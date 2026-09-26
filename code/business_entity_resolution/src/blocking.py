"""
blocking.py
-----------
Candidate pair generation for the Amazon ML Challenge 2026.

Architecture:
  Stage A — Classical blocking (MUST run first, alone, validated)
    Key 1: Normalized name token-sort prefix (first 3 chars of sorted name)
    Key 2: Double Metaphone phonetic code per token
    Key 3: Shared address postal-code / PIN fragment

  Stage B — Embedding ANN (gated — only runs on S1 entities where Stage A
             returned few/no candidates, streamed in batches to disk)

The two stages produce:
  candidate_pairs_classical.tsv  — Stage A output (for recall measurement)
  candidate_pairs.tsv            — Final union of A + B (pipeline output)

Public API:
    run_stage_a(s1, s2, s3) -> pd.DataFrame  (source1_entity_id, candidate_entity_ids)
    run_stage_b(s1, s2, s3, weak_s1_ids, output_dir) -> pd.DataFrame
    build_candidate_pairs(s1, s2, s3, output_dir, embed_threshold) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is only used for progress bars; degrade gracefully
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

logger = logging.getLogger(__name__)

# jellyfish is imported once at module load (not per-token inside the hot loop,
# which is called millions of times). If unavailable, phonetic keys are skipped.
try:
    import jellyfish as _jellyfish
except ImportError:  # pragma: no cover
    _jellyfish = None
    logger.warning("jellyfish not installed — phonetic blocking keys disabled.")

_POSTAL_RE = __import__("re").compile(r"\b(\d{5,6})\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phonetic_keys(name: str) -> list[str]:
    """Return a Metaphone code for each token in name (>=2 chars)."""
    if _jellyfish is None:
        return []
    codes = []
    for t in name.split():
        if len(t) < 2:
            continue
        code = _jellyfish.metaphone(t)
        if code:
            codes.append(code)
    return codes


def _postal_fragments(address: str) -> list[str]:
    """
    Extract plausible postal-code / PIN fragments from a normalized address.
    Looks for:
      - 5-digit or 5+4-digit US ZIP  (e.g. 90210, 90210-1234)
      - 6-digit Indian PIN            (e.g. 400001)
      - French postal code (5-digit starting with 0-9)
    Returns empty list if none found.
    """
    # All digit runs of length 5 or 6 (regex compiled once at module level)
    fragments = _POSTAL_RE.findall(address)
    return list(set(fragments))


# ---------------------------------------------------------------------------
# Blocking index helpers
# ---------------------------------------------------------------------------

def _build_index(
    df: pd.DataFrame,
    source_label: str,
    max_key_freq: int = 1000,
    fn_max_freq: int = 5000,
) -> dict[str, list[str]]:
    """
    Build a blocking index: blocking_key -> [entity_id, ...].
    Keys are constructed from:
      - Exact full sorted normalized name   (FN:) — high precision, prune-exempt
      - Sorted-name first-6-char prefix     (NP:)
      - Phonetic codes of name tokens       (PH:)
      - Postal fragments of address         (PC:)

    Prunes NP/PH/PC keys that map to > max_key_freq entities (stop-keys) to
    avoid combinatorial explosion during intersection. Exact full-name (FN:)
    keys get a much higher ceiling (fn_max_freq) rather than the normal
    stop-key threshold: an exact normalized-name collision is a strong,
    high-precision signal, so exempting it protects legitimate businesses
    whose noisy prefix/phonetic keys happen to be over-frequent. The
    fn_max_freq ceiling is a defensive backstop only — it caps the
    pathological case where a *wildly* common exact name (e.g. a generic
    two-word name shared by thousands of records) would otherwise produce an
    unbounded candidate block for every S1 entity carrying that exact name.
    Set fn_max_freq very high so it never fires in normal operation.
    """
    index: dict[str, list[str]] = {}

    for row in df.itertuples(index=False):
        eid        = row.entity_id
        sorted_name = getattr(row, "norm_name_sorted", "")
        norm_addr  = getattr(row, "norm_address", "")

        keys: list[str] = []

        # Key type 0: exact full sorted normalized name (prune-exempt)
        if sorted_name:
            keys.append(f"FN:{sorted_name}")

        # Key type 1: name prefix (first 6 chars of sorted normalized name)
        if len(sorted_name) >= 3:
            keys.append(f"NP:{sorted_name[:6]}")

        # Key type 2: phonetic codes per token
        for code in _phonetic_keys(sorted_name):
            keys.append(f"PH:{code}")

        # Key type 3: postal fragments
        for fragment in _postal_fragments(norm_addr):
            keys.append(f"PC:{fragment}")

        for k in set(keys):  # use set to avoid double-adding same key for one row
            index.setdefault(k, []).append(eid)

    # Prune highly frequent keys (stop-keys). NP/PH/PC keys use max_key_freq;
    # FN: (exact-name) keys use the much higher fn_max_freq backstop so a
    # pathologically common exact name can't create an unbounded block.
    n_before = len(index)
    n_fn_capped = 0
    pruned_index: dict[str, list[str]] = {}
    for k, v in index.items():
        if k.startswith("FN:"):
            if len(v) <= fn_max_freq:
                pruned_index[k] = v
            else:
                n_fn_capped += 1
        elif len(v) <= max_key_freq:
            pruned_index[k] = v
    index = pruned_index
    n_pruned = n_before - len(index)

    logger.info(
        "Built blocking index for %s: %d keys (%d stop-keys pruned, "
        "%d FN keys capped at fn_max_freq=%d)",
        source_label, len(index), n_pruned, n_fn_capped, fn_max_freq,
    )
    return index


def _get_s1_keys(row) -> list[str]:
    """Return blocking keys for a single S1 row (mirrors _build_index logic)."""
    sorted_name = getattr(row, "norm_name_sorted", "") or ""
    norm_addr   = getattr(row, "norm_address", "") or ""
    keys: list[str] = []
    if sorted_name:
        keys.append(f"FN:{sorted_name}")
    if len(sorted_name) >= 3:
        keys.append(f"NP:{sorted_name[:6]}")
    for code in _phonetic_keys(sorted_name):
        keys.append(f"PH:{code}")
    for fragment in _postal_fragments(norm_addr):
        keys.append(f"PC:{fragment}")
    return keys


# ---------------------------------------------------------------------------
# Stage A — Classical Blocking (memory-efficient streaming)
# ---------------------------------------------------------------------------

def run_stage_a(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """
    Run classical blocking (token-sort prefix + phonetic + postal).

    Memory-efficient design:
      - Builds Sx inverted index (key -> [sx_id]) once in RAM.
      - Iterates S1 rows one-by-one, looks up each row's blocking keys
        against the Sx index, accumulates candidates in a transient set
        per row, then appends result to a growing list.
      - Never holds the full s1->candidates mapping in RAM simultaneously;
        each row's candidate set is serialised to a string immediately.

    Parameters
    ----------
    s1, s2, s3 : DataFrames with columns including
                 entity_id, norm_name_sorted, norm_address
    chunk_size : Rows of S1 to process before logging progress.

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, candidate_entity_ids,
                                n_candidates].
    All S1 entities appear (with empty string if no candidates found).
    """
    logger.info("Stage A: building Sx inverted indexes ...")

    # Only Sx indexes live in RAM (key -> list[sx_id])
    s2_index = _build_index(s2, "S2")
    s3_index = _build_index(s3, "S3")

    logger.info(
        "Stage A: S2 index %d keys | S3 index %d keys | streaming %d S1 rows ...",
        len(s2_index), len(s3_index), len(s1),
    )

    all_rows: list[dict] = []
    s1_reset = s1.reset_index(drop=True)
    n = len(s1_reset)

    for i, row in enumerate(s1_reset.itertuples(index=False)):
        s1_id = row.entity_id
        keys  = _get_s1_keys(row)

        candidates: set[str] = set()
        for k in keys:
            candidates.update(s2_index.get(k, ()))
            candidates.update(s3_index.get(k, ()))
        candidates.discard(s1_id)

        all_rows.append({
            "source1_entity_id":    s1_id,
            "candidate_entity_ids": ",".join(sorted(candidates)),
            "n_candidates":         len(candidates),
        })

        if (i + 1) % chunk_size == 0 or (i + 1) == n:
            logger.info("Stage A: processed %d / %d S1 rows ...", i + 1, n)

    df_candidates = pd.DataFrame(all_rows)
    n_zero = int((df_candidates["n_candidates"] == 0).sum())
    logger.info(
        "Stage A complete: %d S1 entities, %d with >=1 candidate (%.1f%%), "
        "%d with ZERO candidates (%.1f%% — recall-lost tail, Stage B target)",
        len(df_candidates),
        (df_candidates["n_candidates"] > 0).sum(),
        (df_candidates["n_candidates"] > 0).mean() * 100,
        n_zero,
        n_zero / max(1, len(df_candidates)) * 100,
    )
    return df_candidates


# ---------------------------------------------------------------------------
# Stage A — Recall measurement on validation split
# ---------------------------------------------------------------------------

def measure_recall(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure the recall ceiling of a candidate set against ground truth.

    Parameters
    ----------
    candidates : DataFrame with [source1_entity_id, candidate_entity_ids]
    ground_truth : DataFrame with [source1_entity_id, matched_entity_ids]

    Returns
    -------
    dict with keys: recall, n_s1_entities, n_true_matches, n_recovered
    """
    gt_map: dict[str, set[str]] = {}
    for row in ground_truth.itertuples(index=False):
        s1_id   = row.source1_entity_id
        matched = row.matched_entity_ids
        if matched:
            gt_map[s1_id] = set(matched.split(","))
        else:
            gt_map[s1_id] = set()

    cand_map: dict[str, set[str]] = {}
    for row in candidates.itertuples(index=False):
        s1_id    = row.source1_entity_id
        cand_str = row.candidate_entity_ids
        cand_map[s1_id] = set(cand_str.split(",")) if cand_str else set()

    n_true_total = 0
    n_recovered  = 0

    for s1_id, true_ids in gt_map.items():
        if not true_ids:
            continue  # singleton — skip from recall calculation
        n_true_total += len(true_ids)
        cand_ids = cand_map.get(s1_id, set())
        n_recovered += len(true_ids & cand_ids)

    recall = n_recovered / max(1, n_true_total)
    logger.info(
        "Recall ceiling: %.4f  (%d / %d true matches recovered)",
        recall, n_recovered, n_true_total,
    )
    return {
        "recall": recall,
        "n_true_matches": n_true_total,
        "n_recovered": n_recovered,
    }


# ---------------------------------------------------------------------------
# Stage B — Embedding ANN (gated)
# ---------------------------------------------------------------------------

def run_stage_b(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    weak_s1_ids: set[str],
    output_dir: str | Path,
    model_name: str = "sentence-transformers/LaBSE",
    batch_size: int = 512,
    top_k: int = 10,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Embedding-based ANN blocking, gated to S1 entities in `weak_s1_ids`.

    Embeddings are computed in batches and written to a memory-mapped numpy
    array on disk — never held fully in RAM. Supports resuming if interrupted.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        logger.error(
            "sentence-transformers or faiss-cpu not installed: %s. "
            "Install with: pip install sentence-transformers faiss-cpu",
            e,
        )
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_ids"]
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_b_path = output_dir / "stage_b_candidates.tsv"
    if resume and stage_b_path.is_file() and stage_b_path.stat().st_size > 0:
        logger.info("Stage B: loading completed candidate enrichments from %s", stage_b_path)
        try:
            return pd.read_csv(stage_b_path, sep="\t", dtype=str).fillna("")
        except Exception as e:
            logger.warning("Failed to read %s: %s. Recomputing.", stage_b_path, e)

    logger.info("Stage B: loading embedding model '%s' ...", model_name)
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    # ------- Build gallery (S2 + S3 raw name strings) -------
    sx_combined = pd.concat([s2, s3], ignore_index=True)
    sx_names  = sx_combined["business_name"].fillna("").tolist()
    sx_ids    = sx_combined["entity_id"].tolist()
    n_gallery = len(sx_names)

    gallery_path = output_dir / "sx_embeddings.npy"
    gallery_done_flag = output_dir / "gallery_done.flag"
    gallery_progress_path = output_dir / "gallery_progress.json"

    if resume and gallery_path.is_file() and gallery_done_flag.is_file():
        logger.info(
            "Stage B: gallery embeddings already complete (%d rows) at %s. Skipping re-embedding.",
            n_gallery, gallery_path
        )
        gallery_mm = np.lib.format.open_memmap(
            gallery_path, mode="r", dtype="float32", shape=(n_gallery, dim)
        )
    else:
        logger.info(
            "Stage B: embedding %d S2/S3 rows into %s ...", n_gallery, gallery_path
        )
        resume_start = 0
        if resume and gallery_path.is_file() and gallery_progress_path.is_file():
            try:
                import json
                with open(gallery_progress_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                saved_idx = meta.get("last_index", 0)
                if 0 < saved_idx < n_gallery:
                    resume_start = saved_idx
                    logger.info("Stage B: resuming gallery embeddings from index %d / %d ...", resume_start, n_gallery)
            except Exception:
                resume_start = 0

        mode = "r+" if (resume_start > 0 and gallery_path.is_file()) else "w+"
        gallery_mm = np.lib.format.open_memmap(
            gallery_path, mode=mode, dtype="float32", shape=(n_gallery, dim)
        )

        for start in tqdm(range(resume_start, n_gallery, batch_size), desc="Embed S2/S3"):
            batch = sx_names[start : start + batch_size]
            vecs  = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            gallery_mm[start : start + len(batch)] = vecs

            if (start + len(batch)) % (batch_size * 50) == 0 or (start + len(batch)) == n_gallery:
                gallery_mm.flush()
                import json
                with open(gallery_progress_path, "w", encoding="utf-8") as f:
                    json.dump({"last_index": start + len(batch), "n_gallery": n_gallery}, f)

        gallery_mm.flush()
        gallery_done_flag.write_text("DONE\n", encoding="utf-8")
        logger.info("Stage B: gallery embeddings written and verified.")

    # ------- Build FAISS index -------
    faiss_index_path = output_dir / "faiss_index.bin"
    if resume and faiss_index_path.is_file() and faiss_index_path.stat().st_size > 0:
        logger.info("Stage B: loading FAISS index from %s ...", faiss_index_path)
        index = faiss.read_index(str(faiss_index_path))
    else:
        logger.info("Stage B: building FAISS index ...")
        index = faiss.IndexFlatIP(dim)  # inner product on unit vectors = cosine sim
        for start in tqdm(range(0, n_gallery, 4096), desc="FAISS add"):
            index.add(gallery_mm[start : start + 4096])
        logger.info("Stage B: FAISS index built with %d vectors. Saving to %s ...", index.ntotal, faiss_index_path)
        faiss.write_index(index, str(faiss_index_path))

    # ------- Query weak S1 entities -------
    weak_s1 = s1[s1["entity_id"].isin(weak_s1_ids)].reset_index(drop=True)
    query_names = weak_s1["business_name"].fillna("").tolist()
    query_ids   = weak_s1["entity_id"].tolist()
    n_queries   = len(query_names)

    logger.info("Stage B: querying %d weak S1 entities (top-%d) ...", n_queries, top_k)

    results: dict[str, set[str]] = {}
    for start in tqdm(range(0, n_queries, batch_size), desc="Query S1"):
        batch_names = query_names[start : start + batch_size]
        batch_ids   = query_ids[start : start + batch_size]
        vecs = model.encode(
            batch_names, convert_to_numpy=True, normalize_embeddings=True
        )
        scores, neighbors = index.search(vecs, top_k)

        for q_id, neighbor_idxs, neighbor_scores in zip(
            batch_ids, neighbors, scores
        ):
            hits: set[str] = set()
            for idx, score in zip(neighbor_idxs, neighbor_scores):
                if idx < 0:
                    continue
                hits.add(sx_ids[idx])
            results[q_id] = hits

    rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(candidates)),
        }
        for s1_id, candidates in results.items()
    ]
    df_stage_b = pd.DataFrame(rows)
    df_stage_b.to_csv(stage_b_path, sep="\t", index=False)
    logger.info("Stage B complete: %d entities enriched. Saved to %s", len(df_stage_b), stage_b_path)
    return df_stage_b


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    output_dir: str | Path,
    embed_threshold: int = 0,
    run_embedding: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Build the final candidate_pairs DataFrame with checkpointing and resume support.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "candidate_pairs.tsv"
    if resume and final_path.is_file() and final_path.stat().st_size > 0:
        logger.info("Found existing candidate pairs at %s. Loading directly.", final_path)
        try:
            return pd.read_csv(final_path, sep="\t", dtype=str).fillna("")
        except Exception as e:
            logger.warning("Could not load %s (%s). Rebuilding.", final_path, e)

    # ---- Stage A ----
    classical_path = output_dir / "candidate_pairs_classical.tsv"
    if resume and classical_path.is_file() and classical_path.stat().st_size > 0:
        logger.info("Found existing classical candidates at %s. Loading directly.", classical_path)
        try:
            stage_a = pd.read_csv(classical_path, sep="\t", dtype=str).fillna("")
            if "n_candidates" not in stage_a.columns:
                stage_a["n_candidates"] = stage_a["candidate_entity_ids"].apply(
                    lambda s: len(s.split(",")) if (s and str(s).strip()) else 0
                )
        except Exception as e:
            logger.warning("Failed to load %s (%s). Running Stage A anew.", classical_path, e)
            stage_a = run_stage_a(s1, s2, s3)
            (
                stage_a[["source1_entity_id", "candidate_entity_ids"]]
                .to_csv(classical_path, sep="\t", index=False)
            )
    else:
        logger.info("Running Stage A (classical blocking) ...")
        stage_a = run_stage_a(s1, s2, s3)
        (
            stage_a[["source1_entity_id", "candidate_entity_ids"]]
            .to_csv(classical_path, sep="\t", index=False)
        )
        logger.info("Stage A candidates written to %s", classical_path)

    if not run_embedding:
        logger.info("Stage B skipped (run_embedding=False). Returning Stage A only.")
        stage_a[["source1_entity_id", "candidate_entity_ids"]].to_csv(final_path, sep="\t", index=False)
        return stage_a[["source1_entity_id", "candidate_entity_ids"]]

    # ---- Stage B ----
    if embed_threshold == -1:
        # Full-corpus mode: embed every S1 entity (GPU-appropriate)
        weak_s1_ids = set(s1["entity_id"])
        logger.info(
            "Stage B: full-corpus mode — embedding all %d S1 entities",
            len(weak_s1_ids),
        )
    else:
        # Gated mode: only embed entities where classical blocking was weak
        weak_s1_ids = set(
            stage_a.loc[stage_a["n_candidates"] <= embed_threshold, "source1_entity_id"]
        )
        logger.info(
            "Stage B: gated mode — %d S1 entities with n_candidates ≤ %d → embedding pass",
            len(weak_s1_ids),
            embed_threshold,
        )

    stage_b = run_stage_b(
        s1, s2, s3,
        weak_s1_ids=weak_s1_ids,
        output_dir=output_dir / "embeddings",
        resume=resume,
    )

    # ---- Union A + B ----
    merged: dict[str, set[str]] = {}

    for row in stage_a.itertuples(index=False):
        s1_id = row.source1_entity_id
        cands = row.candidate_entity_ids
        merged.setdefault(s1_id, set())
        if cands and str(cands).strip():
            merged[s1_id].update(str(cands).split(","))

    for row in stage_b.itertuples(index=False):
        s1_id = row.source1_entity_id
        cands = row.candidate_entity_ids
        merged.setdefault(s1_id, set())
        if cands and str(cands).strip():
            merged[s1_id].update(str(cands).split(","))

    final_rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cands)),
        }
        for s1_id, cands in merged.items()
    ]
    final = pd.DataFrame(final_rows)
    final.to_csv(final_path, sep="\t", index=False)
    logger.info("Final candidate pairs written to %s", final_path)

    return final
