"""
reranker.py
-----------
Borderline-band cross-encoder reranking.

The LightGBM classifier is confident on the easy majority of pairs. The score
that decides the leaderboard, though, is set by a small band of AMBIGUOUS pairs
sitting right around the decision threshold. This module reranks *only* that
band with a small multilingual cross-encoder, which reads the two names (and
addresses) jointly — resolving exactly the France / Latin↔Indic / chain cases
edit-distance and a bag-of-features GBM struggle with. Because it touches only
a few percent of pairs, the compute cost is trivial even though a cross-encoder
is far heavier per-pair than the GBM.

Design contract:
  * Pairs OUTSIDE the band keep their original (calibrated) LightGBM score.
  * Pairs INSIDE the band [t−δ, t+δ] have their score replaced by the
    cross-encoder's sigmoid score.
  * Reranking is applied identically to val and test, and the decision
    threshold/policy is (re)tuned on the reranked val scores — so the modified
    score distribution stays internally consistent end-to-end.

Model constraints (competition rules): must be MIT/Apache-2.0 and ≤8B params.
The default, `BAAI/bge-reranker-v2-m3`, is an XLM-RoBERTa-based multilingual
reranker (~568M params) — well under 8B and strong on the Indic tail. VERIFY the
license of whatever model you actually download on the lab machine before
submitting; swap via `model_name` if needed (e.g. a multilingual MiniLM CE).

Gated + graceful: if sentence-transformers / torch is unavailable (e.g. the
sandbox), reranking is skipped and the original scores are returned unchanged.

Public API:
    rerank_borderline(scores, pairs, s1_df, sx_df, threshold, band=..., ...)
        -> (np.ndarray adjusted_scores, np.ndarray reranked_mask)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # ≤8B, multilingual; verify license on lab box


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def _entity_text(df_indexed, eid) -> str:
    """'<name> | <address>' for a given entity id (best-effort)."""
    try:
        name = df_indexed.at[eid, "business_name"]
    except Exception:
        name = ""
    try:
        addr = df_indexed.at[eid, "business_address"]
    except Exception:
        addr = ""
    name = "" if not isinstance(name, str) else name
    addr = "" if not isinstance(addr, str) else addr
    return f"{name} | {addr}".strip()


def rerank_borderline(
    scores: np.ndarray,
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    sx_df: pd.DataFrame,
    threshold: float,
    band: float = 0.10,
    model_name: str = DEFAULT_RERANK_MODEL,
    batch_size: int = 128,
    max_length: int = 128,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rerank the pairs whose score lies in [threshold−band, threshold+band].

    Returns
    -------
    (adjusted_scores, reranked_mask)
      adjusted_scores : copy of `scores` with band positions overwritten by the
                        cross-encoder sigmoid score.
      reranked_mask   : bool array marking which positions were reranked.
    """
    scores = np.asarray(scores, dtype="float64").copy()
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return scores, mask

    lo, hi = threshold - band, threshold + band
    band_idx = np.where((scores >= lo) & (scores <= hi))[0]
    logger.info("Reranker: %d / %d pairs in borderline band [%.3f, %.3f] (%.2f%%).",
                len(band_idx), n, lo, hi, 100.0 * len(band_idx) / max(1, n))
    if len(band_idx) == 0:
        return scores, mask

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers/torch not available — cross-encoder reranking "
            "SKIPPED; returning original scores. (Install on the lab machine to "
            "enable this lever.)"
        )
        return scores, mask

    s1_idx = s1_df.set_index("entity_id")
    sx_idx = sx_df.set_index("entity_id")
    s1_ids = pairs["source1_entity_id"].to_numpy()
    sx_ids = pairs["candidate_entity_id"].to_numpy()

    ce_pairs = [
        [_entity_text(s1_idx, s1_ids[i]), _entity_text(sx_idx, sx_ids[i])]
        for i in band_idx
    ]

    logger.info("Reranker: loading cross-encoder '%s' ...", model_name)
    model = CrossEncoder(model_name, max_length=max_length, device=device)
    logits = model.predict(ce_pairs, batch_size=batch_size, show_progress_bar=False)
    ce_scores = _sigmoid(np.asarray(logits, dtype="float64").reshape(-1))

    scores[band_idx] = ce_scores
    mask[band_idx] = True
    logger.info("Reranker: %d borderline pairs rescored by cross-encoder.", len(band_idx))
    return scores, mask
