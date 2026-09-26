"""
pipeline.py
-----------
Top-level orchestrator for the Amazon ML Challenge 2026 Entity Resolution pipeline.

Usage:
    python -m src.pipeline \\
        --train-dir  dataset/train \\
        --test-dir   dataset/test  \\
        --output-dir output/       \\
        [--val-fraction 0.2]       \\
        [--neg-ratio 3]            \\
        [--embed-threshold 0]      \\
        [--run-embedding]

Sequencing:
  1. Parse train + test files (robust csv.reader, quarantine logging)
  2. Normalize all source DataFrames (NFKC → unidecode → lowercase → suffix strip)
  3. Hold out a validation split from train_ground_truth (20% default)
  4. Stage A classical blocking → candidate_pairs_classical.tsv
  5. Measure recall ceiling on validation split
  6. [Optional] Stage B embedding ANN gated to weak entities → merged candidates
  7. Explode candidate pairs → build feature matrix
  8. Train LightGBM on train split
  9. Tune threshold on validation split (maximise F_0.5)
  10. Inference on test set → matching_results.tsv + candidate_pairs.tsv
  11. Run validate_submission.py (prints PASS/FAIL)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .parser        import load_source, load_ground_truth
from .normalization import add_normalized_columns
from .blocking      import build_candidate_pairs, run_stage_a, measure_recall
from .features      import (
    build_feature_matrix, fit_global_tfidf, compute_embedding_cosine_map,
    FEATURE_COLS,
)
from .model         import (
    build_training_pairs, train, tune_threshold, predict,
    f05_score, save_model,
)
from .policy        import ProbabilityCalibrator, ThresholdPolicy, tune_policy, apply_policy
from .reranker      import rerank_borderline, DEFAULT_RERANK_MODEL
from .validation    import (
    report_diagnostics, loco_evaluate, bucket_errors,
    s1_country_map, source_map, annotate_source,
)
from .checkpoint    import (
    save_dataframe, load_dataframe, save_json, load_json,
    save_numpy, load_numpy, save_pickle, load_pickle, checkpoint_exists,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explode_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated candidate_entity_ids into one row per pair."""
    rows = []
    for row in candidates.itertuples(index=False):
        s1_id    = row.source1_entity_id
        cand_str = row.candidate_entity_ids
        if not cand_str:
            continue
        for sx_id in cand_str.split(","):
            sx_id = sx_id.strip()
            if sx_id:
                rows.append({"source1_entity_id": s1_id, "candidate_entity_id": sx_id})
    return pd.DataFrame(rows)


def _gt_pair_set(gt: pd.DataFrame) -> set[tuple[str, str]]:
    """Build a set of (s1_id, sx_id) true-match pairs from a ground-truth frame."""
    gt_set: set[tuple[str, str]] = set()
    for row in gt.itertuples(index=False):
        s1_id   = row.source1_entity_id
        matched = row.matched_entity_ids
        if matched:
            for sx_id in matched.split(","):
                sx_id = sx_id.strip()
                if sx_id:
                    gt_set.add((s1_id, sx_id))
    return gt_set


def _label_pairs(feat_df: pd.DataFrame, gt_set: set[tuple[str, str]]) -> np.ndarray:
    """
    Vectorized 0/1 labelling of a feature frame against a ground-truth pair set.

    Replaces a per-row DataFrame.apply(axis=1), which is ~2 orders of magnitude
    slower and becomes the pipeline's bottleneck on multi-million-pair sets.
    """
    if len(feat_df) == 0:
        return np.zeros(0, dtype=np.int8)
    s1 = feat_df["source1_entity_id"].to_numpy()
    sx = feat_df["candidate_entity_id"].to_numpy()
    return np.fromiter(
        ((s1[i], sx[i]) in gt_set for i in range(len(feat_df))),
        dtype=np.int8,
        count=len(feat_df),
    )


def _write_output(
    pairs: pd.DataFrame,
    predictions: np.ndarray,
    s1_all_ids: list[str],
    output_dir: Path,
    candidates: pd.DataFrame,
    split: str = "test",
) -> None:
    """
    Write matching_results.tsv and candidate_pairs.tsv to output_dir.

    Parameters
    ----------
    pairs       : [source1_entity_id, candidate_entity_id] (exploded)
    predictions : Binary array aligned with pairs
    s1_all_ids  : All S1 entity IDs that must appear in the output
    output_dir  : Target directory
    candidates  : Raw candidate DataFrame (for candidate_pairs.tsv)
    split       : 'test' or 'val' (for logging)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- matching_results.tsv ---
    matched: dict[str, list[str]] = {eid: [] for eid in s1_all_ids}
    for (s1_id, sx_id), pred in zip(
        zip(pairs["source1_entity_id"], pairs["candidate_entity_id"]),
        predictions,
    ):
        if pred == 1:
            matched.setdefault(s1_id, []).append(sx_id)

    match_rows = [
        {"source1_entity_id": eid, "matched_entity_ids": ",".join(sorted(set(ids)))}
        for eid, ids in matched.items()
    ]
    match_df = pd.DataFrame(match_rows)
    match_path = output_dir / "matching_results.tsv"
    match_df.to_csv(match_path, sep="\t", index=False)
    logger.info("[%s] matching_results.tsv written: %d S1 rows", split, len(match_df))

    # --- candidate_pairs.tsv ---
    cand_path = output_dir / "candidate_pairs.tsv"
    candidates[["source1_entity_id", "candidate_entity_ids"]].to_csv(
        cand_path, sep="\t", index=False
    )
    logger.info("[%s] candidate_pairs.tsv written", split)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _advanced_decision_enabled(args: argparse.Namespace) -> bool:
    """
    True when any lever that replaces the baseline tune_threshold/predict path
    is requested. When False, steps 9–10 run EXACTLY as the frozen baseline
    (identical output to the reviewed commit), so every new capability is
    strictly opt-in.
    """
    return bool(
        args.calibrate
        or args.fine_threshold
        or args.per_source_threshold
        or args.abstain
        or args.rerank
        or args.france_margin > 0.0
    )


def _build_model_params(args: argparse.Namespace) -> Optional[dict]:
    """Optional LightGBM overrides for the class-imbalance experiment."""
    mp: dict = {}
    if args.scale_pos_weight is not None:
        # scale_pos_weight and class_weight are two ways to say the same thing;
        # LightGBM should use exactly one. Prefer the explicit numeric knob.
        mp["class_weight"] = None
        mp["scale_pos_weight"] = float(args.scale_pos_weight)
    elif args.drop_class_weight:
        mp["class_weight"] = None
    return mp or None


def _find_validator_script(test_dir: Path) -> Optional[Path]:
    """Find validate_submission.py from various invocation contexts."""
    candidates = [
        Path("utils/validate_submission.py"),
        Path(__file__).resolve().parents[3] / "utils" / "validate_submission.py",
        Path(__file__).resolve().parents[2] / "utils" / "validate_submission.py",
        test_dir.resolve().parent.parent / "utils" / "validate_submission.py",
        test_dir.resolve().parent / "utils" / "validate_submission.py",
        Path(__file__).resolve().parents[3] / "student_resource" / "utils" / "validate_submission.py",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def run_pipeline(args: argparse.Namespace) -> None:
    train_dir   = Path(args.train_dir).resolve()
    test_dir    = Path(args.test_dir).resolve()
    output_dir  = Path(args.output_dir).resolve()
    quarantine  = output_dir / "quarantine"

    # Setup checkpoint subdirectories
    ckpt_root = (Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints").resolve()
    ckpt_norm     = ckpt_root / "normalized"
    ckpt_splits   = ckpt_root / "splits"
    ckpt_blocking = ckpt_root / "blocking"
    ckpt_features = ckpt_root / "features"
    ckpt_model    = ckpt_root / "model"
    ckpt_tuning   = ckpt_root / "tuning"
    ckpt_test     = ckpt_root / "test"

    for d in [ckpt_norm, ckpt_splits, ckpt_blocking, ckpt_features, ckpt_model, ckpt_tuning, ckpt_test]:
        d.mkdir(parents=True, exist_ok=True)

    resume = args.resume

    # -----------------------------------------------------------------------
    # 1. Parse & 2. Normalize (with Checkpointing)
    # -----------------------------------------------------------------------
    logger.info("=== Steps 1 & 2: Parsing & Normalization ===")
    norm_paths = {
        "s1_train": ckpt_norm / "s1_train.parquet",
        "s2_train": ckpt_norm / "s2_train.parquet",
        "s3_train": ckpt_norm / "s3_train.parquet",
        "gt_full":  ckpt_norm / "gt_full.parquet",
        "s1_test":  ckpt_norm / "s1_test.parquet",
        "s2_test":  ckpt_norm / "s2_test.parquet",
        "s3_test":  ckpt_norm / "s3_test.parquet",
    }
    all_norm_exist = all(checkpoint_exists(p) for p in norm_paths.values())

    if resume and all_norm_exist:
        logger.info("Found all normalized datasets in checkpoint (%s). Loading...", ckpt_norm)
        s1_train = load_dataframe(norm_paths["s1_train"])
        s2_train = load_dataframe(norm_paths["s2_train"])
        s3_train = load_dataframe(norm_paths["s3_train"])
        gt_full  = load_dataframe(norm_paths["gt_full"])
        s1_test  = load_dataframe(norm_paths["s1_test"])
        s2_test  = load_dataframe(norm_paths["s2_test"])
        s3_test  = load_dataframe(norm_paths["s3_test"])
        logger.info("Loaded normalized datasets: s1_train=%d, s2_train=%d, s3_train=%d, s1_test=%d, s2_test=%d, s3_test=%d",
                    len(s1_train), len(s2_train), len(s3_train), len(s1_test), len(s2_test), len(s3_test))
    else:
        logger.info("Parsing source TSV files...")
        s1_train = load_source(train_dir / "train_source1.tsv", quarantine)
        s2_train = load_source(train_dir / "train_source2.tsv", quarantine)
        s3_train = load_source(train_dir / "train_source3.tsv", quarantine)
        gt_full  = load_ground_truth(train_dir / "train_ground_truth.tsv", quarantine)

        s1_test  = load_source(test_dir / "test_source1.tsv",  quarantine)
        s2_test  = load_source(test_dir / "test_source2.tsv",  quarantine)
        s3_test  = load_source(test_dir / "test_source3.tsv",  quarantine)

        for df, name in [
            (s1_train, "s1_train"), (s2_train, "s2_train"), (s3_train, "s3_train"),
            (s1_test,  "s1_test"),  (s2_test,  "s2_test"),  (s3_test,  "s3_test"),
        ]:
            add_normalized_columns(df)
            logger.info("Normalized %s: %d rows", name, len(df))
            save_dataframe(df, norm_paths[name])

        save_dataframe(gt_full, norm_paths["gt_full"])
        logger.info("Normalized datasets checkpointed to %s", ckpt_norm)

    # -----------------------------------------------------------------------
    # 3. Validation split (with Checkpointing)
    # -----------------------------------------------------------------------
    logger.info("=== Step 3: Train/val split ===")
    split_paths = {
        "gt_train":    ckpt_splits / "gt_train.parquet",
        "gt_val":      ckpt_splits / "gt_val.parquet",
        "s1_tr_split": ckpt_splits / "s1_tr_split.parquet",
        "s1_va_split": ckpt_splits / "s1_va_split.parquet",
    }
    all_splits_exist = all(checkpoint_exists(p) for p in split_paths.values())

    if resume and all_splits_exist:
        logger.info("Found train/val splits in checkpoint (%s). Loading...", ckpt_splits)
        gt_train    = load_dataframe(split_paths["gt_train"])
        gt_val      = load_dataframe(split_paths["gt_val"])
        s1_tr_split = load_dataframe(split_paths["s1_tr_split"])
        s1_va_split = load_dataframe(split_paths["s1_va_split"])
    else:
        all_s1_ids = gt_full["source1_entity_id"].tolist()
        train_ids, val_ids = train_test_split(
            all_s1_ids, test_size=args.val_fraction, random_state=42
        )
        gt_train = gt_full[gt_full["source1_entity_id"].isin(set(train_ids))].reset_index(drop=True)
        gt_val   = gt_full[gt_full["source1_entity_id"].isin(set(val_ids))].reset_index(drop=True)

        s1_tr_split = s1_train[s1_train["entity_id"].isin(set(train_ids))].reset_index(drop=True)
        s1_va_split = s1_train[s1_train["entity_id"].isin(set(val_ids))].reset_index(drop=True)

        save_dataframe(gt_train, split_paths["gt_train"])
        save_dataframe(gt_val, split_paths["gt_val"])
        save_dataframe(s1_tr_split, split_paths["s1_tr_split"])
        save_dataframe(s1_va_split, split_paths["s1_va_split"])
        logger.info("Splits checkpointed to %s", ckpt_splits)

    logger.info("Train split: %d S1 entities | Val split: %d S1 entities",
                len(s1_tr_split), len(s1_va_split))

    # -----------------------------------------------------------------------
    # 4. Stage A — Classical Blocking (train split)
    # -----------------------------------------------------------------------
    logger.info("=== Step 4: Stage A classical blocking (train split) ===")
    cand_train_a_path = ckpt_blocking / "candidates_train_a.tsv"
    if resume and checkpoint_exists(cand_train_a_path):
        logger.info("Found Stage A train candidates at %s. Loading...", cand_train_a_path)
        candidates_train_a = pd.read_csv(cand_train_a_path, sep="\t", dtype=str).fillna("")
        if "n_candidates" not in candidates_train_a.columns:
            candidates_train_a["n_candidates"] = candidates_train_a["candidate_entity_ids"].apply(
                lambda s: len(str(s).split(",")) if (s and str(s).strip()) else 0
            )
    else:
        candidates_train_a = run_stage_a(s1_tr_split, s2_train, s3_train)
        candidates_train_a.to_csv(cand_train_a_path, sep="\t", index=False)

    # -----------------------------------------------------------------------
    # 5. Recall measurement on validation split
    # -----------------------------------------------------------------------
    logger.info("=== Step 5: Recall ceiling measurement (val split) ===")
    cand_val_a_path = ckpt_blocking / "candidates_val_a.tsv"
    recall_stats_path = ckpt_blocking / "recall_stats.json"

    if resume and checkpoint_exists(cand_val_a_path) and checkpoint_exists(recall_stats_path):
        logger.info("Found Stage A val candidates and recall stats at %s. Loading...", ckpt_blocking)
        candidates_val_a = pd.read_csv(cand_val_a_path, sep="\t", dtype=str).fillna("")
        if "n_candidates" not in candidates_val_a.columns:
            candidates_val_a["n_candidates"] = candidates_val_a["candidate_entity_ids"].apply(
                lambda s: len(str(s).split(",")) if (s and str(s).strip()) else 0
            )
        recall_stats = load_json(recall_stats_path)
    else:
        candidates_val_a = run_stage_a(s1_va_split, s2_train, s3_train)
        candidates_val_a.to_csv(cand_val_a_path, sep="\t", index=False)
        recall_stats = measure_recall(candidates_val_a, gt_val)
        save_json(recall_stats, recall_stats_path)

    logger.info(
        "Recall ceiling: %.4f  (%d / %d true matches recovered)",
        recall_stats["recall"],
        recall_stats["n_recovered"],
        recall_stats["n_true_matches"],
    )

    # Two gate numbers that decide the whole strategy (singleton ratio +
    # blocking recall ceiling → precision-bound vs recall-bound verdict).
    report_diagnostics(gt_full, recall_ceiling=recall_stats["recall"])

    # -----------------------------------------------------------------------
    # 6. [Optional] Stage B — Embedding ANN
    # -----------------------------------------------------------------------
    if args.run_embedding:
        logger.info("=== Step 6: Stage B embedding ANN (gated) ===")
        candidates_train = build_candidate_pairs(
            s1_tr_split, s2_train, s3_train,
            output_dir=output_dir / "blocking_train",
            embed_threshold=args.embed_threshold,
            run_embedding=True,
            resume=resume,
        )
        candidates_val = build_candidate_pairs(
            s1_va_split, s2_train, s3_train,
            output_dir=output_dir / "blocking_val",
            embed_threshold=args.embed_threshold,
            run_embedding=True,
            resume=resume,
        )
    else:
        logger.info("=== Step 6: Stage B skipped ===")
        candidates_train = candidates_train_a
        candidates_val   = candidates_val_a

    # -----------------------------------------------------------------------
    # 7. Feature matrix (with Checkpointing)
    # -----------------------------------------------------------------------
    logger.info("=== Step 7: Building feature matrices ===")
    feat_paths = {
        "tfidf":        ckpt_features / "tfidf_vec.pkl",
        "emb_train":    ckpt_features / "emb_map_train.pkl",
        "emb_val":      ckpt_features / "emb_map_val.pkl",
        "X_train":      ckpt_features / "X_train.parquet",
        "X_val":        ckpt_features / "X_val.parquet",
        "pairs_train":  ckpt_features / "pairs_train.parquet",
        "pairs_val":    ckpt_features / "pairs_val.parquet",
    }

    # Combined S2+S3 frame for the TRAIN sources — needed for feature building,
    # reranking, and error-analysis regardless of whether features are rebuilt
    # or loaded from checkpoint, so define it unconditionally here.
    sx_train_combined = pd.concat([s2_train, s3_train], ignore_index=True)

    if (resume and checkpoint_exists(feat_paths["X_train"]) and
            checkpoint_exists(feat_paths["X_val"]) and
            checkpoint_exists(feat_paths["pairs_val"]) and
            checkpoint_exists(feat_paths["tfidf"])):
        logger.info("Found feature matrices in checkpoint (%s). Loading...", ckpt_features)
        X_train_df  = load_dataframe(feat_paths["X_train"])
        X_val_df    = load_dataframe(feat_paths["X_val"])
        pairs_train = load_dataframe(feat_paths["pairs_train"])
        pairs_val   = load_dataframe(feat_paths["pairs_val"])
        tfidf_vec   = load_pickle(feat_paths["tfidf"])
    else:
        pairs_train = _explode_candidates(candidates_train)
        pairs_val   = _explode_candidates(candidates_val)

        # Fit ONE TF-IDF vectorizer on the full training name corpus
        tfidf_vec = fit_global_tfidf([
            s1_train["norm_name"], s2_train["norm_name"], s3_train["norm_name"],
        ])
        save_pickle(tfidf_vec, feat_paths["tfidf"])

        emb_map_train = emb_map_val = None
        if args.run_embedding:
            emb_map_train = compute_embedding_cosine_map(
                pairs_train, s1_tr_split, sx_train_combined,
                cache_path=feat_paths["emb_train"], resume=resume,
            )
            emb_map_val   = compute_embedding_cosine_map(
                pairs_val,   s1_va_split, sx_train_combined,
                cache_path=feat_paths["emb_val"], resume=resume,
            )

        X_train_df = build_feature_matrix(
            pairs_train, s1_tr_split, sx_train_combined,
            embedding_cosine_map=emb_map_train, tfidf_vectorizer=tfidf_vec,
        )
        X_val_df   = build_feature_matrix(
            pairs_val, s1_va_split, sx_train_combined,
            embedding_cosine_map=emb_map_val, tfidf_vectorizer=tfidf_vec,
        )

        save_dataframe(X_train_df, feat_paths["X_train"])
        save_dataframe(X_val_df, feat_paths["X_val"])
        save_dataframe(pairs_train, feat_paths["pairs_train"])
        save_dataframe(pairs_val, feat_paths["pairs_val"])
        logger.info("Feature matrices checkpointed to %s", ckpt_features)

    # -----------------------------------------------------------------------
    # 8. Build labels + train LightGBM (with Checkpointing & Resume)
    # -----------------------------------------------------------------------
    logger.info("=== Step 8: Training LightGBM ===")
    y_fit_path = ckpt_features / "y_fit.npy"
    X_fit_path = ckpt_features / "X_fit.parquet"

    # Per-source group labels ('S2'/'S3') for optional per-group thresholding
    # and for LOCO's per-source policy. Country map keys the LOCO folds.
    smap_train       = source_map(s2_train, s3_train)
    country_map_all  = s1_country_map(s1_train)
    src_labels_train = annotate_source(pairs_train, smap_train)

    # Full (pre-downsample) labels — needed both for the LOCO diagnostic and to
    # build the downsampled (X_fit, y_fit) training set below.
    gt_set  = _gt_pair_set(gt_train)
    y_train = _label_pairs(X_train_df, gt_set)

    if resume and checkpoint_exists(y_fit_path) and checkpoint_exists(X_fit_path):
        logger.info("Found training features and labels in checkpoint. Loading...")
        X_fit = load_dataframe(X_fit_path)
        y_fit = load_numpy(y_fit_path)
    else:

        # Downsample negatives for training
        pos_mask  = y_train == 1
        neg_mask  = y_train == 0
        n_pos     = pos_mask.sum()
        max_neg   = n_pos * args.neg_ratio
        neg_idx   = np.where(neg_mask)[0]
        rng       = np.random.default_rng(42)
        chosen_neg = rng.choice(neg_idx, size=min(max_neg, len(neg_idx)), replace=False)
        keep = np.concatenate([np.where(pos_mask)[0], chosen_neg])

        X_fit = X_train_df.iloc[keep].reset_index(drop=True)
        y_fit = y_train[keep]
        save_dataframe(X_fit, X_fit_path)
        save_numpy(y_fit, y_fit_path)

    # -----------------------------------------------------------------------
    # 8b. [Optional] LOCO — unseen-country transfer diagnostic
    # -----------------------------------------------------------------------
    france_margin = float(args.france_margin)
    if args.loco:
        logger.info("=== Step 8b: LOCO unseen-country transfer evaluation ===")
        loco = loco_evaluate(
            X_train_df, y_train, gt_train, country_map_all,
            source_labels=src_labels_train if args.per_source_threshold else None,
            neg_ratio=args.neg_ratio,
            calib_method=(args.calib_method if args.calibrate else "isotonic"),
        )
        if args.france_margin <= 0.0:
            france_margin = float(loco["recommended_conservative_margin"])
            logger.info("LOCO-derived France-robust margin: +%.3f (applied to final threshold)",
                        france_margin)

    clf = train(
        X_fit, y_fit,
        model_params=_build_model_params(args),
        checkpoint_dir=ckpt_model,
        checkpoint_freq=args.checkpoint_freq,
        resume=resume,
    )
    save_model(clf, output_dir / "model.pkl")

    # -----------------------------------------------------------------------
    # 9. Decision policy on val split
    # -----------------------------------------------------------------------
    use_advanced = _advanced_decision_enabled(args)
    cal: Optional[ProbabilityCalibrator] = None
    adj_policy: Optional[ThresholdPolicy] = None
    best_threshold = 0.5

    if not use_advanced:
        logger.info("=== Step 9: Threshold tuning (val split) — baseline path ===")
        threshold_ckpt_path = ckpt_tuning / "threshold_checkpoint.json"
        best_threshold, best_f05 = tune_threshold(
            clf, X_val_df, pairs_val, gt_val,
            checkpoint_path=threshold_ckpt_path,
            resume=resume,
        )
        logger.info("Best threshold: %.2f | Val F_0.5: %.4f", best_threshold, best_f05)

        # Diagnostic-only error bucketing on the baseline decision (no change to output).
        if args.error_analysis:
            logger.info("=== Step 9b: Error-analysis bucketing (val split) ===")
            val_preds = predict(clf, X_val_df, threshold=best_threshold)
            bucket_errors(
                pairs_val, val_preds, gt_val, s1_va_split, sx_train_combined,
                country_map=country_map_all,
            )
    else:
        logger.info("=== Step 9: Precision-first decision policy (val split) ===")
        raw_val = clf.predict_proba(X_val_df[FEATURE_COLS])[:, 1]
        y_val   = _label_pairs(X_val_df, _gt_pair_set(gt_val))

        # (a) calibration — identity pass-through unless --calibrate
        cal = ProbabilityCalibrator(
            args.calib_method if args.calibrate else "none"
        ).fit(raw_val, y_val)
        val_scores = cal.transform(raw_val)

        # per-source group labels (val candidates come from the TRAIN sources)
        gl_val = annotate_source(pairs_val, smap_train) if args.per_source_threshold else None

        # (b) optional borderline-band cross-encoder reranking of val scores.
        # A provisional coarse threshold defines the band; the final threshold
        # is then (re)tuned on the reranked scores for end-to-end consistency.
        if args.rerank:
            prov_t = tune_policy(val_scores, pairs_val, gt_val, grid_step=0.02)[0].global_threshold
            val_scores, _ = rerank_borderline(
                val_scores, pairs_val, s1_va_split, sx_train_combined, prov_t,
                band=args.rerank_band, model_name=args.rerank_model,
            )

        # (c) tune the policy on (possibly reranked, calibrated) val scores
        grid_step      = 0.005 if args.fine_threshold else 0.05
        abstain_deltas = None if args.abstain else [0.0]
        policy, best_f05 = tune_policy(
            val_scores, pairs_val, gt_val,
            group_labels=gl_val, labels=y_val,
            grid_step=grid_step, abstain_deltas=abstain_deltas,
        )

        # (d) France-robust conservative margin (unseen country has no labels to
        # tune against, so we shift the boundary up by the LOCO-derived / manual
        # margin). Applied to the global and every per-group threshold.
        adj_policy = ThresholdPolicy(
            global_threshold=min(0.999, policy.global_threshold + france_margin),
            group_thresholds={g: min(0.999, t + france_margin)
                              for g, t in policy.group_thresholds.items()},
            abstain_delta=policy.abstain_delta,
        )
        best_threshold = adj_policy.global_threshold
        logger.info(
            "Policy: val F0.5=%.4f | global_t %.3f→%.3f (france_margin +%.3f) | "
            "abstain_delta=%.3f | per-source=%s",
            best_f05, policy.global_threshold, adj_policy.global_threshold,
            france_margin, adj_policy.abstain_delta, bool(adj_policy.group_thresholds),
        )

        # (e) optional error-analysis bucketing on val (in-distribution policy,
        # i.e. WITHOUT the France margin, so buckets reflect real US/India behaviour)
        if args.error_analysis:
            logger.info("=== Step 9b: Error-analysis bucketing (val split) ===")
            val_preds = apply_policy(val_scores, pairs_val, policy, gl_val)
            bucket_errors(
                pairs_val, val_preds, gt_val, s1_va_split, sx_train_combined,
                country_map=country_map_all,
            )

    # -----------------------------------------------------------------------
    # 10. Inference on TEST set (with Checkpointing)
    # -----------------------------------------------------------------------
    logger.info("=== Step 10: Test set inference ===")
    test_out_matching = output_dir / "matching_results.tsv"
    test_out_candidate = output_dir / "candidate_pairs.tsv"

    if resume and checkpoint_exists(test_out_matching) and checkpoint_exists(test_out_candidate):
        logger.info("Found existing matching_results.tsv and candidate_pairs.tsv at %s. Skipping test inference.", output_dir)
    else:
        # Free training-side frames before loading the test set
        del (s1_train, s2_train, s3_train,
             X_train_df, X_val_df, X_fit, pairs_train, pairs_val,
             candidates_train, candidates_val, candidates_train_a, candidates_val_a,
             s1_tr_split, s1_va_split)
        gc.collect()

        sx_test_combined = pd.concat([s2_test, s3_test], ignore_index=True)

        candidates_test = build_candidate_pairs(
            s1_test, s2_test, s3_test,
            output_dir=output_dir / "blocking_test",
            embed_threshold=args.embed_threshold if args.run_embedding else 0,
            run_embedding=args.run_embedding,
            resume=resume,
        )

        pairs_test = _explode_candidates(candidates_test)
        X_test_path = ckpt_test / "X_test.parquet"
        emb_test_path = ckpt_test / "emb_map_test.pkl"

        if resume and checkpoint_exists(X_test_path):
            logger.info("Found test feature matrix in checkpoint (%s). Loading...", X_test_path)
            X_test_df = load_dataframe(X_test_path)
        else:
            emb_map_test = None
            if args.run_embedding:
                emb_map_test = compute_embedding_cosine_map(
                    pairs_test, s1_test, sx_test_combined,
                    cache_path=emb_test_path, resume=resume,
                )
            X_test_df = build_feature_matrix(
                pairs_test, s1_test, sx_test_combined,
                embedding_cosine_map=emb_map_test, tfidf_vectorizer=tfidf_vec,
            )
            save_dataframe(X_test_df, X_test_path)

        if not use_advanced:
            preds_test = predict(clf, X_test_df, threshold=best_threshold)
        else:
            raw_test    = clf.predict_proba(X_test_df[FEATURE_COLS])[:, 1]
            test_scores = cal.transform(raw_test)
            smap_test   = source_map(s2_test, s3_test)
            gl_test     = annotate_source(pairs_test, smap_test) if args.per_source_threshold else None
            if args.rerank:
                test_scores, _ = rerank_borderline(
                    test_scores, pairs_test, s1_test, sx_test_combined,
                    adj_policy.global_threshold,
                    band=args.rerank_band, model_name=args.rerank_model,
                )
            preds_test = apply_policy(test_scores, pairs_test, adj_policy, gl_test)

        _write_output(
            pairs_test, preds_test,
            s1_all_ids=s1_test["entity_id"].tolist(),
            output_dir=output_dir,
            candidates=candidates_test,
            split="test",
        )

    # -----------------------------------------------------------------------
    # 11. Validate submission
    # -----------------------------------------------------------------------
    logger.info("=== Step 11: Validating submission ===")
    validate_script = _find_validator_script(test_dir)
    if validate_script is not None:
        logger.info("Running validator from %s ...", validate_script)
        result = subprocess.run(
            [
                sys.executable, str(validate_script),
                "--matching", str(output_dir / "matching_results.tsv"),
                "--candidate", str(output_dir / "candidate_pairs.tsv"),
                "--test-dir", str(test_dir),
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode == 0:
            logger.info("Validator: PASS ✓")
        else:
            logger.error("Validator: FAIL ✗\n%s", result.stderr)
    else:
        logger.warning("validate_submission.py not found in search paths — skipping validation.")

    logger.info(
        "Pipeline complete.  Val F_0.5=%.4f  threshold=%.2f",
        best_f05, best_threshold,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Entity Resolution Pipeline"
    )
    p.add_argument("--train-dir",       default="dataset/train", help="Training data directory")
    p.add_argument("--test-dir",        default="dataset/test",  help="Test data directory")
    p.add_argument("--output-dir",      default="output",        help="Output directory")
    p.add_argument("--val-fraction",    type=float, default=0.2, help="Validation split fraction")
    p.add_argument("--neg-ratio",       type=int,   default=3,   help="Negative-to-positive ratio for training")
    p.add_argument("--embed-threshold", type=int,   default=0,   help="Stage B gate: -1=embed all S1 entities (GPU full-corpus mode); >=0=only embed entities with <=N classical candidates (CPU gated mode)")
    p.add_argument("--run-embedding",   action="store_true",      help="Enable Stage B embedding ANN")

    # --- Checkpoint / resume infrastructure ---
    p.add_argument("--resume",          dest="resume", action="store_true", default=True, help="Resume from checkpoints if available (default: True)")
    p.add_argument("--no-resume",       dest="resume", action="store_false", help="Ignore checkpoints and recompute from scratch")
    p.add_argument("--checkpoint-dir",  default=None,            help="Directory to store intermediate checkpoints (default: <output-dir>/checkpoints)")
    p.add_argument("--checkpoint-freq", type=int,   default=25,  help="Save model checkpoint every N trees (default: 25)")

    # --- Optional precision-first decision levers (all OFF by default; with
    #     none set, steps 9–10 reproduce the frozen baseline exactly). ---
    grp = p.add_argument_group("decision levers (opt-in; default path unchanged)")
    grp.add_argument("--calibrate", action="store_true",
                     help="Fit a probability calibrator on the val split before thresholding.")
    grp.add_argument("--calib-method", default="isotonic", choices=["isotonic", "platt", "sigmoid"],
                     help="Calibration method when --calibrate is set (default: isotonic).")
    grp.add_argument("--fine-threshold", action="store_true",
                     help="Use the fine 0.005 F0.5 grid instead of the coarse 0.05 sweep.")
    grp.add_argument("--per-source-threshold", action="store_true",
                     help="Tune separate S2/S3 thresholds via coordinate ascent on joint F0.5.")
    grp.add_argument("--abstain", action="store_true",
                     help="Sweep an abstention margin {0,.02,.05,.1}: an entity commits only if its best candidate clears threshold by the margin.")
    grp.add_argument("--france-margin", type=float, default=0.0,
                     help="Add this conservative margin to the final threshold for unseen-country robustness. 0 = use LOCO's recommendation when --loco is set, else none.")

    # --- Cross-encoder reranking of the borderline band ---
    grp.add_argument("--rerank", action="store_true",
                     help="Rerank borderline-band pairs with a multilingual cross-encoder.")
    grp.add_argument("--rerank-band", type=float, default=0.10,
                     help="Half-width of the reranking band around the threshold (default 0.10).")
    grp.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL,
                     help=f"Cross-encoder model id (default: {DEFAULT_RERANK_MODEL}; verify MIT/Apache & <=8B on the lab box).")

    # --- Diagnostics ---
    grp.add_argument("--loco", action="store_true",
                     help="Run leave-one-country-out transfer evaluation on the train split.")
    grp.add_argument("--error-analysis", action="store_true",
                     help="Bucket val-split errors (singleton/chain/indic/blocking/classifier).")

    # --- Class-imbalance experiment ---
    grp.add_argument("--drop-class-weight", action="store_true",
                     help="Train LightGBM without class_weight='balanced'.")
    grp.add_argument("--scale-pos-weight", type=float, default=None,
                     help="Set LightGBM scale_pos_weight (implies class_weight=None).")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
