"""
model.py
--------
LightGBM binary classifier for entity matching.

Training:
  - Positive pairs: from train_ground_truth.tsv (S1 ↔ S2/S3 matches)
  - Negative pairs: sampled from candidate set (candidates that are NOT
    true matches) — at a configurable negative-to-positive ratio.

Threshold tuning:
  - The default LightGBM threshold of 0.5 is NOT used.
  - We sweep probability cutoffs and select the threshold that maximises
    F_0.5 on a held-out validation split directly.

Public API:
    train(features_train, labels_train) -> LGBMClassifier
    predict_proba(model, features) -> np.ndarray
    tune_threshold(model, features_val, pairs_val, ground_truth_val) -> float
    predict(model, features, threshold) -> np.ndarray  (0/1 labels)
    f05_score(pairs, predictions, ground_truth) -> float
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F_0.5 implementation (macro-averaged, per-S1-entity)
# ---------------------------------------------------------------------------

def f05_score(
    pairs: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth: pd.DataFrame,
) -> float:
    """
    Compute macro-averaged F_0.5 score.

    Parameters
    ----------
    pairs       : DataFrame with [source1_entity_id, candidate_entity_id]
    predictions : Binary array (1 = match predicted, 0 = no match),
                  same length and order as pairs.
    ground_truth: DataFrame with [source1_entity_id, matched_entity_ids]

    Returns
    -------
    float — macro-averaged F_0.5 across all S1 entities in ground_truth.
    """
    # Build predicted matches dict
    pred_matches: dict[str, set[str]] = {}
    pairs_arr = pairs[["source1_entity_id", "candidate_entity_id"]].values
    for (s1_id, sx_id), pred in zip(pairs_arr, predictions):
        if pred == 1:
            pred_matches.setdefault(s1_id, set()).add(sx_id)

    # Build ground-truth dict
    gt_map: dict[str, set[str]] = {}
    for row in ground_truth.itertuples(index=False):
        s1_id   = row.source1_entity_id
        matched = row.matched_entity_ids
        gt_map[s1_id] = set(matched.split(",")) if matched else set()

    beta  = 0.5
    beta2 = beta ** 2

    scores = []
    for s1_id, true_set in gt_map.items():
        pred_set = pred_matches.get(s1_id, set())

        # Singleton case: true_set is empty
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f = (1 + beta2) * precision * recall / (beta2 * precision + recall)
            scores.append(f)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_pairs(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    neg_ratio: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a labelled pair set from candidates + ground truth.

    Returns
    -------
    (pairs_df, labels)
    pairs_df : DataFrame with [source1_entity_id, candidate_entity_id]
    labels   : 1d numpy array of 0/1 labels
    """
    # Build ground-truth set for O(1) lookup
    gt_set: set[tuple[str, str]] = set()
    for row in ground_truth.itertuples(index=False):
        s1_id   = row.source1_entity_id
        matched = row.matched_entity_ids
        if matched:
            for sx_id in matched.split(","):
                gt_set.add((s1_id, sx_id.strip()))

    # Explode candidate_pairs → one row per pair
    logger.info("Exploding candidate pairs ...")
    exploded_rows = []
    for row in candidates.itertuples(index=False):
        s1_id    = row.source1_entity_id
        cand_str = row.candidate_entity_ids
        if not cand_str:
            continue
        for sx_id in cand_str.split(","):
            sx_id = sx_id.strip()
            if sx_id:
                exploded_rows.append((s1_id, sx_id))

    pairs_df = pd.DataFrame(exploded_rows, columns=["source1_entity_id", "candidate_entity_id"])
    pairs_df["label"] = pairs_df.apply(
        lambda r: 1 if (r["source1_entity_id"], r["candidate_entity_id"]) in gt_set else 0,
        axis=1,
    )

    n_pos = (pairs_df["label"] == 1).sum()
    n_neg = (pairs_df["label"] == 0).sum()
    logger.info("Candidate pairs: %d positives, %d negatives", n_pos, n_neg)

    # Downsample negatives
    max_neg = n_pos * neg_ratio
    if n_neg > max_neg:
        pos_idx = pairs_df[pairs_df["label"] == 1].index
        neg_idx = pairs_df[pairs_df["label"] == 0].sample(
            max_neg, random_state=random_state
        ).index
        pairs_df = pairs_df.loc[list(pos_idx) + list(neg_idx)].reset_index(drop=True)
        logger.info(
            "Downsampled to %d positives + %d negatives (ratio 1:%d)",
            n_pos, max_neg, neg_ratio,
        )

    labels = pairs_df["label"].values
    pairs_out = pairs_df[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
    return pairs_out, labels


class LGBMCheckpointCallback:
    """Callback to periodically save LightGBM booster and training metadata."""
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_freq: int = 25,
        base_trees: int = 0,
        total_target_trees: int = 500,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_freq = checkpoint_freq
        self.base_trees = base_trees
        self.total_target_trees = total_target_trees
        self.latest_booster_path = self.checkpoint_dir / "booster_checkpoint_latest.txt"
        self.meta_path = self.checkpoint_dir / "model_checkpoint.json"

    def __call__(self, env):
        current_trees = self.base_trees + env.iteration + 1
        is_periodic = (current_trees % self.checkpoint_freq == 0)
        is_final = (env.iteration == env.end_iteration - 1)
        if is_periodic or is_final:
            try:
                env.model.save_model(str(self.latest_booster_path))
                import json
                meta = {
                    "completed_trees": current_trees,
                    "target_trees": self.total_target_trees,
                    "iteration_in_run": env.iteration,
                    "is_final": is_final,
                }
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                logger.info(
                    "Model checkpoint saved at tree %d / %d -> %s",
                    current_trees, self.total_target_trees, self.latest_booster_path
                )
            except Exception as e:
                logger.warning("Failed to save model checkpoint: %s", e)


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_params: Optional[dict] = None,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_freq: int = 25,
    resume: bool = True,
) -> LGBMClassifier:
    """
    Train a LightGBM binary classifier with checkpointing and resume support.

    Parameters
    ----------
    X_train        : Feature DataFrame (columns must include FEATURE_COLS)
    y_train        : Binary labels
    model_params   : Optional LightGBM hyperparameters (defaults used if None)
    checkpoint_dir : Directory to store model checkpoints and metadata
    checkpoint_freq: Interval of trees at which to checkpoint
    resume         : If True, resume from existing checkpoint if available

    Returns
    -------
    Trained LGBMClassifier.
    """
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if model_params:
        default_params.update(model_params)

    target_trees = default_params["n_estimators"]

    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = ckpt_dir / "model.pkl"
        latest_booster_path = ckpt_dir / "booster_checkpoint_latest.txt"
        meta_path = ckpt_dir / "model_checkpoint.json"

        # 1. Check if fully trained model already exists
        if resume and final_model_path.is_file() and final_model_path.stat().st_size > 0:
            try:
                import json
                is_completed = False
                if meta_path.is_file():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("is_final", False) or meta.get("completed_trees", 0) >= target_trees:
                        is_completed = True
                clf = load_model(final_model_path)
                if is_completed or (hasattr(clf, "booster_") and clf.booster_.num_trees() >= target_trees):
                    logger.info(
                        "Found fully-trained model at %s (%d trees). Skipping training.",
                        final_model_path, getattr(clf.booster_, "num_trees", lambda: 0)(),
                    )
                    return clf
            except Exception as e:
                logger.warning("Could not load existing %s: %s. Checking booster checkpoint.", final_model_path, e)

        # 2. Check if intermediate booster checkpoint exists
        if resume and latest_booster_path.is_file() and latest_booster_path.stat().st_size > 0:
            try:
                import lightgbm as lgb
                existing_booster = lgb.Booster(model_file=str(latest_booster_path))
                completed_trees = existing_booster.num_trees()
                if completed_trees >= target_trees:
                    logger.info(
                        "Booster checkpoint has reached target trees (%d >= %d).",
                        completed_trees, target_trees,
                    )
                    if final_model_path.is_file():
                        try:
                            return load_model(final_model_path)
                        except Exception:
                            pass
                    clf = LGBMClassifier(**default_params)
                    clf._Booster = existing_booster
                    clf.booster_ = existing_booster
                    clf.fitted_ = True
                    clf.classes_ = np.unique(y_train)
                    clf.n_classes_ = len(clf.classes_)
                    clf._n_classes = len(clf.classes_)
                    save_model(clf, final_model_path)
                    return clf

                remaining_trees = target_trees - completed_trees
                logger.info(
                    "Resuming LightGBM training: %d / %d trees completed. Training remaining %d trees...",
                    completed_trees, target_trees, remaining_trees,
                )
                run_params = dict(default_params)
                run_params["n_estimators"] = remaining_trees

                clf = LGBMClassifier(**run_params)
                cb = LGBMCheckpointCallback(
                    ckpt_dir,
                    checkpoint_freq=checkpoint_freq,
                    base_trees=completed_trees,
                    total_target_trees=target_trees,
                )
                clf.fit(
                    X_train[FEATURE_COLS],
                    y_train,
                    init_model=str(latest_booster_path),
                    callbacks=[cb],
                )
                save_model(clf, final_model_path)
                logger.info("Resumed training finished. Total trees: %d", clf.booster_.num_trees())
                return clf
            except Exception as e:
                logger.warning("Failed to resume from %s (%s). Starting fresh.", latest_booster_path, e)

        # 3. Fresh training with checkpointing
        clf = LGBMClassifier(**default_params)
        cb = LGBMCheckpointCallback(
            ckpt_dir,
            checkpoint_freq=checkpoint_freq,
            base_trees=0,
            total_target_trees=target_trees,
        )
        clf.fit(X_train[FEATURE_COLS], y_train, callbacks=[cb])
        save_model(clf, final_model_path)
        logger.info("LightGBM training complete. Total trees: %d", clf.booster_.num_trees())
        return clf

    # No checkpoint dir specified — standard training
    clf = LGBMClassifier(**default_params)
    clf.fit(X_train[FEATURE_COLS], y_train)
    logger.info("LightGBM training complete.")
    return clf


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    model: LGBMClassifier,
    features_val: pd.DataFrame,
    pairs_val: pd.DataFrame,
    ground_truth_val: pd.DataFrame,
    thresholds: Optional[list[float]] = None,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = True,
) -> tuple[float, float]:
    """
    Sweep probability thresholds and return the one that maximises F_0.5
    on the validation split. Supports loading from/saving to checkpoint.

    Returns
    -------
    (best_threshold, best_f05)
    """
    if checkpoint_path is not None and resume:
        ckpt = Path(checkpoint_path)
        if ckpt.is_file() and ckpt.stat().st_size > 0:
            try:
                import json
                with open(ckpt, "r", encoding="utf-8") as f:
                    data = json.load(f)
                best_t = data["best_threshold"]
                best_f = data["best_f05"]
                logger.info("Loaded tuned threshold from %s: threshold=%.2f, val F_0.5=%.4f", ckpt, best_t, best_f)
                return best_t, best_f
            except Exception as e:
                logger.warning("Failed to load threshold checkpoint: %s. Re-tuning.", e)

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    probas = model.predict_proba(features_val[FEATURE_COLS])[:, 1]

    best_threshold = 0.5
    best_f05       = -1.0

    for t in thresholds:
        preds = (probas >= t).astype(int)
        score = f05_score(pairs_val, preds, ground_truth_val)
        logger.debug("threshold=%.2f  F_0.5=%.4f", t, score)
        if score > best_f05:
            best_f05       = score
            best_threshold = t

    logger.info(
        "Best threshold: %.2f  |  Best F_0.5 (val): %.4f",
        best_threshold, best_f05,
    )

    if checkpoint_path is not None:
        try:
            import json
            ckpt = Path(checkpoint_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump({"best_threshold": best_threshold, "best_f05": best_f05}, f, indent=2)
            logger.info("Saved threshold checkpoint to %s", ckpt)
        except Exception as e:
            logger.warning("Failed to save threshold checkpoint: %s", e)

    return best_threshold, best_f05


def predict(
    model: LGBMClassifier,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary predictions using the given threshold."""
    probas = model.predict_proba(features[FEATURE_COLS])[:, 1]
    return (probas >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_model(model: LGBMClassifier, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> LGBMClassifier:
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Model loaded from %s", path)
    return model
