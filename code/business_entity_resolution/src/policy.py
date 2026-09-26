"""
policy.py
---------
Precision-first decision policy for the entity-resolution classifier.

Three upgrades over the coarse 0.05-step F0.5 threshold sweep in model.py:

  1. Probability calibration (isotonic / Platt).  The LightGBM model is trained
     on 3:1 negative-downsampled data, so its raw probabilities are distorted
     relative to the true candidate prior.  We fit a calibrator on a
     non-downsampled validation slice so the scores mean what a threshold
     expects them to.  (Note: the baseline already TUNES the threshold on the
     full, non-downsampled val candidate set — the distortion that remains is
     in the *probabilities themselves* because the model was fit on downsampled
     data, which is exactly what calibration corrects.)

  2. Fine, per-group thresholds.  A <=0.005-grid F0.5 search, optionally tuned
     separately per source (S2 vs S3) and/or per country, because the
     precision/recall trade-off differs across them.  Per-group thresholds are
     fitted by coordinate ascent on the *joint* per-entity F0.5 (not in
     isolation), since one S1 entity can mix S2 and S3 candidates.

  3. Abstention (margin) rule.  For a precision-weighted metric on a
     singleton-heavy set, if an S1 entity's BEST candidate only marginally
     clears its threshold, emit nothing for that entity.  "Stay empty when
     unsure" has high expected value when ~85% of entities are true no-matches.

All threshold selection uses model.f05_score (the exact challenge metric with
the singleton rule) so we optimise the target directly.

Public API:
    ProbabilityCalibrator(method).fit(probas, labels).transform(probas)
    tune_policy(scores, pairs, gt, group_labels=None, ...) -> (ThresholdPolicy, f05)
    apply_policy(scores, pairs, policy, group_labels=None) -> np.ndarray (0/1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .model import f05_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """Isotonic (default) or Platt/sigmoid calibration of classifier scores."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._model = None

    def fit(self, probas, labels) -> "ProbabilityCalibrator":
        probas = np.asarray(probas, dtype="float64")
        labels = np.asarray(labels, dtype="int64")
        if self.method in ("none", "identity"):
            # Explicit pass-through: lets the pipeline always hold a calibrator
            # object (uniform code path) while leaving scores untouched.
            self._model = None
            return self
        if len(np.unique(labels)) < 2:
            logger.warning("Calibrator: only one label class present — identity calibration.")
            self._model = None
            return self
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            m.fit(probas, labels)
        elif self.method in ("platt", "sigmoid"):
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            m.fit(probas.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method!r}")
        self._model = m
        logger.info("Calibrator fitted (method=%s) on %d val pairs.", self.method, len(probas))
        return self

    def transform(self, probas) -> np.ndarray:
        probas = np.asarray(probas, dtype="float64")
        if self._model is None:
            return probas
        if self.method == "isotonic":
            return np.clip(self._model.predict(probas), 0.0, 1.0)
        return self._model.predict_proba(probas.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Threshold policy
# ---------------------------------------------------------------------------

@dataclass
class ThresholdPolicy:
    """A fine, optionally per-group decision policy with an abstention margin."""
    global_threshold: float = 0.5
    group_thresholds: dict = field(default_factory=dict)  # group_label -> threshold
    abstain_delta: float = 0.0

    def threshold_for(self, group_label) -> float:
        return self.group_thresholds.get(group_label, self.global_threshold)


def _pair_thresholds(
    pairs: pd.DataFrame,
    policy: ThresholdPolicy,
    group_labels: Optional[np.ndarray],
) -> np.ndarray:
    """Per-pair applicable threshold, honouring group overrides when present."""
    n = len(pairs)
    if not policy.group_thresholds or group_labels is None:
        return np.full(n, policy.global_threshold, dtype="float64")
    return np.array(
        [policy.threshold_for(g) for g in group_labels], dtype="float64"
    )


def apply_policy(
    scores: np.ndarray,
    pairs: pd.DataFrame,
    policy: ThresholdPolicy,
    group_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Turn (already-calibrated) scores into 0/1 predictions under the policy.

    Steps:
      keep_pair = score >= threshold(group)
      abstention: for each S1 entity, if the best candidate's margin over its
                  own threshold is < abstain_delta, drop ALL of that entity's
                  matches (emit empty).
    """
    scores = np.asarray(scores, dtype="float64")
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    t_arr = _pair_thresholds(pairs, policy, group_labels)
    keep = scores >= t_arr

    if policy.abstain_delta > 0.0:
        margin = scores - t_arr
        s1 = pairs["source1_entity_id"].to_numpy()
        # Best margin per entity (only over pairs that clear their threshold).
        best: dict = {}
        for i in range(n):
            if keep[i]:
                m = margin[i]
                if m > best.get(s1[i], -np.inf):
                    best[s1[i]] = m
        committed = {e for e, m in best.items() if m >= policy.abstain_delta}
        for i in range(n):
            if keep[i] and s1[i] not in committed:
                keep[i] = False

    return keep.astype(np.int8)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def _default_grid(step: float = 0.005) -> np.ndarray:
    # Fine grid — the competition lives in the third decimal, so a 0.05 grid
    # (the model.py baseline) cannot resolve the difference between ranks.
    return np.round(np.arange(0.02, 0.985 + 1e-9, step), 4)


def _score(scores, pairs, gt, policy, group_labels) -> float:
    preds = apply_policy(scores, pairs, policy, group_labels)
    return f05_score(pairs, preds, gt)


def _tune_global(scores, pairs, gt, grid, abstain_delta) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in grid:
        pol = ThresholdPolicy(global_threshold=float(t), abstain_delta=abstain_delta)
        f = _score(scores, pairs, gt, pol, None)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def _eligible_groups(group_labels, labels, min_group_pos) -> list:
    """Groups with enough positive pairs to justify a dedicated threshold."""
    if group_labels is None or labels is None:
        return []
    labels = np.asarray(labels)
    out = []
    for g in pd.unique(pd.Series(group_labels)):
        mask = np.asarray(group_labels) == g
        if int(labels[mask].sum()) >= min_group_pos:
            out.append(g)
    return out


def tune_policy(
    scores: np.ndarray,
    pairs: pd.DataFrame,
    gt: pd.DataFrame,
    group_labels: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    grid_step: float = 0.005,
    abstain_deltas: Optional[list] = None,
    min_group_pos: int = 50,
    coord_passes: int = 2,
) -> tuple[ThresholdPolicy, float]:
    """
    Fit a precision-first ThresholdPolicy that maximises per-entity F0.5.

    scores        : already-calibrated match scores (0..1), aligned to pairs.
    pairs         : [source1_entity_id, candidate_entity_id].
    gt            : ground truth for the entities in `pairs`.
    group_labels  : optional per-pair group key (e.g. 'S2'/'S3' or country).
                    When given, thresholds are tuned per group by coordinate
                    ascent on the JOINT F0.5 (initialised at the global best).
    labels        : optional per-pair 0/1 labels — only used to decide which
                    groups have enough positives to earn their own threshold.
    abstain_deltas: candidate abstention margins to sweep (default {0, .02, .05, .1}).
    """
    grid = _default_grid(grid_step)
    if abstain_deltas is None:
        abstain_deltas = [0.0, 0.02, 0.05, 0.10]

    best_policy: Optional[ThresholdPolicy] = None
    best_f = -1.0

    for delta in abstain_deltas:
        # 1) global threshold at this abstention margin
        g_t, g_f = _tune_global(scores, pairs, gt, grid, delta)
        policy = ThresholdPolicy(global_threshold=g_t, abstain_delta=delta)
        cur_f = g_f

        # 2) per-group coordinate ascent (optional)
        groups = _eligible_groups(group_labels, labels, min_group_pos)
        if groups:
            policy.group_thresholds = {g: g_t for g in groups}
            for _ in range(coord_passes):
                improved = False
                for g in groups:
                    best_gt, best_gf = policy.group_thresholds[g], cur_f
                    for t in grid:
                        trial = dict(policy.group_thresholds)
                        trial[g] = float(t)
                        cand = ThresholdPolicy(g_t, trial, delta)
                        f = _score(scores, pairs, gt, cand, group_labels)
                        if f > best_gf:
                            best_gf, best_gt = f, float(t)
                    if best_gt != policy.group_thresholds[g]:
                        policy.group_thresholds[g] = best_gt
                        cur_f = best_gf
                        improved = True
                if not improved:
                    break

        if cur_f > best_f:
            best_f = cur_f
            best_policy = policy

    logger.info(
        "Best policy: global_t=%.3f  groups=%s  abstain_delta=%.3f  F0.5=%.4f",
        best_policy.global_threshold,
        {k: round(v, 3) for k, v in best_policy.group_thresholds.items()} or "—",
        best_policy.abstain_delta,
        best_f,
    )
    return best_policy, best_f


