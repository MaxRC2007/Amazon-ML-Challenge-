"""
validation.py
-------------
Unseen-country validation (LOCO) + strategy diagnostics + error bucketing.

Why this module exists
----------------------
The baseline holds out a random 20% of ground-truth entities for the val split.
That split is US+India only, so it measures NOTHING about how the decision
boundary transfers to France (~15% of the test, entirely unseen in training).
Blind spots on 15% of the data are exactly where a 0.989 submission loses to a
0.991 one.

Leave-one-country-out (LOCO) simulates the France situation with a country we
*do* have labels for: train + tune the threshold on one country, then measure
F0.5 on a completely held-out country. The in-distribution → held-out F0.5 drop
(and the shift in the optimal threshold) tells you how conservative the
production decision boundary must be to survive an unseen country.

Public API:
    singleton_ratio(gt) -> float
    s1_country_map(s1_df) -> dict[eid, country]
    source_map(s2_df, s3_df) -> dict[sx_id, "S2"|"S3"]
    annotate_source(pairs, source_map) -> np.ndarray
    annotate_country(pairs, country_map) -> np.ndarray
    report_diagnostics(gt, recall_ceiling) -> dict     # the two gate numbers
    loco_evaluate(X, y, gt, country_map, ...) -> dict   # transfer gap + margin
    bucket_errors(pairs, preds, gt, s1_df, sx_df, ...) -> collections.Counter
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from .features import FEATURE_COLS
from .model import f05_score, train
from .policy import ProbabilityCalibrator, tune_policy, apply_policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def singleton_ratio(gt: pd.DataFrame) -> float:
    """Fraction of S1 entities in `gt` that are true no-matches (singletons)."""
    n = len(gt)
    if n == 0:
        return 0.0
    empty = gt["matched_entity_ids"].fillna("").astype(str).str.strip() == ""
    return float(empty.mean())


def s1_country_map(s1_df: pd.DataFrame) -> dict:
    return dict(zip(s1_df["entity_id"], s1_df["country"]))


def source_map(s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> dict:
    m = {eid: "S2" for eid in s2_df["entity_id"]}
    m.update({eid: "S3" for eid in s3_df["entity_id"]})
    return m


def annotate_source(pairs: pd.DataFrame, smap: dict) -> np.ndarray:
    return np.array([smap.get(x, "S?") for x in pairs["candidate_entity_id"]], dtype=object)


def annotate_country(pairs: pd.DataFrame, cmap: dict) -> np.ndarray:
    return np.array([cmap.get(x, "?") for x in pairs["source1_entity_id"]], dtype=object)


def report_diagnostics(gt: pd.DataFrame, recall_ceiling: Optional[float] = None) -> dict:
    """
    Log the two numbers that decide the whole strategy:
      - singleton ratio (how much of the score is 'predict empty correctly')
      - blocking recall ceiling (the achievable-recall upper bound)
    and print an explicit recall-bound vs precision-bound verdict.
    """
    sr = singleton_ratio(gt)
    logger.info("DIAGNOSTIC — singleton ratio: %.4f (%.1f%% of S1 are true no-matches)",
                sr, sr * 100)
    verdict = "unknown (recall ceiling not supplied)"
    if recall_ceiling is not None:
        logger.info("DIAGNOSTIC — blocking recall ceiling: %.4f", recall_ceiling)
        if recall_ceiling >= 0.97 and sr >= 0.80:
            verdict = ("PRECISION-BOUND — recall ceiling high and singletons dominate; "
                       "spend effort on thresholding/calibration/abstention/reranking, "
                       "NOT on generating more candidates.")
        elif recall_ceiling < 0.90:
            verdict = ("RECALL-BOUND — blocking is losing true matches; improve blocking "
                       "(Stage B / more keys) before tuning the decision boundary.")
        else:
            verdict = "MIXED — improve both blocking recall and decision precision."
    logger.info("DIAGNOSTIC — verdict: %s", verdict)
    return {"singleton_ratio": sr, "recall_ceiling": recall_ceiling, "verdict": verdict}


# ---------------------------------------------------------------------------
# LOCO — leave-one-country-out transfer evaluation
# ---------------------------------------------------------------------------

def _downsample_idx(y: np.ndarray, neg_ratio: int, seed: int) -> np.ndarray:
    """Indices keeping all positives + neg_ratio× negatives (matches pipeline)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    max_neg = len(pos) * neg_ratio
    if len(neg) > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    return np.concatenate([pos, neg])


def _gt_subset(gt: pd.DataFrame, entities: set) -> pd.DataFrame:
    return gt[gt["source1_entity_id"].isin(entities)].reset_index(drop=True)


def loco_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    gt: pd.DataFrame,
    country_map: dict,
    source_labels: Optional[np.ndarray] = None,
    neg_ratio: int = 3,
    calib_method: str = "isotonic",
    min_country_entities: int = 200,
    seed: int = 42,
) -> dict:
    """
    Leave-one-country-out transfer evaluation.

    For each country c with enough entities: train + calibrate + tune the
    decision policy on the OTHER countries only, then measure F0.5 on c
    (a stand-in for unseen France). Reports the in-distribution → held-out
    F0.5 drop and the shift in the optimal threshold — the latter is the
    conservative margin to add to the production threshold for France
    robustness.

    X : labelled feature frame (source1_entity_id, candidate_entity_id, FEATURE_COLS).
    y : 0/1 labels aligned to X.
    gt: ground truth covering all entities in X's country pool (incl. singletons).
    country_map : entity_id -> country (from s1_country_map).
    source_labels : optional per-row 'S2'/'S3' for per-source thresholds.
    """
    y = np.asarray(y)
    s1_ids = X["source1_entity_id"].to_numpy()
    row_country = np.array([country_map.get(e, "?") for e in s1_ids], dtype=object)

    all_entities = set(gt["source1_entity_id"])
    counts = pd.Series([country_map.get(e, "?") for e in all_entities]).value_counts()
    fold_countries = [c for c, n in counts.items() if c != "?" and n >= min_country_entities]
    logger.info("LOCO folds (countries with >=%d entities): %s",
                min_country_entities, fold_countries)

    results = {}
    for c in fold_countries:
        held_mask = row_country == c
        train_mask = ~held_mask
        if held_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        # Split TRAIN countries' entities into model-fit vs threshold-tune,
        # so the threshold is never tuned on data the model was fit on.
        train_entities = [e for e in all_entities if country_map.get(e) != c]
        rng = np.random.default_rng(seed)
        train_entities = list(train_entities)
        rng.shuffle(train_entities)
        cut = int(0.75 * len(train_entities))
        fit_ent = set(train_entities[:cut])
        tune_ent = set(train_entities[cut:])

        fit_row = np.array([e in fit_ent for e in s1_ids])
        tune_row = np.array([e in tune_ent for e in s1_ids])

        # --- fit model on downsampled fit rows ---
        X_fit = X[fit_row].reset_index(drop=True)
        y_fit = y[fit_row]
        keep = _downsample_idx(y_fit, neg_ratio, seed)
        clf = train(X_fit.iloc[keep].reset_index(drop=True), y_fit[keep])

        # --- calibrate on tune rows (non-downsampled) ---
        raw_tune = clf.predict_proba(X[tune_row][FEATURE_COLS])[:, 1]
        cal = ProbabilityCalibrator(calib_method).fit(raw_tune, y[tune_row])
        scores_tune = cal.transform(raw_tune)

        pairs_tune = X[tune_row][["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        gl_tune = source_labels[tune_row] if source_labels is not None else None
        gt_tune = _gt_subset(gt, tune_ent)
        policy, in_f05 = tune_policy(
            scores_tune, pairs_tune, gt_tune,
            group_labels=gl_tune, labels=y[tune_row],
        )

        # --- evaluate on held-out country c ---
        raw_c = clf.predict_proba(X[held_mask][FEATURE_COLS])[:, 1]
        scores_c = cal.transform(raw_c)
        pairs_c = X[held_mask][["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        gl_c = source_labels[held_mask] if source_labels is not None else None
        gt_c = _gt_subset(gt, set(gt["source1_entity_id"]) & {e for e in all_entities if country_map.get(e) == c})
        preds_c = apply_policy(scores_c, pairs_c, policy, gl_c)
        held_f05 = f05_score(pairs_c, preds_c, gt_c)

        # --- held-out's OWN optimal policy (diagnostic: how conservative to be) ---
        held_policy, held_opt_f05 = tune_policy(
            scores_c, pairs_c, gt_c, group_labels=gl_c, labels=y[held_mask],
        )
        thr_shift = held_policy.global_threshold - policy.global_threshold

        results[c] = {
            "in_distribution_f05": round(in_f05, 4),
            "held_out_f05": round(held_f05, 4),
            "f05_gap": round(in_f05 - held_f05, 4),
            "in_dist_threshold": round(policy.global_threshold, 3),
            "held_out_optimal_threshold": round(held_policy.global_threshold, 3),
            "threshold_shift_needed": round(thr_shift, 3),
            "held_out_optimal_f05": round(held_opt_f05, 4),
        }
        logger.info("LOCO[hold-out=%s]: in-dist F0.5=%.4f  held-out F0.5=%.4f  "
                    "gap=%.4f  thr %.3f→%.3f (Δ%.3f)", c,
                    in_f05, held_f05, in_f05 - held_f05,
                    policy.global_threshold, held_policy.global_threshold, thr_shift)

    # Recommended France-robust margin: the largest positive threshold shift any
    # held-out country needed (clamped >= 0). Add this to the production threshold.
    shifts = [r["threshold_shift_needed"] for r in results.values()]
    recommended_margin = round(max([s for s in shifts] + [0.0]), 3) if shifts else 0.0
    gaps = [r["f05_gap"] for r in results.values()]
    summary = {
        "folds": results,
        "recommended_conservative_margin": recommended_margin,
        "worst_f05_gap": round(max(gaps), 4) if gaps else 0.0,
        "mean_held_out_f05": round(float(np.mean([r["held_out_f05"] for r in results.values()])), 4) if results else 0.0,
    }
    logger.info("LOCO summary: worst gap=%.4f  recommended conservative margin=+%.3f  "
                "mean held-out F0.5=%.4f",
                summary["worst_f05_gap"], recommended_margin, summary["mean_held_out_f05"])
    return summary


# ---------------------------------------------------------------------------
# Error analysis — bucket the entities you get wrong
# ---------------------------------------------------------------------------

def _has_native_script(s: str) -> bool:
    """True if the string carries non-ASCII (e.g. Devanagari/Tamil/…) characters."""
    return isinstance(s, str) and any(ord(ch) > 127 for ch in s)


def _tokset(s: str) -> set:
    return set(s.split()) if isinstance(s, str) else set()


def _jacc(a: str, b: str) -> float:
    sa, sb = _tokset(a), _tokset(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def bucket_errors(
    pairs: pd.DataFrame,
    preds: np.ndarray,
    gt: pd.DataFrame,
    s1_df: pd.DataFrame,
    sx_df: pd.DataFrame,
    country_map: Optional[dict] = None,
    max_examples: int = 20,
) -> Counter:
    """
    On an evaluation split, classify every entity scored below 1.0 into a
    failure bucket so effort goes to the biggest one first (the error-analysis
    loop that actually moves a 0.0016-spread leaderboard).

    Buckets:
      singleton_false_merge — a true no-match got at least one predicted match
      chain_false_merge     — false merge where names match but addresses differ
      indic_false_merge     — false merge involving a native-script name
      other_false_merge     — remaining false merges
      blocking_miss         — a true match was never even a candidate
      indic_miss            — a missed true match involving a native-script name
      classifier_miss       — a candidate true match the classifier rejected
    """
    preds = np.asarray(preds)
    s1n = s1_df.set_index("entity_id")
    sxn = sx_df.set_index("entity_id")

    def s1_attr(eid, col):
        try:
            return s1n.at[eid, col]
        except Exception:
            return ""

    def sx_attr(eid, col):
        try:
            return sxn.at[eid, col]
        except Exception:
            return ""

    # predicted matches + full candidate set per entity
    pred_matches: dict = {}
    cand_set: dict = {}
    s1_arr = pairs["source1_entity_id"].to_numpy()
    sx_arr = pairs["candidate_entity_id"].to_numpy()
    for i in range(len(pairs)):
        e, x = s1_arr[i], sx_arr[i]
        cand_set.setdefault(e, set()).add(x)
        if preds[i] == 1:
            pred_matches.setdefault(e, set()).add(x)

    buckets: Counter = Counter()
    examples: dict = {}
    for row in gt.itertuples(index=False):
        e = row.source1_entity_id
        matched = row.matched_entity_ids
        true_set = set(matched.split(",")) if (isinstance(matched, str) and matched.strip()) else set()
        pred_set = pred_matches.get(e, set())
        cands = cand_set.get(e, set())

        # singleton
        if not true_set:
            if pred_set:
                buckets["singleton_false_merge"] += 1
                examples.setdefault("singleton_false_merge", []).append((e, sorted(pred_set)[:3]))
            continue

        fp = pred_set - true_set
        fn = true_set - pred_set
        if not fp and not fn:
            continue  # scored 1.0 — skip

        s1_name = s1_attr(e, "norm_name")
        for x in fp:
            x_bizname = sx_attr(x, "business_name")
            if _has_native_script(x_bizname):
                buckets["indic_false_merge"] += 1
            elif _jacc(s1_name, sx_attr(x, "norm_name")) >= 0.9 and \
                    _jacc(s1_attr(e, "norm_address"), sx_attr(x, "norm_address")) < 0.3:
                buckets["chain_false_merge"] += 1
            else:
                buckets["other_false_merge"] += 1

        for x in fn:
            if x not in cands:
                buckets["blocking_miss"] += 1
            elif _has_native_script(sx_attr(x, "business_name")):
                buckets["indic_miss"] += 1
            else:
                buckets["classifier_miss"] += 1

    logger.info("ERROR BUCKETS (largest first):")
    for name, cnt in buckets.most_common():
        logger.info("   %-24s %d", name, cnt)
        for ex in examples.get(name, [])[:3]:
            logger.info("       e.g. %s", ex)
    return buckets


