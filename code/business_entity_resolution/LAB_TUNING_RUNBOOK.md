# Lab-Tuning Runbook — crossing macro F0.5 > 0.99

This runbook covers the six precision-first levers layered on top of the frozen
baseline (`a6110e2`). **Every lever is opt-in.** With no new flags, `pipeline.py`
reproduces the reviewed baseline exactly (steps 9–10 are byte-identical), so you
can turn levers on one at a time and measure each in isolation.

Golden rule (from the leaderboard analysis): **tune and judge on held-out-country
F0.5, never the in-distribution number.** The dataset is ~80–90% singletons and
the win lives in the France / Latin↔Indic / chain tails. If a change lifts
in-distribution F0.5 but not the held-out-country number, it is overfitting —
revert it.

## 0. One-time environment setup (lab machine)

```bash
cd code/business_entity_resolution
pip install -r requirements.txt          # >= constraints
pip freeze > requirements-lock.txt        # commit as a SEPARATE commit after a clean run
```

The reranker additionally needs `sentence-transformers` + `torch`. If they are
absent the reranker **silently no-ops** (returns original scores) — so a missing
install degrades gracefully instead of crashing, but you also get no benefit.
Verify the cross-encoder's license (MIT/Apache) and size (≤8B) on the box before
submitting; the default `BAAI/bge-reranker-v2-m3` is XLM-R ~568M.

## 1. First run — read the two gate numbers (decides the whole strategy)

`report_diagnostics` now logs automatically after Stage-A recall measurement.
Just run the baseline once and read the log:

```bash
python -m src.pipeline --train-dir dataset/train --test-dir dataset/test \
    --output-dir output --run-embedding
```

Look for two log lines:
- `DIAGNOSTIC — singleton ratio: …`
- `DIAGNOSTIC — blocking recall ceiling: …`  → `verdict: PRECISION-BOUND / RECALL-BOUND / MIXED`

If it prints **PRECISION-BOUND** (recall ceiling ≥0.97 and singletons ≥0.80),
stop chasing candidates and spend everything on levers 2–5 below. If
**RECALL-BOUND**, fix blocking first (Stage B / more keys) before thresholding.

## 2. Measure unseen-country transfer (LOCO) — highest-priority lever

```bash
python -m src.pipeline … --run-embedding --loco
```

`loco_evaluate` runs after labelling: for each training country with ≥200
entities it trains + calibrates + tunes the policy on the *other* countries, then
scores the held-out country (a stand-in for unseen France). Read the summary line:

```
LOCO summary: worst gap=…  recommended conservative margin=+X.XXX  mean held-out F0.5=…
```

The **recommended conservative margin** is the largest positive threshold shift a
held-out country needed. When you enable the policy path (lever 3) *without*
setting `--france-margin`, this LOCO value is adopted automatically and added to
the final decision threshold to make it France-robust.

## 3. Precision-first decision policy (calibration + fine/per-source + abstention)

Enabling **any** of these flags switches steps 9–10 from the baseline
`tune_threshold`/`predict` to the calibrated `ProbabilityCalibrator` +
`tune_policy` + `apply_policy` path:

```bash
python -m src.pipeline … --run-embedding --loco \
    --calibrate --calib-method isotonic \
    --fine-threshold \
    --per-source-threshold \
    --abstain
```

- `--calibrate [--calib-method isotonic|platt]` — corrects probabilities distorted
  by 3:1 negative downsampling (fitted on the val split's true prior).
- `--fine-threshold` — 0.005 F0.5 grid instead of the coarse 0.05 sweep (the race
  is in the third decimal).
- `--per-source-threshold` — separate S2 vs S3 thresholds by coordinate ascent on
  the JOINT per-entity F0.5.
- `--abstain` — sweeps an abstention margin {0,.02,.05,.1}; an entity commits only
  if its best candidate clears threshold by the margin ("stay empty when unsure").
- `--france-margin F` — override the LOCO auto-margin with a fixed value (0 = use
  LOCO's recommendation; the margin is added to global and every per-group cut and
  clamped ≤0.999).

Class-imbalance experiment (F0.5 is precision-weighted, so `class_weight="balanced"`
pushes the wrong way): compare on held-out-country F0.5:

```bash
python -m src.pipeline … --drop-class-weight        # or
python -m src.pipeline … --scale-pos-weight 0.5     # implies class_weight=None
```

## 4. Borderline-band cross-encoder reranker

```bash
python -m src.pipeline … --calibrate --fine-threshold \
    --rerank --rerank-band 0.10 --rerank-model BAAI/bge-reranker-v2-m3
```

Only pairs whose calibrated score lands in `[t−band, t+band]` are re-scored by the
cross-encoder (a few % of pairs → trivial compute); everything else keeps its
LightGBM decision. Applied identically to val and test, and the policy is
re-tuned on the reranked val scores so the score distribution stays consistent
end-to-end. Start with `--rerank-band 0.10`; widen only if the band is tiny.

## 5. Error-analysis loop (how you find the winning few hundred entities)

```bash
python -m src.pipeline … --error-analysis        # works with or without the policy path
```

`bucket_errors` dumps every val entity scored below 1.0 into buckets
(`singleton_false_merge`, `chain_false_merge`, `indic_false_merge`,
`blocking_miss`, `indic_miss`, `classifier_miss`, …), largest first. Fix the
biggest bucket, re-run, repeat. At a 0.0016 spread this beats any blind sweep.

## Recommended full send (all levers)

```bash
python -m src.pipeline \
    --train-dir dataset/train --test-dir dataset/test --output-dir output \
    --run-embedding \
    --loco --error-analysis \
    --calibrate --calib-method isotonic \
    --fine-threshold --per-source-threshold --abstain \
    --rerank --rerank-band 0.10
```

Then confirm `utils/validate_submission.py` prints PASS (pipeline runs it as
step 11). Report, for each change: what changed, held-out-country F0.5
before/after, and the singleton false-merge count.

## Suggested ablation order (isolate each lever's contribution)

1. Baseline (no flags) → record in-dist + note it's your reference.
2. `--loco` only → read the transfer gap + recommended margin (diagnostic; output unchanged).
3. `+ --calibrate` → does calibration alone move held-out F0.5?
4. `+ --fine-threshold` → third-decimal threshold resolution.
5. `+ --per-source-threshold` → S2/S3 split.
6. `+ --abstain` → abstention margin.
7. Toggle `--drop-class-weight` / `--scale-pos-weight 0.5` → keep whichever wins on held-out.
8. `+ --rerank` → the borderline band.
9. `--error-analysis` throughout → drive the next fix.

Keep a lever **only** if it improves the held-out-country F0.5.

## Constraints checklist (re-confirm before submitting)

- Cross-encoder + embedding models are MIT/Apache-2.0 and ≤8B params.
- No external data / APIs / geocoding — all signal from the provided files.
- `validate_submission.py` PASSes; every emitted match ⊆ candidate_pairs; every
  S1 entity appears exactly once; only real test S2/S3 IDs.
- Fixed seeds; commit `requirements-lock.txt` as a separate post-run commit.

