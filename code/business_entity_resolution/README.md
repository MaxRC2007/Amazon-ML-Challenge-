# Business Entity Resolution — Run Guide

## Overview

End-to-end pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.
Matches noisy business records from Source 2 & 3 against the deduplicated Source 1 reference.
Scored by macro-averaged F_0.5 (precision weighted 2×).

## Environment Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

> All libraries are MIT/Apache 2.0 licensed. The `sentence-transformers` model
> (`paraphrase-multilingual-MiniLM-L12-v2`) runs **entirely offline and locally**
> — it is a static pretrained artifact, not an external API or lookup service.

## End-to-End Run (classical pipeline only — recommended first run)

From the `student_resource/` root directory:

```bash
python -m code.business_entity_resolution.src.pipeline \
    --train-dir  dataset/train \
    --test-dir   dataset/test  \
    --output-dir output/
```

Output written to `output/`:
- `matching_results.tsv` — final entity matches (upload this to the leaderboard)
- `candidate_pairs.tsv`  — blocking candidate set (submitted in the final zip)

## With Embedding Stage B (after validating Stage A recall)

```bash
python -m code.business_entity_resolution.src.pipeline \
    --train-dir       dataset/train \
    --test-dir        dataset/test  \
    --output-dir      output/       \
    --run-embedding                 \
    --embed-threshold 2
```

`--embed-threshold N` means Stage B runs only on S1 entities where Stage A
returned ≤ N candidates. Set N=0 to restrict Stage B to entities with zero
classical candidates; set N=2 to cover entities with 0–2 candidates.

## Validate Before Submitting

```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv  \
    --test-dir  dataset/test
```

Prints `PASS` (exit 0) or numbered issues (exit 1). Run this after every pipeline run.

## Module Reference

| File | Purpose |
|---|---|
| `src/parser.py` | Robust TSV loading via `csv.reader` with quarantine logging |
| `src/normalization.py` | NFKC → unidecode → lowercase → suffix strip → token-sort |
| `src/blocking.py` | Stage A classical blocking + gated Stage B embedding ANN |
| `src/features.py` | 9 pairwise similarity features (Jaccard, Levenshtein, TF-IDF, etc.) |
| `src/model.py` | LightGBM training + tree-level checkpointing + F_0.5 threshold sweep |
| `src/checkpoint.py` | Atomic Parquet / JSON / Pickle serialization for fault tolerance |
| `src/pipeline.py` | Top-level orchestrator with step-level checkpointing & resume |

## Checkpointing & Resuming (Fault-Tolerant Training)

The pipeline incorporates end-to-end atomic checkpointing across all execution stages:

- **Enabled by Default**: Any run automatically saves checkpoints to `output/checkpoints/` (or `--checkpoint-dir`).
- **Resuming Interrupted Training**: If training or inference is stopped, simply re-run the exact same command. The pipeline detects existing checkpoints and resumes seamlessly:
  - Already normalized datasets load in ~2 seconds (via PyArrow Parquet).
  - Pre-computed inverted indexes, blocking candidate pairs, and TF-IDF vectors are restored.
  - LightGBM model training resumes from the last completed booster iteration (checkpointed every `--checkpoint-freq` trees).
  - Validation threshold and test inference states are saved to prevent duplicate work.
- **Flags**:
  - `--resume` (default `True`): Resume from checkpoints if available.
  - `--no-resume`: Ignore checkpoints and recompute the full pipeline from scratch.
  - `--checkpoint-freq N` (default `25`): Frequency (in trees) for saving LightGBM booster checkpoints.
  - `--checkpoint-dir DIR`: Custom directory for storing pipeline checkpoints.

## Self-tests

```bash
# Normalization self-test
python -m code.business_entity_resolution.src.normalization

# Parser self-test
python -m code.business_entity_resolution.src.parser dataset/train dataset/test
```

## Key Design Decisions

1. **No external APIs / geocoding** — fully offline. Complies with fair-play rules.
2. **Model ≤ 8B params** — LightGBM is a gradient-boosted tree (no parameters in the
   LLM sense); the optional embedding model is ~117M params (MIT license).
3. **F_0.5 threshold tuning** — we sweep thresholds 0.10–0.90 on a held-out
   validation split and select the value maximising F_0.5 directly, not the 0.5 default.
4. **Embedding model is offline** — `paraphrase-multilingual-MiniLM-L12-v2` is a
   static artifact loaded from the local sentence-transformers cache. It does not
   make network calls at inference time and was trained prior to this competition.
   This is standard practice (equivalent to any pretrained embedding) and does not
   constitute an "external lookup."
5. **Singleton handling** — the output contains a row for every S1 entity, with an
   empty `matched_entity_ids` for singletons (correctly scores 1.0 on F_0.5).
