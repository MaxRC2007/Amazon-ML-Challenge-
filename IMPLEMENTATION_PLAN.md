# Business Entity Resolution — Implementation Plan & Current State

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution
**Repo:** `github.com/MaxRC2007/Amazon-ML-Challenge-` (private), branch `main`
**Verified source snapshot:** commit `a6110e2` (in history under merge tip)
**Status:** Reviewed, fixed, verified on sampled data. Ready for full lab-machine run.

---

## 1. Objective & Scoring

Match noisy business records: for every entity in **Source 1**, find all records in **Source 2** and **Source 3** that refer to the same real-world business. Output is one row per S1 entity listing its matched S2-/S3- IDs (empty for singletons).

**Metric — precision-weighted macro F₀.₅**, computed *per S1 entity* then averaged:

```
F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
```

- Precision is weighted 4× more heavily than recall → **false matches hurt far more than misses**.
- A true singleton (no matches) scores **1.0 if predicted empty**, **0.0 if any false match** is emitted.
- This shapes every design choice toward **high precision**: over-generate candidates in blocking (recall ceiling), then let a precision-tuned classifier + F₀.₅-tuned threshold cut aggressively.

---

## 2. Hard Constraints

- Test set includes **France** (absent from training) → must **not** hard-code country logic to {US, India}. Country is used only as a *feature*, never a filter.
- Model must be **MIT/Apache licensed, ≤8B params** → LightGBM (classifier) + `LaBSE` / multilingual MiniLM (embeddings).
- Fully **reproducible** submission package required.
- No external data lookups; matching must be derived from genuine features (audit for fair play).

<!-- APPEND-MARKER -->

---

## 3. End-to-End Pipeline (`pipeline.py` orchestration)

The pipeline runs 11 sequenced steps:

1. **Parse** all 7 files (3 train sources + ground truth + 3 test sources) with a robust TSV reader.
2. **Normalize** every source (name + address normalization columns precomputed once).
3. **Train/val split** — hold out 20% of S1 ground-truth entities (`random_state=42`) for threshold tuning and recall measurement.
4. **Stage A classical blocking** on the train split → candidate pairs.
5. **Recall-ceiling measurement** on the val split (the upper bound on achievable recall given blocking).
6. **[Optional] Stage B embedding ANN** — gated behind `--run-embedding`.
7. **Feature matrix** built for train/val/test on a single shared TF-IDF vocabulary.
8. **Label + train LightGBM** with negative downsampling.
9. **Threshold tuning** on val split — sweep cutoffs, maximize F₀.₅ directly.
10. **Test inference** → `matching_results.tsv` + `candidate_pairs.tsv`.
11. **Validate submission** via `utils/validate_submission.py` (PASS/FAIL gate).

Data flow: `parse → normalize → block → (embed) → featurize → classify → threshold → emit → validate`.

---

## 4. Module-by-Module Detail

### 4.1 `parser.py` — robust ingestion
- Uses `csv.reader(delimiter="\t", quoting=csv.QUOTE_NONE)` — **never** `str.split('\t')` — so any row with the wrong field count is *caught*, not silently misaligned.
- Rows with != expected column count are logged and quarantined (not included), never silently repaired.
- `encoding="utf-8", errors="replace"` for resilience against stray bytes.
- Columns: sources → `[entity_id, business_name, business_address, country]`; GT → `[source1_entity_id, matched_entity_ids]` (kept as raw comma-string).
- **Verified finding:** ZERO malformed rows across all 7 real files. Apparent viewer misalignment is from commas embedded in address fields — tab-parse is clean. Quarantine branch stays as pure defensive logging.

### 4.2 `normalization.py` — text canonicalization
Pipeline applied in order to each name:
1. **Unicode NFKC** (canonical/compatibility folding: ﬁ→fi, ²→2).
2. **`unidecode`** transliteration — non-Latin scripts (Devanagari, Tamil, Telugu, Malayalam, Gujarati, Bengali, Kannada, Cyrillic, French accents) → closest ASCII. **Load-bearing** for cross-script matching.
3. **Lowercase**.
4. **Strip** non-alphanumeric (keep alnum + space).
5. **Strip legal suffixes** (iterative — handles "Pvt Ltd" in two passes).
6. **Token-sort** — sorts tokens alphabetically so word-order variants collide.
7. **Collapse whitespace**.

- **Legal-suffix vocabulary** covers US (`llc, inc, corp, ltd, co, lp, llp, pllc, pc, pa, na, dba`), India (`pvt, private, opc, ngo`), France (`sarl, sas, sasu, eurl, sa, sci, snc, sca, scop, gie, eirl`), Germany (`gmbh, ag, kg, ohg, eg`), UK (`plc`).
- **Precomputed columns** (added once per DataFrame, reused for all pairs): `norm_name`, `norm_name_sorted` (token-sorted), `norm_name_set` (deduped sorted tokens), `norm_address` (lighter normalization — no suffix strip, since suffixes are meaningful in addresses).

### 4.3 `blocking.py` — candidate generation (the recall engine)

**Stage A — classical blocking.** Builds an inverted index `key → [entity_id]` over S2 and S3, then streams S1 rows one at a time (memory-efficient — never holds the full S1→candidates map in RAM). Four key types per entity:

| Key | Definition | Purpose |
|-----|-----------|---------|
| `FN:` | Exact full token-sorted normalized name | High-precision exact-name collision |
| `NP:` | First 6 chars of sorted normalized name | Prefix/near-name grouping |
| `PH:` | Metaphone phonetic code per token (jellyfish) | Spelling/transcription variants |
| `PC:` | 5–6 digit postal/PIN fragment from address | Geographic co-location |

- **Stop-key pruning:** `NP/PH/PC` keys mapping to > `max_key_freq=1000` entities are dropped (avoids combinatorial explosion from generic keys).
- **`FN:` gets a higher `fn_max_freq=5000` ceiling** rather than full exemption — a defensive backstop so a pathologically common exact name can't create an unbounded block, while normal exact-name collisions (high precision) are preserved.
- **Zero-candidate logging:** reports the % of S1 entities with no candidates (the recall-lost tail, and Stage B's target).
- `run_stage_a` output: `[source1_entity_id, candidate_entity_ids, n_candidates]`, all S1 entities present (empty string if none).

**`measure_recall`** — computes the recall ceiling: fraction of true GT matches present in the candidate set. Singletons excluded from the denominator. This is the go/no-go number for Stage B.

**Stage B — embedding ANN (gated, optional).**
- Model: `sentence-transformers/LaBSE` (multilingual, Apache-licensed).
- Embeds S2+S3 names → memory-mapped `.npy` on disk (never fully in RAM) → **FAISS `IndexFlatIP`** (inner product on unit vectors = cosine).
- Queries S1 entities for **top-k=10** nearest neighbors.
- **Two gating modes** (`embed_threshold`): `-1` = embed *all* S1 (GPU full-corpus mode); `≥0` = embed only S1 with ≤ N classical candidates (CPU-friendly, targets the weak-blocking tail).
- Final candidates = **union of Stage A ∪ Stage B**.

### 4.4 `features.py` — pairwise similarity features

For each `(S1, candidate)` pair, computes **9 features** (vectorized via pandas merge + list comprehensions + rapidfuzz C-extensions — no per-row `.apply`):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `token_set_jaccard` | Jaccard on deduped name token sets |
| 2 | `levenshtein_ratio` | Edit-distance ratio on normalized names |
| 3 | `token_sort_ratio` | Levenshtein after token-sort (order-invariant) |
| 4 | `lcs_ratio` | Longest common subsequence / max length |
| 5 | `tfidf_cosine` | Cosine over char 3-grams (symbol-noise robust) |
| 6 | `address_token_jaccard` | Jaccard on address token sets |
| 7 | `address_levenshtein` | Edit-distance ratio on addresses |
| 8 | `country_match` | 1.0 if country labels identical else 0.0 |
| 9 | `embedding_cosine` | Multilingual embedding cosine (0.0 if Stage B off) |

- **`fit_global_tfidf`** fits **one** `TfidfVectorizer` (char_wb 3-grams) on the union of all train name corpora, reused across train/val/test → `tfidf_cosine` shares a stable vocabulary (no train/test distribution skew).
- **`compute_embedding_cosine_map`** scores `embedding_cosine` for **every** candidate pair (embedding only unique names), so the feature is a genuine similarity signal — not a de-facto "came from Stage B" source indicator that would leak.
- rapidfuzz preferred; graceful `difflib` fallback if unavailable.

### 4.5 `model.py` — classifier, metric, thresholding
- **LightGBM** `LGBMClassifier`: `n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=20, class_weight="balanced", random_state=42`.
- **`f05_score`** — exact challenge metric: per-S1 macro F₀.₅ with the singleton rule (empty GT → 1.0 if predicted empty, else 0.0).
- **`tune_threshold`** — sweeps cutoffs `0.10…0.90` step 0.05, picks the one maximizing F₀.₅ on the val split (default 0.5 explicitly *not* used). Also logs a **singleton false-merge diagnostic** (how many true singletons got wrongly merged at the chosen threshold — directly the precision risk).
- `predict` applies the tuned threshold; model persisted via pickle.

### 4.6 `pipeline.py` — orchestration specifics
- **Vectorized labeling** (`_label_pairs`) via `np.fromiter` over a GT pair-set — replaces a slow per-row `.apply(axis=1)` that was the bottleneck on multi-million-pair sets.
- **Negative downsampling**: keep all positives + `neg_ratio × n_pos` negatives (default 3:1), `rng(42)`.
- **Memory management**: explicit `del` of all training-side frames + `gc.collect()` before loading the large test candidate set into the feature builder — keeps peak RAM down at full scale.
- Shared `tfidf_vec` threaded through train/val/test feature builds.

---

## 5. Fixes & Improvements Applied During Review

These are the changes made on top of the original generated pipeline:

**Correctness**
- **France `sarl` suffix** was *missing* entirely from the legal-suffix list → added, plus a `dict.fromkeys` dedup guard so any accidental repeat can't bloat the regex.
- **`embedding_cosine` computed for all pairs** (not just Stage-B-surfaced ones) → prevents the feature from leaking blocking-source membership to the classifier.
- **Global TF-IDF vocabulary** fit once and reused → eliminates train/test feature skew.

**Scale / performance**
- **`jellyfish` import hoisted** to module top (was inside the phonetic hot loop called millions of times); `_POSTAL_RE` precompiled once.
- **Vectorized feature computation** (merge + list comprehensions) replacing per-row rapidfuzz loop.
- **Vectorized labeling** (`np.fromiter`) replacing `.apply(axis=1)`.
- **Explicit memory cleanup** (`del` + `gc.collect()`) before test inference.

**Robustness**
- **`FN:` prune backstop** (`fn_max_freq=5000`) — empirically FN keys are tiny (max 7 on sample), but this caps the pathological common-exact-name case at full scale.
- **Zero-candidate + FN-cap logging** for observability.
- **tqdm graceful fallback** when not installed.

---

## 6. Test-Data Structural Findings (inputs only, no labels)

Sampled 250k rows/source. This materially informed the embedding decision:

- **Countries:** India ~47%, US ~38%, **France ~15%** (confirmed present at volume).
- **Name scripts:** S1 names are entirely Latin; **S2/S3 have ~10–17% native-script names** (Devanagari ~6% + Telugu/Kannada/Tamil/Bengali/Gujarati/Malayalam/Gurmukhi). Many true matches are **Latin-S1 ↔ Indic-S2/S3**.
- **Key risk:** `unidecode` transliterates Indic script *phonetically* ("मॉडर्न फाइनेंस" → "modarn phainens") which does **not** align with the English form ("Modern Finance") under edit distance → classical string features are structurally blind to this tail.
- **Legal suffixes:** every frequent test suffix already in the vocabulary → no change needed.
- **Postal codes:** only ~4–5% of addresses carry a 5–6 digit fragment → `PC:` is a minor assist; name keys carry the load.

---

## 7. Embedding Decision

**Enable `--run-embedding` for the lab run** (treated as core, not optional polish), gated only by the real Stage-A recall ceiling. Rationale: the Latin↔Indic cross-script tail (~10–17% of S2/S3) is exactly the failure mode multilingual semantic embeddings fix and edit-distance can't. GPU confirmed available. If the classical recall ceiling is already very high with a tiny zero-candidate tail, Stage B is insurance; otherwise it's earning its place.

---

## 8. Verification Performed

- All six modules `py_compile` clean.
- **Classical Stage-A smoke test** on an 800-S1 aligned sample: pipeline runs end-to-end; vectorized labeling count **exactly matches** `measure_recall` n_recovered (1733).
- FN-key concern empirically checked: large candidate blocks (max 253) come from generic `NP:` prefixes, not `FN:` (max 7) — the exemption is safe; backstop added anyway.
- Sandbox recall (0.64) is a **floor only** — `unidecode`/`jellyfish`/`rapidfuzz` are absent in the sandbox, so transliteration + phonetic keys are disabled there. Real recall on the lab machine (full deps) will be higher. **Do not anchor on 0.64.**

---

## 9. Reproducibility & Freeze Workflow

- Source frozen at commit **`a6110e2`** ("verified pre-lab-run state"); pushed to private GitHub repo (`dataset/`, `output/`, `*.tsv`, model artifacts all `.gitignore`d — source only, ~1.6 MB).
- **Requirements intentionally left unpinned** (`>=`) until lab install: install once on lab hardware → `pip freeze > requirements-lock.txt` → commit as a *second, distinct* commit. Avoids guessing incompatible faiss/torch/sentence-transformers versions before knowing the lab's Python/CUDA.
- Lab run: clone repo → verify hash `a6110e2` in history → copy dataset in separately → install → run from `code/business_entity_resolution/` with `python -m src.pipeline ... --run-embedding`.

---

## 10. Key Defaults Reference

| Parameter | Default | Location |
|-----------|---------|----------|
| `val_fraction` | 0.20 | pipeline |
| `neg_ratio` | 3 | pipeline |
| `max_key_freq` (stop-key prune) | 1000 | blocking |
| `fn_max_freq` (FN backstop) | 5000 | blocking |
| `NP:` prefix length | 6 chars | blocking |
| Stage B `top_k` | 10 | blocking |
| Embedding model | LaBSE (feature: MiniLM) | blocking/features |
| `embed_threshold` | 0 (`-1`=all/GPU) | pipeline |
| LightGBM `n_estimators` | 500 | model |
| Threshold sweep | 0.10–0.90 step 0.05 | model |
| `random_state` | 42 | throughout |

---

## 11. What to Watch on the First Real Lab Run

The two gate numbers that drive every downstream decision:
1. **`Recall ceiling: 0.XXXX`** — the real achievable recall (deps now active).
2. **`... N with ZERO candidates (X%)`** — the recall-lost tail.

Then: confirm `torch.cuda.is_available()` is True before Stage B; let it run through `Validator: PASS`; lock requirements as the second commit.


