That worked — everything's up. `main` is now `686e7c7`, and your verified `a6110e2` sits right below it in history, exactly as intended. The repo has your full code + the Apache-2.0 LICENSE, and it's private.

Let me record the pushed state so it's anchored across sessions.That worked — everything's up. `main` is now `686e7c7`, with GitHub's Apache-2.0 LICENSE and your reviewed source `a6110e2` both in history. The 1.6 MB push is all source, no data, exactly as intended.

Now on the **lab machine**:

```bash
git clone https://github.com/MaxRC2007/Amazon-ML-Challenge-.git
cd Amazon-ML-Challenge-
git log --oneline -3        # confirm a6110e2 is in the history
```
(Same PAT/`gh` auth as before, since it's private.)

Then the run sequence:

```bash
# 1. put the dataset in place (NOT in the repo — copy it in)
#    dataset/train/{train_source1,2,3, train_ground_truth}.tsv
#    dataset/test/{test_source1,2,3}.tsv

# 2. environment
python -m venv .venv && source .venv/bin/activate    # Win: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
python -c "import torch; print('gpu:', torch.cuda.is_available())"   # must be True

# 3. run (from inside the package dir so -m src.pipeline resolves)
cd code/business_entity_resolution
python -m src.pipeline \
  --train-dir ../../dataset/train \
  --test-dir  ../../dataset/test \
  --output-dir ../../output \
  --run-embedding
```

Watch the log for the two gate numbers — `Recall ceiling: 0.XXXX` and `... N with ZERO candidates (X%)` — then let it finish through `Validator: PASS`. Finally lock the environment as the second commit:

```bash
cd ../..            # back to repo root
pip freeze > requirements-lock.txt
git add requirements-lock.txt
git commit -m "Locked environment: exact versions that ran on lab hardware"
git push
```

Two reminders that already bit us or could: run the pipeline from `code/business_entity_resolution/` (not the repo root), and if `gpu:` prints `False`, fix CUDA/torch before running or the embedding pass crawls on the full test set. You're set — the repo is the clean, frozen, reproducible package.
