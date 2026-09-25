"""
verify_test_files.py
--------------------
Independently verifies the data analysis claims from the initial investigation:
  1. Malformed-row count in test files (wrong column count)
  2. Non-ASCII business name prevalence in test files
  3. Re-verify the same for train files to confirm earlier counts

Runs using Python stdlib + built-in csv only (no external deps).
"""

import csv
import os
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(WORKSPACE, "dataset", "train")
TEST_DIR  = os.path.join(WORKSPACE, "dataset", "test")

FILES = [
    # (label, path, expected_cols)
    ("train_source1", os.path.join(TRAIN_DIR, "train_source1.tsv"), 4),
    ("train_source2", os.path.join(TRAIN_DIR, "train_source2.tsv"), 4),
    ("train_source3", os.path.join(TRAIN_DIR, "train_source3.tsv"), 4),
    ("train_ground_truth", os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), 2),
    ("test_source1",  os.path.join(TEST_DIR,  "test_source1.tsv"),  4),
    ("test_source2",  os.path.join(TEST_DIR,  "test_source2.tsv"),  4),
    ("test_source3",  os.path.join(TEST_DIR,  "test_source3.tsv"),  4),
]

SAMPLE_LIMIT = 10


def analyze(label, filepath, expected_cols):
    if not os.path.exists(filepath):
        print(f"[MISSING] {label}: {filepath}")
        return

    total_rows       = 0
    wrong_col_rows   = 0
    non_ascii_names  = 0
    sample_wrong     = []
    sample_non_ascii = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[EMPTY] {label}")
            return

        for row in reader:
            total_rows += 1
            if len(row) != expected_cols:
                wrong_col_rows += 1
                if len(sample_wrong) < SAMPLE_LIMIT:
                    sample_wrong.append({
                        "n_cols": len(row),
                        "raw": row[:6],   # first 6 fields to avoid huge dumps
                    })

            # business_name is column index 1 for source files (not ground-truth)
            if expected_cols == 4 and len(row) >= 2:
                name = row[1]
                if any(ord(c) > 127 for c in name):
                    non_ascii_names += 1
                    if len(sample_non_ascii) < SAMPLE_LIMIT:
                        sample_non_ascii.append(name)

    pct_wrong     = wrong_col_rows  / max(1, total_rows) * 100
    pct_non_ascii = non_ascii_names / max(1, total_rows) * 100

    print(f"\n=== {label} ===")
    print(f"  Total data rows (excl. header) : {total_rows:,}")
    print(f"  Expected columns               : {expected_cols}")
    print(f"  Wrong column count             : {wrong_col_rows:,}  ({pct_wrong:.4f}%)")
    if expected_cols == 4:
        print(f"  Non-ASCII business names       : {non_ascii_names:,}  ({pct_non_ascii:.4f}%)")

    if sample_wrong:
        print(f"  --- Sample wrong-column rows (up to {SAMPLE_LIMIT}) ---")
        for s in sample_wrong:
            print(f"    n_cols={s['n_cols']}  first fields: {s['raw']}")

    if sample_non_ascii and expected_cols == 4:
        print(f"  --- Sample non-ASCII names (up to {SAMPLE_LIMIT}) ---")
        for n in sample_non_ascii:
            print(f"    {n}")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA VERIFICATION REPORT")
    print("Checking train & test files for malformed rows and")
    print("non-ASCII business names independently.")
    print("=" * 60)

    for label, path, expected_cols in FILES:
        analyze(label, path, expected_cols)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
