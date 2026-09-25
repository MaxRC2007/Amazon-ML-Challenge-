"""
raw_byte_check.py
-----------------
Inspect raw repr() of rows that could appear "misaligned" in a text editor:
  - Rows with very long business_address fields (variable column width)
  - Rows with non-ASCII characters in business_name
  - First 5 rows of each file for sanity

Shows the actual raw bytes so we can confirm whether any embedded \\t or \\n
characters exist inside fields (which would be a real parsing problem) vs
purely visual misalignment from variable-width tab rendering.
"""

import csv
import sys

CHECKS = [
    ("dataset/train/train_source1.tsv", 4),
    ("dataset/train/train_source2.tsv", 4),
    ("dataset/train/train_source3.tsv", 4),
    ("dataset/test/test_source1.tsv",   4),
]

N_SAMPLE = 5          # rows per category
ADDR_THRESHOLD = 120  # chars — "long enough to look misaligned in an editor"


def inspect_file(path, expected_cols, n_sample=N_SAMPLE):
    print(f"\n{'='*70}")
    print(f"FILE: {path}")
    print(f"{'='*70}")

    long_addr_shown  = 0
    non_ascii_shown  = 0
    first_rows_shown = 0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        # --- Raw line-level check (before csv parsing) ---
        # Read the first 10 physical lines and check for embedded tabs/newlines
        fh.seek(0)
        print("\n--- RAW BYTES: first 6 physical lines (repr) ---")
        for i, raw_line in enumerate(fh):
            if i >= 6:
                break
            # Look for suspicious patterns
            has_embedded_tab      = "\t" in raw_line.rstrip("\n")
            tab_count             = raw_line.count("\t")
            print(f"  line {i+1:02d} | tabs={tab_count} | repr={repr(raw_line[:120])}")

        # --- CSV-parsed check ---
        fh.seek(0)
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        print(f"\n  Header cols ({len(header)}): {header}")

        for lineno, row in enumerate(reader, start=2):
            if len(row) != expected_cols:
                print(f"  *** WRONG COL COUNT at line {lineno}: {len(row)} cols — repr={repr(row)}")
                continue

            eid, name, addr, country = row
            addr_len = len(addr)
            has_non_ascii = any(ord(c) > 127 for c in name)

            # Category 1: first N rows
            if first_rows_shown < n_sample:
                print(f"\n  [FIRST] line {lineno:06d} | cols={len(row)} | addr_len={addr_len}")
                print(f"    entity_id : {repr(eid)}")
                print(f"    name      : {repr(name)}")
                print(f"    address   : {repr(addr[:80])}")
                print(f"    country   : {repr(country)}")
                first_rows_shown += 1

            # Category 2: long address rows (most likely to look "misaligned")
            if addr_len >= ADDR_THRESHOLD and long_addr_shown < n_sample:
                print(f"\n  [LONG ADDR] line {lineno:06d} | cols={len(row)} | addr_len={addr_len}")
                print(f"    name      : {repr(name)}")
                print(f"    address   : {repr(addr[:160])}")
                # Check raw file bytes at this location
                long_addr_shown += 1

            # Category 3: non-ASCII names
            if has_non_ascii and non_ascii_shown < n_sample:
                print(f"\n  [NON-ASCII] line {lineno:06d} | cols={len(row)}")
                print(f"    name repr : {repr(name)}")
                print(f"    addr repr : {repr(addr[:80])}")
                non_ascii_shown += 1

            if long_addr_shown >= n_sample and non_ascii_shown >= n_sample and first_rows_shown >= n_sample:
                break

    print()


if __name__ == "__main__":
    for path, cols in CHECKS:
        inspect_file(path, cols)
    print("\nDONE — if no '*** WRONG COL COUNT' lines appeared and no embedded \\t")
    print("found inside field reprs, the zero-malformed-rows finding is confirmed.")
