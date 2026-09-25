"""
parser.py
---------
Robust TSV loader for the Amazon ML Challenge 2026 dataset.

Uses Python's csv.reader with an explicit tab delimiter (never str.split('\t'))
so that any row with an unexpected field count is caught rather than silently
misaligned.  Zero malformed rows were found in all 7 train/test files during
pre-build verification, but the quarantine path stays as a safety net.

Public API:
    load_source(path)        -> pd.DataFrame with columns [entity_id, business_name,
                                                            business_address, country]
    load_ground_truth(path)  -> pd.DataFrame with columns [source1_entity_id,
                                                            matched_entity_ids]
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected column counts
_SOURCE_COLS = 4
_GT_COLS = 2

_SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
_GT_HEADER = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(
    path: str | Path,
    expected_cols: int,
    quarantine_path: Optional[str | Path] = None,
) -> list[list[str]]:
    """
    Read a TSV file row-by-row.  Rows with a wrong column count are logged
    and optionally written to a quarantine file; they are NOT silently
    included in the returned data.

    Returns a list of rows (each a list of strings), excluding the header.
    """
    path = Path(path)
    quarantine_rows: list[list[str]] = []
    good_rows: list[list[str]] = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)  # consume header
        if header is None:
            logger.warning("Empty file: %s", path)
            return []

        for lineno, row in enumerate(reader, start=2):  # 2 because header is line 1
            if len(row) != expected_cols:
                logger.warning(
                    "Malformed row at line %d in %s: expected %d cols, got %d — %s",
                    lineno,
                    path.name,
                    expected_cols,
                    len(row),
                    row[:4],
                )
                quarantine_rows.append(row)
            else:
                good_rows.append(row)

    if quarantine_rows:
        logger.warning(
            "%d malformed rows quarantined from %s", len(quarantine_rows), path.name
        )
        if quarantine_path:
            _write_quarantine(quarantine_path, quarantine_rows)

    return good_rows


def _write_quarantine(path: str | Path, rows: list[list[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for row in rows:
            writer.writerow(row)
    logger.info("Quarantine written to %s (%d rows)", path, len(rows))


def load_source(
    path: str | Path,
    quarantine_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load a source TSV (source1 / source2 / source3).

    Returns a DataFrame with columns:
        entity_id, business_name, business_address, country
    """
    path = Path(path)
    quarantine_path = (
        Path(quarantine_dir) / f"{path.stem}_quarantine.tsv"
        if quarantine_dir
        else None
    )

    rows = _read_tsv(path, _SOURCE_COLS, quarantine_path=quarantine_path)

    df = pd.DataFrame(rows, columns=_SOURCE_HEADER)

    # Ensure string types and strip leading/trailing whitespace
    for col in _SOURCE_HEADER:
        df[col] = df[col].str.strip()

    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def load_ground_truth(
    path: str | Path,
    quarantine_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.

    Returns a DataFrame with columns:
        source1_entity_id, matched_entity_ids

    matched_entity_ids is kept as a raw comma-separated string.
    Use .str.split(',') downstream when you need a list.
    """
    path = Path(path)
    quarantine_path = (
        Path(quarantine_dir) / f"{path.stem}_quarantine.tsv"
        if quarantine_dir
        else None
    )

    rows = _read_tsv(path, _GT_COLS, quarantine_path=quarantine_path)

    df = pd.DataFrame(rows, columns=_GT_HEADER)
    df["source1_entity_id"] = df["source1_entity_id"].str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].str.strip()

    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/train")
    test_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset/test")

    for p in [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
    ]:
        df = load_source(p)
        print(f"{p.name}: {len(df):,} rows  |  dtypes: {df.dtypes.to_dict()}")

    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"Ground truth: {len(gt):,} rows")
