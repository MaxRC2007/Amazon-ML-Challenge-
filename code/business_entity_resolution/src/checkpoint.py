"""
checkpoint.py
-------------
Checkpointing and state recovery utilities for the Entity Resolution pipeline.

Provides fast, fault-tolerant persistence for:
  - Pandas DataFrames (via PyArrow Parquet with fallback to pickle)
  - NumPy arrays (via np.save / np.load)
  - Python dictionaries / metadata (via JSON)
  - Arbitrary objects like vectorizers and models (via Pickle)

Enables instantaneous resume if pipeline or model training is paused or interrupted.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to disk using Parquet with atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temp_path, index=False, engine="pyarrow")
        if path.exists():
            path.unlink()
        temp_path.rename(path)
        logger.debug("Saved DataFrame (%d rows) to %s", len(df), path)
    except Exception as e:
        logger.warning("Parquet save failed for %s (%s). Falling back to pickle.", path, e)
        with open(temp_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        if path.exists():
            path.unlink()
        temp_path.rename(path)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from disk."""
    path = Path(path)
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_json(data: dict | list, path: str | Path) -> None:
    """Save metadata/dict to a JSON file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if path.exists():
        path.unlink()
    temp_path.rename(path)


def load_json(path: str | Path) -> Any:
    """Load metadata/dict from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to disk atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temp_path, arr)
    if path.exists():
        path.unlink()
    temp_path.rename(path)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk."""
    path = Path(path)
    return np.load(path)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an arbitrary object to disk via pickle atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists():
        path.unlink()
    temp_path.rename(path)


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_exists(path: str | Path) -> bool:
    """Check if a checkpoint file exists and is non-empty."""
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0
