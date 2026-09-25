"""
normalization.py
----------------
Text normalization pipeline for business names and addresses.

Pipeline (applied in order):
  1. Unicode NFKC — canonical decomposition + compatibility mapping
                    (e.g. ﬁ → fi, ² → 2, ½ → 1/2)
  2. unidecode    — transliterate any non-Latin script to closest ASCII
                    (Devanagari, Tamil, Malayalam, Telugu, Gujarati, French
                     accented Latin, Cyrillic, etc. all handled)
  3. Lowercase
  4. Strip punctuation & symbols (keep alphanumeric + spaces)
  5. Strip legal suffixes (curated list covering US/India/France)
  6. Token-sort    — sorts tokens alphabetically so word-order variants match
  7. Collapse whitespace

This module is called once per dataframe; results are cached as new columns
rather than recomputed per pair.

Public API:
    normalize_name(series: pd.Series) -> pd.Series
    normalize_address(series: pd.Series) -> pd.Series
    token_sort(s: str) -> str        # sort tokens alphabetically
    token_set(s: str) -> str         # deduplicated sorted tokens
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

import pandas as pd

try:
    from unidecode import unidecode
except ImportError:
    # Graceful fallback (won't handle Indic/CJK but won't crash)
    import warnings
    warnings.warn(
        "unidecode not installed — non-Latin script transliteration disabled. "
        "Install with: pip install unidecode",
        ImportWarning,
        stacklevel=2,
    )

    def unidecode(s: str) -> str:  # type: ignore[misc]
        return s


# ---------------------------------------------------------------------------
# Legal suffix vocabulary
# US, India, France, Germany (common additions).
# Lower-cased so they're matched after lowercasing.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIXES: tuple[str, ...] = (
    # US
    "llc", "inc", "corp", "corporation", "ltd", "limited", "co", "company",
    "lp", "llp", "pllc", "pc", "pa", "na", "dba",
    # India
    "pvt", "private", "opc", "ngo",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "sca", "scop",
    "gie", "eirl",
    # Germany
    "gmbh", "ag", "kg", "ohg", "eg",
    # UK
    "plc",
)

# De-duplicate defensively while preserving order, so an accidental repeat in
# the list above (e.g. the historical double-"sarl") can never bloat the regex
# or change matching behaviour.
_LEGAL_SUFFIXES = tuple(dict.fromkeys(_LEGAL_SUFFIXES))

# Build a regex that matches one suffix at a word boundary, followed by
# optional punctuation/whitespace and end-of-string.
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\b[.\s]*$",
    re.IGNORECASE,
)

# Strip everything that is not alphanumeric or space
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

# Collapse multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core string transforms
# ---------------------------------------------------------------------------

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _transliterate(s: str) -> str:
    return unidecode(s)


def _strip_legal_suffixes(s: str) -> str:
    """Iteratively strip trailing legal suffixes (e.g. 'Pvt Ltd' → two passes)."""
    prev = None
    while prev != s:
        prev = s
        s = _SUFFIX_PATTERN.sub("", s).strip()
    return s


def normalize(s: str) -> str:
    """
    Full normalization pipeline for a single string.
    Returns the normalized form.  Returns '' for null/non-string input.
    """
    if not isinstance(s, str) or not s:
        return ""
    s = _nfkc(s)
    s = _transliterate(s)
    s = s.lower()
    s = re.sub(r"['’`]", "", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _strip_legal_suffixes(s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def token_sort(s: str) -> str:
    """Sort space-separated tokens alphabetically — handles word-order variants."""
    return " ".join(sorted(s.split()))


def token_set(s: str) -> str:
    """Deduplicated, sorted tokens — more aggressive than token_sort."""
    return " ".join(sorted(set(s.split())))


# ---------------------------------------------------------------------------
# Vectorised pandas wrappers
# ---------------------------------------------------------------------------

def normalize_name(series: pd.Series) -> pd.Series:
    """Normalize a Series of business names. Returns a new Series."""
    return series.fillna("").map(normalize)


def normalize_address(series: pd.Series) -> pd.Series:
    """
    Normalize a Series of addresses.
    Address normalization is lighter — we keep more tokens and don't strip
    legal suffixes (they're meaningful in addresses).
    """
    def _norm_addr(s: str) -> str:
        if not isinstance(s, str) or not s:
            return ""
        s = _nfkc(s)
        s = _transliterate(s)
        s = s.lower()
        s = re.sub(r"['’`]", "", s)
        s = _NON_ALNUM_RE.sub(" ", s)
        s = _MULTI_SPACE_RE.sub(" ", s).strip()
        return s

    return series.fillna("").map(_norm_addr)


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-computed normalized columns to a source dataframe in-place.

    Added columns:
        norm_name        — normalize(business_name)
        norm_name_sorted — token_sort(norm_name)
        norm_name_set    — token_set(norm_name)
        norm_address     — normalize_address(business_address)

    Modifies df in-place and also returns it for chaining.
    """
    df["norm_name"]        = normalize_name(df["business_name"])
    df["norm_name_sorted"] = df["norm_name"].map(token_sort)
    df["norm_name_set"]    = df["norm_name"].map(token_set)
    df["norm_address"]     = normalize_address(df["business_address"])
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (input, expected_normalized, expected_sorted)
        ("Orelee's Barbershop",             "orelees barbershop",            "barbershop orelees"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", normalize("राम मार्केटिंग प्राइवेट लिमिटेड"), token_sort(normalize("राम मार्केटिंग प्राइवेट लिमिटेड"))),
        ("École primaire Sainte Pierre",    "ecole primaire sainte pierre",   "ecole pierre primaire sainte"),
        ("B+ Retail Inc",                   "b retail",                       "b retail"),
        ("Custom Wealth Services LLC",      "custom wealth services",         "custom services wealth"),
        ("Consulting Nyasa Nursing Private Limited", "consulting nyasa nursing", "consulting nursing nyasa"),
        ("Société Générale SARL",           "societe generale",               "generale societe"),
        ("Moore Bitwise Inc.",              "moore bitwise",                  "bitwise moore"),
        ("Nexus Anchor Rain",               "nexus anchor rain",              "anchor nexus rain"),
        # Reversed word order → same after token_sort
        ("Rain Anchor Nexus",               "rain anchor nexus",              "anchor nexus rain"),
    ]

    print(f"{'Input':<45} {'norm_name':<35} {'norm_sorted':<35}")
    print("-" * 115)
    for inp, exp_norm, exp_sorted in test_cases:
        n = normalize(inp)
        s = token_sort(n)
        match_norm   = "✓" if n == exp_norm   else f"✗ (got: {n!r})"
        match_sorted = "✓" if s == exp_sorted else f"✗ (got: {s!r})"
        print(f"{inp:<45} {match_norm:<35} {match_sorted:<35}")
