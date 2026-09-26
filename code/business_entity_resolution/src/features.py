"""
features.py
-----------
Pairwise similarity feature computation for candidate pairs.

For each (S1 entity, candidate S2/S3 entity) pair, computes:

Name features:
  1. token_set_jaccard      — Jaccard on token sets of normalized names
  2. levenshtein_ratio      — edit distance ratio (0–1)
  3. token_sort_ratio       — Levenshtein ratio after token-sorting (handles reordering)
  4. lcs_ratio              — longest common subsequence length / max(len)
  5. tfidf_cosine           — TF-IDF cosine over char 3-grams (robust to symbol noise)

Address features:
  6. address_token_jaccard  — Jaccard on normalized address token sets
  7. address_levenshtein    — edit distance ratio on normalized addresses

Metadata features:
  8. country_match          — 1.0 if country labels are identical, else 0.0

Hard-bucket features (model the top-of-leaderboard failure modes):
  9.  name_containment      — |A∩B| / min(|A|,|B|) on name token sets (sub-brands)
  10. acronym_match         — 1.0 if one name is the initialism of the other
  11. house_number_match    — numeric street-number agreement (1/0/-1 sentinel)
  12. name_len_ratio        — min/max character length ratio of the two names
  13. name_addr_mismatch    — chain/franchise trap: name overlap × (1 − addr overlap)

Embedding features (optional, 0.0 when Stage B not run):
  14. embedding_cosine      — cosine similarity from multilingual embedding model

Public API:
    build_feature_matrix(pairs, s1, s2s3_combined) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import re as _re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Any run of digits — used for house/street-number extraction from addresses.
_DIGIT_RUN_RE = _re.compile(r"\d+")

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import LCSseq as _LCSseq

    def _levenshtein_ratio(a: str, b: str) -> float:
        return _fuzz.ratio(a, b) / 100.0

    def _token_sort_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0

    def _lcs_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        lcs = _LCSseq.similarity(a, b)
        return lcs / max(len(a), len(b))

except ImportError:
    logger.warning("rapidfuzz not installed — string similarity features degraded.")
    import difflib

    def _levenshtein_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _token_sort_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        a_s = " ".join(sorted(a.split()))
        b_s = " ".join(sorted(b.split()))
        return difflib.SequenceMatcher(None, a_s, b_s).ratio()

    def _lcs_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        if not a and not b:
            return 1.0
        m = difflib.SequenceMatcher(None, a, b)
        lcs = sum(t.size for t in m.get_matching_blocks())
        return lcs / max(len(a), len(b))


def _token_set_jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Hard-bucket features
# These explicitly model the failure modes that decide the top-of-leaderboard
# margin: name containment (sub-brands), acronym/initialism equivalence,
# house-number agreement (a cross-lingual precision signal), name length
# ratio, and the chain/franchise trap (identical name, different address =>
# usually a DIFFERENT entity, i.e. a strong non-merge signal).
# All are pure-python and vectorised via list comprehensions — no new deps.
# ---------------------------------------------------------------------------

def _containment(a_set: str, b_set: str) -> float:
    """
    Token-set containment: |A ∩ B| / min(|A|, |B|).
    Returns 1.0 when one name's tokens are a subset of the other's
    ("reliance" ⊂ "reliance digital"), which plain Jaccard understates.
    Inputs are space-joined deduped token strings (norm_name_set).
    """
    sa = set(a_set.split())
    sb = set(b_set.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _acronym_match(a: str, b: str) -> float:
    """
    1.0 if one side is the initialism of the other
    ("ibm" ↔ "international business machines"), else 0.0.
    Inputs are normalized (lowercased, space-tokenised) names.
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0

    def _acro(toks: list[str]) -> str:
        return "".join(t[0] for t in toks if t)

    if len(ta) == 1 and len(tb) >= 2 and ta[0] == _acro(tb):
        return 1.0
    if len(tb) == 1 and len(ta) >= 2 and tb[0] == _acro(ta):
        return 1.0
    return 0.0


def _house_numbers(addr: str) -> set[str]:
    return set(_DIGIT_RUN_RE.findall(addr))


def _house_number_match(a_addr: str, b_addr: str) -> float:
    """
    Numeric street/house-number agreement between two addresses.
      1.0  → both have digits and at least one overlaps  (strong same-place)
      0.0  → both have digits but none overlap            (different place)
     -1.0  → at least one side has no digits              (unknown / missing)
    A sentinel-for-missing scheme LightGBM splits on cleanly; survives across
    languages because digits are script-invariant.
    """
    na, nb = _house_numbers(a_addr), _house_numbers(b_addr)
    if not na or not nb:
        return -1.0
    return 1.0 if (na & nb) else 0.0


def _len_ratio(a: str, b: str) -> float:
    """Character length ratio min/max of two normalized names (0–1)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ---------------------------------------------------------------------------
# TF-IDF char n-gram vectorizer (fitted once per call to build_feature_matrix)
# ---------------------------------------------------------------------------

def _fit_tfidf(corpus: list[str], ngram_range=(3, 3)) -> TfidfVectorizer:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1)
    vec.fit(corpus)
    return vec


def fit_global_tfidf(name_series_list: list, ngram_range=(3, 3)) -> TfidfVectorizer:
    """
    Fit ONE TfidfVectorizer on the union of all supplied normalized-name
    corpora (e.g. train S1+S2+S3). Reuse the returned vectorizer for every
    build_feature_matrix call (train, val, test) so tfidf_cosine is computed
    on a single, stable vocabulary and does not skew across splits.
    """
    corpus: list[str] = []
    for s in name_series_list:
        corpus.extend(x for x in s.fillna("").tolist() if x)
    corpus = list(set(corpus)) or [""]
    logger.info("Fitting global TF-IDF on %d unique normalized names ...", len(corpus))
    return _fit_tfidf(corpus, ngram_range=ngram_range)


def _tfidf_cosine_batch(
    a_texts: list[str],
    b_texts: list[str],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Compute diagonal cosine similarities for two parallel lists."""
    A = vectorizer.transform(a_texts)
    B = vectorizer.transform(b_texts)
    # Diagonal of cosine_similarity matrix
    norms_A = np.sqrt(A.multiply(A).sum(axis=1)).A1
    norms_B = np.sqrt(B.multiply(B).sum(axis=1)).A1
    dots = A.multiply(B).sum(axis=1).A1
    denom = norms_A * norms_B
    sims = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom > 0)
    return sims


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    sx_combined: pd.DataFrame,
    embedding_cosine_map: Optional[dict[tuple[str, str], float]] = None,
    tfidf_vectorizer: Optional[TfidfVectorizer] = None,
) -> pd.DataFrame:
    """
    Compute pairwise similarity features for all candidate pairs.

    Parameters
    ----------
    pairs : DataFrame with columns [source1_entity_id, candidate_entity_id]
            (one row per pair — exploded from comma-separated lists)
    s1    : Source 1 DataFrame (with norm_name, norm_name_sorted,
            norm_name_set, norm_address, country columns)
    sx_combined : Concatenated S2+S3 DataFrame (same norm_* columns)
    embedding_cosine_map : Optional dict {(s1_id, sx_id): cosine_score}
                           from Stage B; pairs not in map get 0.0.
    tfidf_vectorizer : Optional pre-fitted TfidfVectorizer. Pass a single
                       vectorizer fitted once (e.g. on the global training
                       corpus) across train/val/test to keep the tfidf_cosine
                       feature on an identical vocabulary — otherwise the
                       feature distribution shifts between splits (train/test
                       skew). If None, a vectorizer is fitted on this pair
                       set's texts (legacy behaviour).

    Returns
    -------
    pd.DataFrame with one row per pair and feature columns + label columns
    (source1_entity_id, candidate_entity_id retained for join-back).
    """
    logger.info("Building feature matrix for %d pairs ...", len(pairs))

    if len(pairs) == 0:
        # Handle empty case gracefully
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "token_set_jaccard",
            "levenshtein_ratio", "token_sort_ratio", "lcs_ratio", "tfidf_cosine",
            "address_token_jaccard", "address_levenshtein", "country_match",
            "name_containment", "acronym_match", "house_number_match",
            "name_len_ratio", "name_addr_mismatch", "embedding_cosine"
        ])

    # ---- 1. Fast alignment via pandas merge ----
    s1_cols = s1[['entity_id', 'norm_name', 'norm_name_set', 'norm_address', 'country']].rename(columns={
        'entity_id': 'source1_entity_id',
        'norm_name': 's1_nn',
        'norm_name_set': 's1_nse',
        'norm_address': 's1_na',
        'country': 's1_cty',
    })
    
    sx_cols = sx_combined[['entity_id', 'norm_name', 'norm_name_set', 'norm_address', 'country']].rename(columns={
        'entity_id': 'candidate_entity_id',
        'norm_name': 'sx_nn',
        'norm_name_set': 'sx_nse',
        'norm_address': 'sx_na',
        'country': 'sx_cty',
    })

    # Merge pairs with source data
    merged = pairs.merge(s1_cols, on='source1_entity_id', how='left')
    merged = merged.merge(sx_cols, on='candidate_entity_id', how='left')

    # Fill NaNs with empty string
    for c in ['s1_nn', 's1_nse', 's1_na', 's1_cty', 'sx_nn', 'sx_nse', 'sx_na', 'sx_cty']:
        merged[c] = merged[c].fillna("")

    # ---- 2. Extract to parallel lists for fast iteration ----
    s1_ids = merged['source1_entity_id'].tolist()
    sx_ids = merged['candidate_entity_id'].tolist()
    s1_nn_list = merged['s1_nn'].tolist()
    sx_nn_list = merged['sx_nn'].tolist()
    s1_nse_list = merged['s1_nse'].tolist()
    sx_nse_list = merged['sx_nse'].tolist()
    s1_na_list = merged['s1_na'].tolist()
    sx_na_list = merged['sx_na'].tolist()
    s1_cty_list = merged['s1_cty'].tolist()
    sx_cty_list = merged['sx_cty'].tolist()

    # ---- 3. TF-IDF Cosine ----
    if tfidf_vectorizer is not None:
        tfidf_sims = _tfidf_cosine_batch(s1_nn_list, sx_nn_list, tfidf_vectorizer)
    else:
        all_texts = list(set(s1_nn_list + sx_nn_list))
        if all_texts:
            tfidf_vec  = _fit_tfidf(all_texts)
            tfidf_sims = _tfidf_cosine_batch(s1_nn_list, sx_nn_list, tfidf_vec)
        else:
            tfidf_sims = np.zeros(len(pairs))

    # ---- 4. Compute features via fast list comprehensions ----
    # List comprehensions + rapidfuzz C-extensions are extremely fast
    token_set_jaccard = [_token_set_jaccard(a, b) for a, b in zip(s1_nse_list, sx_nse_list)]
    levenshtein_ratio = [_levenshtein_ratio(a, b) for a, b in zip(s1_nn_list, sx_nn_list)]
    token_sort_ratio = [_token_sort_ratio(a, b) for a, b in zip(s1_nn_list, sx_nn_list)]
    lcs_ratio = [_lcs_ratio(a, b) for a, b in zip(s1_nn_list, sx_nn_list)]
    address_token_jaccard = [_token_set_jaccard(a, b) for a, b in zip(s1_na_list, sx_na_list)]
    address_levenshtein = [_levenshtein_ratio(a, b) for a, b in zip(s1_na_list, sx_na_list)]
    country_match = [1.0 if a == b else 0.0 for a, b in zip(s1_cty_list, sx_cty_list)]

    # ---- 4b. Hard-bucket features (sub-brands, acronyms, chains, house #) ----
    name_containment  = [_containment(a, b)        for a, b in zip(s1_nse_list, sx_nse_list)]
    acronym_match     = [_acronym_match(a, b)      for a, b in zip(s1_nn_list, sx_nn_list)]
    house_number_match = [_house_number_match(a, b) for a, b in zip(s1_na_list, sx_na_list)]
    name_len_ratio    = [_len_ratio(a, b)          for a, b in zip(s1_nn_list, sx_nn_list)]
    # Chain/franchise trap: identical name + divergent address => likely a
    # DIFFERENT entity. High when name overlap is strong but address overlap
    # is weak; the tree learns this as a non-merge signal.
    _tsj = np.asarray(token_set_jaccard, dtype="float64")
    _atj = np.asarray(address_token_jaccard, dtype="float64")
    name_addr_mismatch = (_tsj * (1.0 - _atj)).tolist()

    if embedding_cosine_map:
        embedding_cosine = [embedding_cosine_map.get((a, b), 0.0) for a, b in zip(s1_ids, sx_ids)]
    else:
        embedding_cosine = [0.0] * len(pairs)

    # ---- 5. Construct final DataFrame ----
    df_features = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": sx_ids,
        "token_set_jaccard": token_set_jaccard,
        "levenshtein_ratio": levenshtein_ratio,
        "token_sort_ratio": token_sort_ratio,
        "lcs_ratio": lcs_ratio,
        "tfidf_cosine": tfidf_sims,
        "address_token_jaccard": address_token_jaccard,
        "address_levenshtein": address_levenshtein,
        "country_match": country_match,
        "name_containment": name_containment,
        "acronym_match": acronym_match,
        "house_number_match": house_number_match,
        "name_len_ratio": name_len_ratio,
        "name_addr_mismatch": name_addr_mismatch,
        "embedding_cosine": embedding_cosine,
    })

    logger.info("Feature matrix built: %d rows × %d cols", len(df_features), len(df_features.columns))
    return df_features


FEATURE_COLS = [
    "token_set_jaccard",
    "levenshtein_ratio",
    "token_sort_ratio",
    "lcs_ratio",
    "tfidf_cosine",
    "address_token_jaccard",
    "address_levenshtein",
    "country_match",
    "name_containment",
    "acronym_match",
    "house_number_match",
    "name_len_ratio",
    "name_addr_mismatch",
    "embedding_cosine",
]


def compute_embedding_cosine_map(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    sx_combined: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 512,
    cache_path: Optional[str | Path] = None,
    resume: bool = True,
) -> dict[tuple[str, str], float]:
    """
    Compute an embedding cosine score for EVERY candidate pair (not just the
    ones Stage B's ANN surfaced). This makes `embedding_cosine` a genuine
    similarity feature across the whole candidate set, rather than a de-facto
    "came from Stage B" source-indicator (which would leak and mislead the
    classifier).

    Supports disk checkpoint caching for instant resume.
    """
    if len(pairs) == 0:
        return {}

    if cache_path is not None and resume:
        cp = Path(cache_path)
        if cp.is_file() and cp.stat().st_size > 0:
            try:
                import pickle
                with open(cp, "rb") as fh:
                    cached_map = pickle.load(fh)
                logger.info("Loaded embedding cosine map (%d pairs) from cache: %s", len(cached_map), cp)
                return cached_map
            except Exception as e:
                logger.warning("Failed to load embedding map cache from %s (%s). Recomputing.", cp, e)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — embedding_cosine feature "
            "will be 0.0 for all pairs."
        )
        return {}

    s1_names = s1.set_index("entity_id")["norm_name"].to_dict()
    sx_names = sx_combined.set_index("entity_id")["norm_name"].to_dict()

    # Collect the distinct entity ids actually referenced by the pairs.
    s1_ids = pairs["source1_entity_id"].unique().tolist()
    sx_ids = pairs["candidate_entity_id"].unique().tolist()

    def _encode(ids, name_lookup):
        texts = [name_lookup.get(i, "") or "" for i in ids]
        vecs = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return {i: v for i, v in zip(ids, vecs)}

    logger.info("Embedding %d unique S1 + %d unique Sx names for pair cosine ...",
                len(s1_ids), len(sx_ids))
    model = SentenceTransformer(model_name)
    s1_vec = _encode(s1_ids, s1_names)
    sx_vec = _encode(sx_ids, sx_names)

    out: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        a = s1_vec.get(row.source1_entity_id)
        b = sx_vec.get(row.candidate_entity_id)
        if a is not None and b is not None:
            out[(row.source1_entity_id, row.candidate_entity_id)] = float(np.dot(a, b))

    if cache_path is not None:
        try:
            import pickle
            cp = Path(cache_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "wb") as fh:
                pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved embedding cosine map (%d pairs) to cache: %s", len(out), cp)
        except Exception as e:
            logger.warning("Failed to save embedding map cache: %s", e)

    return out
