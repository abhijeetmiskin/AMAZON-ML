"""
High-performance pairwise feature engineering using RapidFuzz and 64-core parallelism.
Extracts string similarity, token overlap, numeric address alignment,
character trigram Jaccard, and structural indicators across candidate pairs.
"""

import re
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import numpy as np
from rapidfuzz import fuzz

from preprocessing import (
    clean_business_name,
    clean_business_address,
    extract_numbers
)

FEATURE_NAMES = [
    "name_exact",
    "name_token_sort",
    "name_token_set",
    "name_ratio",
    "name_partial_ratio",
    "name_trigram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "addr_exact",
    "addr_token_sort",
    "addr_token_set",
    "addr_num_overlap",
    "addr_is_null",
    "source_prefix",
    "heuristic_score",
]


def _get_char_trigrams(text: str) -> set:
    s = re.sub(r"[^\w]", "", text.lower())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str,
    t_id: str,
    country: str
) -> List[float]:
    """Extract a 15-dimensional feature vector for a candidate pair."""
    cn1 = clean_business_name(s1_name, country)
    cn2 = clean_business_name(t_name, country)

    ca1 = clean_business_address(s1_addr, country)
    ca2 = clean_business_address(t_addr, country)

    # Name features
    name_exact = 1.0 if cn1 and cn1 == cn2 else 0.0
    name_token_sort = float(fuzz.token_sort_ratio(cn1, cn2))
    name_token_set = float(fuzz.token_set_ratio(cn1, cn2))
    name_ratio = float(fuzz.ratio(cn1, cn2))
    name_partial_ratio = float(fuzz.partial_ratio(cn1, cn2))

    # Character trigram jaccard
    tri1 = _get_char_trigrams(cn1)
    tri2 = _get_char_trigrams(cn2)
    name_trigram_jaccard = float(len(tri1 & tri2) / max(len(tri1 | tri2), 1)) if (tri1 or tri2) else 0.0

    l1, l2 = len(cn1), len(cn2)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    # Address features
    addr_is_null = 1.0 if not ca1 or not ca2 else 0.0
    if not addr_is_null:
        addr_exact = 1.0 if ca1 == ca2 else 0.0
        addr_token_sort = float(fuzz.token_sort_ratio(ca1, ca2))
        addr_token_set = float(fuzz.token_set_ratio(ca1, ca2))

        nums1 = set(extract_numbers(ca1))
        nums2 = set(extract_numbers(ca2))
        addr_num_overlap = 1.0 if nums1 and nums2 and (nums1 & nums2) else 0.0
    else:
        addr_exact = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_num_overlap = 0.0

    source_prefix = 2.0 if t_id.startswith("S2-") else 3.0

    # Heuristic combined indicator
    if addr_is_null:
        heuristic_score = name_token_set
    else:
        heuristic_score = 0.50 * name_token_set + 0.50 * addr_token_set

    return [
        name_exact,
        name_token_sort,
        name_token_set,
        name_ratio,
        name_partial_ratio,
        name_trigram_jaccard,
        name_len_diff,
        name_len_ratio,
        addr_exact,
        addr_token_sort,
        addr_token_set,
        addr_num_overlap,
        addr_is_null,
        source_prefix,
        heuristic_score,
    ]


def _extract_chunk(chunk_pairs, s1_data, target_data):
    features = []
    for s1_id, t_id in chunk_pairs:
        s1_name, s1_addr, country = s1_data[s1_id]
        t_name, t_addr, _ = target_data[t_id]
        feat = extract_pair_features(s1_name, s1_addr, t_name, t_addr, t_id, country)
        features.append(feat)
    return features


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    s1_data: Dict[str, Tuple[str, str, str]],
    target_data: Dict[str, Tuple[str, str, str]],
    n_jobs: int = 32
) -> np.ndarray:
    """Build feature matrix across a list of candidate pairs in parallel across CPU cores."""
    if not candidate_pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    n_pairs = len(candidate_pairs)
    if n_pairs < 20000 or n_jobs == 1:
        features = _extract_chunk(candidate_pairs, s1_data, target_data)
        return np.array(features, dtype=np.float32)

    chunk_size = max(5000, n_pairs // n_jobs)
    chunks = [candidate_pairs[i:i+chunk_size] for i in range(0, n_pairs, chunk_size)]

    results = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_extract_chunk)(c, s1_data, target_data) for c in chunks
    )
    features = [f for sub in results for f in sub]
    return np.array(features, dtype=np.float32)
