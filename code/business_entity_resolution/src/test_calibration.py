"""
Test Calibration Script: Evaluates realistic hard-negative training,
enhanced 22D feature engineering, cardinality capping, margin pruning,
and singleton gating on 5,000 validation S1 entities.
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.append("/home/cim/AMZON ML/code/business_entity_resolution/src")
from preprocessing import clean_business_name, clean_business_address, extract_numbers
from metrics import compute_macro_f05


FEATURE_NAMES_22 = [
    "name_exact",
    "name_token_sort",
    "name_token_set",
    "name_ratio",
    "name_partial_ratio",
    "name_trigram_jaccard",
    "name_word_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_first_token_match",
    "addr_exact",
    "addr_token_sort",
    "addr_token_set",
    "addr_num_overlap",
    "addr_num_mismatch",
    "addr_is_null",
    "s1_addr_null",
    "t_addr_null",
    "name_x_addr",
    "name_sort_x_addr_sort",
    "source_prefix",
    "heuristic_score",
]


def _get_char_trigrams(text: str) -> set:
    s = re.sub(r"[^\w]", "", text.lower())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}


def extract_pair_features_22(
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str,
    t_id: str,
    country: str,
) -> List[float]:
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

    tri1 = _get_char_trigrams(cn1)
    tri2 = _get_char_trigrams(cn2)
    name_trigram_jaccard = float(len(tri1 & tri2) / max(len(tri1 | tri2), 1)) if (tri1 or tri2) else 0.0

    w1 = set(cn1.split())
    w2 = set(cn2.split())
    name_word_jaccard = float(len(w1 & w2) / max(len(w1 | w2), 1)) if (w1 or w2) else 0.0

    l1, l2 = len(cn1), len(cn2)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    toks1 = cn1.split()
    toks2 = cn2.split()
    name_first_token_match = 1.0 if toks1 and toks2 and toks1[0] == toks2[0] else 0.0

    # Address features
    s1_addr_null = 1.0 if not ca1 else 0.0
    t_addr_null = 1.0 if not ca2 else 0.0
    addr_is_null = 1.0 if (s1_addr_null or t_addr_null) else 0.0

    nums1 = set(extract_numbers(ca1)) if ca1 else set()
    nums2 = set(extract_numbers(ca2)) if ca2 else set()

    if not addr_is_null:
        addr_exact = 1.0 if ca1 == ca2 else 0.0
        addr_token_sort = float(fuzz.token_sort_ratio(ca1, ca2))
        addr_token_set = float(fuzz.token_set_ratio(ca1, ca2))
        addr_num_overlap = 1.0 if nums1 and nums2 and (nums1 & nums2) else 0.0
        addr_num_mismatch = 1.0 if nums1 and nums2 and not (nums1 & nums2) else 0.0
    else:
        addr_exact = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_num_overlap = 0.0
        addr_num_mismatch = 0.0

    name_x_addr = (name_token_set / 100.0) * (addr_token_set / 100.0 if not addr_is_null else (name_token_set / 100.0))
    name_sort_x_addr_sort = (name_token_sort / 100.0) * (addr_token_sort / 100.0 if not addr_is_null else (name_token_sort / 100.0))

    source_prefix = 2.0 if t_id.startswith("S2-") else 3.0

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
        name_word_jaccard,
        name_len_diff,
        name_len_ratio,
        name_first_token_match,
        addr_exact,
        addr_token_sort,
        addr_token_set,
        addr_num_overlap,
        addr_num_mismatch,
        addr_is_null,
        s1_addr_null,
        t_addr_null,
        name_x_addr,
        name_sort_x_addr_sort,
        source_prefix,
        heuristic_score,
    ]


def build_matrix_parallel(pairs, s1_data, target_data, n_jobs=32):
    from joblib import Parallel, delayed

    def _chunk(chk):
        feats = []
        for s1_id, t_id in chk:
            s_name, s_addr, country = s1_data[s1_id]
            t_name, t_addr, _ = target_data[t_id]
            feats.append(extract_pair_features_22(s_name, s_addr, t_name, t_addr, t_id, country))
        return feats

    if len(pairs) < 10000 or n_jobs == 1:
        return np.array(_chunk(pairs), dtype=np.float32)

    chunk_size = max(2000, len(pairs) // n_jobs)
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]
    results = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_chunk)(c) for c in chunks)
    all_feats = [f for sub in results for f in sub]
    return np.array(all_feats, dtype=np.float32)


def main():
    print("=" * 60, flush=True)
    print("RUNNING ADVANCED CALIBRATION EXPERIMENT", flush=True)
    print("=" * 60, flush=True)

    train_dir = "/home/cim/AMZON ML/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    t0 = time.time()
    s1 = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t")
    s2 = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t")
    s3 = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t")
    gt = pl.read_csv(f"{train_dir}/train_ground_truth.tsv", separator="\t")
    targets = pl.concat([s2, s3])
    print(f"Loaded datasets in {time.time()-t0:.2f}s", flush=True)

    # Sample 15,000 S1 entities (10k train, 5k val)
    s1_sampled = s1.sample(15000, seed=42)
    s1_train = s1_sampled.slice(0, 10000)
    s1_val = s1_sampled.slice(10000, 5000)

    val_ids = set(s1_val["entity_id"])
    train_ids = set(s1_train["entity_id"])

    gt_map = {}
    needed_tids = set()
    for r in gt.filter(pl.col("source1_entity_id").is_in(set(s1_sampled["entity_id"]))).iter_rows(named=True):
        m = r["matched_entity_ids"]
        tids = set(m.split(",")) if m else set()
        gt_map[r["source1_entity_id"]] = tids
        needed_tids.update(tids)

    print(f"Sampled 15,000 S1 queries with {len(needed_tids):,} true target IDs", flush=True)

    # Create target pool: true targets + 350,000 distractors
    true_targets = targets.filter(pl.col("entity_id").is_in(needed_tids))
    distractors = targets.filter(~pl.col("entity_id").is_in(needed_tids)).sample(350000, seed=42)
    pool = pl.concat([true_targets, distractors])
    print(f"Total target pool: {len(pool):,} records", flush=True)

    # Pre-index pool
    target_data = {r["entity_id"]: (r["business_name"], r.get("business_address") or "", r["country"]) for r in pool.iter_rows(named=True)}
    s1_data = {r["entity_id"]: (r["business_name"], r.get("business_address") or "", r["country"]) for r in s1_sampled.iter_rows(named=True)}

    name_idx = defaultdict(list)
    digit_idx = defaultdict(list)
    for tid, (tname, taddr, country) in target_data.items():
        cn = clean_business_name(tname, country)
        if cn: name_idx[(country, cn)].append(tid)
        ca = clean_business_address(taddr, country)
        nums = extract_numbers(ca)
        words = [w for w in ca.split() if len(w) >= 4 and not w.isdigit()]
        for num in nums:
            if len(num) >= 4:
                digit_idx[(country, num)].append(tid)
            elif len(num) >= 2 and words:
                for w in words[:2]:
                    digit_idx[(country, num + "_" + w[:4])].append(tid)

    # Generate candidate pairs for train and val
    def get_cands(df_s1):
        cand_dict = {}
        for r in df_s1.iter_rows(named=True):
            sid = r["entity_id"]
            c = r["country"]
            cands = set()
            cn = clean_business_name(r["business_name"], c)
            if cn and (c, cn) in name_idx:
                cands.update(name_idx[(c, cn)])
            ca = clean_business_address(r.get("business_address"), c)
            nums = extract_numbers(ca)
            words = [w for w in ca.split() if len(w) >= 4 and not w.isdigit()]
            for num in nums:
                if len(num) >= 4 and (c, num) in digit_idx:
                    cands.update(digit_idx[(c, num)][:20])
                elif len(num) >= 2 and words:
                    for w in words[:2]:
                        k = (c, num + "_" + w[:4])
                        if k in digit_idx:
                            cands.update(digit_idx[k][:20])
            cand_dict[sid] = cands
        return cand_dict

    print("Generating candidates for train and val...", flush=True)
    train_cands = get_cands(s1_train)
    val_cands = get_cands(s1_val)

    # Guarantee true matches are in train candidates so model sees positives
    train_pairs = []
    for sid, cands in train_cands.items():
        all_cands = set(cands) | gt_map.get(sid, set())
        for tid in all_cands:
            if tid in target_data:
                train_pairs.append((sid, tid))

    val_pairs = []
    for sid, cands in val_cands.items():
        for tid in cands:
            if tid in target_data:
                val_pairs.append((sid, tid))

    y_train = np.array([1 if tid in gt_map.get(sid, set()) else 0 for sid, tid in train_pairs], dtype=np.int32)
    y_val = np.array([1 if tid in gt_map.get(sid, set()) else 0 for sid, tid in val_pairs], dtype=np.int32)

    print(f"Train candidate pairs: {len(train_pairs):,} (Pos: {y_train.sum():,}, Neg: {(1-y_train).sum():,})", flush=True)
    print(f"Val candidate pairs:   {len(val_pairs):,} (Pos: {y_val.sum():,}, Neg: {(1-y_val).sum():,})", flush=True)

    # Extract 22D features
    t_f = time.time()
    print("Building 22D feature matrices in parallel (32 cores)...", flush=True)
    X_train = build_matrix_parallel(train_pairs, s1_data, target_data, n_jobs=32)
    X_val = build_matrix_parallel(val_pairs, s1_data, target_data, n_jobs=32)
    print(f"Extracted features in {time.time()-t_f:.2f}s!", flush=True)

    # Train LightGBM model with n_jobs=8 to prevent OpenMP contention
    print("Training calibrated LightGBM classifier (8 OpenMP threads)...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
        verbosity=-1
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"Trained LightGBM in {time.time()-t_train:.2f}s!", flush=True)

    val_probs = clf.predict_proba(X_val)[:, 1]

    # Pre-sort candidate lists once per entity
    s1_to_scored = {s: [] for s in val_ids}
    for (sid, tid), p in zip(val_pairs, val_probs):
        s1_to_scored[sid].append((tid, float(p)))

    for s in val_ids:
        s1_to_scored[s].sort(key=lambda x: -x[1])

    # Feature importances
    print("\nFeature Importances:")
    for fn, imp in sorted(zip(FEATURE_NAMES_22, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {fn:25s}: {imp}", flush=True)

    # Sweep thresholds and post-processing policies
    print("\n" + "=" * 60, flush=True)
    print("EVALUATING CALIBRATION POLICIES ON 5,000 VALIDATION ENTITIES", flush=True)
    print("=" * 60, flush=True)

    best_macro = 0.0
    best_config = None

    for tau in [0.70, 0.75, 0.80, 0.83, 0.86, 0.89, 0.92]:
        for max_k in [3, 4, 5, 6, 8]:
            for margin in [0.15, 0.25, 0.40, 1.0]:
                for singleton_gate in [0.0, 0.80, 0.85, 0.88]:
                    preds = {}
                    for sid in val_ids:
                        cands = s1_to_scored.get(sid, [])
                        if not cands:
                            preds[sid] = set()
                            continue

                        top_p = cands[0][1]

                        # Singleton gate
                        if singleton_gate > 0 and top_p < singleton_gate:
                            preds[sid] = set()
                            continue

                        # Filter by threshold and margin
                        selected = []
                        for tid, p in cands:
                            if p >= tau and (top_p - p) <= margin:
                                selected.append(tid)
                            if len(selected) >= max_k:
                                break

                        preds[sid] = set(selected)

                    res = compute_macro_f05(gt_map, preds, all_s1_ids=val_ids)
                    score = res["macro_f05"]
                    if score > best_macro:
                        best_macro = score
                        best_config = (tau, max_k, margin, singleton_gate, res)
                        print(f"New Best: F0.5={score:.4f} | Prec={res['macro_precision']:.4f} | Rec={res['macro_recall']:.4f} | SingleAcc={res['singleton_acc']:.2%} | (tau={tau}, max_k={max_k}, margin={margin}, gate={singleton_gate})", flush=True)

    b_tau, b_k, b_margin, b_gate, b_res = best_config
    print("\n" + "=" * 60, flush=True)
    print("OPTIMAL CALIBRATED CONFIGURATION:", flush=True)
    print(f"  Best Validation Macro F_0.5: {best_macro:.4f}", flush=True)
    print(f"  Best Precision:              {b_res['macro_precision']:.4f}", flush=True)
    print(f"  Best Recall:                 {b_res['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy:          {b_res['singleton_acc']:.2%}", flush=True)
    print(f"  Threshold (tau):             {b_tau}", flush=True)
    print(f"  Max Matches per Entity (k):  {b_k}", flush=True)
    print(f"  Score Margin from Top:       {b_margin}", flush=True)
    print(f"  Singleton Gate Threshold:    {b_gate}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
