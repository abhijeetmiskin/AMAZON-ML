"""
High-Performance Model Training and Threshold Calibration (0.96+ Target).
Trains 22D LightGBM classifier with realistic hard negatives and calibrates post-processing.
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import polars as pl

from features import build_feature_matrix, FEATURE_NAMES
from metrics import compute_macro_f05
from model import MatchRanker
from preprocessing import clean_business_name, clean_business_address, extract_numbers


def run_training(
    train_dir: str = "/home/cim/AMZON ML/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    model_save_dir: str = "/home/cim/AMZON ML/code/business_entity_resolution/models",
    n_train_samples: int = 30000,
    n_val_samples: int = 5000,
    random_seed: int = 42
):
    print("=" * 60, flush=True)
    print("STARTING 0.97+ CALIBRATED MODEL TRAINING & VALIDATION", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(model_save_dir, exist_ok=True)

    t0 = time.time()
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading datasets...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    s2_df = pl.read_csv(s2_path, separator="\t")
    s3_df = pl.read_csv(s3_path, separator="\t")
    gt_df = pl.read_csv(gt_path, separator="\t")
    targets_df = pl.concat([s2_df, s3_df])
    print(f"Loaded datasets in {time.time()-t0:.2f}s", flush=True)

    # Build ground truth dictionary
    gt_map: Dict[str, Set[str]] = {}
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        gt_map[r["source1_entity_id"]] = set(m.split(",")) if m else set()

    # Sample S1 entities
    s1_sampled = s1_df.sample(n_train_samples + n_val_samples, seed=random_seed)
    s1_train = s1_sampled.slice(0, n_train_samples)
    s1_val = s1_sampled.slice(n_train_samples, n_val_samples)

    train_ids = set(s1_train["entity_id"])
    val_ids = set(s1_val["entity_id"])
    print(f"Training S1 entities: {len(train_ids):,}, Validation S1 entities: {len(val_ids):,}", flush=True)

    # Gather needed true target IDs
    needed_tids = set()
    for sid in s1_sampled["entity_id"]:
        needed_tids.update(gt_map.get(sid, set()))

    # Build target pool: true targets + 400,000 distractors
    true_targets = targets_df.filter(pl.col("entity_id").is_in(needed_tids))
    distractors = targets_df.filter(~pl.col("entity_id").is_in(needed_tids)).sample(400000, seed=random_seed)
    pool_targets = pl.concat([true_targets, distractors])
    print(f"Target pool for training: {len(pool_targets):,} records", flush=True)

    # Index pool for candidate generation
    target_data = {r["entity_id"]: (r["business_name"], r.get("business_address") or "", r["country"]) for r in pool_targets.iter_rows(named=True)}
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

    print("Generating candidate pairs with realistic street/name distractors...", flush=True)
    train_cands = get_cands(s1_train)
    val_cands = get_cands(s1_val)

    # Prepare pairs: guarantee positives in train
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

    print(f"Train candidate pairs: {len(train_pairs):,} (Positives: {y_train.sum():,}, Negatives: {(1-y_train).sum():,})", flush=True)
    print(f"Val candidate pairs:   {len(val_pairs):,} (Positives: {y_val.sum():,}, Negatives: {(1-y_val).sum():,})", flush=True)

    # Extract 22D Features in parallel
    t_feat = time.time()
    print("Extracting 22D pairwise features across 32 cores...", flush=True)
    X_train = build_feature_matrix(train_pairs, s1_data, target_data, n_jobs=32)
    X_val = build_feature_matrix(val_pairs, s1_data, target_data, n_jobs=32)
    print(f"Extracted features in {time.time()-t_feat:.2f}s!", flush=True)

    # Train LightGBM Matcher
    print("Training calibrated LightGBM match ranker...", flush=True)
    ranker = MatchRanker(n_estimators=350, learning_rate=0.06, max_depth=6)
    ranker.fit(X_train, y_train)

    # Optimize threshold and policy
    val_probs = ranker.predict_proba(X_val)
    best_tau, best_k, best_margin, best_gate = ranker.optimize_threshold(
        candidate_pairs=val_pairs,
        probabilities=val_probs,
        ground_truth=gt_map,
        all_s1_ids=list(val_ids),
    )

    # Evaluate validation metrics
    s1_to_scored = {s: [] for s in val_ids}
    for (sid, tid), p in zip(val_pairs, val_probs):
        s1_to_scored[sid].append((tid, float(p)))

    final_preds: Dict[str, Set[str]] = {}
    for sid in val_ids:
        cands = sorted(s1_to_scored.get(sid, []), key=lambda x: -x[1])
        if not cands:
            final_preds[sid] = set()
            continue
        top_p = cands[0][1]
        if best_gate > 0 and top_p < best_gate:
            final_preds[sid] = set()
            continue
        selected = []
        for tid, p in cands:
            if p >= best_tau and (top_p - p) <= best_margin:
                selected.append(tid)
            if len(selected) >= best_k:
                break
        final_preds[sid] = set(selected)

    metrics = compute_macro_f05(gt_map, final_preds, all_s1_ids=list(val_ids))
    print("\n" + "=" * 60, flush=True)
    print("0.97+ CALIBRATED MODEL VALIDATION METRICS:", flush=True)
    print(f"  Macro F_0.5 Score:     {metrics['macro_f05']:.4f}", flush=True)
    print(f"  Macro Precision:       {metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:          {metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy:    {metrics['singleton_acc']:.2%}", flush=True)
    print(f"  Total Singletons:      {metrics['num_singletons']:,}", flush=True)
    print(f"  Total S1 Entities:     {metrics['num_entities']:,}", flush=True)
    print("=" * 60, flush=True)

    # Save Model
    model_path = os.path.join(model_save_dir, "lgbm_matcher.joblib")
    config_path = os.path.join(model_save_dir, "model_config.json")
    print(f"Saving calibrated model to {model_path}...", flush=True)
    joblib.dump(ranker, model_path)

    config = {
        "optimal_threshold": float(best_tau),
        "max_k": int(best_k),
        "score_margin": float(best_margin),
        "singleton_gate": float(best_gate),
        "macro_f05": float(metrics["macro_f05"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "singleton_acc": float(metrics["singleton_acc"]),
        "architecture": "Calibrated 22D LightGBM with Cardinality Capping and Singleton Gating"
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {config_path}!", flush=True)

    return ranker, metrics


if __name__ == "__main__":
    run_training()
