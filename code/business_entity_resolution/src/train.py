"""
High-Performance Model Training and Threshold Calibration (0.96+ Target).
Trains LightGBM using Hybrid (Dense GPU + Lexical) Candidate Pairs.
"""

import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import polars as pl

from blocking import HybridBlocker
from features import build_feature_matrix
from metrics import compute_macro_f05
from model import MatchRanker


def run_training(
    train_dir: str = "/home/cim/AMZON ML/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
    model_save_dir: str = "/home/cim/AMZON ML/code/business_entity_resolution/models",
    n_train_samples: int = 10000,
    n_val_samples: int = 3000,
    random_seed: int = 42
):
    print("=" * 60)
    print("STARTING 0.96+ HYBRID MODEL TRAINING & VALIDATION")
    print("=" * 60)

    os.makedirs(model_save_dir, exist_ok=True)

    t0 = time.time()
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading datasets...")
    s1_df = pl.read_csv(s1_path, separator="\t")
    s2_df = pl.read_csv(s2_path, separator="\t")
    s3_df = pl.read_csv(s3_path, separator="\t")
    gt_df = pl.read_csv(gt_path, separator="\t")
    targets_df = pl.concat([s2_df, s3_df])
    print(f"Loaded datasets in {time.time()-t0:.2f}s")

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
    print(f"Training S1 entities: {len(train_ids):,}, Validation S1 entities: {len(val_ids):,}")

    # Gather needed true target IDs + representative distractors
    needed_tids = set()
    for sid in s1_sampled["entity_id"]:
        needed_tids.update(gt_map.get(sid, set()))

    distractors = targets_df.filter(~pl.col("entity_id").is_in(needed_tids)).sample(50000, seed=random_seed)
    pool_targets = pl.concat([targets_df.filter(pl.col("entity_id").is_in(needed_tids)), distractors])
    print(f"Target pool for training: {len(pool_targets):,} records")

    # Fit Hybrid Blocker per country
    blocker = HybridBlocker(dense_top_k=15)
    countries = s1_sampled["country"].unique().to_list()

    all_train_cands: Dict[str, Set[str]] = {}
    all_val_cands: Dict[str, Set[str]] = {}

    for country in countries:
        c_targets = pool_targets.filter(pl.col("country") == country)
        c_s1_train = s1_train.filter(pl.col("country") == country)
        c_s1_val = s1_val.filter(pl.col("country") == country)

        blocker.fit_country_targets(country, c_targets)
        c_train_cands = blocker.query_country_candidates(country, c_s1_train)
        c_val_cands = blocker.query_country_candidates(country, c_s1_val)

        all_train_cands.update(c_train_cands)
        all_val_cands.update(c_val_cands)

    val_gt_pairs = sum(len(gt_map[sid]) for sid in val_ids)
    val_recovered = sum(len(gt_map[sid] & all_val_cands[sid]) for sid in val_ids)
    print(f"\nValidation Hybrid Blocking Recall: {val_recovered:,} / {val_gt_pairs:,} ({val_recovered/val_gt_pairs:.2%})")

    # Prepare data dicts for feature extraction
    s1_data: Dict[str, Tuple[str, str, str]] = {}
    for r in s1_sampled.iter_rows(named=True):
        s1_data[r["entity_id"]] = (r["business_name"], r.get("business_address") or "", r["country"])

    target_data = blocker.target_data

    # Flatten candidate pairs
    train_pairs = [(sid, tid) for sid, cands in all_train_cands.items() for tid in cands]
    y_train = np.array([1 if tid in gt_map[sid] else 0 for sid, tid in train_pairs], dtype=np.int32)

    val_pairs = [(sid, tid) for sid, cands in all_val_cands.items() for tid in cands]
    y_val = np.array([1 if tid in gt_map[sid] else 0 for sid, tid in val_pairs], dtype=np.int32)

    print(f"Train candidate pairs: {len(train_pairs):,} (Positives: {sum(y_train):,})")
    print(f"Val candidate pairs: {len(val_pairs):,} (Positives: {sum(y_val):,})")

    # Extract Features
    t_feat = time.time()
    print("Extracting pairwise features...")
    X_train = build_feature_matrix(train_pairs, s1_data, target_data)
    X_val = build_feature_matrix(val_pairs, s1_data, target_data)
    print(f"Extracted features in {time.time()-t_feat:.2f}s!")

    # Train LightGBM Matcher
    print("Training LightGBM match ranker...")
    ranker = MatchRanker(n_estimators=400, learning_rate=0.06, max_depth=6)
    ranker.fit(X_train, y_train, X_val, y_val)

    # Threshold Optimization for Macro F_0.5
    val_probs = ranker.predict_proba(X_val)
    best_tau = ranker.optimize_threshold(
        candidate_pairs=val_pairs,
        probabilities=val_probs,
        ground_truth=gt_map,
        all_s1_ids=list(val_ids),
    )

    final_preds: Dict[str, Set[str]] = {s: set() for s in val_ids}
    for (s1_id, tid), p in zip(val_pairs, val_probs):
        if p >= best_tau:
            final_preds[s1_id].add(tid)

    metrics = compute_macro_f05(gt_map, final_preds, all_s1_ids=list(val_ids))
    print("\n" + "=" * 60)
    print("0.96+ HYBRID MODEL VALIDATION METRICS:")
    print(f"  Macro F_0.5 Score:     {metrics['macro_f05']:.4f}")
    print(f"  Macro Precision:       {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:          {metrics['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:    {metrics['singleton_acc']:.2%}")
    print(f"  Total Singletons:      {metrics['num_singletons']:,}")
    print(f"  Total S1 Entities:     {metrics['num_entities']:,}")
    print("=" * 60)

    # Save Model
    model_path = os.path.join(model_save_dir, "lgbm_matcher.joblib")
    config_path = os.path.join(model_save_dir, "model_config.json")
    print(f"Saving 0.96+ model to {model_path}...")
    joblib.dump(ranker, model_path)

    config = {
        "optimal_threshold": float(best_tau),
        "macro_f05": float(metrics["macro_f05"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "singleton_acc": float(metrics["singleton_acc"]),
        "architecture": "Hybrid (Dense GPU + Exact Name + Invariant Address Digits)"
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {config_path}!")

    return ranker, blocker, metrics


if __name__ == "__main__":
    run_training()
