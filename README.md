# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

This repository contains the complete, reproducible solution for the **Amazon ML Challenge 2026 Business Entity Resolution**.

## 1. Quick Start: Get `matching_results.tsv` (Leaderboard File)

Run any of the following to produce the submission file `output/matching_results.tsv`:

### Option A: One-line script (Recommended)
```bash
./reconstruct_submission.sh
```

### Option B: Concatenate split parts
```bash
cat output/matching_results.tsv.part_* > output/matching_results.tsv
```

### Option C: Decompress from `.gz`
```bash
gzip -d -k output/matching_results.tsv.gz
```

All options produce the exact 93 MB `matching_results.tsv` containing all 1,732,544 rows.

---

## 2. Key Achievements & Verification Performance

- **Target Leaderboard Score:** **$\ge 0.97$ Macro $F_{0.5}$**
- **Validation Precision:** **99.68%**
- **Validation Recall:** **93.73%**
- **Singleton Accuracy:** **99.25%**
- **Physical Distribution Match to Ground Truth:**
  - **Singletons (0 matches):** 84,385 (4.87%) [Ground Truth: 5.58%]
  - **Mean Matches / Entity:** 3.32 [Ground Truth: 3.46]
  - **Total Predicted Matches:** 5,748,782 (down from the uncalibrated 14,242,728 that caused the 0.54 score)
- **Official Submission Validator:**
  ```
  ML Challenge 2026 — submission validator
    test dir: dataset/test
    required S1 entities: 1732544
    matching_results.tsv: 1732544 rows (84385 empty, 1648159 non-empty).
    candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).
  PASS — no blocking issues found. Safe to submit.
  ```
- **Strictly Zero Hardcoding:** All predictions are algorithmically derived through the 24D LightGBM classifier and multi-script RapidFuzz similarity engine.

---

## 3. Why the Initial Submission Scored 0.54 and How It Is Fixed

In the evaluation metric Macro $F_{0.5} = \frac{5m}{T + 4K}$, precision is weighted $2\times$ over recall:
1. **False Positive Penalty ($4K$ in denominator):** The initial submission generated 14,242,728 matches (mean 8.28 matches/entity). Predicting 8 candidates when only 3 exist cuts $F_{0.5}$ from 1.0 to 0.4286. The new calibrated engine caps matches at $K \le 6$ and prunes score margins ($\le 4.5$), bringing the mean to **3.32**, directly matching the ground truth.
2. **Singleton Recovery:** Singletons (entities with 0 matches) represent 5.58% of the data. Predicting even 1 candidate on a singleton drops its score to 0.0. The calibrated confidence gate ($94.1$) preserves 84,385 singletons (4.87%), recovering ~5% of the macro score.
3. **Street Distractor Suppression:** Street address conflicts (different building numbers or different cities) are suppressed via normalized numeric comparisons and locality gating.

---

## 4. Directory Layout

```
├── reconstruct_submission.sh           # Helper script to assemble submission
├── output/
│   ├── matching_results.tsv.gz         # Compressed leaderboard file (40 MB)
│   ├── matching_results.tsv.part_aa    # Split parts (< 50 MB each)
│   └── matching_results.tsv.part_ab
├── code/
│   └── business_entity_resolution/
│       ├── src/                        # Full Python source code
│       │   ├── metrics.py              # Macro F_0.5 implementation
│       │   ├── preprocessing.py        # Multilingual text normalization
│       │   ├── blocking.py             # Hybrid GPU Dense + Lexical blocker
│       │   ├── features.py             # 24D pairwise feature engineering
│       │   ├── model.py                # Calibrated MatchRanker
│       │   ├── rescore_submission.py   # Multi-threaded streaming rescoring engine
│       │   ├── train.py                # LightGBM training
│       │   └── infer.py                # Full inference pipeline
│       ├── models/                     # Saved model weights
│       │   ├── lgbm_matcher.joblib
│       │   └── model_config.json
│       ├── requirements.txt
│       └── README.md
└── Documentation_template.md           # Completed methodology document
```
