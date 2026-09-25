# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

This repository contains the complete, reproducible solution for the **Amazon ML Challenge 2026 Business Entity Resolution**.

## Key Achievements & Validation Performance
- **Validation Macro F_0.5 Score:** **0.9598 (~0.96)**
- **Validation Precision:** **98.92%**
- **Validation Recall:** **92.04%**
- **Singleton Accuracy:** **97.14%**
- **Official Validator:** `PASS — no blocking issues found. Safe to submit.` (verified across all 1,732,544 reference entities and 9,969,589 target IDs with zero errors/warnings).
- **Zero Hardcoding**: 100% algorithmic and model-driven.

---

## 1. Getting `matching_results.tsv` (Leaderboard File)

Because GitHub enforces a strict 100 MB per-file upload limit, the 197 MB `matching_results.tsv` file is provided in two ready-to-use formats inside `output/`:

### Option A: Reassemble from Split Parts (Fastest, No Tools Needed)
```bash
cat output/matching_results.tsv.part_* > output/matching_results.tsv
```

### Option B: Decompress from `.gz`
```bash
gzip -d -k output/matching_results.tsv.gz
```

Both options produce the exact 197 MB `matching_results.tsv` containing all 1,732,544 rows.

---

## 2. Solution Architecture

1. **Multilingual Text Preprocessing**: Normalizes Latin, Devanagari (Hindi), Tamil, and French company strings, removing legal suffixes (`Inc`, `Pvt Ltd`, `प्राइवेट लिमिटेड`, `எல்எல்பி`, `SARL`, `SASU`, `& Fils`).
2. **Zero Cross-Border Leakage**: Strictly partitions comparisons by country (`US`, `India`, `France`), eliminating cross-border false merges.
3. **Hybrid High-Recall Blocking**: Combines GPU Dense Semantic Retrieval (multilingual MiniLM bi-encoder on RTX PRO 4000 Blackwell) with invariant numeric address signatures (`116`, `1705`, `9487203`, `6(29)`) and clean exact names (93.18% ground-truth recall).
4. **15-Dimensional Pairwise Feature Engineering**: Computes RapidFuzz token sort/set ratios, character 3-gram Jaccard, address number overlap, length differences, and structural signals.
5. **LightGBM Match Ranker**: Tree-based ranker with optimal threshold calibration ($\tau = 0.62$) tailored to the precision-heavy Macro $F_{0.5}$ metric.

---

## 3. Directory Layout

```
├── output/
│   ├── matching_results.tsv.gz         # Compressed leaderboard file (83 MB)
│   ├── matching_results.tsv.part_aa    # Split parts (< 100 MB each)
│   ├── matching_results.tsv.part_ab
│   └── matching_results.tsv.part_ac
├── code/
│   └── business_entity_resolution/
│       ├── src/                        # Full Python source code
│       │   ├── metrics.py              # Macro F_0.5 implementation
│       │   ├── preprocessing.py        # Multilingual cleaning
│       │   ├── blocking.py             # HybridBlocker (GPU Dense + Lexical)
│       │   ├── features.py             # 15D RapidFuzz feature extraction
│       │   ├── model.py                # LightGBM ranker
│       │   ├── train.py                # Training & threshold calibration
│       │   └── infer.py                # Test set inference
│       ├── models/                     # Saved model artifacts
│       │   ├── lgbm_matcher.joblib
│       │   └── model_config.json
│       ├── requirements.txt            # Pinned dependencies
│       └── README.md                   # Replication guide
└── Documentation_template.md           # Completed methodology document
```
