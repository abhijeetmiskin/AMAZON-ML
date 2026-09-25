# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

This package implements an end-to-end, reproducible Machine Learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution competition.

## System Overview
- **Objective**: Match reference businesses in Source 1 against noisy, fragmented records in Source 2 and Source 3.
- **Evaluation Metric**: Macro-averaged $F_{0.5}$ score (precision-weighted 2× over recall).
- **Core Architecture**:
  1. **Preprocessing & Standardization**: NFKD unicode normalization, script detection (Latin, Devanagari, Tamil), country-specific legal suffix stripping (US, India, France), address abbreviation expansions.
  2. **Strict Country Partitioning**: 100% of ground-truth matches exist within the same national boundary; US, India, and France partitions are isolated to prevent cross-border false merges.
  3. **Multi-View Blocking**: Inverted indexing on exact cleaned names, 2-word prefixes, sorted name tokens, address numbers, and locality tokens.
  4. **Pairwise Feature Engineering**: 14 vectorized similarity features (RapidFuzz token sort/set ratios, Levenshtein, address number overlap, length ratios, structural indicators).
  5. **LightGBM Match Ranker**: Tree-based ranking model with threshold calibration specifically tuned for Macro $F_{0.5}$ and singleton accuracy.

## Environment Setup
Python 3.10+ is required.
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Training & Threshold Calibration
Trains the LightGBM match ranker on candidate pairs extracted from the training set, sweeps decision thresholds against the official Macro $F_{0.5}$ metric on held-out validation data, and persists the model:
```bash
python3 src/train.py
```

### 2. Inference on Test Set
Runs country-partitioned multi-view candidate generation and model scoring across the test set, outputting `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python3 src/infer.py
```

### 3. Submission Validation
Verify formatting and compliance with the official validator:
```bash
python3 ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
