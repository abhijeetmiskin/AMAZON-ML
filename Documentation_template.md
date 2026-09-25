# ML Challenge 2026: Business Entity Resolution Solution Document

**Challenge:** Amazon ML Challenge 2026  
**Problem:** Multi-Source Business Entity Resolution  
**Evaluation Metric:** Macro-Averaged F_0.5 Score  
**Validation F_0.5 Score:** 0.9598 (~0.96)  
**Validation Precision:** 98.92%  
**Validation Recall:** 92.04%  

---

## 1. Executive Summary

This solution presents an end-to-end, high-recall, precision-calibrated machine learning pipeline for large-scale multi-source Business Entity Resolution across 26.4 million records. By uniting GPU-accelerated Dense Semantic Retrieval (multilingual MiniLM on NVIDIA RTX PRO 4000 Blackwell) with invariant address digit sequence indexing and exact cleaned names, the system achieves a 93.18% candidate blocking recall while restricting candidate pool size to ~20 pairs per entity. A LightGBM pairwise ranker trained on hard negative candidate pairs with decision threshold optimization tailored specifically to the Macro $F_{0.5}$ metric achieves **0.9598 validation Macro $F_{0.5}$** with **98.92% precision** and **97.14% singleton accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis
* **Scale**: 2.2M Source 1 reference entities and 10.3M Source 2/3 target records in train; 1.73M reference entities and 10.0M targets in test.
* **Country Constraint**: Empirical analysis across all 7,638,365 ground-truth matched pairs confirmed that **0 matches cross national borders** (100% intra-country).
* **Noise Patterns**:
  * **Transliterations (India)**: Identical businesses represented in Latin, Devanagari (Hindi), Tamil, and Malayalam scripts.
  * **Invariant Address Digits**: Across 90%+ of addresses, street numbers, door numbers, shop numbers, and postal digits (`116`, `1705`, `9487203`, `6(29)`) remain strictly invariant even when locality tokens are permuted or translated.
  * **Unseen Domain (France)**: 15% of the test set originates from France, completely absent from training data. Handled via country-agnostic string processing, French legal suffixes (`SARL`, `SAS`, `EURL`, `& Fils`), and French address normalizations.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-View Blocking (Dense GPU + Invariant Digit Lexical) + GBDT Pairwise Ranker + F_0.5 Calibrated Threshold Optimizer.  
**Core Innovation:** High-throughput GPU Bi-Encoder dense semantic embedding coupled with invariant numeric address signatures, completely eliminating the need for prohibited external geocoding/translation APIs while maintaining >93% recall and 98.9% precision.

---

## 3. Candidate Generation (Blocking)

- **Blocking Channels**:
  1. *GPU Dense Semantic Retrieval*: Top-15 cosine neighbors using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` batched across 24GB GPU VRAM.
  2. *Clean Exact Name*: Normalized business name stripped of country-specific legal suffixes.
  3. *Invariant Address Digits*: Extracted digit tokens ($\ge 4$ digits standalone, or $2-3$ digits combined with significant street words).
- **Candidate pairs generated per entity:** ~20.5 on average.
- **How true matches were preserved:** The union of semantic embeddings (bridging multilingual script differences and severe name typos) and invariant numeric address keys (bridging business acronyms and name rebrandings) guarantees a **93.18% recall ceiling** on ground truth.

---

## 4. Matching Model

**Features Used (15 Dimensions):**
- Name Features: Exact clean match, RapidFuzz Token Sort Ratio, RapidFuzz Token Set Ratio, Levenshtein Ratio, Partial Ratio, Character 3-Gram Jaccard similarity, Name Length Difference, Name Length Ratio.
- Address Features: Exact clean match, Address Token Sort Ratio, Address Token Set Ratio, Address Numeric Overlap indicator, Address Null indicator.
- Structural & Source: Source prefix indicator (S2 vs S3), Heuristic composite score.

**Model Type:** LightGBM Binary Classifier / Pairwise Ranker (`num_leaves=31, max_depth=6, learning_rate=0.06`).  
**Threshold Selection Method:** Direct grid optimization against the official Macro $F_{0.5}$ metric on a held-out validation set. The optimal decision threshold was identified at $\tau^* = 0.62$, optimizing the precision-recall balance and awarding 1.0 to singletons.

---

## 5. Results & Error Analysis

- **Macro F_0.5 Score:** **0.9598**
- **Macro Precision:** **98.92%**
- **Macro Recall:** **92.04%**
- **Singleton Accuracy:** **97.14%**
- **False Positives:** Minimized to $< 1.1\%$ due to the calibrated decision cutoff ($\tau = 0.62$), preserving singleton integrity.
- **False Negatives:** Primarily confined to rare records containing both completely transformed names without shared trigrams and completely missing/null addresses.

---

## 6. Conclusion

By strictly aligning with the competition's evaluation metric, exploiting country partitioning, and leveraging modern GPU sentence embeddings alongside invariant numeric address indexing, our solution delivers state-of-the-art Entity Resolution performance without relying on any prohibited external APIs.

---

## Appendix: Code Artefacts

- `output/matching_results.tsv`: Final test matches scored on the leaderboard.
- `output/candidate_pairs.tsv`: Candidate blocking set.
- `code/business_entity_resolution/src/`:
  - `metrics.py`: Official Macro F_0.5 evaluation implementation.
  - `preprocessing.py`: Multilingual text normalizer (Latin, Devanagari, Tamil, French).
  - `blocking.py`: HybridBlocker (GPU Dense + Lexical Invariant Digits).
  - `features.py`: 15-dimensional pairwise RapidFuzz feature extractor.
  - `model.py`: LightGBM ranker with F_0.5 threshold tuner.
  - `train.py`: Training and validation runner.
  - `infer.py`: Test inference pipeline.
- `code/business_entity_resolution/requirements.txt`: Pinned dependencies.
