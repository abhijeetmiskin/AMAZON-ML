"""
Calibrated LightGBM Matching Classifier and Macro F_0.5 Post-Processing.
Supports precision-optimized thresholding, cardinality capping (max_k),
margin pruning, and dedicated singleton confidence gating.
"""

from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES
from metrics import compute_macro_f05


class MatchRanker:
    """LightGBM-based pairwise matching classifier with calibrated Macro F_0.5 post-processing."""

    def __init__(self, n_estimators: int = 350, learning_rate: float = 0.05, max_depth: int = 6):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=8,
            verbosity=-1,
        )
        self.optimal_threshold: float = 0.75
        self.max_k: int = 5
        self.score_margin: float = 0.35
        self.singleton_gate: float = 0.75

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train LightGBM model."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        candidate_pairs: List[Tuple[str, str]],
        probabilities: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: List[str],
    ) -> Tuple[float, int, float, float]:
        """Find decision threshold, max_k, margin, and singleton gate that maximize Macro F_0.5."""
        s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {s: [] for s in all_s1_ids}
        for (s1_id, t_id), prob in zip(candidate_pairs, probabilities):
            if s1_id in s1_to_scored_cands:
                s1_to_scored_cands[s1_id].append((t_id, float(prob)))

        for sid in all_s1_ids:
            s1_to_scored_cands[sid].sort(key=lambda x: -x[1])

        best_f05 = -1.0
        best_cfg = (0.75, 5, 0.35, 0.75)

        for tau in [0.70, 0.75, 0.80, 0.83, 0.86, 0.89]:
            for max_k in [3, 4, 5, 6]:
                for margin in [0.25, 0.35, 0.50]:
                    for s_gate in [0.0, 0.75, 0.80, 0.85]:
                        preds: Dict[str, Set[str]] = {}
                        for s1_id in all_s1_ids:
                            cands = s1_to_scored_cands.get(s1_id, [])
                            if not cands:
                                preds[s1_id] = set()
                                continue

                            top_p = cands[0][1]
                            if s_gate > 0 and top_p < s_gate:
                                preds[s1_id] = set()
                                continue

                            selected = []
                            for t_id, p in cands:
                                if p >= tau and (top_p - p) <= margin:
                                    selected.append(t_id)
                                if len(selected) >= max_k:
                                    break
                            preds[s1_id] = set(selected)

                        res = compute_macro_f05(ground_truth, preds, all_s1_ids=all_s1_ids)
                        score = res["macro_f05"]
                        if score > best_f05:
                            best_f05 = score
                            best_cfg = (tau, max_k, margin, s_gate)

        tau, k, margin, gate = best_cfg
        print(f"Optimal Policy: tau={tau}, max_k={k}, margin={margin}, s_gate={gate} -> Macro F_0.5: {best_f05:.4f}")
        self.optimal_threshold = tau
        self.max_k = k
        self.score_margin = margin
        self.singleton_gate = gate
        return best_cfg
