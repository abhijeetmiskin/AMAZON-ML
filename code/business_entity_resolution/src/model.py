"""
LightGBM Matching Classifier and Macro F_0.5 Threshold Calibration.
Calibrates decision thresholds specifically for the precision-heavy F_0.5 objective.
"""

from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES
from metrics import compute_macro_f05


class MatchRanker:
    """LightGBM-based pairwise matching classifier with F_0.5 calibration."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        self.optimal_threshold: float = 0.65

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train LightGBM model with early stopping."""
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

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
        thresholds: Optional[List[float]] = None,
    ) -> float:
        """Find decision threshold that maximizes Macro F_0.5 on validation split."""
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.40, 0.92, 0.02)]

        # Group candidate probabilities by S1 entity
        s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {s: [] for s in all_s1_ids}
        for (s1_id, t_id), prob in zip(candidate_pairs, probabilities):
            if s1_id in s1_to_scored_cands:
                s1_to_scored_cands[s1_id].append((t_id, float(prob)))

        best_f05 = -1.0
        best_tau = 0.65

        for tau in thresholds:
            preds: Dict[str, Set[str]] = {}
            for s1_id in all_s1_ids:
                matches = {t_id for t_id, p in s1_to_scored_cands.get(s1_id, []) if p >= tau}
                preds[s1_id] = matches

            res = compute_macro_f05(ground_truth, preds, all_s1_ids=all_s1_ids)
            f05 = res["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_tau = tau

        print(f"Optimal Threshold: {best_tau:.2f} -> Validation Macro F_0.5: {best_f05:.4f}")
        self.optimal_threshold = best_tau
        return best_tau
