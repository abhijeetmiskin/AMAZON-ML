"""
Evaluation Metric: Macro-Averaged F_0.5 Score.
Matches the official Amazon ML Challenge 2026 scoring criteria exactly.

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed as a macro-average across all Source 1 entities in the evaluation set.
Singletons:
- Ground truth is empty, predicted is empty: 1.0 (True Negative)
- Ground truth is empty, predicted is non-empty: 0.0 (False Positive / False Merge)
- Ground truth is non-empty, predicted is empty: 0.0 (False Negative / Missed)
- Ground truth is non-empty, predicted has zero overlap: 0.0
"""

from typing import Dict, Iterable, Optional, Set, Tuple


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute (f05, precision, recall) for a single Source 1 entity."""
    len_true = len(true_ids)
    len_pred = len(pred_ids)

    # Edge cases involving empty sets
    if len_true == 0:
        if len_pred == 0:
            return 1.0, 1.0, 1.0  # Correct singleton identification
        return 0.0, 0.0, 1.0      # False merge on singleton

    if len_pred == 0:
        return 0.0, 1.0, 0.0      # Missed all matches

    # Non-empty sets
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_pred
    recall = tp / len_true

    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute Macro-Averaged F_0.5 across all Source 1 entities.

    Args:
        ground_truth: Mapping {s1_id: set_of_matched_ids}
        predictions: Mapping {s1_id: set_of_predicted_ids}
        all_s1_ids: Optional complete set of required S1 IDs (defaults to union of keys)

    Returns:
        Dictionary containing:
            - 'macro_f05': Macro-averaged F_0.5 score
            - 'macro_precision': Macro-averaged Precision
            - 'macro_recall': Macro-averaged Recall
            - 'singleton_acc': Accuracy on singletons
            - 'num_entities': Total S1 entities evaluated
            - 'num_singletons': Total ground-truth singletons
    """
    if all_s1_ids is None:
        eval_ids = set(ground_truth.keys()) | set(predictions.keys())
    else:
        eval_ids = set(all_s1_ids)

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    singleton_correct = 0
    total_singletons = 0
    non_singleton_count = 0

    for s1_id in eval_ids:
        t_ids = ground_truth.get(s1_id, set())
        p_ids = predictions.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(t_ids, p_ids)
        total_f05 += f05
        total_prec += prec
        total_rec += rec

        if len(t_ids) == 0:
            total_singletons += 1
            if len(p_ids) == 0:
                singleton_correct += 1
        else:
            non_singleton_count += 1

    n = len(eval_ids)
    if n == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_acc": 0.0,
            "num_entities": 0,
            "num_singletons": 0,
        }

    return {
        "macro_f05": total_f05 / n,
        "macro_precision": total_prec / n,
        "macro_recall": total_rec / n,
        "singleton_acc": singleton_correct / total_singletons if total_singletons > 0 else 0.0,
        "num_entities": n,
        "num_singletons": total_singletons,
    }
