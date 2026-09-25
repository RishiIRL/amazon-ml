from typing import Any, Dict, List, Set, Tuple
import numpy as np


def compute_entity_f0_5(
    predicted_matches: Set[str], ground_truth_matches: Set[str]
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F0.5 for a single Source 1 entity.

    Handles singletons (where ground truth is empty):
    - If both predicted and ground truth are empty: P = 1.0, R = 1.0, F0.5 = 1.0.
    - If ground truth is empty but predicted is non-empty: P = 0.0, R = 0.0, F0.5 = 0.0.
    - If ground truth is non-empty but predicted is empty: P = 0.0, R = 0.0, F0.5 = 0.0.
    """
    if len(ground_truth_matches) == 0 and len(predicted_matches) == 0:
        return 1.0, 1.0, 1.0

    if len(predicted_matches) == 0 or len(ground_truth_matches) == 0:
        return 0.0, 0.0, 0.0

    tp = len(predicted_matches.intersection(ground_truth_matches))
    precision = tp / len(predicted_matches)
    recall = tp / len(ground_truth_matches)

    if precision + recall == 0:
        f0_5 = 0.0
    else:
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    return precision, recall, f0_5


def _to_id_set(matches) -> Set[str]:
    """Helper to convert predictions/ground_truth entries into a set of entity IDs."""
    if isinstance(matches, tuple) and len(matches) == 2:
        s2, s3 = matches
        return set(s2).union(set(s3))
    elif isinstance(matches, (list, set, tuple)):
        return set(matches)
    return set()


def compute_macro_f0_5(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """Compute Macro F0.5, Macro Precision, and Macro Recall across all Source 1 entities."""
    all_s1_ids = set(predictions.keys()).union(set(ground_truth.keys()))
    if not all_s1_ids:
        return {"macro_f0_5": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    precisions: List[float] = []
    recalls: List[float] = []
    f0_5_scores: List[float] = []

    for s1_id in all_s1_ids:
        pred_set = _to_id_set(predictions.get(s1_id, []))
        gt_set = _to_id_set(ground_truth.get(s1_id, []))

        p, r, f = compute_entity_f0_5(pred_set, gt_set)
        precisions.append(p)
        recalls.append(r)
        f0_5_scores.append(f)

    return {
        "macro_f0_5": float(np.mean(f0_5_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
    }


def compute_candidate_recall(
    candidate_pairs: List[Tuple[str, str]],
    ground_truth: Dict[str, Any],
) -> float:
    """Compute candidate generation (blocking) recall against ground truth pairs."""
    gt_pairs: Set[Tuple[str, str]] = set()
    for s1_id, val in ground_truth.items():
        matched_set = _to_id_set(val)
        for m_id in matched_set:
            if m_id:
                gt_pairs.add((s1_id, m_id))

    if not gt_pairs:
        return 1.0

    cand_set = set(candidate_pairs)
    retrieved = len(gt_pairs.intersection(cand_set))
    return float(retrieved / len(gt_pairs))
