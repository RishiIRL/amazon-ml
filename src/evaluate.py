import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.data.loader import DatasetLoader
from src.evaluation.metrics import compute_macro_f0_5, compute_candidate_recall, compute_entity_f0_5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    candidate_pairs: Optional[list] = None,
) -> Dict[str, float]:
    """Evaluate predictions against ground truth dictionary.
    
    Args:
        predictions: Dictionary mapping s1_id to list/tuple/set of matched entity IDs.
        ground_truth: Dictionary mapping s1_id to ground truth matched entity IDs.
        candidate_pairs: Optional list of (s1_id, target_id) tuples for candidate recall evaluation.
        
    Returns:
        Dictionary with macro F0.5, precision, recall, and candidate recall.
    """
    logger.info("Computing evaluation metrics for %d predicted entities...", len(predictions))
    metrics = compute_macro_f0_5(predictions, ground_truth)
    
    if candidate_pairs is not None:
        cand_recall = compute_candidate_recall(candidate_pairs, ground_truth)
        metrics["blocking_recall"] = cand_recall

    return metrics


def evaluate_pipeline(config: Union[str, Dict[str, Any]] = "configs/config.yaml") -> Dict[str, float]:
    """Run pipeline evaluation using configuration file.
    
    Args:
        config: Path to YAML config file or config dictionary.
        
    Returns:
        Evaluation metrics dictionary.
    """
    from src.train import train
    results = train(config)
    return results["metrics"]


if __name__ == "__main__":
    evaluate_pipeline()
