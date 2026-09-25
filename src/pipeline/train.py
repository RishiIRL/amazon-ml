import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.data.loader import DatasetLoader
from src.preprocessing.normalize import UnicodeNormalizer
from src.preprocessing.transliteration import ScriptAwareTransliterator
from src.blocking.candidate_generator import SimpleCandidateGenerator
from src.features.pair_features import PairFeatureExtractor
from src.models.matcher import EntityMatcher
from src.evaluation.metrics import compute_macro_f0_5, compute_candidate_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config path %s not found. Using default parameters.", config_path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_training_pipeline(config: Union[str, Dict[str, Any]] = "configs/config.yaml") -> Dict[str, Any]:
    """Run full entity resolution training pipeline using configuration.
    
    Args:
        config: Path to YAML config file or config dictionary.
        
    Returns:
        Dictionary containing evaluation metrics and pipeline execution statistics.
    """
    if isinstance(config, str):
        cfg = load_config(config)
    else:
        cfg = config

    paths_cfg = cfg.get("paths", {})
    prep_cfg = cfg.get("preprocessing", {})
    block_cfg = cfg.get("blocking", {})
    model_cfg = cfg.get("model", {})

    train_dir = paths_cfg.get("train_dir", "student_resource/dataset/train")
    model_checkpoint = paths_cfg.get("model_checkpoint", "models/entity_matcher.pkl")

    logger.info("Loading training data from %s...", train_dir)
    loader = DatasetLoader(train_dir)
    s1, s2, s3 = loader.load_sources()
    ground_truth = loader.load_ground_truth()
    logger.info("Loaded %d s1, %d s2, %d s3, and %d ground truth records", len(s1), len(s2), len(s3), len(ground_truth))

    logger.info("Normalizing and transliterating records...")
    normalizer = UnicodeNormalizer(
        form=prep_cfg.get("unicode_form", "NFKC"),
        lowercase=prep_cfg.get("lowercase", True),
        strip_whitespace=True,
    )
    transliterator = ScriptAwareTransliterator(
        target_scheme=prep_cfg.get("target_scheme", "ITRANS"),
        use_ai4bharat=prep_cfg.get("use_ai4bharat", True),
    )

    s1_norm = [transliterator.transliterate_record(r) for r in normalizer.normalize_dataset(s1)]
    s2_norm = [transliterator.transliterate_record(r) for r in normalizer.normalize_dataset(s2)]
    s3_norm = [transliterator.transliterate_record(r) for r in normalizer.normalize_dataset(s3)]

    logger.info("Generating candidate pairs...")
    candidate_gen = SimpleCandidateGenerator(
        max_candidates_per_entity=block_cfg.get("max_candidates_per_entity", 100)
    )
    pairs = candidate_gen.generate_candidates(s1_norm, s2_norm, s3_norm)
    pair_ids = [(r1.entity_id, r2.entity_id) for r1, r2 in pairs]
    logger.info("Generated %d candidate pairs", len(pairs))

    blocking_recall = compute_candidate_recall(pair_ids, ground_truth)
    logger.info("Candidate generation (blocking) recall: %.4f", blocking_recall)

    logger.info("Extracting pair features...")
    feature_extractor = PairFeatureExtractor()
    features_df = feature_extractor.extract_batch_features(pairs)

    logger.info("Fitting entity matcher model...")
    matcher = EntityMatcher(threshold=model_cfg.get("threshold", 0.5))
    
    # Create ground truth binary labels for pair candidate dataset
    gt_pairs_set = set()
    for s1_id, matched in ground_truth.items():
        if isinstance(matched, tuple) and len(matched) == 2:
            s2_m, s3_m = matched
            for m in list(s2_m) + list(s3_m):
                gt_pairs_set.add((s1_id, m))
        elif isinstance(matched, (list, set)):
            for m in matched:
                gt_pairs_set.add((s1_id, m))

    labels = [1 if pair in gt_pairs_set else 0 for pair in pair_ids]
    matcher.fit(features_df, labels)

    logger.info("Predicting match pairs and optimizing decision threshold for Macro F_0.5...")
    pair_scores = matcher.predict_pairs(features_df)
    
    # Format ground truth for threshold tuning
    gt_dict: Dict[str, Set[str]] = {}
    for s1_id, matched in ground_truth.items():
        if isinstance(matched, tuple) and len(matched) == 2:
            s2_m, s3_m = matched
            gt_dict[s1_id] = set(s2_m).union(set(s3_m))
        elif isinstance(matched, (list, set)):
            gt_dict[s1_id] = set(matched)

    best_thresh = matcher.optimize_threshold(s1_norm, pairs, pair_scores, gt_dict)
    logger.info("Optimized Macro F_0.5 threshold: %.4f", best_thresh)

    predictions = matcher.predict_clusters(s1_norm, pairs, pair_scores, threshold=best_thresh)

    logger.info("Evaluating predictions against ground truth...")
    eval_predictions_set = matcher.predict_clusters_as_sets(s1_norm, pairs, pair_scores, threshold=best_thresh)
    eval_res = compute_macro_f0_5(eval_predictions_set, gt_dict)
    eval_metrics = {
        "macro_f0_5": eval_res.get("macro_f0_5", 0.0),
        "macro_precision": eval_res.get("macro_precision", 0.0),
        "macro_recall": eval_res.get("macro_recall", 0.0),
        "blocking_recall": blocking_recall,
        "best_threshold": best_thresh
    }

    logger.info("Training pipeline metrics: %s", eval_metrics)

    if model_checkpoint:
        checkpoint_path = Path(model_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(matcher, f)
        logger.info("Saved trained entity matcher to %s", checkpoint_path)

    return {
        "metrics": eval_metrics,
        "num_candidates": len(pairs),
        "num_predictions": len(predictions),
    }


if __name__ == "__main__":
    run_training_pipeline()


if __name__ == "__main__":
    run_training_pipeline()
