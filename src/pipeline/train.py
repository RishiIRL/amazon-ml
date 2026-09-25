import logging
from typing import Dict, Any

from src.data.loader import DatasetLoader
from src.preprocessing.normalize import UnicodeNormalizer
from src.blocking.candidate_generator import SimpleCandidateGenerator
from src.features.pair_features import PairFeatureExtractor
from src.models.matcher import EntityMatcher
from src.evaluation.metrics import compute_macro_f0_5, compute_candidate_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(dataset_dir: str = "student_resource/dataset/train") -> Dict[str, float]:
    """Run baseline training and evaluation pipeline skeleton."""
    logger.info("Loading training data from %s", dataset_dir)
    loader = DatasetLoader(dataset_dir)
    s1, s2, s3 = loader.load_sources()
    ground_truth = loader.load_ground_truth()

    logger.info("Loaded %d s1, %d s2, %d s3, and %d ground truth records", len(s1), len(s2), len(s3), len(ground_truth))

    logger.info("Normalizing text fields...")
    normalizer = UnicodeNormalizer()
    s1 = normalizer.normalize_dataset(s1)
    s2 = normalizer.normalize_dataset(s2)
    s3 = normalizer.normalize_dataset(s3)

    logger.info("Generating candidate pairs...")
    candidate_gen = SimpleCandidateGenerator()
    pairs = candidate_gen.generate_candidates(s1, s2, s3)
    logger.info("Generated %d candidate pairs", len(pairs))

    # Calculate blocking recall
    pair_ids = [(r1.entity_id, r2.entity_id) for r1, r2 in pairs]
    blocking_recall = compute_candidate_recall(pair_ids, ground_truth)
    logger.info("Candidate generation recall: %.4f", blocking_recall)

    logger.info("Extracting pair features...")
    feature_extractor = PairFeatureExtractor()
    features_df = feature_extractor.extract_batch_features(pairs)

    logger.info("Fitting entity matcher...")
    matcher = EntityMatcher()
    dummy_labels = [0] * len(features_df)
    matcher.fit(features_df, dummy_labels)

    logger.info("Predicting match pairs and clusters...")
    pair_scores = matcher.predict_pairs(features_df)
    predictions = matcher.predict_clusters(s1, pairs, pair_scores, threshold=0.5)

    logger.info("Evaluating Macro F0.5...")
    eval_metrics = compute_macro_f0_5(predictions, ground_truth)
    eval_metrics["blocking_recall"] = blocking_recall

    logger.info("Training pipeline metrics: %s", eval_metrics)
    return eval_metrics


if __name__ == "__main__":
    run_training_pipeline()
