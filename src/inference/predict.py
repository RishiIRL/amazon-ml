import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from src.data.loader import DatasetLoader
from src.data.schema import BusinessRecord
from src.preprocessing.normalize import UnicodeNormalizer
from src.blocking.candidate_generator import SimpleCandidateGenerator
from src.features.pair_features import PairFeatureExtractor
from src.models.matcher import EntityMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_submission(
    s1_records: Sequence[BusinessRecord],
    predictions: Dict[str, Union[Sequence[str], Tuple[List[str], List[str]], Set[str]]],
    candidate_pairs: Sequence[Tuple[str, str]],
    matching_filepath: str = "matching_results.tsv",
    candidate_filepath: str = "candidate_pairs.tsv",
) -> None:
    """Export predictions and candidate pairs to submission TSV files.

    Adheres strictly to submission formatting rules:
    - matching_results.tsv header: source1_entity_id\tmatched_entity_ids
    - candidate_pairs.tsv header: source1_entity_id\tcandidate_entity_ids
    - Includes every Source 1 entity in s1_records exactly once.
    """
    Path(matching_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(candidate_filepath).parent.mkdir(parents=True, exist_ok=True)

    # Build candidate lookup: s1_id -> set of candidate target IDs
    cand_map: Dict[str, Set[str]] = {}
    for s1_id, target_id in candidate_pairs:
        cand_map.setdefault(s1_id, set()).add(target_id)

    logger.info("Exporting matching results to %s", matching_filepath)
    with open(matching_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for rec in s1_records:
            s1_id = rec.entity_id
            val = predictions.get(s1_id, [])
            if isinstance(val, tuple) and len(val) == 2:
                s2_m, s3_m = val
                matched_ids = sorted(list(set(s2_m).union(set(s3_m))))
            else:
                matched_ids = sorted(list(set(val)))
            
            matched_str = ",".join(matched_ids) if matched_ids else ""
            writer.writerow([s1_id, matched_str])

    logger.info("Exporting candidate pairs to %s", candidate_filepath)
    with open(candidate_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for rec in s1_records:
            s1_id = rec.entity_id
            cands = sorted(list(cand_map.get(s1_id, set())))
            cand_str = ",".join(cands) if cands else ""
            writer.writerow([s1_id, cand_str])


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config path %s not found. Using default parameters.", config_path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f) or {}


def run_inference_pipeline(
    config: Union[str, Dict[str, Any]] = "configs/config.yaml"
) -> None:
    """Run complete test dataset inference pipeline using configuration."""
    if isinstance(config, str):
        cfg = load_config(config)
    else:
        cfg = config

    paths_cfg = cfg.get("paths", {})
    prep_cfg = cfg.get("preprocessing", {})
    block_cfg = cfg.get("blocking", {})
    model_cfg = cfg.get("model", {})

    test_dir = paths_cfg.get("test_dir", "student_resource/dataset/test")
    matching_filepath = paths_cfg.get("matching_results", "output/matching_results.tsv")
    candidate_filepath = paths_cfg.get("candidate_pairs", "output/candidate_pairs.tsv")
    model_checkpoint = paths_cfg.get("model_checkpoint", "models/entity_matcher.pkl")

    logger.info("Loading test datasets from %s...", test_dir)
    loader = DatasetLoader(test_dir)
    s1, s2, s3 = loader.load_sources()

    logger.info("Normalizing and transliterating test records...")
    from src.preprocessing.transliteration import ScriptAwareTransliterator
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
    logger.info("Generated %d candidate pairs for test set.", len(pairs))

    logger.info("Extracting pair features...")
    feature_extractor = PairFeatureExtractor()
    features_df = feature_extractor.extract_batch_features(pairs)

    matcher = None
    if model_checkpoint and Path(model_checkpoint).exists():
        try:
            import pickle
            with open(model_checkpoint, "rb") as f:
                matcher = pickle.load(f)
            logger.info("Loaded trained matcher from %s", model_checkpoint)
        except Exception as e:
            logger.warning("Could not load model checkpoint from %s: %s. Using default matcher.", model_checkpoint, e)

    if matcher is None:
        matcher = EntityMatcher(threshold=model_cfg.get("threshold", 0.5))

    threshold = getattr(matcher, "threshold", model_cfg.get("threshold", 0.5))
    logger.info("Predicting matches with EntityMatcher (threshold=%.4f)...", threshold)
    pair_scores = matcher.predict_pairs(features_df)
    predictions = matcher.predict_clusters(s1_norm, pairs, pair_scores, threshold=threshold)

    export_submission(
        s1_records=s1,
        predictions=predictions,
        candidate_pairs=pair_ids,
        matching_filepath=matching_filepath,
        candidate_filepath=candidate_filepath,
    )
    logger.info("Inference complete. Output written to %s and %s", matching_filepath, candidate_filepath)


if __name__ == "__main__":
    run_inference_pipeline()
