import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import DataLoader, DatasetLoader
from src.data.schema import BusinessRecord
from src.preprocessing.normalize import UnicodeNormalizer
from src.preprocessing.transliteration import ScriptAwareTransliterator
from src.blocking.candidate_generator import SimpleCandidateGenerator
from src.features.pair_features import PairFeatureExtractor
from src.models.matcher import EntityMatcher
from src.pipeline.train import run_training_pipeline
from src.inference.predict import run_inference_pipeline, export_submission


class TestPipelineSynthetic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.test_dir = os.path.join(self.temp_dir, "test")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.models_dir = os.path.join(self.temp_dir, "models")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Create dummy TSV files for training
        s1_train = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                   "s1_101\tAmazon India Pvt Ltd\tOuter Ring Road Bangalore\tIN\n" \
                   "s1_102\tFlipkart Logistics\tElectronic City Bangalore\tIN\n"

        s2_train = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                   "s2_201\tAmazon India Private Limited\tOuter Ring Rd Bengaluru\tIN\n" \
                   "s2_202\tFlipkart Internet Pvt Ltd\tElectronic City Bengaluru\tIN\n"

        s3_train = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                   "s3_301\tAmazon Retail India\tOuter Ring Road Bangalore\tIN\n" \
                   "s3_302\tFlipkart India\tElectronic City Bangalore\tIN\n"

        gt_train = "source1_entity_id\tmatched_entity_ids\n" \
                   "s1_101\ts2_201,s3_301\n" \
                   "s1_102\ts2_202,s3_302\n"

        with open(os.path.join(self.train_dir, "train_source1.tsv"), "w", encoding="utf-8") as f:
            f.write(s1_train)
        with open(os.path.join(self.train_dir, "train_source2.tsv"), "w", encoding="utf-8") as f:
            f.write(s2_train)
        with open(os.path.join(self.train_dir, "train_source3.tsv"), "w", encoding="utf-8") as f:
            f.write(s3_train)
        with open(os.path.join(self.train_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
            f.write(gt_train)

        # Create dummy TSV files for testing
        s1_test = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "s1_101\tAmazon India Pvt Ltd\tOuter Ring Road Bangalore\tIN\n"

        s2_test = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "s2_201\tAmazon India Private Limited\tOuter Ring Rd Bengaluru\tIN\n"

        s3_test = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "s3_301\tAmazon Retail India\tOuter Ring Road Bangalore\tIN\n"

        with open(os.path.join(self.test_dir, "test_source1.tsv"), "w", encoding="utf-8") as f:
            f.write(s1_test)
        with open(os.path.join(self.test_dir, "test_source2.tsv"), "w", encoding="utf-8") as f:
            f.write(s2_test)
        with open(os.path.join(self.test_dir, "test_source3.tsv"), "w", encoding="utf-8") as f:
            f.write(s3_test)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_synthetic_pipeline(self):
        config = {
            "paths": {
                "train_dir": self.train_dir,
                "test_dir": self.test_dir,
                "ground_truth_path": os.path.join(self.train_dir, "train_ground_truth.tsv"),
                "output_dir": self.output_dir,
                "matching_results": os.path.join(self.output_dir, "matching_results.tsv"),
                "candidate_pairs": os.path.join(self.output_dir, "candidate_pairs.tsv"),
                "model_dir": self.models_dir,
                "model_checkpoint": os.path.join(self.models_dir, "entity_matcher.pkl"),
            },
            "preprocessing": {
                "unicode_form": "NFKC",
                "lowercase": True,
                "use_ai4bharat": False,
            },
            "blocking": {
                "max_candidates_per_entity": 10,
                "min_blocking_score": 0.05,
            },
            "model": {
                "threshold": 0.3,
            }
        }

        # 1. Run training pipeline
        results = run_training_pipeline(config)
        self.assertIn("metrics", results)
        self.assertIn("macro_f0_5", results["metrics"])
        self.assertTrue(os.path.exists(config["paths"]["model_checkpoint"]))

        # 2. Run inference pipeline
        run_inference_pipeline(config)
        self.assertTrue(os.path.exists(config["paths"]["matching_results"]))
        self.assertTrue(os.path.exists(config["paths"]["candidate_pairs"]))

        # 3. Check submission file contents
        matching_df = pd.read_csv(config["paths"]["matching_results"], sep="\t", dtype=str)
        candidate_df = pd.read_csv(config["paths"]["candidate_pairs"], sep="\t", dtype=str)

        self.assertListEqual(list(matching_df.columns), ["source1_entity_id", "matched_entity_ids"])
        self.assertListEqual(list(candidate_df.columns), ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(len(matching_df), 1)
        self.assertEqual(len(candidate_df), 1)


if __name__ == "__main__":
    unittest.main()
