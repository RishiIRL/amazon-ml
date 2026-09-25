import tempfile
import os
from src.data.loader import DatasetLoader
from src.data.schema import BusinessRecord


def test_load_sources_from_tsv():
    with tempfile.TemporaryDirectory() as tmp_dir:
        s1_path = os.path.join(tmp_dir, "train_source1.tsv")
        s2_path = os.path.join(tmp_dir, "train_source2.tsv")
        s3_path = os.path.join(tmp_dir, "train_source3.tsv")
        gt_path = os.path.join(tmp_dir, "train_ground_truth.tsv")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("src1_1\tAcme Corp\t123 Main St\tUS\n")

        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("src2_1\tAcme Corp LLC\t123 Main St\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("src3_1\tAcme\tMain Street\tUS\n")

        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tsource2_matches\tsource3_matches\n")
            f.write("src1_1\tsrc2_1\tsrc3_1\n")

        loader = DatasetLoader(tmp_dir)
        s1, s2, s3 = loader.load_sources()
        gt = loader.load_ground_truth()

        assert len(s1) == 1
        assert len(s2) == 1
        assert len(s3) == 1
        assert s1[0].entity_id == "src1_1"
        assert s1[0].business_name == "Acme Corp"
        assert gt["src1_1"] == (["src2_1"], ["src3_1"])
