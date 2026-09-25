import tempfile
import os
from src.data.schema import BusinessRecord
from src.inference.predict import export_submission
from src.evaluation.metrics import compute_macro_f0_5, compute_entity_f0_5


def test_export_submission_format():
    s1_records = [
        BusinessRecord(entity_id="src1_1", business_name="A", business_address="1", country="US", source="source1"),
        BusinessRecord(entity_id="src1_2", business_name="B", business_address="2", country="US", source="source1"),
        BusinessRecord(entity_id="src1_3", business_name="C", business_address="3", country="US", source="source1"),
    ]
    predictions = {
        "src1_1": (["src2_10", "src2_11"], ["src3_20"]),
        "src1_2": ([], []),
        "src1_3": (["src2_30"], []),
    }
    candidate_pairs = [("src1_1", "src2_10"), ("src1_1", "src3_20"), ("src1_3", "src2_30")]

    with tempfile.TemporaryDirectory() as tmp_dir:
        matching_file = os.path.join(tmp_dir, "matching_results.tsv")
        candidate_file = os.path.join(tmp_dir, "candidate_pairs.tsv")
        export_submission(
            s1_records=s1_records,
            predictions=predictions,
            candidate_pairs=candidate_pairs,
            matching_filepath=matching_file,
            candidate_filepath=candidate_file,
        )

        with open(matching_file, "r", encoding="utf-8") as f:
            matching_lines = [line.rstrip("\r\n") for line in f.readlines()]

        assert matching_lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert matching_lines[1] == "src1_1\tsrc2_10,src2_11,src3_20"
        assert matching_lines[2] == "src1_2\t"
        assert matching_lines[3] == "src1_3\tsrc2_30"

        with open(candidate_file, "r", encoding="utf-8") as f:
            cand_lines = [line.rstrip("\r\n") for line in f.readlines()]

        assert cand_lines[0] == "source1_entity_id\tcandidate_entity_ids"
        assert cand_lines[1] == "src1_1\tsrc2_10,src3_20"
        assert cand_lines[2] == "src1_2\t"
        assert cand_lines[3] == "src1_3\tsrc2_30"



def test_macro_f0_5_metric_singleton():
    # Singleton case: both GT and pred are empty -> F0.5 = 1.0
    p, r, f = compute_entity_f0_5(set(), set())
    assert f == 1.0

    # Match case
    p, r, f = compute_entity_f0_5({"s2_1"}, {"s2_1"})
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0

    # Macro average check
    preds = {
        "e1": (["s2_1"], []),
        "e2": ([], []),
    }
    gt = {
        "e1": (["s2_1"], []),
        "e2": ([], []),
    }
    metrics = compute_macro_f0_5(preds, gt)
    assert metrics["macro_f0_5"] == 1.0
