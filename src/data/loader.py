from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from src.data.schema import BusinessRecord


class DataLoader:
    """Utility for loading entity resolution datasets from tab-separated (.tsv) files."""

    @staticmethod
    def load_tsv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a TSV file with explicit tab separation.

        Args:
            file_path: Absolute or relative path to TSV file.

        Returns:
            DataFrame with loaded records.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Explicit tab separator is required for TSV files
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
        )
        return df

    @classmethod
    def load_source(cls, file_path: Union[str, Path], source_name: Optional[str] = None) -> pd.DataFrame:
        """Load a source entity TSV file (Source 1, Source 2, or Source 3).

        Args:
            file_path: Path to source TSV.
            source_name: Optional identifier for source name.

        Returns:
            DataFrame containing entity records.
        """
        df = cls.load_tsv(file_path)
        required_cols = {"entity_id", "business_name", "business_address", "country"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"File {file_path} is missing required columns: {missing_cols}")

        if source_name:
            df["source"] = source_name
        else:
            df["source"] = df["entity_id"].apply(
                lambda eid: "source1" if str(eid).startswith("S1-")
                else ("source2" if str(eid).startswith("S2-")
                      else ("source3" if str(eid).startswith("S3-") else "unknown"))
            )
        return df

    @classmethod
    def load_ground_truth(cls, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load ground truth matching labels TSV file.

        Args:
            file_path: Path to train_ground_truth.tsv.

        Returns:
            DataFrame with ground truth columns.
        """
        df = cls.load_tsv(file_path)
        return df

    @classmethod
    def load_all_sources(cls, data_dir: Union[str, Path], split: str = "train") -> Dict[str, pd.DataFrame]:
        """Load all three source files for a split independently.

        Args:
            data_dir: Base directory containing dataset subfolders.
            split: Split name ('train' or 'test').

        Returns:
            Dictionary mapping 'source1', 'source2', 'source3' to their DataFrames.
        """
        dir_path = Path(data_dir) / split
        return {
            "source1": cls.load_source(dir_path / f"{split}_source1.tsv", source_name="source1"),
            "source2": cls.load_source(dir_path / f"{split}_source2.tsv", source_name="source2"),
            "source3": cls.load_source(dir_path / f"{split}_source3.tsv", source_name="source3"),
        }


class DatasetLoader:
    """High-level dataset loader converting TSVs into BusinessRecord objects and GT dicts."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def _find_file(self, prefix: str) -> Path:
        """Find TSV file matching prefix in data_dir."""
        for file in self.data_dir.glob("*.tsv"):
            if file.name.startswith(prefix) or prefix in file.name:
                return file
        matches = list(self.data_dir.glob(f"*{prefix}*.tsv"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No TSV file matching prefix '{prefix}' found in {self.data_dir}")

    def load_source_records(self, prefix: str, source_name: str) -> List[BusinessRecord]:
        """Load records for a single source file into BusinessRecord objects."""
        file_path = self._find_file(prefix)
        df = DataLoader.load_source(file_path, source_name=source_name)
        records = []
        for _, row in df.iterrows():
            records.append(
                BusinessRecord(
                    entity_id=str(row.get("entity_id", "")),
                    business_name=str(row.get("business_name", "")),
                    business_address=str(row.get("business_address", "")),
                    country=str(row.get("country", "")),
                    source=source_name,
                )
            )
        return records

    def load_sources(self) -> Tuple[List[BusinessRecord], List[BusinessRecord], List[BusinessRecord]]:
        """Load Source 1, Source 2, and Source 3 records."""
        s1 = self.load_source_records("source1", "source1")
        s2 = self.load_source_records("source2", "source2")
        s3 = self.load_source_records("source3", "source3")
        return s1, s2, s3

    def load_ground_truth(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Load ground truth mapping entity_id -> (source2_matches, source3_matches)."""
        gt_path = self._find_file("ground_truth")
        df = DataLoader.load_tsv(gt_path)

        mapping: Dict[str, Tuple[List[str], List[str]]] = {}
        for _, row in df.iterrows():
            s1_id = str(row.get("source1_entity_id", row.get("entity_id", "")))
            
            # Check for combined matched_entity_ids column (from train_ground_truth.tsv)
            if "matched_entity_ids" in row:
                raw_matches = str(row.get("matched_entity_ids", ""))
                all_matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
                s2_matches = [m for m in all_matches if m.startswith("S2-")]
                s3_matches = [m for m in all_matches if m.startswith("S3-")]
            else:
                s2_raw = str(row.get("source2_matches", ""))
                s3_raw = str(row.get("source3_matches", ""))
                s2_matches = [m.strip() for m in s2_raw.split(",") if m.strip()]
                s3_matches = [m.strip() for m in s3_raw.split(",") if m.strip()]

            mapping[s1_id] = (s2_matches, s3_matches)

        return mapping

