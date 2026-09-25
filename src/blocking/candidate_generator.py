from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
from src.data.schema import BusinessRecord


class BaseCandidateGenerator(ABC):
    """Abstract base class for candidate generators (blocking strategies)."""

    @abstractmethod
    def generate_candidates(
        self,
        source1: Union[pd.DataFrame, List[BusinessRecord]],
        *target_sources: Union[pd.DataFrame, List[BusinessRecord], List[Union[pd.DataFrame, List[BusinessRecord]]]]
    ) -> Any:
        """Generate candidate matching pairs.

        Returns:
            Dictionary or list of candidate matching pairs.
        """
        pass


class SimpleCandidateGenerator(BaseCandidateGenerator):
    """Simple initial candidate generator interface stub.
    
    Supports basic blocking strategies (e.g. country-matching block or prefix block)
    to generate candidate pairs for downstream scoring.
    """

    def __init__(self, block_by_country: bool = True, max_candidates_per_s1: Optional[int] = 100) -> None:
        self.block_by_country = block_by_country
        self.max_candidates_per_s1 = max_candidates_per_s1

    def generate_candidates(
        self,
        source1: Union[pd.DataFrame, List[BusinessRecord]],
        *target_sources: Any
    ) -> Any:
        """Generate candidate pairs stub for each Source 1 entity."""
        # Normalize input target sources
        targets: List[Union[pd.DataFrame, List[BusinessRecord]]] = []
        for ts in target_sources:
            if isinstance(ts, list) and len(ts) > 0 and isinstance(ts[0], list):
                targets.extend(ts)
            else:
                targets.append(ts)

        # Helper to extract records/dict
        if isinstance(source1, list):
            s1_records = source1
        elif isinstance(source1, pd.DataFrame):
            s1_records = [
                BusinessRecord(
                    entity_id=str(row["entity_id"]),
                    business_name=str(row.get("business_name", "")),
                    business_address=str(row.get("business_address", "")),
                    country=str(row.get("country", "")),
                    source="source1",
                )
                for _, row in source1.iterrows()
            ]
        else:
            s1_records = []

        target_records: List[BusinessRecord] = []
        for target in targets:
            if isinstance(target, list):
                target_records.extend(target)
            elif isinstance(target, pd.DataFrame):
                for _, row in target.iterrows():
                    target_records.append(
                        BusinessRecord(
                            entity_id=str(row["entity_id"]),
                            business_name=str(row.get("business_name", "")),
                            business_address=str(row.get("business_address", "")),
                            country=str(row.get("country", "")),
                            source="target",
                        )
                    )

        pairs: List[Tuple[BusinessRecord, BusinessRecord]] = []
        for r1 in s1_records:
            # Candidate matching stub: returns candidate pairs (r1, r2)
            for r2 in target_records[: self.max_candidates_per_s1 if self.max_candidates_per_s1 else len(target_records)]:
                pairs.append((r1, r2))
                break  # Stub returns 1 candidate per s1 for high speed baseline

        return pairs
