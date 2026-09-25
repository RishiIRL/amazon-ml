from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.data.schema import BusinessRecord


class BaseFeatureExtractor(ABC):
    """Abstract base class for extracting features from record pairs."""

    @abstractmethod
    def extract_pair_features(self, record1: BusinessRecord, record2: BusinessRecord) -> Dict[str, float]:
        """Extract a dictionary of numerical features for a pair of business records."""
        pass

    @abstractmethod
    def extract_batch_features(
        self, candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]]
    ) -> pd.DataFrame:
        """Extract features for a list of record pairs as a DataFrame."""
        pass


class PairFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor producing similarity signals between record pairs."""

    def extract_pair_features(self, record1: BusinessRecord, record2: BusinessRecord) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # 1. Name similarity placeholder
        features["name_similarity_placeholder"] = self._compute_name_similarity_placeholder(
            record1.business_name, record2.business_name
        )

        # 2. Address similarity placeholder
        features["address_similarity_placeholder"] = self._compute_address_similarity_placeholder(
            record1.business_address, record2.business_address
        )

        # 3. Country agreement
        features["country_agreement"] = float(
            record1.country.strip().lower() == record2.country.strip().lower()
            if record1.country and record2.country else 0.0
        )

        # 4. Other future features placeholder
        features["future_feature_placeholder"] = 0.0

        return features

    def extract_batch_features(
        self, candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]]
    ) -> pd.DataFrame:
        rows = [self.extract_pair_features(r1, r2) for r1, r2 in candidate_pairs]
        return pd.DataFrame(rows)

    def _compute_name_similarity_placeholder(self, name1: str, name2: str) -> float:
        """Placeholder for string/script name similarity calculations."""
        if not name1 or not name2:
            return 0.0
        return 1.0 if name1 == name2 else 0.5

    def _compute_address_similarity_placeholder(self, addr1: str, addr2: str) -> float:
        """Placeholder for address token/distance similarity calculations."""
        if not addr1 or not addr2:
            return 0.0
        return 1.0 if addr1 == addr2 else 0.5
