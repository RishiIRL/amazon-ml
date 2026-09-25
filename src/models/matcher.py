from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.data.schema import BusinessRecord


class BaseMatcher(ABC):
    """Abstract base interface for pair classification and entity clustering matchers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: List[int]) -> "BaseMatcher":
        """Train the matcher model on pair features X and binary target y."""
        pass

    @abstractmethod
    def predict_pairs(self, X: pd.DataFrame) -> List[float]:
        """Predict match probabilities or confidence scores for pair features X."""
        pass

    @abstractmethod
    def predict_clusters(
        self,
        source1_records: List[BusinessRecord],
        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]],
        pair_scores: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Group candidate pairs into entity clusters and assign matches to Source 1 entities."""
        pass


class EntityMatcher(BaseMatcher):
    """Placeholder Entity Matcher for baseline pair classification and clustering."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: List[int]) -> "EntityMatcher":
        """Dummy fit method to be replaced by ML classifier (e.g., LightGBM / XGBoost)."""
        self.is_fitted = True
        return self

    def predict_pairs(self, X: pd.DataFrame) -> List[float]:
        """Dummy prediction returning placeholder similarity scores or 0.0s."""
        if X.empty:
            return []
        if "name_similarity_placeholder" in X.columns:
            return X["name_similarity_placeholder"].tolist()
        return [0.0] * len(X)

    def predict_clusters(
        self,
        source1_records: List[BusinessRecord],
        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]],
        pair_scores: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Predict cluster matches for Source 1 records based on thresholded pair scores."""
        thresh = threshold or self.threshold
        results: Dict[str, Tuple[List[str], List[str]]] = {}

        # Initialize every Source 1 record with empty matches (or non-matched baseline)
        for s1_rec in source1_records:
            results[s1_rec.entity_id] = ([], [])

        # Assign matches exceeding threshold
        for (r1, r2), score in zip(candidate_pairs, pair_scores):
            if score >= thresh:
                s1_id = r1.entity_id if r1.source == "source1" else (r2.entity_id if r2.source == "source1" else None)
                other_rec = r2 if r1.source == "source1" else r1
                if s1_id and s1_id in results:
                    s2_list, s3_list = results[s1_id]
                    if other_rec.source == "source2" and other_rec.entity_id not in s2_list:
                        s2_list.append(other_rec.entity_id)
                    elif other_rec.source == "source3" and other_rec.entity_id not in s3_list:
                        s3_list.append(other_rec.entity_id)

        return results
