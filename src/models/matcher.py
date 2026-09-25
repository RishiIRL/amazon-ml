from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.data.schema import BusinessRecord


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """Compute Macro F_0.5 score across all Source 1 entities.
    
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singleton entities with no true matches score 1.0 if prediction is empty, 0.0 otherwise.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not true_set and pred_set:
            scores.append(0.0)
            continue
        if true_set and not pred_set:
            scores.append(0.0)
            continue

        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


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
    """ML Entity Matcher with classifier model and Macro F_0.5 threshold optimization."""

    def __init__(self, threshold: float = 0.5, classifier: Optional[Any] = None) -> None:
        self.threshold = threshold
        self.classifier = classifier or HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, random_state=42
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: List[int]) -> "EntityMatcher":
        """Fit ML classifier model on pair features and binary match labels."""
        if X.empty or len(y) == 0:
            self.is_fitted = True
            return self

        # Fill any missing values with 0.0
        X_clean = X.fillna(0.0)
        self.classifier.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict_pairs(self, X: pd.DataFrame) -> List[float]:
        """Predict pair match probabilities."""
        if X.empty:
            return []
        if not self.is_fitted:
            # Fallback simple heuristic average of string similarity features
            sim_cols = [c for c in X.columns if "ratio" in c or "sim" in c or "jaccard" in c]
            if sim_cols:
                return X[sim_cols].mean(axis=1).tolist()
            return [0.0] * len(X)

        X_clean = X.fillna(0.0)
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_clean)[:, 1]
            return probs.tolist()
        else:
            preds = self.classifier.predict(X_clean)
            return [float(p) for p in preds]

    def optimize_threshold(
        self,
        source1_records: List[BusinessRecord],
        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]],
        pair_scores: List[float],
        ground_truth: Dict[str, Set[str]],
    ) -> float:
        """Find decision threshold that maximizes Macro F_0.5 metric."""
        best_threshold = self.threshold
        best_score = -1.0

        thresholds = np.linspace(0.1, 0.95, 35)
        for t in thresholds:
            pred_clusters = self.predict_clusters_as_sets(
                source1_records, candidate_pairs, pair_scores, threshold=float(t)
            )
            score = compute_macro_f05(ground_truth, pred_clusters)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        self.threshold = best_threshold
        return best_threshold

    def predict_clusters_as_sets(
        self,
        source1_records: List[BusinessRecord],
        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]],
        pair_scores: List[float],
        threshold: Optional[float] = None,
    ) -> Dict[str, Set[str]]:
        """Map predictions into a dictionary of {s1_entity_id: set(matched_entity_ids)}."""
        thresh = threshold if threshold is not None else self.threshold
        results: Dict[str, Set[str]] = {r.entity_id: set() for r in source1_records}

        for (r1, r2), score in zip(candidate_pairs, pair_scores):
            if score >= thresh:
                s1_id = r1.entity_id
                target_id = r2.entity_id
                if s1_id in results:
                    results[s1_id].add(target_id)

        return results

    def predict_clusters(
        self,
        source1_records: List[BusinessRecord],
        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]],
        pair_scores: List[float],
        threshold: Optional[float] = None,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Predict cluster matches formatted as {s1_id: (s2_matches_list, s3_matches_list)}."""
        thresh = threshold if threshold is not None else self.threshold
        results: Dict[str, Tuple[List[str], List[str]]] = {
            r.entity_id: ([], []) for r in source1_records
        }

        for (r1, r2), score in zip(candidate_pairs, pair_scores):
            if score >= thresh:
                s1_id = r1.entity_id
                target_id = r2.entity_id
                if s1_id in results:
                    s2_list, s3_list = results[s1_id]
                    if target_id.startswith("S2-") and target_id not in s2_list:
                        s2_list.append(target_id)
                    elif target_id.startswith("S3-") and target_id not in s3_list:
                        s3_list.append(target_id)

        return results

