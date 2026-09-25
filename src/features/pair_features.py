import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Tuple
import Levenshtein
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


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _trigram_jaccard(s1: str, s2: str) -> float:
    t1 = set(f" #{s1.lower()}# "[i : i + 3] for i in range(max(1, len(s1) - 2))) if s1 else set()
    t2 = set(f" #{s2.lower()}# "[i : i + 3] for i in range(max(1, len(s2) - 2))) if s2 else set()
    if not t1 or not t2:
        return 0.0
    u = t1.union(t2)
    return len(t1.intersection(t2)) / len(u) if u else 0.0


def _token_sort_ratio(s1: str, s2: str) -> float:
    t1 = " ".join(sorted(_tokenize(s1)))
    t2 = " ".join(sorted(_tokenize(s2)))
    if not t1 or not t2:
        return 0.0
    return Levenshtein.ratio(t1, t2)


def _token_set_ratio(s1: str, s2: str) -> float:
    toks1 = set(_tokenize(s1))
    toks2 = set(_tokenize(s2))
    if not toks1 or not toks2:
        return 0.0
    intersection = toks1.intersection(toks2)
    sorted_inter = " ".join(sorted(intersection))
    sorted_t1 = " ".join(sorted(toks1))
    sorted_t2 = " ".join(sorted(toks2))

    t1_inter = f"{sorted_inter} {sorted_t1}".strip()
    t2_inter = f"{sorted_inter} {sorted_t2}".strip()

    r1 = Levenshtein.ratio(sorted_inter, sorted_t1)
    r2 = Levenshtein.ratio(sorted_inter, sorted_t2)
    r3 = Levenshtein.ratio(t1_inter, t2_inter)
    return max(r1, r2, r3)


def _digit_overlap(s1: str, s2: str) -> Tuple[float, float]:
    d1 = re.findall(r"\d+", s1 or "")
    d2 = re.findall(r"\d+", s2 or "")
    if not d1 or not d2:
        return (0.0, 0.0)
    set1, set2 = set(d1), set(d2)
    inter = set1.intersection(set2)
    jaccard = len(inter) / len(set1.union(set2)) if set1.union(set2) else 0.0
    exact = float(set1 == set2)
    return (jaccard, exact)


class PairFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor producing similarity signals between record pairs."""

    def extract_pair_features(self, record1: BusinessRecord, record2: BusinessRecord) -> Dict[str, float]:
        features: Dict[str, float] = {}

        n1, n2 = record1.business_name or "", record2.business_name or ""
        a1, a2 = record1.business_address or "", record2.business_address or ""
        c1, c2 = (record1.country or "").strip().lower(), (record2.country or "").strip().lower()

        # 1. Name Features
        features["name_lev_ratio"] = Levenshtein.ratio(n1.lower(), n2.lower()) if n1 and n2 else 0.0
        features["name_jaro_winkler"] = Levenshtein.jaro_winkler(n1.lower(), n2.lower()) if n1 and n2 else 0.0
        features["name_token_sort"] = _token_sort_ratio(n1, n2)
        features["name_token_set"] = _token_set_ratio(n1, n2)
        features["name_trigram_jaccard"] = _trigram_jaccard(n1, n2)

        # 2. Address Features
        features["address_lev_ratio"] = Levenshtein.ratio(a1.lower(), a2.lower()) if a1 and a2 else 0.0
        features["address_jaro_winkler"] = Levenshtein.jaro_winkler(a1.lower(), a2.lower()) if a1 and a2 else 0.0
        features["address_token_sort"] = _token_sort_ratio(a1, a2)
        features["address_token_set"] = _token_set_ratio(a1, a2)
        features["address_trigram_jaccard"] = _trigram_jaccard(a1, a2)

        # 3. Digit matching in address (street numbers, zip codes)
        digit_jaccard, digit_exact = _digit_overlap(a1, a2)
        features["address_digit_jaccard"] = digit_jaccard
        features["address_digit_exact"] = digit_exact

        # 4. Country Features
        features["country_exact_match"] = float(c1 == c2 and len(c1) > 0)
        features["country_compatible"] = float(
            not c1 or not c2 or c1 == "unknown" or c2 == "unknown" or c1 == c2 or c1 in c2 or c2 in c1
        )

        # 5. Composite / Structural interaction features
        features["name_len_diff"] = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
        features["addr_len_diff"] = abs(len(a1) - len(a2)) / max(1, max(len(a1), len(a2)))
        
        # High address similarity even if name similarity is moderate (common when brand name varies)
        features["high_addr_sim"] = float(features["address_token_set"] > 0.85 or features["address_trigram_jaccard"] > 0.8)
        
        # High name similarity even if address is short
        features["high_name_sim"] = float(features["name_token_set"] > 0.85 or features["name_jaro_winkler"] > 0.85)

        # Target source indicator (S2 vs S3)
        features["is_target_s2"] = float(record2.entity_id.startswith("S2-"))
        features["is_target_s3"] = float(record2.entity_id.startswith("S3-"))

        return features

    def extract_batch_features(
        self, candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]]
    ) -> pd.DataFrame:
        rows = [self.extract_pair_features(r1, r2) for r1, r2 in candidate_pairs]
        return pd.DataFrame(rows)

