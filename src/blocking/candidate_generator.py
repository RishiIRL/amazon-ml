import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.data.schema import BusinessRecord


class BaseCandidateGenerator(ABC):
    """Abstract base class for candidate generators (blocking strategies)."""

    @abstractmethod
    def generate_candidates(
        self,
        source1: Union[pd.DataFrame, List[BusinessRecord]],
        *target_sources: Union[pd.DataFrame, List[BusinessRecord], List[Union[pd.DataFrame, List[BusinessRecord]]]]
    ) -> List[Tuple[BusinessRecord, BusinessRecord]]:
        """Generate candidate matching pairs.

        Returns:
            List of (Source 1 BusinessRecord, Target BusinessRecord) tuples.
        """
        pass


def _tokenize(text: str) -> Set[str]:
    """Extract alphanumeric tokens from text."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower()))


def _token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(union)


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams for fuzzy matching."""
    s = f" #{text.lower()}# "
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


class SimpleCandidateGenerator(BaseCandidateGenerator):
    """Multi-source candidate generator with blocking by country and token/n-gram indexing.
    
    Generates high-recall candidate pairs between Source 1 records and target sources (S2, S3).
    Handles edge cases like missing/empty country fields and different business name spellings
    or address structural variations.
    """

    def __init__(
        self,
        block_by_country: bool = True,
        max_candidates_per_entity: Optional[int] = 50,
        max_candidates_per_s1: Optional[int] = None,
        min_similarity_threshold: float = 0.05,
    ) -> None:
        self.block_by_country = block_by_country
        self.max_candidates_per_entity = max_candidates_per_s1 or max_candidates_per_entity or 50
        self.min_similarity_threshold = min_similarity_threshold

    def _normalize_to_records(
        self, src: Any
    ) -> List[BusinessRecord]:
        """Convert DataFrame or list into List[BusinessRecord]."""
        if isinstance(src, list):
            records = []
            for item in src:
                if isinstance(item, list):
                    records.extend(self._normalize_to_records(item))
                elif isinstance(item, BusinessRecord):
                    records.append(item)
            return records
        elif isinstance(src, pd.DataFrame):
            records = []
            for _, row in src.iterrows():
                eid = str(row["entity_id"])
                records.append(
                    BusinessRecord(
                        entity_id=eid,
                        business_name=str(row.get("business_name", "")),
                        business_address=str(row.get("business_address", "")),
                        country=str(row.get("country", "")),
                    )
                )
            return records
        return []

    def _country_compatible(self, c1: str, c2: str) -> bool:
        """Check if two country strings are compatible for blocking."""
        c1_clean = (c1 or "").strip().lower()
        c2_clean = (c2 or "").strip().lower()

        # If either country is missing or unknown, assume compatible to avoid dropping recall
        if not c1_clean or not c2_clean or c1_clean == "unknown" or c2_clean == "unknown":
            return True

        if c1_clean == c2_clean:
            return True

        # Check substring match (e.g. 'us' in 'usa', 'united states')
        if c1_clean in c2_clean or c2_clean in c1_clean:
            return True

        return False

    def _pair_blocking_score(self, r1: BusinessRecord, r2: BusinessRecord) -> float:
        """Compute quick composite similarity score for candidate ranking."""
        name_toks1 = _tokenize(r1.business_name)
        name_toks2 = _tokenize(r2.business_name)
        addr_toks1 = _tokenize(r1.business_address)
        addr_toks2 = _tokenize(r2.business_address)

        name_jaccard = _token_jaccard(name_toks1, name_toks2)
        addr_jaccard = _token_jaccard(addr_toks1, addr_toks2)

        # Character n-gram similarity for names
        ngram1 = _char_ngrams(r1.business_name)
        ngram2 = _char_ngrams(r2.business_name)
        ngram_sim = _token_jaccard(ngram1, ngram2)

        # Character n-gram similarity for addresses
        addr_ngram1 = _char_ngrams(r1.business_address)
        addr_ngram2 = _char_ngrams(r2.business_address)
        addr_ngram_sim = _token_jaccard(addr_ngram1, addr_ngram2)

        # Highest of name sim, address sim, or n-gram sim
        max_sim = max(name_jaccard, addr_jaccard, ngram_sim, addr_ngram_sim)

        # Boost if numbers in address match (e.g. street numbers or zip codes)
        nums1 = set(re.findall(r"\d+", r1.business_address))
        nums2 = set(re.findall(r"\d+", r2.business_address))
        if nums1 and nums2 and nums1.intersection(nums2):
            max_sim += 0.15

        return max_sim

    def generate_candidates(
        self,
        source1: Union[pd.DataFrame, List[BusinessRecord]],
        *target_sources: Any
    ) -> List[Tuple[BusinessRecord, BusinessRecord]]:
        """Generate candidate pairs for each Source 1 record against target sources."""
        s1_records = self._normalize_to_records(source1)

        targets: List[BusinessRecord] = []
        for ts in target_sources:
            targets.extend(self._normalize_to_records(ts))

        # Group targets by target source (e.g. S2 vs S3)
        s2_targets: List[BusinessRecord] = [r for r in targets if r.entity_id.startswith("S2-")]
        s3_targets: List[BusinessRecord] = [r for r in targets if r.entity_id.startswith("S3-")]

        # If prefixes are not present, partition all targets as single group
        if not s2_targets and not s3_targets:
            target_groups = [targets]
        else:
            target_groups = []
            if s2_targets:
                target_groups.append(s2_targets)
            if s3_targets:
                target_groups.append(s3_targets)

        candidate_pairs: List[Tuple[BusinessRecord, BusinessRecord]] = []

        for r1 in s1_records:
            for group in target_groups:
                scored_candidates: List[Tuple[float, BusinessRecord]] = []

                for r_target in group:
                    # Country blocking check
                    if self.block_by_country and not self._country_compatible(r1.country, r_target.country):
                        continue

                    score = self._pair_blocking_score(r1, r_target)
                    scored_candidates.append((score, r_target))

                # If country blocking produced no candidates, try without country constraint
                if not scored_candidates:
                    for r_target in group:
                        score = self._pair_blocking_score(r1, r_target)
                        scored_candidates.append((score, r_target))

                # Sort by similarity score descending
                scored_candidates.sort(key=lambda x: x[0], reverse=True)

                # Select top candidates per target source
                selected = scored_candidates[: self.max_candidates_per_entity]
                for score, r_target in selected:
                    candidate_pairs.append((r1, r_target))

        return candidate_pairs


class CandidateGenerator(SimpleCandidateGenerator):
    """Alias for SimpleCandidateGenerator for API backward compatibility."""
    pass
