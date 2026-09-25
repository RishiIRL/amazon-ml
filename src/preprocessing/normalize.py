import re
import unicodedata
from typing import Optional
from src.data.schema import BusinessRecord


class BaseNormalizer:
    """Abstract interface for text normalization."""

    def normalize(self, text: Optional[str]) -> str:
        """Normalize input string."""
        raise NotImplementedError


class UnicodeNormalizer(BaseNormalizer):
    """Safe, Unicode-preserving normalizer for business names and addresses.
    
    Preserves all native character sets (Latin, Devanagari, Kannada, Telugu,
    Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi, etc.).
    """

    def __init__(
        self,
        unicode_form: str = "NFKC",
        lowercase: bool = True,
        form: Optional[str] = None,
        strip_whitespace: bool = True,
    ) -> None:
        self.unicode_form = form or unicode_form
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace

    def normalize(self, text: Optional[str]) -> str:
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode normalization (NFKC decomposes compatibility chars safely)
        normalized = unicodedata.normalize(self.unicode_form, text)

        # 2. Lowercase conversion preserving script casing where applicable
        if self.lowercase:
            normalized = normalized.lower()

        # 3. Safe punctuation cleanup (collapse multiple punctuation marks)
        normalized = re.sub(r"[\s\t\n\r]+", " ", normalized)
        
        # 4. Strip surrounding whitespace
        return normalized.strip()

    def normalize_text(self, text: Optional[str]) -> str:
        """Alias for normalize."""
        return self.normalize(text)

    def normalize_record(self, record: BusinessRecord) -> BusinessRecord:
        """Normalize all fields of a BusinessRecord returning a new BusinessRecord."""
        return BusinessRecord(
            entity_id=record.entity_id,
            business_name=self.normalize(record.business_name),
            business_address=self.normalize(record.business_address),
            country=self.normalize(record.country),
            source=record.source,
        )

    def normalize_dataset(self, records: list) -> list:
        """Normalize all records in a list of BusinessRecords."""
        return [self.normalize_record(r) for r in records]


def normalize_business_record(name: str, address: str, normalizer: Optional[BaseNormalizer] = None) -> tuple:
    """Normalize business name and address using Unicode-preserving normalization."""
    if normalizer is None:
        normalizer = UnicodeNormalizer()
    norm_name = normalizer.normalize(name)
    norm_address = normalizer.normalize(address)
    return norm_name, norm_address
