import logging
import unicodedata
from typing import Dict, Optional
from src.data.schema import BusinessRecord

logger = logging.getLogger(__name__)

# Optional import for Indic transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False
    logger.warning("indic_transliteration package not available. Transliteration will fallback to pass-through.")

# Mapping Unicode block name fragments to sanscript scheme constants
SCRIPT_MAP: Dict[str, str] = {}
if INDIC_TRANSLITERATION_AVAILABLE:
    SCRIPT_MAP = {
        "DEVANAGARI": sanscript.DEVANAGARI,
        "BENGALI": sanscript.BENGALI,
        "TAMIL": sanscript.TAMIL,
        "TELUGU": sanscript.TELUGU,
        "KANNADA": sanscript.KANNADA,
        "MALAYALAM": sanscript.MALAYALAM,
        "GUJARATI": sanscript.GUJARATI,
        "GURMUKHI": sanscript.GURMUKHI,
        "ORIYA": sanscript.ORIYA,
    }


class ScriptAwareTransliterator:
    """Script-aware transliterator for Indic non-Latin text to Latin representation.
    
    Preserves original Unicode text without translation, external network calls,
    or destructive character stripping.
    """

    def __init__(self, target_scheme: str = "ITRANS") -> None:
        self.target_scheme = target_scheme
        if INDIC_TRANSLITERATION_AVAILABLE:
            self.target_scheme_const = getattr(sanscript, target_scheme, sanscript.ITRANS)

    def detect_primary_script(self, text: str) -> Optional[str]:
        """Detect primary non-Latin Indic script in the given text based on Unicode character blocks."""
        if not text:
            return None

        script_counts: Dict[str, int] = {}
        for char in text:
            try:
                block = unicodedata.name(char, "").split()[0]
                if block in SCRIPT_MAP:
                    script_counts[block] = script_counts.get(block, 0) + 1
            except (ValueError, IndexError):
                continue

        if not script_counts:
            return None

        # Return the script with highest character count
        return max(script_counts, key=lambda s: script_counts[s])

    def detect_indic_scripts(self, text: str) -> list[str]:
        """Detect all non-Latin Indic scripts present in the text based on Unicode character blocks."""
        if not text:
            return []

        scripts_found: list[str] = []
        for char in text:
            try:
                block = unicodedata.name(char, "").split()[0]
                if block in SCRIPT_MAP and block not in scripts_found:
                    scripts_found.append(block)
            except (ValueError, IndexError):
                continue

        return scripts_found

    def transliterate_text(self, text: Optional[str]) -> str:
        """Transliterate text if non-Latin Indic scripts are detected.
        
        Returns transliterated text while preserving non-Indic / Latin portions.
        Falls back to original text if no Indic script is detected or if transliteration fails.
        """
        if not text or not isinstance(text, str):
            return ""

        if not INDIC_TRANSLITERATION_AVAILABLE:
            return text

        scripts = self.detect_indic_scripts(text)
        if not scripts:
            return text

        result = text
        for script_name in scripts:
            source_scheme = SCRIPT_MAP.get(script_name)
            if not source_scheme:
                continue
            try:
                result = transliterate(result, source_scheme, self.target_scheme_const)
            except Exception as e:
                logger.debug("Transliteration failed for script %s on text '%s': %s", script_name, text, e)

        return result

    def transliterate_record(self, record: BusinessRecord) -> BusinessRecord:
        """Transliterate fields of a BusinessRecord returning a new BusinessRecord.
        
        Note: The original non-Latin text is preserved if no transliteration occurs,
        or transliterated fields can be used for feature extraction alongside original text.
        """
        return BusinessRecord(
            entity_id=record.entity_id,
            business_name=self.transliterate_text(record.business_name),
            business_address=self.transliterate_text(record.business_address),
            country=record.country,
            source=record.source,
        )
