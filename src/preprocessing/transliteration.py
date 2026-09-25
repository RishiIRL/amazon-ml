import logging
import unicodedata
from typing import Dict, List, Optional
from src.data.schema import BusinessRecord

logger = logging.getLogger(__name__)

# Optional import for Indic transliteration via indic-transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False
    sanscript = None
    transliterate = None
    logger.info("indic_transliteration package not available locally; will fallback if ai4bharat is missing.")

# Optional import for AI4Bharat transliteration engine (XlitEngine)
try:
    from ai4bharat.transliteration import XlitEngine
    AI4BHARAT_AVAILABLE = True
except ImportError:
    AI4BHARAT_AVAILABLE = False
    XlitEngine = None
    logger.info("ai4bharat-transliteration package not available locally; fallback engines will be used.")

# Mapping Unicode block name fragments to sanscript scheme constants
SCRIPT_MAP: Dict[str, str] = {}
if INDIC_TRANSLITERATION_AVAILABLE and sanscript is not None:
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
else:
    SCRIPT_MAP = {
        "DEVANAGARI": "devanagari",
        "BENGALI": "bengali",
        "TAMIL": "tamil",
        "TELUGU": "telugu",
        "KANNADA": "kannada",
        "MALAYALAM": "malayalam",
        "GUJARATI": "gujarati",
        "GURMUKHI": "gurmukhi",
        "ORIYA": "oriya",
    }

# Mapping Unicode block names to AI4Bharat language codes
AI4BHARAT_LANG_MAP: Dict[str, str] = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "TAMIL": "ta",
    "TELUGU": "te",
    "KANNADA": "kn",
    "MALAYALAM": "ml",
    "GUJARATI": "gu",
    "GURMUKHI": "pa",
    "ORIYA": "or",
}


class ScriptAwareTransliterator:
    """Script-aware transliterator supporting both AI4Bharat XlitEngine and indic-transliteration.
    
    Converts Indic non-Latin text to Latin script while preserving original content.
    Provides automatic fallback if specific packages or engines are missing in runtime.
    """

    def __init__(self, target_scheme: str = "ITRANS", use_ai4bharat: bool = True) -> None:
        self.target_scheme = target_scheme
        self.use_ai4bharat = use_ai4bharat
        self.target_scheme_const = getattr(sanscript, target_scheme, "ITRANS") if sanscript else "ITRANS"
        
        self.xlit_engine = None
        if self.use_ai4bharat and AI4BHARAT_AVAILABLE and XlitEngine is not None:
            try:
                import argparse
                import torch
                try:
                    torch.serialization.add_safe_globals([argparse.Namespace])
                except Exception:
                    pass
                # Initialize XlitEngine for Indic to English transliteration
                self.xlit_engine = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)
                logger.info("Successfully initialized AI4Bharat XlitEngine.")
            except Exception as e:
                logger.warning("Failed to initialize AI4Bharat XlitEngine: %s. Falling back to sanscript.", e)

    def detect_primary_script(self, text: str) -> Optional[str]:
        """Detect primary non-Latin Indic script in text based on Unicode character blocks."""
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

        return max(script_counts, key=lambda s: script_counts[s])

    def detect_indic_scripts(self, text: str) -> List[str]:
        """Detect all non-Latin Indic scripts present in text based on Unicode character blocks."""
        if not text:
            return []

        scripts_found: List[str] = []
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
        
        Attempts transliteration using AI4Bharat XlitEngine first, then sanscript,
        and falls back to original text if engines are unavailable or transliteration fails.
        """
        if not text or not isinstance(text, str):
            return ""

        scripts = self.detect_indic_scripts(text)
        if not scripts:
            return text

        # 1. Try AI4Bharat XlitEngine if available
        if self.xlit_engine is not None:
            try:
                result = text
                for script_name in scripts:
                    lang_code = AI4BHARAT_LANG_MAP.get(script_name)
                    if lang_code and hasattr(self.xlit_engine, "translit_sentence"):
                        result = self.xlit_engine.translit_sentence(result, lang_code)
                return result
            except Exception as e:
                logger.debug("AI4Bharat XlitEngine transliteration failed: %s. Falling back to sanscript.", e)

        # 2. Try indic-transliteration (sanscript) if available
        if INDIC_TRANSLITERATION_AVAILABLE and transliterate is not None:
            result = text
            for script_name in scripts:
                source_scheme = SCRIPT_MAP.get(script_name)
                if not isinstance(source_scheme, str):
                    continue
                try:
                    result = transliterate(result, source_scheme, self.target_scheme_const)
                except Exception as e:
                    logger.debug("Sanscript transliteration failed for script %s on '%s': %s", script_name, text, e)
            return result

        # 3. Fallback: Return original text unchanged
        return text

    def transliterate_record(self, record: BusinessRecord) -> BusinessRecord:
        """Transliterate fields of a BusinessRecord returning a new BusinessRecord."""
        return BusinessRecord(
            entity_id=record.entity_id,
            business_name=self.transliterate_text(record.business_name),
            business_address=self.transliterate_text(record.business_address),
            country=record.country,
            source=record.source,
        )

