from unittest.mock import patch
import pytest
from src.data.schema import BusinessRecord
from src.preprocessing.transliteration import (
    ScriptAwareTransliterator,
    INDIC_TRANSLITERATION_AVAILABLE,
)


@pytest.fixture
def transliterator() -> ScriptAwareTransliterator:
    return ScriptAwareTransliterator(target_scheme="ITRANS")


# 1. Test every supported Indic script
@pytest.mark.parametrize(
    "script_name,sample_text,expected_sub",
    [
        ("DEVANAGARI", "किराना दुकान", "kirAnA"),
        ("BENGALI", "মুদি দোকান", "mudi"),
        ("TAMIL", "மளிகை கடை", "maLighai"),
        ("TELUGU", "కిరాణా కొట్టు", "kirANA"),
        ("KANNADA", "ಕಿರಣ ಅಂಗಡಿ", "kiraNa"),
        ("MALAYALAM", "മളികക്കട", "maLikakkaTa"),
        ("GUJARATI", "કરિયાણાની દુકાન", "kariyANAnI"),
        ("GURMUKHI", "ਕਰਿਆਨਾ ਦੁਕਾਨ", "kariAnA"),
        ("ORIYA", "ମୁଦି ଦୋକାନ", "mudi"),
    ],
)
def test_all_supported_indic_scripts(
    transliterator: ScriptAwareTransliterator,
    script_name: str,
    sample_text: str,
    expected_sub: str,
) -> None:
    result = transliterator.transliterate_text(sample_text)
    assert result != ""
    if INDIC_TRANSLITERATION_AVAILABLE:
        assert expected_sub in result


# 2. Test mixed-script strings
@pytest.mark.parametrize(
    "mixed_text,expected_parts",
    [
        ("ABC किराना Store", ["ABC", "kirAnA", "Store"]),
        ("Hotel कृष्णा Palace", ["Hotel", "kRRiShNA", "Palace"]),
        ("12 MG Road, कृष्णा Nagar", ["12 MG Road,", "kRRiShNA", "Nagar"]),
    ],
)
def test_mixed_script_strings(
    transliterator: ScriptAwareTransliterator,
    mixed_text: str,
    expected_parts: list[str],
) -> None:
    result = transliterator.transliterate_text(mixed_text)
    if INDIC_TRANSLITERATION_AVAILABLE:
        for part in expected_parts:
            assert part in result


# 3. Test business names AND addresses
def test_business_names_and_addresses(transliterator: ScriptAwareTransliterator) -> None:
    rec = BusinessRecord(
        entity_id="S1-999",
        business_name="Hotel कृष्णा Palace",
        business_address="12 MG Road, कृष्णा Nagar, Mumbai 400001",
        country="India",
        source="source1",
    )
    t_rec = transliterator.transliterate_record(rec)
    assert t_rec.entity_id == "S1-999"
    assert t_rec.country == "India"
    assert t_rec.source == "source1"
    if INDIC_TRANSLITERATION_AVAILABLE:
        assert "Hotel kRRiShNA Palace" in t_rec.business_name
        assert "12 MG Road, kRRiShNA Nagar, Mumbai 400001" in t_rec.business_address


# 4. Verify Latin text is preserved exactly
def test_latin_text_preservation(transliterator: ScriptAwareTransliterator) -> None:
    latin_inputs = [
        "Amazon Retail LLC",
        "Pvt Ltd 123 Main Street Suite #400",
        "Global Enterprises (Branch A)",
    ]
    for text in latin_inputs:
        assert transliterator.transliterate_text(text) == text


# 5. Verify punctuation and numbers are preserved appropriately
def test_punctuation_and_numbers_preservation(transliterator: ScriptAwareTransliterator) -> None:
    text = "Unit 4-B/12, #99 (किराना Store) - Pin: 560001!"
    result = transliterator.transliterate_text(text)
    if INDIC_TRANSLITERATION_AVAILABLE:
        assert result.startswith("Unit 4-B/12, #99 (")
        assert "kirAnA Store) - Pin: 560001!" in result


# 6. Verify transliteration never calls translation or external services
def test_no_external_service_calls(transliterator: ScriptAwareTransliterator) -> None:
    def block_network(*args, **kwargs):
        raise RuntimeError("Network call attempted during transliteration!")

    with patch("socket.socket.connect", side_effect=block_network):
        res = transliterator.transliterate_text("Hotel कृष्णा Palace")
        if INDIC_TRANSLITERATION_AVAILABLE:
            assert res == "Hotel kRRiShNA Palace"


# 7. Check whether ITRANS output is deterministic
def test_transliteration_determinism(transliterator: ScriptAwareTransliterator) -> None:
    text = "12 MG Road, कृष्णा Nagar, মুদি দোকান, 560001"
    first_run = transliterator.transliterate_text(text)
    for _ in range(50):
        assert transliterator.transliterate_text(text) == first_run


# 8. Test multi-script strings containing multiple Indic scripts in a single input
def test_multi_indic_script_string(transliterator: ScriptAwareTransliterator) -> None:
    text = "किराना (Devanagari) and মুদি (Bengali)"
    result = transliterator.transliterate_text(text)
    if INDIC_TRANSLITERATION_AVAILABLE:
        assert "kirAnA" in result
        assert "mudi" in result
        assert "(Devanagari) and" in result
        assert "(Bengali)" in result


# 9. Document library quirks (Telugu Dravidian short vowels & Oriya Nukta)
def test_library_quirks_documented(transliterator: ScriptAwareTransliterator) -> None:
    if not INDIC_TRANSLITERATION_AVAILABLE:
        pytest.skip("indic_transliteration not installed")

    # Telugu short vowel 'e' / 'o' mapping to accented 'è' / 'ò' in ITRANS
    telugu_text = "తెలుగు"
    res_telugu = transliterator.transliterate_text(telugu_text)
    assert res_telugu == "tèlugu"  # 'è' is Latin letter with grave accent

    # Oriya Nukta sign (\u0b3c) remaining as combining character
    oriya_text = "ଓଡ଼ିଆ"
    res_oriya = transliterator.transliterate_text(oriya_text)
    assert "oDa" in res_oriya and "iA" in res_oriya
