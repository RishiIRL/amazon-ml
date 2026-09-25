from src.data.schema import BusinessRecord
from src.preprocessing.normalize import UnicodeNormalizer


def test_unicode_normalizer():
    normalizer = UnicodeNormalizer(lowercase=True)

    # Test full-width characters and accents
    raw_name = "Ａｃｍｅ  Ｃｏｒｐ  Café"
    normalized_name = normalizer.normalize_text(raw_name)
    assert normalized_name == "acme corp café"

    # Test non-Latin script preservation (Devanagari, Japanese)
    raw_non_latin = "  अमज़ॉन   株式会社  "
    normalized_non_latin = normalizer.normalize_text(raw_non_latin)
    assert normalized_non_latin == "अमज़ॉन 株式会社"

    # Test record normalization
    record = BusinessRecord("e1", "  TEST Corp  ", " 100 Main St ", " US ", "source1")
    norm_rec = normalizer.normalize_record(record)
    assert norm_rec.business_name == "test corp"
    assert norm_rec.business_address == "100 main st"
    assert norm_rec.country == "us"
