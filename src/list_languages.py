import os
import glob
import pandas as pd
from langdetect import detect_langs, DetectorFactory
import unicodedata
from collections import Counter

# Set seed for reproducible langdetect results
DetectorFactory.seed = 0

LANG_NAMES = {
    'en': 'English',
    'fr': 'French',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'ml': 'Malayalam',
    'or': 'Odia',
    'pa': 'Punjabi',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'id': 'Indonesian',
    'ca': 'Catalan',
    'tl': 'Tagalog',
    'et': 'Estonian',
    'so': 'Somali',
    'sv': 'Swedish',
    'ro': 'Romanian',
}

SCRIPT_NAMES = {
    'LATIN': 'Latin (English / French / Transliterated text)',
    'DEVANAGARI': 'Devanagari (Hindi / Marathi)',
    'KANNADA': 'Kannada',
    'TELUGU': 'Telugu',
    'TAMIL': 'Tamil',
    'BENGALI': 'Bengali',
    'GUJARATI': 'Gujarati',
    'MALAYALAM': 'Malayalam',
    'ORIYA': 'Odia',
    'GURMUKHI': 'Gurmukhi (Punjabi)'
}

def detect_unicode_scripts(text):
    scripts = set()
    for char in text:
        if char.isalpha():
            try:
                script_name = unicodedata.name(char).split()[0]
                scripts.add(script_name)
            except ValueError:
                pass
    return scripts

def analyze_dataset():
    data_dir = "student_resource/dataset"
    tsv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tsv"), recursive=True))
    tsv_files = [f for f in tsv_files if "ground_truth" not in f]

    countries = Counter()
    lang_counter = Counter()
    script_counter = Counter()

    total_records = 0

    for filepath in tsv_files:
        df = pd.read_csv(filepath, sep="\t", on_bad_lines="skip")
        
        if "country" in df.columns:
            countries.update(df["country"].dropna())
        
        total_records += len(df)
        
        sample_df = df.sample(n=min(3000, len(df)), random_state=42) if len(df) > 3000 else df

        for _, row in sample_df.iterrows():
            name = str(row.get("business_name", ""))
            addr = str(row.get("business_address", ""))
            text = f"{name} {addr}".strip()

            if not text or len(text) < 3:
                continue

            for script in detect_unicode_scripts(text):
                if script in SCRIPT_NAMES:
                    script_counter[script] += 1

            try:
                predictions = detect_langs(text)
                for pred in predictions:
                    if pred.prob > 0.6:
                        lang_counter[pred.lang] += 1
            except Exception:
                pass

    print("="*60)
    print("DATABASE LANGUAGE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Records across TSV files: {total_records:,}")
    print("\nCountries in Database:")
    for country, count in countries.items():
        print(f"  - {country}: {count:,} records")

    print("\nDetected Text Scripts / Alphabets:")
    for script, count in script_counter.most_common():
        name = SCRIPT_NAMES.get(script, script)
        print(f"  - {name}: {count:,} occurrences in sample")

    print("\nDetected Languages:")
    for code, count in lang_counter.most_common():
        lang_name = LANG_NAMES.get(code, f"Code ({code})")
        print(f"  - {lang_name} ({code}): {count:,} detected in sample")

if __name__ == "__main__":
    analyze_dataset()
