import pandas as pd
import unicodedata

def get_scripts_set(text):
    if not isinstance(text, str):
        return set(["LATIN"])
    scripts = set()
    for char in text:
        if char.isalpha():
            try:
                name = unicodedata.name(char).split()[0]
                scripts.add(name)
            except ValueError:
                pass
    if not scripts:
        return set(["LATIN"])
    return scripts

def analyze_true_matches():
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    gt_df = pd.read_csv(gt_path, sep="\t")
    
    needed_ids = set()
    pairs = []
    
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str or matched_str.strip() == "":
            continue
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        needed_ids.add(s1_id)
        for m_id in matched_ids:
            needed_ids.add(m_id)
            pairs.append((s1_id, m_id))

    print(f"Total True Matched Pairs in Ground Truth: {len(pairs):,}")
    print(f"Total Unique Entity IDs to load: {len(needed_ids):,}")

    records_dict = {}
    
    for src_file in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src_file}"
        print(f"Filtering and loading {src_file}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=100000, on_bad_lines="skip"):
            chunk_filtered = chunk[chunk["entity_id"].isin(needed_ids)]
            for _, row in chunk_filtered.iterrows():
                eid = row["entity_id"]
                name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
                records_dict[eid] = f"{name} {addr}"

    total_pairs = 0
    different_scripts_pairs = 0
    different_primary_script = 0
    script_mismatch_types = {}

    for s1_id, m_id in pairs:
        if s1_id not in records_dict or m_id not in records_dict:
            continue
        
        t1 = records_dict[s1_id]
        t2 = records_dict[m_id]
        
        s1_scripts = get_scripts_set(t1)
        m_scripts = get_scripts_set(t2)
        
        total_pairs += 1
        
        # Check if set of scripts differs
        if s1_scripts != m_scripts:
            different_scripts_pairs += 1
            
            # Identify non-Latin scripts if any
            s1_non_latin = sorted([s for s in s1_scripts if s != "LATIN"])
            m_non_latin = sorted([s for s in m_scripts if s != "LATIN"])
            
            s1_main = s1_non_latin[0] if s1_non_latin else "LATIN"
            m_main = m_non_latin[0] if m_non_latin else "LATIN"
            
            if s1_main != m_main:
                different_primary_script += 1
                pair_type = tuple(sorted([s1_main, m_main]))
                script_mismatch_types[pair_type] = script_mismatch_types.get(pair_type, 0) + 1

    print("\n" + "="*60)
    print("MATCHED PAIRS SCRIPT & TRANSLITERATION ANALYSIS")
    print("="*60)
    print(f"Total True Matched Pairs Analyzed: {total_pairs:,}")
    print(f"Pairs with Any Script Set Difference: {different_scripts_pairs:,} ({different_scripts_pairs / total_pairs * 100:.4f}%)")
    print(f"Pairs Across Different Primary Scripts (e.g., Non-Latin vs Latin): {different_primary_script:,} ({different_primary_script / total_pairs * 100:.4f}%)")
    print(f"Pairs Using Same Primary Script: {total_pairs - different_primary_script:,} ({(total_pairs - different_primary_script) / total_pairs * 100:.4f}%)")

    print("\nPrimary Script Mismatch Breakdown (Cross-script pairs):")
    for pair, count in sorted(script_mismatch_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {pair[0]} <---> {pair[1]}: {count:,} pairs ({count / total_pairs * 100:.4f}%)")

if __name__ == "__main__":
    analyze_true_matches()
