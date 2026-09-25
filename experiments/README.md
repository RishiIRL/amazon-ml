# Experiment Log & Performance Tracking

This directory contains records of model iterations, candidate generation strategies, hyperparameter configurations, and evaluation metrics for the Amazon Business Entity Resolution challenge.

## Evaluation Metric
- Primary Metric: **Macro $F_{0.5}$** across all Source 1 entities (including singletons).
- Secondary Metric: **Candidate Generation (Blocking) Recall**.

## Experiment History

| Exp ID | Date | Blocking Strategy | Feature Set | Model / Classifier | Threshold | Blocking Recall | Macro $F_{0.5}$ | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-000 | Baseline | Simple exact name match stub | Placeholders (Name, Address, Country) | Baseline Entity Matcher | 0.5 | TBD | TBD | Initial pipeline skeleton verification |

## Future Hypotheses & Roadmap
1. **Transliteration & Unicode Normalization**: Test script normalization across Devanagari, Japanese (Kanji/Kana), Cyrillic, Chinese, and Latin scripts.
2. **Candidate Blocking**: Implement blocking using character n-gram TF-IDF, MinHash / LSH, and country-constrained candidate retrieval.
3. **Similarity Features**: Extract Levenshtein/Jaro-Winkler string distances, token overlap, soft TF-IDF, entity embeddings, and address components.
4. **Machine Learning Classifier**: Train LightGBM / XGBoost classifier on candidate pairs using positive pairs from ground truth and random negative samples.
