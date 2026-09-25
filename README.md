# Amazon Business Entity Resolution Challenge

A clean, modular Python codebase for resolving business entities across heterogeneous data sources (Source 1, Source 2, Source 3) under multi-lingual, cross-script, and noisy text conditions.

## Project Structure

```
.
├── configs/
│   └── baseline.yaml              # Pipeline configurations
├── docs/
│   └── problem.md                 # Detailed problem specification
├── experiments/
│   └── README.md                  # Experiment tracking log
├── src/
│   ├── data/
│   │   ├── schema.py              # BusinessRecord dataclass
│   │   └── loader.py              # TSV data loader
│   ├── preprocessing/
│   │   └── normalize.py           # NFKC Unicode normalizer
│   ├── blocking/
│   │   └── candidate_generator.py # Candidate blocking interface & generator
│   ├── features/
│   │   └── pair_features.py       # Pairwise feature extractor
│   ├── models/
│   │   └── matcher.py             # Matcher interface & model baseline
│   ├── evaluation/
│   │   └── metrics.py             # Precision, Recall, Candidate Recall & Macro F0.5
│   ├── pipeline/
│   │   └── train.py               # Training & evaluation pipeline
│   └── inference/
│       └── predict.py             # Test inference & submission exporter
├── student_resource/              # Official problem dataset and resources
│   ├── dataset/
│   │   ├── train/
│   │   └── test/
│   └── utils/
│       └── validate_submission.py # Official submission validation script
├── tests/                         # Pytest suite
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_output.py
├── requirements.txt
└── README.md
```

## Setup & Requirements

1. **Python Environment**: Python 3.8+ is recommended.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Training Pipeline

To run the training and evaluation pipeline on `student_resource/dataset/train`:
```bash
python -m src.pipeline.train
```

## Running Inference

To generate `submission.tsv` for the test dataset `student_resource/dataset/test`:
```bash
python -m src.inference.predict
```

## Validating Submission Output

Run the official validation script on the generated `submission.tsv` against the test source file:
```bash
python student_resource/utils/validate_submission.py student_resource/dataset/test/test_source1.tsv submission.tsv
```

## Running Tests

Execute unit tests with `pytest`:
```bash
pytest tests/
```

## Evaluation Metric

The primary evaluation metric is **Macro $F_{0.5}$** computed per Source 1 entity:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons (entities with 0 ground truth matches) receive $F_{0.5} = 1.0$ if no matches are predicted, and $0.0$ if false matches are predicted.
- The final score is macro-averaged across all Source 1 entities.
