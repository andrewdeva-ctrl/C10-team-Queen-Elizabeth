# AgriSLM TF-IDF Submission Pipeline

## Overview

This directory contains a production-ready implementation of the TF-IDF based question-answer matching pipeline for the TRI Agriculture & Climate SLM benchmark.

## Files

- **`tfidf_submission.py`** — Main pipeline module with `TfidfSubmissionPipeline` class
- **`test_tfidf_submission.py`** — Comprehensive unit tests (pytest)
- **`queenelizabeth-tfidf-submission.ipynb`** — Original Jupyter notebook (legacy)
- **`data_validation.py`** — Data integrity validation utility
- **`requirements.txt`** — Python dependencies

## Quick Start

### Installation

```bash
cd scripts
pip install -r requirements.txt
```

### Run Pipeline (Local)

```bash
python tfidf_submission.py --data-dir ../data --output-dir ../data --verbose
```

### Run with Options

```bash
# Generate detailed results with confidence scores
python tfidf_submission.py --save-detailed --top-k 1

# Retrieve top-3 candidate answers per question
python tfidf_submission.py --top-k 3 --save-detailed
```

### Run Tests

```bash
pytest test_tfidf_submission.py -v

# With coverage report
pytest test_tfidf_submission.py --cov=tfidf_submission --cov-report=html
```

## API Usage

### Basic Pipeline

```python
from tfidf_submission import TfidfSubmissionPipeline

pipeline = TfidfSubmissionPipeline(
    data_dir="../data",
    output_dir="../data",
    top_k=1
)

# Run entire pipeline
submission_path = pipeline.run(save_detailed=True)
print(f"Submission saved to {submission_path}")
```

### Step-by-Step Control

```python
pipeline = TfidfSubmissionPipeline(data_dir="../data")

# Load datasets
pipeline.load_data()
print(f"Loaded {len(pipeline.train_df)} training examples")

# Build vectorizer
pipeline.build_vectorizer()

# Match questions
pipeline.match_questions()

# Save results
pipeline.save_submission("my_submission.csv")
pipeline.save_detailed_results("results_detailed.json")

# Access results programmatically
for result in pipeline.results:
    print(f"Q{result['QuestionId']}: {result['Confidence']:.3f}")
```

## Algorithm

### Overview

1. **Vectorization**: Convert all training and test questions to TF-IDF vectors using unigrams and bigrams
2. **Similarity**: Compute cosine similarity between each test question and all training questions
3. **Retrieval**: Select the training question with highest similarity
4. **Answer Matching**: Return the reference answer from the best-matching training example
5. **Topic Filtering** (optional): If both datasets have a `topic` column, restrict matching within the same topic

### Hyperparameters

- **ngram_range**: (1, 2) — Use both unigrams and bigrams
- **lowercase**: True — Normalize text case
- **stop_words**: 'english' — Remove common English words (the, a, is, etc.)
- **max_features**: 5000 — Limit vocabulary to 5000 most frequent terms
- **top_k**: 1 (default) — Number of top candidates to retrieve

## Output

### submission.csv

Kaggle-compliant submission file with two columns:

```csv
QuestionId,Answer
1,Use organic compost and manure regularly.
2,Rotate crops to prevent soil depletion.
...
```

### submission_detailed.json (with --save-detailed)

Detailed results including confidence scores and top-k candidates:

```json
[
  {
    "QuestionId": 1,
    "Answer": "Use organic compost and manure regularly.",
    "Confidence": 0.847,
    "CandidateCount": 45,
    "TopKCandidates": [
      {"rank": 1, "score": 0.847, "answer": "..."},
      {"rank": 2, "score": 0.712, "answer": "..."}
    ]
  }
]
```

## Design Principles

### Production-Ready

- ✅ Comprehensive error handling and validation
- ✅ Structured logging for debugging
- ✅ Type hints for clarity
- ✅ Docstrings for all public methods
- ✅ Unit tests with pytest

### Reproducible

- ✅ Fixed random seeds (via sklearn defaults)
- ✅ Environment-agnostic path resolution (Kaggle vs local)
- ✅ All parameters configurable via CLI or API
- ✅ Clear dependency management

### Maintainable

- ✅ Modular class-based design
- ✅ Separation of concerns (load, vectorize, match, save)
- ✅ Extensible for future improvements (ensemble, fine-tuning, etc.)
- ✅ No hard-coded paths or magic numbers

## Improvements Over Notebook

| Feature | Notebook | Production |
|---------|----------|------------|
| Error Handling | Minimal | Comprehensive |
| Logging | None | Structured logging |
| Type Safety | None | Full type hints |
| Testing | None | Full pytest suite |
| Top-K Retrieval | No | Yes |
| Confidence Scores | No | Yes |
| Reusability | Single-use | Modular API |
| Documentation | Minimal | Extensive |
| CLI Interface | No | Full argparse |
| Data Validation | Basic | Comprehensive |

## Kaggle Submission

To run on Kaggle:

```python
from tfidf_submission import TfidfSubmissionPipeline, resolve_data_path

data_dir, output_dir = resolve_data_path()
pipeline = TfidfSubmissionPipeline(data_dir=data_dir, output_dir=output_dir)
pipeline.run()

print("Submission ready in /kaggle/working/submission.csv")
```

The pipeline automatically detects Kaggle environment and resolves paths accordingly.

## Future Enhancements

1. **Ensemble Methods**: Combine multiple vectorizers (TF-IDF, BM25, semantic embeddings)
2. **Fine-Tuning**: Add lightweight semantic model fine-tuning
3. **Preprocessing**: Advanced text preprocessing (lemmatization, domain-specific cleaning)
4. **Caching**: Cache vectorizer state for faster re-runs
5. **Hyperparameter Tuning**: Grid search over ngram_range, max_features, similarity threshold
6. **Answer Generation**: Move beyond retrieval to generative models

## License

Team-generated code: **CC BY-SA 4.0**

## Contact

Team QueenElizabeth — TRI AI Cohort 10
- Ernest Nchabeng (Team Leader)
- Andrew Devadason (Team Member)
