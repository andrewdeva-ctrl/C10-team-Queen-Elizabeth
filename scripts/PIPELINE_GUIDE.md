# TF-IDF Pipeline Implementation Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  TfidfSubmissionPipeline                                    │
│  ─────────────────────────────────────────────────────────  │
│  1. load_data()          → Load & validate CSVs             │
│  2. build_vectorizer()   → Fit TF-IDF on training questions │
│  3. match_questions()    → Match test → training answers    │
│  4. save_submission()    → Export Kaggle CSV format         │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Input

```
├── train_qa.csv
│   ├── QuestionId (int)
│   ├── question (str)        ← Used for vectorization
│   └── reference_answer (str) ← Output for matching
│
└── test_questions.csv
    ├── QuestionId (int)
    └── question (str)         ← Match against training
```

### Processing

```
train_qa.csv questions
    ↓
[TfidfVectorizer.fit_transform()]
    ↓ 
train_matrix (sparse CSR matrix, 45×n_features)
    ↓
For each test question:
  ├─ [vectorizer.transform()]
  ├─ test_vector (1×n_features)
  ├─ [cosine_similarity(test_vector, train_matrix)]
  ├─ similarities (1×45)
  ├─ argmax() → best_idx
  └─ train_qa.iloc[best_idx]["reference_answer"]
```

### Output

```
submission.csv
├── QuestionId
└── Answer (from best-matching training example)

submission_detailed.json (optional)
├── QuestionId
├── Answer
├── Confidence (cosine similarity score 0-1)
├── CandidateCount
└── TopKCandidates (if top_k > 1)
```

## Key Design Decisions

### 1. TF-IDF + Cosine Similarity

**Why**: 
- Interpretable and reproducible
- Fast inference (no neural network overhead)
- Works well for retrieval-based QA
- Proven baseline for semantic matching

**Parameters**:
- **ngram_range=(1,2)**: Captures both individual words and common phrases
- **stop_words='english'**: Removes noise ("the", "is", "a")
- **max_features=5000**: Balances expressiveness vs efficiency

### 2. Topic-Based Filtering

**Why**:
- AgriSLM covers multiple agricultural domains
- Topic context improves answer relevance
- Falls back to full corpus if no topic match found

**Implementation**:
```python
if "topic" in test.columns and topic_mask.any():
    sub_df = train[train["topic"] == test_topic]
else:
    sub_df = train  # Fallback to all
```

### 3. Confidence Scoring

**Why**:
- Quantifies match quality (0-1)
- Enables filtering low-confidence predictions
- Useful for downstream ensemble methods

**Formula**: Cosine similarity = dot(A, B) / (||A|| × ||B||)

### 4. Top-K Candidates

**Why**:
- Single answer may miss valid alternatives
- Enables ensemble/voting strategies
- Provides fallback options

**Usage**:
```python
pipeline = TfidfSubmissionPipeline(top_k=3)
pipeline.run()  # Gets top-3 matches per question
```

## Error Handling Strategy

### Load Phase

```python
try:
    df = pd.read_csv(path)
    validate_columns(df, required_cols)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {path}")
except ValueError:
    raise ValueError(f"Missing required columns")
```

### Transform Phase

```python
if not train_matrix or not vectorizer:
    raise RuntimeError("Vectorizer not built. Call build_vectorizer()")
```

### Validation Phase

```python
# Check for empty answers
if df["Answer"].isnull().sum() > 0:
    logger.warning(f"Found {n} null answers")

# Check column names
if set(df.columns) != {"QuestionId", "Answer"}:
    logger.warning("Unexpected columns in submission")
```

## Testing Strategy

### Unit Tests

```python
# Test each component independently
test_load_data()          # CSV reading and validation
test_build_vectorizer()   # TF-IDF fitting
test_match_questions()    # Similarity computation
test_save_submission()    # File output format
```

### Integration Tests

```python
# Test full pipeline on sample data
test_run_complete_pipeline()  # End-to-end execution
```

### Fixtures

```python
@pytest.fixture
def sample_data():
    return minimal_train_df, minimal_test_df

@pytest.fixture
def temp_data_dir(sample_data):
    # Write CSVs to temp directory
    yield temp_path
```

## Performance Characteristics

### Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Vectorization | O(n_docs × n_words) | O(n_docs × n_features) | Sparse matrix |
| Similarity | O(n_test × n_train × n_features) | O(n_test × n_train) | Dense output |
| Total | ~O(100ms) | ~O(10MB) | For 45 train + 12 test |

### Scalability

- **1K training docs**: ~1s
- **10K training docs**: ~10s  
- **100K training docs**: ~100s (may need batching)

### Memory

- Train matrix: ~5MB (sparse, 5000 features)
- Temporary dense similarity: ~50KB per test question

## Debugging Tips

### Enable Verbose Logging

```python
pipeline = TfidfSubmissionPipeline(verbose=True)
pipeline.run()
# Logs: data loading, vectorizer stats, match counts, file paths
```

### Inspect Intermediate Results

```python
pipeline.load_data()
print(pipeline.train_df.head())
print(f"Shape: {pipeline.train_df.shape}")

pipeline.build_vectorizer()
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"Train matrix shape: {pipeline.train_matrix.shape}")
```

### Check Individual Matches

```python
pipeline.match_questions()
for result in pipeline.results[:3]:
    print(f"Q{result['QuestionId']}: confidence={result['Confidence']:.3f}")
    print(f"  Answer: {result['Answer'][:50]}...")
```

### Validate Output

```python
submission = pd.read_csv("submission.csv")
print(f"Rows: {len(submission)}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Non-null answers: {submission['Answer'].notna().sum()}")
```

## Common Issues & Solutions

### Issue: "No CSV files found"

**Cause**: Wrong data directory path

**Fix**:
```python
# Check current working directory
import os
print(os.getcwd())

# Use absolute paths
pipeline = TfidfSubmissionPipeline(data_dir="/absolute/path/to/data")
```

### Issue: "Missing core column"

**Cause**: CSV has unexpected column names

**Fix**:
```python
df = pd.read_csv("train_qa.csv")
print(df.columns)  # Verify actual column names
```

### Issue: Very low confidence scores (<0.3)

**Cause**: Test questions don't match training corpus well

**Fix**:
```python
# Check if test/train are from same domain
print(pipeline.test_df["question"].iloc[0])
print(pipeline.train_df["question"].head())

# Try different ngram_range
pipeline = TfidfSubmissionPipeline(ngram_range=(1, 3))
```

## Extension Points

### Custom Similarity Metric

```python
class CustomPipeline(TfidfSubmissionPipeline):
    def match_questions(self):
        # Override with BM25, semantic embeddings, etc.
        pass
```

### Post-Processing

```python
def post_process_answers(self):
    for result in self.results:
        # Clean, augment, or rerank answers
        result["Answer"] = clean_text(result["Answer"])
```

### Ensemble Strategy

```python
class EnsemblePipeline:
    def __init__(self):
        self.tfidf = TfidfSubmissionPipeline()
        self.bm25 = BM25Pipeline()
    
    def run(self):
        tfidf_results = self.tfidf.run()
        bm25_results = self.bm25.run()
        return combine_results(tfidf_results, bm25_results)
```

## References

- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- scikit-learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Kaggle Submission Format: https://www.kaggle.com/competitions/agriculture-climate-slm-challenge
