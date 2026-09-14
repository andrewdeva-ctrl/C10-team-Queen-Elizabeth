"""Production-ready TF-IDF based question-answer matching pipeline.

This module implements a semantic matching solution using TF-IDF vectorization
and cosine similarity to match test questions with reference training answers.
Designed for the TRI Agriculture & Climate SLM benchmark.

Author: Team QueenElizabeth
License: CC BY-SA 4.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TfidfSubmissionPipeline:
    """Production pipeline for TF-IDF based question-answer matching.
    
    This class encapsulates the entire workflow:
    1. Load and validate training/test datasets
    2. Build TF-IDF representations
    3. Match test questions to training answers
    4. Generate and save submission
    """
    
    def __init__(
        self,
        data_dir: str = "../data",
        output_dir: str = "../",
        ngram_range: Tuple[int, int] = (1, 2),
        top_k: int = 1,
        verbose: bool = True
    ):
        """Initialize the pipeline.
        
        Args:
            data_dir: Path to data directory containing CSV files
            output_dir: Path to output directory for submission CSV
            ngram_range: N-gram range for TfidfVectorizer (default: unigrams + bigrams)
            top_k: Number of top candidates to retrieve (default: 1)
            verbose: Whether to print progress messages
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.ngram_range = ngram_range
        self.top_k = max(1, top_k)  # Ensure at least 1
        self.verbose = verbose
        
        # Data containers
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.train_matrix = None
        self.results: List[Dict] = []
        
        self._log(f"Pipeline initialized (top_k={self.top_k}, ngrams={ngram_range})")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Centralized logging."""
        if self.verbose:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
    
    def load_data(self) -> None:
        """Load and validate training and test datasets.
        
        Raises:
            FileNotFoundError: If required CSV files are missing
            ValueError: If DataFrames lack required columns
        """
        self._log("Loading datasets...")
        
        # Load training data
        train_path = self.data_dir / "train_qa.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        self.train_df = pd.read_csv(train_path)
        self._validate_dataframe(self.train_df, "train_qa.csv", ["question", "reference_answer"])
        self._log(f"Loaded {len(self.train_df)} training Q&A pairs")
        
        # Load test data
        test_path = self.data_dir / "test_questions.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        self.test_df = pd.read_csv(test_path)
        self._validate_dataframe(self.test_df, "test_questions.csv", ["question"])
        self._log(f"Loaded {len(self.test_df)} test questions")
    
    def _validate_dataframe(self, df: pd.DataFrame, name: str, required_cols: List[str]) -> None:
        """Validate DataFrame has required columns and no critical nulls.
        
        Args:
            df: DataFrame to validate
            name: Name of the DataFrame (for logging)
            required_cols: List of required column names
            
        Raises:
            ValueError: If validation fails
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} missing columns: {missing_cols}")
        
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self._log(
                    f"Warning: {name} has {null_count} null values in '{col}'",
                    level="warning"
                )
    
    def build_vectorizer(self) -> None:
        """Build TF-IDF vectorizer and fit on training questions.
        
        Raises:
            RuntimeError: If training data not yet loaded
        """
        if self.train_df is None:
            raise RuntimeError("Training data not loaded. Call load_data() first.")
        
        self._log(f"Building TF-IDF vectorizer (ngram_range={self.ngram_range})...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            lowercase=True,
            stop_words='english',
            max_features=5000  # Limit vocabulary for efficiency
        )
        
        # Fit on training questions only
        self.train_matrix = self.vectorizer.fit_transform(self.train_df["question"])
        self._log(f"Vectorizer fitted. Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
    
    def match_questions(self) -> None:
        """Match test questions to training answers using cosine similarity.
        
        For each test question:
        1. Transform to TF-IDF vector
        2. Compute similarity with all training questions
        3. Find top-k most similar training examples
        4. Retrieve answers from top match(es)
        
        Raises:
            RuntimeError: If vectorizer or test data not ready
        """
        if self.vectorizer is None or self.train_matrix is None:
            raise RuntimeError("Vectorizer not built. Call build_vectorizer() first.")
        if self.test_df is None:
            raise RuntimeError("Test data not loaded. Call load_data() first.")
        
        self._log(f"Matching {len(self.test_df)} test questions to training answers...")
        self.results = []
        
        for idx, test_row in self.test_df.iterrows():
            question_id = test_row.get("QuestionId", idx)
            question_text = test_row["question"]
            
            # Handle topic-based filtering if available
            if "topic" in self.test_df.columns and "topic" in self.train_df.columns:
                test_topic = test_row.get("topic")
                topic_mask = (self.train_df["topic"] == test_topic)
                
                if topic_mask.any():
                    sub_df = self.train_df[topic_mask]
                    sub_matrix = self.train_matrix[topic_mask]
                else:
                    sub_df = self.train_df
                    sub_matrix = self.train_matrix
            else:
                sub_df = self.train_df
                sub_matrix = self.train_matrix
            
            # Transform test question
            test_vec = self.vectorizer.transform([question_text])
            
            # Compute similarities
            sims = cosine_similarity(test_vec, sub_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(sims)[-self.top_k:][::-1]
            
            # Build result entry
            best_idx = top_indices[0]
            best_score = sims[best_idx]
            answer = sub_df.iloc[best_idx]["reference_answer"]
            
            result = {
                "QuestionId": question_id,
                "Answer": answer,
                "Confidence": float(best_score),
                "CandidateCount": len(sub_df)
            }
            
            # Optionally include top-k candidates
            if self.top_k > 1:
                candidates = []
                for rank, top_idx in enumerate(top_indices, 1):
                    candidates.append({
                        "rank": rank,
                        "answer": sub_df.iloc[top_idx]["reference_answer"],
                        "score": float(sims[top_idx])
                    })
                result["TopKCandidates"] = candidates
            
            self.results.append(result)
        
        self._log(f"Matched {len(self.results)} questions")
    
    def save_submission(self, filename: str = "submission.csv") -> Path:
        """Save submission to CSV in Kaggle-required format.
        
        Args:
            filename: Output filename (default: submission.csv)
            
        Returns:
            Path to the saved submission file
            
        Raises:
            RuntimeError: If no results to save
            IOError: If file write fails
        """
        if not self.results:
            raise RuntimeError("No results to save. Call match_questions() first.")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create submission DataFrame with only required columns
        submission_df = pd.DataFrame([
            {"QuestionId": r["QuestionId"], "Answer": r["Answer"]}
            for r in self.results
        ])
        
        output_path = self.output_dir / filename
        submission_df.to_csv(output_path, index=False)
        self._log(f"Submission saved to {output_path}")
        
        # Validate submission format
        self._validate_submission(submission_df, output_path)
        
        return output_path
    
    def _validate_submission(self, df: pd.DataFrame, filepath: Path) -> None:
        """Validate submission format meets Kaggle requirements.
        
        Args:
            df: Submission DataFrame
            filepath: Path to submission file
        """
        # Check required columns
        if set(df.columns) != {"QuestionId", "Answer"}:
            self._log(
                f"Warning: Submission has unexpected columns: {list(df.columns)}",
                level="warning"
            )
        
        # Check for empty answers
        empty_answers = df["Answer"].isnull().sum() + (df["Answer"] == "").sum()
        if empty_answers > 0:
            self._log(
                f"Warning: {empty_answers} empty answers in submission",
                level="warning"
            )
        
        # Check row count
        self._log(f"Submission validation: {len(df)} rows, {len(df.columns)} columns")
    
    def save_detailed_results(self, filename: str = "submission_detailed.json") -> Path:
        """Save detailed results with confidence scores and candidates.
        
        Args:
            filename: Output filename (default: submission_detailed.json)
            
        Returns:
            Path to the saved file
        """
        if not self.results:
            raise RuntimeError("No results to save. Call match_questions() first.")
        
        import json
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self._log(f"Detailed results saved to {output_path}")
        return output_path
    
    def run(self, save_detailed: bool = False) -> Path:
        """Execute the complete pipeline.
        
        Args:
            save_detailed: Whether to save detailed results with scores
            
        Returns:
            Path to submission.csv
        """
        self._log("Starting TF-IDF submission pipeline...")
        self.load_data()
        self.build_vectorizer()
        self.match_questions()
        submission_path = self.save_submission()
        
        if save_detailed:
            self.save_detailed_results()
        
        self._log("Pipeline completed successfully!")
        return submission_path


def resolve_data_path() -> Tuple[str, str]:
    """Resolve data and output paths for Kaggle vs local environments.
    
    Returns:
        Tuple of (data_dir, output_dir)
    """
    # Try Kaggle competition path first
    if os.path.exists("/kaggle/input/competitions/agriculture-climate-slm-challenge/"):
        return (
            "/kaggle/input/competitions/agriculture-climate-slm-challenge/",
            "/kaggle/working/"
        )
    
    # Try generic Kaggle input path
    if os.path.exists("/kaggle/input/"):
        return (
            "/kaggle/input/agriculture-and-climate-slm/",
            "/kaggle/working/"
        )
    
    # Default to local relative paths
    return ("../data/", "../")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TF-IDF based submission generator for AgriSLM benchmark"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top candidates to consider (default: 1)"
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed results with confidence scores"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging (default: True)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir, output_dir = resolve_data_path()
    if args.data_dir:
        data_dir = args.data_dir
    if args.output_dir:
        output_dir = args.output_dir
    
    # Run pipeline
    pipeline = TfidfSubmissionPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        verbose=args.verbose
    )
    
    try:
        pipeline.run(save_detailed=args.save_detailed)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
