"""Unit tests for TF-IDF submission pipeline.

Run with: pytest test_tfidf_submission.py -v
"""

import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from tfidf_submission import TfidfSubmissionPipeline, resolve_data_path


@pytest.fixture
def sample_data():
    """Create minimal sample datasets for testing."""
    train_df = pd.DataFrame({
        "QuestionId": [1, 2, 3],
        "question": [
            "How do I improve soil fertility?",
            "What is crop rotation?",
            "How to treat plant diseases?"
        ],
        "reference_answer": [
            "Use organic compost and manure regularly.",
            "Rotate crops to prevent soil depletion.",
            "Apply fungicides and improve drainage."
        ]
    })
    
    test_df = pd.DataFrame({
        "QuestionId": [10, 11],
        "question": [
            "How can I make my soil better?",
            "What are crop rotation benefits?"
        ]
    })
    
    return train_df, test_df


@pytest.fixture
def temp_data_dir(sample_data):
    """Create temporary directory with sample CSVs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        train_df, test_df = sample_data
        
        train_df.to_csv(tmppath / "train_qa.csv", index=False)
        test_df.to_csv(tmppath / "test_questions.csv", index=False)
        
        yield tmppath


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTfidfSubmissionPipeline:
    """Test suite for TfidfSubmissionPipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = TfidfSubmissionPipeline(verbose=False)
        assert pipeline.train_df is None
        assert pipeline.test_df is None
        assert pipeline.vectorizer is None
        assert pipeline.top_k == 1
    
    def test_load_data_success(self, temp_data_dir, temp_output_dir):
        """Test successful data loading."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            verbose=False
        )
        pipeline.load_data()
        
        assert pipeline.train_df is not None
        assert len(pipeline.train_df) == 3
        assert "question" in pipeline.train_df.columns
        assert "reference_answer" in pipeline.train_df.columns
        
        assert pipeline.test_df is not None
        assert len(pipeline.test_df) == 2
    
    def test_load_data_missing_file(self, temp_output_dir):
        """Test error when data files are missing."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_output_dir),  # Empty directory
            output_dir=str(temp_output_dir),
            verbose=False
        )
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_data()
    
    def test_build_vectorizer(self, temp_data_dir, temp_output_dir):
        """Test TF-IDF vectorizer building."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            verbose=False
        )
        pipeline.load_data()
        pipeline.build_vectorizer()
        
        assert pipeline.vectorizer is not None
        assert pipeline.train_matrix is not None
        assert pipeline.train_matrix.shape[0] == 3  # 3 training questions
    
    def test_build_vectorizer_without_data(self, temp_output_dir):
        """Test error when building vectorizer without loading data."""
        pipeline = TfidfSubmissionPipeline(
            output_dir=str(temp_output_dir),
            verbose=False
        )
        
        with pytest.raises(RuntimeError):
            pipeline.build_vectorizer()
    
    def test_match_questions(self, temp_data_dir, temp_output_dir):
        """Test question matching."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            top_k=1,
            verbose=False
        )
        pipeline.load_data()
        pipeline.build_vectorizer()
        pipeline.match_questions()
        
        assert len(pipeline.results) == 2
        assert all("QuestionId" in r for r in pipeline.results)
        assert all("Answer" in r for r in pipeline.results)
        assert all("Confidence" in r for r in pipeline.results)
        
        # Check that confidence scores are between 0 and 1
        for result in pipeline.results:
            assert 0 <= result["Confidence"] <= 1
    
    def test_match_questions_top_k(self, temp_data_dir, temp_output_dir):
        """Test top-k candidate retrieval."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            top_k=2,
            verbose=False
        )
        pipeline.load_data()
        pipeline.build_vectorizer()
        pipeline.match_questions()
        
        assert len(pipeline.results) == 2
        for result in pipeline.results:
            assert "TopKCandidates" in result
            assert len(result["TopKCandidates"]) == 2
    
    def test_save_submission(self, temp_data_dir, temp_output_dir):
        """Test submission file generation."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            verbose=False
        )
        pipeline.load_data()
        pipeline.build_vectorizer()
        pipeline.match_questions()
        
        output_path = pipeline.save_submission()
        
        assert output_path.exists()
        submission_df = pd.read_csv(output_path)
        assert len(submission_df) == 2
        assert set(submission_df.columns) == {"QuestionId", "Answer"}
    
    def test_save_detailed_results(self, temp_data_dir, temp_output_dir):
        """Test detailed results JSON export."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            verbose=False
        )
        pipeline.load_data()
        pipeline.build_vectorizer()
        pipeline.match_questions()
        
        output_path = pipeline.save_detailed_results()
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
    
    def test_run_complete_pipeline(self, temp_data_dir, temp_output_dir):
        """Test complete end-to-end pipeline execution."""
        pipeline = TfidfSubmissionPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            verbose=False
        )
        
        output_path = pipeline.run()
        
        assert output_path.exists()
        submission_df = pd.read_csv(output_path)
        assert len(submission_df) == 2
        assert "Answer" in submission_df.columns
    
    def test_path_resolution(self):
        """Test Kaggle vs local path resolution."""
        data_dir, output_dir = resolve_data_path()
        assert isinstance(data_dir, str)
        assert isinstance(output_dir, str)


class TestDataValidation:
    """Test data validation functions."""
    
    def test_validate_dataframe_success(self, sample_data, temp_output_dir):
        """Test successful DataFrame validation."""
        train_df, _ = sample_data
        pipeline = TfidfSubmissionPipeline(
            output_dir=str(temp_output_dir),
            verbose=False
        )
        
        # Should not raise
        pipeline._validate_dataframe(train_df, "test.csv", ["question", "reference_answer"])
    
    def test_validate_dataframe_missing_columns(self, sample_data, temp_output_dir):
        """Test validation with missing columns."""
        train_df, _ = sample_data
        pipeline = TfidfSubmissionPipeline(
            output_dir=str(temp_output_dir),
            verbose=False
        )
        
        with pytest.raises(ValueError):
            pipeline._validate_dataframe(train_df, "test.csv", ["nonexistent_col"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
