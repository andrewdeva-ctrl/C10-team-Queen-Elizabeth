import csv
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Core schemas required by the benchmark competition datasets
CORE_TRAIN_COLS = ["QuestionId", "question", "reference_answer"]
CORE_TEST_COLS = ["QuestionId", "question"]

class DataValidator:
    """Validates data files for consistency, structure, and required schemas."""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.validation_report = {}
        self.errors = []
        self.warnings = []

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print("🔍 Starting AgriSLM Data Validation...\n")
        self.validate_csv_files()
        self.validate_jsonl_files()
        self.validate_file_correspondence()
        self.validate_core_schemas()
        self.generate_report()
        return self.validation_report

    def validate_csv_files(self) -> None:
        """Validate all CSV files for basic read integrity."""
        print("📊 Validating CSV files...")
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print("  ⚠️ No CSV files found in data directory.")
            return

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                self.validation_report[csv_file.name] = {
                    "status": "✅ Valid",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "shape": df.shape
                }
                print(f"  ✅ {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                self.errors.append(f"CSV Error in {csv_file.name}: {str(e)}")
                print(f"  ❌ {csv_file.name}: {str(e)}")

    def validate_jsonl_files(self) -> None:
        """Validate all JSONL files."""
        print("\n📋 Validating JSONL files...")
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            try:
                lines = []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)
                            lines.append(line)
                self.validation_report[jsonl_file.name] = {
                    "status": "✅ Valid",
                    "lines": len(lines)
                }
                print(f"  ✅ {jsonl_file.name}: {len(lines)} valid JSON lines")
            except Exception as e:
                self.errors.append(f"JSONL Error in {jsonl_file.name}: {str(e)}")
                print(f"  ❌ {jsonl_file.name}: {str(e)}")

    def validate_file_correspondence(self) -> None:
        """Validate existence of critical benchmark files."""
        print("\n🔗 Checking file presence...")
        expected_files = ['train_qa.csv', 'test_questions.csv']
        for fname in expected_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                print(f"  ✅ Found benchmark file: {fname}")
            else:
                self.warnings.append(f"Benchmark file missing: {fname}")
                print(f"  ⚠️ Missing benchmark file: {fname}")

    def validate_core_schemas(self) -> None:
        """Validate specific column requirements for training and testing datasets."""
        print("\n🔎 Verifying core schemas...")
        train_path = self.data_dir / "train_qa.csv"
        test_path = self.data_dir / "test_questions.csv"

        if train_path.exists():
            train_df = pd.read_csv(train_path)
            for col in CORE_TRAIN_COLS:
                if col not in train_df.columns:
                    self.errors.append(f"train_qa.csv missing core column: {col}")
                    print(f"  ❌ train_qa.csv missing core column: {col}")
            if train_df["question"].isnull().sum() > 0:
                self.errors.append("Null question values found in train_qa.csv")

        if test_path.exists():
            test_df = pd.read_csv(test_path)
            for col in CORE_TEST_COLS:
                if col not in test_df.columns:
                    self.errors.append(f"test_questions.csv missing core column: {col}")
                    print(f"  ❌ test_questions.csv missing core column: {col}")
            if test_df["question"].isnull().sum() > 0:
                self.errors.append("Null question values found in test_questions.csv")

    def generate_report(self) -> None:
        """Print summary report of errors and warnings."""
        print("\n" + "="*40)
        print("📋 VALIDATION SUMMARY REPORT")
        print("="*40)
        if self.errors:
            print(f"❌ Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")
        else:
            print("✅ No critical errors found.")

        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")
        print("="*40 + "\n")


if __name__ == "__main__":
    validator = DataValidator()
    validator.validate_all()
