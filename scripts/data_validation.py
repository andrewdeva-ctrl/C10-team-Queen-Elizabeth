"""
Data Validation Script for AgriSLM Project
This script validates the correspondence and integrity of data files.
"""

import os
import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

class DataValidator:
    """Validates data files for consistency and structure."""
    
    def __init__(self, data_dir: str = "data"):
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
        self.validate_data_consistency()
        
        self.generate_report()
        return self.validation_report
    
    def validate_csv_files(self) -> None:
        """Validate all CSV files."""
        print("📊 Validating CSV files...")
        csv_files = list(self.data_dir.glob("*.csv"))
        
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
                with open(jsonl_file, 'r') as f:
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
        """Validate correspondence between related data files."""
        print("\n🔗 Checking file correspondence...")
        
        try:
            # Check if benchmark files exist
            benchmark_files = [
                'documents.csv',
                'train_qa.csv',
                'test_questions.csv',
                'baseline_submission.csv',
                'dataset-metadata.json'
            ]
            
            missing_files = [f for f in benchmark_files if not (self.data_dir / f).exists()]
            if missing_files:
                self.warnings.append(f"Missing benchmark files: {missing_files}")
                print(f"  ⚠️  Missing files: {missing_files}")
            else:
                print(f"  ✅ All benchmark files present")
            
            # Check if demo data files exist
            demo_files = [
                'agrislm_team_demo_dataset.csv',
                'agrislm_team_demo_dataset.jsonl'
            ]
            
            missing_demo = [f for f in demo_files if not (self.data_dir / f).exists()]
            if missing_demo:
                self.warnings.append(f"Missing demo data files: {missing_demo}")
                print(f"  ⚠️  Missing demo files: {missing_demo}")
            else:
                print(f"  ✅ All demo data files present")
        
        except Exception as e:
            self.errors.append(f"Correspondence check error: {str(e)}")
            print(f"  ❌ Error: {str(e)}")
    
    def validate_data_consistency(self) -> None:
        """Validate consistency between CSV and JSONL formats."""
        print("\n🔄 Validating data consistency...")
        
        try:
            csv_path = self.data_dir / "agrislm_team_demo_dataset.csv"
            jsonl_path = self.data_dir / "agrislm_team_demo_dataset.jsonl"
            
            if csv_path.exists() and jsonl_path.exists():
                csv_df = pd.read_csv(csv_path)
                
                jsonl_lines = []
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            jsonl_lines.append(json.loads(line))
                
                if len(csv_df) == len(jsonl_lines):
                    print(f"  ✅ Row count matches: {len(csv_df)} rows in both formats")
                else:
                    self.warnings.append(
                        f"Row mismatch: CSV has {len(csv_df)} rows, JSONL has {len(jsonl_lines)} rows"
                    )
                    print(f"  ⚠️  Row count mismatch: CSV ({len(csv_df)}) vs JSONL ({len(jsonl_lines)})")
        
        except Exception as e:
            self.warnings.append(f"Consistency check error: {str(e)}")
            print(f"  ⚠️  Warning: {str(e)}")
    
    def generate_report(self) -> None:
        """Generate and print validation report."""
        print("\n" + "="*60)
        print("📋 VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ Files Validated: {len(self.validation_report)}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors:
            print("\n🎉 All validations passed!")
        
        print("\n" + "="*60)


def main():
    """Run data validation."""
    validator = DataValidator()
    report = validator.validate_all()
    
    # Save report to JSON
    report_path = Path("data_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validator.validation_report, f, indent=2)
    
    print(f"\n📁 Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()
