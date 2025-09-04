#!/usr/bin/env python3
"""
Test Enhanced Data Validation System
Creates sample data and tests both Pandera and Great Expectations
"""

import os
import sys
import logging
import json
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for testing validation"""
    print("📄 Creating sample data for validation testing...")
    
    # Create sample contracts data
    contracts_data = [
        {
            "contract_id": "contract_001",
            "file_name": "sample_contract_001.pdf",
            "contract_type": "employment",
            "parties": "['ABC Corp', 'John Doe']",
            "effective_date": "2024-01-01",
            "jurisdiction": "California",
            "total_clauses": 15,
            "file_size": 1024,
            "parsed_ok": 1,
        },
        {
            "contract_id": "contract_002",
            "file_name": "sample_contract_002.pdf",
            "contract_type": "service",
            "parties": "['XYZ Inc', 'Jane Smith']",
            "effective_date": "2024-01-15",
            "jurisdiction": "New York",
            "total_clauses": 12,
            "file_size": 2048,
            "parsed_ok": 1,
        },
        {
            "contract_id": "contract_003",
            "file_name": "sample_contract_003.pdf",
            "contract_type": "nda",
            "parties": "['Tech Corp', 'Consultant LLC']",
            "effective_date": "2024-02-01",
            "jurisdiction": "Delaware",
            "total_clauses": 8,
            "file_size": 512,
            "parsed_ok": 1,
        },
    ]
    
    # Create sample clauses data
    clauses_data = [
        {
            "clause_id": "contract_001_clause_001",
            "contract_id": "contract_001",
            "text": "This agreement shall be governed by the laws of California.",
            "clause_type": "governing_law",
            "confidence": 0.9,
            "text_length": 65,
        },
        {
            "clause_id": "contract_001_clause_002",
            "contract_id": "contract_001",
            "text": "Party A shall be liable for all damages up to $100,000.",
            "clause_type": "liability",
            "confidence": 0.8,
            "text_length": 58,
        },
        {
            "clause_id": "contract_002_clause_001",
            "contract_id": "contract_002",
            "text": "All information shall be kept confidential.",
            "clause_type": "confidentiality",
            "confidence": 0.95,
            "text_length": 45,
        },
        {
            "clause_id": "contract_003_clause_001",
            "contract_id": "contract_003",
            "text": "Either party may terminate this agreement with 30 days notice.",
            "clause_type": "termination",
            "confidence": 0.85,
            "text_length": 72,
        },
    ]
    
    # Create DataFrames
    contracts_df = pd.DataFrame(contracts_data)
    clauses_df = pd.DataFrame(clauses_data)
    
    # Save to files
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    contracts_df.to_csv("data/processed/contracts_index.csv", index=False)
    clauses_df.to_csv("data/processed/clause_spans.csv", index=False)
    
    print(f"   ✅ Created {len(contracts_data)} sample contracts")
    print(f"   ✅ Created {len(clauses_data)} sample clauses")
    
    return contracts_df, clauses_df


def test_great_expectations():
    """Test Great Expectations validation"""
    print("\n🏷️ Testing Great Expectations Validation...")
    
    try:
        from src.validation.great_expectations_validator import GreatExpectationsValidator
        
        config = {}
        validator = GreatExpectationsValidator(config)
        
        # Run validation
        report = validator.run_comprehensive_validation("data/processed")
        
        print("   ✅ Great Expectations validation completed")
        print(f"   📄 Report saved to: reports/great_expectations_report.json")
        
        return report
        
    except Exception as e:
        print(f"   ❌ Great Expectations validation failed: {e}")
        return None


def test_pandera():
    """Test Pandera validation"""
    print("\n📊 Testing Pandera Validation...")
    
    try:
        from src.validation.run_checks import DataValidator
        
        config = {"data_dir": "data/processed"}
        validator = DataValidator(config)
        
        # Run validation
        results = validator.validate_all()
        
        print("   ✅ Pandera validation completed")
        print(f"   📊 Results: {results}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Pandera validation failed: {e}")
        return None


def main():
    """Main test function"""
    print("🎯 Testing Enhanced Data Validation System")
    print("=" * 50)
    
    # Create sample data
    contracts_df, clauses_df = create_sample_data()
    
    # Test Great Expectations
    ge_results = test_great_expectations()
    
    # Test Pandera
    pandera_results = test_pandera()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION TEST SUMMARY")
    print("=" * 50)
    
    if ge_results:
        print("✅ Great Expectations: Working")
    else:
        print("❌ Great Expectations: Failed")
    
    if pandera_results:
        print("✅ Pandera: Working")
    else:
        print("❌ Pandera: Failed")
    
    print("\n📁 Generated Files:")
    files = [
        "data/processed/contracts_index.csv",
        "data/processed/clause_spans.csv",
        "reports/great_expectations_report.json",
    ]
    
    for file_path in files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    print("\n🚀 Next Steps:")
    if ge_results and pandera_results:
        print("   • Both validation systems are working!")
        print("   • Ready for Phase 2: Model Development")
    else:
        print("   • Fix validation system issues")
        print("   • Ensure all dependencies are properly installed")
    
    print("=" * 50)


if __name__ == "__main__":
    main()

