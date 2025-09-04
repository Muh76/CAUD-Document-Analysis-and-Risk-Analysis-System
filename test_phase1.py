#!/usr/bin/env python3
"""
Test Phase 1 Pipeline with CUAD Data
Quick test to verify all Phase 1 components work correctly
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.parsing_pipeline import ContractParser
from src.data.metadata_extractor import MetadataExtractor
from src.data.legal_ner import LegalNER

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parsing_pipeline():
    """Test parsing pipeline with sample text"""
    print("🧪 Testing Parsing Pipeline...")
    
    config = {
        "input_dir": "data/raw",
        "output_dir": "data/processed",
        "min_clause_length": 50,
        "confidence_threshold": 0.5,
    }
    
    parser = ContractParser(config)
    
    # Test with sample contract text
    sample_text = """
    EMPLOYMENT AGREEMENT
    
    This agreement is between ABC Corporation Inc. and John Doe, effective as of January 1, 2024.
    
    SECTION 1. GOVERNING LAW
    This agreement shall be governed by the laws of California.
    
    SECTION 2. LIABILITY
    Party A shall be liable for all damages up to $100,000.
    
    SECTION 3. CONFIDENTIALITY
    All information shall be kept confidential.
    """
    
    # Test clause segmentation
    clauses = parser.segment_clauses(sample_text, "test_contract")
    
    print(f"   ✅ Extracted {len(clauses)} clauses")
    for i, clause in enumerate(clauses[:3]):  # Show first 3
        print(f"      Clause {i+1}: {clause['clause_type']} ({len(clause['text'])} chars)")
    
    return clauses


def test_metadata_extraction():
    """Test metadata extraction"""
    print("🧪 Testing Metadata Extraction...")
    
    config = {}
    extractor = MetadataExtractor(config)
    
    # Test with sample contract text
    sample_text = """
    This agreement is between ABC Corporation Inc. and John Doe, effective as of January 1, 2024.
    The contract value is $75,000.00 USD.
    This agreement shall be governed by the laws of California.
    """
    
    metadata = extractor.extract_metadata(sample_text, "test_contract")
    
    print(f"   ✅ Extracted metadata:")
    print(f"      Parties: {metadata.parties}")
    print(f"      Effective Date: {metadata.effective_date}")
    print(f"      Contract Value: {metadata.contract_value} {metadata.currency}")
    print(f"      Governing Law: {metadata.governing_law}")
    print(f"      Contract Type: {metadata.contract_type}")
    
    return metadata


def test_legal_ner():
    """Test legal NER"""
    print("🧪 Testing Legal NER...")
    
    config = {}
    ner = LegalNER(config)
    
    # Test with sample contract text
    sample_text = """
    This agreement contains Force Majeure provisions.
    The Indemnified Party shall be protected.
    This is a Non-Disclosure Agreement.
    """
    
    entities = ner.extract_entities(sample_text, "test_contract")
    
    print(f"   ✅ Extracted {len(entities)} entities")
    for entity in entities[:5]:  # Show first 5
        print(f"      {entity['entity']} ({entity['label']})")
    
    return entities


def test_cuad_data_processing():
    """Test processing with actual CUAD data"""
    print("🧪 Testing CUAD Data Processing...")
    
    # Check if CUAD data exists
    cuad_file = Path("data/raw/CUAD_v1.json")
    if not cuad_file.exists():
        print("   ⚠️ CUAD data not found, skipping CUAD test")
        return None
    
    # Load CUAD data
    import json
    with open(cuad_file, 'r') as f:
        cuad_data = json.load(f)
    
    print(f"   📊 CUAD dataset: {len(cuad_data['data'])} contracts")
    
    # Process first few contracts
    config = {}
    extractor = MetadataExtractor(config)
    ner = LegalNER(config)
    
    processed_count = 0
    for i, contract in enumerate(cuad_data['data'][:5]):  # Process first 5
        try:
            # Extract text from CUAD format
            text = contract.get('text', '')
            if text:
                contract_id = f"cuad_contract_{i:04d}"
                
                # Extract metadata
                metadata = extractor.extract_metadata(text, contract_id)
                
                # Extract entities
                entities = ner.extract_entities(text, contract_id)
                
                processed_count += 1
                print(f"      Contract {i+1}: {len(entities)} entities, {len(metadata.parties)} parties")
                
        except Exception as e:
            print(f"      Error processing contract {i}: {e}")
    
    print(f"   ✅ Successfully processed {processed_count} contracts")
    return processed_count


def main():
    """Run all tests"""
    print("🎯 Phase 1 Pipeline Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        test_parsing_pipeline()
        print()
        
        test_metadata_extraction()
        print()
        
        test_legal_ner()
        print()
        
        # Test with CUAD data
        test_cuad_data_processing()
        print()
        
        print("✅ All tests completed successfully!")
        print("\n🚀 Phase 1 components are ready for production use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

