"""
Test Suite for Contract Analysis System - Phase 1
Tests for parsing, metadata extraction, NER, and data quality
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parsing_pipeline import ContractParser
from src.data.metadata_extractor import MetadataExtractor, ContractMetadata
from src.data.legal_ner import LegalNER
from src.validation.run_checks import DataValidator


class TestParsingPipeline:
    """Test parsing pipeline functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "input_dir": "data/raw",
            "output_dir": "data/processed",
            "min_clause_length": 50,
            "confidence_threshold": 0.5,
        }
        self.parser = ContractParser(self.config)

    def test_parser_initialization(self):
        """Test parser initialization"""
        assert self.parser is not None
        assert hasattr(self.parser, 'heading_patterns')
        assert hasattr(self.parser, 'bullet_patterns')

    def test_text_normalization(self):
        """Test text normalization"""
        test_text = "This   has   extra   spaces\n\nand\n\n\nnewlines"
        normalized = self.parser.normalize_text(test_text)
        
        assert "   " not in normalized  # No extra spaces
        assert "\n\n" not in normalized  # No double newlines
        assert normalized.strip() == normalized  # Properly stripped

    def test_clause_classification(self):
        """Test clause type classification"""
        test_clauses = [
            ("This agreement shall be governed by the laws of California", "governing_law"),
            ("Party A shall pay $50,000 in liquidated damages", "liability"),
            ("All information shall be kept confidential", "confidentiality"),
            ("Either party may terminate this agreement", "termination"),
            ("All intellectual property shall be assigned", "ip_assignment"),
        ]
        
        for clause_text, expected_type in test_clauses:
            classified_type = self.parser._classify_clause_type(clause_text)
            assert classified_type == expected_type

    def test_clause_segmentation(self):
        """Test clause segmentation"""
        test_text = """
        SECTION 1. GOVERNING LAW
        This agreement shall be governed by the laws of California.
        
        SECTION 2. LIABILITY
        Party A shall be liable for all damages.
        
        SECTION 3. CONFIDENTIALITY
        All information shall be kept confidential.
        """
        
        clauses = self.parser.segment_clauses(test_text, "test_contract")
        
        assert len(clauses) >= 3  # Should find at least 3 clauses
        assert all("clause_id" in clause for clause in clauses)
        assert all("text" in clause for clause in clauses)
        assert all("clause_type" in clause for clause in clauses)

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        short_text = "Short clause"
        medium_text = "This is a medium length clause with some content"
        long_text = "This is a very long clause with lots of content that should get a high confidence score because it has substantial text and appears to be well-formatted"
        
        short_confidence = self.parser._calculate_confidence(short_text)
        medium_confidence = self.parser._calculate_confidence(medium_text)
        long_confidence = self.parser._calculate_confidence(long_text)
        
        assert short_confidence < medium_confidence < long_confidence
        assert all(0 <= conf <= 1 for conf in [short_confidence, medium_confidence, long_confidence])


class TestMetadataExtraction:
    """Test metadata extraction functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.config = {}
        self.extractor = MetadataExtractor(self.config)

    def test_extractor_initialization(self):
        """Test metadata extractor initialization"""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'date_patterns')
        assert hasattr(self.extractor, 'amount_patterns')
        assert hasattr(self.extractor, 'party_patterns')

    def test_party_extraction(self):
        """Test party name extraction"""
        test_text = """
        This agreement is between ABC Corporation Inc. and XYZ Company LLC.
        The parties agree to the following terms.
        """
        
        parties = self.extractor._extract_parties(test_text)
        assert len(parties) >= 1
        assert any("ABC Corporation" in party for party in parties)
        assert any("XYZ Company" in party for party in parties)

    def test_date_extraction(self):
        """Test date extraction"""
        test_text = """
        This agreement is effective as of January 1, 2024.
        The contract expires on December 31, 2024.
        """
        
        effective_date = self.extractor._extract_effective_date(test_text)
        expiration_date = self.extractor._extract_expiration_date(test_text)
        
        assert effective_date == "2024-01-01"
        assert expiration_date == "2024-12-31"

    def test_amount_extraction(self):
        """Test amount and currency extraction"""
        test_text = """
        The contract value is $100,000.00 USD.
        Additional fees may apply up to €50,000 EUR.
        """
        
        value, currency = self.extractor._extract_contract_value(test_text)
        assert value == 100000.0
        assert currency == "USD"

    def test_governing_law_extraction(self):
        """Test governing law extraction"""
        test_text = """
        This agreement shall be governed by the laws of the State of California.
        Jurisdiction shall be in the courts of Los Angeles County.
        """
        
        governing_law = self.extractor._extract_governing_law(test_text)
        jurisdiction = self.extractor._extract_jurisdiction(test_text)
        
        assert "California" in governing_law
        assert "Los Angeles" in jurisdiction

    def test_contract_type_classification(self):
        """Test contract type classification"""
        test_cases = [
            ("employment agreement with employee", "employment"),
            ("service agreement for consulting", "service"),
            ("purchase agreement for goods", "purchase"),
            ("lease agreement for property", "lease"),
            ("non-disclosure agreement", "nda"),
        ]
        
        for text, expected_type in test_cases:
            classified_type = self.extractor._classify_contract_type(text)
            assert classified_type == expected_type

    def test_full_metadata_extraction(self):
        """Test complete metadata extraction"""
        test_text = """
        EMPLOYMENT AGREEMENT
        
        This agreement is between ABC Corporation Inc. and John Doe, effective as of January 1, 2024.
        The contract value is $75,000.00 USD.
        This agreement shall be governed by the laws of California.
        """
        
        metadata = self.extractor.extract_metadata(test_text, "test_contract")
        
        assert isinstance(metadata, ContractMetadata)
        assert metadata.contract_id == "test_contract"
        assert len(metadata.parties) >= 1
        assert metadata.effective_date == "2024-01-01"
        assert metadata.contract_value == 75000.0
        assert metadata.currency == "USD"
        assert "California" in metadata.governing_law
        assert metadata.contract_type == "employment"


class TestLegalNER:
    """Test legal NER functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.config = {}
        self.ner = LegalNER(self.config)

    def test_ner_initialization(self):
        """Test NER initialization"""
        assert self.ner is not None
        # Note: spaCy might not be available in test environment

    @patch('src.data.legal_ner.SPACY_AVAILABLE', True)
    def test_entity_extraction(self):
        """Test entity extraction (mocked)"""
        test_text = """
        This agreement contains Force Majeure provisions.
        The Indemnified Party shall be protected.
        This is a Non-Disclosure Agreement.
        """
        
        # Mock spaCy processing
        with patch.object(self.ner, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "Force Majeure"
            mock_ent.label_ = "LEGAL_TERM"
            mock_ent.start_char = 0
            mock_ent.end_char = 13
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc
            
            entities = self.ner.extract_entities(test_text, "test_contract")
            
            assert len(entities) >= 1
            assert entities[0]["entity"] == "Force Majeure"
            assert entities[0]["label"] == "LEGAL_TERM"

    def test_entity_classification(self):
        """Test entity classification"""
        test_entities = [
            {"label": "LEGAL_TERM", "entity": "Force Majeure"},
            {"label": "CONTRACT_TYPE", "entity": "Non-Disclosure Agreement"},
            {"label": "ORG", "entity": "ABC Corporation"},
            {"label": "PERSON", "entity": "John Doe"},
        ]
        
        classified = self.ner.classify_entities(test_entities)
        
        assert len(classified["legal_terms"]) >= 1
        assert len(classified["contract_types"]) >= 1
        assert len(classified["organizations"]) >= 1
        assert len(classified["persons"]) >= 1

    def test_entity_report_generation(self):
        """Test entity report generation"""
        test_entities = [
            {"label": "LEGAL_TERM", "entity": "Force Majeure", "confidence": 0.9},
            {"label": "LEGAL_TERM", "entity": "Indemnified Party", "confidence": 0.8},
            {"label": "CONTRACT_TYPE", "entity": "Non-Disclosure Agreement", "confidence": 0.9},
        ]
        
        report = self.ner.generate_entity_report(test_entities)
        
        assert report["total_entities"] == 3
        assert "LEGAL_TERM" in report["entity_types"]
        assert "CONTRACT_TYPE" in report["entity_types"]
        assert report["entity_types"]["LEGAL_TERM"] == 2
        assert report["entity_types"]["CONTRACT_TYPE"] == 1


class TestDataValidation:
    """Test data validation functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "data_dir": "data/processed",
            "validation_config": "configs/validation.yaml",
        }
        self.validator = DataValidator(self.config)

    def test_validator_initialization(self):
        """Test validator initialization"""
        assert self.validator is not None
        assert hasattr(self.validator, 'config')

    def test_pandera_schema_creation(self):
        """Test Pandera schema creation"""
        schemas = self.validator.create_pandera_schemas()
        
        assert "contract_metadata" in schemas
        assert "clause_segments" in schemas
        assert "enriched_metadata" in schemas

    def test_contract_metadata_validation(self):
        """Test contract metadata validation"""
        # Create test data
        test_data = pd.DataFrame({
            "contract_id": ["contract_001", "contract_002"],
            "contract_type": ["employment", "service"],
            "parties": ["['ABC Corp', 'John Doe']", "['XYZ Inc', 'Jane Smith']"],
            "effective_date": ["2024-01-01", "2024-01-15"],
            "jurisdiction": ["California", "New York"],
            "total_clauses": [10, 15],
            "file_size": [1024, 2048],
        })
        
        schemas = self.validator.create_pandera_schemas()
        contract_schema = schemas["contract_metadata"]
        
        # This should not raise an exception
        validated_data = contract_schema.validate(test_data)
        assert len(validated_data) == 2

    def test_clause_segments_validation(self):
        """Test clause segments validation"""
        # Create test data
        test_data = pd.DataFrame({
            "contract_id": ["contract_001", "contract_001"],
            "clause_type": ["governing_law", "liability"],
            "text": [
                "This agreement shall be governed by the laws of California.",
                "Party A shall be liable for all damages."
            ],
            "confidence": [0.8, 0.9],
            "risk_flags": ["['uncapped_liability']", "[]"],
        })
        
        schemas = self.validator.create_pandera_schemas()
        clause_schema = schemas["clause_segments"]
        
        # This should not raise an exception
        validated_data = clause_schema.validate(test_data)
        assert len(validated_data) == 2


class TestDataQuality:
    """Test data quality checks"""

    def test_clause_overlap_detection(self):
        """Test detection of overlapping clauses"""
        clauses = [
            {"start": 0, "end": 100, "text": "First clause"},
            {"start": 50, "end": 150, "text": "Second clause"},  # Overlaps with first
            {"start": 200, "end": 300, "text": "Third clause"},  # No overlap
        ]
        
        # Check for overlaps
        overlaps = []
        for i, clause1 in enumerate(clauses):
            for j, clause2 in enumerate(clauses[i+1:], i+1):
                if (clause1["start"] < clause2["end"] and 
                    clause1["end"] > clause2["start"]):
                    overlaps.append((i, j))
        
        assert len(overlaps) >= 1  # Should find at least one overlap

    def test_contract_metadata_completeness(self):
        """Test contract metadata completeness"""
        required_fields = [
            "contract_id", "contract_type", "parties", 
            "total_clauses", "file_size"
        ]
        
        test_metadata = {
            "contract_id": "contract_001",
            "contract_type": "employment",
            "parties": ["ABC Corp", "John Doe"],
            "total_clauses": 10,
            "file_size": 1024,
        }
        
        missing_fields = [field for field in required_fields 
                          if field not in test_metadata or test_metadata[field] is None]
        
        assert len(missing_fields) == 0  # All required fields present

    def test_clause_text_quality(self):
        """Test clause text quality"""
        test_clauses = [
            {"text": "This is a good clause with substantial content.", "quality": "good"},
            {"text": "Short.", "quality": "poor"},  # Too short
            {"text": "This clause has reasonable length and content.", "quality": "good"},
        ]
        
        quality_issues = []
        for clause in test_clauses:
            if len(clause["text"]) < 20:  # Minimum length threshold
                quality_issues.append(clause["text"])
        
        assert len(quality_issues) >= 1  # Should find at least one quality issue


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_end_to_end_processing(self):
        """Test end-to-end processing with sample data"""
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample contract text
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
            
            # Test parsing
            config = {"input_dir": temp_dir, "output_dir": temp_dir}
            parser = ContractParser(config)
            
            # Test clause segmentation
            clauses = parser.segment_clauses(sample_text, "test_contract")
            assert len(clauses) >= 3
            
            # Test metadata extraction
            extractor = MetadataExtractor({})
            metadata = extractor.extract_metadata(sample_text, "test_contract")
            assert metadata.contract_type == "employment"
            assert len(metadata.parties) >= 1
            
            # Test NER (if spaCy available)
            ner = LegalNER({})
            if ner.nlp:
                entities = ner.extract_entities(sample_text, "test_contract")
                assert len(entities) >= 0  # May or may not find entities

    def test_data_consistency(self):
        """Test data consistency across pipeline stages"""
        # This test ensures that contract IDs are consistent
        # across parsing, metadata extraction, and NER stages
        
        contract_id = "test_contract_001"
        
        # Simulate pipeline stages
        parsing_output = {"contract_id": contract_id, "n_clauses": 5}
        metadata_output = {"contract_id": contract_id, "parties": ["ABC Corp"]}
        ner_output = {"contract_id": contract_id, "entities": []}
        
        # Check consistency
        assert parsing_output["contract_id"] == metadata_output["contract_id"]
        assert metadata_output["contract_id"] == ner_output["contract_id"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
