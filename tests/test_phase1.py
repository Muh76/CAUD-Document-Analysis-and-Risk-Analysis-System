"""
Tests for Phase 1 - Data Pipeline and Baseline Models
Comprehensive test suite for data processing, validation, and baseline models
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.append('src')

from data.pipeline import ContractDataPipeline, ContractMetadata, ClauseSegment
from models.baseline_models import BaselineClauseClassifier, KeywordRiskScorer, BaselineEvaluator

class TestContractDataPipeline:
    """Test suite for data pipeline functionality"""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            'cuad_file_path': 'data/raw/CUAD_v1.json',
            'output_dir': 'data/processed',
            'validation_enabled': True,
            'logging_level': 'INFO'
        }
    
    @pytest.fixture
    def sample_contract_text(self):
        return """
        AGREEMENT
        
        This Agreement is entered into on January 1, 2024, between ABC Corporation ("Company") 
        and John Doe ("Employee").
        
        Section 1. Employment
        The Company agrees to employ the Employee as a Software Engineer.
        
        Section 2. Compensation
        The Employee shall receive a salary of $100,000 per year.
        
        Section 3. Termination
        Either party may terminate this agreement with 30 days written notice.
        
        Section 4. Confidentiality
        The Employee agrees to maintain confidentiality of company information.
        
        Section 5. Governing Law
        This agreement shall be governed by the laws of California.
        """
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        return ContractDataPipeline(pipeline_config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.nlp is not None
        assert pipeline.tokenizer is not None
    
    def test_extract_metadata(self, pipeline, sample_contract_text):
        """Test metadata extraction"""
        metadata = pipeline.extract_metadata(sample_contract_text, "test_contract_001")
        
        assert isinstance(metadata, ContractMetadata)
        assert metadata.contract_id == "test_contract_001"
        assert metadata.contract_type == "EMPLOYMENT"
        assert len(metadata.parties) > 0
        assert metadata.total_clauses > 0
        assert metadata.file_size > 0
    
    def test_segment_clauses(self, pipeline, sample_contract_text):
        """Test clause segmentation"""
        clauses = pipeline.segment_clauses(sample_contract_text, "test_contract_001")
        
        assert isinstance(clauses, list)
        assert len(clauses) > 0
        
        for clause in clauses:
            assert isinstance(clause, ClauseSegment)
            assert clause.clause_id is not None
            assert len(clause.text) > 0
            assert clause.start_position >= 0
            assert clause.end_position > clause.start_position
            assert 0 <= clause.confidence <= 1
    
    def test_classify_contract_type(self, pipeline):
        """Test contract type classification"""
        # Test employment contract
        employment_text = "This employment agreement..."
        assert pipeline._classify_contract_type(employment_text) == "EMPLOYMENT"
        
        # Test NDA
        nda_text = "This non-disclosure agreement..."
        assert pipeline._classify_contract_type(nda_text) == "NDA"
        
        # Test unknown type
        unknown_text = "This is some random text..."
        assert pipeline._classify_contract_type(unknown_text) is None
    
    def test_classify_clause_type(self, pipeline):
        """Test clause type classification"""
        # Test liability clause
        liability_text = "The party shall indemnify..."
        assert pipeline._classify_clause_type("Liability", liability_text) == "liability"
        
        # Test termination clause
        termination_text = "Either party may terminate..."
        assert pipeline._classify_clause_type("Termination", termination_text) == "termination"
        
        # Test unknown clause type
        unknown_text = "This is some random clause..."
        assert pipeline._classify_clause_type("Unknown", unknown_text) is None
    
    def test_identify_risk_flags(self, pipeline):
        """Test risk flag identification"""
        # Test high risk clause
        high_risk_text = "The party shall have unlimited liability for all damages."
        risk_flags = pipeline._identify_risk_flags(high_risk_text)
        assert "uncapped_liability" in risk_flags
        
        # Test medium risk clause
        medium_risk_text = "The party shall use reasonable efforts."
        risk_flags = pipeline._identify_risk_flags(medium_risk_text)
        assert "vague_terms" in risk_flags
        
        # Test low risk clause
        low_risk_text = "Both parties agree to standard terms."
        risk_flags = pipeline._identify_risk_flags(low_risk_text)
        assert len(risk_flags) == 0
    
    def test_process_contract(self, pipeline, sample_contract_text):
        """Test complete contract processing"""
        metadata, clauses = pipeline.process_contract(sample_contract_text, "test_contract_001")
        
        assert isinstance(metadata, ContractMetadata)
        assert isinstance(clauses, list)
        assert len(clauses) > 0
        assert metadata.total_clauses == len(clauses)
    
    def test_create_data_contracts(self, pipeline):
        """Test data contract creation"""
        data_contracts = pipeline.create_data_contracts()
        
        assert isinstance(data_contracts, dict)
        assert 'contract_metadata' in data_contracts
        assert 'clause_segments' in data_contracts
        assert 'required_fields' in data_contracts['contract_metadata']
        assert 'validation_rules' in data_contracts['contract_metadata']
    
    def test_generate_data_report(self, pipeline):
        """Test data report generation"""
        # Create sample metadata and clauses
        metadata_list = [
            ContractMetadata(
                contract_id="test_001",
                contract_type="EMPLOYMENT",
                parties=["ABC Corp", "John Doe"],
                effective_date="2024-01-01",
                expiration_date=None,
                jurisdiction="California",
                governing_law="California Law",
                total_clauses=5,
                file_size=1000,
                processing_timestamp="2024-01-01T00:00:00"
            )
        ]
        
        clauses_list = [
            [
                ClauseSegment(
                    clause_id="test_001_clause_0",
                    text="Test clause text",
                    clause_type="liability",
                    start_position=0,
                    end_position=20,
                    confidence=0.8,
                    entities=[],
                    risk_flags=["uncapped_liability"]
                )
            ]
        ]
        
        report = pipeline.generate_data_report(metadata_list, clauses_list)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'contract_types' in report
        assert 'clause_types' in report
        assert 'risk_flags' in report
        assert 'entity_types' in report
        assert 'data_quality' in report

class TestBaselineClauseClassifier:
    """Test suite for baseline clause classifier"""
    
    @pytest.fixture
    def classifier_config(self):
        return {
            'data_path': 'data/processed/clause_segments.csv',
            'output_dir': 'models/baseline',
            'test_size': 0.2,
            'random_state': 42
        }
    
    @pytest.fixture
    def sample_clauses_df(self):
        return pd.DataFrame({
            'text': [
                "The party shall indemnify all damages.",
                "Either party may terminate with notice.",
                "The employee shall maintain confidentiality.",
                "Payment shall be made monthly.",
                "This agreement is governed by California law."
            ],
            'clause_type': [
                'liability',
                'termination',
                'confidentiality',
                'payment',
                'governing_law'
            ]
        })
    
    @pytest.fixture
    def classifier(self, classifier_config):
        return BaselineClauseClassifier(classifier_config)
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert classifier.config is not None
        assert classifier.label_encoder is not None
        assert classifier.vectorizer is not None
        assert classifier.model is None
    
    def test_prepare_data(self, classifier, sample_clauses_df):
        """Test data preparation"""
        X, y = classifier.prepare_data(sample_clauses_df)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_clauses_df)
        assert len(y) == len(sample_clauses_df)
        assert classifier.clause_types is not None
        assert len(classifier.clause_types) > 0
    
    def test_train_model(self, classifier, sample_clauses_df):
        """Test model training"""
        X, y = classifier.prepare_data(sample_clauses_df)
        metrics = classifier.train_model(X, y, model_type='logistic')
        
        assert classifier.model is not None
        assert isinstance(metrics, dict)
        assert 'model_type' in metrics
        assert 'cv_f1_macro_mean' in metrics
        assert 'f1_macro' in metrics
        assert 'training_samples' in metrics
    
    def test_predict(self, classifier, sample_clauses_df):
        """Test prediction functionality"""
        X, y = classifier.prepare_data(sample_clauses_df)
        classifier.train_model(X, y, model_type='logistic')
        
        test_texts = ["This is a test clause.", "Another test clause."]
        predictions, confidence_scores = classifier.predict(test_texts)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(confidence_scores, np.ndarray)
        assert len(predictions) == len(test_texts)
        assert len(confidence_scores) == len(test_texts)
        assert all(0 <= score <= 1 for score in confidence_scores)
    
    def test_get_feature_importance(self, classifier, sample_clauses_df):
        """Test feature importance extraction"""
        X, y = classifier.prepare_data(sample_clauses_df)
        classifier.train_model(X, y, model_type='logistic')
        
        feature_importance = classifier.get_feature_importance(top_n=5)
        
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0
        
        for clause_type, features in feature_importance.items():
            assert isinstance(features, list)
            assert len(features) <= 5
            for feature, score in features:
                assert isinstance(feature, str)
                assert isinstance(score, (int, float))
    
    def test_save_and_load_model(self, classifier, sample_clauses_df, tmp_path):
        """Test model saving and loading"""
        X, y = classifier.prepare_data(sample_clauses_df)
        classifier.train_model(X, y, model_type='logistic')
        
        model_path = tmp_path / "test_model.joblib"
        classifier.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_classifier = BaselineClauseClassifier.load_model(str(model_path))
        
        assert loaded_classifier.model is not None
        assert loaded_classifier.clause_types is not None
        assert loaded_classifier.feature_names is not None

class TestKeywordRiskScorer:
    """Test suite for keyword-based risk scorer"""
    
    @pytest.fixture
    def risk_scorer(self):
        return KeywordRiskScorer()
    
    def test_risk_scorer_initialization(self, risk_scorer):
        """Test risk scorer initialization"""
        assert risk_scorer is not None
        assert 'high_risk' in risk_scorer.risk_patterns
        assert 'medium_risk' in risk_scorer.risk_patterns
        assert 'low_risk' in risk_scorer.risk_patterns
        assert 'high_risk' in risk_scorer.risk_weights
    
    def test_score_clause_high_risk(self, risk_scorer):
        """Test high risk clause scoring"""
        high_risk_text = "The party shall have unlimited liability for all damages."
        result = risk_scorer.score_clause(high_risk_text)
        
        assert isinstance(result, dict)
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'risk_breakdown' in result
        assert 'detected_risks' in result
        assert result['risk_level'] == 'HIGH'
        assert result['risk_score'] > 7.0
    
    def test_score_clause_medium_risk(self, risk_scorer):
        """Test medium risk clause scoring"""
        medium_risk_text = "The party shall use reasonable efforts to complete the work."
        result = risk_scorer.score_clause(medium_risk_text)
        
        assert isinstance(result, dict)
        assert result['risk_level'] == 'MEDIUM'
        assert 4.0 <= result['risk_score'] < 7.0
    
    def test_score_clause_low_risk(self, risk_scorer):
        """Test low risk clause scoring"""
        low_risk_text = "Both parties agree to standard terms and conditions."
        result = risk_scorer.score_clause(low_risk_text)
        
        assert isinstance(result, dict)
        assert result['risk_level'] == 'LOW'
        assert result['risk_score'] < 4.0
    
    def test_score_contract(self, risk_scorer):
        """Test contract-level risk scoring"""
        clauses = [
            "The party shall have unlimited liability.",
            "The party shall use reasonable efforts.",
            "Both parties agree to standard terms."
        ]
        
        result = risk_scorer.score_contract(clauses)
        
        assert isinstance(result, dict)
        assert 'contract_risk_score' in result
        assert 'contract_risk_level' in result
        assert 'total_clauses' in result
        assert 'high_risk_clauses' in result
        assert 'medium_risk_clauses' in result
        assert 'low_risk_clauses' in result
        assert 'clause_scores' in result
        assert 'risk_distribution' in result
        assert result['total_clauses'] == 3

class TestBaselineEvaluator:
    """Test suite for baseline evaluator"""
    
    @pytest.fixture
    def evaluator_config(self):
        return {
            'data_path': 'data/processed/clause_segments.csv',
            'output_dir': 'models/baseline',
            'test_size': 0.2,
            'random_state': 42
        }
    
    @pytest.fixture
    def evaluator(self, evaluator_config):
        return BaselineEvaluator(evaluator_config)
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator is not None
        assert evaluator.config is not None
        assert evaluator.results == {}
    
    def test_evaluate_classifier(self, evaluator):
        """Test classifier evaluation"""
        # Mock classifier and test data
        classifier = Mock()
        classifier.model = Mock()
        classifier.clause_types = ['liability', 'termination', 'confidentiality']
        classifier.get_feature_importance.return_value = {'liability': [('test', 0.5)]}
        
        X_test = np.random.rand(10, 100)
        y_test = np.random.randint(0, 3, 10)
        
        # Mock predictions
        classifier.model.predict.return_value = np.random.randint(0, 3, 10)
        classifier.model.predict_proba.return_value = np.random.rand(10, 3)
        
        result = evaluator.evaluate_classifier(classifier, X_test, y_test)
        
        assert isinstance(result, dict)
        assert 'f1_macro' in result
        assert 'f1_weighted' in result
        assert 'f1_per_class' in result
        assert 'classification_report' in result
        assert 'confusion_matrix' in result
        assert 'feature_importance' in result
    
    def test_evaluate_risk_scorer(self, evaluator):
        """Test risk scorer evaluation"""
        risk_scorer = KeywordRiskScorer()
        test_clauses = [
            "The party shall have unlimited liability.",
            "The party shall use reasonable efforts.",
            "Both parties agree to standard terms."
        ]
        
        result = evaluator.evaluate_risk_scorer(risk_scorer, test_clauses)
        
        assert isinstance(result, dict)
        assert 'avg_risk_score' in result
        assert 'std_risk_score' in result
        assert 'risk_distribution' in result
        assert 'risk_type_distribution' in result
        assert 'total_clauses' in result
        assert 'clauses_with_risks' in result
    
    def test_generate_report(self, evaluator, tmp_path):
        """Test report generation"""
        # Add some results
        evaluator.results = {
            'classifier': {'f1_macro': 0.8},
            'risk_scorer': {'avg_risk_score': 5.0}
        }
        
        report_path = tmp_path / "test_report.json"
        evaluator.generate_report(str(report_path))
        
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        assert 'evaluation_summary' in report
        assert 'results' in report
        assert 'classifier' in report['results']
        assert 'risk_scorer' in report['results']

class TestDataValidation:
    """Test suite for data validation"""
    
    def test_pandera_schema_validation(self):
        """Test Pandera schema validation"""
        import pandera as pa
        from pandera.typing import Series
        
        class TestSchema(pa.SchemaModel):
            text: Series[str] = pa.Field(str_len={"min_value": 1})
            clause_type: Series[str] = pa.Field(nullable=True)
            confidence: Series[float] = pa.Field(ge=0.0, le=1.0)
        
        # Valid data
        valid_data = pd.DataFrame({
            'text': ['Test clause'],
            'clause_type': ['liability'],
            'confidence': [0.8]
        })
        
        # Should not raise exception
        TestSchema.validate(valid_data)
        
        # Invalid data
        invalid_data = pd.DataFrame({
            'text': [''],
            'clause_type': ['liability'],
            'confidence': [1.5]  # Invalid confidence
        })
        
        # Should raise exception
        with pytest.raises(Exception):
            TestSchema.validate(invalid_data)

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_end_to_end_pipeline(self, tmp_path):
        """Test complete end-to-end pipeline"""
        # Create sample data
        sample_data = {
            'data': [
                {
                    'id': 'test_contract_001',
                    'title': 'Test Employment Agreement',
                    'context': 'This employment agreement is between ABC Corp and John Doe.',
                    'questions': [],
                    'answers': []
                }
            ]
        }
        
        data_file = tmp_path / "test_cuad.json"
        with open(data_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Test pipeline
        config = {
            'cuad_file_path': str(data_file),
            'output_dir': str(tmp_path / 'processed'),
            'validation_enabled': True,
            'logging_level': 'INFO'
        }
        
        pipeline = ContractDataPipeline(config)
        
        # Load data
        cuad_df = pipeline.load_cuad_dataset(str(data_file))
        assert len(cuad_df) == 1
        
        # Process contract
        contract = cuad_df.iloc[0]
        metadata, clauses = pipeline.process_contract(contract['context'], contract['contract_id'])
        
        assert isinstance(metadata, ContractMetadata)
        assert isinstance(clauses, list)
        assert len(clauses) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
