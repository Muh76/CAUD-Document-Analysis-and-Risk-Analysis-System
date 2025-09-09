"""
Golden tests for end-to-end pipeline validation.
"""

import pytest
import json
import tempfile
from pathlib import Path

from core.pipeline import ContractAnalyzer

class TestGoldenPipeline:
    """Test end-to-end pipeline with golden data."""

    def test_sample_contract_analysis(self):
        """Test analysis with sample contract data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create label_map.json
            label_map = {i: f"label_{i}" for i in range(5)}
            with open(artifacts_dir / "label_map.json", 'w') as f:
                json.dump(label_map, f)

            # Create thresholds.json
            thresholds = {
                "per_label_thresholds": {f"label_{i}": 0.5 for i in range(5)},
                "global_thresholds": {"HIGH_RISK_THRESHOLD": 0.3}
            }
            with open(artifacts_dir / "thresholds.json", 'w') as f:
                json.dump(thresholds, f)

            analyzer = ContractAnalyzer(str(artifacts_dir))

            # Sample contract clauses
            clauses = [
                "This agreement is entered into between Party A and Party B.",
                "The term of this agreement shall be for a period of one year.",
                "Either party may terminate this agreement with 30 days notice."
            ]

            result = analyzer.analyze("sample_contract", clauses)

            # Validate result structure
            assert "contract_id" in result
            assert "results" in result
            assert "latency_ms" in result
            assert "thresholds_used" in result
            assert "model_snapshot" in result

            # Validate results
            assert len(result["results"]) == 3
            for i, clause_result in enumerate(result["results"]):
                assert "clause_id" in clause_result
                assert "text" in clause_result
                assert "probs" in clause_result
                assert "risk" in clause_result
                assert clause_result["clause_id"] == i
                assert len(clause_result["probs"]) == 5

    def test_risk_scoring_consistency(self):
        """Test that risk scoring is consistent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create label_map.json
            label_map = {i: f"label_{i}" for i in range(5)}
            with open(artifacts_dir / "label_map.json", 'w') as f:
                json.dump(label_map, f)

            # Create thresholds.json
            thresholds = {
                "per_label_thresholds": {f"label_{i}": 0.5 for i in range(5)},
                "global_thresholds": {"HIGH_RISK_THRESHOLD": 0.3}
            }
            with open(artifacts_dir / "thresholds.json", 'w') as f:
                json.dump(thresholds, f)

            analyzer = ContractAnalyzer(str(artifacts_dir))

            # Test same clause multiple times
            clause = "This is a test clause for consistency testing."
            results = []

            for i in range(5):
                result = analyzer.predict_clause(clause)
                results.append(result)

            # Check that probabilities are consistent (within tolerance)
            for i in range(1, len(results)):
                for j in range(len(results[0]["probs"])):
                    diff = abs(results[0]["probs"][j] - results[i]["probs"][j])
                    assert diff < 0.01  # Allow small variance

if __name__ == "__main__":
    pytest.main([__file__])
