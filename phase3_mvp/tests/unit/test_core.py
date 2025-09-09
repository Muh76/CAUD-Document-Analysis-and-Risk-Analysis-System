"""
Unit tests for core functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.pipeline import ContractAnalyzer
from core.schemas import ContractAnalysis, ClauseResult
from core.settings import Settings
from core.io import normalize_contract_id
from core.text_ingest import TextIngestion
from core.export import ExportManager

class TestContractAnalyzer:
    """Test ContractAnalyzer functionality."""

    def test_init(self):
        """Test analyzer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock artifacts
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
            assert analyzer.label_map == label_map
            assert analyzer.tau == thresholds

    def test_predict_clause(self):
        """Test clause prediction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create mock artifacts
            label_map = {i: f"label_{i}" for i in range(5)}
            with open(artifacts_dir / "label_map.json", 'w') as f:
                json.dump(label_map, f)

            thresholds = {
                "per_label_thresholds": {f"label_{i}": 0.5 for i in range(5)},
                "global_thresholds": {"HIGH_RISK_THRESHOLD": 0.3}
            }
            with open(artifacts_dir / "thresholds.json", 'w') as f:
                json.dump(thresholds, f)

            analyzer = ContractAnalyzer(str(artifacts_dir))
            result = analyzer.predict_clause("Test clause text")

            assert "probs" in result
            assert len(result["probs"]) == 5
            assert all(0 <= p <= 1 for p in result["probs"])

    def test_analyze(self):
        """Test contract analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create mock artifacts
            label_map = {i: f"label_{i}" for i in range(5)}
            with open(artifacts_dir / "label_map.json", 'w') as f:
                json.dump(label_map, f)

            thresholds = {
                "per_label_thresholds": {f"label_{i}": 0.5 for i in range(5)},
                "global_thresholds": {"HIGH_RISK_THRESHOLD": 0.3}
            }
            with open(artifacts_dir / "thresholds.json", 'w') as f:
                json.dump(thresholds, f)

            analyzer = ContractAnalyzer(str(artifacts_dir))
            clauses = ["Clause 1", "Clause 2", "Clause 3"]
            result = analyzer.analyze("test_contract", clauses)

            assert result["contract_id"] == "test_contract"
            assert len(result["results"]) == 3
            assert "latency_ms" in result
            assert "thresholds_used" in result

class TestSettings:
    """Test Settings functionality."""

    def test_init(self):
        """Test settings initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(temp_dir)
            assert settings.artifacts_dir == Path(temp_dir)
            assert settings.model_snapshot == Path(temp_dir).name

class TestIO:
    """Test IO utilities."""

    def test_normalize_contract_id(self):
        """Test contract ID normalization."""
        assert normalize_contract_id("Test Contract") == "test_contract"
        assert normalize_contract_id("Contract-123") == "contract_123"
        assert normalize_contract_id("CONTRACT_456") == "contract_456"

class TestExport:
    """Test export functionality."""

    def test_export_manager_init(self):
        """Test export manager initialization."""
        export_manager = ExportManager()
        assert export_manager is not None

    def test_get_export_formats(self):
        """Test export formats."""
        export_manager = ExportManager()
        formats = export_manager.get_export_formats()
        assert "csv" in formats
        assert "json" in formats
        assert "portfolio_csv" in formats
        assert "risk_report" in formats

if __name__ == "__main__":
    pytest.main([__file__])
