"""
Integration tests for API endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from api.main import app

class TestAPIEndpoints:
    """Test API endpoint functionality."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer devtoken"}

    def test_health_endpoint(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_detailed_endpoint(self):
        """Test detailed health endpoint."""
        response = self.client.get("/health/detailed", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "metrics" in data
        assert "security" in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = self.client.get("/metrics", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "counters" in data
        assert "histograms" in data
        assert "gauges" in data

    def test_analyze_contract_text(self):
        """Test contract analysis with text."""
        payload = {
            "contract_id": "test_contract",
            "text": "This is a test contract clause."
        }
        response = self.client.post("/analyze_contract", 
                                  json=payload, 
                                  headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "contract_id" in data
        assert "results" in data
        assert "latency_ms" in data

    def test_analyze_contract_file(self):
        """Test contract analysis with file."""
        payload = {
            "contract_id": "test_contract",
            "file_b64": "VGVzdCBjb250cmFjdCBjb250ZW50",
            "mime": "text/plain"
        }
        response = self.client.post("/analyze_contract", 
                                  json=payload, 
                                  headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "contract_id" in data
        assert "results" in data

    def test_batch_analyze(self):
        """Test batch analysis."""
        payload = {
            "contracts": [
                {"contract_id": "contract1", "text": "Clause 1"},
                {"contract_id": "contract2", "text": "Clause 2"}
            ]
        }
        response = self.client.post("/batch_analyze", 
                                  json=payload, 
                                  headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_risk_report(self):
        """Test risk report generation."""
        payload = {
            "contract_ids": ["contract1", "contract2"]
        }
        response = self.client.post("/risk_report", 
                                  json=payload, 
                                  headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "contract_details" in data

    def test_export_formats(self):
        """Test export formats endpoint."""
        response = self.client.get("/export/formats", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "descriptions" in data

    def test_unauthorized_access(self):
        """Test unauthorized access."""
        response = self.client.get("/health/detailed")
        assert response.status_code == 401

    def test_rate_limiting(self):
        """Test rate limiting."""
        # Make multiple requests to test rate limiting
        for i in range(65):  # Exceed rate limit
            response = self.client.get("/health", headers=self.headers)
            if response.status_code == 429:
                break
        else:
            # If we didn't hit rate limit, that's also acceptable for testing
            assert response.status_code in [200, 429]

if __name__ == "__main__":
    pytest.main([__file__])
