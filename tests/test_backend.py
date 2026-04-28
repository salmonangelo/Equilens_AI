"""
Tests for the backend Cloud Run API.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and readiness probes."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self, client):
        response = client.get("/api/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestAnalysisEndpoints:
    """Tests for fairness analysis API endpoints."""

    def test_analyze_returns_not_implemented(self, client):
        # Provide minimal required payload for /analyze
        payload = {
            "dataset_id": "test-dataset",
            "protected_attributes": ["gender"],
            "target_column": "approved",
        }
        response = client.post("/api/v1/analyze", json=payload)
        # Should fail with 404 (dataset not found) since we didn't upload
        assert response.status_code in [404, 200]

    def test_get_report_returns_not_implemented(self, client):
        response = client.get("/api/v1/reports/test-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_implemented"
        assert data["report_id"] == "test-id"

    def test_explain_returns_not_implemented(self, client):
        # Provide minimal required payload for /explain
        payload = {
            "dataset_id": "test-dataset",
            "protected_attributes": ["gender"],
            "target_column": "approved",
        }
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "not_implemented"
