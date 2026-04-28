"""
Tests for the MCP server.
"""

import pytest
from fastapi.testclient import TestClient

from mcp_server.server import app


@pytest.fixture
def client():
    """FastAPI test client for MCP server."""
    return TestClient(app)


class TestMCPRoot:
    """Tests for the MCP server root endpoint."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "mcp-server"
        assert data["status"] == "running"


class TestMCPToolEndpoints:
    """Tests for MCP tool listing and invocation."""

    def test_list_tools(self, client):
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_invoke_tool_returns_stub(self, client):
        response = client.post("/mcp/tools/test_tool/invoke")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_implemented"
        assert data["tool"] == "test_tool"


class TestMCPResourceEndpoints:
    """Tests for MCP resource listing."""

    def test_list_resources(self, client):
        response = client.get("/mcp/resources")
        assert response.status_code == 200
        data = response.json()
        assert "resources" in data
        assert isinstance(data["resources"], list)
