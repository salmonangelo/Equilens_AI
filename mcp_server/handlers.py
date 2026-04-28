"""
EquiLens AI — MCP Handlers

Request handlers that bridge MCP tool invocations to the
fairness engine's core functionality.
"""

from __future__ import annotations


class MCPToolHandler:
    """
    Base handler for MCP tool invocations.

    Each tool maps to a handler method that validates input,
    calls the fairness engine, and formats the response.
    """

    def __init__(self) -> None:
        # TODO: Initialize fairness engine dependency
        pass

    async def handle_analyze_fairness(self, params: dict) -> dict:
        """Handle the 'analyze_fairness' tool invocation."""
        # TODO: Implement fairness analysis dispatch
        raise NotImplementedError

    async def handle_get_metrics(self, params: dict) -> dict:
        """Handle the 'get_metrics' tool invocation."""
        # TODO: Implement metrics retrieval
        raise NotImplementedError

    async def handle_suggest_remediation(self, params: dict) -> dict:
        """Handle the 'suggest_remediation' tool invocation."""
        # TODO: Implement remediation suggestion generation
        raise NotImplementedError
