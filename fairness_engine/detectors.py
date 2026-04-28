"""
EquiLens AI — Bias Detectors

Bias detection strategies that wrap metric computations with
threshold logic, severity classification, and remediation hints.
"""

from __future__ import annotations

from enum import Enum


class BiasSeverity(Enum):
    """Classification of bias severity levels."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class BiasDetector:
    """
    Base class for bias detection strategies.

    Subclass this to implement custom detection logic
    with domain-specific thresholds and severity mappings.
    """

    def __init__(self, name: str, thresholds: dict[str, float] | None = None) -> None:
        self.name = name
        self.thresholds = thresholds or {}

    def detect(self, **kwargs) -> dict:
        """
        Run bias detection and return findings.

        Returns:
            Dict with keys: 'severity', 'findings', 'recommendations'
        """
        # TODO: Implement detection logic
        raise NotImplementedError("Detection logic not yet implemented")

    def classify_severity(self, metric_value: float, threshold: float) -> BiasSeverity:
        """Map a metric deviation to a severity level."""
        # TODO: Implement severity classification
        raise NotImplementedError("Severity classification not yet implemented")
