"""
EquiLens AI — Fairness Engine

Core module for computing bias metrics, evaluating model fairness,
and detecting disparities across protected attributes.
"""

from fairness_engine.metrics import (
    disparate_impact_ratio,
    demographic_parity_difference,
    equal_opportunity_difference,
    compute_all_metrics,
    MetricResult,
)
from fairness_engine.evaluator import FairnessEvaluator, EvaluationReport
from fairness_engine.anonymizer import (
    anonymize,
    detect_pii_columns,
    AnonymizationConfig,
    AnonymizationStrategy,
    get_anonymization_summary,
)
from fairness_engine.scoring import (
    compute_fairness_risk_score,
    compute_frs_from_dataframe,
    compute_frs_from_metrics,
    ScoringConfig,
    RiskLevel,
    FairnessRiskResult,
)

__all__ = [
    "disparate_impact_ratio",
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "compute_all_metrics",
    "MetricResult",
    "FairnessEvaluator",
    "EvaluationReport",
    "anonymize",
    "detect_pii_columns",
    "AnonymizationConfig",
    "AnonymizationStrategy",
    "get_anonymization_summary",
    "compute_fairness_risk_score",
    "compute_frs_from_dataframe",
    "compute_frs_from_metrics",
    "ScoringConfig",
    "RiskLevel",
    "FairnessRiskResult",
]

