"""
EquiLens AI — Fairness Risk Score (FRS)

Combines Disparate Impact (DI), Demographic Parity Difference (DPD),
and Equal Opportunity Difference (EOD) into a single, interpretable
risk score on a 0–1 scale.

Scoring methodology
═══════════════════

The FRS is **not** an arbitrary weighted average.  Each metric is first
transformed into a per-metric risk value via a piecewise-linear function
anchored to established fairness thresholds from the literature, then the
three per-metric risks are combined with configurable weights.

1. Per-metric risk functions
────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Disparate Impact Ratio (DI)                                       │
  │                                                                   │
  │ Ideal value: 1.0  (equal selection rates)                         │
  │ 80% rule threshold: 0.8 ≤ DI ≤ 1.25                              │
  │                                                                   │
  │ risk_DI(x) is defined by distance from 1.0, normalised so that:   │
  │   • DI = 1.0          → risk = 0.0   (perfect parity)             │
  │   • DI = 0.8 or 1.25  → risk = 0.5   (threshold boundary)        │
  │   • DI ≤ 0.5 or ≥ 2.0 → risk = 1.0   (severe disparity)          │
  │                                                                   │
  │ Between these anchors the function is linearly interpolated.       │
  │ DI = inf or DI ≤ 0    → risk = 1.0                                │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Demographic Parity Difference (DPD) — absolute value used          │
  │                                                                   │
  │ Ideal value: 0.0  (equal positive prediction rates)               │
  │                                                                   │
  │ risk_DPD(x) = linear mapping of |DPD|:                            │
  │   • |DPD| = 0.0   → risk = 0.0                                    │
  │   • |DPD| = 0.1   → risk = 0.5   (standard threshold)             │
  │   • |DPD| ≥ 0.3   → risk = 1.0   (severe)                         │
  │   • Linear interpolation between anchors.                          │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Equal Opportunity Difference (EOD) — absolute value used           │
  │                                                                   │
  │ Ideal value: 0.0  (equal TPR across groups)                       │
  │                                                                   │
  │ risk_EOD(x) = same shape as DPD:                                   │
  │   • |EOD| = 0.0   → risk = 0.0                                    │
  │   • |EOD| = 0.1   → risk = 0.5                                    │
  │   • |EOD| ≥ 0.3   → risk = 1.0                                    │
  │   • NaN (undefined TPR) → risk = 1.0 + warning flag               │
  └─────────────────────────────────────────────────────────────────────┘

2. Composite Fairness Risk Score
────────────────────────────────

  FRS = w_DI · risk_DI  +  w_DPD · risk_DPD  +  w_EOD · risk_EOD

  Default weights: DI=0.35, DPD=0.35, EOD=0.30
  (DI and DPD are given equal weight as complementary group-fairness
   measures; EOD is slightly lower because it conditions on ground truth
   which can itself be biased.)

  Weights are normalised to sum to 1.0 internally, so you can pass
  any positive ratios.

3. Risk level classification
────────────────────────────

  ┌──────────┬───────────────┬──────────────────────────────────────┐
  │ Level    │ FRS range     │ Interpretation                      │
  ├──────────┼───────────────┼──────────────────────────────────────┤
  │ LOW      │ [0.0, 0.3)    │ Model meets fairness thresholds.    │
  │ MEDIUM   │ [0.3, 0.6)    │ Marginal — review recommended.      │
  │ HIGH     │ [0.6, 1.0]    │ Significant bias detected.          │
  └──────────┴───────────────┴──────────────────────────────────────┘

  Both the weight vector and risk-level boundaries are configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from fairness_engine.metrics import (
    compute_all_metrics,
    disparate_impact_ratio,
    demographic_parity_difference,
    equal_opportunity_difference,
)


# ===================================================================
# Enums & config
# ===================================================================

class RiskLevel(Enum):
    """Fairness risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ScoringConfig:
    """
    Configuration for the Fairness Risk Score computation.

    Attributes:
        weight_di:  Weight for Disparate Impact risk component.
        weight_dpd: Weight for Demographic Parity Difference risk component.
        weight_eod: Weight for Equal Opportunity Difference risk component.

        di_fair_lower:    Lower bound of the DI fair range (80% rule = 0.8).
        di_fair_upper:    Upper bound of the DI fair range (80% rule = 1.25).
        di_severe_lower:  DI value at or below which risk is maximum.
        di_severe_upper:  DI value at or above which risk is maximum.

        dpd_threshold:    |DPD| at which risk = 0.5.
        dpd_severe:       |DPD| at or above which risk = 1.0.

        eod_threshold:    |EOD| at which risk = 0.5.
        eod_severe:       |EOD| at or above which risk = 1.0.

        risk_low_upper:    FRS below this → LOW risk.
        risk_medium_upper: FRS below this → MEDIUM risk; at or above → HIGH.
    """

    # --- Weights (will be normalised internally) ---
    weight_di: float = 0.35
    weight_dpd: float = 0.35
    weight_eod: float = 0.30

    # --- Disparate Impact anchors ---
    di_fair_lower: float = 0.8
    di_fair_upper: float = 1.25
    di_severe_lower: float = 0.5
    di_severe_upper: float = 2.0

    # --- Demographic Parity Difference anchors ---
    dpd_threshold: float = 0.1
    dpd_severe: float = 0.3

    # --- Equal Opportunity Difference anchors ---
    eod_threshold: float = 0.1
    eod_severe: float = 0.3

    # --- Risk level boundaries ---
    risk_low_upper: float = 0.3
    risk_medium_upper: float = 0.6

    def __post_init__(self) -> None:
        if any(w < 0 for w in (self.weight_di, self.weight_dpd, self.weight_eod)):
            raise ValueError("All weights must be non-negative.")
        total = self.weight_di + self.weight_dpd + self.weight_eod
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        if not (0 < self.di_severe_lower < self.di_fair_lower < 1.0):
            raise ValueError(
                "DI anchors must satisfy: 0 < di_severe_lower < di_fair_lower < 1.0"
            )
        if not (1.0 < self.di_fair_upper < self.di_severe_upper):
            raise ValueError(
                "DI anchors must satisfy: 1.0 < di_fair_upper < di_severe_upper"
            )

    @property
    def normalised_weights(self) -> tuple[float, float, float]:
        """Return weights normalised to sum to 1.0."""
        total = self.weight_di + self.weight_dpd + self.weight_eod
        return (
            self.weight_di / total,
            self.weight_dpd / total,
            self.weight_eod / total,
        )


# ===================================================================
# Per-metric risk functions
# ===================================================================

def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation: maps x ∈ [x0, x1] → [y0, y1], clamped."""
    if x1 == x0:
        return y1
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def compute_di_risk(di_value: float, config: ScoringConfig) -> float:
    """
    Transform a Disparate Impact ratio into a [0, 1] risk score.

    Piecewise-linear mapping:
        DI = 1.0                          → 0.0  (perfect)
        DI = di_fair_lower or di_fair_upper → 0.5  (threshold boundary)
        DI ≤ di_severe_lower or ≥ di_severe_upper → 1.0  (severe)
        DI ≤ 0, NaN, or inf              → 1.0  (undefined / extreme)

    Args:
        di_value: The Disparate Impact ratio.
        config: Scoring configuration with DI anchors.

    Returns:
        Risk score in [0.0, 1.0].
    """
    if not math.isfinite(di_value) or di_value <= 0:
        return 1.0

    # Distance from ideal (1.0), mapped differently below/above 1.0
    if di_value <= config.di_severe_lower:
        return 1.0
    elif di_value <= config.di_fair_lower:
        # Interpolate: severe_lower → 1.0, fair_lower → 0.5
        return _lerp(di_value, config.di_severe_lower, config.di_fair_lower, 1.0, 0.5)
    elif di_value <= 1.0:
        # Interpolate: fair_lower → 0.5, 1.0 → 0.0
        return _lerp(di_value, config.di_fair_lower, 1.0, 0.5, 0.0)
    elif di_value <= config.di_fair_upper:
        # Interpolate: 1.0 → 0.0, fair_upper → 0.5
        return _lerp(di_value, 1.0, config.di_fair_upper, 0.0, 0.5)
    elif di_value <= config.di_severe_upper:
        # Interpolate: fair_upper → 0.5, severe_upper → 1.0
        return _lerp(di_value, config.di_fair_upper, config.di_severe_upper, 0.5, 1.0)
    else:
        return 1.0


def compute_dpd_risk(dpd_value: float, config: ScoringConfig) -> float:
    """
    Transform a Demographic Parity Difference into a [0, 1] risk score.

    Uses |DPD| with piecewise-linear mapping:
        |DPD| = 0.0            → 0.0  (perfect parity)
        |DPD| = dpd_threshold  → 0.5  (standard threshold)
        |DPD| ≥ dpd_severe     → 1.0  (severe)

    Args:
        dpd_value: The Demographic Parity Difference (can be negative).
        config: Scoring configuration with DPD anchors.

    Returns:
        Risk score in [0.0, 1.0].
    """
    if not math.isfinite(dpd_value):
        return 1.0

    abs_dpd = abs(dpd_value)

    if abs_dpd >= config.dpd_severe:
        return 1.0
    elif abs_dpd >= config.dpd_threshold:
        return _lerp(abs_dpd, config.dpd_threshold, config.dpd_severe, 0.5, 1.0)
    else:
        return _lerp(abs_dpd, 0.0, config.dpd_threshold, 0.0, 0.5)


def compute_eod_risk(eod_value: float, config: ScoringConfig) -> float:
    """
    Transform an Equal Opportunity Difference into a [0, 1] risk score.

    Uses |EOD| with piecewise-linear mapping:
        |EOD| = 0.0            → 0.0  (perfect equality)
        |EOD| = eod_threshold  → 0.5  (standard threshold)
        |EOD| ≥ eod_severe     → 1.0  (severe)
        NaN (undefined TPR)    → 1.0  (cannot assess)

    Args:
        eod_value: The Equal Opportunity Difference (can be NaN).
        config: Scoring configuration with EOD anchors.

    Returns:
        Risk score in [0.0, 1.0].
    """
    if not math.isfinite(eod_value):
        return 1.0

    abs_eod = abs(eod_value)

    if abs_eod >= config.eod_severe:
        return 1.0
    elif abs_eod >= config.eod_threshold:
        return _lerp(abs_eod, config.eod_threshold, config.eod_severe, 0.5, 1.0)
    else:
        return _lerp(abs_eod, 0.0, config.eod_threshold, 0.0, 0.5)


# ===================================================================
# Composite FRS
# ===================================================================

@dataclass
class FairnessRiskResult:
    """
    Complete Fairness Risk Score result.

    Attributes:
        score:          Composite FRS in [0.0, 1.0].
        risk_level:     Classified risk level (LOW / MEDIUM / HIGH).
        di_risk:        Per-metric risk for Disparate Impact.
        dpd_risk:       Per-metric risk for Demographic Parity Difference.
        eod_risk:       Per-metric risk for Equal Opportunity Difference.
        di_value:       Raw Disparate Impact ratio used.
        dpd_value:      Raw DPD value used.
        eod_value:      Raw EOD value used.
        weights:        Normalised weights applied [w_DI, w_DPD, w_EOD].
        explanation:    Human-readable breakdown of the score.
        warnings:       List of diagnostic warnings (e.g. undefined TPR).
    """

    score: float
    risk_level: RiskLevel
    di_risk: float
    dpd_risk: float
    eod_risk: float
    di_value: float
    dpd_value: float
    eod_value: float
    weights: tuple[float, float, float]
    explanation: str
    warnings: list[str] = field(default_factory=list)


def _classify_risk(score: float, config: ScoringConfig) -> RiskLevel:
    """Map a composite FRS to a risk level."""
    if score < config.risk_low_upper:
        return RiskLevel.LOW
    elif score < config.risk_medium_upper:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH


def _build_explanation(
    score: float,
    risk_level: RiskLevel,
    di_value: float,
    dpd_value: float,
    eod_value: float,
    di_risk: float,
    dpd_risk: float,
    eod_risk: float,
    weights: tuple[float, float, float],
) -> str:
    """Build a human-readable explanation of the FRS computation."""
    lines = [
        f"Fairness Risk Score: {score:.4f} ({risk_level.value.upper()} risk)",
        "",
        "Per-metric breakdown:",
        f"  Disparate Impact     = {di_value:.4f}  →  risk = {di_risk:.4f}  (weight: {weights[0]:.2f})",
        f"  Demographic Parity   = {dpd_value:+.4f}  →  risk = {dpd_risk:.4f}  (weight: {weights[1]:.2f})",
    ]

    if math.isnan(eod_value):
        lines.append(
            f"  Equal Opportunity    = NaN (undefined)  →  risk = {eod_risk:.4f}  (weight: {weights[2]:.2f})"
        )
    else:
        lines.append(
            f"  Equal Opportunity    = {eod_value:+.4f}  →  risk = {eod_risk:.4f}  (weight: {weights[2]:.2f})"
        )

    lines.extend([
        "",
        f"  Composite = {weights[0]:.2f}×{di_risk:.4f} + {weights[1]:.2f}×{dpd_risk:.4f} + {weights[2]:.2f}×{eod_risk:.4f} = {score:.4f}",
    ])
    return "\n".join(lines)


def compute_fairness_risk_score(
    di_value: float,
    dpd_value: float,
    eod_value: float,
    config: ScoringConfig | None = None,
) -> FairnessRiskResult:
    """
    Compute the composite Fairness Risk Score from raw metric values.

    This is the low-level function — pass pre-computed metric values
    directly.  For a DataFrame-level convenience function, see
    ``compute_frs_from_dataframe``.

    Args:
        di_value:  Disparate Impact Ratio (ideal = 1.0).
        dpd_value: Demographic Parity Difference (ideal = 0.0).
        eod_value: Equal Opportunity Difference (ideal = 0.0, may be NaN).
        config:    Scoring configuration (uses defaults if None).

    Returns:
        FairnessRiskResult with score, risk level, breakdown, and explanation.
    """
    config = config or ScoringConfig()
    w_di, w_dpd, w_eod = config.normalised_weights

    # --- Per-metric risks ---
    di_risk = compute_di_risk(di_value, config)
    dpd_risk = compute_dpd_risk(dpd_value, config)
    eod_risk = compute_eod_risk(eod_value, config)

    # --- Composite score ---
    score = w_di * di_risk + w_dpd * dpd_risk + w_eod * eod_risk
    score = round(max(0.0, min(1.0, score)), 6)

    # --- Classify ---
    risk_level = _classify_risk(score, config)

    # --- Warnings ---
    warn_list: list[str] = []
    if not math.isfinite(di_value) or di_value <= 0:
        warn_list.append(
            f"DI value ({di_value}) is non-finite or non-positive; "
            "risk set to maximum (1.0)."
        )
    if math.isnan(eod_value):
        warn_list.append(
            "EOD is undefined (NaN) — likely due to zero actual positives "
            "in one group. Risk set to maximum (1.0)."
        )
    if not math.isfinite(dpd_value):
        warn_list.append(
            f"DPD value ({dpd_value}) is non-finite; risk set to maximum (1.0)."
        )

    # --- Explanation ---
    explanation = _build_explanation(
        score, risk_level,
        di_value, dpd_value, eod_value,
        di_risk, dpd_risk, eod_risk,
        (w_di, w_dpd, w_eod),
    )

    return FairnessRiskResult(
        score=score,
        risk_level=risk_level,
        di_risk=round(di_risk, 6),
        dpd_risk=round(dpd_risk, 6),
        eod_risk=round(eod_risk, 6),
        di_value=di_value,
        dpd_value=dpd_value,
        eod_value=eod_value,
        weights=(w_di, w_dpd, w_eod),
        explanation=explanation,
        warnings=warn_list,
    )


def compute_frs_from_metrics(metrics: dict[str, dict]) -> FairnessRiskResult:
    """
    Compute FRS from a ``compute_all_metrics()`` result dict.

    Args:
        metrics: Dict returned by ``fairness_engine.metrics.compute_all_metrics``.

    Returns:
        FairnessRiskResult.
    """
    di_value = metrics["disparate_impact_ratio"]["value"]
    dpd_value = metrics["demographic_parity_difference"]["value"]
    eod_value = metrics["equal_opportunity_difference"]["value"]
    return compute_fairness_risk_score(di_value, dpd_value, eod_value)


def compute_frs_from_dataframe(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
    prediction_col: str | None = None,
    *,
    privileged_value: int = 0,
    config: ScoringConfig | None = None,
) -> FairnessRiskResult:
    """
    End-to-end: compute all fairness metrics from a DataFrame, then
    derive the composite Fairness Risk Score.

    This is the recommended high-level entry point.

    Args:
        df: DataFrame containing the data.
        protected_col: Binary protected attribute column.
        target_col: Binary ground-truth label column.
        prediction_col: Binary prediction column (defaults to target_col).
        privileged_value: Value identifying the privileged group.
        config: Scoring configuration (uses defaults if None).

    Returns:
        FairnessRiskResult with score, risk level, and full breakdown.

    Example:
        >>> result = compute_frs_from_dataframe(
        ...     df, "gender", "label", "prediction"
        ... )
        >>> print(result.score, result.risk_level.value)
        0.62 high
    """
    config = config or ScoringConfig()
    pred = prediction_col or target_col

    metrics = compute_all_metrics(
        df, protected_col, target_col, pred,
        privileged_value=privileged_value,
    )

    di_value = metrics["disparate_impact_ratio"]["value"]
    dpd_value = metrics["demographic_parity_difference"]["value"]
    eod_value = metrics["equal_opportunity_difference"]["value"]

    return compute_fairness_risk_score(di_value, dpd_value, eod_value, config)
