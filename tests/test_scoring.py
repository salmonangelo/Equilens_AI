"""
Tests for the fairness_engine.scoring module (Fairness Risk Score).

Covers:
    - Per-metric risk functions (DI, DPD, EOD) at anchor points and edges
    - Composite FRS computation with default and custom weights
    - Risk level classification (LOW, MEDIUM, HIGH)
    - Edge cases (inf, NaN, zero, boundary values)
    - DataFrame-level convenience function
    - Explanation string correctness
    - ScoringConfig validation
"""

import math

import numpy as np
import pandas as pd
import pytest

from fairness_engine.scoring import (
    RiskLevel,
    ScoringConfig,
    FairnessRiskResult,
    compute_di_risk,
    compute_dpd_risk,
    compute_eod_risk,
    compute_fairness_risk_score,
    compute_frs_from_metrics,
    compute_frs_from_dataframe,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def default_config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture
def fair_dataset() -> pd.DataFrame:
    """Both groups have ~60% positive rate, similar TPR."""
    return pd.DataFrame({
        "protected":  [0]*50 + [1]*50,
        "prediction": [1]*30 + [0]*20 + [1]*29 + [0]*21,
        "label":      [1]*25 + [0]*25 + [1]*24 + [0]*26,
    })


@pytest.fixture
def biased_dataset() -> pd.DataFrame:
    """Privileged: 90% positive, Unprivileged: 30% positive."""
    return pd.DataFrame({
        "protected":  [0]*50 + [1]*50,
        "prediction": [1]*45 + [0]*5 + [1]*15 + [0]*35,
        "label":      [1]*40 + [0]*10 + [1]*20 + [0]*30,
    })


# ===================================================================
# DI risk function
# ===================================================================

class TestDIRisk:
    """Tests for compute_di_risk at known anchor points."""

    def test_perfect_parity(self, default_config):
        """DI = 1.0 → risk = 0.0"""
        assert compute_di_risk(1.0, default_config) == 0.0

    def test_at_lower_threshold(self, default_config):
        """DI = 0.8 → risk = 0.5"""
        assert abs(compute_di_risk(0.8, default_config) - 0.5) < 1e-9

    def test_at_upper_threshold(self, default_config):
        """DI = 1.25 → risk = 0.5"""
        assert abs(compute_di_risk(1.25, default_config) - 0.5) < 1e-9

    def test_at_severe_lower(self, default_config):
        """DI = 0.5 → risk = 1.0"""
        assert compute_di_risk(0.5, default_config) == 1.0

    def test_at_severe_upper(self, default_config):
        """DI = 2.0 → risk = 1.0"""
        assert compute_di_risk(2.0, default_config) == 1.0

    def test_below_severe_lower(self, default_config):
        """DI = 0.3 → risk = 1.0"""
        assert compute_di_risk(0.3, default_config) == 1.0

    def test_above_severe_upper(self, default_config):
        """DI = 5.0 → risk = 1.0"""
        assert compute_di_risk(5.0, default_config) == 1.0

    def test_infinity(self, default_config):
        """DI = inf → risk = 1.0"""
        assert compute_di_risk(float("inf"), default_config) == 1.0

    def test_zero(self, default_config):
        """DI = 0 → risk = 1.0"""
        assert compute_di_risk(0.0, default_config) == 1.0

    def test_negative(self, default_config):
        """DI = -0.5 → risk = 1.0"""
        assert compute_di_risk(-0.5, default_config) == 1.0

    def test_midpoint_below_one(self, default_config):
        """DI = 0.9 → risk = 0.25 (halfway between 0.8→0.5 and 1.0→0.0)"""
        risk = compute_di_risk(0.9, default_config)
        assert abs(risk - 0.25) < 1e-9

    def test_midpoint_above_one(self, default_config):
        """DI = 1.125 → risk = 0.25 (halfway between 1.0→0.0 and 1.25→0.5)"""
        risk = compute_di_risk(1.125, default_config)
        assert abs(risk - 0.25) < 1e-9

    def test_monotonic_below_one(self, default_config):
        """Risk should increase as DI moves away from 1.0 downward."""
        values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
        risks = [compute_di_risk(v, default_config) for v in values]
        for i in range(len(risks) - 1):
            assert risks[i] <= risks[i + 1]


# ===================================================================
# DPD risk function
# ===================================================================

class TestDPDRisk:
    """Tests for compute_dpd_risk at known anchor points."""

    def test_perfect_parity(self, default_config):
        """DPD = 0.0 → risk = 0.0"""
        assert compute_dpd_risk(0.0, default_config) == 0.0

    def test_at_threshold_positive(self, default_config):
        """DPD = +0.1 → risk = 0.5"""
        assert abs(compute_dpd_risk(0.1, default_config) - 0.5) < 1e-9

    def test_at_threshold_negative(self, default_config):
        """DPD = -0.1 → risk = 0.5 (uses absolute value)"""
        assert abs(compute_dpd_risk(-0.1, default_config) - 0.5) < 1e-9

    def test_at_severe(self, default_config):
        """DPD = 0.3 → risk = 1.0"""
        assert compute_dpd_risk(0.3, default_config) == 1.0

    def test_above_severe(self, default_config):
        """DPD = 0.5 → risk = 1.0"""
        assert compute_dpd_risk(0.5, default_config) == 1.0

    def test_midpoint_below_threshold(self, default_config):
        """DPD = 0.05 → risk = 0.25"""
        risk = compute_dpd_risk(0.05, default_config)
        assert abs(risk - 0.25) < 1e-9

    def test_midpoint_between_threshold_and_severe(self, default_config):
        """DPD = 0.2 → risk = 0.75"""
        risk = compute_dpd_risk(0.2, default_config)
        assert abs(risk - 0.75) < 1e-9

    def test_infinity(self, default_config):
        assert compute_dpd_risk(float("inf"), default_config) == 1.0

    def test_symmetry(self, default_config):
        """Positive and negative DPD of same magnitude → same risk."""
        for val in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            assert compute_dpd_risk(val, default_config) == compute_dpd_risk(-val, default_config)


# ===================================================================
# EOD risk function
# ===================================================================

class TestEODRisk:
    """Tests for compute_eod_risk at known anchor points."""

    def test_perfect_equality(self, default_config):
        assert compute_eod_risk(0.0, default_config) == 0.0

    def test_at_threshold(self, default_config):
        assert abs(compute_eod_risk(0.1, default_config) - 0.5) < 1e-9

    def test_at_severe(self, default_config):
        assert compute_eod_risk(0.3, default_config) == 1.0

    def test_nan(self, default_config):
        """NaN (undefined TPR) → risk = 1.0"""
        assert compute_eod_risk(float("nan"), default_config) == 1.0

    def test_negative_symmetry(self, default_config):
        assert compute_eod_risk(-0.15, default_config) == compute_eod_risk(0.15, default_config)


# ===================================================================
# Composite FRS
# ===================================================================

class TestCompositeFRS:
    """Tests for compute_fairness_risk_score."""

    def test_perfect_model(self):
        """All metrics at ideal values → FRS = 0.0, LOW risk."""
        result = compute_fairness_risk_score(1.0, 0.0, 0.0)
        assert result.score == 0.0
        assert result.risk_level == RiskLevel.LOW

    def test_severely_biased_model(self):
        """All metrics at worst values → FRS = 1.0, HIGH risk."""
        result = compute_fairness_risk_score(0.0, 0.5, 0.5)
        assert result.score == 1.0
        assert result.risk_level == RiskLevel.HIGH

    def test_threshold_boundary_model(self):
        """All metrics at threshold boundaries → FRS = 0.5, MEDIUM risk."""
        result = compute_fairness_risk_score(0.8, 0.1, 0.1)
        assert abs(result.score - 0.5) < 1e-6
        assert result.risk_level == RiskLevel.MEDIUM

    def test_mixed_metrics(self):
        """DI perfect, DPD at threshold, EOD severe → weighted mix."""
        result = compute_fairness_risk_score(1.0, 0.1, 0.3)
        # di_risk=0.0, dpd_risk=0.5, eod_risk=1.0
        # FRS = 0.35*0 + 0.35*0.5 + 0.30*1.0 = 0.175 + 0.30 = 0.475
        assert abs(result.score - 0.475) < 1e-6
        assert result.risk_level == RiskLevel.MEDIUM

    def test_score_bounds(self):
        """FRS must always be in [0, 1]."""
        test_cases = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.3, 0.3),
            (2.0, -0.2, 0.15),
            (float("inf"), 0.5, float("nan")),
        ]
        for di, dpd, eod in test_cases:
            result = compute_fairness_risk_score(di, dpd, eod)
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of bounds for ({di}, {dpd}, {eod})"

    def test_returns_all_fields(self):
        result = compute_fairness_risk_score(0.9, -0.05, 0.02)
        assert isinstance(result, FairnessRiskResult)
        assert isinstance(result.score, float)
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.di_risk, float)
        assert isinstance(result.dpd_risk, float)
        assert isinstance(result.eod_risk, float)
        assert isinstance(result.explanation, str)
        assert isinstance(result.warnings, list)
        assert len(result.weights) == 3


# ===================================================================
# Risk level classification
# ===================================================================

class TestRiskLevels:

    def test_low_boundary(self):
        """Score just below 0.3 → LOW."""
        result = compute_fairness_risk_score(1.0, 0.0, 0.0)
        assert result.risk_level == RiskLevel.LOW

    def test_medium_boundary(self):
        """Score at exactly 0.5 → MEDIUM."""
        result = compute_fairness_risk_score(0.8, 0.1, 0.1)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_high_boundary(self):
        """Score at or above 0.6 → HIGH."""
        result = compute_fairness_risk_score(0.5, 0.3, 0.3)
        assert result.risk_level == RiskLevel.HIGH

    def test_custom_boundaries(self):
        """Custom risk boundaries change classification."""
        strict_config = ScoringConfig(
            risk_low_upper=0.2,
            risk_medium_upper=0.4,
        )
        # DI=0.9 → di_risk=0.25, DPD=0.05 → dpd_risk=0.25, EOD=0.05 → eod_risk=0.25
        # FRS = 0.35*0.25 + 0.35*0.25 + 0.30*0.25 = 0.25
        result = compute_fairness_risk_score(0.9, 0.05, 0.05, strict_config)
        assert result.risk_level == RiskLevel.MEDIUM  # 0.25 ≥ 0.2


# ===================================================================
# Warnings
# ===================================================================

class TestWarnings:

    def test_infinite_di_warns(self):
        result = compute_fairness_risk_score(float("inf"), 0.0, 0.0)
        assert len(result.warnings) >= 1
        assert any("DI" in w for w in result.warnings)

    def test_nan_eod_warns(self):
        result = compute_fairness_risk_score(1.0, 0.0, float("nan"))
        assert len(result.warnings) >= 1
        assert any("EOD" in w or "NaN" in w for w in result.warnings)

    def test_zero_di_warns(self):
        result = compute_fairness_risk_score(0.0, 0.0, 0.0)
        assert any("DI" in w for w in result.warnings)

    def test_no_warnings_on_normal_input(self):
        result = compute_fairness_risk_score(1.0, 0.0, 0.0)
        assert len(result.warnings) == 0


# ===================================================================
# Explanation string
# ===================================================================

class TestExplanation:

    def test_contains_score(self):
        result = compute_fairness_risk_score(0.9, -0.05, 0.02)
        assert str(round(result.score, 4)) in result.explanation

    def test_contains_risk_level(self):
        result = compute_fairness_risk_score(0.5, 0.3, 0.3)
        assert "HIGH" in result.explanation

    def test_contains_metric_values(self):
        result = compute_fairness_risk_score(0.85, -0.08, 0.05)
        assert "0.85" in result.explanation
        assert "0.08" in result.explanation or "-0.08" in result.explanation

    def test_nan_eod_noted(self):
        result = compute_fairness_risk_score(1.0, 0.0, float("nan"))
        assert "NaN" in result.explanation or "undefined" in result.explanation


# ===================================================================
# Custom weights
# ===================================================================

class TestCustomWeights:

    def test_equal_weights(self):
        config = ScoringConfig(weight_di=1, weight_dpd=1, weight_eod=1)
        result = compute_fairness_risk_score(0.8, 0.1, 0.1, config)
        # All risks = 0.5, equal weights → FRS = 0.5
        assert abs(result.score - 0.5) < 1e-6

    def test_di_only(self):
        """Weight entirely on DI."""
        config = ScoringConfig(weight_di=1.0, weight_dpd=0.0, weight_eod=0.0)
        result = compute_fairness_risk_score(0.8, 0.3, 0.3, config)
        # Only DI matters: di_risk = 0.5
        assert abs(result.score - 0.5) < 1e-6

    def test_dpd_only(self):
        config = ScoringConfig(weight_di=0.0, weight_dpd=1.0, weight_eod=0.0)
        result = compute_fairness_risk_score(0.0, 0.1, 0.3, config)
        # Only DPD matters: dpd_risk = 0.5
        assert abs(result.score - 0.5) < 1e-6

    def test_weights_normalized(self):
        """Weights (2, 2, 1) should behave same as (0.4, 0.4, 0.2)."""
        c1 = ScoringConfig(weight_di=2, weight_dpd=2, weight_eod=1)
        c2 = ScoringConfig(weight_di=0.4, weight_dpd=0.4, weight_eod=0.2)
        r1 = compute_fairness_risk_score(0.85, 0.12, -0.08, c1)
        r2 = compute_fairness_risk_score(0.85, 0.12, -0.08, c2)
        assert abs(r1.score - r2.score) < 1e-6


# ===================================================================
# Config validation
# ===================================================================

class TestConfigValidation:

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ScoringConfig(weight_di=-1)

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ScoringConfig(weight_di=0, weight_dpd=0, weight_eod=0)

    def test_invalid_di_anchors_raises(self):
        with pytest.raises(ValueError, match="DI anchors"):
            ScoringConfig(di_severe_lower=0.9, di_fair_lower=0.8)


# ===================================================================
# compute_frs_from_metrics
# ===================================================================

class TestFRSFromMetrics:

    def test_from_metrics_dict(self):
        metrics = {
            "disparate_impact_ratio": {"value": 0.85},
            "demographic_parity_difference": {"value": -0.08},
            "equal_opportunity_difference": {"value": 0.05},
        }
        result = compute_frs_from_metrics(metrics)
        assert isinstance(result, FairnessRiskResult)
        assert 0.0 <= result.score <= 1.0


# ===================================================================
# DataFrame-level convenience
# ===================================================================

class TestFRSFromDataFrame:

    def test_fair_dataset_low_risk(self, fair_dataset):
        result = compute_frs_from_dataframe(
            fair_dataset, "protected", "label", "prediction"
        )
        assert result.risk_level == RiskLevel.LOW
        assert result.score < 0.3

    def test_biased_dataset_high_risk(self, biased_dataset):
        result = compute_frs_from_dataframe(
            biased_dataset, "protected", "label", "prediction"
        )
        assert result.risk_level == RiskLevel.HIGH
        assert result.score >= 0.6

    def test_returns_correct_type(self, fair_dataset):
        result = compute_frs_from_dataframe(
            fair_dataset, "protected", "label", "prediction"
        )
        assert isinstance(result, FairnessRiskResult)
        assert isinstance(result.risk_level, RiskLevel)

    def test_custom_config_passed_through(self, fair_dataset):
        strict = ScoringConfig(risk_low_upper=0.01, risk_medium_upper=0.02)
        result = compute_frs_from_dataframe(
            fair_dataset, "protected", "label", "prediction",
            config=strict,
        )
        # Even a fair dataset will fail very strict thresholds
        assert result.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
