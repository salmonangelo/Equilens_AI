"""
Tests for the fairness_engine module.

Includes a synthetic dataset and comprehensive unit tests covering:
- Normal operation for all three metrics
- Edge cases: division by zero, NaN handling, empty groups
- Input validation errors
- The compute_all_metrics convenience function
- The FairnessEvaluator orchestrator
"""

import math

import numpy as np
import pandas as pd
import pytest

from fairness_engine.metrics import (
    MetricResult,
    disparate_impact_ratio,
    demographic_parity_difference,
    equal_opportunity_difference,
    compute_all_metrics,
)
from fairness_engine.evaluator import FairnessEvaluator


# ===================================================================
# Synthetic datasets
# ===================================================================

@pytest.fixture
def fair_dataset() -> pd.DataFrame:
    """
    A synthetic dataset where the model is roughly fair.

    100 rows, two groups of 50 each.
    Both groups have ~60% positive prediction rate.
    Ground truth positive rate ~50% per group.
    """
    np.random.seed(42)
    n = 100
    protected = np.array([0] * 50 + [1] * 50)
    # Similar prediction rates: 60% for both groups
    preds_priv = np.array([1] * 30 + [0] * 20)
    preds_unpriv = np.array([1] * 29 + [0] * 21)
    predictions = np.concatenate([preds_priv, preds_unpriv])
    # Ground truth: ~50% positive per group
    labels_priv = np.array([1] * 25 + [0] * 25)
    labels_unpriv = np.array([1] * 24 + [0] * 26)
    labels = np.concatenate([labels_priv, labels_unpriv])

    return pd.DataFrame({
        "protected": protected,
        "prediction": predictions,
        "label": labels,
    })


@pytest.fixture
def biased_dataset() -> pd.DataFrame:
    """
    A synthetic dataset where the model is clearly biased.

    Privileged group: 90% positive rate.
    Unprivileged group: 30% positive rate.
    """
    protected = np.array([0] * 50 + [1] * 50)
    preds_priv = np.array([1] * 45 + [0] * 5)
    preds_unpriv = np.array([1] * 15 + [0] * 35)
    predictions = np.concatenate([preds_priv, preds_unpriv])
    labels_priv = np.array([1] * 40 + [0] * 10)
    labels_unpriv = np.array([1] * 20 + [0] * 30)
    labels = np.concatenate([labels_priv, labels_unpriv])

    return pd.DataFrame({
        "protected": protected,
        "prediction": predictions,
        "label": labels,
    })


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Minimal 10-row dataset for hand-verifiable calculations."""
    return pd.DataFrame({
        "protected": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "prediction": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        "label":      [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    })


# ===================================================================
# MetricResult dataclass
# ===================================================================

class TestMetricResult:
    def test_creation(self):
        r = MetricResult(name="test", value=0.5, threshold=0.1, is_fair=True)
        assert r.name == "test"
        assert r.value == 0.5
        assert r.description == ""

    def test_with_description(self):
        r = MetricResult(name="x", value=0.0, threshold=0.1,
                         is_fair=True, description="ok")
        assert r.description == "ok"


# ===================================================================
# Disparate Impact Ratio
# ===================================================================

class TestDisparateImpactRatio:
    def test_hand_calculated(self, small_dataset):
        """Privileged rate=3/5=0.6, Unprivileged rate=1/5=0.2, DI=0.2/0.6=0.333"""
        result = disparate_impact_ratio(
            small_dataset, "protected", "prediction"
        )
        assert result["metric"] == "disparate_impact_ratio"
        assert abs(result["value"] - (1 / 3)) < 1e-4
        assert abs(result["privileged_rate"] - 0.6) < 1e-6
        assert abs(result["unprivileged_rate"] - 0.2) < 1e-6
        assert result["is_fair"] is False  # 0.333 < 0.8

    def test_fair_dataset(self, fair_dataset):
        result = disparate_impact_ratio(
            fair_dataset, "protected", "prediction"
        )
        # 29/50 / 30/50 = 0.9667 → within [0.8, 1.25]
        assert result["is_fair"] is True

    def test_biased_dataset(self, biased_dataset):
        result = disparate_impact_ratio(
            biased_dataset, "protected", "prediction"
        )
        # 15/50 / 45/50 = 0.333 → outside [0.8, 1.25]
        assert result["is_fair"] is False
        assert result["value"] < 0.8

    def test_division_by_zero(self):
        """When privileged group has 0% positive rate → inf."""
        df = pd.DataFrame({
            "protected": [0, 0, 0, 1, 1, 1],
            "prediction": [0, 0, 0, 1, 1, 0],
        })
        result = disparate_impact_ratio(df, "protected", "prediction")
        assert result["value"] == float("inf")
        assert result["is_fair"] is False

    def test_nan_rows_dropped(self):
        df = pd.DataFrame({
            "protected": [0, 0, 1, 1, np.nan],
            "prediction": [1, 0, 1, 0, 1],
        })
        result = disparate_impact_ratio(df, "protected", "prediction")
        assert result["rows_dropped"] == 1
        assert result["rows_used"] == 4

    def test_custom_fair_range(self, small_dataset):
        result = disparate_impact_ratio(
            small_dataset, "protected", "prediction",
            fair_range=(0.2, 1.5),
        )
        assert result["is_fair"] is True  # 0.333 ≥ 0.2


# ===================================================================
# Demographic Parity Difference
# ===================================================================

class TestDemographicParityDifference:
    def test_hand_calculated(self, small_dataset):
        """DPD = 0.2 - 0.6 = -0.4"""
        result = demographic_parity_difference(
            small_dataset, "protected", "prediction"
        )
        assert result["metric"] == "demographic_parity_difference"
        assert abs(result["value"] - (-0.4)) < 1e-6
        assert result["is_fair"] is False  # |-0.4| > 0.1

    def test_fair_dataset(self, fair_dataset):
        result = demographic_parity_difference(
            fair_dataset, "protected", "prediction"
        )
        # 29/50 - 30/50 = -0.02 → |DPD| ≤ 0.1
        assert result["is_fair"] is True

    def test_biased_dataset(self, biased_dataset):
        result = demographic_parity_difference(
            biased_dataset, "protected", "prediction"
        )
        # 0.3 - 0.9 = -0.6 → unfair
        assert result["is_fair"] is False
        assert result["value"] < 0

    def test_perfect_parity(self):
        df = pd.DataFrame({
            "protected": [0, 0, 1, 1],
            "prediction": [1, 0, 1, 0],
        })
        result = demographic_parity_difference(df, "protected", "prediction")
        assert result["value"] == 0.0
        assert result["is_fair"] is True

    def test_custom_threshold(self, small_dataset):
        result = demographic_parity_difference(
            small_dataset, "protected", "prediction",
            threshold=0.5,
        )
        assert result["is_fair"] is True  # |-0.4| ≤ 0.5


# ===================================================================
# Equal Opportunity Difference
# ===================================================================

class TestEqualOpportunityDifference:
    def test_hand_calculated(self, small_dataset):
        """
        Privileged actual positives: rows 0,1 → labels [1,1], preds [1,1] → TPR=1.0
        Unprivileged actual positives: rows 5,6 → labels [1,1], preds [1,0] → TPR=0.5
        EOD = 0.5 - 1.0 = -0.5
        """
        result = equal_opportunity_difference(
            small_dataset, "protected", "label", "prediction"
        )
        assert result["metric"] == "equal_opportunity_difference"
        assert abs(result["value"] - (-0.5)) < 1e-6
        assert abs(result["privileged_tpr"] - 1.0) < 1e-6
        assert abs(result["unprivileged_tpr"] - 0.5) < 1e-6
        assert result["is_fair"] is False

    def test_fair_dataset(self, fair_dataset):
        result = equal_opportunity_difference(
            fair_dataset, "protected", "label", "prediction"
        )
        # Check it returns a valid result with tpr values
        assert "privileged_tpr" in result
        assert "unprivileged_tpr" in result

    def test_no_positives_in_group(self):
        """When a group has zero actual positives, TPR is undefined → NaN."""
        df = pd.DataFrame({
            "protected": [0, 0, 0, 1, 1, 1],
            "prediction": [1, 1, 0, 1, 0, 0],
            "label":      [1, 1, 0, 0, 0, 0],  # no Y=1 in unprivileged
        })
        result = equal_opportunity_difference(
            df, "protected", "label", "prediction"
        )
        assert math.isnan(result["unprivileged_tpr"])
        assert result["is_fair"] is False

    def test_defaults_to_target_col(self, small_dataset):
        """When prediction_col is None, use target_col → TPR=1.0 for both."""
        result = equal_opportunity_difference(
            small_dataset, "protected", "label"
        )
        # Self-evaluation: every actual positive is "predicted" positive
        assert abs(result["privileged_tpr"] - 1.0) < 1e-6
        assert abs(result["unprivileged_tpr"] - 1.0) < 1e-6
        assert result["value"] == 0.0


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
        with pytest.raises(ValueError, match="not found"):
            disparate_impact_ratio(df, "missing", "b")

    def test_non_binary_column_raises(self):
        df = pd.DataFrame({"p": [0, 1, 2], "t": [1, 0, 1]})
        with pytest.raises(ValueError, match="binary"):
            disparate_impact_ratio(df, "p", "t")

    def test_all_nan_raises(self):
        df = pd.DataFrame({
            "p": [np.nan, np.nan],
            "t": [np.nan, np.nan],
        })
        with pytest.raises(ValueError, match="No valid"):
            disparate_impact_ratio(df, "p", "t")

    def test_eod_missing_prediction_col_raises(self):
        df = pd.DataFrame({"p": [0, 1], "t": [1, 0]})
        with pytest.raises(ValueError, match="not found"):
            equal_opportunity_difference(df, "p", "t", "nonexistent")


# ===================================================================
# compute_all_metrics
# ===================================================================

class TestComputeAllMetrics:
    def test_returns_all_three(self, small_dataset):
        results = compute_all_metrics(
            small_dataset, "protected", "label", "prediction"
        )
        assert "disparate_impact_ratio" in results
        assert "demographic_parity_difference" in results
        assert "equal_opportunity_difference" in results

    def test_each_has_expected_keys(self, small_dataset):
        results = compute_all_metrics(
            small_dataset, "protected", "label", "prediction"
        )
        for key, val in results.items():
            assert "metric" in val
            assert "value" in val
            assert "is_fair" in val


# ===================================================================
# FairnessEvaluator
# ===================================================================

class TestFairnessEvaluator:
    def test_evaluate_single_attribute(self, biased_dataset):
        evaluator = FairnessEvaluator(model_name="test_model")
        report = evaluator.evaluate(
            df=biased_dataset,
            protected_cols=["protected"],
            target_col="label",
            prediction_col="prediction",
        )
        assert report.model_name == "test_model"
        assert "protected" in report.results
        assert report.overall_fair is False  # biased dataset

    def test_evaluate_fair_dataset(self, fair_dataset):
        evaluator = FairnessEvaluator(model_name="fair_model")
        report = evaluator.evaluate(
            df=fair_dataset,
            protected_cols=["protected"],
            target_col="label",
            prediction_col="prediction",
        )
        assert report.summary.endswith("FAIR.")

    def test_report_summary_format(self, small_dataset):
        evaluator = FairnessEvaluator()
        report = evaluator.evaluate(
            df=small_dataset,
            protected_cols=["protected"],
            target_col="label",
            prediction_col="prediction",
        )
        assert "1 protected attribute(s)" in report.summary
