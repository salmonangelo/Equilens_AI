"""
EquiLens AI — Fairness Evaluator

Orchestrates the evaluation pipeline: validates inputs, applies all
fairness metrics per protected attribute, aggregates results, and
generates structured fairness reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fairness_engine.metrics import (
    compute_all_metrics,
    disparate_impact_ratio,
    demographic_parity_difference,
    equal_opportunity_difference,
)


@dataclass
class EvaluationReport:
    """Aggregated fairness evaluation report."""

    model_name: str
    protected_attributes: list[str]
    results: dict = field(default_factory=dict)
    overall_fair: bool = True
    summary: str = ""


class FairnessEvaluator:
    """
    Main evaluator class that runs a suite of fairness metrics
    against model predictions for one or more protected attributes.

    Usage:
        evaluator = FairnessEvaluator(model_name="loan_model")
        report = evaluator.evaluate(
            df=data,
            protected_cols=["gender", "race"],
            target_col="approved",
            prediction_col="predicted",
        )
    """

    def __init__(self, model_name: str = "unnamed_model") -> None:
        self.model_name = model_name

    def evaluate(
        self,
        df: pd.DataFrame,
        protected_cols: list[str],
        target_col: str,
        prediction_col: str | None = None,
        *,
        privileged_value: int = 0,
    ) -> EvaluationReport:
        """
        Run all fairness metrics for each protected attribute and return
        an aggregated report.

        Args:
            df: DataFrame containing the data.
            protected_cols: List of binary protected attribute column names.
            target_col: Binary ground-truth label column.
            prediction_col: Binary prediction column (defaults to target_col).
            privileged_value: Value identifying the privileged group.

        Returns:
            EvaluationReport with per-attribute metric results.
        """
        report = EvaluationReport(
            model_name=self.model_name,
            protected_attributes=protected_cols,
        )

        all_fair = True

        for col in protected_cols:
            metrics = compute_all_metrics(
                df=df,
                protected_col=col,
                target_col=target_col,
                prediction_col=prediction_col,
                privileged_value=privileged_value,
            )
            report.results[col] = metrics

            # Check if any metric is unfair
            for metric_result in metrics.values():
                if not metric_result.get("is_fair", True):
                    all_fair = False

        report.overall_fair = all_fair
        report.summary = (
            f"Evaluated {len(protected_cols)} protected attribute(s) "
            f"across 3 metrics. Overall: {'FAIR' if all_fair else 'UNFAIR'}."
        )

        return report
