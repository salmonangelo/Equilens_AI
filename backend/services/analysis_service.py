"""
EquiLens AI — Analysis Service

Orchestrates the full analysis pipeline:
    1. Anonymize PII columns (preserving protected attributes).
    2. Compute fairness metrics per protected attribute.
    3. Compute composite Fairness Risk Scores.

Returns a single consolidated JSON-serializable result dict.
No external AI APIs — fully deterministic.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from fairness_engine.anonymizer import (
    anonymize,
    AnonymizationConfig,
    get_anonymization_summary,
)
from fairness_engine.evaluator import FairnessEvaluator
from fairness_engine.scoring import (
    compute_frs_from_metrics,
    FairnessRiskResult,
)
from fairness_engine.metrics import compute_all_metrics


def _sanitize_value(val: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python primitives."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if math.isnan(val) or math.isinf(val):
            return str(val)
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(v) for v in val]
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return str(val)
    return val


def _frs_to_dict(result: FairnessRiskResult) -> dict[str, Any]:
    """Convert a FairnessRiskResult dataclass to a JSON-safe dict."""
    d = asdict(result)
    d["risk_level"] = result.risk_level.value
    d["weights"] = list(result.weights)
    return _sanitize_value(d)


def run_analysis(
    df: pd.DataFrame,
    *,
    protected_cols: list[str],
    target_col: str,
    prediction_col: str | None = None,
    model_name: str = "unnamed_model",
    skip_anonymization: bool = False,
) -> dict[str, Any]:
    """
    Execute the full EquiLens analysis pipeline.

    Args:
        df: Raw dataset DataFrame.
        protected_cols: Binary protected attribute column names.
        target_col: Binary ground-truth label column.
        prediction_col: Binary prediction column (defaults to target_col).
        model_name: Label for the model being audited.
        skip_anonymization: If True, skip PII detection/anonymization.

    Returns:
        Consolidated JSON-serializable result dictionary with sections:
            - report_id, timestamp, model_name
            - anonymization (summary of PII handling)
            - fairness_metrics (per-attribute metric breakdown)
            - risk_scores (per-attribute FRS)
            - overall (aggregate verdict)

    Raises:
        ValueError: On missing columns or invalid data.
    """
    pred_col = prediction_col or target_col
    report_id = f"rpt_{uuid.uuid4().hex[:10]}"

    # ------------------------------------------------------------------
    # 1. Validate required columns exist
    # ------------------------------------------------------------------
    required = set(protected_cols) | {target_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 2. Anonymization
    # ------------------------------------------------------------------
    anonymization_result: dict[str, Any] = {"skipped": True, "summary": {}}

    if not skip_anonymization:
        anon_config = AnonymizationConfig(
            skip_columns=list(required),  # preserve analysis columns
        )
        anon_df, detections = anonymize(df, anon_config, report=False)
        anonymization_result = {
            "skipped": False,
            "summary": get_anonymization_summary(detections),
            "rows_after": len(anon_df),
            "columns_after": len(anon_df.columns),
        }
    else:
        anon_df = df.copy()

    # ------------------------------------------------------------------
    # 3. Fairness metrics + risk scores  (per protected attribute)
    # ------------------------------------------------------------------
    per_attribute: dict[str, Any] = {}
    overall_fair = True

    for col in protected_cols:
        # --- Metrics ---
        metrics = compute_all_metrics(
            anon_df,
            protected_col=col,
            target_col=target_col,
            prediction_col=pred_col,
        )

        # --- Risk score ---
        frs = compute_frs_from_metrics(metrics)

        # --- Per-metric fairness check ---
        attr_fair = all(
            m.get("is_fair", True) for m in metrics.values()
        )
        if not attr_fair:
            overall_fair = False

        per_attribute[col] = {
            "metrics": _sanitize_value(metrics),
            "risk_score": _frs_to_dict(frs),
            "is_fair": attr_fair,
        }

    # ------------------------------------------------------------------
    # 4. Evaluator summary (cross-attribute)
    # ------------------------------------------------------------------
    evaluator = FairnessEvaluator(model_name=model_name)
    eval_report = evaluator.evaluate(
        anon_df,
        protected_cols=protected_cols,
        target_col=target_col,
        prediction_col=pred_col,
    )

    # ------------------------------------------------------------------
    # 5. Assemble response
    # ------------------------------------------------------------------
    return {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "anonymization": anonymization_result,
        "protected_attributes": protected_cols,
        "per_attribute": per_attribute,
        "overall": {
            "is_fair": overall_fair,
            "summary": eval_report.summary,
        },
    }
