"""
EquiLens AI — Full Analysis Pipeline Orchestrator

End-to-end analysis workflow:
1. Load dataset (CSV or DataFrame)
2. Run anonymization + metrics computation
3. Validate privacy constraints
4. Generate Gemini explanation (optional)
5. Return comprehensive JSON result

This is the PRIMARY entry point for fairness auditing.
"""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.analysis_service import run_analysis
from backend.services.explanation_service_v2 import (
    generate_fairness_explanation,
    create_fallback_response,
    FairnessAnalysisResponse,
)
from prompts.fairness_analyst import FairnessMetricsInput
from backend.privacy.validator import (
    validate_gemini_payload,
    PrivacyValidationError,
    PrivacyValidator,
)


logger = logging.getLogger(__name__)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid CSV
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    try:
        df = pd.read_csv(path)
        logger.info(f"✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")


async def run_full_analysis(
    file_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    *,
    protected_cols: list[str],
    target_col: str,
    prediction_col: str | None = None,
    model_name: str = "audit_model",
    use_case: str = "general",
    application_domain: str = "not_specified",
    regulatory_requirements: str = "",
    generate_explanation: bool = True,
    min_group_size: int = 10,
) -> dict[str, Any]:
    """
    Execute a complete fairness audit end-to-end.

    This is the PRIMARY API for running EquiLens fairness analysis.

    Pipeline:
    1. Load dataset (CSV or accept DataFrame)
    2. Anonymize PII columns
    3. Compute fairness metrics (DI, DPD, EOD, FRS)
    4. Validate privacy (no raw data to external APIs)
    5. Generate Gemini explanation (optional, with fallback)
    6. Return comprehensive report

    Args:
        file_path: Path to CSV file (mutually exclusive with df)
        df: Pre-loaded DataFrame (mutually exclusive with file_path)
        protected_cols: List of binary protected attribute column names
        target_col: Binary ground-truth label column
        prediction_col: Binary prediction column (defaults to target_col)
        model_name: Label for the model being audited
        use_case: Use case description (e.g., "loan_approval")
        application_domain: Domain (e.g., "lending")
        regulatory_requirements: Applicable regulations (e.g., "Fair Lending, EU AI Act")
        generate_explanation: Whether to call Gemini for AI explanation (default: True)
        min_group_size: Minimum samples per group for privacy validation (default: 10)

    Returns:
        Comprehensive JSON report with:
        - report_id, timestamp, model_name
        - dataset shape
        - anonymization summary
        - per-attribute metrics and risk scores
        - overall fairness verdict
        - (optional) AI-generated explanation with remediation strategies

    Raises:
        ValueError: On missing/invalid dataset or columns
        PrivacyValidationError: If privacy constraints violated
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting full EquiLens fairness analysis pipeline...")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if file_path is not None:
        logger.info(f"Loading dataset from: {file_path}")
        dataset = load_dataset(file_path)
    elif df is not None:
        logger.info(f"Using provided DataFrame ({len(df)} rows)")
        dataset = df.copy()
    else:
        raise ValueError("Must provide either file_path or df parameter")

    # ------------------------------------------------------------------
    # 2. Run core analysis (anonymization + metrics)
    # ------------------------------------------------------------------
    logger.info("Running core analysis (anonymization + metrics)...")
    analysis_result = run_analysis(
        dataset,
        protected_cols=protected_cols,
        target_col=target_col,
        prediction_col=prediction_col,
        model_name=model_name,
        skip_anonymization=False,
    )
    logger.info("✓ Core analysis complete")

    # ------------------------------------------------------------------
    # 3. Validate privacy constraints (before external API calls)
    # ------------------------------------------------------------------
    logger.info("Validating privacy constraints...")
    privacy_validator = PrivacyValidator(min_group_size=min_group_size)
    
    for attr_name, attr_data in analysis_result.get("per_attribute", {}).items():
        metrics = attr_data.get("metrics", {})
        total_rows = analysis_result.get("dataset_shape", {}).get("rows", 1)

        # Extract group statistics from metrics
        di_metric = metrics.get("disparate_impact_ratio", {})
        priv_rate = di_metric.get("privileged_rate", 0)
        unpriv_rate = di_metric.get("unprivileged_rate", 0)
        priv_count = int(priv_rate * total_rows) if priv_rate > 0 else 0
        unpriv_count = int(unpriv_rate * total_rows) if unpriv_rate > 0 else 0

        # Build payload for privacy validation
        payload = {
            "model_name": model_name,
            "protected_attribute": attr_name,
            "disparate_impact_ratio": di_metric.get("value", 0),
            "demographic_parity_difference": metrics.get("demographic_parity_difference", {}).get("value", 0),
            "equal_opportunity_difference": metrics.get("equal_opportunity_difference", {}).get("value", 0),
            "group_statistics": {
                "privileged": {
                    "sample_size": priv_count,
                    "percentage": priv_rate * 100
                },
                "unprivileged": {
                    "sample_size": unpriv_count,
                    "percentage": unpriv_rate * 100
                }
            },
            "total_samples": total_rows,
            "use_case": use_case,
            "application_domain": application_domain,
            "regulatory_requirements": regulatory_requirements,
            "data_quality_notes": "",
        }

        try:
            result = privacy_validator.validate_payload_for_gemini(payload)
            if not result.is_valid:
                raise result.errors[0]
            logger.info(f"✓ Privacy validation passed for {attr_name}")
        except PrivacyValidationError as e:
            logger.error(f"✗ Privacy validation FAILED for {attr_name}: {e}")
            raise

    # ------------------------------------------------------------------
    # 4. Generate explanation (Gemini)
    # ------------------------------------------------------------------
    explanation_result = None
    if generate_explanation:
        logger.info("Generating AI-powered fairness explanation...")
        try:
            # Extract first protected attribute for explanation
            first_attr = protected_cols[0] if protected_cols else "primary_attribute"
            first_attr_data = analysis_result.get("per_attribute", {}).get(first_attr, {})
            metrics_obj = first_attr_data.get("metrics", {})
            total_rows = analysis_result.get("dataset_shape", {}).get("rows", 1)

            # Extract group statistics from metrics
            di_metric = metrics_obj.get("disparate_impact_ratio", {})
            priv_rate = di_metric.get("privileged_rate", 0)
            unpriv_rate = di_metric.get("unprivileged_rate", 0)
            priv_count = int(priv_rate * total_rows) if priv_rate > 0 else 0
            unpriv_count = int(unpriv_rate * total_rows) if unpriv_rate > 0 else 0

            metrics_input = FairnessMetricsInput(
                model_name=model_name,
                protected_attribute=first_attr,
                disparate_impact_ratio=di_metric.get("value", 0),
                demographic_parity_difference=metrics_obj.get("demographic_parity_difference", {}).get("value", 0),
                equal_opportunity_difference=metrics_obj.get("equal_opportunity_difference", {}).get("value", 0),
                group_statistics={
                    "privileged": {
                        "sample_size": priv_count,
                        "percentage": priv_rate * 100
                    },
                    "unprivileged": {
                        "sample_size": unpriv_count,
                        "percentage": unpriv_rate * 100
                    }
                },
                total_samples=total_rows,
                use_case=use_case,
                application_domain=application_domain,
                regulatory_requirements=regulatory_requirements,
                data_quality_notes=f"Anonymization completed. {len(dataset)} samples analyzed.",
            )

            # Try Gemini explanation
            explanation_result = await generate_fairness_explanation(metrics_input)

            if explanation_result:
                logger.info("✓ Gemini explanation generated successfully")
            else:
                logger.warning("Gemini explanation failed; using fallback")
                explanation_result = create_fallback_response(
                    metrics_input,
                    reason="Gemini API call failed or privacy validation rejected request"
                )
        except Exception as e:
            logger.error(f"Exception during explanation generation: {e}")
            logger.info("Proceeding with fallback response...")
            try:
                explanation_result = create_fallback_response(
                    metrics_input,
                    reason=f"Exception: {str(e)}"
                )
            except:
                explanation_result = None

    # ------------------------------------------------------------------
    # 5. Assemble final report
    # ------------------------------------------------------------------
    logger.info("Assembling final report...")
    final_report = {
        "report_id": analysis_result.get("report_id"),
        "timestamp": analysis_result.get("timestamp"),
        "model_name": model_name,
        "dataset_shape": analysis_result.get("dataset_shape"),
        "anonymization": analysis_result.get("anonymization"),
        "protected_attributes": protected_cols,
        "per_attribute": analysis_result.get("per_attribute"),
        "overall": analysis_result.get("overall"),
        "explanation": (
            explanation_result.model_dump() if explanation_result else None
        ),
        "pipeline_status": "success",
    }

    logger.info("=" * 80)
    logger.info(f"✓ Analysis complete: {model_name}")
    logger.info(f"  Report ID: {final_report['report_id']}")
    logger.info(f"  Overall fairness: {final_report['overall'].get('is_fair')}")
    logger.info("=" * 80)

    return final_report


# Synchronous wrapper for environments without async
def run_full_analysis_sync(
    file_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    *,
    protected_cols: list[str],
    target_col: str,
    prediction_col: str | None = None,
    model_name: str = "audit_model",
    use_case: str = "general",
    application_domain: str = "not_specified",
    regulatory_requirements: str = "",
    generate_explanation: bool = True,
    min_group_size: int = 10,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_full_analysis.

    Use this in CLI/demo scripts or non-async contexts.
    """
    return asyncio.run(
        run_full_analysis(
            file_path=file_path,
            df=df,
            protected_cols=protected_cols,
            target_col=target_col,
            prediction_col=prediction_col,
            model_name=model_name,
            use_case=use_case,
            application_domain=application_domain,
            regulatory_requirements=regulatory_requirements,
            generate_explanation=generate_explanation,
            min_group_size=min_group_size,
        )
    )
