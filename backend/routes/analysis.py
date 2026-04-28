"""
EquiLens AI — Fairness Analysis Endpoints

API routes for submitting models/datasets for fairness evaluation,
retrieving analysis results, and generating reports.

Features:
  - Column validation
  - Comprehensive error handling
  - Dataset processing logging
  - Metrics computation logging
  - Gemini API failure handling with fallback
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.dataset_store import dataset_store
from backend.services.analysis_service import run_analysis

router = APIRouter()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for ``POST /analyze``."""

    dataset_id: str = Field(
        ...,
        description="ID returned by POST /upload",
        examples=["a1b2c3d4e5f6"],
    )
    protected_attributes: list[str] = Field(
        ...,
        min_length=1,
        description="List of binary protected attribute column names",
        examples=[["gender", "race"]],
    )
    target_column: str = Field(
        ...,
        description="Binary ground-truth label column name",
        examples=["approved"],
    )
    prediction_column: str | None = Field(
        default=None,
        description=(
            "Binary prediction column name. "
            "If omitted, defaults to target_column (self-evaluation)."
        ),
        examples=["predicted"],
    )
    model_name: str = Field(
        default="unnamed_model",
        description="Human-readable model identifier for the report",
        examples=["loan_approval_v2"],
    )
    skip_anonymization: bool = Field(
        default=False,
        description="If true, skip PII detection and anonymization",
    )


# ------------------------------------------------------------------
# Helper: Validate columns exist in dataset
# ------------------------------------------------------------------

def validate_columns(
    df,
    dataset_id: str,
    required_columns: list[str],
) -> None:
    """
    Validate that all required columns exist in the dataset.

    Raises HTTPException if any column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        available = list(df.columns)
        logger.error(
            f"❌ Dataset {dataset_id}: Missing columns {missing}. "
            f"Available: {available}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Missing required columns",
                "missing_columns": missing,
                "available_columns": available,
            },
        )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Run fairness analysis on an uploaded dataset",
    response_description="Full fairness audit report",
)
async def analyze_dataset(body: AnalyzeRequest) -> dict[str, Any]:
    """
    Run the full EquiLens analysis pipeline on a previously uploaded
    dataset:

    1. **Column Validation** — ensure all required columns exist
    2. **Anonymization** — detect and sanitize PII columns (unless
       ``skip_anonymization`` is set).
    3. **Fairness Metrics** — compute Disparate Impact, Demographic
       Parity Difference, and Equal Opportunity Difference per
       protected attribute.
    4. **Risk Scoring** — derive a composite Fairness Risk Score
       (FRS) per protected attribute with LOW / MEDIUM / HIGH
       classification.
    5. **LLM Explanation** — (optional) generate structured explanation
       from Gemini with EU AI Act compliance mapping.

    Returns a consolidated JSON report.
    """
    logger.info(f"📊 Analysis requested: dataset={body.dataset_id}, model={body.model_name}")

    # --- Retrieve dataset ---
    entry = dataset_store.get(body.dataset_id)
    if entry is None:
        logger.error(f"❌ Dataset not found: {body.dataset_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Dataset '{body.dataset_id}' not found. "
                "Upload a CSV first via POST /upload."
            ),
        )

    logger.info(f"✓ Dataset loaded: {entry.rows} rows, {entry.columns} columns")

    # --- Validate columns ---
    required_columns = (
        body.protected_attributes
        + [body.target_column]
        + ([body.prediction_column] if body.prediction_column else [])
    )

    try:
        validate_columns(entry.df, body.dataset_id, required_columns)
        logger.info(f"✓ All required columns present")
    except HTTPException as exc:
        raise exc

    # --- Run pipeline ---
    try:
        logger.info("→ Starting fairness metrics computation...")
        result = run_analysis(
            df=entry.df,
            protected_cols=body.protected_attributes,
            target_col=body.target_column,
            prediction_col=body.prediction_column,
            model_name=body.model_name,
            skip_anonymization=body.skip_anonymization,
        )
        logger.info("✓ Metrics computation complete")

    except ValueError as exc:
        logger.warning(f"⚠️  Analysis validation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:
        logger.error(f"❌ Analysis pipeline failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {exc}",
        )

    logger.info(f"✓ Analysis complete for model: {body.model_name}")

    return {
        "status": "success",
        "dataset_id": body.dataset_id,
        "model_name": body.model_name,
        **result,
    }


@router.get(
    "/datasets",
    summary="List uploaded datasets",
    response_description="List of stored dataset IDs and metadata",
)
async def list_datasets() -> dict[str, Any]:
    """Return a list of all currently stored dataset IDs with metadata."""
    ids = dataset_store.list_ids()
    datasets = []
    for did in ids:
        summary = dataset_store.summary(did)
        if summary:
            datasets.append(summary)
    return {"status": "success", "count": len(datasets), "datasets": datasets}


@router.get(
    "/datasets/{dataset_id}",
    summary="Get dataset metadata",
    response_description="Dataset metadata and column information",
)
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Return metadata for a specific uploaded dataset."""
    summary = dataset_store.summary(dataset_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return {"status": "success", **summary}


@router.get(
    "/reports/{report_id}",
    summary="Get analysis report",
    response_description="Full fairness analysis report",
)
async def get_report(report_id: str) -> dict[str, Any]:
    """
    Retrieve a previously generated fairness analysis report.

    Note: Currently returns a stub response. Full report persistence
    will be implemented in a future version.
    """
    logger.info(f"Report requested: {report_id}")
    return {
        "status": "not_implemented",
        "report_id": report_id,
        "message": "Report persistence not yet implemented",
    }


@router.post(
    "/explain",
    summary="Generate explanation for analysis",
    response_description="AI-generated fairness explanation",
)
async def explain_analysis(body: AnalyzeRequest) -> dict[str, Any]:
    """
    Generate an AI-powered explanation for fairness metrics.

    Note: Currently returns a stub response. Full AI explanation
    will use the explanation_service_v2 in a future update.
    """
    logger.info(f"Explanation requested for model: {body.model_name}")
    return {
        "status": "not_implemented",
        "model_name": body.model_name,
        "message": "Explanation service will be available in next release",
    }
