"""
EquiLens AI — API Schemas

Pydantic models for request validation and response serialization.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# === Request Models ===


class AnalysisRequest(BaseModel):
    """Request body for fairness analysis submission."""

    model_name: str = Field(..., description="Name or identifier of the model")
    dataset_id: str = Field(..., description="Reference to the dataset")
    protected_attributes: list[str] = Field(
        ..., description="List of protected attribute column names"
    )
    predictions_column: str = Field(
        default="prediction", description="Column name for model predictions"
    )
    labels_column: str = Field(
        default="label", description="Column name for ground truth labels"
    )
    metrics: list[str] = Field(
        default=["spd", "di", "eod"],
        description="List of fairness metrics to compute",
    )


class ExplanationRequest(BaseModel):
    """Request body for LLM-powered bias explanation."""

    report_id: str = Field(..., description="ID of the analysis report to explain")
    detail_level: str = Field(
        default="standard",
        description="Level of detail: 'brief', 'standard', or 'detailed'",
    )
    audience: str = Field(
        default="technical",
        description="Target audience: 'technical', 'executive', or 'general'",
    )


# === Response Models ===


class MetricResponse(BaseModel):
    """Single fairness metric result."""

    name: str
    value: float
    threshold: float
    is_fair: bool
    description: str = ""


class AnalysisResponse(BaseModel):
    """Response body for fairness analysis."""

    report_id: str
    model_name: str
    metrics: list[MetricResponse] = []
    overall_fair: bool = True
    summary: str = ""


class ExplanationResponse(BaseModel):
    """Response body for LLM-generated explanation."""

    report_id: str
    explanation: str = ""
    recommendations: list[str] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
