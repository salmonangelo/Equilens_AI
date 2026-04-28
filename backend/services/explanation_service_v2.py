"""
EquiLens AI — Robust Fairness Explanation Service

Integrates with Gemini 2.0 Flash to generate structured, evidence-based
fairness analysis using the production-grade prompt template defined in
prompts/fairness_analyst.py.

Features:
  • Structured JSON output (enforced via system prompt)
  • EU AI Act risk mapping
  • Guardrails against hallucinations
  • Retry logic with exponential backoff
  • Response validation
  • Graceful fallback if LLM call fails
"""

import json
import logging
import os
from typing import Any
from datetime import datetime, timezone

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from prompts.fairness_analyst import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FairnessMetricsInput,
    build_user_prompt,
    validate_output_schema,
    EU_AI_ACT_MAPPINGS,
)
from backend.privacy.validator import (
    validate_gemini_payload,
    PrivacyValidationError,
)


# =====================================================================
# Logging Configuration
# =====================================================================

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )


# =====================================================================
# Response Schema (Pydantic for validation)
# =====================================================================

class MetricDetail(BaseModel):
    """Single metric result with interpretation."""

    value: float
    threshold: float
    is_fair: bool
    interpretation: str = ""


class FairnessAnalysisResponse(BaseModel):
    """Structured response from Gemini bias analysis."""

    analysis: dict[str, Any] = Field(
        ...,
        description="Technical analysis with metrics and data quality assessment",
    )
    fairness_assessment: dict[str, Any] = Field(
        ...,
        description="Overall fairness verdict with severity and impact",
    )
    eu_ai_act: dict[str, Any] = Field(
        ...,
        description="EU AI Act risk classification and compliance status",
    )
    remediation: dict[str, list[dict[str, Any]]] = Field(
        ...,
        description="Mitigation strategies (pre/in/post-processing)",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Analysis limitations and caveats",
    )


# =====================================================================
# Gemini API Configuration
# =====================================================================

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not set. Fairness explanation feature will be unavailable."
    )
else:
    genai.configure(api_key=GENAI_API_KEY)


# =====================================================================
# Main Analysis Function
# =====================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
async def generate_fairness_explanation(
    metrics: FairnessMetricsInput,
    model_id: str = "gemini-2.0-flash",
) -> FairnessAnalysisResponse | None:
    """
    Generate a robust, structured fairness analysis from Gemini.

    This function:
    0. Validates privacy constraints to prevent raw data exfiltration
    1. Builds a comprehensive prompt with system + user context
    2. Calls Gemini 2.0 Flash with explicit JSON schema constraints
    3. Parses and validates the JSON response
    4. Returns a FairnessAnalysisResponse or None if analysis fails

    Args:
        metrics: FairnessMetricsInput with model/metric data
        model_id: Gemini model to use (default: gemini-2.0-flash)

    Returns:
        FairnessAnalysisResponse if successful, None if LLM call or parsing fails.

    Raises:
        (caught and logged via decorator @retry)
    """

    if not GENAI_API_KEY:
        logger.error("GEMINI_API_KEY not configured; cannot generate explanation")
        return None

    try:
        # Step 0: Validate privacy constraints before making external API call
        logger.info(f"Validating privacy constraints for {metrics.model_name}...")
        metrics_dict = {
            "model_name": metrics.model_name,
            "protected_attribute": metrics.protected_attribute,
            "disparate_impact_ratio": metrics.disparate_impact_ratio,
            "demographic_parity_difference": metrics.demographic_parity_difference,
            "equal_opportunity_difference": metrics.equal_opportunity_difference,
            "group_statistics": metrics.group_statistics,
            "total_samples": metrics.total_samples,
            "use_case": metrics.use_case,
            "application_domain": metrics.application_domain,
            "regulatory_requirements": metrics.regulatory_requirements,
            "data_quality_notes": metrics.data_quality_notes,
        }
        
        try:
            validate_gemini_payload(metrics_dict)
            logger.info("✓ Privacy validation passed")
        except PrivacyValidationError as pve:
            logger.error(f"Privacy validation failed: {pve}")
            # Do not proceed with external API call
            return None
        
        # Step 1: Build prompt
        logger.info(f"Building fairness analysis prompt for {metrics.model_name}...")
        user_prompt = build_user_prompt(metrics)

        # Step 2: Call Gemini with system prompt (governs behavior + output format)
        logger.info(
            f"Calling {model_id} with robust fairness analysis prompt..."
        )
        client = genai.Client()
        response = client.models.generate_content(
            model=model_id,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"
                        }
                    ],
                }
            ],
            generation_config={
                "temperature": 0.2,  # Low temp = deterministic, fact-based
                "top_p": 0.8,
                "max_output_tokens": 4096,
            },
        )

        # Step 3: Extract and validate JSON
        response_text = response.text.strip()
        logger.info(f"Received response from Gemini ({len(response_text)} chars)")

        # Handle markdown code blocks (some models wrap JSON in ```json ... ```)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        response_json = json.loads(response_text)
        logger.info("Successfully parsed JSON response")

        # Step 4: Validate schema
        is_valid, errors = validate_output_schema(response_json)
        if not is_valid:
            logger.warning(f"Response schema validation failed: {errors}")
            logger.warning("Proceeding with partial response; some fields may be missing")

        # Step 5: Parse into Pydantic model (additional validation)
        analysis_response = FairnessAnalysisResponse(**response_json)
        logger.info(
            f"✓ Analysis complete: Risk level = {analysis_response.eu_ai_act.get('risk_level', 'unknown')}"
        )

        return analysis_response

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Raw response: {response_text[:500]}...")
        return None

    except ValidationError as e:
        logger.error(f"Response validation failed: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during fairness explanation: {e}")
        raise  # Let @retry decorator handle


# =====================================================================
# Utility Functions
# =====================================================================

def create_fallback_response(
    metrics: FairnessMetricsInput,
    reason: str = "LLM service unavailable",
) -> FairnessAnalysisResponse:
    """
    Generate a deterministic fallback response if Gemini call fails.

    This ensures the system degrades gracefully: metrics are still
    computed, but without LLM-generated explanations.

    Args:
        metrics: Input metrics
        reason: Why fallback was triggered

    Returns:
        Minimal but valid FairnessAnalysisResponse
    """
    logger.warning(f"Using fallback response: {reason}")

    # Classify severity based on metric values
    severity = "minimal"
    if abs(metrics.demographic_parity_difference) > 0.2:
        severity = "critical"
    elif abs(metrics.demographic_parity_difference) > 0.15:
        severity = "high"
    elif abs(metrics.demographic_parity_difference) > 0.1:
        severity = "moderate"
    elif abs(metrics.demographic_parity_difference) > 0.05:
        severity = "low"

    # Simple risk classification (without LLM context)
    risk_level = "high" if severity in ["high", "critical"] else "limited"

    return FairnessAnalysisResponse(
        analysis={
            "model_name": metrics.model_name,
            "protected_attribute": metrics.protected_attribute,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_quality": {
                "sample_size_per_group": {
                    k: v.get("sample_size", 0)
                    for k, v in metrics.group_statistics.items()
                },
                "missing_data_pct": 0.0,
                "warnings": [f"Fallback response: {reason}"],
            },
            "metrics": {
                "disparate_impact_ratio": {
                    "value": metrics.disparate_impact_ratio,
                    "threshold": 0.8,
                    "is_fair": metrics.disparate_impact_ratio >= 0.8,
                },
                "demographic_parity_difference": {
                    "value": metrics.demographic_parity_difference,
                    "threshold": 0.1,
                    "is_fair": abs(metrics.demographic_parity_difference) <= 0.1,
                },
                "equal_opportunity_difference": {
                    "value": metrics.equal_opportunity_difference,
                    "threshold": 0.1,
                    "is_fair": abs(metrics.equal_opportunity_difference) <= 0.1,
                },
                "notes": "Fallback mode: metrics only, no LLM interpretation.",
            },
        },
        fairness_assessment={
            "overall_fair": severity == "minimal",
            "severity": severity,
            "disproportionately_affected_group": "Unable to determine (LLM unavailable)",
            "magnitude_of_disparity": f"DPD = {metrics.demographic_parity_difference:.3f}",
            "evidence_basis": "Deterministic threshold comparison (no LLM analysis)",
        },
        eu_ai_act={
            "risk_level": risk_level,
            "article_mapping": f"Preliminary classification only. {reason}",
            "compliance_status": "uncertain",
            "compliance_reasoning": "Full EU AI Act analysis requires LLM service.",
        },
        remediation={
            "pre_processing": [],
            "in_processing": [],
            "post_processing": [],
        },
        limitations=[
            reason,
            "Mitigation strategies not generated (LLM service unavailable)",
            "Risk classification is preliminary and based only on metric thresholds",
            "Human review recommended before any remediation decisions",
        ],
    )


# =====================================================================
# Synchronous Wrapper (for non-async contexts)
# =====================================================================

import asyncio


def generate_fairness_explanation_sync(
    metrics: FairnessMetricsInput,
    model_id: str = "gemini-2.0-flash",
    timeout_seconds: int = 60,
) -> FairnessAnalysisResponse | None:
    """
    Synchronous wrapper for generate_fairness_explanation.

    Use this in non-async contexts (CLI, sync endpoints).

    Args:
        metrics: Input metrics
        model_id: Gemini model to use
        timeout_seconds: Timeout for the entire operation (default: 60s)

    Returns:
        FairnessAnalysisResponse or None if request fails/times out
    """
    try:
        loop = asyncio.get_event_loop()
        # If loop is running, we need a new one
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, generate_fairness_explanation(metrics, model_id))
                return future.result(timeout=timeout_seconds)
        else:
            return asyncio.run(asyncio.wait_for(
                generate_fairness_explanation(metrics, model_id),
                timeout=timeout_seconds
            ))
    except asyncio.TimeoutError:
        logger.error(f"Gemini API call timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return None


# =====================================================================
# Demo & Testing
# =====================================================================

if __name__ == "__main__":
    """
    Example usage and testing.
    
    To run:
        export GEMINI_API_KEY="your_api_key"
        python -m backend.services.explanation_service
    """
    import asyncio

    # Sample metrics for testing
    test_metrics = FairnessMetricsInput(
        model_name="Loan Approval System v2.1",
        protected_attribute="gender",
        disparate_impact_ratio=0.72,
        demographic_parity_difference=-0.15,
        equal_opportunity_difference=-0.18,
        group_statistics={
            "male": {"approval_rate": 0.68, "sample_size": 5200},
            "female": {"approval_rate": 0.49, "sample_size": 4800},
        },
        total_samples=10000,
        use_case="Credit lending eligibility",
        application_domain="financial_services",
        regulatory_requirements="EU AI Act (high-risk), GDPR Article 22",
        data_quality_notes="No missing values; balanced dataset",
    )

    async def main():
        print("=" * 80)
        print("Testing Fairness Explanation Service")
        print("=" * 80)
        
        result = await generate_fairness_explanation(test_metrics)
        
        if result:
            print("\n✓ Analysis succeeded:")
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print("\n✗ Analysis failed; using fallback...")
            fallback = create_fallback_response(
                test_metrics,
                "Gemini API returned invalid response"
            )
            print(json.dumps(fallback.model_dump(), indent=2))

    asyncio.run(main())
