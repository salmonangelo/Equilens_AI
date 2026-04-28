"""explanation_service
~~~~~~~~~~~~~~~~~~~~~~
Utility module that contacts the Gemini LLM to produce a structured fairness
explanation from aggregated metrics, group statistics and a risk score.

The module:
* defines a Pydantic schema for the expected JSON output,
* builds a clean, dedented prompt,
* configures the Gemini API key from ``GEMINI_API_KEY`` environment variable,
* provides a retry‑wrapped ``generate_fairness_explanation`` function with
  exponential back‑off and detailed logging.
"""

import json
import textwrap
import logging
import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Configure Gemini API key from environment variable. The variable name can be overridden as needed.
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set. Please set it to your Gemini API key.")

genai.configure(api_key=GENAI_API_KEY)


# Configure a module-level logger. If the application already sets up logging this is a no-op.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structured Output Schema
class FairnessExplanation(BaseModel):
    explanation: str = Field(
        description="Plain English explanation of the fairness metrics and group statistics."
    )
    risk_interpretation: str = Field(
        description="Interpretation of the overall risk score and what it implies for the system."
    )
    suggested_fixes: List[str] = Field(
        description="Actionable, suggested fixes or next steps to mitigate any fairness issues found."
    )

# Clean Prompt Formatting
# System instruction that forces guardrails and the EU‑AI‑Act mapping.
SYSTEM_INSTRUCTION = textwrap.dedent(
    """
    You are a specialist in responsible AI and EU AI regulation. Your job is to analyse the supplied fairness metrics, explain any bias in plain English, map the findings to the EU AI Act risk categories (high‑risk, limited‑risk, minimal‑risk, no‑risk), and provide concrete mitigation suggestions.

    Guardrails:
    - Do NOT fabricate data that was not provided.
    - If any required metric or group statistic is missing, output "missing data" for that field and skip the related analysis.
    - The final answer MUST be a single JSON object that conforms to the schema described in the user prompt.
    - Keep explanations concise (≤ 150 words per section) and avoid speculative statements.
    """
).strip()

# User‑visible prompt template – the aggregated data placeholders are filled by the caller.
USER_PROMPT_TEMPLATE = textwrap.dedent(
    """
    {fairness_metrics}\n\n    {group_statistics}\n\n    Risk Score: {risk_score}\n    """
).strip()

# Helper to build the final prompt – concatenates system instruction and user data.
def _build_prompt(fairness_metrics: dict, group_statistics: dict, risk_score: float) -> str:
    """Return the full prompt sent to Gemini.

    The function JSON‑encodes the supplied dictionaries and inserts them into the
    user‑visible template, then prefixes the system instruction that contains the
    guardrails and EU‑AI‑Act mapping requirements.
    """
    user_part = USER_PROMPT_TEMPLATE.format(
        fairness_metrics=json.dumps(fairness_metrics, indent=2),
        group_statistics=json.dumps(group_statistics, indent=2),
        risk_score=risk_score,
    )
    # Combine system + user parts – Gemini treats the whole string as a single prompt.
    return f"{SYSTEM_INSTRUCTION}\n\n{user_part}"


# Helper to build the final prompt – keeps the main function tidy.
def _build_prompt(fairness_metrics: dict, group_statistics: dict, risk_score: float) -> str:
    """Return the full prompt sent to Gemini.

    The function JSON‑encodes the supplied dictionaries and inserts them into the
    user‑visible template, then prefixes the system instruction that contains the
    guardrails and EU‑AI‑Act mapping requirements.
    """
    user_part = USER_PROMPT_TEMPLATE.format(
        fairness_metrics=json.dumps(fairness_metrics, indent=2),
        group_statistics=json.dumps(group_statistics, indent=2),
        risk_score=risk_score,
    )
    return f"{SYSTEM_INSTRUCTION}\n\n{user_part}"



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # Retry on any Gemini API‑related exception as well as generic errors.
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_fairness_explanation(
    fairness_metrics: Dict[str, Any],
    group_statistics: Dict[str, Any],
    risk_score: float,
    model_name: str = "gemini-1.5-pro",
    timeout_seconds: int = 30,
) -> FairnessExplanation:
    """Generate a structured fairness explanation using Gemini.

    Parameters
    ----------
    fairness_metrics: Dict[str, Any]
        Aggregated fairness metrics (e.g., disparate impact, demographic parity).
    group_statistics: Dict[str, Any]
        Summary statistics per protected group (counts, prevalences, etc.).
    risk_score: float
        Overall risk score, typically in the range [0, 1].
    model_name: str, optional
        Gemini model identifier; defaults to ``gemini-1.5-pro``.
    timeout_seconds: int, optional
        Maximum seconds to wait for the Gemini response before raising a timeout.

    Returns
    -------
    FairnessExplanation
        A validated Pydantic model containing ``explanation``, ``risk_interpretation`` and ``suggested_fixes``.

    Raises
    ------
    RuntimeError
        If the Gemini response cannot be parsed or the API returns an error.
    """
    """Generate a structured fairness explanation using Gemini.

    Parameters
    ----------
    fairness_metrics: Dict[str, Any]
        Aggregated fairness metrics (e.g., disparate impact, demographic parity).
    group_statistics: Dict[str, Any]
        Summary statistics per protected group (counts, prevalences, etc.).
    risk_score: float
        Overall risk score, typically in the range ``[0, 1]``.
    model_name: str, optional
        Gemini model identifier; defaults to ``gemini-1.5-pro``.
    timeout_seconds: int, optional
        Maximum seconds to wait for the Gemini response before raising a timeout.

    Returns
    -------
    FairnessExplanation
        A validated Pydantic model containing ``explanation``, ``risk_interpretation`` and ``suggested_fixes``.

    Raises
    ------
    RuntimeError
        If the Gemini response cannot be parsed or the API returns an error.
    """

    
    # 1. Initialise the Gemini model
    model = genai.GenerativeModel(model_name)

    # 2. Build the full prompt (system + user parts)
    prompt = _build_prompt(fairness_metrics, group_statistics, risk_score)
    
    try:
        # 3. Call the API with Structured Output constraints and a timeout
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=FairnessExplanation,
                temperature=0.2,  # Low temperature for deterministic output
                max_output_tokens=1024,
                # Note: Gemini SDK may not expose direct timeout; we rely on Tenacity's retry.
            ),
            request_options={"timeout": timeout_seconds},
        )

        # 4. Ensure the response is JSON and validate against the schema
        if not response.text:
            raise RuntimeError("Empty response from Gemini API.")
        try:
            return FairnessExplanation.model_validate_json(response.text)
        except Exception as json_err:
            logger.error(f"Failed to parse Gemini response as JSON: {json_err}")
            raise RuntimeError("Invalid JSON response from Gemini API.")

    except Exception as e:
        logger.error(f"Unexpected error during Gemini call: {e}")
        raise
