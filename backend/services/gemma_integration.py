"""
EquiLens AI — Gemma Local LLM Integration (Optional)

Integrates Gemma via Ollama API for LOCAL column classification:
- Column type detection (PII / protected / target / feature)
- Schema understanding
- Data quality assessment

IMPORTANT: Gemma is used ONLY for schema analysis, NOT for fairness metrics.
Fairness metrics are always computed locally (deterministic).

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull Gemma: ollama pull gemma:7b
    3. Start server: ollama serve (default: http://localhost:11434)

This module gracefully degrades if Ollama is unavailable.
"""

import logging
import json
from typing import Any
from urllib.parse import urljoin

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_ENDPOINT = urljoin(OLLAMA_BASE_URL, "/api/generate")
OLLAMA_MODEL = "gemma:7b"
OLLAMA_TIMEOUT = 30  # seconds


# =====================================================================
# Gemma Integration
# =====================================================================

def is_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(
            urljoin(OLLAMA_BASE_URL, "/api/tags"),
            timeout=5,
        )
        return response.status_code == 200
    except (ConnectionError, Timeout, RequestException):
        return False


def classify_column(column_name: str, sample_values: list[Any]) -> dict[str, Any]:
    """
    Use Gemma to classify a column's purpose.

    LOCAL ONLY: Used for schema understanding, not sensitive data analysis.

    Args:
        column_name: Name of the column
        sample_values: Sample values from the column

    Returns:
        Dictionary with:
        - type: "pii" | "protected" | "target" | "feature" | "unknown"
        - confidence: float in [0, 1]
        - reasoning: explanation string
        - method: "gemma" | "fallback"

    Example:
        >>> classify_column("email", ["user1@example.com", "user2@example.com"])
        {
            "type": "pii",
            "confidence": 0.95,
            "reasoning": "Column contains email addresses (PII)",
            "method": "gemma"
        }
    """
    if not is_ollama_available():
        logger.warning("Ollama not available; using fallback schema detection")
        return _fallback_classification(column_name, sample_values)

    try:
        return _gemma_classify(column_name, sample_values)
    except Exception as e:
        logger.warning(f"Gemma classification failed: {e}. Using fallback.")
        return _fallback_classification(column_name, sample_values)


def _gemma_classify(column_name: str, sample_values: list[Any]) -> dict[str, Any]:
    """Call Gemma to classify a column."""
    sample_str = ", ".join(str(v)[:50] for v in sample_values[:5])

    prompt = f"""Classify this column based on name and sample values.

Column name: {column_name}
Sample values: {sample_str}

Respond with JSON:
{{
  "type": ("pii" | "protected" | "target" | "feature" | "unknown"),
  "confidence": (0.0 to 1.0),
  "reasoning": "brief explanation"
}}

Only respond with the JSON object, no extra text."""

    try:
        response = requests.post(
            OLLAMA_API_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # Deterministic
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "").strip()

        # Parse JSON from response
        classification = json.loads(response_text)
        classification["method"] = "gemma"

        logger.info(f"Gemma classified '{column_name}' as {classification['type']}")
        return classification

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemma response as JSON: {e}")
        raise
    except RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        raise


def _fallback_classification(
    column_name: str,
    sample_values: list[Any],
) -> dict[str, Any]:
    """
    Fallback heuristic-based column classification.

    Used when Gemma is unavailable.
    """
    name_lower = column_name.lower()

    # PII patterns
    pii_patterns = [
        "email", "phone", "ssn", "social_security",
        "credit_card", "card_number", "name", "surname",
        "address", "zip", "postal", "street",
    ]

    # Protected attribute patterns
    protected_patterns = [
        "gender", "sex", "race", "ethnicity", "age",
        "religion", "disability", "national_origin",
    ]

    # Target/label patterns
    target_patterns = [
        "target", "label", "outcome", "result",
        "approved", "denied", "accepted", "rejected",
    ]

    col_type = "feature"
    confidence = 0.5

    if any(p in name_lower for p in pii_patterns):
        col_type = "pii"
        confidence = 0.8
    elif any(p in name_lower for p in protected_patterns):
        col_type = "protected"
        confidence = 0.9
    elif any(p in name_lower for p in target_patterns):
        col_type = "target"
        confidence = 0.85

    return {
        "type": col_type,
        "confidence": confidence,
        "reasoning": f"Heuristic match: {col_type}",
        "method": "fallback",
    }


def analyze_schema(df_columns: list[str], sample_rows: list[dict]) -> dict[str, dict]:
    """
    Analyze dataset schema and classify all columns.

    Args:
        df_columns: List of column names
        sample_rows: Sample rows from dataset

    Returns:
        Dictionary mapping column name → classification result
    """
    logger.info(f"Analyzing schema for {len(df_columns)} columns...")

    classifications = {}

    for col in df_columns:
        sample_values = [row.get(col) for row in sample_rows if col in row]
        classifications[col] = classify_column(col, sample_values)

    # Log summary
    pii_cols = [c for c, v in classifications.items() if v.get("type") == "pii"]
    protected_cols = [c for c, v in classifications.items() if v.get("type") == "protected"]
    target_cols = [c for c, v in classifications.items() if v.get("type") == "target"]

    logger.info(f"  PII columns: {pii_cols}")
    logger.info(f"  Protected attributes: {protected_cols}")
    logger.info(f"  Target columns: {target_cols}")

    return classifications
