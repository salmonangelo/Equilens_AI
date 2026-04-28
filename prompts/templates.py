"""
EquiLens AI — Prompt Templates

Structured prompt templates for LLM interactions.
Templates use Python f-string or Jinja2 formatting with
clearly defined input variables.
"""

from __future__ import annotations

# === Bias Explanation Prompts ===

BIAS_EXPLANATION_TEMPLATE = """
You are EquiLens AI, an expert in machine learning fairness and bias analysis.

Given the following fairness metrics for model "{model_name}":

{metrics_summary}

Protected attribute analyzed: {protected_attribute}

Provide a clear, actionable explanation of:
1. What these metrics indicate about the model's fairness
2. Which groups are disproportionately affected
3. The potential real-world impact of these biases
4. Severity assessment (none / low / moderate / high / critical)

Target audience: {audience}
Detail level: {detail_level}
""".strip()


# === Remediation Suggestion Prompts ===

REMEDIATION_TEMPLATE = """
You are EquiLens AI, an expert in ML fairness remediation strategies.

A fairness audit of model "{model_name}" revealed the following issues:

{bias_findings}

Suggest concrete remediation strategies, categorized as:
1. **Pre-processing**: Data-level interventions
2. **In-processing**: Algorithm-level modifications
3. **Post-processing**: Output-level adjustments

For each strategy, include:
- Description of the approach
- Expected impact on fairness metrics
- Potential tradeoffs with model performance
- Implementation complexity (low / medium / high)
""".strip()


# === Report Narrative Prompts ===

REPORT_NARRATIVE_TEMPLATE = """
You are EquiLens AI. Generate a professional fairness audit report narrative.

Model: {model_name}
Date: {analysis_date}
Overall assessment: {overall_assessment}

Metrics:
{metrics_details}

Write a structured report narrative suitable for {audience} stakeholders.
Include an executive summary, detailed findings, and next steps.
""".strip()


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given variables.

    Args:
        template: The prompt template string.
        **kwargs: Variables to substitute into the template.

    Returns:
        Formatted prompt string.
    """
    return template.format(**kwargs)
