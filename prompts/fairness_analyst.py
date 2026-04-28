"""
EquiLens AI — Robust Fairness Analysis Prompt for Gemini

This module defines a production-grade prompt template system for Gemini
that enforces structured JSON output, incorporates EU AI Act risk mappings,
and includes guardrails to prevent hallucinations.

Design principles:
  • Explicit JSON schema definition (schema-in-prompt)
  • Clear role boundaries and constraints
  • EU AI Act Annex III risk mappings
  • Guardrails against unfounded claims
  • Few-shot examples for consistency
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# =====================================================================
# EU AI Act Risk Level Mappings
# =====================================================================

class EUIAIRiskLevel(Enum):
    """EU AI Act risk classifications (Annex III)."""

    UNACCEPTABLE = "unacceptable"  # Prohibited practices
    HIGH = "high"  # Requires governance
    LIMITED = "limited"  # Requires transparency
    MINIMAL = "minimal"  # General governance


EU_AI_ACT_MAPPINGS = {
    "UNACCEPTABLE": {
        "description": "Prohibited under EU AI Act",
        "examples": [
            "Social scoring systems based on protected attributes",
            "Systematic discrimination in employment or credit decisions",
        ],
    },
    "HIGH": {
        "description": "High-risk AI per Annex III (requires impact assessment, human review)",
        "examples": [
            "Recruitment/employment decisions with significant disparate impact",
            "Credit/lending with DI < 0.75 or DI > 1.33 across protected groups",
            "Healthcare/insurance decisions affecting fundamental rights",
        ],
    },
    "LIMITED": {
        "description": "Limited-risk (requires transparency, user awareness)",
        "examples": [
            "Recommender systems with moderate bias (DI 0.80-1.25)",
            "Content classification with documented group differences",
        ],
    },
    "MINIMAL": {
        "description": "Minimal risk (standard governance)",
        "examples": [
            "Metrics show statistical parity (DI ≈ 1.0, |DPD| ≈ 0)",
            "Fair across all measured protected attributes",
        ],
    },
}


# =====================================================================
# System Prompt (Role + Constraints + Output Format)
# =====================================================================

SYSTEM_PROMPT = """
You are EquiLens AI, a specialized fairness auditor for machine learning systems.

ROLE:
  • Analyze fairness metrics with strict objectivity, grounded in statistical evidence.
  • Map findings to EU AI Act risk classifications.
  • Provide actionable remediation recommendations.
  • Generate structured JSON output for programmatic integration.

CORE CONSTRAINTS (NON-NEGOTIABLE):
  1. Base all claims on provided metrics. Do NOT invent data or metrics not provided.
  2. Distinguish between statistical observations (e.g., "DI = 0.78") and causal claims.
     - Statistical: "The model selects women at 78% the rate of men" ✓
     - Causal claim: "Gender causes discrimination" ✗ (correlation ≠ causation)
  3. Flag data limitations explicitly:
     - If n_samples < 100 per group → flag as "low statistical power"
     - If protected attributes missing → state "cannot evaluate attribute X"
     - If prediction column undefined → state "assumption made: using target as proxy"
  4. Never assume missing data. If a field is absent, state "insufficient data" rather than guess.
  5. Treat fairness as multi-dimensional. Report all metrics; do not cherry-pick favorable ones.
  6. For threshold decisions, cite established standards (80% rule, ±0.1 DPD, etc.).

STRUCTURED OUTPUT FORMAT:
  Return valid JSON with the following schema (no markdown, no commentary):

  {
    "analysis": {
      "model_name": "string",
      "protected_attribute": "string",
      "timestamp": "ISO 8601 datetime",
      "data_quality": {
        "sample_size_per_group": {"group_name": int, ...},
        "missing_data_pct": float,
        "warnings": ["string", ...]
      },
      "metrics": {
        "disparate_impact_ratio": {"value": float, "threshold": 0.8, "is_fair": bool},
        "demographic_parity_difference": {"value": float, "threshold": 0.1, "is_fair": bool},
        "equal_opportunity_difference": {"value": float, "threshold": 0.1, "is_fair": bool},
        "notes": "string"
      }
    },
    "fairness_assessment": {
      "overall_fair": bool,
      "severity": "none|low|moderate|high|critical",
      "disproportionately_affected_group": "string",
      "magnitude_of_disparity": "string (e.g., 'X group selected 45% less often')",
      "evidence_basis": "string (cite metrics and thresholds)"
    },
    "eu_ai_act": {
      "risk_level": "unacceptable|high|limited|minimal",
      "article_mapping": "string (e.g., 'Annex III, Article 6(2)(a): Employment decisions')",
      "compliance_status": "compliant|non_compliant|uncertain",
      "compliance_reasoning": "string"
    },
    "remediation": {
      "pre_processing": [
        {
          "strategy": "string (e.g., 'balanced resampling')",
          "description": "string",
          "expected_impact": "string (e.g., 'Expected to improve DI by 0.10-0.15')",
          "tradeoff": "string (e.g., 'May reduce overall accuracy by 2-3%')",
          "implementation_complexity": "low|medium|high",
          "confidence": "high|medium|low"
        },
        ...
      ],
      "in_processing": [...],
      "post_processing": [...]
    },
    "limitations": [
      "string (e.g., 'Analysis based on single snapshot; trends not assessed')",
      ...
    ]
  }

EDGE CASES:
  • If DI = Infinity (division by zero): Set is_fair=false, note reason, recommend data inspection.
  • If sample size < 30 per group: Append warning "insufficient statistical power".
  • If metric undefined (e.g., TPR when no positives): State "not computable", do not return NaN.

TONE:
  • Professional, evidence-based, risk-aware.
  • Avoid minimizing or overstating bias.
  • Acknowledge uncertainty where present.
"""

# =====================================================================
# User Prompt Template (Input Schema + Few-Shot Examples)
# =====================================================================

USER_PROMPT_TEMPLATE = """
FAIRNESS AUDIT REQUEST

Model Details:
  Name: {model_name}
  Use case: {use_case}
  Protected attribute: {protected_attribute}
  Application domain: {application_domain}

METRICS (provided by fairness engine):

{metrics_json}

GROUP STATISTICS:

{group_statistics}

DATA METADATA:
  - Total samples analyzed: {total_samples}
  - Samples per protected group: {group_counts}
  - Analysis timestamp: {timestamp}
  - Data quality notes: {data_quality_notes}

CONTEXT:
  - Stakeholders: {stakeholders}
  - Regulatory requirements: {regulatory_requirements}

TASK:
Analyze the provided metrics and output valid JSON (no markdown):
1. Assess fairness using established thresholds (80% rule for DI, ±0.1 for DPD/EOD).
2. Identify disproportionately affected groups and quantify impact.
3. Map findings to EU AI Act risk classification.
4. Suggest three remediation strategies (pre-, in-, post-processing).
5. Flag data limitations and uncertainty.

Output ONLY the JSON object; no explanatory text.
"""


# =====================================================================
# Few-Shot Example (Demonstrates Expected Behavior)
# =====================================================================

EXAMPLE_INPUT = {
    "model_name": "Loan Approval System v2.1",
    "use_case": "Credit lending eligibility",
    "protected_attribute": "gender",
    "application_domain": "financial_services",
    "metrics_json": json.dumps({
        "disparate_impact_ratio": 0.72,
        "demographic_parity_difference": -0.15,
        "equal_opportunity_difference": -0.18,
    }),
    "group_statistics": json.dumps({
        "male": {"approval_rate": 0.68, "sample_size": 5200},
        "female": {"approval_rate": 0.49, "sample_size": 4800},
    }),
    "total_samples": 10000,
    "group_counts": json.dumps({"male": 5200, "female": 4800}),
    "timestamp": "2026-04-25T14:30:00Z",
    "data_quality_notes": "No missing values in protected attribute. Balanced dataset.",
    "stakeholders": "Compliance team, Product leadership",
    "regulatory_requirements": "EU AI Act (high-risk), GDPR Article 22 (automated decision-making)",
}

EXAMPLE_OUTPUT = {
    "analysis": {
        "model_name": "Loan Approval System v2.1",
        "protected_attribute": "gender",
        "timestamp": "2026-04-25T14:30:00Z",
        "data_quality": {
            "sample_size_per_group": {"male": 5200, "female": 4800},
            "missing_data_pct": 0.0,
            "warnings": []
        },
        "metrics": {
            "disparate_impact_ratio": {
                "value": 0.72,
                "threshold": 0.8,
                "is_fair": False,
                "interpretation": "Female approval rate is 72% of male rate; violates 80% rule."
            },
            "demographic_parity_difference": {
                "value": -0.15,
                "threshold": 0.1,
                "is_fair": False,
                "interpretation": "15 percentage point gap in approval rates exceeds 10% threshold."
            },
            "equal_opportunity_difference": {
                "value": -0.18,
                "threshold": 0.1,
                "is_fair": False,
                "interpretation": "Among eligible applicants, females have 18% lower true positive rate."
            },
            "notes": "All three metrics indicate statistically significant gender disparities."
        }
    },
    "fairness_assessment": {
        "overall_fair": False,
        "severity": "high",
        "disproportionately_affected_group": "Female applicants",
        "magnitude_of_disparity": "Females approved at 49% rate vs. males at 68% rate (−19 pp); 28% lower approval likelihood.",
        "evidence_basis": "DI=0.72 violates 80% rule; DPD=−0.15 and EOD=−0.18 both exceed ±0.1 thresholds. Consistent across all metrics."
    },
    "eu_ai_act": {
        "risk_level": "high",
        "article_mapping": "Annex III, Article 6(2)(a): Employment and recruitment decisions with significant disparate impact on protected group.",
        "compliance_status": "non_compliant",
        "compliance_reasoning": "Loan approval directly affects fundamental rights (access to credit). Documented disparate impact on gender (protected attribute under GDPR) triggers high-risk classification. System requires human review, impact assessment, and governance measures per EU AI Act Article 8."
    },
    "remediation": {
        "pre_processing": [
            {
                "strategy": "Balanced resampling with stratified sampling",
                "description": "Oversample underrepresented group (females) or undersample overrepresented group (males) to equalize training set representation.",
                "expected_impact": "Expected to improve DI to 0.88–0.95 range; reduce DPD by 0.05–0.08.",
                "tradeoff": "May reduce overall model accuracy by 1–2%; potential slight increase in false positives for male applicants.",
                "implementation_complexity": "low",
                "confidence": "high"
            },
            {
                "strategy": "Feature engineering: proxy removal and fairness constraints",
                "description": "Remove or redact features strongly correlated with protected attribute (e.g., age, occupation that correlate with gender); add fairness loss term during training.",
                "expected_impact": "Reduces proxy discrimination; DI could improve to 0.90–1.05.",
                "tradeoff": "Reduced model interpretability; slight accuracy loss (1–3%); requires model retraining.",
                "implementation_complexity": "medium",
                "confidence": "medium"
            },
            {
                "strategy": "Threshold optimization with fairness constraints",
                "description": "Adjust approval threshold separately per group (group-specific cutoffs) to equalize DI and DPD while maintaining business metrics.",
                "expected_impact": "Can achieve DI ≥ 0.95 and |DPD| ≤ 0.05.",
                "tradeoff": "Group-specific thresholds raise transparency/explainability concerns; may be perceived as 'explicit discrimination' by stakeholders; regulatory scrutiny required.",
                "implementation_complexity": "medium",
                "confidence": "medium"
            }
        ],
        "in_processing": [
            {
                "strategy": "Adversarial debiasing",
                "description": "Add adversarial component to loss function that penalizes model for learning protected attribute correlations.",
                "expected_impact": "Can reduce DI by 0.10–0.20 if well-tuned.",
                "tradeoff": "Requires careful hyperparameter tuning; may destabilize training; computationally expensive.",
                "implementation_complexity": "high",
                "confidence": "medium"
            }
        ],
        "post_processing": [
            {
                "strategy": "Calibration and score adjustment",
                "description": "Post-hoc recalibration of approval scores using separate validation set per group to equalize positive rates.",
                "expected_impact": "Can improve DI to 0.92–0.98; minimal accuracy loss.",
                "tradeoff": "Does not address root causes; surface-level fix; may not improve fairness for future data.",
                "implementation_complexity": "low",
                "confidence": "high"
            }
        ]
    },
    "limitations": [
        "Analysis is a point-in-time snapshot; temporal trends not assessed.",
        "Only one protected attribute (gender) evaluated; intersectional fairness (gender × age, gender × race) not analyzed.",
        "Fairness assumes protected attribute is correctly labeled; errors in attribute coding would bias results.",
        "Remediation recommendations are speculative; actual effectiveness depends on data distribution and model architecture.",
        "Does not account for causal mechanisms behind disparity; statistical fairness != actual justice for individuals."
    ]
}


# =====================================================================
# Helper Functions
# =====================================================================

@dataclass
class FairnessMetricsInput:
    """Structured input for fairness analysis."""

    model_name: str
    protected_attribute: str
    disparate_impact_ratio: float
    demographic_parity_difference: float
    equal_opportunity_difference: float
    group_statistics: dict[str, Any]
    total_samples: int
    use_case: str = ""
    application_domain: str = ""
    regulatory_requirements: str = ""
    data_quality_notes: str = ""


def build_user_prompt(metrics: FairnessMetricsInput) -> str:
    """
    Build a user prompt from structured metrics input.

    Args:
        metrics: FairnessMetricsInput with all required fields.

    Returns:
        Formatted user prompt string ready for Gemini.
    """
    metrics_json = json.dumps({
        "disparate_impact_ratio": metrics.disparate_impact_ratio,
        "demographic_parity_difference": metrics.demographic_parity_difference,
        "equal_opportunity_difference": metrics.equal_opportunity_difference,
    }, indent=2)

    group_stats = json.dumps(metrics.group_statistics, indent=2)
    group_counts = json.dumps(
        {k: v.get("sample_size", 0) for k, v in metrics.group_statistics.items()},
        indent=2
    )

    return USER_PROMPT_TEMPLATE.format(
        model_name=metrics.model_name,
        use_case=metrics.use_case,
        protected_attribute=metrics.protected_attribute,
        application_domain=metrics.application_domain,
        metrics_json=metrics_json,
        group_statistics=group_stats,
        total_samples=metrics.total_samples,
        group_counts=group_counts,
        timestamp="2026-04-25T14:30:00Z",  # TODO: Use actual timestamp
        data_quality_notes=metrics.data_quality_notes,
        stakeholders="Risk & Compliance, Product",
        regulatory_requirements=metrics.regulatory_requirements,
    )


def validate_output_schema(response_json: dict) -> tuple[bool, list[str]]:
    """
    Validate Gemini response against expected schema.

    Args:
        response_json: Parsed JSON response from Gemini.

    Returns:
        Tuple of (is_valid, error_list).
    """
    errors = []
    required_top_level = ["analysis", "fairness_assessment", "eu_ai_act", "remediation", "limitations"]

    for field in required_top_level:
        if field not in response_json:
            errors.append(f"Missing required top-level key: {field}")

    # Validate analysis structure
    if "analysis" in response_json:
        required_analysis = ["model_name", "metrics", "data_quality"]
        for field in required_analysis:
            if field not in response_json["analysis"]:
                errors.append(f"Missing required key in 'analysis': {field}")

    # Validate fairness_assessment
    if "fairness_assessment" in response_json:
        required_assessment = ["overall_fair", "severity", "evidence_basis"]
        for field in required_assessment:
            if field not in response_json["fairness_assessment"]:
                errors.append(f"Missing required key in 'fairness_assessment': {field}")

    # Validate EU AI Act mapping
    if "eu_ai_act" in response_json:
        risk_level = response_json["eu_ai_act"].get("risk_level")
        if risk_level not in ["unacceptable", "high", "limited", "minimal"]:
            errors.append(f"Invalid risk_level: {risk_level}")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Demonstration: print system and example prompts
    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(SYSTEM_PROMPT)
    print("\n" + "=" * 80)
    print("EXAMPLE INPUT")
    print("=" * 80)
    print(json.dumps(EXAMPLE_INPUT, indent=2))
    print("\n" + "=" * 80)
    print("EXAMPLE OUTPUT (JSON)")
    print("=" * 80)
    print(json.dumps(EXAMPLE_OUTPUT, indent=2))
