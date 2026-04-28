# EquiLens AI — Robust Fairness Analysis Prompt for Gemini

## Executive Summary

This document defines a **production-grade prompt template** for Gemini 2.0 Flash that analyzes machine learning fairness metrics with:

- ✅ **Structured JSON output** (enforced via system prompt with explicit schema)
- ✅ **EU AI Act mappings** (Risk classifications per Annex III)
- ✅ **Hallucination prevention** (Data-grounded claims only; explicit guardrails)
- ✅ **Mitigation recommendations** (Pre-, in-, and post-processing strategies)
- ✅ **Response validation** (Pydantic schema checking)
- ✅ **Graceful degradation** (Fallback responses if LLM unavailable)

---

## 1. Design Principles

### 1.1 Core Tenets

| Principle | Rationale | Implementation |
|-----------|-----------|-----------------|
| **Evidence-Based** | All claims must be grounded in provided metrics. No invented data. | System prompt explicitly forbids claims not in metrics; asks for citation of thresholds. |
| **Multi-Dimensional** | Fairness is not one metric; report all relevant dimensions. | Template requires DI, DPD, EOD; flags if incomplete. |
| **Statistical Honesty** | Distinguish correlation from causation; flag uncertainty. | Prompt requires "Statistical observation" vs. "causal claim" distinction. |
| **Data-Aware** | No analysis without acknowledging data limitations. | Template sections: "data_quality" warnings, "limitations" array. |
| **Actionable** | Recommendations must be implementable with known tradeoffs. | Each mitigation includes: description, expected impact, tradeoff, complexity, confidence. |

### 1.2 Why Structured JSON?

1. **Programmatic Integration**: Parse response directly into Python; no NLP on NLP.
2. **Validation**: Check schema before using; catch LLM errors early.
3. **Reproducibility**: Same input → same JSON structure (deterministic parsing).
4. **Scalability**: Feed into dashboards, compliance reports, alerting systems.

---

## 2. System Prompt

The **system prompt** defines:
- Role and expertise boundaries
- Hard constraints (non-hallucination rules)
- Exact JSON schema (with field descriptions)
- Edge case handling
- Tone and communication style

### 2.1 Key Sections

#### 2.1.1 ROLE
```
You are EquiLens AI, a specialized fairness auditor for machine learning systems.
```

Narrow role prevents drift into legal advice, feature engineering, etc.

#### 2.1.2 CORE CONSTRAINTS (Non-Negotiable)

```
1. Base all claims on provided metrics. Do NOT invent data.
2. Distinguish between statistical observations and causal claims.
3. Flag data limitations explicitly:
   - If n < 100 per group → "low statistical power"
   - If attribute missing → "cannot evaluate"
   - If prediction column undefined → state assumption
4. Never assume missing data.
5. Treat fairness as multi-dimensional; report all metrics.
6. Cite established standards (80% rule, ±0.1 DPD, etc.).
```

These constraints directly prevent:
- 🚫 Making up missing metrics
- 🚫 Claiming causation from correlation
- 🚫 Ignoring sample size limitations
- 🚫 Cherry-picking favorable metrics
- 🚫 Using arbitrary thresholds

#### 2.1.3 JSON Schema

The system prompt includes the **exact schema** Gemini must produce:

```json
{
  "analysis": {
    "model_name": "string",
    "protected_attribute": "string",
    "timestamp": "ISO 8601 datetime",
    "data_quality": {
      "sample_size_per_group": {"group_name": int},
      "missing_data_pct": float,
      "warnings": ["string"]
    },
    "metrics": {
      "disparate_impact_ratio": {
        "value": float,
        "threshold": 0.8,
        "is_fair": bool
      },
      // ... (DPD, EOD similarly)
    }
  },
  "fairness_assessment": {...},
  "eu_ai_act": {...},
  "remediation": {...},
  "limitations": [...]
}
```

**Why in the prompt?** Gemini uses this as a blueprint; fewer parsing errors, more consistent output.

#### 2.1.4 EU AI Act Mappings

System prompt includes reference table:

| Risk Level | Description | Examples |
|------------|-------------|----------|
| **UNACCEPTABLE** | Prohibited under EU AI Act | Social scoring; systematic discrimination |
| **HIGH** | Requires governance (Annex III) | Employment/credit decisions; DI < 0.75 |
| **LIMITED** | Requires transparency | Recommender systems; documented bias |
| **MINIMAL** | Standard governance | Statistical parity achieved |

Gemini uses this to classify responses.

---

## 3. User Prompt Template

The **user prompt** provides:
- Model/use case context
- Raw metrics (DI, DPD, EOD)
- Group statistics (approval rates, sample sizes)
- Data metadata (sample size, quality notes)
- Regulatory context (compliance requirements)

### 3.1 Input Schema

```python
@dataclass
class FairnessMetricsInput:
    # Required
    model_name: str                    # e.g., "Loan Approval v2.1"
    protected_attribute: str           # e.g., "gender"
    disparate_impact_ratio: float      # e.g., 0.72
    demographic_parity_difference: float  # e.g., -0.15
    equal_opportunity_difference: float   # e.g., -0.18
    group_statistics: dict             # {"male": {...}, "female": {...}}
    total_samples: int                 # e.g., 10000
    
    # Optional context
    use_case: str = ""                 # e.g., "Credit lending"
    application_domain: str = ""       # e.g., "financial_services"
    regulatory_requirements: str = ""  # e.g., "EU AI Act (high-risk)"
    data_quality_notes: str = ""       # e.g., "No missing values"
```

### 3.2 Example User Prompt

```
FAIRNESS AUDIT REQUEST

Model Details:
  Name: Loan Approval System v2.1
  Use case: Credit lending eligibility
  Protected attribute: gender
  Application domain: financial_services

METRICS (provided by fairness engine):
{
  "disparate_impact_ratio": 0.72,
  "demographic_parity_difference": -0.15,
  "equal_opportunity_difference": -0.18
}

GROUP STATISTICS:
{
  "male": {"approval_rate": 0.68, "sample_size": 5200},
  "female": {"approval_rate": 0.49, "sample_size": 4800}
}

DATA METADATA:
  - Total samples analyzed: 10000
  - Samples per protected group: {"male": 5200, "female": 4800}
  - Analysis timestamp: 2026-04-25T14:30:00Z
  - Data quality notes: No missing values in protected attribute. Balanced dataset.

CONTEXT:
  - Stakeholders: Compliance team, Product leadership
  - Regulatory requirements: EU AI Act (high-risk), GDPR Article 22 (automated decision-making)

TASK:
Analyze the provided metrics and output valid JSON (no markdown):
1. Assess fairness using established thresholds (80% rule for DI, ±0.1 for DPD/EOD).
2. Identify disproportionately affected groups and quantify impact.
3. Map findings to EU AI Act risk classification.
4. Suggest three remediation strategies (pre-, in-, post-processing).
5. Flag data limitations and uncertainty.

Output ONLY the JSON object; no explanatory text.
```

---

## 4. Expected Output (JSON Schema)

### 4.1 Full Schema with Descriptions

```json
{
  "analysis": {
    "model_name": "string (e.g., 'Loan Approval System v2.1')",
    "protected_attribute": "string (e.g., 'gender')",
    "timestamp": "ISO 8601 datetime when analysis was performed",
    "data_quality": {
      "sample_size_per_group": {
        "group_name": "integer (e.g., 'male': 5200)"
      },
      "missing_data_pct": "float (% of missing values)",
      "warnings": [
        "string (e.g., 'Low statistical power: n=45 < 100 per group')"
      ]
    },
    "metrics": {
      "disparate_impact_ratio": {
        "value": "float (provided metric)",
        "threshold": "float (standard threshold, e.g., 0.8 for 80% rule)",
        "is_fair": "bool (value >= threshold)",
        "interpretation": "string (e.g., 'Female selection rate is 72% of male rate')"
      },
      "demographic_parity_difference": {
        "value": "float (provided metric)",
        "threshold": "float (standard, e.g., 0.1)",
        "is_fair": "bool (abs(value) <= threshold)",
        "interpretation": "string"
      },
      "equal_opportunity_difference": {
        "value": "float",
        "threshold": "float",
        "is_fair": "bool",
        "interpretation": "string"
      },
      "notes": "string (summary of metric consistency, e.g., 'All three metrics indicate disparities')"
    }
  },
  "fairness_assessment": {
    "overall_fair": "bool",
    "severity": "enum: none|low|moderate|high|critical",
    "disproportionately_affected_group": "string (e.g., 'Female applicants')",
    "magnitude_of_disparity": "string (e.g., 'X group approved 19 percentage points less often')",
    "evidence_basis": "string (cite metrics and thresholds, e.g., 'DI=0.72 violates 80% rule')"
  },
  "eu_ai_act": {
    "risk_level": "enum: unacceptable|high|limited|minimal",
    "article_mapping": "string (e.g., 'Annex III, Article 6(2)(a): Employment decisions')",
    "compliance_status": "enum: compliant|non_compliant|uncertain",
    "compliance_reasoning": "string (explain mapping and why risk level applies)"
  },
  "remediation": {
    "pre_processing": [
      {
        "strategy": "string (e.g., 'balanced resampling')",
        "description": "string (how to implement)",
        "expected_impact": "string (e.g., 'Expected to improve DI to 0.88–0.95')",
        "tradeoff": "string (e.g., 'May reduce accuracy by 2–3%')",
        "implementation_complexity": "enum: low|medium|high",
        "confidence": "enum: high|medium|low"
      }
    ],
    "in_processing": [...],
    "post_processing": [...]
  },
  "limitations": [
    "string (e.g., 'Single time snapshot; trends not assessed')",
    "string (e.g., 'Only one protected attribute evaluated; intersectionality not considered')"
  ]
}
```

### 4.2 Example Output (Loan Approval Case)

```json
{
  "analysis": {
    "model_name": "Loan Approval System v2.1",
    "protected_attribute": "gender",
    "timestamp": "2026-04-25T14:30:00Z",
    "data_quality": {
      "sample_size_per_group": {
        "male": 5200,
        "female": 4800
      },
      "missing_data_pct": 0.0,
      "warnings": []
    },
    "metrics": {
      "disparate_impact_ratio": {
        "value": 0.72,
        "threshold": 0.8,
        "is_fair": false,
        "interpretation": "Female approval rate is 72% of male rate; violates 80% rule."
      },
      "demographic_parity_difference": {
        "value": -0.15,
        "threshold": 0.1,
        "is_fair": false,
        "interpretation": "15 percentage point gap in approval rates exceeds 10% threshold."
      },
      "equal_opportunity_difference": {
        "value": -0.18,
        "threshold": 0.1,
        "is_fair": false,
        "interpretation": "Among eligible applicants, females have 18% lower true positive rate."
      },
      "notes": "All three metrics indicate statistically significant gender disparities."
    }
  },
  "fairness_assessment": {
    "overall_fair": false,
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
        "description": "Remove or redact features strongly correlated with protected attribute (e.g., age, occupation); add fairness loss term during training.",
        "expected_impact": "Reduces proxy discrimination; DI could improve to 0.90–1.05.",
        "tradeoff": "Reduced model interpretability; 1–3% accuracy loss; requires retraining.",
        "implementation_complexity": "medium",
        "confidence": "medium"
      },
      {
        "strategy": "Threshold optimization with fairness constraints",
        "description": "Adjust approval threshold separately per group to equalize DI and DPD while maintaining business metrics.",
        "expected_impact": "Can achieve DI ≥ 0.95 and |DPD| ≤ 0.05.",
        "tradeoff": "Raises transparency concerns; may be perceived as explicit discrimination; regulatory scrutiny required.",
        "implementation_complexity": "medium",
        "confidence": "medium"
      }
    ],
    "in_processing": [
      {
        "strategy": "Adversarial debiasing",
        "description": "Add adversarial component to loss function that penalizes learning protected attribute correlations.",
        "expected_impact": "Can reduce DI by 0.10–0.20 if well-tuned.",
        "tradeoff": "Requires careful hyperparameter tuning; may destabilize training; computationally expensive.",
        "implementation_complexity": "high",
        "confidence": "medium"
      }
    ],
    "post_processing": [
      {
        "strategy": "Calibration and score adjustment",
        "description": "Post-hoc recalibration using separate validation set per group to equalize positive rates.",
        "expected_impact": "Can improve DI to 0.92–0.98; minimal accuracy loss.",
        "tradeoff": "Doesn't address root causes; surface-level fix; may not generalize to future data.",
        "implementation_complexity": "low",
        "confidence": "high"
      }
    ]
  },
  "limitations": [
    "Analysis is a point-in-time snapshot; temporal trends not assessed.",
    "Only one protected attribute (gender) evaluated; intersectional fairness (gender × age, gender × race) not analyzed.",
    "Fairness assumes protected attribute is correctly labeled; errors would bias results.",
    "Remediation recommendations are speculative; actual effectiveness depends on data distribution and model architecture.",
    "Does not account for causal mechanisms; statistical fairness ≠ actual justice for individuals."
  ]
}
```

---

## 5. Hallucination Prevention Guardrails

### 5.1 What We Guard Against

| Hallucination Type | Guardrail | Example |
|-------------------|-----------|---------|
| **Invented metrics** | "Base all claims on provided metrics. Do NOT invent data." | ❌ Gemini cannot invent "False Negative Rate Parity" if not provided. |
| **Causal claims** | "Distinguish statistical observations from causal claims." | ❌ Cannot claim "Gender causes discrimination"; only "disparate impact observed." |
| **Undefined groups** | "If protected attribute missing, state 'cannot evaluate'." | ❌ Cannot analyze race if race column not provided. |
| **Assumptions** | "Never assume missing data. State 'insufficient data'." | ❌ Cannot guess prediction column; must ask or flag. |
| **Cherry-picking** | "Treat fairness as multi-dimensional; report all metrics." | ❌ Cannot hide DI if DPD looks good. |
| **Arbitrary thresholds** | "For threshold decisions, cite established standards." | ✓ "80% rule threshold of 0.8 is per EEOC guidance." |

### 5.2 Example Guardrail in Action

**User Input (missing prediction_column):**
```json
{
  "disparate_impact_ratio": 0.72,
  "demographic_parity_difference": -0.15,
  // equal_opportunity_difference NOT PROVIDED
}
```

**Bad Gemini Response (Hallucination):**
```
"equal_opportunity_difference": {
  "value": -0.17,  // ← MADE UP!
  "is_fair": false
}
```

**Good Gemini Response (w/ Guardrails):**
```json
{
  "data_quality": {
    "warnings": [
      "equal_opportunity_difference not provided in metrics",
      "Unable to compute EOD; assumption made that target column = prediction proxy"
    ]
  },
  "metrics": {
    "equal_opportunity_difference": {
      "value": null,
      "is_fair": null,
      "interpretation": "Not computable from provided data"
    }
  }
}
```

---

## 6. Integration Example

### 6.1 Using the Prompt in Code

```python
from prompts.fairness_analyst import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FairnessMetricsInput,
    build_user_prompt,
)
from backend.services.explanation_service_v2 import generate_fairness_explanation

# Step 1: Collect metrics from fairness engine
metrics = FairnessMetricsInput(
    model_name="Loan Approval v2.1",
    protected_attribute="gender",
    disparate_impact_ratio=0.72,
    demographic_parity_difference=-0.15,
    equal_opportunity_difference=-0.18,
    group_statistics={
        "male": {"approval_rate": 0.68, "sample_size": 5200},
        "female": {"approval_rate": 0.49, "sample_size": 4800},
    },
    total_samples=10000,
    use_case="Credit lending",
    application_domain="financial_services",
    regulatory_requirements="EU AI Act (high-risk)",
)

# Step 2: Call Gemini with robust prompt
response = await generate_fairness_explanation(metrics)

# Step 3: Response is automatically validated + parsed into Pydantic
if response:
    print(f"Risk Level: {response.eu_ai_act['risk_level']}")
    print(f"Compliance: {response.eu_ai_act['compliance_status']}")
    for strategy in response.remediation["pre_processing"]:
        print(f"- {strategy['strategy']}: {strategy['expected_impact']}")
else:
    # Fallback if LLM unavailable
    fallback = create_fallback_response(metrics)
    print("Using fallback response (metrics only)")
```

### 6.2 Handling Edge Cases

**Case 1: LLM returns invalid JSON**
```python
try:
    response = await generate_fairness_explanation(metrics)
except json.JSONDecodeError:
    logger.error("Gemini returned invalid JSON")
    response = create_fallback_response(metrics, "Invalid JSON response")
```

**Case 2: Response schema validation fails**
```python
is_valid, errors = validate_output_schema(response_json)
if not is_valid:
    logger.warning(f"Schema validation failed: {errors}")
    # System continues with partial response
    # Client can check 'limitations' field for details
```

**Case 3: Low statistical power**
```python
if response.data_quality["warnings"]:
    for warning in response.data_quality["warnings"]:
        if "statistical power" in warning:
            logger.info("Analysis flagged low power; human review recommended")
```

---

## 7. Best Practices

### 7.1 Prompt Engineering

| Best Practice | Why | Implementation |
|---------------|-----|-----------------|
| **Low temperature** | Reduces hallucination; favors consistency | `temperature=0.2` in generation_config |
| **Schema in prompt** | Gemini uses as blueprint; fewer parsing errors | System prompt includes exact JSON structure |
| **Explicit constraints** | Prevents common failure modes | "Do NOT invent data"; "Distinguish correlation from causation" |
| **Citations** | Forces grounding; enables verification | "For threshold decisions, cite established standards" |
| **Staged parsing** | Catches errors early | JSON parse → schema validate → Pydantic parse |

### 7.2 Fairness Analysis

| Best Practice | Why | Implementation |
|---------------|-----|-----------------|
| **Multi-metric** | No single metric tells the whole story | Report DI, DPD, EOD; flag if any unavailable |
| **Data quality first** | Garbage in = garbage out | Always include "data_quality" warnings section |
| **Acknowledge limits** | Honesty builds trust | "Limitations" array is mandatory |
| **Evidence-based** | Reproducible, defensible | All claims cite metrics + thresholds |
| **Regulatory context** | Compliance matters | Map to EU AI Act per Annex III |

### 7.3 Response Validation

```python
# Always validate before use
is_valid, errors = validate_output_schema(response_json)
if not is_valid:
    logger.warning(f"Schema errors: {errors}")
    # Option 1: Proceed with caution (fields may be incomplete)
    # Option 2: Reject response; use fallback
    # Choice depends on downstream requirements
```

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

| Limitation | Mitigation |
|-----------|-----------|
| Single protected attribute only | Use loop for each attribute; then intersectional analysis |
| Point-in-time snapshot | Log historical analyses; compute trend metrics |
| No causal analysis | Requires domain knowledge; beyond LLM scope |
| Assumes binary protected attribute | Extend schema for multi-class (race, age buckets) |
| No feedback loops | Collect human ratings; fine-tune Gemini via RLHF |

### 8.2 Future Enhancements

1. **Multi-attribute Analysis**: Loop over protected_attributes; return intersectional matrices
2. **Temporal Trends**: Ingest historical metrics; compute drift, acceleration
3. **Causal Inference**: Integrate with causal discovery libraries (DoWhy, CausalML)
4. **Custom Thresholds**: Accept domain-specific fairness standards
5. **Feedback Integration**: Log analyst feedback; improve recommendations over time

---

## 9. Files & Artifacts

### 9.1 Code Files

- **[prompts/fairness_analyst.py](../prompts/fairness_analyst.py)**
  - System prompt definition
  - User prompt template
  - Input/output schemas
  - EU AI Act mappings
  - Helper functions (build_user_prompt, validate_output_schema)

- **[backend/services/explanation_service_v2.py](../backend/services/explanation_service_v2.py)**
  - Gemini integration with retry logic
  - Response validation via Pydantic
  - Fallback response generation
  - Example usage / demo

### 9.2 Test Data

**Input Example:**
```json
{
  "model_name": "Loan Approval System v2.1",
  "protected_attribute": "gender",
  "disparate_impact_ratio": 0.72,
  "demographic_parity_difference": -0.15,
  "equal_opportunity_difference": -0.18,
  "group_statistics": {
    "male": {"approval_rate": 0.68, "sample_size": 5200},
    "female": {"approval_rate": 0.49, "sample_size": 4800}
  },
  "total_samples": 10000
}
```

**Output Example:** See Section 4.2 (full loan approval example)

---

## 10. Checklist for Use

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Import `FairnessMetricsInput` from `prompts.fairness_analyst`
- [ ] Populate input with fairness metrics (DI, DPD, EOD)
- [ ] Call `generate_fairness_explanation(metrics)`
- [ ] Check response.eu_ai_act['risk_level'] for compliance risk
- [ ] Review response.remediation for actionable strategies
- [ ] Log response.limitations for caveats
- [ ] If LLM call fails, use `create_fallback_response()` (graceful degradation)
- [ ] Validate schema before downstream integration: `validate_output_schema(response_json)`

---

## 11. References

### EU AI Act
- [EU AI Act - Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- Annex III: High-risk AI systems
- Article 6(2): Employment/recruitment decisions
- Article 8: Governance requirements

### Fairness Metrics
- **80% Rule** (Disparate Impact): EEOC Uniform Guidelines on Employee Selection
- **Demographic Parity Difference (DPD)**: |P(Ŷ=1|A=0) − P(Ŷ=1|A=1)| ≤ 0.1
- **Equal Opportunity Difference (EOD)**: |P(Ŷ=1|A=0,Y=1) − P(Ŷ=1|A=1,Y=1)| ≤ 0.1

### Prompt Engineering
- "Prompting as a Programming Language" - Microsoft Research
- OpenAI: "Techniques to improve reliability" (System prompts, schema constraints)
- Constitutional AI (Anthropic): Guardrails design patterns

---

## 12. Support & Troubleshooting

### Q: Gemini returns incomplete JSON
**A:** Check `data_quality.warnings` for hints. If schema validation fails, fall back to deterministic response (metrics only).

### Q: Response says "cannot evaluate attribute X"
**A:** Likely missing data. Check if protected attribute column provided; if not, this is correct behavior (preventing hallucination).

### Q: Why EU AI Act mapping?
**A:** Regulatory compliance is critical for high-stakes ML. Mapping ensures transparency and accountability under emerging regulations.

### Q: Can I customize thresholds?
**A:** Currently hardcoded (80% rule, ±0.1). Extend `FairnessMetricsInput` with optional `thresholds` dict if needed.

---

**Last Updated:** April 25, 2026  
**Version:** 1.0 (Production-Ready)  
**Maintainer:** EquiLens AI Team
