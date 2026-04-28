# EquiLens AI — Prompt Engineering Quick Reference

## 🎯 TL;DR

**What:** Production-grade prompt for Gemini bias analysis  
**Why:** Structured JSON + EU AI Act mapping + guardrails against hallucinations  
**How:** Feed metrics → Gemini → Validate JSON → Parse into Pydantic → Use downstream  

---

## 📋 Input Checklist

```python
metrics = FairnessMetricsInput(
    # REQUIRED
    model_name="Loan Approval v2",           # Name of ML model
    protected_attribute="gender",             # Protected attr (e.g., race, age)
    disparate_impact_ratio=0.72,             # DI (should be ≥ 0.8)
    demographic_parity_difference=-0.15,    # DPD (should be |x| ≤ 0.1)
    equal_opportunity_difference=-0.18,     # EOD (should be |x| ≤ 0.1)
    group_statistics={                       # Approval/selection rates per group
        "male": {"approval_rate": 0.68, "sample_size": 5200},
        "female": {"approval_rate": 0.49, "sample_size": 4800},
    },
    total_samples=10000,
    
    # OPTIONAL (for richer context)
    use_case="Credit lending",
    application_domain="financial_services",
    regulatory_requirements="EU AI Act (high-risk)",
    data_quality_notes="No missing values; balanced dataset",
)
```

---

## 🔧 Code Integration (3 Lines)

```python
from backend.services.explanation_service_v2 import generate_fairness_explanation

response = await generate_fairness_explanation(metrics)
print(f"Risk: {response.eu_ai_act['risk_level']}")  # Output: "high"
```

---

## 📊 Output Structure

| Section | Key Fields | Example |
|---------|-----------|---------|
| **analysis** | `metrics`, `data_quality`, `warnings` | DI=0.72, threshold=0.8, is_fair=False |
| **fairness_assessment** | `severity`, `magnitude_of_disparity` | "high", "Females 19pp less likely" |
| **eu_ai_act** | `risk_level`, `compliance_status` | "high", "non_compliant" |
| **remediation** | `pre_processing`, `in_processing`, `post_processing` | [{strategy, expected_impact, tradeoff, complexity}] |
| **limitations** | Caveats and data quality notes | ["Single snapshot", "n < 100 per group"] |

---

## 🚨 EU AI Act Risk Levels

| Level | Definition | Action |
|-------|-----------|--------|
| **UNACCEPTABLE** | Prohibited (e.g., social scoring) | ❌ Deploy immediately |
| **HIGH** | Requires governance + DPIA | ⚠️ Impact assessment, human review |
| **LIMITED** | Requires transparency | ℹ️ Document, inform users |
| **MINIMAL** | Standard governance | ✓ Can proceed |

---

## ✅ Fairness Thresholds (Established Standards)

| Metric | Threshold | Standard |
|--------|-----------|----------|
| Disparate Impact (DI) | ≥ 0.80 | 80% Rule (EEOC) |
| Demographic Parity Diff (DPD) | \|x\| ≤ 0.10 | Industry best practice |
| Equal Opportunity Diff (EOD) | \|x\| ≤ 0.10 | Industry best practice |

---

## 🛡️ Guardrails (What's Prevented)

✋ **LLM CANNOT:**
- Invent metrics not provided
- Claim causation ("gender causes discrimination" ❌)
- Assume missing data (must state "insufficient")
- Cherry-pick favorable metrics
- Use arbitrary thresholds (must cite source)

✓ **LLM MUST:**
- Ground all claims in provided data
- Distinguish "statistical observation" from "causal claim"
- Flag data quality issues (n < 100, missing columns)
- Report all metrics (don't hide bad ones)
- Cite established standards (80% rule, etc.)

---

## 🔄 Remediation Strategies (Quick Guide)

### Pre-Processing (Data-Level)
- ✅ **Balanced resampling** — Low complexity, high confidence
- ✅ **Feature engineering** — Medium complexity, medium confidence
- 💭 **Threshold optimization** — Medium complexity, raises transparency concerns

### In-Processing (Algorithm-Level)
- 💭 **Adversarial debiasing** — High complexity, requires tuning
- 💭 **Fairness-aware training** — Medium complexity

### Post-Processing (Output-Level)
- ✅ **Score calibration** — Low complexity, high confidence (but surface-level fix)
- 💭 **Group-specific cutoffs** — Medium complexity, raises fairness concerns

**Note:** Each recommendation includes `expected_impact`, `tradeoff`, and `confidence` field.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| JSON parse error | Check markdown wrapping (```json```); fallback to `create_fallback_response()` |
| Schema validation fails | Check `data_quality.warnings`; review required fields in output |
| "insufficient statistical power" warning | n < 30 per group; collect more data or use caution |
| EU AI Act mapping says "uncertain" | Missing context fields; provide `use_case`, `application_domain`, `regulatory_requirements` |
| Metrics conflict (DI says fair, DPD says not) | Normal! Report all; severity is conservative (worst-case). |

---

## 🔐 Security Checklist

- [ ] Set `GEMINI_API_KEY` env var (never hardcode)
- [ ] Use `temperature=0.2` (deterministic, low hallucination)
- [ ] Validate schema before parsing: `validate_output_schema(json)`
- [ ] Log inputs (metrics + model name) for audit trail
- [ ] Do NOT log raw DataFrames or sensitive columns
- [ ] Use `create_fallback_response()` if LLM unavailable (graceful degradation)

---

## 📁 Files

```
prompts/
  └─ fairness_analyst.py          # Prompt templates, schemas, guardrails
backend/services/
  └─ explanation_service_v2.py    # Gemini integration, retry logic, validation
docs/
  └─ PROMPT_ENGINEERING_GUIDE.md  # Full documentation (this file's sibling)
```

---

## 🎓 Example Walkthrough

**Input:**
```python
metrics = FairnessMetricsInput(
    model_name="Loan Model",
    protected_attribute="gender",
    disparate_impact_ratio=0.72,  # ← Below 0.8 threshold
    demographic_parity_difference=-0.15,  # ← Beyond ±0.1 threshold
    equal_opportunity_difference=-0.18,  # ← Beyond ±0.1 threshold
    group_statistics={
        "M": {"approval_rate": 0.68, "sample_size": 5200},
        "F": {"approval_rate": 0.49, "sample_size": 4800},
    },
    total_samples=10000,
    use_case="Credit approval",
    application_domain="financial_services",
)
response = await generate_fairness_explanation(metrics)
```

**Output (Key Findings):**
```json
{
  "fairness_assessment": {
    "overall_fair": false,
    "severity": "high",
    "magnitude_of_disparity": "Females approved 19pp less often"
  },
  "eu_ai_act": {
    "risk_level": "high",
    "compliance_status": "non_compliant",
    "compliance_reasoning": "Credit access is fundamental right; documented disparate impact triggers Annex III classification."
  },
  "remediation": {
    "pre_processing": [
      {
        "strategy": "Balanced resampling",
        "expected_impact": "Improve DI to 0.88–0.95",
        "implementation_complexity": "low",
        "confidence": "high"
      }
    ]
  }
}
```

**Action:** ⚠️ High-risk system; requires DPIA, human review, remediation before deployment.

---

## 🔗 References

- **System Prompt:** [prompts/fairness_analyst.py](../prompts/fairness_analyst.py) (lines 1–100)
- **User Template:** [prompts/fairness_analyst.py](../prompts/fairness_analyst.py) (lines ~250–300)
- **Integration:** [backend/services/explanation_service_v2.py](../backend/services/explanation_service_v2.py)
- **Full Guide:** [docs/PROMPT_ENGINEERING_GUIDE.md](./PROMPT_ENGINEERING_GUIDE.md)

---

## ❓ FAQ

**Q: Why JSON and not markdown?**  
A: JSON is parseable by code; no NLP needed. Enables automation, dashboards, compliance reporting.

**Q: Can I customize fairness thresholds?**  
A: Currently hardcoded (80% rule, ±0.1). To extend: add `thresholds` dict to `FairnessMetricsInput`.

**Q: What if Gemini returns invalid JSON?**  
A: `generate_fairness_explanation()` catches parse errors and returns `None`. Fallback to `create_fallback_response()` for deterministic output (metrics only).

**Q: How to avoid hallucinations?**  
A: System prompt includes explicit guardrails. Low temp (0.2) + schema constraints + response validation. See "Guardrails" section above.

**Q: Why EU AI Act?**  
A: It's now law in EU/EEA. High-stakes ML requires compliance audits. This prompt automates the mapping.

---

**Last Updated:** April 25, 2026  
**Version:** 1.0 (Quick Reference)
