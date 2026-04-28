# EquiLens AI — Robust Fairness Prompt: Deliverables Summary

## 📦 What Has Been Delivered

A **production-ready prompt engineering system** for Gemini 2.0 Flash that generates structured, evidence-based fairness analysis with EU AI Act compliance mapping.

---

## 📂 Files Created/Modified

### Core Prompt System
| File | Purpose | Status |
|------|---------|--------|
| `prompts/fairness_analyst.py` | System prompt, user template, schemas, guardrails | ✅ Created (650 lines) |
| `backend/services/explanation_service_v2.py` | Gemini integration, retry logic, validation, fallback | ✅ Created (400 lines) |
| `requirements.txt` | Added `tenacity>=8.2.0` for retry logic | ✅ Updated |

### Documentation
| File | Purpose | Status |
|------|---------|--------|
| `docs/PROMPT_ENGINEERING_GUIDE.md` | Full technical documentation (12 sections) | ✅ Created |
| `docs/PROMPT_QUICK_REFERENCE.md` | Quick reference & cheat sheet | ✅ Created |
| `tests/test_prompt_system.py` | Validation suite (11 tests, all passing) | ✅ Created |

### Testing
```
✅ 11/11 validation tests passing:
  ✓ System prompt structure
  ✓ User prompt template
  ✓ Input/output schema validation
  ✓ EU AI Act mappings
  ✓ Guardrails against hallucinations
  ✓ Remediation strategies
  ✓ Edge case handling
```

---

## 🎯 Key Features

### 1. **Structured JSON Output** (Enforced)
- Exact schema defined in system prompt
- Pydantic validation layer
- Schema validation helper function
- No markdown wrapping or explanatory text

### 2. **EU AI Act Compliance Mapping**
- Risk classification (UNACCEPTABLE → HIGH → LIMITED → MINIMAL)
- Annex III article mapping
- Compliance status assessment
- Regulatory requirements integration

### 3. **Hallucination Prevention** (6 Core Guardrails)
✋ System prompt explicitly forbids:
1. Inventing metrics not provided
2. Claiming causation from correlation
3. Assuming missing data
4. Cherry-picking favorable metrics
5. Using arbitrary thresholds
6. Incomplete multi-dimensional analysis

✓ System prompt requires:
- Evidence-based claims (cite metrics + thresholds)
- Statistical honesty (distinguish observation vs. causation)
- Data quality flagging (sample size, missing values)
- All metrics reported (no hiding bad results)

### 4. **Actionable Mitigations**
Three categories (pre-, in-, post-processing):
- Strategy name + description
- Expected impact (quantified)
- Implementation tradeoffs
- Complexity assessment (low/medium/high)
- Confidence level

### 5. **Graceful Degradation**
- Retry logic with exponential backoff
- Fallback response (deterministic, metrics-only)
- Response validation before parsing
- Detailed error logging

---

## 🚀 Quick Start (3 Steps)

### Step 1: Import
```python
from prompts.fairness_analyst import FairnessMetricsInput
from backend.services.explanation_service_v2 import generate_fairness_explanation
```

### Step 2: Prepare Metrics
```python
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
)
```

### Step 3: Call Gemini
```python
response = await generate_fairness_explanation(metrics)
print(f"Risk Level: {response.eu_ai_act['risk_level']}")
# Output: "high" → Requires governance
```

---

## 📊 Example Input/Output

### Input (Loan Approval Case)
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

### Output (Summary)
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
    "article_mapping": "Annex III, Article 6(2)(a): Employment decisions"
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
  },
  "limitations": [...]
}
```

**Action:** ⚠️ High-risk system; requires DPIA, human review, remediation.

---

## 🔒 Security & Best Practices

### Configuration
- `GEMINI_API_KEY` in environment variables (never hardcode)
- `temperature=0.2` (deterministic, reduces hallucinations)
- Exponential backoff retry (3 attempts max)

### Validation
```python
# Always validate before parsing
is_valid, errors = validate_output_schema(response_json)
if not is_valid:
    logger.warning(f"Schema errors: {errors}")
    response = create_fallback_response(metrics)
```

### Logging & Audit
- All inputs logged (model name, metrics)
- PII sanitization (never log raw DataFrames)
- Error traces captured for debugging

---

## 📚 Documentation

| Document | Target Audience | Key Sections |
|----------|-----------------|--------------|
| **PROMPT_ENGINEERING_GUIDE.md** | Technical leads, ML engineers | Design principles, system prompt, schema, examples, best practices |
| **PROMPT_QUICK_REFERENCE.md** | Practitioners, data scientists | Input checklist, code examples, thresholds, guardrails, FAQ |
| **test_prompt_system.py** | QA, DevOps | Validation suite, edge cases, integration tests |

---

## 🧪 Testing & Validation

Run all tests:
```bash
cd c:\Users\Asus\Desktop\solution_challenge
python -m pytest tests/test_prompt_system.py -v
```

Expected output:
```
============================= 11 passed in 0.20s ==============================
  ✓ test_system_prompt_exists
  ✓ test_user_prompt_template
  ✓ test_input_schema
  ✓ test_build_user_prompt
  ✓ test_example_input_output
  ✓ test_output_schema_validation
  ✓ test_eu_ai_act_mappings
  ✓ test_guardrails
  ✓ test_metric_interpretation
  ✓ test_remediation_structure
  ✓ test_edge_case_handling
```

---

## 🏗️ Architecture Integration

### Before (Original Code)
```
fairness_engine/ (metrics computation)
    ↓
backend/services/analysis_service.py (orchestration)
    ↓
[LLM call] → Unstructured explanation
    ↓
API response
```

### After (With Robust Prompt)
```
fairness_engine/ (metrics computation)
    ↓
backend/services/analysis_service.py (orchestration)
    ↓
[Robust Prompt] → Gemini 2.0 Flash
    ↓ (retry + validation + fallback)
FairnessAnalysisResponse (structured JSON)
    ↓ (schema validated)
API response + dashboard/compliance report
```

---

## 📋 EU AI Act Compliance Mapping

System automatically classifies fairness findings to EU AI Act risk levels:

| Metric | Threshold | Risk If Violated | EU AI Act |
|--------|-----------|------------------|-----------|
| DI (Disparate Impact) | ≥ 0.80 | HIGH | Annex III, Art. 6(2) |
| DPD (Demographic Parity Diff) | \|x\| ≤ 0.10 | HIGH | Annex III, Art. 6(2) |
| EOD (Equal Opportunity Diff) | \|x\| ≤ 0.10 | HIGH | Annex III, Art. 6(2) |

System outputs:
- `compliance_status`: compliant | non_compliant | uncertain
- `article_mapping`: Specific annex/article citation
- `risk_level`: unacceptable | high | limited | minimal

---

## 🔄 Remediation Recommendations

System suggests strategies in three categories:

### Pre-Processing (Data-Level)
- Balanced resampling ✅ Low complexity
- Feature engineering 💭 Medium complexity
- Threshold optimization 💭 Medium complexity

### In-Processing (Algorithm-Level)
- Adversarial debiasing 💭 High complexity
- Fairness-aware training 💭 High complexity

### Post-Processing (Output-Level)
- Score calibration ✅ Low complexity
- Group-specific cutoffs 💭 Medium complexity

Each includes: expected impact, tradeoff, complexity, confidence.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| JSON parse error | Fallback to `create_fallback_response()` |
| Schema validation fails | Check `data_quality.warnings`; review required fields |
| "insufficient statistical power" | n < 30 per group; collect more data |
| Metrics conflict (DI fair, DPD not) | Normal! Report all; severity is conservative |
| Gemini returns invalid response | Retry logic catches; falls back after 3 attempts |

---

## 📦 Dependencies Added

```
tenacity>=8.2.0,<9.0.0  # Retry logic with exponential backoff
```

All other dependencies (FastAPI, Pydantic, Google Generative AI, etc.) were already present.

---

## ✅ Checklist for Deployment

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Run `pytest tests/test_prompt_system.py` (all 11 tests passing)
- [ ] Review `docs/PROMPT_ENGINEERING_GUIDE.md` (section 8: limitations)
- [ ] Integrate `generate_fairness_explanation()` into `analysis_service.py`
- [ ] Update API response schema to include EU AI Act fields
- [ ] Test with sample datasets (loan approval, hiring, etc.)
- [ ] Document in API docs (/docs endpoint)
- [ ] Set up monitoring for LLM API usage and costs
- [ ] Configure compliance dashboard to display `risk_level` field

---

## 🎓 Learning Resources

- **Prompt Engineering Best Practices**: See `docs/PROMPT_ENGINEERING_GUIDE.md` (Section 7)
- **EU AI Act Overview**: See `docs/PROMPT_ENGINEERING_GUIDE.md` (Section 11 References)
- **Fairness Metrics**: See `docs/PROMPT_QUICK_REFERENCE.md` (Fairness Thresholds table)
- **Example Usage**: See `backend/services/explanation_service_v2.py` (main block, line ~350)

---

## 📞 Support

### For Questions About:
- **System Prompt Design** → See `prompts/fairness_analyst.py` (lines 50–150)
- **JSON Schema** → See `prompts/fairness_analyst.py` (lines 180–250)
- **Gemini Integration** → See `backend/services/explanation_service_v2.py` (lines 100–200)
- **EU AI Act Mappings** → See `docs/PROMPT_ENGINEERING_GUIDE.md` (Section 2.1.3)
- **Remediation Strategies** → See `docs/PROMPT_QUICK_REFERENCE.md` (Remediation section)

---

## 🎯 Success Metrics

Once integrated, you should see:
1. ✅ 100% of fairness analyses return valid JSON (no parsing errors)
2. ✅ All responses include EU AI Act risk classification
3. ✅ Remediation recommendations average 3+ strategies per analysis
4. ✅ Zero hallucinated metrics (all grounded in input)
5. ✅ Response time < 5 seconds (with caching)

---

## 📝 Version Info

- **Created:** April 25, 2026
- **Status:** Production-Ready
- **Tested:** 11/11 validation tests passing
- **Gemini Model:** 2.0 Flash (recommended)
- **Python:** 3.10+ (tested on 3.13.3)

---

**Next Steps:**
1. Review `docs/PROMPT_QUICK_REFERENCE.md` (5 min read)
2. Run `pytest tests/test_prompt_system.py` (verify setup)
3. Try the example in `backend/services/explanation_service_v2.py` (main block)
4. Integrate into `analysis_service.py` response pipeline
5. Deploy with `GEMINI_API_KEY` set

**Ready to ship! 🚀**
