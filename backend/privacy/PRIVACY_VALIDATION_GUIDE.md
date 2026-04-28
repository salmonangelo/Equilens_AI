# Privacy Validation Layer - Security & Compliance

A **production-grade privacy validation middleware** that enforces strict constraints before any external API call, ensuring no raw dataset records or personally identifiable information (PII) leaves your system.

## Overview

The privacy validation layer prevents privacy violations by:

✅ **Blocking raw dataset records** - Detects and rejects payloads containing raw data  
✅ **Enforcing minimum group size** - Prevents re-identification via small group analysis (< 10 records)  
✅ **Validating aggregation** - Ensures only aggregated metrics and statistics are sent to external APIs  
✅ **Detecting suspicious patterns** - Identifies row-level data indicators (ID fields, records arrays)  
✅ **Clear error messages** - Provides actionable feedback on validation failures  

## Architecture

```
Dataset Upload
    ↓
Analysis Pipeline (local, in-memory)
    ├─ Anonymization (PII redaction)
    ├─ Metrics computation (aggregated)
    └─ Risk scoring
    ↓
[PRIVACY VALIDATION CHECKPOINT] ← Validates before external API
    │
    ├─ Detects banned fields (raw_data, dataframe, records)
    ├─ Checks group sizes (min 10 per group)
    ├─ Validates metric-only payload
    ├─ Identifies row-level indicators
    │
    └─ ✓ PASS → Gemini API call with aggregated metrics only
       ✗ FAIL → Reject, return error, NO external call made
```

## Components

### 1. Privacy Validator (`backend/privacy/validator.py`)

Core validation engine with granular checks:

```python
from backend.privacy.validator import PrivacyValidator, validate_gemini_payload

# Option A: Use the validator directly
validator = PrivacyValidator(min_group_size=10)
result = validator.validate_payload_for_gemini(payload_dict)

if not result.is_valid:
    for error in result.errors:
        print(f"❌ {error.violation_type}: {error.message}")

# Option B: Use the helper function (raises exception)
try:
    validate_gemini_payload(payload_dict)
except PrivacyValidationError as e:
    print(f"Privacy violation: {e}")
```

### 2. Integration with Explanation Service

Privacy validation is **automatically applied** before Gemini API calls:

```python
# In backend/services/explanation_service_v2.py
async def generate_fairness_explanation(metrics: FairnessMetricsInput):
    # Step 0: Privacy validation (automatic)
    validate_gemini_payload(metrics_dict)  # ← Blocks external call if invalid
    
    # Step 1+: Safe to call Gemini API
    response = client.models.generate_content(...)
```

## Validation Rules

### Banned Fields (Immediate Rejection)

These fields **always** trigger rejection:

| Field | Reason |
|-------|--------|
| `raw_data` | Obvious raw dataset indicator |
| `df` / `dataframe` | Pandas DataFrame (row-level) |
| `records` / `rows` | Record collections |
| `data` | Generic data field (ambiguous) |

**Example:**
```python
# ❌ REJECTED
payload = {
    "disparate_impact_ratio": 0.8,
    "raw_data": [{"id": 1, "name": "John"}]  # Banned!
}

# ✓ ACCEPTED
payload = {
    "disparate_impact_ratio": 0.8,
    "group_statistics": {"male": {"sample_size": 100}}
}
```

### Group Size Constraints

**Minimum group size: 10 records per protected group**

Prevents re-identification risk through demographic inference.

```python
# ❌ REJECTED (female group too small)
group_statistics = {
    "male": {"sample_size": 500},
    "female": {"sample_size": 3}  # < 10, rejected!
}

# ✓ ACCEPTED
group_statistics = {
    "male": {"sample_size": 500},
    "female": {"sample_size": 100}  # >= 10, accepted
}
```

### Allowed Fields (Whitelist)

Only these aggregated fields may be sent to external APIs:

**Fairness Metrics:**
- `disparate_impact_ratio`
- `demographic_parity_difference`
- `equal_opportunity_difference`
- `true_positive_rate` / `false_positive_rate`
- `selection_rate`, `precision`, `recall`

**Risk Scores:**
- `fairness_risk_score`
- `risk_level` (LOW/MEDIUM/HIGH)
- `risk_weight`, `weights`

**Metadata (non-sensitive):**
- `model_name`, `protected_attribute`
- `use_case`, `application_domain`
- `total_samples`, `timestamp`

**Aggregated Statistics:**
- `group_statistics` (counts + rates only)
- `sample_size_per_group`
- `data_quality_notes`

**Compliance:**
- `regulatory_requirements`
- `stakeholders`

## Error Messages & Remediation

### Error Type: RAW_DATA_DETECTED

```
❌ raw_data_detected: Banned field 'raw_data' detected in payload. 
   Raw dataset records cannot be sent to external APIs.
```

**Fix:** Remove the banned field. Send only aggregated metrics.

### Error Type: SMALL_GROUP_SIZE

```
❌ small_group_size: Small group size detected. Groups smaller than 10: 
   'female' (n=5), 'asian' (n=8). Risk of re-identification. 
   Cannot send to external API.
```

**Fix:** 
- Combine small groups: `"female_combined": {"sample_size": 50}`
- Apply de-identification: Increase aggregation granularity
- Use differential privacy: Add noise to protect individuals

### Error Type: ROW_LEVEL_DATA

```
❌ row_level_data: Row-level indicator 'record_id' found with 1000 entries. 
   Cannot send raw records to external API.
```

**Fix:** Remove individual record IDs. Use aggregate counts only.

### Error Type: INSUFFICIENT_AGGREGATION

```
⚠️  Unexpected field 'results' in payload. 
    Verify this contains only aggregated data, not raw records.
```

**Fix:** Review the field. Ensure it contains aggregate statistics, not row data.

## Usage Examples

### Valid Payload ✅

```python
valid_payload = {
    "model_name": "Credit Risk Model v2.1",
    "protected_attribute": "race",
    
    # Aggregated metrics only
    "disparate_impact_ratio": 0.72,
    "demographic_parity_difference": -0.15,
    "equal_opportunity_difference": -0.18,
    
    # Aggregated group statistics
    "group_statistics": {
        "white": {
            "sample_size": 5000,           # ← Count (safe)
            "approval_rate": 0.65,          # ← Aggregate statistic
        },
        "black": {
            "sample_size": 3000,            # >= 10, safe
            "approval_rate": 0.47,
        },
    },
    
    "total_samples": 8000,
    
    # Metadata (non-sensitive)
    "use_case": "Credit card approval",
    "regulatory_requirements": "Fair Lending (ECOA), EU AI Act",
}

# Validate
validate_gemini_payload(valid_payload)  # ✅ Passes
```

### Invalid Payload ❌

```python
invalid_payload = {
    "model_name": "Model v1",
    "disparate_impact_ratio": 0.75,
    
    # ❌ Raw records (BANNED)
    "records": [
        {"user_id": 1, "income": 50000, "credit_score": 720},
        {"user_id": 2, "income": 75000, "credit_score": 780},
    ],
    
    "group_statistics": {
        "approved": {"sample_size": 100},
        "denied": {"sample_size": 5}  # ❌ Too small
    },
}

# Validation fails
validate_gemini_payload(invalid_payload)
# Raises PrivacyValidationError with details
```

## Testing

### Run Tests

```bash
pytest tests/test_privacy_validation.py -v
```

### Test Coverage

The test suite includes:

✅ **Valid payloads** - Minimal, complete, with multiple attributes  
✅ **Raw data detection** - Banned fields, multiple violations  
✅ **Small group size** - Single/multiple groups, boundary cases  
✅ **Row-level indicators** - ID fields, high cardinality  
✅ **Malformed payloads** - Wrong types, missing fields  
✅ **Warnings** - Unexpected fields, suspicious patterns  
✅ **Edge cases** - Empty payloads, NaN values, zero samples  
✅ **Integration scenarios** - Real-world fairness analysis payloads  

### Example Test

```python
def test_small_group_rejected():
    """Verify that small groups are rejected."""
    payload = {
        "model_name": "model",
        "group_statistics": {
            "male": {"sample_size": 100},
            "female": {"sample_size": 3}  # Too small!
        },
        "total_samples": 103,
    }
    
    with pytest.raises(PrivacyValidationError) as exc:
        validate_gemini_payload(payload)
    
    assert exc.value.violation_type == PrivacyViolationType.SMALL_GROUP_SIZE
    assert "female" in exc.value.message
```

## Configuration

### Custom Minimum Group Size

```python
# Default: 10 records per group
validator = PrivacyValidator(min_group_size=10)

# Custom: Stricter privacy (20 records per group)
validator = PrivacyValidator(min_group_size=20)

# Custom: Looser privacy (5 records per group)
validator = PrivacyValidator(min_group_size=5)
```

### Allowed Fields Customization

Edit `backend/privacy/validator.py`:

```python
ALLOWED_METRIC_FIELDS = {
    "disparate_impact_ratio",
    "demographic_parity_difference",
    # ... add your custom aggregated fields
}
```

## Compliance & Standards

This implementation addresses:

### ✅ GDPR Compliance
- **Article 22** (Automated Decision-Making): Validates that explanations use only aggregated metrics
- **Article 9** (Special Categories): Prevents transmission of raw protected attributes
- **Anonymization**: Ensures data cannot be linked to individuals

### ✅ Fair Lending (ECOA/FHA)
- **Disparate Impact Analysis**: Safe aggregation prevents individual re-identification
- **Adverse Action Notices**: Aggregated explanations protect applicants

### ✅ EU AI Act
- **Annex III (High-Risk)**: Validation ensures transparency without exposing raw data
- **Article 13 (Transparency)**: Provides explanations based on aggregated metrics only

### ✅ CCPA/CPRA
- **Data Minimization**: No unnecessary raw data sent to third parties
- **Purpose Limitation**: Data used only for fairness analysis

## Deployment

### Local Testing

```bash
# Start backend with privacy validation enabled
python main.py

# Validation runs automatically before Gemini API calls
```

### Production Deployment

Privacy validation is **always enabled**. No configuration needed.

```python
# In explanation_service_v2.py
async def generate_fairness_explanation(metrics):
    # Automatic validation - blocks API call if invalid
    validate_gemini_payload(metrics_dict)
    # If we reach here, payload is safe
    ...
```

### Monitoring

Check backend logs:

```
✓ Privacy validation passed
❌ Privacy validation failed: raw_data_detected
⚠️  Privacy validation warning: Unexpected field
```

## Troubleshooting

### "Cannot connect to backend" after deploying privacy validation

**Cause:** Privacy validation is stricter and rejects some payloads  
**Fix:** Review error messages in logs, ensure payload is properly aggregated

### "Small group size detected"

**Cause:** Protected group has < 10 records  
**Fix:** Options:
1. Increase sample size
2. Combine groups (e.g., combine underrepresented races)
3. Apply differential privacy
4. Reduce sensitivity of the analysis

### "Unexpected field in payload"

**Cause:** Field name not in whitelist  
**Fix:** Either:
1. Remove the field (if not needed)
2. Rename to match whitelist (if it's aggregated metrics)
3. Request whelist update (if it's a new metric type)

## Future Enhancements

- [ ] Differential privacy integration
- [ ] Homomorphic encryption for Gemini payloads
- [ ] Audit logging for all validation decisions
- [ ] Machine learning-based anomaly detection
- [ ] Integration with secrets management (HashiCorp Vault)
- [ ] Custom validation rules per deployment

## Support

For questions or issues:

1. **Check logs**: Backend logs show validation details
2. **Review error message**: Specific violation type and remediation
3. **Run tests**: `pytest tests/test_privacy_validation.py`
4. **Read docs**: See above for detailed guidance

---

**Privacy by Design:** This validation layer ensures that fairness analysis benefits users without compromising their privacy. 🔒
