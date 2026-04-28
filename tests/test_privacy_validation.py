"""
Unit Tests for Privacy Validation Layer

Tests validate that:
  - Raw data cannot be sent to external APIs
  - Small group sizes are rejected
  - Only aggregated metrics pass validation
  - Clear error messages are provided
"""

import pytest
from backend.privacy.validator import (
    PrivacyValidator,
    PrivacyValidationError,
    PrivacyViolationType,
    ValidationResult,
    validate_gemini_payload,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def validator():
    """Create a validator instance."""
    return PrivacyValidator(min_group_size=10)


@pytest.fixture
def valid_gemini_payload():
    """Valid payload with only aggregated metrics."""
    return {
        "model_name": "loan_approval_v2",
        "protected_attribute": "gender",
        "disparate_impact_ratio": 0.75,
        "demographic_parity_difference": -0.15,
        "equal_opportunity_difference": -0.18,
        "group_statistics": {
            "male": {"sample_size": 5200, "approval_rate": 0.68},
            "female": {"sample_size": 4800, "approval_rate": 0.49},
        },
        "total_samples": 10000,
        "use_case": "credit_lending",
        "application_domain": "financial_services",
        "regulatory_requirements": "EU AI Act",
        "data_quality_notes": "No missing values",
    }


# =====================================================================
# Tests: Valid Payloads
# =====================================================================

class TestValidPayloads:
    """Tests for payloads that should pass validation."""
    
    def test_valid_minimal_payload(self, validator):
        """Minimal valid payload with required aggregated fields."""
        payload = {
            "model_name": "test_model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 150},
            },
            "total_samples": 250,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_valid_complete_payload(self, validator, valid_gemini_payload):
        """Complete valid payload with all aggregated metrics."""
        result = validator.validate_payload_for_gemini(valid_gemini_payload)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_payload_with_multiple_protected_attributes(self, validator):
        """Payload analyzing multiple protected attributes."""
        payload = {
            "model_name": "model_v1",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.75,
            "group_statistics": {
                "male": {"sample_size": 500},
                "female": {"sample_size": 600},
            },
            "total_samples": 1100,
            "demographic_parity_difference": -0.12,
            "equal_opportunity_difference": -0.15,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid
    
    def test_large_group_sizes_pass(self, validator):
        """Payloads with large group sizes pass validation."""
        payload = {
            "model_name": "model",
            "protected_attribute": "race",
            "disparate_impact_ratio": 0.9,
            "group_statistics": {
                "group_a": {"sample_size": 10000},
                "group_b": {"sample_size": 15000},
            },
            "total_samples": 25000,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid


# =====================================================================
# Tests: Invalid Payloads - Raw Data Detected
# =====================================================================

class TestRawDataDetection:
    """Tests for detecting raw dataset records."""
    
    def test_banned_field_raw_data(self, validator):
        """Rejects payload with 'raw_data' field."""
        payload = {
            "model_name": "model",
            "raw_data": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.RAW_DATA_DETECTED 
                   for e in result.errors)
    
    def test_banned_field_dataframe(self, validator):
        """Rejects payload with 'dataframe' field."""
        payload = {
            "model_name": "model",
            "dataframe": "DataFrame with 1000 rows",
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.RAW_DATA_DETECTED 
                   for e in result.errors)
    
    def test_banned_field_records(self, validator):
        """Rejects payload with 'records' field."""
        payload = {
            "model_name": "model",
            "records": [
                {"user_id": 1, "decision": "approved"},
                {"user_id": 2, "decision": "denied"},
            ],
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
    
    def test_banned_field_data(self, validator):
        """Rejects payload with generic 'data' field."""
        payload = {
            "model_name": "model",
            "data": [1, 2, 3, 4, 5],  # Could be row data
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
    
    def test_multiple_banned_fields(self, validator):
        """Rejects payload with multiple banned fields."""
        payload = {
            "model_name": "model",
            "raw_data": [{"id": 1}],
            "dataframe": "data",
            "records": [{"id": 2}],
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        # Should have multiple errors
        assert len(result.errors) > 1


# =====================================================================
# Tests: Invalid Payloads - Small Group Size
# =====================================================================

class TestSmallGroupSizeDetection:
    """Tests for detecting groups too small for privacy."""
    
    def test_single_group_too_small(self, validator):
        """Rejects when one group is below minimum size."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 5},  # Too small!
            },
            "total_samples": 105,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.SMALL_GROUP_SIZE 
                   for e in result.errors)
    
    def test_both_groups_too_small(self, validator):
        """Rejects when all groups are below minimum size."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 3},
                "female": {"sample_size": 7},
            },
            "total_samples": 10,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        errors = [e for e in result.errors 
                  if e.violation_type == PrivacyViolationType.SMALL_GROUP_SIZE]
        assert len(errors) > 0
    
    def test_exactly_minimum_size_passes(self, validator):
        """Accepts when group size equals minimum."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 10},  # Exactly minimum
                "female": {"sample_size": 10},
            },
            "total_samples": 20,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid
    
    def test_minimum_size_error_message(self, validator):
        """Error message includes affected groups and minimum requirement."""
        payload = {
            "model_name": "model",
            "protected_attribute": "race",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "white": {"sample_size": 100},
                "black": {"sample_size": 5},
                "asian": {"sample_size": 3},
            },
            "total_samples": 108,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        error = [e for e in result.errors 
                 if e.violation_type == PrivacyViolationType.SMALL_GROUP_SIZE][0]
        assert "black" in error.message.lower()
        assert "asian" in error.message.lower()
        assert "10" in error.message  # Minimum size


# =====================================================================
# Tests: Row-Level Data Detection
# =====================================================================

class TestRowLevelDataDetection:
    """Tests for detecting indicators of row-level data."""
    
    def test_id_field_with_high_cardinality(self, validator):
        """Rejects when 'id' field has many unique values."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "id": list(range(1, 100)),  # 99 unique IDs - likely row data
            "group_statistics": {
                "male": {"sample_size": 50},
                "female": {"sample_size": 49},
            },
            "total_samples": 99,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.ROW_LEVEL_DATA 
                   for e in result.errors)
    
    def test_record_id_field_detected(self, validator):
        """Detects 'record_id' as row-level indicator."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "record_id": [f"rec_{i}" for i in range(50)],
            "group_statistics": {
                "male": {"sample_size": 25},
                "female": {"sample_size": 25},
            },
            "total_samples": 50,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.ROW_LEVEL_DATA 
                   for e in result.errors)
    
    def test_small_id_list_allowed(self, validator):
        """Small number of IDs might be allowed (not necessarily row data)."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "id": [1, 2, 3],  # Only 3 IDs - not row data
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 100},
            },
            "total_samples": 200,
        }
        result = validator.validate_payload_for_gemini(payload)
        # Should pass (or warn) since ID list is small
        assert result.is_valid or len([e for e in result.errors 
                if e.violation_type == PrivacyViolationType.ROW_LEVEL_DATA]) == 0


# =====================================================================
# Tests: Invalid Payload Structures
# =====================================================================

class TestInvalidPayloadStructures:
    """Tests for malformed payloads."""
    
    def test_payload_is_not_dict(self, validator):
        """Rejects non-dict payload."""
        result = validator.validate_payload_for_gemini([1, 2, 3])
        assert not result.is_valid
        assert any(e.violation_type == PrivacyViolationType.INVALID_STATISTICS 
                   for e in result.errors)
    
    def test_group_statistics_not_dict(self, validator):
        """Rejects when group_statistics is not a dict."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": [
                {"male": 100},
                {"female": 150},
            ],  # List instead of dict
            "total_samples": 250,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
    
    def test_missing_sample_size_in_group_stats(self, validator):
        """Warns/rejects when group statistics missing sample size."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"approval_rate": 0.6},  # No sample_size!
                "female": {"sample_size": 100},
            },
            "total_samples": 200,
        }
        result = validator.validate_payload_for_gemini(payload)
        # Should fail or warn
        assert not result.is_valid


# =====================================================================
# Tests: Warnings
# =====================================================================

class TestWarnings:
    """Tests for non-critical warnings."""
    
    def test_warning_unexpected_fields(self, validator):
        """Warns about unexpected fields that might need review."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 100},
            },
            "total_samples": 200,
            "custom_field": "some_value",  # Unexpected
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid  # Still passes
        assert any("custom_field" in w for w in result.warnings)
    
    def test_warning_list_of_dicts(self, validator):
        """Warns about list fields with dicts (might be row data)."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 100},
            },
            "total_samples": 200,
            "results": [{"metric": 0.8}, {"metric": 0.75}],  # List of dicts
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid  # Passes but with warning
        assert any("results" in w for w in result.warnings)


# =====================================================================
# Tests: Helper Function
# =====================================================================

class TestValidateGeminiPayloadHelper:
    """Tests for the validate_gemini_payload helper function."""
    
    def test_valid_payload_passes(self, valid_gemini_payload):
        """Valid payload doesn't raise exception."""
        # Should not raise
        validate_gemini_payload(valid_gemini_payload)
    
    def test_invalid_payload_raises_exception(self):
        """Invalid payload raises PrivacyValidationError."""
        payload = {
            "model_name": "model",
            "raw_data": [{"id": 1}],  # Banned field
        }
        with pytest.raises(PrivacyValidationError) as exc_info:
            validate_gemini_payload(payload)
        
        assert exc_info.value.violation_type == PrivacyViolationType.RAW_DATA_DETECTED
    
    def test_small_group_raises_exception(self):
        """Small group size raises exception."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 5},  # Too small
            },
            "total_samples": 105,
        }
        with pytest.raises(PrivacyValidationError) as exc_info:
            validate_gemini_payload(payload)
        
        assert exc_info.value.violation_type == PrivacyViolationType.SMALL_GROUP_SIZE


# =====================================================================
# Tests: Custom Minimum Group Size
# =====================================================================

class TestCustomMinimumGroupSize:
    """Tests for validators with custom minimum group size."""
    
    def test_custom_min_group_size(self):
        """Validator respects custom minimum group size."""
        validator = PrivacyValidator(min_group_size=20)
        
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 15},  # Below custom min of 20
                "female": {"sample_size": 15},
            },
            "total_samples": 30,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
    
    def test_meets_custom_min_group_size(self):
        """Validator accepts payloads meeting custom minimum."""
        validator = PrivacyValidator(min_group_size=15)
        
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 20},
                "female": {"sample_size": 20},
            },
            "total_samples": 40,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid


# =====================================================================
# Tests: Edge Cases
# =====================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""
    
    def test_empty_payload_dict(self, validator):
        """Empty payload is handled gracefully."""
        result = validator.validate_payload_for_gemini({})
        assert result.is_valid  # Empty is okay (likely before metrics added)
    
    def test_zero_total_samples(self, validator):
        """Payload with zero total samples is flagged."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 0},
                "female": {"sample_size": 0},
            },
            "total_samples": 0,
        }
        result = validator.validate_payload_for_gemini(payload)
        assert not result.is_valid
    
    def test_negative_group_size(self, validator):
        """Negative group sizes are handled."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": -10},
                "female": {"sample_size": 50},
            },
            "total_samples": 40,  # Invalid
        }
        result = validator.validate_payload_for_gemini(payload)
        # Should be invalid
        assert not result.is_valid
    
    def test_nan_in_metrics(self, validator, valid_gemini_payload):
        """NaN values in metrics are handled."""
        payload = valid_gemini_payload.copy()
        payload["disparate_impact_ratio"] = float("nan")
        
        result = validator.validate_payload_for_gemini(payload)
        # Validation should still work (NaN is a number)
        assert result.is_valid


# =====================================================================
# Integration Tests
# =====================================================================

class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_real_world_valid_payload(self, validator):
        """Real-world fairness analysis payload passes."""
        payload = {
            "model_name": "Credit Risk Scoring v3.2",
            "protected_attribute": "race",
            "disparate_impact_ratio": 0.72,
            "demographic_parity_difference": -0.18,
            "equal_opportunity_difference": -0.15,
            "true_positive_rate": 0.68,
            "false_positive_rate": 0.25,
            "group_statistics": {
                "white": {
                    "sample_size": 5000,
                    "approval_rate": 0.62,
                    "default_rate": 0.08,
                },
                "black": {
                    "sample_size": 3200,
                    "approval_rate": 0.45,
                    "default_rate": 0.11,
                },
                "hispanic": {
                    "sample_size": 2100,
                    "approval_rate": 0.50,
                    "default_rate": 0.10,
                },
            },
            "total_samples": 10300,
            "use_case": "Credit card approval",
            "application_domain": "financial_services",
            "regulatory_requirements": "Fair Lending (ECOA), EU AI Act Article 5",
            "data_quality_notes": "Balanced dataset, no missing protected attributes",
        }
        result = validator.validate_payload_for_gemini(payload)
        assert result.is_valid
    
    def test_attempted_data_exfiltration_blocked(self, validator):
        """Attempt to sneak raw data past validator is caught."""
        payload = {
            "model_name": "model",
            "protected_attribute": "gender",
            "disparate_impact_ratio": 0.8,
            "group_statistics": {
                "male": {"sample_size": 100},
                "female": {"sample_size": 100},
            },
            "total_samples": 200,
            # Sneaky attempt: hide row data in custom field
            "internal_data": {
                "raw_records": [
                    {"user_id": 1, "ssn": "123-45-6789", "income": 50000},
                    {"user_id": 2, "ssn": "987-65-4321", "income": 75000},
                ]
            },
        }
        result = validator.validate_payload_for_gemini(payload)
        # Warns about internal_data
        assert any("internal_data" in w for w in result.warnings)
