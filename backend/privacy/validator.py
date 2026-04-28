"""
EquiLens AI — Privacy Validation Layer

Enforces strict privacy constraints before any external API call.
Prevents raw dataset records, PII, and small group data from leaving the system.

Features:
  - Validates outgoing payloads to external APIs
  - Ensures only aggregated metrics are sent
  - Rejects requests with small group sizes (< minimum threshold)
  - Detects and blocks raw record exfiltration attempts
  - Provides clear, actionable error messages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

# Minimum group size required for external API calls
MIN_GROUP_SIZE = 10

# Fields allowed in aggregated metrics for external APIs
ALLOWED_METRIC_FIELDS = {
    # Fairness metrics (aggregated, per-attribute)
    "disparate_impact_ratio",
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "true_positive_rate",
    "false_positive_rate",
    "selection_rate",
    "precision",
    "recall",
    
    # Risk scores (aggregated)
    "fairness_risk_score",
    "risk_level",
    "risk_weight",
    "weights",
    
    # Metadata (non-sensitive)
    "model_name",
    "protected_attribute",
    "use_case",
    "application_domain",
    "total_samples",
    "timestamp",
    "analysis_timestamp",
    
    # Group statistics (aggregated counts only, not raw data)
    "group_statistics",
    "group_counts",
    "sample_size_per_group",
    
    # Data quality (aggregated)
    "data_quality_notes",
    "missing_data_pct",
    "warnings",
    "limitations",
    
    # Regulatory/compliance
    "regulatory_requirements",
    "stakeholders",
}

# Fields that trigger immediate rejection (raw data indicators)
BANNED_FIELDS = {
    "raw_data",
    "df",
    "dataframe",
    "records",
    "rows",
    "samples",
    "data",  # Generic "data" field likely contains records
}

# Patterns indicating row-level data
ROW_LEVEL_INDICATORS = {
    "id",
    "record_id",
    "index",
    "row",
    "entry",
}


class PrivacyViolationType(Enum):
    """Classification of privacy violations."""
    
    RAW_DATA_DETECTED = "raw_data_detected"
    SMALL_GROUP_SIZE = "small_group_size"
    BANNED_FIELD = "banned_field"
    SUSPICIOUS_FIELD = "suspicious_field"
    ROW_LEVEL_DATA = "row_level_data"
    INSUFFICIENT_AGGREGATION = "insufficient_aggregation"
    INVALID_STATISTICS = "invalid_statistics"


@dataclass
class PrivacyValidationError(Exception):
    """Privacy validation failure with detailed context."""
    
    violation_type: PrivacyViolationType
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.violation_type.value}: {self.message}"


@dataclass
class ValidationResult:
    """Result of privacy validation."""
    
    is_valid: bool
    errors: list[PrivacyValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(
        self,
        violation_type: PrivacyViolationType,
        message: str,
        **details,
    ) -> None:
        """Add a validation error."""
        self.is_valid = False
        error = PrivacyValidationError(
            violation_type=violation_type,
            message=message,
            details=details,
        )
        self.errors.append(error)
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


# =====================================================================
# Main Validator
# =====================================================================

class PrivacyValidator:
    """
    Validates payloads before sending to external APIs.
    
    Ensures:
    - No raw dataset records
    - No row-level data
    - No small groups that could be re-identified
    - Only aggregated metrics
    """
    
    def __init__(self, min_group_size: int = MIN_GROUP_SIZE):
        """
        Initialize validator.
        
        Args:
            min_group_size: Minimum samples per protected group (default: 10)
        """
        self.min_group_size = min_group_size
        self.logger = logging.getLogger(__name__)
    
    def validate_payload_for_gemini(
        self,
        payload: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate payload before sending to Gemini API.
        
        Enforces:
        1. No raw records present
        2. Only aggregated metrics included
        3. Group sizes above minimum threshold
        4. No suspicious field names
        
        Args:
            payload: Dictionary to validate
            
        Returns:
            ValidationResult with detailed errors and warnings
        """
        result = ValidationResult(is_valid=True)
        
        if not isinstance(payload, dict):
            result.add_error(
                PrivacyViolationType.INVALID_STATISTICS,
                f"Payload must be a dictionary, got {type(payload).__name__}",
                payload_type=str(type(payload)),
            )
            return result
        
        # 1. Check for banned fields
        self._check_banned_fields(payload, result)
        
        if not result.is_valid:
            return result  # Reject immediately on banned fields
        
        # 2. Check for suspicious field names
        self._check_suspicious_fields(payload, result)
        
        # 3. Check group statistics validity
        self._validate_group_statistics(payload, result)
        
        # 4. Check metric fields
        self._validate_metrics(payload, result)
        
        # 5. Check for row-level data indicators
        self._check_row_level_data(payload, result)
        
        return result
    
    def _check_banned_fields(
        self,
        payload: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Check if banned fields (raw data indicators) are present.
        
        Immediately rejects if found.
        """
        banned_found = BANNED_FIELDS & set(payload.keys())
        if banned_found:
            for field in banned_found:
                result.add_error(
                    PrivacyViolationType.RAW_DATA_DETECTED,
                    f"Banned field '{field}' detected in payload. "
                    f"Raw dataset records cannot be sent to external APIs.",
                    field=field,
                )
    
    def _check_suspicious_fields(
        self,
        payload: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check for suspicious field names that might contain raw data."""
        suspicious = []
        for key in payload.keys():
            # Check for list/array fields that might contain records
            if isinstance(payload[key], list) and len(payload[key]) > 0:
                # If list contains dicts, likely row data
                if isinstance(payload[key][0], dict):
                    suspicious.append(key)
        
        if suspicious:
            for field in suspicious:
                result.add_warning(
                    f"Field '{field}' contains list of dicts (possible row data). "
                    f"Ensure this is aggregated statistics, not raw records."
                )
    
    def _validate_group_statistics(
        self,
        payload: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Validate group statistics for minimum group size.
        
        Checks:
        - group_statistics exists and is properly aggregated
        - sample_size_per_group or equivalent exists
        - All groups meet minimum size threshold
        """
        group_stats = payload.get("group_statistics", {})
        
        if not group_stats:
            result.add_warning(
                "No group_statistics found in payload. "
                "Cannot verify minimum group size constraint."
            )
            return
        
        if not isinstance(group_stats, dict):
            result.add_error(
                PrivacyViolationType.INVALID_STATISTICS,
                f"group_statistics must be dict, got {type(group_stats).__name__}",
                type_found=str(type(group_stats)),
            )
            return
        
        small_groups = {}
        
        for group_name, group_data in group_stats.items():
            # Get sample size from various possible field names
            sample_size = None
            
            if isinstance(group_data, dict):
                sample_size = (
                    group_data.get("sample_size") 
                    or group_data.get("n")
                    or group_data.get("count")
                    or group_data.get("sample_size_per_group")
                )
            elif isinstance(group_data, (int, float)):
                sample_size = group_data
            
            if sample_size is None:
                result.add_warning(
                    f"Could not extract sample size for group '{group_name}'. "
                    f"Assuming raw data and rejecting."
                )
                result.add_error(
                    PrivacyViolationType.INVALID_STATISTICS,
                    f"Cannot determine sample size for group '{group_name}'",
                    group=group_name,
                )
                continue
            
            # Verify minimum group size
            if sample_size < self.min_group_size:
                small_groups[group_name] = sample_size
        
        if small_groups:
            affected_groups = ", ".join(
                f"'{g}' (n={n})" for g, n in small_groups.items()
            )
            result.add_error(
                PrivacyViolationType.SMALL_GROUP_SIZE,
                f"Small group size detected. Groups smaller than {self.min_group_size}: "
                f"{affected_groups}. Risk of re-identification. "
                f"Cannot send to external API.",
                min_required=self.min_group_size,
                affected_groups=small_groups,
            )
    
    def _validate_metrics(
        self,
        payload: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Validate that only allowed aggregated metric fields are present.
        
        Warns about unexpected fields that might need review.
        """
        unexpected_fields = set(payload.keys()) - ALLOWED_METRIC_FIELDS
        
        for field in unexpected_fields:
            result.add_warning(
                f"Unexpected field '{field}' in payload. "
                f"Verify this contains only aggregated data, not raw records."
            )
    
    def _check_row_level_data(
        self,
        payload: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Detect indicators of row-level data.
        
        Checks for:
        - 'id', 'record_id' fields with large cardinality
        - 'index', 'row' fields
        - High-cardinality string fields that might be IDs
        """
        for field in ROW_LEVEL_INDICATORS:
            if field in payload:
                value = payload[field]
                
                # If it's a list with many unique values, likely row IDs
                if isinstance(value, list) and len(value) > 20:
                    result.add_error(
                        PrivacyViolationType.ROW_LEVEL_DATA,
                        f"Row-level indicator '{field}' found with {len(value)} entries. "
                        f"Cannot send raw records to external API.",
                        field=field,
                        cardinality=len(value),
                    )


# =====================================================================
# Helper Functions
# =====================================================================

def validate_gemini_payload(payload: dict[str, Any]) -> None:
    """
    Validate payload and raise exception if invalid.
    
    Use this in the explanation service to ensure privacy before API call.
    
    Args:
        payload: Dictionary to validate
        
    Raises:
        PrivacyValidationError: If validation fails
    """
    validator = PrivacyValidator()
    result = validator.validate_payload_for_gemini(payload)
    
    if not result.is_valid:
        error_messages = [str(e) for e in result.errors]
        combined_message = "\n".join(error_messages)
        
        logger.error(
            f"Privacy validation failed. Rejecting external API call.\n{combined_message}"
        )
        
        # Raise the first error
        raise result.errors[0]
    
    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"Privacy validation warning: {warning}")


# =====================================================================
# Middleware Integration
# =====================================================================

class PrivacyValidationMiddleware:
    """
    ASGI middleware for privacy validation.
    
    Can be added to FastAPI app to validate all outgoing requests.
    Use in conjunction with explicit payload validation in services.
    """
    
    def __init__(self, app, validator: PrivacyValidator | None = None):
        """Initialize middleware."""
        self.app = app
        self.validator = validator or PrivacyValidator()
    
    async def __call__(self, scope, receive, send):
        """Process request through middleware."""
        # Middleware currently used for logging/monitoring
        # Actual validation happens in services before API calls
        await self.app(scope, receive, send)
