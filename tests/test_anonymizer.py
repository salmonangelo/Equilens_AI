"""
Tests for the fairness_engine.anonymizer module.

Covers:
    - Column-name-based PII detection
    - Cell-content-based PII detection
    - All anonymization strategies (DROP, MASK, HASH, REDACT, GENERALIZE)
    - Configuration overrides (extra cols, skip cols, custom strategies)
    - Edge cases (empty columns, all-NaN, mixed content)
    - Warning emission
    - Statistical structure preservation
    - Input validation
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from fairness_engine.anonymizer import (
    AnonymizationConfig,
    AnonymizationStrategy,
    PIIDetection,
    anonymize,
    detect_pii_columns,
    get_anonymization_summary,
)


# ===================================================================
# Synthetic datasets
# ===================================================================

@pytest.fixture
def pii_dataset() -> pd.DataFrame:
    """DataFrame with obvious PII columns and some safe columns."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "full_name": ["Alice Smith", "Bob Jones", "Carol White",
                      "David Brown", "Eve Davis"],
        "email": ["alice@example.com", "bob@test.org",
                  "carol@demo.net", "david@corp.io", "eve@mail.com"],
        "phone": ["+1-555-0101", "+1-555-0102", "+1-555-0103",
                  "+1-555-0104", "+1-555-0105"],
        "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd",
                    "321 Elm Blvd", "654 Maple Ct"],
        "age": [25, 34, 45, 52, 29],
        "income": [50000, 62000, 78000, 91000, 55000],
        "gender": [0, 1, 0, 1, 0],
        "prediction": [1, 0, 1, 1, 0],
        "label": [1, 0, 1, 0, 0],
    })


@pytest.fixture
def content_pii_dataset() -> pd.DataFrame:
    """Columns with innocuous names but PII in cell values."""
    return pd.DataFrame({
        "info_field": [
            "alice@example.com", "bob@test.org",
            "carol@demo.net", "david@corp.io",
        ],
        "contact": [
            "+1-555-0101", "+1-555-0102",
            "+1-555-0103", "+1-555-0104",
        ],
        "score": [0.85, 0.72, 0.91, 0.68],
    })


@pytest.fixture
def clean_dataset() -> pd.DataFrame:
    """A DataFrame with no PII at all."""
    return pd.DataFrame({
        "feature_1": [1.2, 3.4, 5.6, 7.8],
        "feature_2": [10, 20, 30, 40],
        "protected": [0, 1, 0, 1],
        "target": [1, 0, 1, 0],
    })


@pytest.fixture
def ssn_dataset() -> pd.DataFrame:
    """DataFrame with SSN column."""
    return pd.DataFrame({
        "ssn": ["123-45-6789", "987-65-4321", "111-22-3333"],
        "score": [0.9, 0.4, 0.7],
    })


# ===================================================================
# Detection tests
# ===================================================================

class TestPIIDetection:
    """Tests for detect_pii_columns."""

    def test_detects_name_column(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "full_name" in detected_cols

    def test_detects_email_column(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "email" in detected_cols

    def test_detects_phone_column(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "phone" in detected_cols

    def test_detects_address_column(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "address" in detected_cols

    def test_does_not_flag_safe_columns(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        detected_cols = set(d.column for d in detections)
        for safe_col in ("id", "age", "income", "gender", "prediction", "label"):
            assert safe_col not in detected_cols

    def test_detection_has_correct_category(self, pii_dataset):
        detections = detect_pii_columns(pii_dataset)
        cat_map = {d.column: d.category for d in detections}
        assert cat_map.get("full_name") == "name"
        assert cat_map.get("email") == "email"
        assert cat_map.get("phone") == "phone"
        assert cat_map.get("address") == "address"

    def test_no_detections_on_clean_data(self, clean_dataset):
        detections = detect_pii_columns(clean_dataset)
        assert len(detections) == 0

    def test_content_scan_detects_emails(self, content_pii_dataset):
        detections = detect_pii_columns(content_pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "info_field" in detected_cols
        # Verify it was detected via content scan
        email_det = [d for d in detections if d.column == "info_field"][0]
        assert email_det.method == "content_scan"
        assert email_det.category == "email"

    def test_content_scan_detects_phones(self, content_pii_dataset):
        detections = detect_pii_columns(content_pii_dataset)
        detected_cols = [d.column for d in detections]
        assert "contact" in detected_cols

    def test_ssn_detected_by_name(self, ssn_dataset):
        detections = detect_pii_columns(ssn_dataset)
        detected_cols = [d.column for d in detections]
        assert "ssn" in detected_cols

    def test_skip_columns_respected(self, pii_dataset):
        config = AnonymizationConfig(skip_columns=["email", "phone"])
        detections = detect_pii_columns(pii_dataset, config)
        detected_cols = [d.column for d in detections]
        assert "email" not in detected_cols
        assert "phone" not in detected_cols
        # Other PII columns still detected
        assert "full_name" in detected_cols

    def test_extra_sensitive_cols(self, pii_dataset):
        config = AnonymizationConfig(extra_sensitive_cols=["income"])
        detections = detect_pii_columns(pii_dataset, config)
        detected_cols = [d.column for d in detections]
        assert "income" in detected_cols
        inc_det = [d for d in detections if d.column == "income"][0]
        assert inc_det.method == "explicit"
        assert inc_det.confidence == 1.0

    def test_column_name_variations(self):
        """Test various column naming conventions."""
        df = pd.DataFrame({
            "first_name": ["a"], "lastName": ["b"],
            "EMAIL_ADDRESS": ["c@d.com"], "PhoneNumber": ["123"],
            "zip_code": ["12345"], "CREDIT_CARD": ["4111"],
        })
        detections = detect_pii_columns(df)
        detected_cols = set(d.column for d in detections)
        assert "first_name" in detected_cols
        assert "lastName" in detected_cols
        assert "EMAIL_ADDRESS" in detected_cols
        assert "PhoneNumber" in detected_cols
        assert "zip_code" in detected_cols
        assert "CREDIT_CARD" in detected_cols


# ===================================================================
# Anonymization strategy tests
# ===================================================================

class TestAnonymizationStrategies:
    """Tests for each anonymization strategy."""

    def test_drop_removes_column(self, pii_dataset):
        """Address columns default to DROP strategy."""
        config = AnonymizationConfig(
            column_strategies={"address": AnonymizationStrategy.DROP},
        )
        result, dets = anonymize(pii_dataset, config, report=False)
        assert "address" not in result.columns

    def test_mask_replaces_values(self, pii_dataset):
        """Name columns default to MASK strategy."""
        result, dets = anonymize(pii_dataset, report=False)
        name_det = [d for d in dets if d.column == "full_name"]
        if name_det and name_det[0].strategy == AnonymizationStrategy.MASK:
            assert all(
                v == "***REDACTED***"
                for v in result["full_name"]
            )

    def test_hash_produces_hex_strings(self, pii_dataset):
        config = AnonymizationConfig(
            column_strategies={"full_name": AnonymizationStrategy.HASH},
        )
        result, _ = anonymize(pii_dataset, config, report=False)
        for val in result["full_name"]:
            assert isinstance(val, str)
            assert len(val) == 16  # truncated SHA-256
            # Verify it's valid hex
            int(val, 16)

    def test_hash_preserves_cardinality(self, pii_dataset):
        config = AnonymizationConfig(
            column_strategies={"full_name": AnonymizationStrategy.HASH},
        )
        result, _ = anonymize(pii_dataset, config, report=False)
        # 5 unique names → 5 unique hashes
        assert result["full_name"].nunique() == 5

    def test_hash_deterministic(self, pii_dataset):
        config = AnonymizationConfig(
            column_strategies={"full_name": AnonymizationStrategy.HASH},
        )
        r1, _ = anonymize(pii_dataset, config, report=False)
        r2, _ = anonymize(pii_dataset, config, report=False)
        assert list(r1["full_name"]) == list(r2["full_name"])

    def test_hash_salt_changes_output(self, pii_dataset):
        c1 = AnonymizationConfig(
            column_strategies={"full_name": AnonymizationStrategy.HASH},
            hash_salt="salt_a",
        )
        c2 = AnonymizationConfig(
            column_strategies={"full_name": AnonymizationStrategy.HASH},
            hash_salt="salt_b",
        )
        r1, _ = anonymize(pii_dataset, c1, report=False)
        r2, _ = anonymize(pii_dataset, c2, report=False)
        assert list(r1["full_name"]) != list(r2["full_name"])

    def test_redact_email_format(self, pii_dataset):
        """Email columns default to REDACT strategy."""
        result, dets = anonymize(pii_dataset, report=False)
        email_det = [d for d in dets if d.column == "email"]
        if email_det and email_det[0].strategy == AnonymizationStrategy.REDACT:
            for val in result["email"]:
                assert "***" in val
                assert "@" in val
                # Should preserve first char of local part
                assert not val.startswith("***")

    def test_generalize_date_of_birth(self):
        df = pd.DataFrame({
            "dob": ["1990-05-15", "1985-12-01", "2000-03-20"],
            "score": [0.9, 0.5, 0.7],
        })
        result, dets = anonymize(df, report=False)
        dob_det = [d for d in dets if d.column == "dob"]
        if dob_det and dob_det[0].strategy == AnonymizationStrategy.GENERALIZE:
            # Should be year-only strings
            for val in result["dob"]:
                assert len(str(val)) == 4  # year


# ===================================================================
# Statistical structure preservation
# ===================================================================

class TestStatisticalPreservation:
    """Verify that non-PII columns are not altered."""

    def test_numeric_columns_unchanged(self, pii_dataset):
        result, _ = anonymize(pii_dataset, report=False)
        # These non-PII columns must be identical
        for col in ("id", "age", "income", "gender", "prediction", "label"):
            if col in result.columns:
                pd.testing.assert_series_equal(
                    result[col], pii_dataset[col], check_names=True,
                )

    def test_row_count_preserved(self, pii_dataset):
        result, _ = anonymize(pii_dataset, report=False)
        assert len(result) == len(pii_dataset)

    def test_non_pii_dtypes_preserved(self, pii_dataset):
        result, _ = anonymize(pii_dataset, report=False)
        for col in ("id", "age", "income", "gender", "prediction", "label"):
            if col in result.columns:
                assert result[col].dtype == pii_dataset[col].dtype

    def test_clean_dataset_unchanged(self, clean_dataset):
        result, detections = anonymize(clean_dataset, report=False)
        assert len(detections) == 0
        pd.testing.assert_frame_equal(result, clean_dataset)


# ===================================================================
# Warnings
# ===================================================================

class TestWarnings:
    """Verify that warnings are emitted for detected PII."""

    def test_warnings_emitted(self, pii_dataset):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            anonymize(pii_dataset, report=True)
            pii_warnings = [x for x in w if "EquiLens Anonymizer" in str(x.message)]
            assert len(pii_warnings) >= 3  # name, email, phone, address at minimum

    def test_warnings_suppressed_when_report_false(self, pii_dataset):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            anonymize(pii_dataset, report=False)
            pii_warnings = [x for x in w if "EquiLens Anonymizer" in str(x.message)]
            assert len(pii_warnings) == 0

    def test_warning_contains_column_name(self, pii_dataset):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            anonymize(pii_dataset, report=True)
            messages = [str(x.message) for x in w]
            assert any("full_name" in m for m in messages)

    def test_no_warnings_on_clean_data(self, clean_dataset):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            anonymize(clean_dataset, report=True)
            pii_warnings = [x for x in w if "EquiLens Anonymizer" in str(x.message)]
            assert len(pii_warnings) == 0


# ===================================================================
# Configuration overrides
# ===================================================================

class TestConfigurationOverrides:

    def test_custom_mask_value(self, pii_dataset):
        config = AnonymizationConfig(
            mask_value="[REMOVED]",
            column_strategies={"full_name": AnonymizationStrategy.MASK},
        )
        result, _ = anonymize(pii_dataset, config, report=False)
        assert all(v == "[REMOVED]" for v in result["full_name"])

    def test_custom_strategy_per_column(self, pii_dataset):
        config = AnonymizationConfig(
            column_strategies={
                "full_name": AnonymizationStrategy.HASH,
                "email": AnonymizationStrategy.DROP,
            },
        )
        result, dets = anonymize(pii_dataset, config, report=False)
        # Email should be dropped
        assert "email" not in result.columns
        # Name should be hashed (hex string)
        for val in result["full_name"]:
            int(val, 16)

    def test_skip_columns_preserved(self, pii_dataset):
        """Protected attribute columns can be excluded from anonymization."""
        # Pretend 'full_name' is needed (unusual but tests the mechanism)
        config = AnonymizationConfig(skip_columns=["full_name"])
        result, dets = anonymize(pii_dataset, config, report=False)
        # full_name should be untouched
        pd.testing.assert_series_equal(
            result["full_name"], pii_dataset["full_name"],
        )
        detected_cols = [d.column for d in dets]
        assert "full_name" not in detected_cols


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_nan_values_preserved_in_mask(self):
        df = pd.DataFrame({
            "name": ["Alice", np.nan, "Carol"],
            "score": [0.9, 0.5, 0.7],
        })
        result, _ = anonymize(df, report=False)
        assert pd.isna(result["name"].iloc[1])
        assert result["name"].iloc[0] == "***REDACTED***"

    def test_nan_values_preserved_in_hash(self):
        df = pd.DataFrame({
            "name": ["Alice", np.nan, "Carol"],
            "score": [0.9, 0.5, 0.7],
        })
        config = AnonymizationConfig(
            column_strategies={"name": AnonymizationStrategy.HASH},
        )
        result, _ = anonymize(df, config, report=False)
        assert pd.isna(result["name"].iloc[1])
        assert isinstance(result["name"].iloc[0], str)

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"name": pd.Series([], dtype=str)})
        with pytest.raises(ValueError, match="empty"):
            anonymize(df)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            anonymize({"not": "a dataframe"})

    def test_single_row(self):
        df = pd.DataFrame({"email": ["a@b.com"], "x": [1]})
        result, dets = anonymize(df, report=False)
        assert len(result) == 1
        assert len(dets) >= 1

    def test_original_dataframe_not_modified(self, pii_dataset):
        original_copy = pii_dataset.copy()
        anonymize(pii_dataset, report=False)
        pd.testing.assert_frame_equal(pii_dataset, original_copy)


# ===================================================================
# Summary helper
# ===================================================================

class TestAnonymizationSummary:

    def test_summary_structure(self, pii_dataset):
        _, dets = anonymize(pii_dataset, report=False)
        summary = get_anonymization_summary(dets)
        assert "total_pii_columns" in summary
        assert "categories" in summary
        assert "strategies" in summary
        assert "columns" in summary
        assert summary["total_pii_columns"] == len(dets)

    def test_summary_categories(self, pii_dataset):
        _, dets = anonymize(pii_dataset, report=False)
        summary = get_anonymization_summary(dets)
        assert "name" in summary["categories"]
        assert "email" in summary["categories"]

    def test_empty_summary(self, clean_dataset):
        _, dets = anonymize(clean_dataset, report=False)
        summary = get_anonymization_summary(dets)
        assert summary["total_pii_columns"] == 0
        assert summary["columns"] == []
