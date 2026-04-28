"""
EquiLens AI — Data Anonymizer

Rule-based PII detection and anonymization module that sanitizes
DataFrames before fairness analysis. No LLMs, no external APIs —
fully deterministic, regex-driven logic.

Strategies:
    DROP        — Remove the column entirely.
    MASK        — Replace cell values with a fixed placeholder string.
    HASH        — One-way SHA-256 hash (preserves cardinality, not values).
    REDACT      — Pattern-aware partial masking (e.g. j***@***.com).
    GENERALIZE  — Reduce precision (ages → buckets, zip → prefix).

Design goals:
    • Detect PII via column-name patterns AND cell-content scanning.
    • Allow user-supplied overrides (extra sensitive cols, custom strategy).
    • Emit Python warnings when PII is auto-detected.
    • Preserve non-PII columns bit-for-bit so statistical structure is intact.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ===================================================================
# Constants & enums
# ===================================================================

class AnonymizationStrategy(Enum):
    """Available anonymization strategies."""

    DROP = "drop"
    MASK = "mask"
    HASH = "hash"
    REDACT = "redact"
    GENERALIZE = "generalize"


# Pre-compiled regex patterns for column-name-based PII detection.
# Keys are human-readable PII categories; values are compiled regexes
# that match common column names (case-insensitive).
_COLUMN_NAME_PATTERNS: dict[str, re.Pattern] = {
    "name": re.compile(
        r"(?:^|[_\-.])"
        r"(?:full[_\-.]?name|first[_\-.]?name|last[_\-.]?name|"
        r"middle[_\-.]?name|surname|given[_\-.]?name|"
        r"customer[_\-.]?name|user[_\-.]?name|person[_\-.]?name|"
        r"display[_\-.]?name|(?<!col)(?<!column)name)"
        r"(?:$|[_\-.])",
        re.IGNORECASE,
    ),
    "email": re.compile(
        r"(?:^|[_\-.])"
        r"(?:e[_\-.]?mail|email[_\-.]?addr(?:ess)?|contact[_\-.]?email)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(?:^|[_\-.])"
        r"(?:phone|telephone|tel|mobile|cell|fax|contact[_\-.]?number)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "address": re.compile(
        r"(?:^|[_\-.])"
        r"(?:address|street|city|state|zip[_\-.]?code|postal[_\-.]?code|"
        r"country|addr|residence|location|house[_\-.]?no)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "ssn": re.compile(
        r"(?:^|[_\-.])"
        r"(?:ssn|social[_\-.]?security|national[_\-.]?id|"
        r"tax[_\-.]?id|passport|driver[_\-.]?license)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "financial": re.compile(
        r"(?:^|[_\-.])"
        r"(?:credit[_\-.]?card|card[_\-.]?number|account[_\-.]?number|"
        r"bank[_\-.]?account|iban|routing[_\-.]?number|cvv)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "ip_address": re.compile(
        r"(?:^|[_\-.])"
        r"(?:ip[_\-.]?addr(?:ess)?|ip[_\-.]?v[46]|remote[_\-.]?addr)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
    "date_of_birth": re.compile(
        r"(?:^|[_\-.])"
        r"(?:dob|date[_\-.]?of[_\-.]?birth|birth[_\-.]?date|birthday)"
        r"(?:$|[_\-.])?",
        re.IGNORECASE,
    ),
}

# Compiled regexes for cell-content-based PII detection.
# Applied to string columns by sampling values.
_CONTENT_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    ),
    "phone": re.compile(
        r"(?:\+?\d{1,3}[\s\-.]?)"            # country code (required anchor)
        r"(?:"
        r"\(?\d{2,4}\)?[\s\-.]?"              # area code
        r"\d{3,4}[\s\-.]?\d{0,4}"             # local number
        r"|"
        r"\d{3}[\s\-.]?\d{4}"                 # simple 7-digit after country code
        r")",
    ),
    "ssn": re.compile(
        r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b",
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    ),
}

# Default strategy per PII category
_DEFAULT_STRATEGIES: dict[str, AnonymizationStrategy] = {
    "name": AnonymizationStrategy.MASK,
    "email": AnonymizationStrategy.REDACT,
    "phone": AnonymizationStrategy.MASK,
    "address": AnonymizationStrategy.DROP,
    "ssn": AnonymizationStrategy.DROP,
    "financial": AnonymizationStrategy.DROP,
    "ip_address": AnonymizationStrategy.HASH,
    "date_of_birth": AnonymizationStrategy.GENERALIZE,
    "credit_card": AnonymizationStrategy.DROP,
}

_MASK_PLACEHOLDER = "***REDACTED***"


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class AnonymizationConfig:
    """
    Configuration for the anonymization pipeline.

    Attributes:
        extra_sensitive_cols: Additional column names to treat as PII.
        column_strategies: Override strategy per column name.
        default_strategy: Fallback strategy for detected PII columns.
        content_scan_sample: Max rows to sample for content scanning.
        content_scan_threshold: Fraction of sampled values that must
            match a PII pattern to flag the column (0.0–1.0).
        mask_value: Placeholder string used by the MASK strategy.
        hash_salt: Optional salt prepended before hashing.
        generalize_age_bins: Bin edges for age generalization.
        skip_columns: Columns to explicitly exclude from anonymization
            (e.g. protected attributes needed for fairness analysis).
    """

    extra_sensitive_cols: list[str] = field(default_factory=list)
    column_strategies: dict[str, AnonymizationStrategy] = field(
        default_factory=dict,
    )
    default_strategy: AnonymizationStrategy = AnonymizationStrategy.MASK
    content_scan_sample: int = 200
    content_scan_threshold: float = 0.3
    mask_value: str = _MASK_PLACEHOLDER
    hash_salt: str = ""
    generalize_age_bins: list[int] = field(
        default_factory=lambda: [0, 18, 25, 35, 45, 55, 65, 100],
    )
    skip_columns: list[str] = field(default_factory=list)


# ===================================================================
# Detection
# ===================================================================

@dataclass
class PIIDetection:
    """Result of PII detection for a single column."""

    column: str
    category: str
    method: str          # "column_name" or "content_scan"
    strategy: AnonymizationStrategy
    confidence: float    # 0.0–1.0


def detect_pii_columns(
    df: pd.DataFrame,
    config: AnonymizationConfig | None = None,
) -> list[PIIDetection]:
    """
    Scan a DataFrame for columns likely containing PII.

    Detection is two-phase:
        1. Column-name matching against known PII patterns.
        2. Cell-content scanning on string columns (sampled).

    Args:
        df: The input DataFrame to scan.
        config: Anonymization configuration (uses defaults if None).

    Returns:
        List of PIIDetection results, one per detected column.
    """
    config = config or AnonymizationConfig()
    detections: list[PIIDetection] = []
    detected_cols: set[str] = set()

    skip = set(config.skip_columns)

    # --- Phase 1: Column-name pattern matching ---
    for col in df.columns:
        if col in skip:
            continue
        for category, pattern in _COLUMN_NAME_PATTERNS.items():
            if pattern.search(col):
                strategy = config.column_strategies.get(
                    col,
                    _DEFAULT_STRATEGIES.get(category, config.default_strategy),
                )
                detections.append(PIIDetection(
                    column=col,
                    category=category,
                    method="column_name",
                    strategy=strategy,
                    confidence=0.95,
                ))
                detected_cols.add(col)
                break

    # --- Phase 1b: Explicit extra_sensitive_cols ---
    for col in config.extra_sensitive_cols:
        if col in df.columns and col not in detected_cols and col not in skip:
            strategy = config.column_strategies.get(
                col, config.default_strategy,
            )
            detections.append(PIIDetection(
                column=col,
                category="user_specified",
                method="explicit",
                strategy=strategy,
                confidence=1.0,
            ))
            detected_cols.add(col)

    # --- Phase 2: Cell-content scanning ---
    for col in df.columns:
        if col in detected_cols or col in skip:
            continue

        # Only scan string/object columns
        if df[col].dtype not in (object, "string"):
            continue

        sample = df[col].dropna()
        if len(sample) == 0:
            continue

        if len(sample) > config.content_scan_sample:
            sample = sample.sample(
                n=config.content_scan_sample, random_state=42,
            )

        sample_str = sample.astype(str)

        for category, pattern in _CONTENT_PATTERNS.items():
            match_count = sample_str.apply(
                lambda v, p=pattern: bool(p.search(v))
            ).sum()
            match_ratio = match_count / len(sample_str)

            if match_ratio >= config.content_scan_threshold:
                strategy = config.column_strategies.get(
                    col,
                    _DEFAULT_STRATEGIES.get(category, config.default_strategy),
                )
                detections.append(PIIDetection(
                    column=col,
                    category=category,
                    method="content_scan",
                    strategy=strategy,
                    confidence=round(match_ratio, 4),
                ))
                detected_cols.add(col)
                break  # one detection per column

    return detections


# ===================================================================
# Anonymization transforms
# ===================================================================

def _apply_mask(series: pd.Series, mask_value: str) -> pd.Series:
    """Replace all non-NaN values with the mask placeholder."""
    return series.where(series.isna(), other=mask_value)


def _apply_hash(series: pd.Series, salt: str) -> pd.Series:
    """SHA-256 hash each value. Preserves NaN and cardinality."""
    def _hash_value(val: Any) -> str | float:
        if pd.isna(val):
            return np.nan
        raw = f"{salt}{val}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    return series.apply(_hash_value)


def _apply_redact_email(series: pd.Series) -> pd.Series:
    """Redact emails: john.doe@example.com → j***@***.com"""
    email_re = _CONTENT_PATTERNS["email"]

    def _redact(val: Any) -> Any:
        if pd.isna(val):
            return val
        s = str(val)
        match = email_re.search(s)
        if match:
            email = match.group()
            local, domain = email.split("@", 1)
            parts = domain.rsplit(".", 1)
            tld = parts[-1] if len(parts) > 1 else ""
            masked = f"{local[0]}***@***.{tld}" if tld else f"{local[0]}***@***"
            return email_re.sub(masked, s, count=1)
        return _MASK_PLACEHOLDER
    return series.apply(_redact)


def _apply_redact_generic(series: pd.Series) -> pd.Series:
    """Generic redaction: keep first and last char, mask middle."""
    def _redact(val: Any) -> Any:
        if pd.isna(val):
            return val
        s = str(val)
        if len(s) <= 2:
            return "*" * len(s)
        return f"{s[0]}{'*' * (len(s) - 2)}{s[-1]}"
    return series.apply(_redact)


def _apply_generalize_date(series: pd.Series) -> pd.Series:
    """Generalize dates to year only."""
    try:
        dates = pd.to_datetime(series, errors="coerce")
        return dates.dt.year.astype("Int64").astype(str).where(
            dates.notna(), other=np.nan,
        )
    except Exception:
        return _apply_mask(series, _MASK_PLACEHOLDER)


def _apply_generalize_numeric(
    series: pd.Series,
    bins: list[int],
) -> pd.Series:
    """Bin numeric values into ranges (e.g. age → 25-35)."""
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
        return pd.cut(
            numeric, bins=bins, labels=labels,
            include_lowest=True, right=False,
        ).astype(str).replace("nan", np.nan)
    except Exception:
        return _apply_mask(series, _MASK_PLACEHOLDER)


# ===================================================================
# Main anonymization function
# ===================================================================

def anonymize(
    df: pd.DataFrame,
    config: AnonymizationConfig | None = None,
    *,
    report: bool = True,
) -> tuple[pd.DataFrame, list[PIIDetection]]:
    """
    Anonymize a DataFrame by detecting and transforming PII columns.

    This function:
        1. Detects PII columns via name patterns and content scanning.
        2. Emits warnings for each auto-detected PII column.
        3. Applies the configured anonymization strategy per column.
        4. Returns the cleaned DataFrame and a detection report.

    Non-PII columns are returned **unchanged**, preserving the
    statistical structure needed for downstream fairness analysis.

    Args:
        df: Input DataFrame (not modified in place).
        config: Anonymization configuration. Uses sensible defaults
                if None.
        report: If True, emit Python warnings for each PII column
                detected.

    Returns:
        Tuple of (anonymized DataFrame, list of PIIDetection results).

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty.

    Example:
        >>> config = AnonymizationConfig(
        ...     extra_sensitive_cols=["nickname"],
        ...     skip_columns=["gender"],  # needed for fairness
        ... )
        >>> clean_df, detections = anonymize(df, config)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if len(df) == 0:
        raise ValueError("Cannot anonymize an empty DataFrame.")

    config = config or AnonymizationConfig()

    # --- Detect PII ---
    detections = detect_pii_columns(df, config)

    # --- Emit warnings ---
    if report and detections:
        for det in detections:
            warnings.warn(
                f"[EquiLens Anonymizer] PII detected in column '{det.column}' "
                f"(category: {det.category}, method: {det.method}, "
                f"confidence: {det.confidence:.0%}). "
                f"Strategy: {det.strategy.value}.",
                UserWarning,
                stacklevel=2,
            )

    # --- Apply transforms ---
    result = df.copy()

    for det in detections:
        col = det.column
        strategy = det.strategy

        if strategy == AnonymizationStrategy.DROP:
            result = result.drop(columns=[col])

        elif strategy == AnonymizationStrategy.MASK:
            result[col] = _apply_mask(result[col], config.mask_value)

        elif strategy == AnonymizationStrategy.HASH:
            result[col] = _apply_hash(result[col], config.hash_salt)

        elif strategy == AnonymizationStrategy.REDACT:
            if det.category == "email":
                result[col] = _apply_redact_email(result[col])
            else:
                result[col] = _apply_redact_generic(result[col])

        elif strategy == AnonymizationStrategy.GENERALIZE:
            if det.category == "date_of_birth":
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = _apply_generalize_numeric(
                        result[col], config.generalize_age_bins,
                    )
                else:
                    result[col] = _apply_generalize_date(result[col])
            else:
                result[col] = _apply_mask(result[col], config.mask_value)

    return result, detections


def get_anonymization_summary(
    detections: list[PIIDetection],
) -> dict[str, Any]:
    """
    Generate a human-readable summary dict from detection results.

    Returns:
        Dictionary with counts, column lists per category, and
        strategy breakdown.
    """
    summary: dict[str, Any] = {
        "total_pii_columns": len(detections),
        "categories": {},
        "strategies": {},
        "columns": [],
    }

    for det in detections:
        summary["columns"].append(det.column)

        # Category grouping
        cat = det.category
        if cat not in summary["categories"]:
            summary["categories"][cat] = []
        summary["categories"][cat].append(det.column)

        # Strategy grouping
        strat = det.strategy.value
        if strat not in summary["strategies"]:
            summary["strategies"][strat] = []
        summary["strategies"][strat].append(det.column)

    return summary
