"""
EquiLens AI — Fairness Metrics

Deterministic statistical fairness metric implementations for binary
classification models. Each function accepts a pandas DataFrame, a
protected attribute column name (binary: 0=privileged, 1=unprivileged),
and a target/prediction column name (binary: 0=negative, 1=positive).

Implemented metrics:
    - Disparate Impact Ratio (DI)
    - Demographic Parity Difference (DPD) — also called Statistical Parity Difference
    - Equal Opportunity Difference (EOD)

Design decisions:
    - All functions return plain dicts for easy serialization / aggregation.
    - NaN / missing values are dropped before computation with a warning count.
    - Division-by-zero returns float('inf') with is_fair=False and a diagnostic message.
    - No external API or LLM calls — fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Container for a single fairness metric computation."""

    name: str
    value: float
    threshold: float
    is_fair: bool
    description: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_clean(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, int]:
    """
    Validate inputs and drop rows with NaN in the relevant columns.

    Args:
        df: Input DataFrame.
        protected_col: Name of the binary protected attribute column.
        target_col: Name of the binary target / prediction column.

    Returns:
        Tuple of (cleaned DataFrame, number of rows dropped).

    Raises:
        ValueError: If columns are missing or contain non-binary values.
    """
    # --- Column existence ---
    for col in (protected_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. "
                             f"Available: {list(df.columns)}")

    # --- Drop NaN ---
    subset = df[[protected_col, target_col]].copy()
    clean = subset.dropna()
    dropped = len(subset) - len(clean)

    if len(clean) == 0:
        raise ValueError("No valid (non-NaN) rows remain after cleaning.")

    # --- Auto-encode text to binary (0/1) ---
    for col in (protected_col, target_col):
        unique_vals = set(clean[col].unique())
        
        # Already numeric binary
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue
        
        # Text binary - map to 0/1 alphabetically
        if len(unique_vals) == 2:
            sorted_vals = sorted(list(unique_vals))
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            clean[col] = clean[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
        else:
            raise ValueError(
                f"Column '{col}' must be binary (exactly 2 unique values). "
                f"Found {len(unique_vals)}: {unique_vals}"
            )

    return clean, dropped


def _selection_rate(series: pd.Series) -> float:
    """Compute P(Y=1) for a binary Series. Returns 0.0 if empty."""
    if len(series) == 0:
        return 0.0
    return float(series.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def disparate_impact_ratio(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
    *,
    privileged_value: int = 0,
    fair_range: tuple[float, float] = (0.8, 1.25),
) -> dict:
    """
    Compute the Disparate Impact Ratio (DI).

    Formula:
        DI = P(Ŷ=1 | G=unprivileged) / P(Ŷ=1 | G=privileged)

    Interpretation:
        - DI = 1.0  → perfect parity
        - DI < 0.8  → adverse impact on unprivileged group (80% rule)
        - DI > 1.25 → adverse impact on privileged group

    Args:
        df: DataFrame containing the data.
        protected_col: Name of the binary protected attribute column
                       (0=privileged, 1=unprivileged by default).
        target_col: Name of the binary prediction / target column.
        privileged_value: Value identifying the privileged group (default 0).
        fair_range: (lower, upper) bounds for the fairness threshold
                    (default 0.8–1.25, the 80% rule).

    Returns:
        Dictionary with keys:
            metric, value, privileged_rate, unprivileged_rate,
            fair_range, is_fair, rows_used, rows_dropped, description

    Edge cases:
        - If P(Ŷ=1 | privileged) = 0 → returns inf with is_fair=False.
        - NaN rows are dropped; count reported in rows_dropped.
    """
    clean, dropped = _validate_and_clean(df, protected_col, target_col)

    priv_mask = clean[protected_col] == privileged_value
    unpriv_mask = ~priv_mask

    rate_priv = _selection_rate(clean.loc[priv_mask, target_col])
    rate_unpriv = _selection_rate(clean.loc[unpriv_mask, target_col])

    # --- Division by zero guard ---
    if rate_priv == 0.0:
        di_value = float("inf")
        description = ("Disparate Impact is undefined (inf): the privileged "
                       "group has a 0% positive rate, making the ratio "
                       "mathematically undefined.")
        is_fair = False
    else:
        di_value = rate_unpriv / rate_priv
        lower, upper = fair_range
        is_fair = lower <= di_value <= upper
        if is_fair:
            description = (f"DI = {di_value:.4f} is within the fair range "
                           f"[{lower}, {upper}]. No significant disparate impact.")
        else:
            description = (f"DI = {di_value:.4f} falls outside the fair range "
                           f"[{lower}, {upper}], indicating potential disparate impact.")

    return {
        "metric": "disparate_impact_ratio",
        "value": round(di_value, 6) if np.isfinite(di_value) else float("inf"),
        "privileged_rate": round(rate_priv, 6),
        "unprivileged_rate": round(rate_unpriv, 6),
        "fair_range": list(fair_range),
        "is_fair": is_fair,
        "rows_used": len(clean),
        "rows_dropped": dropped,
        "description": description,
    }


def demographic_parity_difference(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
    *,
    privileged_value: int = 0,
    threshold: float = 0.1,
) -> dict:
    """
    Compute the Demographic Parity Difference (DPD).

    Also known as Statistical Parity Difference (SPD).

    Formula:
        DPD = P(Ŷ=1 | G=unprivileged) − P(Ŷ=1 | G=privileged)

    Interpretation:
        - DPD = 0   → perfect demographic parity
        - DPD > 0   → unprivileged group has higher positive rate
        - DPD < 0   → privileged group has higher positive rate
        - |DPD| ≤ threshold → considered fair

    Args:
        df: DataFrame containing the data.
        protected_col: Name of the binary protected attribute column.
        target_col: Name of the binary prediction / target column.
        privileged_value: Value identifying the privileged group (default 0).
        threshold: Maximum acceptable |DPD| for fairness (default 0.1).

    Returns:
        Dictionary with keys:
            metric, value, privileged_rate, unprivileged_rate,
            threshold, is_fair, rows_used, rows_dropped, description

    Edge cases:
        - Empty groups yield a rate of 0.0 (difference is still computable).
        - NaN rows are dropped; count reported in rows_dropped.
    """
    clean, dropped = _validate_and_clean(df, protected_col, target_col)

    priv_mask = clean[protected_col] == privileged_value
    unpriv_mask = ~priv_mask

    rate_priv = _selection_rate(clean.loc[priv_mask, target_col])
    rate_unpriv = _selection_rate(clean.loc[unpriv_mask, target_col])

    dpd_value = rate_unpriv - rate_priv
    is_fair = abs(dpd_value) <= threshold

    if is_fair:
        description = (f"DPD = {dpd_value:+.4f} (|DPD| = {abs(dpd_value):.4f} "
                       f"≤ {threshold}). Demographic parity is satisfied.")
    else:
        favored = "unprivileged" if dpd_value > 0 else "privileged"
        description = (f"DPD = {dpd_value:+.4f} (|DPD| = {abs(dpd_value):.4f} "
                       f"> {threshold}). The {favored} group has a higher "
                       f"positive prediction rate.")

    return {
        "metric": "demographic_parity_difference",
        "value": round(dpd_value, 6),
        "privileged_rate": round(rate_priv, 6),
        "unprivileged_rate": round(rate_unpriv, 6),
        "threshold": threshold,
        "is_fair": is_fair,
        "rows_used": len(clean),
        "rows_dropped": dropped,
        "description": description,
    }


def equal_opportunity_difference(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
    prediction_col: str | None = None,
    *,
    privileged_value: int = 0,
    threshold: float = 0.1,
) -> dict:
    """
    Compute the Equal Opportunity Difference (EOD).

    Measures the difference in True Positive Rates (recall) between
    the unprivileged and privileged groups — i.e., among individuals
    who truly deserve a positive outcome, are both groups equally
    likely to receive one?

    Formula:
        EOD = TPR_unprivileged − TPR_privileged

        where TPR_g = P(Ŷ=1 | Y=1, G=g)

    Interpretation:
        - EOD = 0   → equal opportunity (equal recall across groups)
        - EOD > 0   → unprivileged group has higher TPR
        - EOD < 0   → privileged group has higher TPR
        - |EOD| ≤ threshold → considered fair

    Args:
        df: DataFrame containing the data.
        protected_col: Name of the binary protected attribute column.
        target_col: Name of the binary ground-truth label column.
        prediction_col: Name of the binary prediction column. If None,
                        defaults to target_col (self-evaluation).
        privileged_value: Value identifying the privileged group (default 0).
        threshold: Maximum acceptable |EOD| for fairness (default 0.1).

    Returns:
        Dictionary with keys:
            metric, value, privileged_tpr, unprivileged_tpr,
            threshold, is_fair, rows_used, rows_dropped, description

    Edge cases:
        - If a group has zero actual positives (Y=1), its TPR is
          undefined. Returns float('nan') for that group's TPR and
          is_fair=False with a diagnostic message.
        - NaN rows are dropped across all relevant columns.
    """
    if prediction_col is None:
        prediction_col = target_col

    # --- Validate all required columns ---
    cols = list({protected_col, target_col, prediction_col})
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. "
                             f"Available: {list(df.columns)}")

    # --- Clean NaN across all columns ---
    subset = df[cols].copy()
    clean = subset.dropna()
    dropped = len(subset) - len(clean)

    if len(clean) == 0:
        raise ValueError("No valid (non-NaN) rows remain after cleaning.")

    # --- Auto-encode text to binary (0/1) ---
    for col in cols:
        unique_vals = set(clean[col].unique())
        
        # Already numeric binary
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue
        
        # Text binary - map to 0/1 alphabetically
        if len(unique_vals) == 2:
            sorted_vals = sorted(list(unique_vals))
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            clean[col] = clean[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
        else:
            raise ValueError(
                f"Column '{col}' must be binary (exactly 2 unique values). "
                f"Found {len(unique_vals)}: {unique_vals}"
            )

    # --- Filter to actual positives (Y=1) ---
    positives = clean[clean[target_col] == 1]

    priv_mask = positives[protected_col] == privileged_value
    unpriv_mask = ~priv_mask

    priv_positives = positives.loc[priv_mask]
    unpriv_positives = positives.loc[unpriv_mask]

    # --- Compute TPR per group ---
    def _tpr(group_positives: pd.DataFrame) -> float:
        if len(group_positives) == 0:
            return float("nan")
        return float(group_positives[prediction_col].mean())

    tpr_priv = _tpr(priv_positives)
    tpr_unpriv = _tpr(unpriv_positives)

    # --- Handle undefined TPR ---
    if np.isnan(tpr_priv) or np.isnan(tpr_unpriv):
        eod_value = float("nan")
        is_fair = False
        missing = []
        if np.isnan(tpr_priv):
            missing.append("privileged")
        if np.isnan(tpr_unpriv):
            missing.append("unprivileged")
        description = (f"EOD is undefined: the {' and '.join(missing)} group(s) "
                       f"have zero actual positives (Y=1), making TPR undefined.")
    else:
        eod_value = tpr_unpriv - tpr_priv
        is_fair = abs(eod_value) <= threshold
        if is_fair:
            description = (f"EOD = {eod_value:+.4f} (|EOD| = {abs(eod_value):.4f} "
                           f"≤ {threshold}). Equal opportunity is satisfied.")
        else:
            favored = "unprivileged" if eod_value > 0 else "privileged"
            description = (f"EOD = {eod_value:+.4f} (|EOD| = {abs(eod_value):.4f} "
                           f"> {threshold}). The {favored} group has a higher "
                           f"true positive rate.")

    value_out = round(eod_value, 6) if np.isfinite(eod_value) else eod_value

    return {
        "metric": "equal_opportunity_difference",
        "value": value_out,
        "privileged_tpr": round(tpr_priv, 6) if np.isfinite(tpr_priv) else tpr_priv,
        "unprivileged_tpr": round(tpr_unpriv, 6) if np.isfinite(tpr_unpriv) else tpr_unpriv,
        "threshold": threshold,
        "is_fair": is_fair,
        "rows_used": len(clean),
        "rows_dropped": dropped,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Convenience: run all three metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str,
    prediction_col: str | None = None,
    *,
    privileged_value: int = 0,
) -> dict:
    """
    Run all fairness metrics and return a consolidated dictionary.

    Args:
        df: DataFrame containing the data.
        protected_col: Binary protected attribute column.
        target_col: Binary ground-truth label column.
        prediction_col: Binary prediction column (defaults to target_col).
        privileged_value: Value identifying the privileged group.

    Returns:
        Dictionary with keys 'disparate_impact_ratio',
        'demographic_parity_difference', 'equal_opportunity_difference',
        each mapping to that metric's result dict.
    """
    pred_col = prediction_col or target_col

    return {
        "disparate_impact_ratio": disparate_impact_ratio(
            df, protected_col, pred_col,
            privileged_value=privileged_value,
        ),
        "demographic_parity_difference": demographic_parity_difference(
            df, protected_col, pred_col,
            privileged_value=privileged_value,
        ),
        "equal_opportunity_difference": equal_opportunity_difference(
            df, protected_col, target_col, pred_col,
            privileged_value=privileged_value,
        ),
    }
