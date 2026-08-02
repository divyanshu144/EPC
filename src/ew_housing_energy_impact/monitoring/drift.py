"""Explainable data drift checks for EPC feature datasets.

These checks are operational heuristics, not statistical significance tests.
They are intentionally lightweight so they can run locally and in CI without a
monitoring service or external database.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_NUMERIC_SHIFT_THRESHOLD = 0.20
DEFAULT_MISSINGNESS_SHIFT_THRESHOLD = 0.10
DEFAULT_CATEGORICAL_SHIFT_THRESHOLD = 0.20
DEFAULT_COUNT_SHIFT_THRESHOLD = 0.30
DEFAULT_AUTHORITY_COVERAGE_THRESHOLD = 0.10

TARGET_COLUMNS = ("CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT", "BELOW_C", "BELOW_C_INT")
YEAR_COLUMN = "YEAR"
AUTHORITY_COLUMN = "ANALYSIS_AUTHORITY"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_category(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    return str(value)


def _distribution(series: pd.Series) -> dict[str, float]:
    values = series.map(_clean_category)
    if values.empty:
        return {}
    return values.value_counts(normalize=True, dropna=False).to_dict()


def _top_distribution(distribution: dict[str, float], limit: int = 10) -> dict[str, float]:
    return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:limit])


def total_variation_distance(reference: pd.Series, current: pd.Series) -> float:
    """Return total variation distance between two categorical distributions."""

    ref_dist = _distribution(reference)
    cur_dist = _distribution(current)
    categories = set(ref_dist) | set(cur_dist)
    if not categories:
        return 0.0
    return 0.5 * sum(
        abs(ref_dist.get(category, 0.0) - cur_dist.get(category, 0.0)) for category in categories
    )


def numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Iterable[str],
    *,
    relative_shift_threshold: float = DEFAULT_NUMERIC_SHIFT_THRESHOLD,
    missingness_shift_threshold: float = DEFAULT_MISSINGNESS_SHIFT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Calculate numeric feature drift summaries."""

    results: list[dict[str, Any]] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue

        ref = pd.to_numeric(reference[column], errors="coerce")
        cur = pd.to_numeric(current[column], errors="coerce")
        ref_mean = ref.mean()
        cur_mean = cur.mean()
        abs_shift = (
            abs(cur_mean - ref_mean) if pd.notna(ref_mean) and pd.notna(cur_mean) else np.nan
        )
        denominator = max(abs(ref_mean), 1.0) if pd.notna(ref_mean) else 1.0
        relative_shift = abs_shift / denominator if pd.notna(abs_shift) else np.nan
        ref_missing = float(ref.isna().mean()) if len(ref) else 0.0
        cur_missing = float(cur.isna().mean()) if len(cur) else 0.0
        missing_shift = abs(cur_missing - ref_missing)

        triggered_checks = []
        if pd.notna(relative_shift) and relative_shift > relative_shift_threshold:
            triggered_checks.append("numeric_relative_mean_shift")
        if missing_shift > missingness_shift_threshold:
            triggered_checks.append("missingness_shift")

        results.append(
            {
                "column": column,
                "reference_mean": None if pd.isna(ref_mean) else float(ref_mean),
                "current_mean": None if pd.isna(cur_mean) else float(cur_mean),
                "absolute_mean_shift": None if pd.isna(abs_shift) else float(abs_shift),
                "relative_mean_shift": None if pd.isna(relative_shift) else float(relative_shift),
                "reference_std": None if pd.isna(ref.std()) else float(ref.std()),
                "current_std": None if pd.isna(cur.std()) else float(cur.std()),
                "reference_missing_rate": ref_missing,
                "current_missing_rate": cur_missing,
                "missingness_shift": float(missing_shift),
                "alert": bool(triggered_checks),
                "triggered_checks": triggered_checks,
            }
        )
    return results


def categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Iterable[str],
    *,
    distribution_shift_threshold: float = DEFAULT_CATEGORICAL_SHIFT_THRESHOLD,
    missingness_shift_threshold: float = DEFAULT_MISSINGNESS_SHIFT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Calculate categorical feature drift summaries."""

    results: list[dict[str, Any]] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue

        ref = reference[column]
        cur = current[column]
        ref_missing = float(ref.isna().mean()) if len(ref) else 0.0
        cur_missing = float(cur.isna().mean()) if len(cur) else 0.0
        missing_shift = abs(cur_missing - ref_missing)
        distribution_shift = total_variation_distance(ref, cur)

        triggered_checks = []
        if distribution_shift > distribution_shift_threshold:
            triggered_checks.append("categorical_distribution_shift")
        if missing_shift > missingness_shift_threshold:
            triggered_checks.append("missingness_shift")

        results.append(
            {
                "column": column,
                "reference_top_distribution": _top_distribution(_distribution(ref)),
                "current_top_distribution": _top_distribution(_distribution(cur)),
                "distribution_shift": float(distribution_shift),
                "reference_missing_rate": ref_missing,
                "current_missing_rate": cur_missing,
                "missingness_shift": float(missing_shift),
                "alert": bool(triggered_checks),
                "triggered_checks": triggered_checks,
            }
        )
    return results


def _coverage_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    count_shift_threshold: float = DEFAULT_COUNT_SHIFT_THRESHOLD,
    authority_coverage_threshold: float = DEFAULT_AUTHORITY_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    reference_count = int(len(reference))
    current_count = int(len(current))
    denominator = max(reference_count, 1)
    certificate_count_change = (current_count - reference_count) / denominator

    triggered_checks = []
    if abs(certificate_count_change) > count_shift_threshold:
        triggered_checks.append("certificate_count_shift")

    coverage: dict[str, Any] = {
        "reference_certificate_count": reference_count,
        "current_certificate_count": current_count,
        "certificate_count_change": float(certificate_count_change),
        "triggered_checks": triggered_checks,
    }

    if YEAR_COLUMN in reference.columns and YEAR_COLUMN in current.columns:
        ref_year = pd.to_numeric(reference[YEAR_COLUMN], errors="coerce")
        cur_year = pd.to_numeric(current[YEAR_COLUMN], errors="coerce")
        coverage["reference_year_min"] = None if pd.isna(ref_year.min()) else int(ref_year.min())
        coverage["reference_year_max"] = None if pd.isna(ref_year.max()) else int(ref_year.max())
        coverage["current_year_min"] = None if pd.isna(cur_year.min()) else int(cur_year.min())
        coverage["current_year_max"] = None if pd.isna(cur_year.max()) else int(cur_year.max())
        if (
            coverage["reference_year_min"] != coverage["current_year_min"]
            or coverage["reference_year_max"] != coverage["current_year_max"]
        ):
            triggered_checks.append("year_range_change")

    if AUTHORITY_COLUMN in reference.columns and AUTHORITY_COLUMN in current.columns:
        ref_authorities = set(reference[AUTHORITY_COLUMN].dropna().astype(str))
        cur_authorities = set(current[AUTHORITY_COLUMN].dropna().astype(str))
        missing_authorities = sorted(ref_authorities - cur_authorities)
        new_authorities = sorted(cur_authorities - ref_authorities)
        authority_change_rate = len(missing_authorities) / max(len(ref_authorities), 1)
        coverage["reference_authority_count"] = len(ref_authorities)
        coverage["current_authority_count"] = len(cur_authorities)
        coverage["missing_authorities"] = missing_authorities
        coverage["new_authorities"] = new_authorities
        coverage["authority_coverage_change_rate"] = float(authority_change_rate)
        if authority_change_rate > authority_coverage_threshold or new_authorities:
            triggered_checks.append("authority_coverage_change")

    below_c_col = "BELOW_C_INT" if "BELOW_C_INT" in reference.columns else "BELOW_C"
    if below_c_col in reference.columns and below_c_col in current.columns:
        ref_below_c = pd.to_numeric(reference[below_c_col], errors="coerce")
        cur_below_c = pd.to_numeric(current[below_c_col], errors="coerce")
        ref_rate = ref_below_c.mean()
        cur_rate = cur_below_c.mean()
        class_balance_shift = (
            abs(cur_rate - ref_rate) if pd.notna(ref_rate) and pd.notna(cur_rate) else np.nan
        )
        coverage["reference_below_c_rate"] = None if pd.isna(ref_rate) else float(ref_rate)
        coverage["current_below_c_rate"] = None if pd.isna(cur_rate) else float(cur_rate)
        coverage["below_c_class_balance_shift"] = (
            None if pd.isna(class_balance_shift) else float(class_balance_shift)
        )

    coverage["alert"] = bool(triggered_checks)
    return coverage


def _default_numeric_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    common = [column for column in reference.columns if column in current.columns]
    return [
        column
        for column in common
        if pd.api.types.is_numeric_dtype(reference[column])
        or pd.api.types.is_numeric_dtype(current[column])
    ]


def _default_categorical_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    common = [column for column in reference.columns if column in current.columns]
    numeric = set(_default_numeric_columns(reference, current))
    return [
        column
        for column in common
        if column not in numeric
        and (
            pd.api.types.is_object_dtype(reference[column])
            or pd.api.types.is_categorical_dtype(reference[column])
            or pd.api.types.is_bool_dtype(reference[column])
        )
    ]


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
    numeric_shift_threshold: float = DEFAULT_NUMERIC_SHIFT_THRESHOLD,
    missingness_shift_threshold: float = DEFAULT_MISSINGNESS_SHIFT_THRESHOLD,
    categorical_shift_threshold: float = DEFAULT_CATEGORICAL_SHIFT_THRESHOLD,
) -> dict[str, Any]:
    """Build a JSON-serializable drift report for two feature datasets."""

    numeric_cols = list(numeric_columns or _default_numeric_columns(reference, current))
    categorical_cols = list(categorical_columns or _default_categorical_columns(reference, current))
    target_cols = [
        column
        for column in TARGET_COLUMNS
        if column in reference.columns and column in current.columns
    ]

    numeric_results = numeric_drift(
        reference,
        current,
        numeric_cols,
        relative_shift_threshold=numeric_shift_threshold,
        missingness_shift_threshold=missingness_shift_threshold,
    )
    categorical_results = categorical_drift(
        reference,
        current,
        categorical_cols,
        distribution_shift_threshold=categorical_shift_threshold,
        missingness_shift_threshold=missingness_shift_threshold,
    )
    target_results = [result for result in numeric_results if result["column"] in target_cols] + [
        result for result in categorical_results if result["column"] in target_cols
    ]
    coverage = _coverage_drift(reference, current)

    drifted_numeric = [result for result in numeric_results if result["alert"]]
    drifted_categorical = [result for result in categorical_results if result["alert"]]
    drifted_targets = [result for result in target_results if result["alert"]]

    return {
        "checked_at": _now_iso(),
        "thresholds": {
            "numeric_relative_mean_shift": numeric_shift_threshold,
            "categorical_distribution_shift": categorical_shift_threshold,
            "missingness_shift": missingness_shift_threshold,
            "certificate_count_shift": DEFAULT_COUNT_SHIFT_THRESHOLD,
            "authority_coverage_change": DEFAULT_AUTHORITY_COVERAGE_THRESHOLD,
        },
        "summary": {
            "numeric_columns_checked": len(numeric_results),
            "categorical_columns_checked": len(categorical_results),
            "target_columns_checked": len(target_results),
            "numeric_alert_count": len(drifted_numeric),
            "categorical_alert_count": len(drifted_categorical),
            "target_alert_count": len(drifted_targets),
            "coverage_alert": bool(coverage["alert"]),
        },
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
        "target_drift": target_results,
        "coverage": coverage,
    }
