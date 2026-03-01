"""Data validation utilities for EPC datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from scripts.pipeline.schemas import CLEAN_KEEP_COLS


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]
    summary: dict


def validate_clean_data(df: pd.DataFrame) -> ValidationResult:
    issues: list[str] = []

    # Required columns
    # Allow dropped high-missingness columns (cleaning step may remove them).
    optional_dropped = {"FLOOR_HEIGHT", "MAIN_HEATING_CONTROLS"}
    required = set(CLEAN_KEEP_COLS + ["YEAR", "POLICY_PERIOD", "BELOW_C"]) - optional_dropped
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")

    # Year range
    if "YEAR" in df.columns:
        yr_min = df["YEAR"].min()
        yr_max = df["YEAR"].max()
        if yr_min < 2008 or yr_max > 2025:
            issues.append(f"YEAR out of expected range: {yr_min}-{yr_max}")

    # EPC efficiency bounds
    if "CURRENT_ENERGY_EFFICIENCY" in df.columns:
        bad = (~df["CURRENT_ENERGY_EFFICIENCY"].between(1, 100)).sum()
        if bad > 0:
            issues.append(f"CURRENT_ENERGY_EFFICIENCY out of [1,100]: {bad} rows")

    # CO2 emissions reasonable range
    if "CO2_EMISSIONS_CURRENT" in df.columns:
        bad = (~df["CO2_EMISSIONS_CURRENT"].between(0, 30)).sum()
        if bad > 0:
            issues.append(f"CO2_EMISSIONS_CURRENT outside [0,30]: {bad} rows")

    # Floor area sanity
    if "TOTAL_FLOOR_AREA" in df.columns:
        bad = (~df["TOTAL_FLOOR_AREA"].between(20, 1000)).sum()
        if bad > 0:
            issues.append(f"TOTAL_FLOOR_AREA outside [20,1000]: {bad} rows")

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct_top10": (
            df.isna().mean().sort_values(ascending=False).head(10).to_dict()
        ),
    }

    return ValidationResult(ok=len(issues) == 0, issues=issues, summary=summary)


def write_validation_report(result: ValidationResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": result.ok,
        "issues": result.issues,
        "summary": result.summary,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def validate_imputed_csv(path: Path) -> ValidationResult:
    df = pd.read_csv(path)
    return validate_clean_data(df)
