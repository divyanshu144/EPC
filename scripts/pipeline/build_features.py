#!/usr/bin/env python3
"""
Build modelling-ready features from the cleaned South West EPC data.

Steps:
1) Drop high-missingness columns used only in EDA.
2) Impute missing values with simple, report-aligned rules.
3) Add missingness flags for moderate-missing fields.
4) Add derived fields (LOG_FLOOR_AREA, POLICY_PERIOD).
5) Save imputed + policy datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EPC_CLEAN_SW_DIR
from scripts.pipeline.policy import POLICY_ORDER, policy_period
from ew_housing_energy_impact.registry import register_artifact


IN_CSV = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw.csv"
OUT_IMPUTED = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw_imputed.csv"
OUT_WITH_POLICY = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw_with_policy.csv"
REGISTRY_PATH = EPC_CLEAN_SW_DIR.parent.parent.parent / "reports" / "artifacts" / "registry.jsonl"


def load_clean() -> pd.DataFrame:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {IN_CSV}")
    return pd.read_csv(IN_CSV)


def ensure_year(df: pd.DataFrame) -> pd.DataFrame:
    if "YEAR" in df.columns:
        return df
    if "LODGEMENT_DATE" not in df.columns:
        raise ValueError("YEAR missing and LODGEMENT_DATE not available.")
    df["LODGEMENT_DATE"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce", utc=True)
    df["YEAR"] = df["LODGEMENT_DATE"].dt.year
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # A: Drop high-missingness variables (report-aligned)
    drop_cols = ["FLOOR_HEIGHT", "MAIN_HEATING_CONTROLS"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # B: Impute minor categorical missingness with "Unknown"
    minor_cats = ["CONSTRUCTION_AGE_BAND", "MAIN_FUEL", "BUILT_FORM"]
    for col in minor_cats:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # C: Moderate missingness -> impute + missing flags
    moderate_cols = [
        "NUMBER_HABITABLE_ROOMS",
        "NUMBER_HEATED_ROOMS",
        "MAINS_GAS_FLAG",
        "MECHANICAL_VENTILATION",
        "TENURE",
    ]

    for col in moderate_cols:
        if col not in df.columns:
            continue
        df[f"{col}_MISSING"] = df[col].isna().astype(int)
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_year(df)

    # Hard clip efficiency scores to valid bounds
    if "CURRENT_ENERGY_EFFICIENCY" in df.columns:
        df = df[df["CURRENT_ENERGY_EFFICIENCY"].between(1, 100)]
    if "POTENTIAL_ENERGY_EFFICIENCY" in df.columns:
        df = df[df["POTENTIAL_ENERGY_EFFICIENCY"].between(1, 100)]

    if "LOG_FLOOR_AREA" not in df.columns and "TOTAL_FLOOR_AREA" in df.columns:
        df["LOG_FLOOR_AREA"] = np.log(df["TOTAL_FLOOR_AREA"])

    # Policy period assignment (report-aligned)
    df["POLICY_PERIOD"] = df["YEAR"].apply(policy_period)
    df["POLICY_PERIOD"] = pd.Categorical(
        df["POLICY_PERIOD"],
        categories=POLICY_ORDER,
        ordered=True,
    )
    return df


def main() -> None:
    df = load_clean()
    df = impute_missing(df)
    df = add_derived(df)

    OUT_IMPUTED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_IMPUTED, index=False)
    df.to_csv(OUT_WITH_POLICY, index=False)
    register_artifact(REGISTRY_PATH, "dataset", OUT_IMPUTED, {"stage": "imputed"})
    register_artifact(REGISTRY_PATH, "dataset", OUT_WITH_POLICY, {"stage": "with_policy"})

    print("✓ Feature build complete.")
    print(f"Saved: {OUT_IMPUTED}")
    print(f"Saved: {OUT_WITH_POLICY}")


if __name__ == "__main__":
    main()
