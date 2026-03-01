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

from config import EPC_CLEAN_SW_DIR, REGISTRY_PATH
from scripts.pipeline.policy import POLICY_ORDER, policy_period
from ew_housing_energy_impact.registry import register_artifact


IN_CSV = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw.csv"
OUT_IMPUTED = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw_imputed.csv"


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


def add_new_build_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add NEW_BUILD_SHARE: fraction of EPC lodgements that are new-build per LA-year.

    Why this matters
    ----------------
    EPC lodgements are event-driven (sale, rental, construction). In years with
    high new-build activity, more high-efficiency dwellings enter the dataset,
    mechanically raising the mean EPC score independent of retrofit policy.
    Including NEW_BUILD_SHARE as a covariate in the fixed-effects models isolates
    genuine within-LA efficiency improvements from this compositional shift.

    Computation
    -----------
    TRANSACTION_TYPE == 'new dwelling' flags new builds. The share is computed
    at the (LOCAL_AUTHORITY, YEAR) level and merged back to each individual
    record so the regression can use it as a continuous control.
    """
    if "TRANSACTION_TYPE" not in df.columns:
        df["NEW_BUILD_SHARE"] = np.nan
        return df

    df["_is_new_build"] = (
        df["TRANSACTION_TYPE"]
        .str.strip()
        .str.lower()
        .eq("new dwelling")
        .astype(int)
    )
    nb_share = (
        df.groupby(["LOCAL_AUTHORITY", "YEAR"], as_index=False)["_is_new_build"]
        .mean()
        .rename(columns={"_is_new_build": "NEW_BUILD_SHARE"})
    )
    df = df.drop(columns=["_is_new_build"]).merge(
        nb_share, on=["LOCAL_AUTHORITY", "YEAR"], how="left"
    )
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

    # New-build share covariate (controls for compositional shift)
    df = add_new_build_share(df)

    return df


def main() -> None:
    df = load_clean()
    df = impute_missing(df)
    df = add_derived(df)

    OUT_IMPUTED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_IMPUTED, index=False)
    register_artifact(REGISTRY_PATH, "dataset", OUT_IMPUTED, {"stage": "imputed"})

    print("✓ Feature build complete.")
    print(f"Saved: {OUT_IMPUTED}")


if __name__ == "__main__":
    main()
