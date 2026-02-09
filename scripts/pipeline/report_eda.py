#!/usr/bin/env python3
"""
Generate EDA figures/tables aligned with the dissertation report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import ROOT
from scripts.pipeline.policy import POLICY_ORDER


DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

sns.set(style="whitegrid", font_scale=1.0)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["POLICY_PERIOD"] = pd.Categorical(df["POLICY_PERIOD"], categories=POLICY_ORDER, ordered=True)
    return df


def simplify_age_band(age: str) -> str:
    s = str(age)
    if "1900" in s or "pre" in s.lower():
        return "Pre-1900"
    if "1900-1929" in s:
        return "1900-1929"
    if "1930-1949" in s:
        return "1930-1949"
    if "1950-1966" in s:
        return "1950-1966"
    if "1967-1975" in s:
        return "1967-1975"
    if "1976-1982" in s:
        return "1976-1982"
    if "1983-1990" in s:
        return "1983-1990"
    if "1991-1995" in s:
        return "1991-1995"
    if "1996-2002" in s:
        return "1996-2002"
    if "2003-2006" in s:
        return "2003-2006"
    if "2007-2011" in s:
        return "2007-2011"
    if "2012" in s or "2012-2021" in s:
        return "2012-2021"
    return "Unknown"


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    df = load_data()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) EPC Rating Distribution (South West)
    epc_order = ["A", "B", "C", "D", "E", "F", "G"]
    df["CURRENT_ENERGY_RATING"] = pd.Categorical(df["CURRENT_ENERGY_RATING"], categories=epc_order, ordered=True)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="CURRENT_ENERGY_RATING", order=epc_order, palette="viridis")
    plt.title("EPC Rating Distribution (South West)")
    plt.xlabel("EPC Rating")
    plt.ylabel("Count")
    save_fig(FIG_DIR / "figure_3_1_epc_rating_distribution.png")

    # 2) EPC Rating by Construction Age Band (grouped)
    df["AGE_BAND_GROUPED"] = df["AGE_BAND_STD"].apply(simplify_age_band)
    age_order = [
        "Pre-1900",
        "1900-1929",
        "1930-1949",
        "1950-1966",
        "1967-1975",
        "1976-1982",
        "1983-1990",
        "1991-1995",
        "1996-2002",
        "2003-2006",
        "2007-2011",
        "2012-2021",
    ]
    df_age = df[~df["AGE_BAND_GROUPED"].isin(["Unknown"])]
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df_age,
        x="AGE_BAND_GROUPED",
        hue="CURRENT_ENERGY_RATING",
        hue_order=epc_order,
        order=age_order,
        palette="viridis",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("EPC Rating Distribution by Construction Age Band (South West)")
    plt.xlabel("Construction Age Band")
    plt.ylabel("Count")
    save_fig(FIG_DIR / "figure_3_2_epc_by_age_band.png")

    # 3) CO2 Emissions by Fuel Type
    df_plot = df[df["CO2_EMISSIONS_CURRENT"].between(0, 30)]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_plot, x="MAIN_FUEL_STD", y="CO2_EMISSIONS_CURRENT", palette="viridis")
    plt.title("CO2 Emissions by Heating Fuel Type (South West)")
    plt.xlabel("Main Fuel")
    plt.ylabel("CO2 Emissions (tonnes/year)")
    save_fig(FIG_DIR / "figure_3_3_co2_by_fuel.png")

    # 4) Exeter vs Plymouth EPC trend
    df_city = df[df["LOCAL_AUTHORITY_LABEL"].isin(["Exeter", "Plymouth"])].copy()
    trend = (
        df_city.groupby(["YEAR", "LOCAL_AUTHORITY_LABEL"], as_index=False)
        .agg(mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"))
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=trend, x="YEAR", y="mean_epc", hue="LOCAL_AUTHORITY_LABEL", marker="o")
    plt.title("EPC Efficiency Trend Comparison (Exeter vs Plymouth)")
    plt.xlabel("Year")
    plt.ylabel("Mean EPC Efficiency")
    save_fig(FIG_DIR / "figure_3_4_exeter_plymouth_trend.png")

    # 5) Correlation heatmap
    num_cols = [
        "CURRENT_ENERGY_EFFICIENCY",
        "POTENTIAL_ENERGY_EFFICIENCY",
        "ENERGY_CONSUMPTION_CURRENT",
        "CO2_EMISSIONS_CURRENT",
        "TOTAL_FLOOR_AREA",
        "EFFICIENCY_GAP",
        "EPC_BAND_SCORE",
        "NUMBER_HABITABLE_ROOMS",
        "NUMBER_HEATED_ROOMS",
        "LOG_FLOOR_AREA",
        "YEAR",
    ]
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".1f", cmap="viridis", linewidths=0.7)
    plt.title("Correlation Heatmap of Key Numerical Variables (South West EPC Dataset)")
    save_fig(FIG_DIR / "figure_3_5_correlation_heatmap.png")

    # Table: Policy period summary
    policy_summary = (
        df.groupby("POLICY_PERIOD", as_index=False)
        .agg(
            mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
            median_epc=("CURRENT_ENERGY_EFFICIENCY", "median"),
            mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
            median_co2=("CO2_EMISSIONS_CURRENT", "median"),
            below_c_share=("BELOW_C", "mean"),
            n_cert=("LMK_KEY", "count"),
        )
    )
    policy_summary.to_csv(TABLE_DIR / "table_4_1_policy_period_summary.csv", index=False)

    print("✓ EDA figures and tables generated.")


if __name__ == "__main__":
    main()
