#!/usr/bin/env python3
"""
Generate a simple HTML data quality dashboard:
 - Missingness table
 - Basic stats
 - Drift proxy: compare last 2 years vs previous years
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import ROOT
from ew_housing_energy_impact.registry import register_artifact


DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
OUT_HTML = ROOT / "reports" / "artifacts" / "data_quality_report.html"
REGISTRY_PATH = ROOT / "reports" / "artifacts" / "registry.jsonl"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = (
        df.isna().mean().sort_values(ascending=False).rename("missing_pct").head(25) * 100
    ).to_frame()

    stats = df.describe(include="all").transpose().head(30)

    # Drift proxy: compare last 2 years vs previous years on numeric columns
    if "YEAR" in df.columns:
        recent_years = sorted(df["YEAR"].dropna().unique())[-2:]
        recent = df[df["YEAR"].isin(recent_years)]
        prev = df[~df["YEAR"].isin(recent_years)]
        num_cols = df.select_dtypes(include="number").columns
        drift = pd.DataFrame({
            "recent_mean": recent[num_cols].mean(),
            "prev_mean": prev[num_cols].mean(),
        })
        drift["mean_diff"] = drift["recent_mean"] - drift["prev_mean"]
        drift = drift.sort_values("mean_diff", key=lambda s: s.abs(), ascending=False).head(25)
    else:
        drift = pd.DataFrame()

    html = """
    <html>
      <head>
        <title>Data Quality Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; }}
          th {{ background: #f4f4f4; }}
          h2 {{ margin-top: 32px; }}
        </style>
      </head>
      <body>
        <h1>Data Quality Report</h1>
        <h2>Missingness (Top 25)</h2>
        {missing}
        <h2>Summary Stats (Top 30)</h2>
        {stats}
        <h2>Drift Proxy (Recent vs Previous Years)</h2>
        {drift}
      </body>
    </html>
    """
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(
        html.format(
            missing=missing.to_html(),
            stats=stats.to_html(),
            drift=drift.to_html() if not drift.empty else "<p>No YEAR column available.</p>",
        ),
        encoding="utf-8",
    )
    register_artifact(REGISTRY_PATH, "quality_report", OUT_HTML, {"type": "html"})

    print(f"✓ Data quality report written: {OUT_HTML}")


if __name__ == "__main__":
    main()
