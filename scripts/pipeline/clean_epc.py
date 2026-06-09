#!/usr/bin/env python3
"""
Clean EPC core CSV and filter to South West region.

- Reads ew_epc_core.csv in streaming chunks.
- Filters to South West rows and rolls lower-tier/unitary authorities up to the
  14 analysis authorities (config.SOUTH_WEST_EPC_AUTHORITIES), preserving the raw
  EPC identifiers in *_RAW columns and exposing ANALYSIS_AUTHORITY.
- Standardises age band, fuel, tenure.
- Engineers useful features (gap, band score, flags).
- Applies simple outlier rules.
- Writes a single clean CSV:
    data/processed/ew_epc_core_clean_sw.csv
"""

from pathlib import Path
import pandas as pd

from collections import Counter

from config import (
    EPC_MERGED_CSV,
    EPC_CLEAN_SW_DIR,      # we’ll still use the directory, but for now just 1 csv
    ANALYSIS_AUTHORITIES,
    CODE_TO_ANALYSIS_AUTHORITY,
    LABEL_TO_ANALYSIS_AUTHORITY,
    BAND_TO_SCORE,
    FUEL_MAP,
    TENURE_MAP,
)
from scripts.pipeline.schemas import CLEAN_KEEP_COLS, CLEAN_DTYPES

# ---------- CONFIG ----------
IN_CSV = EPC_MERGED_CSV
OUT_CSV = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw.csv"
CHUNK = 1_000_000  # adjust based on RAM

CORE_KEEP = CLEAN_KEEP_COLS
DTYPES_LIGHT = CLEAN_DTYPES
# ----------------------------


def norm_age_band(x: str) -> str:
    """Normalise age band textual variants into consistent labels."""
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    s = s.replace("England and Wales: 2012 onwards", "England and Wales: 2012-2021")
    s = s.replace("England and Wales: 2007 onwards", "England and Wales: 2007-2011")
    return s


def roll_up_to_analysis_authority(df: pd.DataFrame) -> pd.Series:
    """Map each row to one of the 14 South West analysis authorities.

    EPC supplies data at lower-tier/unitary level (e.g. 'Exeter', 'Cheltenham',
    'Mendip'); these must roll up to the dissertation's upper-tier clusters
    (Devon, Gloucestershire, Somerset, …). Matching is by GSS code first
    (LOCAL_AUTHORITY), falling back to the certificate LOCAL_AUTHORITY_LABEL.
    Returns a Series of analysis-authority names (NaN where not South West).
    """
    code = df["LOCAL_AUTHORITY"].astype("string")
    analysis = code.map(CODE_TO_ANALYSIS_AUTHORITY)
    # Fallback to label for any rows whose code did not resolve.
    need = analysis.isna()
    if need.any() and "LOCAL_AUTHORITY_LABEL" in df.columns:
        label = df["LOCAL_AUTHORITY_LABEL"].astype("string")
        analysis = analysis.where(~need, label.map(LABEL_TO_ANALYSIS_AUTHORITY))
    return analysis


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Filter to South West rows by rolling up to the 14 analysis authorities.
    #    Works on lower-tier/unitary codes *and* labels, so district-level rows
    #    (e.g. Exeter -> Devon) are kept rather than dropped.
    analysis = roll_up_to_analysis_authority(df)
    mask = analysis.notna().to_numpy()
    df = df[mask].copy()
    if df.empty:
        return df
    analysis = analysis[mask]

    # 2) Keep only needed cols that exist in this chunk, then attach the rollup
    #    while preserving the raw EPC identifiers.
    keep = [c for c in CORE_KEEP if c in df.columns]
    raw_code = df["LOCAL_AUTHORITY"].to_numpy()
    raw_label = df["LOCAL_AUTHORITY_LABEL"].to_numpy()
    df = df[keep].copy()
    df["LOCAL_AUTHORITY_RAW"] = raw_code
    df["LOCAL_AUTHORITY_LABEL_RAW"] = raw_label
    df["ANALYSIS_AUTHORITY"] = analysis.to_numpy()
    # Overwrite the entity key + label with the analysis authority so all
    # downstream code (FE entity index, clustering, new-build share, EDA) operates
    # at the 14-authority level. Raw values are preserved above.
    df["LOCAL_AUTHORITY"] = df["ANALYSIS_AUTHORITY"]
    df["LOCAL_AUTHORITY_LABEL"] = df["ANALYSIS_AUTHORITY"]

    # 3) Dates & Year
    df["LODGEMENT_DATE"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce", utc=True)
    df["YEAR"] = df["LODGEMENT_DATE"].dt.year

    # 4) Standardise categorical text
    df["MAIN_FUEL_STD"] = df["MAIN_FUEL"].map(FUEL_MAP).fillna("other")
    df["TENURE_STD"] = df["TENURE"].map(TENURE_MAP).fillna("Unknown")
    df["AGE_BAND_STD"] = df["CONSTRUCTION_AGE_BAND"].apply(norm_age_band)

    # 5) Useful engineered fields
    df["EFFICIENCY_GAP"] = df["POTENTIAL_ENERGY_EFFICIENCY"] - df["CURRENT_ENERGY_EFFICIENCY"]
    df = df[df["EFFICIENCY_GAP"] >= 0]  # enforce plausible gap

    df["EPC_BAND_SCORE"] = df["CURRENT_ENERGY_RATING"].map(BAND_TO_SCORE)
    df["HAS_MAINS_GAS"] = df["MAINS_GAS_FLAG"].fillna("Unknown").eq("Y")
    df["IS_ELECTRIC_HEAT"] = df["MAIN_FUEL_STD"].eq("electricity")
    df["BELOW_C"] = df["CURRENT_ENERGY_EFFICIENCY"] < 69

    # 6) Simple outlier rules
    df = df[df["TOTAL_FLOOR_AREA"].between(20, 1000)]
    if "CO2_EMISSIONS_CURRENT" in df.columns:
        df = df[df["CO2_EMISSIONS_CURRENT"].between(0, 30)]

    return df


def main() -> None:
    EPC_CLEAN_SW_DIR.mkdir(parents=True, exist_ok=True)

    # fresh run: remove old CSV if exists
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    n_rows_in = 0
    n_rows_out = 0
    first = True
    auth_counts: Counter = Counter()

    for chunk in pd.read_csv(
        IN_CSV,
        dtype=DTYPES_LIGHT,
        low_memory=False,
        chunksize=CHUNK,
    ):
        n_rows_in += len(chunk)
        clean = clean_chunk(chunk)
        if clean.empty:
            continue

        n_rows_out += len(clean)
        auth_counts.update(clean["ANALYSIS_AUTHORITY"].value_counts().to_dict())

        mode = "w" if first else "a"
        header = first
        clean.to_csv(OUT_CSV, index=False, mode=mode, header=header)
        first = False

        print(f"Wrote {len(clean):,} rows (total out: {n_rows_out:,})")

    print(f"✓ Done. In: {n_rows_in:,}  ->  Out (SW clean): {n_rows_out:,}")
    print(f"Saved clean CSV to: {OUT_CSV}")
    _report_geography(auth_counts)


def _report_geography(auth_counts: Counter) -> None:
    """Print the analysis-authority rollup summary and flag missing clusters."""
    present = [a for a in ANALYSIS_AUTHORITIES if auth_counts.get(a, 0) > 0]
    missing = [a for a in ANALYSIS_AUTHORITIES if auth_counts.get(a, 0) == 0]

    print("\n--- Geography rollup ---")
    print(
        f"Analysis authorities present: {len(present)} of "
        f"{len(ANALYSIS_AUTHORITIES)} expected"
    )
    for auth in ANALYSIS_AUTHORITIES:
        print(f"    {auth:40s} {auth_counts.get(auth, 0):>12,d} rows")
    if missing:
        print(
            f"⚠ Missing expected analysis authorities (0 rows): {missing}. "
            "Check that the download step fetched their lower-tier/unitary files."
        )


if __name__ == "__main__":
    main()
