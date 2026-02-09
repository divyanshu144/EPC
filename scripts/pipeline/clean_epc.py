#!/usr/bin/env python3
"""
Clean EPC core CSV and filter to South West region.

- Reads ew_epc_core.csv in streaming chunks.
- Filters to SOUTH_WEST_LAS.
- Standardises age band, fuel, tenure.
- Engineers useful features (gap, band score, flags).
- Applies simple outlier rules.
- Writes a single clean CSV:
    data/processed/ew_epc_core_clean_sw.csv
"""

from pathlib import Path
import pandas as pd

from config import (
    EPC_MERGED_CSV,
    EPC_CLEAN_SW_DIR,      # we’ll still use the directory, but for now just 1 csv
    SOUTH_WEST_LAS,
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


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Filter to South West LAs
    df = df[df["LOCAL_AUTHORITY_LABEL"].isin(SOUTH_WEST_LAS)]
    if df.empty:
        return df

    # 2) Keep only needed cols that exist in this chunk
    keep = [c for c in CORE_KEEP if c in df.columns]
    df = df[keep].copy()

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

        mode = "w" if first else "a"
        header = first
        clean.to_csv(OUT_CSV, index=False, mode=mode, header=header)
        first = False

        print(f"Wrote {len(clean):,} rows (total out: {n_rows_out:,})")

    print(f"✓ Done. In: {n_rows_in:,}  ->  Out (SW clean): {n_rows_out:,}")
    print(f"Saved clean CSV to: {OUT_CSV}")


if __name__ == "__main__":
    main()
