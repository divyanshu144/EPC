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

import re
from collections import Counter

import pandas as pd

from config import (
    ANALYSIS_AUTHORITIES,
    BAND_TO_SCORE,
    CODE_TO_ANALYSIS_AUTHORITY,
    EPC_CLEAN_SW_DIR,  # we’ll still use the directory, but for now just 1 csv
    EPC_MERGED_CSV,
    FUEL_MAP,
    LABEL_TO_ANALYSIS_AUTHORITY,
    TENURE_MAP,
)
from scripts.pipeline.schemas import CLEAN_DTYPES, CLEAN_KEEP_COLS

# ---------- CONFIG ----------
IN_CSV = EPC_MERGED_CSV
OUT_CSV = EPC_CLEAN_SW_DIR / "ew_epc_core_clean_sw.csv"
CHUNK = 1_000_000  # adjust based on RAM

CORE_KEEP = CLEAN_KEEP_COLS
DTYPES_LIGHT = CLEAN_DTYPES
# ----------------------------


# Canonical construction-age bands, ordered, with inclusive numeric bounds.
# `None` for the lower bound means open-ended below, `None` for the upper bound
# means open-ended above. This is the single source of truth for age-band
# vocabulary; report_eda.simplify_age_band derives its grouping from it.
AGE_BANDS: tuple[tuple[int | None, int | None, str], ...] = (
    (None, 1899, "England and Wales: before 1900"),
    (1900, 1929, "England and Wales: 1900-1929"),
    (1930, 1949, "England and Wales: 1930-1949"),
    (1950, 1966, "England and Wales: 1950-1966"),
    (1967, 1975, "England and Wales: 1967-1975"),
    (1976, 1982, "England and Wales: 1976-1982"),
    (1983, 1990, "England and Wales: 1983-1990"),
    (1991, 1995, "England and Wales: 1991-1995"),
    (1996, 2002, "England and Wales: 1996-2002"),
    (2003, 2006, "England and Wales: 2003-2006"),
    (2007, 2011, "England and Wales: 2007-2011"),
    (2012, 2021, "England and Wales: 2012-2021"),
    (2022, None, "England and Wales: 2022 onwards"),
)

_BARE_YEAR = re.compile(r"^\d{4}$")


def _band_for_year(year: int) -> str | None:
    """Return the canonical band containing `year`, or None if no band covers it."""
    for low, high, label in AGE_BANDS:
        if (low is None or year >= low) and (high is None or year <= high):
            return label
    return None


def norm_age_band(x: str) -> str:
    """Normalise construction age into a canonical band label.

    EPC records construction age two ways: as a band ("England and Wales:
    1950-1966") and, for many newer certificates, as a bare year ("2023"). On
    real South West data this produced 40 distinct levels, 27 of them bare years
    covering 13,338 rows (10.7%).

    That fragmentation is not cosmetic. Sparse single-year levels are what made
    three test rows carry categories unseen in training, which combined with a
    rank-deficient design matrix to produce a reported R2 of -1.13e15 (see
    tests/test_preprocessor.py). Collapsing a bare year into its band removes the
    sparse levels and makes the categorical dense.

    The bulk of the bare years were 2022-2025 (12,367 rows), which had no band at
    all because the published vocabulary stopped at "2012 onwards". AGE_BANDS
    therefore adds an open-ended "2022 onwards" band rather than forcing those
    rows into 2012-2021, which would be simply wrong.

    Unrecognised text is returned unchanged rather than bucketed into Unknown: a
    new label appearing in the source is information, and silently absorbing it
    would hide a vocabulary change.
    """
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()

    # Historical "onwards" phrasing for bands that have since closed.
    s = s.replace("England and Wales: 2012 onwards", "England and Wales: 2012-2021")
    s = s.replace("England and Wales: 2007 onwards", "England and Wales: 2007-2011")

    if _BARE_YEAR.match(s):
        band = _band_for_year(int(s))
        if band is not None:
            return band

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
    print(f"Analysis authorities present: {len(present)} of {len(ANALYSIS_AUTHORITIES)} expected")
    for auth in ANALYSIS_AUTHORITIES:
        print(f"    {auth:40s} {auth_counts.get(auth, 0):>12,d} rows")
    if missing:
        print(
            f"⚠ Missing expected analysis authorities (0 rows): {missing}. "
            "Check that the download step fetched their lower-tier/unitary files."
        )


if __name__ == "__main__":
    main()
