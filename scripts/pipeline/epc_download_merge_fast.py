#!/usr/bin/env python3
"""
EPC Bulk Data Downloader and Merger
-----------------------------------
Downloads the annual EPC bulk archives covering all of England & Wales, then
filters to SW LAs during the clean step. Per-authority LA-mode selection is
retired: the rebuilt EPC service serves one national archive per period, not
one archive per local authority (see ew_housing_energy_impact.ingestion.epc_source).

Download/validate/extract mechanics live in
`ew_housing_energy_impact.ingestion.epc_source` and are re-exported here so
this module keeps importing for existing callers (the Prefect task, the CLI).

Usage:
    1. Set EPC_BEARER_TOKEN in your .env file (root directory).
    2. Optionally set YEARS to restrict which annual archives are fetched.
    3. Run from project root:
         python -m scripts.pipeline.epc_download_merge_fast
"""

import glob
import os

import pandas as pd

from config import EPC_MERGED_CSV, RAW_EXTRACT_DIR
from ew_housing_energy_impact.ingestion.epc_source import (  # noqa: F401
    _is_valid_zip,
    auth_headers,
    download_all,
    download_full_load,
    download_one_file,
    ensure_dirs,
    extract_all_zips,
    fetch_download_info,
)
from scripts.pipeline.full_load_schema import normalise_full_load_columns, resolve_usecols
from scripts.pipeline.schemas import CORE_COLS, CORE_DTYPES

# -------------- CONFIG -------------- #

YEARS = None  # None = all years, or e.g. {2023, 2024}

OUT_PATH = str(EPC_MERGED_CSV)

CHUNK = 2_000_000
MAX_WORKERS = 4
TIMEOUT = 120

DTYPES = CORE_DTYPES

# ------------------------------------ #


def list_available_files(headers: dict) -> dict:
    """Get list of all files available via EPC API."""
    del headers
    raise NotImplementedError(
        "Listing individual files is retired. The rebuilt EPC service serves "
        "one national archive; download_full_load() fetches it directly."
    )


def pick_year_from_name(name: str) -> int | None:
    """Extract year from filename like domestic-2024.zip."""
    if name.startswith("domestic-") and name.endswith(".zip"):
        try:
            return int(name.split("-")[1].split(".")[0])
        except Exception:
            return None
    return None


def merge_core_columns() -> None:
    """Merge every year's certificates.csv into one master CSV.

    Column names are resolved per file rather than assumed. The full-load
    archive and the older monthly archives use different vocabularies — the
    former is lowercase with `certificate_number` and `uprn` where the latter
    had `LMK_KEY` and `BUILDING_REFERENCE_NUMBER` — so reading with a fixed
    `usecols=CORE_COLS` fails outright against the full load. Resolving from
    each file's own header lets both shapes merge into one canonical output.
    """
    csv_files = sorted(glob.glob(str(RAW_EXTRACT_DIR / "*" / "certificates.csv")))
    if not csv_files:
        print("No certificates.csv files found.")
        return

    # Fresh write every run
    if EPC_MERGED_CSV.exists():
        EPC_MERGED_CSV.unlink()

    first = True
    for csv_file in csv_files:
        header = pd.read_csv(csv_file, nrows=0).columns.tolist()
        usecols, rename_map = resolve_usecols(CORE_COLS, header)
        if not usecols:
            print(f"Skipping {csv_file}: no recognised columns.")
            continue

        missing = sorted(set(CORE_COLS) - set(rename_map.values()))
        note = f"  (absent: {', '.join(missing)})" if missing else ""
        print(f"Merging {csv_file} ...{note}")

        for chunk in pd.read_csv(
            csv_file,
            usecols=usecols,
            dtype=str,
            chunksize=CHUNK,
            low_memory=True,
        ):
            chunk = normalise_full_load_columns(chunk.rename(columns=rename_map))
            # Keep a stable column order so appended chunks line up.
            chunk = chunk.reindex(columns=[c for c in CORE_COLS if c in chunk.columns])
            mode, header_flag = ("w", True) if first else ("a", False)
            chunk.to_csv(OUT_PATH, index=False, mode=mode, header=header_flag)
            first = False

    print(f"✓ Merged output saved to: {OUT_PATH}")


def main(mode: str = "la") -> None:
    """Download the national EPC archive, extract it, and merge core columns.

    Args:
        mode: accepted for backward compatibility with existing callers (the
            Prefect task, the CLI). No longer changes behaviour — the rebuilt
            EPC service serves one national archive rather than one per local
            authority, so there is nothing to select between.
    """
    del mode

    ensure_dirs()
    os.makedirs(EPC_MERGED_CSV.parent, exist_ok=True)
    headers = auth_headers()

    # Ask for the advertised size first so a truncated transfer is caught rather
    # than cached as a valid-looking archive.
    info = fetch_download_info(headers)
    expected = info.get("fileSize")
    if expected:
        print(f"Archive is {int(expected):,} bytes, last updated {info.get('lastUpdated')}.")

    print(download_full_load(headers, expected_bytes=expected))
    extract_all_zips()
    merge_core_columns()


if __name__ == "__main__":
    main()
