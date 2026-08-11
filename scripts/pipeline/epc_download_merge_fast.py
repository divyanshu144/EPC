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
import requests

from config import EPC_MERGED_CSV, RAW_EXTRACT_DIR
from ew_housing_energy_impact.ingestion.epc_source import (  # noqa: F401
    _is_valid_zip,
    auth_headers,
    download_all,
    download_one_file,
    ensure_dirs,
    extract_all_zips,
)
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
    resp = requests.get(
        "https://epc.opendatacommunities.org/api/v1/files",
        headers={**headers, "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("files", {})


def pick_year_from_name(name: str) -> int | None:
    """Extract year from filename like domestic-2024.zip."""
    if name.startswith("domestic-") and name.endswith(".zip"):
        try:
            return int(name.split("-")[1].split(".")[0])
        except Exception:
            return None
    return None


def merge_core_columns() -> None:
    """Merge certificates.csv from all years into one master CSV."""
    csv_files = sorted(glob.glob(str(RAW_EXTRACT_DIR / "*" / "certificates.csv")))
    if not csv_files:
        print("No certificates.csv files found.")
        return

    # Fresh write every run
    if EPC_MERGED_CSV.exists():
        EPC_MERGED_CSV.unlink()

    first = True
    for csv_file in csv_files:
        print(f"Merging {csv_file} ...")
        for chunk in pd.read_csv(
            csv_file,
            usecols=CORE_COLS,
            dtype=DTYPES,
            chunksize=CHUNK,
            low_memory=True,
        ):
            mode, header = ("w", True) if first else ("a", False)
            chunk.to_csv(OUT_PATH, index=False, mode=mode, header=header)
            first = False

    print(f"✓ Merged output saved to: {OUT_PATH}")


def main(mode: str = "la") -> None:
    """Full pipeline: authenticate, list, download, extract, merge.

    Args:
        mode: accepted for backward compatibility with existing callers (the
            Prefect task, the CLI). No longer changes behaviour — the rebuilt
            EPC service serves one national archive per period rather than
            one archive per local authority, so there is no per-authority
            selection to branch on any more.
    """
    ensure_dirs()
    os.makedirs(EPC_MERGED_CSV.parent, exist_ok=True)
    headers = auth_headers()
    files = list_available_files(headers)

    candidates = []
    for name in files.keys():
        year = pick_year_from_name(name)
        if year and (YEARS is None or year in YEARS):
            candidates.append(name)
    candidates.sort()
    print(f"Found {len(candidates)} annual bulk zips.")

    download_all(headers, candidates)
    extract_all_zips()
    merge_core_columns()


if __name__ == "__main__":
    main()
