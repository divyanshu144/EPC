#!/usr/bin/env python3
"""
EPC Bulk Data Downloader and Merger
-----------------------------------
Downloads all domestic EPC zip files from the UK Open Data API,
extracts them, and merges only the key columns into one clean CSV.

Usage:
    1. Add EPC_EMAIL and EPC_API_KEY in your .env file (root directory).
    2. Optional: set YEARS = {2023, 2024} to limit downloads.
    3. Run from project root:
         python -m scripts.pipeline.epc_download_merge_fast
"""

import os
import glob
import base64
import zipfile
import concurrent.futures as futures

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import RAW_BULK_DIR, RAW_EXTRACT_DIR, EPC_MERGED_CSV
from scripts.pipeline.schemas import CORE_COLS, CORE_DTYPES

# -------------- CONFIG -------------- #

YEARS = None  # or e.g. {2023, 2024, 2025}

OUT_PATH = str(EPC_MERGED_CSV)

CHUNK = 2_000_000
MAX_WORKERS = 4
TIMEOUT = 120

DTYPES = CORE_DTYPES

# ------------------------------------ #


def ensure_dirs() -> None:
    """Make sure output folders exist."""
    for d in (RAW_BULK_DIR, RAW_EXTRACT_DIR, EPC_MERGED_CSV.parent):
        os.makedirs(d, exist_ok=True)


def auth_headers() -> dict:
    """Generate authentication header using .env credentials."""
    load_dotenv()
    email = os.getenv("EPC_EMAIL")
    api_key = os.getenv("EPC_API_KEY")

    if not email or not api_key:
        raise ValueError("Missing EPC_EMAIL or EPC_API_KEY in .env file")

    token = base64.b64encode(f"{email}:{api_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


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


def download_one_file(args) -> str:
    """Download one EPC zip file."""
    fname, headers = args
    url = f"https://epc.opendatacommunities.org/api/v1/files/{fname}"
    out_path = RAW_BULK_DIR / fname

    if out_path.exists():
        return f"✓ Skipped (exists): {fname}"

    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8 << 20):  # 8 MB chunks
                f.write(chunk)
    return f"⬇️  Downloaded: {fname}"


def download_all(headers: dict, candidates: list[str]) -> None:
    """Download all candidate files in parallel."""
    if not candidates:
        print("No candidate files to download.")
        return

    print(f"Downloading {len(candidates)} files...")
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for msg in tqdm(
            executor.map(download_one_file, [(c, headers) for c in candidates]),
            total=len(candidates),
        ):
            print(msg)


def extract_all_zips() -> None:
    """Unzip all downloaded EPC archives."""
    for zip_path in glob.glob(str(RAW_BULK_DIR / "*.zip")):
        subfolder = os.path.splitext(os.path.basename(zip_path))[0]
        dest = RAW_EXTRACT_DIR / subfolder
        cert_csv = dest / "certificates.csv"

        if cert_csv.exists():
            continue  # skip already extracted

        os.makedirs(dest, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            print(f"⚠️  Bad zip file detected, removing: {zip_path}")
            try:
                os.remove(zip_path)
            except OSError:
                pass
            continue
    print("✓ All archives extracted.")


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


def main() -> None:
    """Full pipeline: authenticate, list, download, extract, merge."""
    ensure_dirs()
    headers = auth_headers()

    files = list_available_files(headers)

    # Filter domestic zip files by year
    candidates: list[str] = []
    for name in files.keys():
        year = pick_year_from_name(name)
        if year and (YEARS is None or year in YEARS):
            candidates.append(name)
    candidates.sort()

    download_all(headers, candidates)
    extract_all_zips()
    merge_core_columns()


if __name__ == "__main__":
    main()
