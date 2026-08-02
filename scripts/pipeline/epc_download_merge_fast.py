#!/usr/bin/env python3
"""
EPC Bulk Data Downloader and Merger
-----------------------------------
Two download modes are supported:

  year  (default) — Download annual bulk zips covering all of England & Wales
                    then filter to SW LAs during the clean step.
                    ~6.7 GB compressed, ~30–50 GB unzipped. Suitable when you
                    need the full national dataset or want all years at once.

  la              — Download only the 14 South West LA-specific zips directly.
                    ~200–300 MB compressed. Recommended for development and
                    when disk space is limited. ~95% less data than year mode.

Set DOWNLOAD_MODE below, or call main(mode='la') / main(mode='year').

Usage:
    1. Add EPC_EMAIL and EPC_API_KEY in your .env file (root directory).
    2. Set DOWNLOAD_MODE and optionally YEARS (year mode only).
    3. Run from project root:
         python -m scripts.pipeline.epc_download_merge_fast
"""

import base64
import concurrent.futures as futures
import glob
import os
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    ANALYSIS_AUTHORITIES,
    CODE_TO_ANALYSIS_AUTHORITY,
    EPC_MERGED_CSV,
    RAW_BULK_DIR,
    RAW_EXTRACT_DIR,
    SOUTH_WEST_EPC_AUTHORITIES,
    SOUTH_WEST_EPC_CODES,
)
from scripts.pipeline.schemas import CORE_COLS, CORE_DTYPES

# -------------- CONFIG -------------- #

# 'la'   → download only SW local authority zips (~200 MB, recommended)
# 'year' → download full annual E&W bulk zips (~6.7 GB)
DOWNLOAD_MODE: str = "la"

YEARS = None  # only used in 'year' mode; None = all years, or e.g. {2023, 2024}

OUT_PATH = str(EPC_MERGED_CSV)

CHUNK = 2_000_000
MAX_WORKERS = 4
TIMEOUT = 120

DTYPES = CORE_DTYPES

# LA-mode download selection uses the single South West geography mapping in
# config.py (SOUTH_WEST_EPC_AUTHORITIES). That mapping holds lower-tier/unitary
# GSS codes ONLY (E06/E07) — never county codes (E10), which have no EPC bulk
# file — and records the analysis authority each one rolls up to. The same
# mapping drives the clean-step rollup, so download and cleaning can never drift.

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


def _is_valid_zip(path) -> bool:
    """Cheap integrity check: can the central directory be read?

    A truncated/interrupted download has no valid end-of-central-directory
    record, so ZipFile() raises BadZipFile. This validates the archive is
    complete without paying the full-CRC cost of testzip() on large files.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def download_one_file(args) -> str:
    """Download one EPC zip file atomically.

    Writes to a temporary ``.part`` file and renames to the final path only on
    success, so an interrupted download never leaves a truncated file that later
    looks like a complete cache. An existing cached zip is validated before
    skipping; if it is corrupt/partial it is removed and re-downloaded.
    """
    fname, headers = args
    url = f"https://epc.opendatacommunities.org/api/v1/files/{fname}"
    out_path = RAW_BULK_DIR / fname

    if out_path.exists():
        if _is_valid_zip(out_path):
            return f"✓ Skipped (exists): {fname}"
        out_path.unlink()  # corrupt/partial cache — re-download
        re_dl = True
    else:
        re_dl = False

    tmp_path = out_path.with_name(out_path.name + ".part")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(8 << 20):  # 8 MB chunks
                    f.write(chunk)
        os.replace(tmp_path, out_path)  # atomic on the same filesystem
    finally:
        if tmp_path.exists():
            tmp_path.unlink()  # clean up on any failure

    return f"⬇️  Re-downloaded (was corrupt): {fname}" if re_dl else f"⬇️  Downloaded: {fname}"


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


def pick_la_files(files: dict) -> list[str]:
    """Select domestic LA-specific zip files for South West authorities.

    Matches filenames like 'domestic-E06000023-Bristol-City-of.zip' against
    the known South West LA code list. This pulls only the ~14 relevant files
    (~200 MB total) instead of the full 18-year national bulk (~6.7 GB).
    """
    candidates = []
    for name in files.keys():
        if not name.startswith("domestic-") or not name.endswith(".zip"):
            continue
        parts = name.split("-")
        if len(parts) < 2:
            continue
        la_code = parts[1]
        if la_code in SOUTH_WEST_EPC_CODES:
            candidates.append(name)
    return sorted(candidates)


def _missing_analysis_authorities(matched: set[str]) -> list[str]:
    """Analysis authorities with NO matched download file (panel-fatal gaps)."""
    covered = {CODE_TO_ANALYSIS_AUTHORITY[c] for c in matched if c in CODE_TO_ANALYSIS_AUTHORITY}
    return sorted(set(ANALYSIS_AUTHORITIES) - covered)


def _matched_codes(candidates: list[str]) -> set[str]:
    """Return the set of SW LA codes that appear in the selected filenames."""
    matched = set()
    for name in candidates:
        parts = name.split("-")
        if len(parts) >= 2:
            matched.add(parts[1])
    return matched


def main(mode: str = DOWNLOAD_MODE, strict: bool = True) -> None:
    """Full pipeline: authenticate, list, download, extract, merge.

    Args:
        mode: 'la' (default) to download only SW LA files (~200 MB), or
              'year' to download full annual E&W bulk files (~6.7 GB).
        strict: in 'la' mode, raise if any of the 14 analysis authorities has no
            matching download file (a panel-fatal gap). Defaults to True for
            dissertation reproducibility — a warning is not enough, since a
            missing Devon/Gloucestershire would silently invalidate the panel.
            Individual redundant reorganisation codes that don't match are only
            reported, not fatal.
    """
    ensure_dirs()
    headers = auth_headers()
    files = list_available_files(headers)

    if mode == "la":
        candidates = pick_la_files(files)
        matched = _matched_codes(candidates)
        unmatched = {
            c: meta["download_name"]
            for c, meta in SOUTH_WEST_EPC_AUTHORITIES.items()
            if c not in matched
        }
        missing_authorities = _missing_analysis_authorities(matched)

        print(
            f"LA mode: requested {len(SOUTH_WEST_EPC_AUTHORITIES)} SW download codes; "
            f"{len(matched)} matched a file; {len(candidates)} zips selected."
        )
        print(
            f"Analysis-authority coverage: "
            f"{len(ANALYSIS_AUTHORITIES) - len(missing_authorities)} of "
            f"{len(ANALYSIS_AUTHORITIES)} covered."
        )
        if unmatched:
            print(
                "ℹ Requested codes with no file in the API listing "
                "(redundant reorganisation vintages are expected here):"
            )
            for code, label in sorted(unmatched.items()):
                covers = CODE_TO_ANALYSIS_AUTHORITY.get(code, "?")
                print(f"    - {code}  {label}  -> {covers}")

        if missing_authorities:
            msg = (
                "South West analysis authorities with NO matching EPC file "
                f"(panel would be incomplete): {missing_authorities}. "
                "Check the GSS codes in config.SOUTH_WEST_EPC_AUTHORITIES against "
                "the live EPC file listing."
            )
            if strict:
                raise RuntimeError(msg)
            print(f"⚠ {msg}")
    else:
        # Year mode: download full annual bulk files
        candidates = []
        for name in files.keys():
            year = pick_year_from_name(name)
            if year and (YEARS is None or year in YEARS):
                candidates.append(name)
        candidates.sort()
        print(f"Year mode: found {len(candidates)} annual bulk zips.")

    download_all(headers, candidates)
    extract_all_zips()
    merge_core_columns()


if __name__ == "__main__":
    main()
