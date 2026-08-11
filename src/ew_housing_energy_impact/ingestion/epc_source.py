"""Download, validate and extract EPC bulk source archives.

The EPC service moved to GOV.UK in 2025: the old opendatacommunities host
301-redirects to an HTML landing page, so a request there returns 200 with a
body that is not JSON. Auth changed from Basic to Bearer at the same time, and
the bulk endpoint now serves one national archive split by period rather than
one archive per local authority. See spec section 0.
"""

from __future__ import annotations

import concurrent.futures as futures
import glob
import os
import re
import zipfile
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm

from config import (
    ANALYSIS_AUTHORITIES,
    CODE_TO_ANALYSIS_AUTHORITY,
    RAW_BULK_DIR,
    RAW_EXTRACT_DIR,
)

API_BASE = "https://api.get-energy-performance-data.communities.gov.uk"
DOMESTIC_CSV_ENDPOINT = f"{API_BASE}/api/files/domestic/csv"
DOMESTIC_CSV_INFO_ENDPOINT = f"{API_BASE}/api/files/domestic/csv/info"

TIMEOUT = 120
MAX_WORKERS = 4

# 'domestic-2025-09' or 'domestic-2024', with or without a .zip suffix.
_PERIOD_RE = re.compile(r"^domestic-(\d{4}(?:-\d{2})?)(?:\.zip)?$")


def auth_headers() -> dict:
    """Bearer auth for the rebuilt EPC service.

    The token comes from the account page on the service website; there is no
    programmatic way to mint one. EPC_EMAIL/EPC_API_KEY are obsolete.
    """
    from ew_housing_energy_impact.settings import get_settings

    token = get_settings().epc_bearer_token
    if not token:
        raise ValueError("Missing EPC_BEARER_TOKEN — get one from your EPC account page")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def period_from_name(name: str) -> str | None:
    """Extract the period from an archive or extracted-folder name.

    'domestic-2025-09' -> '2025-09'; 'domestic-2024.zip' -> '2024'.
    Returns None for anything that is not a domestic archive.
    """
    match = _PERIOD_RE.match(name)
    return match.group(1) if match else None


def ensure_dirs() -> None:
    """Make sure output folders exist."""
    for d in (RAW_BULK_DIR, RAW_EXTRACT_DIR):
        os.makedirs(d, exist_ok=True)


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
        except zipfile.BadZipFile as exc:
            # The archive passed download validation but will not extract. Delete
            # it so the next run re-downloads, and fail now rather than leaving a
            # hole to be discovered after the load.
            os.remove(zip_path)
            raise RuntimeError(
                f"Corrupt archive removed, re-run to re-download: {zip_path}"
            ) from exc
    print("✓ All archives extracted.")


def _missing_analysis_authorities(matched: set[str]) -> list[str]:
    """Analysis authorities with NO matched download file (panel-fatal gaps)."""
    covered = {CODE_TO_ANALYSIS_AUTHORITY[c] for c in matched if c in CODE_TO_ANALYSIS_AUTHORITY}
    return sorted(set(ANALYSIS_AUTHORITIES) - covered)


SOURCE_KINDS = ("certificates", "recommendations")


class ExtractedSource(NamedTuple):
    """One extracted CSV belonging to one source period."""

    period: str
    kind: str
    path: Path


def iter_extracted_sources(root: Path | None = None) -> list[ExtractedSource]:
    """List every certificates/recommendations CSV under the extract root.

    Folders holding only metadata (columns.csv, schema.json, LICENCE.txt) yield
    nothing — that is the state of the 2008-2024 folders locally, where the CSVs
    were deleted to reclaim disk.
    """
    base = Path(root) if root is not None else Path(RAW_EXTRACT_DIR)
    sources: list[ExtractedSource] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        period = period_from_name(folder.name)
        if period is None:
            continue
        for kind in SOURCE_KINDS:
            csv_path = folder / f"{kind}.csv"
            if csv_path.exists():
                sources.append(ExtractedSource(period, kind, csv_path))
    return sources
