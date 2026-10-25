"""Create an auditable South West release without overwriting legacy outputs.

Run: PYTHONPATH=src:. python -m scripts.pipeline.build_release --release-id NAME
Missing measurements are retained. Nonmissing values outside declared cohort
bounds are excluded with mutually exclusive stage counts. No learned imputation
is performed here. Certificate uniqueness is checked across ALL selected South
West source rows, before eligibility filtering, using an on-disk unique index.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import ANALYSIS_AUTHORITIES, FUEL_MAP, RAW_EXTRACT_DIR, ROOT, TENURE_MAP
from ew_housing_energy_impact.registry import sha256_file
from scripts.pipeline.clean_epc import norm_age_band, roll_up_to_analysis_authority
from scripts.pipeline.full_load_schema import normalise_full_load_columns, resolve_usecols
from scripts.pipeline.policy import policy_period
from scripts.pipeline.schemas import CORE_COLS

NUMERIC = [
    "CURRENT_ENERGY_EFFICIENCY",
    "POTENTIAL_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    "ENERGY_CONSUMPTION_POTENTIAL",
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISSIONS_POTENTIAL",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "TOTAL_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "NUMBER_HEATED_ROOMS",
    "FLOOR_HEIGHT",
    "LOW_ENERGY_LIGHTING",
]
BOUNDS = {
    "CURRENT_ENERGY_EFFICIENCY": (1, 100),
    "POTENTIAL_ENERGY_EFFICIENCY": (1, 100),
    "TOTAL_FLOOR_AREA": (20, 1000),
    "CO2_EMISSIONS_CURRENT": (0, 30),
}
REQUIRED = {
    "LMK_KEY",
    "BUILDING_REFERENCE_NUMBER",
    "LOCAL_AUTHORITY",
    "LOCAL_AUTHORITY_LABEL",
    "LODGEMENT_DATE",
    "CURRENT_ENERGY_EFFICIENCY",
    "TOTAL_FLOOR_AREA",
    "PROPERTY_TYPE",
    "CONSTRUCTION_AGE_BAND",
    "MAIN_FUEL",
    "TENURE",
    "TRANSACTION_TYPE",
    "CO2_EMISSIONS_CURRENT",
    "POTENTIAL_ENERGY_EFFICIENCY",
}


def normalise_fuel(series: pd.Series) -> pd.Series:
    """Conservative lexical mapping; preserve unrecognised fuel in the raw column."""
    text = series.astype("string").str.strip().str.lower()
    mapping = {key.lower(): value for key, value in FUEL_MAP.items()}
    out = text.map(mapping)
    for prefix, category in [
        ("mains gas", "gas"),
        ("gas: mains gas", "gas"),
        ("electricity", "electricity"),
        ("oil", "oil"),
        ("gas: bulk lpg", "lpg"),
        ("lpg", "lpg"),
    ]:
        out = out.mask(out.isna() & text.str.startswith(prefix, na=False), category)
    return out.fillna("other").mask(text.isna() | text.eq(""), "Unknown").astype("string")


def prepare_chunk(raw: pd.DataFrame, counters: Counter) -> pd.DataFrame:
    """Apply stateless cohort rules; count each exclusion only at its first cause."""
    df = raw.copy()
    counters["south_west_input"] += len(df)

    def retain(mask: pd.Series, reason: str) -> None:
        nonlocal df
        mask = mask.fillna(False)
        counters[reason] += int((~mask).sum())
        df = df.loc[mask].copy()

    dates = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce", utc=True)
    df["YEAR"] = dates.dt.year.astype("float64")
    retain(df["YEAR"].notna(), "invalid_date")
    retain(df["YEAR"].between(2008, 2025), "outside_2008_2025")
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    retain(df["CURRENT_ENERGY_EFFICIENCY"].notna(), "missing_regression_target")
    for column, (lo, hi) in BOUNDS.items():
        retain(df[column].isna() | df[column].between(lo, hi), f"outside_bounds_{column}")
    gap = df["POTENTIAL_ENERGY_EFFICIENCY"] - df["CURRENT_ENERGY_EFFICIENCY"]
    retain(gap.isna() | gap.ge(0), "negative_efficiency_gap")

    df["LOCAL_AUTHORITY_RAW"] = df["LOCAL_AUTHORITY"]
    df["LOCAL_AUTHORITY_LABEL_RAW"] = df["LOCAL_AUTHORITY_LABEL"]
    df["ANALYSIS_AUTHORITY"] = roll_up_to_analysis_authority(df).astype("string")
    df["LOCAL_AUTHORITY"] = df["ANALYSIS_AUTHORITY"]
    df["LOCAL_AUTHORITY_LABEL"] = df["ANALYSIS_AUTHORITY"]
    df["AGE_BAND_STD"] = df["CONSTRUCTION_AGE_BAND"].apply(norm_age_band).astype("string")
    df["MAIN_FUEL_STD"] = normalise_fuel(df["MAIN_FUEL"])
    tenure_map = {k.lower(): v for k, v in TENURE_MAP.items()}
    df["TENURE_STD"] = (
        df["TENURE"].str.strip().str.lower().map(tenure_map).fillna("Unknown").astype("string")
    )
    df["LOG_FLOOR_AREA"] = np.log(df["TOTAL_FLOOR_AREA"])
    df["EFFICIENCY_GAP"] = df["POTENTIAL_ENERGY_EFFICIENCY"] - df["CURRENT_ENERGY_EFFICIENCY"]
    df["BELOW_C_INT"] = df["CURRENT_ENERGY_EFFICIENCY"].lt(69).astype("float64")
    keys = df["BUILDING_REFERENCE_NUMBER"].astype("string")
    df["PROPERTY_IDENTIFIED"] = (
        keys.notna() & keys.str.strip().ne("") & ~keys.str.startswith("NOUPRN:", na=False)
    ).astype(bool)
    df["IS_NEW_BUILD"] = (
        df["TRANSACTION_TYPE"].str.strip().str.lower().eq("new dwelling").fillna(False)
    )
    df["POLICY_PERIOD"] = df["YEAR"].map(policy_period).astype("string")
    for col in ["TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS"]:
        df[f"{col}_MISSING"] = df[col].isna()
    counters["retained"] += len(df)
    return df


def register_certificates(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Fail on duplicate or missing certificate IDs, including across chunks/files."""
    ids = df["LMK_KEY"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError("Missing certificate identifiers in South West sources")
    try:
        conn.executemany("INSERT INTO certificates VALUES (?)", ((x,) for x in ids))
        conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate certificate identifiers across South West sources") from exc


def _table(df: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    # Explicit strings prevent a wholly missing first batch becoming Arrow null type.
    for col in df:
        if (
            col not in NUMERIC
            and col
            not in {
                "YEAR",
                "LOG_FLOOR_AREA",
                "EFFICIENCY_GAP",
                "BELOW_C_INT",
                "NEW_BUILD_SHARE",
            }
            and not pd.api.types.is_bool_dtype(df[col])
        ):
            df[col] = df[col].astype("string")
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def build_release(source_root: Path, output: Path, chunk_size: int = 200_000) -> dict:
    sources = sorted(
        p
        for p in source_root.glob("domestic-*/certificates.csv")
        if re.fullmatch(r"domestic-\d{4}", p.parent.name)
    )
    if not sources:
        raise ValueError("No annual certificate source files found")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Release is not valid until manifest.json is complete.\n")
    code_files = [
        Path(__file__),
        ROOT / "config.py",
        ROOT / "scripts/pipeline/clean_epc.py",
        ROOT / "scripts/pipeline/full_load_schema.py",
        ROOT / "scripts/pipeline/policy.py",
        ROOT / "scripts/pipeline/schemas.py",
    ]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in code_files}
    ledger: list[dict] = []
    missing: Counter = Counter()
    years: Counter = Counter()
    authorities: Counter = Counter()
    policy: Counter = Counter()
    nb_counts: Counter = Counter()
    group_counts: Counter = Counter()
    writer = None
    schema = None
    row_count = 0
    unidentified = 0
    stage_path = output / "prepared.parquet"
    db_path = output / "audit.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE certificates (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE properties (id TEXT PRIMARY KEY)")
    try:
        for source in sources:
            initial_stat = source.stat()
            header = pd.read_csv(source, nrows=0).columns.tolist()
            usecols, renames = resolve_usecols(CORE_COLS, header)
            absent = REQUIRED - set(renames.values())
            if absent:
                raise ValueError(f"Required source columns absent in {source}: {sorted(absent)}")
            counts: Counter = Counter()
            source_missing: Counter = Counter()
            for raw in pd.read_csv(source, usecols=usecols, dtype="string", chunksize=chunk_size):
                counts["national_input"] += len(raw)
                raw = normalise_full_load_columns(raw.rename(columns=renames))
                raw = raw.reindex(columns=CORE_COLS)
                selected = roll_up_to_analysis_authority(raw).notna()
                counts["outside_south_west"] += int((~selected).sum())
                raw = raw.loc[selected].copy()
                if raw.empty:
                    continue
                register_certificates(conn, raw)
                raw["SOURCE_PERIOD"] = source.parent.name.removeprefix("domestic-")
                for col in NUMERIC:
                    source_missing[col] += int(
                        pd.to_numeric(raw[col], errors="coerce").isna().sum()
                    )
                df = prepare_chunk(raw, counts)
                if df.empty:
                    continue
                table = _table(df, schema)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(stage_path, schema, compression="zstd")
                writer.write_table(table)
                row_count += len(df)
                missing.update(df.isna().sum().to_dict())
                years.update({str(int(k)): int(v) for k, v in df.YEAR.value_counts().items()})
                authorities.update(df.ANALYSIS_AUTHORITY.value_counts().to_dict())
                policy.update(df.POLICY_PERIOD.value_counts().to_dict())
                unidentified += int((~df.PROPERTY_IDENTIFIED).sum())
                conn.executemany(
                    "INSERT OR IGNORE INTO properties VALUES (?)",
                    ((x,) for x in df.loc[df.PROPERTY_IDENTIFIED, "BUILDING_REFERENCE_NUMBER"]),
                )
                conn.commit()
                groups = df.groupby(["ANALYSIS_AUTHORITY", "YEAR"])["IS_NEW_BUILD"].agg(
                    ["sum", "size"]
                )
                nb_counts.update(groups["sum"].to_dict())
                group_counts.update(groups["size"].to_dict())
            digest = sha256_file(source)
            final_stat = source.stat()
            if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
                final_stat.st_size,
                final_stat.st_mtime_ns,
            ):
                raise RuntimeError(f"Source changed during release: {source}")
            entry = {
                "path": str(source),
                "sha256": digest,
                "bytes": final_stat.st_size,
                "source_columns": header,
                "counts": dict(counts),
                "south_west_numeric_missing_before_filters": dict(source_missing),
            }
            ledger.append(entry)
            print(json.dumps({"source": source.parent.name, **counts}), flush=True)
        if writer is not None:
            writer.close()
            writer = None
        if not row_count:
            raise ValueError("No eligible South West rows")
        # Group-derived covariate is for descriptive/policy analysis only, not prediction.
        shares = {key: nb_counts[key] / count for key, count in group_counts.items()}
        final_path = output / "certificates.parquet"
        final_writer = None
        try:
            for batch in pq.ParquetFile(stage_path).iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                df["NEW_BUILD_SHARE"] = [
                    shares[(a, y)] for a, y in zip(df.ANALYSIS_AUTHORITY, df.YEAR)
                ]
                table = _table(df)
                if final_writer is None:
                    final_writer = pq.ParquetWriter(final_path, table.schema, compression="zstd")
                final_writer.write_table(table)
        finally:
            if final_writer is not None:
                final_writer.close()
        total_counts = Counter()
        for entry in ledger:
            total_counts.update(entry["counts"])
        exclusion_total = sum(
            v
            for k, v in total_counts.items()
            if k
            not in {
                "national_input",
                "outside_south_west",
                "south_west_input",
                "retained",
            }
        )
        if total_counts["south_west_input"] != row_count + exclusion_total:
            raise AssertionError("Exclusion accounting does not reconcile")
        if (
            total_counts["national_input"]
            != total_counts["outside_south_west"] + total_counts["south_west_input"]
        ):
            raise AssertionError("Geographic accounting does not reconcile")
        manifest = {
            "status": "complete",
            "release_id": output.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "code_sha256": code_hashes,
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "scope": {
                "years": [2008, 2025],
                "bounds": BOUNDS,
                "missing_measurements": "retained; impute within training folds",
                "duplicate_check_scope": "all South West source rows before cohort filtering",
                "new_build_share_scope": "retained authority-year cohort; not a predictive feature",
            },
            "sources": ledger,
            "counts": dict(total_counts),
            "rows": row_count,
            "unique_source_certificates": conn.execute(
                "SELECT COUNT(*) FROM certificates"
            ).fetchone()[0],
            "unique_identified_properties": conn.execute(
                "SELECT COUNT(*) FROM properties"
            ).fetchone()[0],
            "unidentified_property_rows": unidentified,
            "year_counts": dict(sorted(years.items())),
            "authority_counts": dict(sorted(authorities.items())),
            "missing_authorities": sorted(set(ANALYSIS_AUTHORITIES) - set(authorities)),
            "policy_period_counts": dict(policy),
            "missing_counts": dict(missing),
            "columns": pq.read_schema(final_path).names,
            "dataset_sha256": sha256_file(final_path),
            "limitations": [
                "Certificates are not unique homes.",
                "UPRN linkage can be imperfect.",
                "Bounds define an analysis cohort, not universal EPC validity.",
                "Exclude unidentified properties from the primary property-held-out benchmark.",
            ],
        }
        if manifest["missing_authorities"]:
            raise ValueError(f"Missing analysis authorities: {manifest['missing_authorities']}")
        for p in code_files:
            if sha256_file(p) != code_hashes[str(p.relative_to(ROOT))]:
                raise RuntimeError("Release implementation changed during build")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "INCOMPLETE").unlink()
        stage_path.unlink()
        return manifest
    finally:
        if writer is not None:
            writer.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-root", type=Path, default=RAW_EXTRACT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.release_id):
        parser.error("release-id must be a simple directory name")
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    manifest = build_release(
        args.source_root, ROOT / "data/releases" / args.release_id, args.chunk_size
    )
    print(
        json.dumps(
            {
                "release": args.release_id,
                "rows": manifest["rows"],
                "sha256": manifest["dataset_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
