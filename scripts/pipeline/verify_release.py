"""Independently inspect the released Parquet against its manifest and cohort rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import ANALYSIS_AUTHORITIES
from ew_housing_energy_impact.registry import sha256_file


def verify_release(release: Path, expected_years=range(2008, 2026)) -> dict:
    manifest = json.loads((release / "manifest.json").read_text())
    path = release / "certificates.parquet"
    checks = {
        "complete_marker": not (release / "INCOMPLETE").exists(),
        "manifest_complete": manifest.get("status") == "complete",
        "dataset_hash": sha256_file(path) == manifest["dataset_sha256"],
    }
    seen = set()
    properties = set()
    counts = Counter()
    years = Counter()
    authorities = Counter()
    missing = Counter()
    invalid = Counter()
    unidentified = 0
    policy_missing = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
        df = batch.to_pandas()
        ids = df.LMK_KEY
        counts["rows"] += len(df)
        counts["blank_ids"] += int(ids.isna().sum() + ids.eq("").sum())
        counts["duplicate_ids"] += len(ids) - ids.nunique() + len(seen.intersection(ids))
        seen.update(ids)
        keys = df.BUILDING_REFERENCE_NUMBER.astype("string")
        identified = (
            keys.notna() & keys.str.strip().ne("") & ~keys.str.startswith("NOUPRN:", na=False)
        )
        properties.update(keys[identified])
        unidentified += int((~identified).sum())
        invalid["property_identified_flag"] += int(identified.ne(df.PROPERTY_IDENTIFIED).sum())
        years.update({str(int(y)): int(n) for y, n in df.YEAR.value_counts().items()})
        authorities.update(df.ANALYSIS_AUTHORITY.value_counts().to_dict())
        missing.update(df.isna().sum().to_dict())
        policy_missing += int(df.POLICY_PERIOD.isna().sum() + df.POLICY_PERIOD.eq("Other").sum())
        for col, (low, high) in manifest["scope"]["bounds"].items():
            value = pd.to_numeric(df[col], errors="coerce")
            invalid[col] += int((value.notna() & ~value.between(low, high)).sum())
        invalid["missing_target"] += int(df.CURRENT_ENERGY_EFFICIENCY.isna().sum())
        invalid["below_c_label"] += int(
            df.BELOW_C_INT.ne(df.CURRENT_ENERGY_EFFICIENCY.lt(69)).sum()
        )
        invalid["negative_gap"] += int(df.EFFICIENCY_GAP.lt(0).sum())
    checks.update(
        {
            "row_count": counts["rows"] == manifest["rows"],
            "certificate_uniqueness": counts["duplicate_ids"] == 0 and counts["blank_ids"] == 0,
            "year_coverage": set(years) == {str(y) for y in expected_years},
            "year_counts": dict(years) == manifest["year_counts"],
            "authority_coverage": set(authorities) == set(ANALYSIS_AUTHORITIES),
            "authority_counts": dict(authorities) == manifest["authority_counts"],
            "numeric_bounds_and_labels": not any(invalid.values()),
            "policy_period_present": policy_missing == 0,
            "missingness": all(
                manifest["missing_counts"].get(k, 0) == v for k, v in missing.items()
            ),
            "identified_property_count": len(properties)
            == manifest["unique_identified_properties"],
            "unidentified_rows": unidentified == manifest["unidentified_property_rows"],
        }
    )
    stages = manifest["counts"]
    excluded = sum(
        v
        for k, v in stages.items()
        if k
        not in {
            "national_input",
            "south_west_input",
            "outside_south_west",
            "retained",
        }
    )
    checks["stage_reconciliation"] = (
        stages["national_input"] == stages["south_west_input"] + stages["outside_south_west"]
        and stages["south_west_input"] == manifest["rows"] + excluded
    )
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "rows": counts["rows"],
        "unique_certificates": len(seen),
        "unique_identified_properties": len(properties),
        "year_counts": dict(sorted(years.items())),
        "unidentified_rows": unidentified,
        "invalid_counts": dict(invalid),
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": sha256_file(release / "manifest.json"),
        "source_hashes_independently_recomputed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_release(args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
