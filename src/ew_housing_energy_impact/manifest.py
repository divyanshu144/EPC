"""Dataset manifest utilities."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def build_manifest(df: pd.DataFrame) -> dict:
    manifest = {
        "rows": len(df),
        "columns": list(df.columns),
        "year_min": int(df["YEAR"].min()) if "YEAR" in df.columns else None,
        "year_max": int(df["YEAR"].max()) if "YEAR" in df.columns else None,
        "local_authorities": int(df["LOCAL_AUTHORITY_LABEL"].nunique())
        if "LOCAL_AUTHORITY_LABEL" in df.columns
        else None,
        "policy_period_counts": df["POLICY_PERIOD"].value_counts().to_dict()
        if "POLICY_PERIOD" in df.columns
        else None,
    }
    return manifest


def write_manifest(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(df)
    out_path.write_text(json.dumps(manifest, indent=2))
