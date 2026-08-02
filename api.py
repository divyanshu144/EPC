from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
)
ARTIFACTS_DIR = ROOT / "reports" / "artifacts"
TABLES_DIR = ROOT / "reports" / "tables"
VALIDATION_REPORT_PATH = ARTIFACTS_DIR / "validation_report.json"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.jsonl"
DRIFT_REPORT_PATH = ARTIFACTS_DIR / "drift_report.json"
REGRESSION_METRICS_PATH = TABLES_DIR / "regression_model_performance.csv"
CLASSIFICATION_METRICS_PATH = TABLES_DIR / "classification_model_performance.csv"

app = FastAPI(title="EPC South West API", version="0.2.0")
logger = logging.getLogger(__name__)


class TrendResponse(BaseModel):
    year: int
    mean_epc: float | None = None
    mean_co2: float | None = None


class AuthoritiesResponse(BaseModel):
    count: int
    source: str
    authorities: list[str]


class AuthoritySummaryResponse(BaseModel):
    authority: str
    source: str
    year_min: int | None = None
    year_max: int | None = None
    certificate_count: int
    latest_mean_epc: float | None = None
    latest_mean_co2: float | None = None
    latest_below_c_rate: float | None = None


class EmptyStateResponse(BaseModel):
    status: str = "unavailable"
    source: str | None = None
    message: str


def _display_path(path: str | Path | None) -> str | None:
    """Return a repo-relative path when possible to avoid exposing local paths."""

    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT.resolve()))
    except (OSError, ValueError):
        return candidate.name


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload["path"] = _display_path(payload.get("path"))
        rows.append(payload)
    return list(reversed(rows))[:limit]


def _database_status() -> dict[str, Any]:
    from ew_housing_energy_impact.db.engine import get_engine
    from ew_housing_energy_impact.settings import get_settings

    settings = get_settings()
    status: dict[str, Any] = {
        "enabled": settings.enable_database,
        "reachable": False,
        "fallback": "csv",
    }
    if not settings.enable_database:
        return status
    try:
        engine = get_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        status["reachable"] = True
        status["fallback"] = None
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _mlflow_configured() -> bool:
    from ew_housing_energy_impact.ml.tracking import is_mlflow_enabled

    return is_mlflow_enabled()


def _db_session_query(query_name: str, query_fn):
    from ew_housing_energy_impact.db.engine import get_engine, session_scope
    from ew_housing_energy_impact.settings import get_settings

    settings = get_settings()
    if not settings.enable_database:
        return None
    try:
        engine = get_engine(settings.database_url)
        with session_scope(engine) as session:
            return query_fn(session)
    except Exception as exc:
        logger.warning(
            "Database %s query failed; falling back to CSV/artifacts: %s", query_name, exc
        )
        return None


def _csv_exists() -> bool:
    return DATA_PATH.exists()


def _csv_columns() -> set[str]:
    if not DATA_PATH.exists():
        return set()
    return set(pd.read_csv(DATA_PATH, nrows=0).columns)


def _authority_column(columns: set[str]) -> str | None:
    for column in ("ANALYSIS_AUTHORITY", "LOCAL_AUTHORITY"):
        if column in columns:
            return column
    return None


def _read_feature_csv(usecols: list[str] | None = None) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Data not found. Run pipeline first.")
    available = _csv_columns()
    if usecols is not None:
        safe_usecols = [column for column in usecols if column in available]
        return pd.read_csv(DATA_PATH, usecols=safe_usecols)
    return pd.read_csv(DATA_PATH)


def _db_authorities() -> list[str] | None:
    from ew_housing_energy_impact.db.queries import get_authorities

    return _db_session_query("authorities", get_authorities)


def _csv_authorities() -> list[str]:
    columns = _csv_columns()
    authority_col = _authority_column(columns)
    if authority_col is None:
        return []
    df = _read_feature_csv([authority_col])
    return sorted(df[authority_col].dropna().astype(str).unique().tolist())


def _resolve_authority(authority_name: str, authorities: list[str]) -> str | None:
    if not authority_name or len(authority_name) > 255:
        raise HTTPException(status_code=400, detail="Invalid authority name.")
    lookup = {authority.lower(): authority for authority in authorities}
    return lookup.get(authority_name.strip().lower())


def _db_trends(authority: str | None = None) -> list[dict[str, Any]] | None:
    from ew_housing_energy_impact.db.queries import get_trends

    rows = _db_session_query("trends", lambda session: get_trends(session, authority=authority))
    if rows is None:
        return None
    if authority:
        return [
            {
                "year": row["year"],
                "mean_epc": row["mean_current_energy_efficiency"],
                "mean_co2": row["mean_co2"],
            }
            for row in rows
        ]
    by_year: dict[int, dict[str, float]] = {}
    for row in rows:
        year = int(row["year"])
        count = float(row["certificate_count"] or 0)
        bucket = by_year.setdefault(year, {"count": 0.0, "epc": 0.0, "co2": 0.0})
        bucket["count"] += count
        bucket["epc"] += float(row["mean_current_energy_efficiency"] or 0) * count
        bucket["co2"] += float(row["mean_co2"] or 0) * count
    return [
        {
            "year": year,
            "mean_epc": values["epc"] / values["count"] if values["count"] else None,
            "mean_co2": values["co2"] / values["count"] if values["count"] else None,
        }
        for year, values in sorted(by_year.items())
    ]


def _db_policy_summary(authority: str | None = None) -> list[dict[str, Any]] | None:
    from ew_housing_energy_impact.db.queries import get_policy_summary

    rows = _db_session_query(
        "policy-summary", lambda session: get_policy_summary(session, authority=authority)
    )
    if rows is None:
        return None
    return [
        {
            "POLICY_PERIOD": row["policy_period"],
            "mean_epc": row["mean_current_energy_efficiency"],
            "mean_co2": row["mean_co2"],
            "below_c_share": row["below_c_rate"],
        }
        for row in rows
    ]


def _db_authority_summary(authority: str) -> dict[str, Any] | None:
    from ew_housing_energy_impact.db.queries import get_trends

    rows = _db_session_query(
        "authority-summary", lambda session: get_trends(session, authority=authority)
    )
    if rows is None:
        return None
    if not rows:
        raise HTTPException(status_code=404, detail=f"Authority not found: {authority}")
    latest = max(rows, key=lambda row: row["year"])
    return {
        "authority": authority,
        "source": "database",
        "year_min": min(row["year"] for row in rows),
        "year_max": max(row["year"] for row in rows),
        "certificate_count": sum(int(row["certificate_count"] or 0) for row in rows),
        "latest_mean_epc": latest.get("mean_current_energy_efficiency"),
        "latest_mean_co2": latest.get("mean_co2"),
        "latest_below_c_rate": latest.get("below_c_rate"),
    }


def _csv_authority_summary(authority: str) -> dict[str, Any]:
    columns = _csv_columns()
    authority_col = _authority_column(columns)
    if authority_col is None:
        raise HTTPException(status_code=404, detail="No authority column is available.")
    authorities = _csv_authorities()
    resolved = _resolve_authority(authority, authorities)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Authority not found: {authority}")

    usecols = [
        authority_col,
        "YEAR",
        "CURRENT_ENERGY_EFFICIENCY",
        "CO2_EMISSIONS_CURRENT",
        "BELOW_C",
        "BELOW_C_INT",
    ]
    df = _read_feature_csv(usecols)
    subset = df[df[authority_col].astype(str) == resolved].copy()
    latest_year = (
        int(pd.to_numeric(subset["YEAR"], errors="coerce").max()) if "YEAR" in subset else None
    )
    latest = subset[subset["YEAR"] == latest_year] if latest_year is not None else subset
    below_col = "BELOW_C_INT" if "BELOW_C_INT" in latest.columns else "BELOW_C"
    return {
        "authority": resolved,
        "source": "csv",
        "year_min": int(pd.to_numeric(subset["YEAR"], errors="coerce").min())
        if "YEAR" in subset
        else None,
        "year_max": latest_year,
        "certificate_count": int(len(subset)),
        "latest_mean_epc": (
            float(latest["CURRENT_ENERGY_EFFICIENCY"].mean())
            if "CURRENT_ENERGY_EFFICIENCY" in latest.columns
            else None
        ),
        "latest_mean_co2": (
            float(latest["CO2_EMISSIONS_CURRENT"].mean())
            if "CO2_EMISSIONS_CURRENT" in latest.columns
            else None
        ),
        "latest_below_c_rate": (
            float(pd.to_numeric(latest[below_col], errors="coerce").mean())
            if below_col in latest.columns
            else None
        ),
    }


def _filter_csv_authority(
    df: pd.DataFrame,
    authority: str | None,
    authority_col: str | None,
) -> pd.DataFrame:
    """Filter a CSV-backed DataFrame to a known authority when requested."""

    if authority is None:
        return df
    if authority_col is None:
        raise HTTPException(status_code=404, detail="No authority column is available.")
    resolved = _resolve_authority(authority, _csv_authorities())
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Authority not found: {authority}")
    return df[df[authority_col].astype(str) == resolved].copy()


def _fallback_validation() -> dict[str, Any]:
    payload = _read_json(VALIDATION_REPORT_PATH)
    if payload is None:
        return {
            "status": "unavailable",
            "source": None,
            "message": "No validation report is available.",
        }
    return {"status": "available", "source": "artifact", "validation": payload}


def _fallback_model_metadata() -> dict[str, Any]:
    files = [
        ("regression", REGRESSION_METRICS_PATH),
        ("classification", CLASSIFICATION_METRICS_PATH),
    ]
    models = []
    for task_type, path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns:
            continue
        for row in df.to_dict(orient="records"):
            model_name = str(row.pop("model"))
            models.append(
                {
                    "task_type": task_type,
                    "model_name": model_name,
                    "metrics": row,
                    "artifact_path": _display_path(path),
                }
            )
    if not models:
        return {
            "status": "unavailable",
            "source": None,
            "message": "No model metadata is available.",
        }
    return {"status": "available", "source": "artifact", "models": models}


@app.get("/health")
def health():
    paths_checked = {
        "feature_dataset": {"available": DATA_PATH.exists(), "path": _display_path(DATA_PATH)},
        "validation_report": {
            "available": VALIDATION_REPORT_PATH.exists(),
            "path": _display_path(VALIDATION_REPORT_PATH),
        },
        "artifact_registry": {
            "available": REGISTRY_PATH.exists(),
            "path": _display_path(REGISTRY_PATH),
        },
        "drift_report": {
            "available": DRIFT_REPORT_PATH.exists(),
            "path": _display_path(DRIFT_REPORT_PATH),
        },
    }
    return {
        "status": "ok",
        "database": _database_status(),
        "csv_fallback_available": DATA_PATH.exists(),
        "paths_checked": paths_checked,
    }


@app.get("/metadata")
def metadata():
    database = _database_status()
    data_source_mode = "database" if database["enabled"] and database["reachable"] else "csv"
    return {
        "data_source_mode": data_source_mode,
        "database": database,
        "csv_data_available": DATA_PATH.exists(),
        "optional_systems": {
            "database": database,
            "mlflow_configured": _mlflow_configured(),
            "monitoring_report_available": DRIFT_REPORT_PATH.exists(),
        },
    }


@app.get("/authorities", response_model=AuthoritiesResponse)
def authorities():
    db_rows = _db_authorities()
    if db_rows is not None:
        return {"count": len(db_rows), "source": "database", "authorities": db_rows}
    csv_rows = _csv_authorities()
    return {"count": len(csv_rows), "source": "csv", "authorities": csv_rows}


@app.get("/authority/{authority_name}/summary", response_model=AuthoritySummaryResponse)
def authority_summary(authority_name: str):
    db_authorities = _db_authorities()
    if db_authorities is not None:
        resolved = _resolve_authority(authority_name, db_authorities)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"Authority not found: {authority_name}")
        summary = _db_authority_summary(resolved)
        if summary is not None:
            return summary
    return _csv_authority_summary(authority_name)


@app.get("/trends", response_model=list[TrendResponse])
def trends(authority: str | None = None):
    db_rows = _db_trends(authority=authority)
    if db_rows is not None:
        return db_rows
    authority_col = _authority_column(_csv_columns())
    usecols = ["YEAR", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT"]
    if authority_col:
        usecols.append(authority_col)
    df = _read_feature_csv(usecols)
    df = _filter_csv_authority(df, authority, authority_col)
    yearly = (
        df.groupby("YEAR", as_index=False)
        .agg(
            mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
            mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
        )
        .rename(columns={"YEAR": "year"})
        .sort_values("year")
    )
    return yearly.to_dict(orient="records")


@app.get("/policy-summary")
def policy_summary(authority: str | None = None):
    db_rows = _db_policy_summary(authority=authority)
    if db_rows is not None:
        return db_rows
    authority_col = _authority_column(_csv_columns())
    usecols = ["POLICY_PERIOD", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT", "BELOW_C"]
    if authority_col:
        usecols.append(authority_col)
    df = _read_feature_csv(usecols)
    df = _filter_csv_authority(df, authority, authority_col)
    summary = df.groupby("POLICY_PERIOD", as_index=False).agg(
        mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
        mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
        below_c_share=("BELOW_C", "mean"),
    )
    return summary.to_dict(orient="records")


@app.get("/validation/latest")
def latest_validation():
    from ew_housing_energy_impact.db.queries import get_latest_validation

    row = _db_session_query("latest-validation", get_latest_validation)
    if row is not None:
        return {"status": "available", "source": "database", "validation": row}
    return _fallback_validation()


@app.get("/artifacts")
def artifacts(limit: int = Query(default=100, ge=1, le=500)):
    from ew_housing_energy_impact.db.queries import get_artifacts

    rows = _db_session_query("artifacts", lambda session: get_artifacts(session, limit=limit))
    if rows is not None:
        for row in rows:
            row["path"] = _display_path(row.get("path"))
        return {"status": "available", "source": "database", "count": len(rows), "artifacts": rows}
    fallback = _read_jsonl(REGISTRY_PATH, limit=limit)
    if not fallback:
        return {"status": "unavailable", "source": None, "count": 0, "artifacts": []}
    return {
        "status": "available",
        "source": "artifact",
        "count": len(fallback),
        "artifacts": fallback,
    }


@app.get("/model/metadata")
def model_metadata(task_type: str | None = None):
    from ew_housing_energy_impact.db.queries import get_model_metrics

    rows = _db_session_query(
        "model-metadata", lambda session: get_model_metrics(session, task_type=task_type)
    )
    if rows is not None:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["task_type"], row["model_name"])
            model = grouped.setdefault(
                key,
                {
                    "task_type": row["task_type"],
                    "model_name": row["model_name"],
                    "metrics": {},
                    "artifact_paths": [],
                },
            )
            model["metrics"][row["metric_name"]] = row["metric_value"]
            if row.get("artifact_path"):
                model["artifact_paths"].append(_display_path(row["artifact_path"]))
        return {"status": "available", "source": "database", "models": list(grouped.values())}
    return _fallback_model_metadata()


@app.get("/drift/latest")
def latest_drift():
    payload = _read_json(DRIFT_REPORT_PATH)
    if payload is None:
        return {"status": "unavailable", "source": None, "message": "No drift report is available."}
    return {
        "status": "available",
        "source": "artifact",
        "summary": payload.get("summary", {}),
        "retraining_recommendation": payload.get("retraining_recommendation", {}),
        "drift": payload.get("drift", {}),
    }


@app.get("/retraining/status")
def retraining_status():
    payload = _read_json(DRIFT_REPORT_PATH)
    if payload is None:
        return {
            "status": "unavailable",
            "source": None,
            "message": "No retraining recommendation is available.",
            "automatic_retraining": False,
        }
    return {
        "status": "available",
        "source": "artifact",
        "automatic_retraining": False,
        "message": "This is a monitoring recommendation only; no retraining is triggered.",
        "recommendation": payload.get("retraining_recommendation", {}),
    }
