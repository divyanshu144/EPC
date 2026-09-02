"""Load existing CSV/report artifacts into optional analytical serving tables."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ew_housing_energy_impact.db.engine import create_tables, get_engine, session_scope
from ew_housing_energy_impact.db.models import (
    ArtifactRegistry,
    AuthorityYearSummary,
    ModelMetric,
    PolicyPeriodSummary,
    ValidationRun,
)
from ew_housing_energy_impact.paths import repo_root

logger = logging.getLogger(__name__)


def default_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the default generated artifact paths used by the loader."""

    base = root or repo_root()
    return {
        "features": base
        / "data"
        / "processed"
        / "ew_epc_core_clean_sw"
        / "ew_epc_core_clean_sw_imputed.csv",
        "validation": base / "reports" / "artifacts" / "validation_report.json",
        "registry": base / "reports" / "artifacts" / "registry.jsonl",
        "regression_metrics": base / "reports" / "tables" / "regression_model_performance.csv",
        "classification_metrics": base
        / "reports"
        / "tables"
        / "classification_model_performance.csv",
    }


def _read_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def _none_if_missing(value: Any) -> Any:
    return None if pd.isna(value) else value


def _require_grouping_keys(df: pd.DataFrame, columns: list[str], table: str) -> None:
    """Fail loudly when a grouping key is missing, naming the column and row count.

    These columns identify the row; a gap in one is not a missing measurement but
    a broken upstream stage — the geography rollup or the year derivation. Two
    failure modes made that invisible:

    * ``groupby(..., dropna=False)`` keeps NaN groups, and ``str(NaN)`` yields the
      literal string ``"nan"``, so a missing authority was written to the serving
      database and surfaced in /authorities as though it were real.
    * The same NaN reached ``int(NaN)`` for YEAR and raised "cannot convert float
      NaN to integer", naming no column and no row.

    Raising here converts both into one message that says what is wrong and how
    much of the data it affects.
    """
    problems = []
    for column in columns:
        if column not in df.columns:
            continue
        n_missing = int(df[column].isna().sum())
        if n_missing:
            problems.append(f"{column} ({n_missing} rows)")

    if problems:
        raise ValueError(
            f"Cannot load {table}: grouping keys contain missing values — "
            f"{', '.join(problems)}. These identify the row, so a gap means an "
            f"upstream stage failed (geography rollup or year derivation). "
            f"Fix the source data rather than dropping the rows."
        )


def _optional_mean(grouped, column: str) -> pd.Series | None:
    source = grouped.obj
    if column not in source.columns:
        return None
    return grouped[column].mean()


def load_authority_year_summary(
    session: Session,
    data: pd.DataFrame | str | Path,
    replace: bool = True,
) -> int:
    """Load authority/year trend summaries from the modelling-ready dataset."""

    df = _read_dataframe(data)
    required = {
        "LOCAL_AUTHORITY",
        "YEAR",
        "CURRENT_ENERGY_EFFICIENCY",
        "CO2_EMISSIONS_CURRENT",
        "BELOW_C",
        "LMK_KEY",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot load authority-year summaries; missing columns: {sorted(missing)}"
        )

    _require_grouping_keys(df, ["LOCAL_AUTHORITY", "YEAR"], "authority_year_summary")

    grouped = df.groupby(["LOCAL_AUTHORITY", "YEAR"], dropna=False)
    summary = grouped.agg(
        mean_current_energy_efficiency=("CURRENT_ENERGY_EFFICIENCY", "mean"),
        mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
        below_c_rate=("BELOW_C", "mean"),
        certificate_count=("LMK_KEY", "count"),
    )

    gas_share = _optional_mean(grouped, "HAS_MAINS_GAS")
    if gas_share is not None:
        summary["gas_share"] = gas_share
        summary["non_gas_share"] = 1 - gas_share
    else:
        summary["gas_share"] = None
        summary["non_gas_share"] = None

    new_build_share = _optional_mean(grouped, "NEW_BUILD_SHARE")
    summary["new_build_share"] = new_build_share if new_build_share is not None else None
    summary = summary.reset_index()

    if replace:
        session.execute(delete(AuthorityYearSummary))

    rows = []
    for record in summary.to_dict(orient="records"):
        mean_epc = _none_if_missing(record["mean_current_energy_efficiency"])
        rows.append(
            AuthorityYearSummary(
                authority=str(record["LOCAL_AUTHORITY"]),
                year=int(record["YEAR"]),
                mean_epc_score=mean_epc,
                mean_current_energy_efficiency=mean_epc,
                mean_co2=_none_if_missing(record["mean_co2"]),
                below_c_rate=_none_if_missing(record["below_c_rate"]),
                certificate_count=int(record["certificate_count"]),
                gas_share=_none_if_missing(record.get("gas_share")),
                non_gas_share=_none_if_missing(record.get("non_gas_share")),
                new_build_share=_none_if_missing(record.get("new_build_share")),
            )
        )
    session.add_all(rows)
    return len(rows)


def load_policy_period_summary(
    session: Session,
    data: pd.DataFrame | str | Path,
    replace: bool = True,
) -> int:
    """Load policy-period summaries overall and by authority."""

    df = _read_dataframe(data)
    required = {
        "POLICY_PERIOD",
        "CURRENT_ENERGY_EFFICIENCY",
        "CO2_EMISSIONS_CURRENT",
        "BELOW_C",
        "LMK_KEY",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot load policy summaries; missing columns: {sorted(missing)}")

    # Same failure as authority_year_summary: str(NaN) becomes "nan" and would be
    # written as a real policy period or authority.
    _require_grouping_keys(df, ["POLICY_PERIOD", "LOCAL_AUTHORITY"], "policy_period_summary")

    if replace:
        session.execute(delete(PolicyPeriodSummary))

    rows: list[PolicyPeriodSummary] = []

    def build_rows(group_cols: list[str], authority: bool) -> None:
        grouped = df.groupby(group_cols, dropna=False).agg(
            mean_current_energy_efficiency=("CURRENT_ENERGY_EFFICIENCY", "mean"),
            mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
            below_c_rate=("BELOW_C", "mean"),
            certificate_count=("LMK_KEY", "count"),
        )
        for index, record in grouped.reset_index().iterrows():
            del index
            mean_epc = _none_if_missing(record["mean_current_energy_efficiency"])
            rows.append(
                PolicyPeriodSummary(
                    policy_period=str(record["POLICY_PERIOD"]),
                    authority=str(record["LOCAL_AUTHORITY"]) if authority else None,
                    mean_epc_score=mean_epc,
                    mean_current_energy_efficiency=mean_epc,
                    mean_co2=_none_if_missing(record["mean_co2"]),
                    below_c_rate=_none_if_missing(record["below_c_rate"]),
                    certificate_count=int(record["certificate_count"]),
                )
            )

    build_rows(["POLICY_PERIOD"], authority=False)
    if "LOCAL_AUTHORITY" in df.columns:
        build_rows(["POLICY_PERIOD", "LOCAL_AUTHORITY"], authority=True)

    session.add_all(rows)
    return len(rows)


def load_validation_run(
    session: Session,
    report_path: str | Path,
    run_id: str | None = None,
    replace: bool = True,
) -> int:
    """Load the JSON validation report into the validation_runs table."""

    path = Path(report_path)
    if not path.exists():
        logger.warning("Validation report not found; skipping: %s", path)
        return 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    geography = summary.get("geography", {})
    issues = payload.get("issues", [])
    resolved_run_id = run_id or f"validation-{int(path.stat().st_mtime)}"

    if replace:
        session.execute(delete(ValidationRun).where(ValidationRun.run_id == resolved_run_id))

    session.add(
        ValidationRun(
            run_id=resolved_run_id,
            status="ok" if payload.get("ok") else "failed",
            checked_at=datetime.fromtimestamp(path.stat().st_mtime),
            row_count=summary.get("rows"),
            year_min=summary.get("year_min"),
            year_max=summary.get("year_max"),
            authority_count=len(geography.get("present", [])) if geography else None,
            error_count=len(issues),
            warning_count=0,
            payload_json=payload,
        )
    )
    return 1


def load_artifact_registry(
    session: Session,
    registry_path: str | Path,
    replace: bool = True,
) -> int:
    """Load JSONL artifact registry entries into the analytical registry table."""

    path = Path(registry_path)
    if not path.exists():
        logger.warning("Artifact registry not found; skipping: %s", path)
        return 0

    if replace:
        session.execute(delete(ArtifactRegistry))

    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        artifact_path = payload.get("path", "")
        session.add(
            ArtifactRegistry(
                artifact_name=Path(artifact_path).name or payload.get("kind", "artifact"),
                artifact_type=payload.get("kind", "artifact"),
                path=artifact_path,
                sha256=payload.get("sha256"),
                created_at=datetime.fromisoformat(
                    payload.get("timestamp_utc", datetime.now(UTC).isoformat())
                ).replace(tzinfo=None),
                metadata_json=payload.get("metadata", {}),
            )
        )
        count += 1
    return count


def load_model_metrics(
    session: Session,
    regression_metrics_path: str | Path | None = None,
    classification_metrics_path: str | Path | None = None,
    replace: bool = True,
) -> int:
    """Load model performance CSVs into long-form model_metrics rows."""

    if replace:
        session.execute(delete(ModelMetric))

    count = 0

    def load_one(path_like: str | Path | None, task_type: str) -> None:
        nonlocal count
        if path_like is None:
            return
        path = Path(path_like)
        if not path.exists():
            logger.warning("%s metrics not found; skipping: %s", task_type, path)
            return
        df = pd.read_csv(path)
        if "model" not in df.columns:
            logger.warning("%s metrics missing model column; skipping: %s", task_type, path)
            return
        for _, row in df.iterrows():
            model_name = str(row["model"])
            for metric_name, value in row.items():
                if metric_name == "model" or pd.isna(value):
                    continue
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                split = "test" if metric_name.endswith("_test") else None
                session.add(
                    ModelMetric(
                        task_type=task_type,
                        model_name=model_name,
                        metric_name=str(metric_name),
                        metric_value=metric_value,
                        split=split,
                        artifact_path=str(path),
                        metadata_json={"source_file": str(path)},
                    )
                )
                count += 1

    load_one(regression_metrics_path, "regression")
    load_one(classification_metrics_path, "classification")
    return count


def refresh_analytics_database(
    *,
    engine: Engine | None = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int | list[str]]:
    """Refresh optional analytical tables from existing generated outputs."""

    paths = default_paths(root)
    planned = [
        f"authority_year_summary <- {paths['features']}",
        f"policy_period_summary <- {paths['features']}",
        f"validation_runs <- {paths['validation']}",
        f"artifact_registry <- {paths['registry']}",
        f"model_metrics <- {paths['regression_metrics']}",
        f"model_metrics <- {paths['classification_metrics']}",
    ]
    if dry_run:
        return {"planned": planned}

    target_engine = engine or get_engine()
    create_tables(target_engine)
    loaded: dict[str, int | list[str]] = {}
    with session_scope(target_engine) as session:
        if paths["features"].exists():
            loaded["authority_year_summary"] = load_authority_year_summary(
                session, paths["features"]
            )
            loaded["policy_period_summary"] = load_policy_period_summary(session, paths["features"])
        else:
            logger.warning("Feature dataset not found; skipping summaries: %s", paths["features"])
            loaded["authority_year_summary"] = 0
            loaded["policy_period_summary"] = 0
        loaded["validation_runs"] = load_validation_run(session, paths["validation"])
        loaded["artifact_registry"] = load_artifact_registry(session, paths["registry"])
        loaded["model_metrics"] = load_model_metrics(
            session,
            regression_metrics_path=paths["regression_metrics"],
            classification_metrics_path=paths["classification_metrics"],
        )
    return loaded
