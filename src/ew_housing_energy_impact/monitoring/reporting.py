"""Report generation for lightweight monitoring outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ew_housing_energy_impact.monitoring.drift import build_drift_report
from ew_housing_energy_impact.monitoring.retraining import recommend_retraining
from ew_housing_energy_impact.paths import repo_root


def default_feature_dataset_path() -> Path:
    """Return the default modelling feature dataset path."""

    return (
        repo_root()
        / "data"
        / "processed"
        / "ew_epc_core_clean_sw"
        / "ew_epc_core_clean_sw_imputed.csv"
    )


def default_json_report_path() -> Path:
    """Return the default JSON monitoring report path."""

    return repo_root() / "reports" / "artifacts" / "drift_report.json"


def default_markdown_report_path(json_path: Path | None = None) -> Path:
    """Return the Markdown report path matching a JSON report path."""

    if json_path is None:
        return repo_root() / "reports" / "artifacts" / "drift_report.md"
    return json_path.with_suffix(".md")


def monitoring_plan(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """Return the files a monitoring run would use."""

    reference = Path(reference_path) if reference_path else default_feature_dataset_path()
    current = Path(current_path) if current_path else default_feature_dataset_path()
    output = Path(output_path) if output_path else default_json_report_path()
    return {
        "reference": str(reference),
        "current": str(current),
        "json_output": str(output),
        "markdown_output": str(default_markdown_report_path(output)),
    }


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_monitoring_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    model_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a complete monitoring report with drift and recommendation."""

    drift = build_drift_report(reference, current)
    recommendation = recommend_retraining(drift, model_metrics=model_metrics)
    return {
        "summary": {
            "checked_at": drift["checked_at"],
            "should_retrain": recommendation["should_retrain"],
            "severity": recommendation["severity"],
            "numeric_alert_count": drift["summary"]["numeric_alert_count"],
            "categorical_alert_count": drift["summary"]["categorical_alert_count"],
            "target_alert_count": drift["summary"]["target_alert_count"],
            "coverage_alert": drift["summary"]["coverage_alert"],
        },
        "drift": drift,
        "retraining_recommendation": recommendation,
        "limitations": [
            "Drift checks are threshold heuristics, not statistical significance tests.",
            "A retraining recommendation does not retrain or deploy a model automatically.",
            "The report compares CSV feature datasets and may miss upstream data issues.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a compact Markdown monitoring report."""

    summary = report["summary"]
    recommendation = report["retraining_recommendation"]
    drift = report["drift"]
    drifted_numeric = [item for item in drift["numeric_features"] if item["alert"]]
    drifted_categorical = [item for item in drift["categorical_features"] if item["alert"]]
    drifted_targets = [item for item in drift["target_drift"] if item["alert"]]

    lines = [
        "# EPC Drift Monitoring Report",
        "",
        "## Summary",
        f"- Checked at: `{summary['checked_at']}`",
        f"- Severity: `{summary['severity']}`",
        f"- Retraining recommended: `{summary['should_retrain']}`",
        f"- Numeric drift alerts: `{summary['numeric_alert_count']}`",
        f"- Categorical drift alerts: `{summary['categorical_alert_count']}`",
        f"- Target drift alerts: `{summary['target_alert_count']}`",
        f"- Coverage alert: `{summary['coverage_alert']}`",
        "",
        "## Drifted Numeric Features",
    ]
    if drifted_numeric:
        for item in drifted_numeric:
            lines.append(
                f"- `{item['column']}`: relative mean shift `{item['relative_mean_shift']}`, "
                f"missingness shift `{item['missingness_shift']}`"
            )
    else:
        lines.append("- None")

    lines += ["", "## Drifted Categorical Features"]
    if drifted_categorical:
        for item in drifted_categorical:
            lines.append(
                f"- `{item['column']}`: distribution shift `{item['distribution_shift']}`, "
                f"missingness shift `{item['missingness_shift']}`"
            )
    else:
        lines.append("- None")

    lines += ["", "## Target Drift"]
    if drifted_targets:
        for item in drifted_targets:
            metric = item.get("relative_mean_shift", item.get("distribution_shift"))
            lines.append(f"- `{item['column']}`: shift `{metric}`")
    else:
        lines.append("- None")

    coverage = drift["coverage"]
    lines += [
        "",
        "## Coverage Drift",
        f"- Reference rows: `{coverage['reference_certificate_count']}`",
        f"- Current rows: `{coverage['current_certificate_count']}`",
        f"- Certificate count change: `{coverage['certificate_count_change']}`",
        f"- Triggered checks: `{coverage['triggered_checks']}`",
        "",
        "## Retraining Recommendation",
        f"- Should retrain: `{recommendation['should_retrain']}`",
        f"- Severity: `{recommendation['severity']}`",
        f"- Triggered checks: `{recommendation['triggered_checks']}`",
    ]
    for reason in recommendation["reasons"]:
        lines.append(f"- Reason: {reason}")

    lines += ["", "## Limitations"]
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    return "\n".join(lines) + "\n"


def write_monitoring_reports(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write JSON and Markdown monitoring reports."""

    json_path = Path(output_path) if output_path else default_json_report_path()
    markdown_path = default_markdown_report_path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def run_monitoring(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    require_inputs: bool = True,
) -> dict[str, Any]:
    """Run monitoring from CSV files and write JSON/Markdown reports."""

    plan = monitoring_plan(reference_path, current_path, output_path)
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    reference = Path(plan["reference"])
    current = Path(plan["current"])
    missing = [str(path) for path in (reference, current) if not path.exists()]
    if missing:
        if require_inputs:
            raise FileNotFoundError(f"Monitoring input file(s) not found: {missing}")
        return {"status": "skipped_missing_inputs", "missing": missing, "plan": plan}

    report = build_monitoring_report(_read_csv(reference), _read_csv(current))
    json_path, markdown_path = write_monitoring_reports(report, plan["json_output"])

    try:
        from ew_housing_energy_impact.ml import tracking

        tracking.log_artifact_safe(json_path, artifact_path="monitoring")
        tracking.log_artifact_safe(markdown_path, artifact_path="monitoring")
    except RuntimeError:
        raise
    except Exception:
        pass

    return {
        "status": "completed",
        "json_output": str(json_path),
        "markdown_output": str(markdown_path),
        "should_retrain": report["summary"]["should_retrain"],
        "severity": report["summary"]["severity"],
    }
