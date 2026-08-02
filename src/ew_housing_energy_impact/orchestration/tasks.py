"""Prefect task wrappers around the existing EPC pipeline entry points.

The orchestration layer intentionally delegates to the established script/CLI
code paths. That keeps this phase focused on workflow coordination rather than
rewriting business logic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from ew_housing_energy_impact.logging_utils import setup_logger
from ew_housing_energy_impact.manifest import write_manifest as write_manifest_file
from ew_housing_energy_impact.monitoring.reporting import run_monitoring as run_monitoring_report
from ew_housing_energy_impact.paths import repo_root
from ew_housing_energy_impact.registry import register_artifact
from ew_housing_energy_impact.validation import validate_imputed_csv, write_validation_report

try:  # pragma: no cover - exercised when Prefect is installed.
    from prefect import get_run_logger, task
except ModuleNotFoundError:  # pragma: no cover - local test fallback before dependency install.
    get_run_logger = None
    F = TypeVar("F", bound=Callable[..., Any])

    def task(*task_args: Any, **task_kwargs: Any) -> Callable[[F], F] | F:
        """Fallback no-op decorator used only when Prefect is not installed."""

        if task_args and callable(task_args[0]) and not task_kwargs:
            return task_args[0]

        def decorator(fn: F) -> F:
            return fn

        return decorator


PIPELINE_TASK_NAMES: tuple[str, ...] = (
    "download_or_merge_epc",
    "clean_south_west",
    "build_features",
    "validate_clean_dataset",
    "generate_eda",
    "run_policy_models",
    "train_ml_models",
    "generate_quality_report",
    "run_monitoring",
    "write_manifest",
    "register_artifacts",
)


def _logger():
    if get_run_logger is not None:
        try:
            return get_run_logger()
        except RuntimeError:
            pass
    return setup_logger("ewhei.orchestration")


def _imputed_data_path() -> Path:
    return (
        repo_root()
        / "data"
        / "processed"
        / "ew_epc_core_clean_sw"
        / "ew_epc_core_clean_sw_imputed.csv"
    )


def _artifact_path(filename: str) -> Path:
    return repo_root() / "reports" / "artifacts" / filename


@task(name="download_or_merge_epc")
def download_or_merge_epc(
    mode: str = "la", skip_download: bool = False, dry_run: bool = False
) -> str:
    """Download/extract/merge EPC certificates using the existing ingestion script."""

    log = _logger()
    if dry_run or skip_download:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping EPC download/merge (%s).", reason)
        return "skipped"

    log.info("Starting EPC download/merge in %s mode.", mode)
    from scripts.pipeline import epc_download_merge_fast

    epc_download_merge_fast.main(mode=mode)
    log.info("Finished EPC download/merge.")
    return "completed"


@task(name="clean_south_west")
def clean_south_west(dry_run: bool = False) -> str:
    """Clean merged EPC data and roll records into South West analysis authorities."""

    log = _logger()
    if dry_run:
        log.info("Dry run: would clean and roll up South West EPC data.")
        return "planned"

    log.info("Starting South West cleaning step.")
    from scripts.pipeline import clean_epc

    clean_epc.main()
    log.info("Finished South West cleaning step.")
    return "completed"


@task(name="build_features")
def build_features(dry_run: bool = False) -> str:
    """Build the modelling-ready feature dataset from the cleaned EPC data."""

    log = _logger()
    if dry_run:
        log.info("Dry run: would build modelling features.")
        return "planned"

    log.info("Starting feature build step.")
    from scripts.pipeline import build_features as build_features_module

    build_features_module.main()
    log.info("Finished feature build step.")
    return "completed"


@task(name="validate_clean_dataset")
def validate_clean_dataset(dry_run: bool = False) -> bool:
    """Validate the imputed modelling dataset and stop the flow on validation failure."""

    log = _logger()
    if dry_run:
        log.info("Dry run: would validate modelling dataset.")
        return True

    data_path = _imputed_data_path()
    report_path = _artifact_path("validation_report.json")
    registry_path = _artifact_path("registry.jsonl")

    log.info("Validating dataset at %s.", data_path)
    result = validate_imputed_csv(data_path)
    write_validation_report(result, report_path)
    register_artifact(registry_path, "validation_report", report_path, {"ok": result.ok})

    if not result.ok:
        log.error("Validation failed with issues: %s", result.issues)
        raise RuntimeError(f"Validation failed: {result.issues}")

    log.info("Validation passed. Report written to %s.", report_path)
    return True


@task(name="generate_eda")
def generate_eda(skip_eda: bool = False, dry_run: bool = False) -> str:
    """Generate EDA figures and summary tables from the modelling dataset."""

    log = _logger()
    if dry_run or skip_eda:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping EDA generation (%s).", reason)
        return "skipped" if skip_eda else "planned"

    log.info("Starting EDA generation.")
    from scripts.pipeline import report_eda

    report_eda.main()
    log.info("Finished EDA generation.")
    return "completed"


@task(name="run_policy_models")
def run_policy_models(skip_policy: bool = False, dry_run: bool = False) -> str:
    """Run fixed-effects policy-period models and clustering outputs."""

    log = _logger()
    if dry_run or skip_policy:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping policy models (%s).", reason)
        return "skipped" if skip_policy else "planned"

    log.info("Starting fixed-effects policy modelling.")
    from scripts.pipeline import run_did

    run_did.main()
    log.info("Finished fixed-effects policy modelling.")
    return "completed"


@task(name="train_ml_models")
def train_ml_models(skip_training: bool = False, dry_run: bool = False) -> str:
    """Train predictive regression/classification models using the existing ML script."""

    log = _logger()
    if dry_run or skip_training:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping ML training (%s).", reason)
        return "skipped" if skip_training else "planned"

    log.info("Starting ML training.")
    from scripts.pipeline import train_models

    train_models.main()
    log.info("Finished ML training.")
    return "completed"


@task(name="generate_quality_report")
def generate_quality_report(skip_quality: bool = False, dry_run: bool = False) -> str:
    """Generate the HTML data quality report from the modelling dataset."""

    log = _logger()
    if dry_run or skip_quality:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping quality report (%s).", reason)
        return "skipped" if skip_quality else "planned"

    log.info("Starting quality report generation.")
    from scripts.pipeline import quality_report

    quality_report.main()
    log.info("Finished quality report generation.")
    return "completed"


@task(name="run_monitoring")
def run_monitoring(skip_monitoring: bool = False, dry_run: bool = False) -> str:
    """Run lightweight drift monitoring and write retraining recommendation reports."""

    log = _logger()
    if dry_run or skip_monitoring:
        reason = "dry run" if dry_run else "skip flag"
        log.info("Skipping monitoring (%s).", reason)
        return "skipped" if skip_monitoring else "planned"

    log.info("Starting lightweight drift monitoring.")
    result = run_monitoring_report(require_inputs=False)
    log.info("Monitoring result: %s", result)
    return str(result.get("status", "completed"))


@task(name="write_manifest")
def write_manifest(dry_run: bool = False) -> str:
    """Write a dataset manifest for the imputed modelling dataset."""

    log = _logger()
    if dry_run:
        log.info("Dry run: would write dataset manifest.")
        return "planned"

    data_path = _imputed_data_path()
    manifest_path = _artifact_path("data_manifest.json")
    registry_path = _artifact_path("registry.jsonl")

    log.info("Writing manifest for %s.", data_path)
    df = pd.read_csv(data_path)
    write_manifest_file(df, manifest_path)
    register_artifact(registry_path, "manifest", manifest_path, {"type": "data_manifest"})
    log.info("Manifest written to %s.", manifest_path)
    return "completed"


@task(name="register_artifacts")
def register_artifacts(dry_run: bool = False) -> str:
    """Register a lightweight orchestration marker in the artifact registry."""

    log = _logger()
    if dry_run:
        log.info("Dry run: would register orchestration marker artifact.")
        return "planned"

    marker_path = _artifact_path("prefect_pipeline_complete.txt")
    registry_path = _artifact_path("registry.jsonl")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("Prefect EPC pipeline completed.\n", encoding="utf-8")
    register_artifact(
        registry_path,
        "orchestration_marker",
        marker_path,
        {"orchestrator": "prefect"},
    )
    log.info("Registered orchestration marker at %s.", marker_path)
    return "completed"
