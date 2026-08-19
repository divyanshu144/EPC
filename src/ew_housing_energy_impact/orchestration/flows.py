"""Prefect flows for the EPC pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ew_housing_energy_impact.logging_utils import setup_logger
from ew_housing_energy_impact.orchestration.tasks import (
    PIPELINE_TASK_NAMES,
    PREFECT_AVAILABLE,
    build_features,
    clean_south_west,
    download_or_merge_epc,
    generate_eda,
    generate_quality_report,
    register_artifacts,
    run_monitoring,
    run_policy_models,
    train_ml_models,
    validate_clean_dataset,
    write_manifest,
)

try:  # pragma: no cover - exercised when Prefect is installed.
    from prefect import flow, get_run_logger
    from prefect.task_runners import SequentialTaskRunner
except ModuleNotFoundError:  # pragma: no cover - local test fallback before dependency install.
    get_run_logger = None
    SequentialTaskRunner = None
    F = TypeVar("F", bound=Callable[..., Any])

    def flow(*flow_args: Any, **flow_kwargs: Any) -> Callable[[F], F] | F:
        """Fallback no-op decorator used only when Prefect is not installed."""

        if flow_args and callable(flow_args[0]) and not flow_kwargs:
            return flow_args[0]

        def decorator(fn: F) -> F:
            return fn

        return decorator


def _logger():
    if get_run_logger is not None:
        try:
            return get_run_logger()
        except RuntimeError:
            pass
    return setup_logger("ewhei.orchestration")


def planned_task_names(
    *,
    skip_download: bool = False,
    skip_eda: bool = False,
    skip_policy: bool = False,
    skip_training: bool = False,
    skip_quality: bool = False,
    skip_monitoring: bool = False,
) -> list[str]:
    """Return the task plan after applying skip flags."""

    skipped = set()
    if skip_download:
        skipped.add("download_or_merge_epc")
    if skip_eda:
        skipped.add("generate_eda")
    if skip_policy:
        skipped.add("run_policy_models")
    if skip_training:
        skipped.add("train_ml_models")
    if skip_quality:
        skipped.add("generate_quality_report")
    if skip_monitoring:
        skipped.add("run_monitoring")
    return [name for name in PIPELINE_TASK_NAMES if name not in skipped]


_FLOW_KWARGS = {"task_runner": SequentialTaskRunner()} if SequentialTaskRunner is not None else {}


@flow(name="epc-pipeline", **_FLOW_KWARGS)
def epc_pipeline_flow(
    mode: str = "la",
    skip_download: bool = False,
    skip_eda: bool = False,
    skip_policy: bool = False,
    skip_training: bool = False,
    skip_quality: bool = False,
    skip_monitoring: bool = False,
    dry_run: bool = False,
) -> dict[str, str | bool | list[str]]:
    """Run the EPC pipeline in dependency order using Prefect orchestration."""

    log = _logger()
    plan = planned_task_names(
        skip_download=skip_download,
        skip_eda=skip_eda,
        skip_policy=skip_policy,
        skip_training=skip_training,
        skip_quality=skip_quality,
        skip_monitoring=skip_monitoring,
    )
    log.info("EPC pipeline plan: %s", " -> ".join(plan))

    if dry_run:
        for task_name in plan:
            log.info("Dry run task: %s", task_name)
        return {
            "status": "dry_run",
            "mode": mode,
            "planned_tasks": plan,
            "prefect_available": PREFECT_AVAILABLE,
        }

    download_or_merge_epc(mode=mode, skip_download=skip_download, dry_run=False)
    clean_south_west(dry_run=False)
    build_features(dry_run=False)
    validate_clean_dataset(dry_run=False)
    generate_eda(skip_eda=skip_eda, dry_run=False)
    run_policy_models(skip_policy=skip_policy, dry_run=False)
    train_ml_models(skip_training=skip_training, dry_run=False)
    generate_quality_report(skip_quality=skip_quality, dry_run=False)
    run_monitoring(skip_monitoring=skip_monitoring, dry_run=False)
    write_manifest(dry_run=False)
    register_artifacts(dry_run=False)

    return {
        "status": "completed",
        "mode": mode,
        "planned_tasks": plan,
        # Surfaced so a run that silently degraded to plain sequential calls is
        # visible in the result rather than indistinguishable from a real one.
        "prefect_available": PREFECT_AVAILABLE,
    }
