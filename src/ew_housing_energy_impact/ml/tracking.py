"""Optional MLflow tracking helpers.

All MLflow imports are lazy. When MLflow is disabled, every public helper is a
safe no-op so local training keeps its existing behaviour.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ew_housing_energy_impact.settings import get_settings

logger = logging.getLogger(__name__)


def is_mlflow_enabled(force_enable: bool | None = None) -> bool:
    """Return whether MLflow tracking should be active."""

    if force_enable is not None:
        return force_enable
    return get_settings().enable_mlflow


def _import_mlflow(force_enable: bool | None = None):
    if not is_mlflow_enabled(force_enable):
        return None
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled but mlflow is not installed. "
            "Install requirements.txt or set ENABLE_MLFLOW=false."
        ) from exc
    return mlflow


def setup_mlflow(force_enable: bool | None = None):
    """Configure MLflow tracking if enabled; otherwise return None."""

    mlflow = _import_mlflow(force_enable)
    if mlflow is None:
        return None

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return mlflow


def start_run(
    run_name: str | None = None,
    nested: bool = False,
    force_enable: bool | None = None,
):
    """Start an MLflow run when enabled, otherwise return a no-op context."""

    mlflow = setup_mlflow(force_enable)
    if mlflow is None:
        return nullcontext(None)
    return mlflow.start_run(run_name=run_name, nested=nested)


def _safe_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | int | float):
        return value
    if value is None:
        return "None"
    text = str(value)
    return text if len(text) <= 250 else text[:247] + "..."


def log_params_safe(params: dict[str, Any] | None, force_enable: bool | None = None) -> None:
    """Log MLflow params when enabled, dropping empty values."""

    mlflow = setup_mlflow(force_enable)
    if mlflow is None or not params:
        return
    safe_params = {
        str(key): _safe_value(value) for key, value in params.items() if value is not None
    }
    if safe_params:
        mlflow.log_params(safe_params)


def log_metrics_safe(metrics: dict[str, Any] | None, force_enable: bool | None = None) -> None:
    """Log numeric MLflow metrics when enabled."""

    mlflow = setup_mlflow(force_enable)
    if mlflow is None or not metrics:
        return
    safe_metrics = {}
    for key, value in metrics.items():
        try:
            safe_metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def log_artifact_safe(
    path: str | Path | None,
    artifact_path: str | None = None,
    force_enable: bool | None = None,
) -> None:
    """Log a file artifact when MLflow is enabled and the file exists."""

    mlflow = setup_mlflow(force_enable)
    if mlflow is None or path is None:
        return
    artifact = Path(path)
    if artifact.exists():
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
    else:
        logger.warning("MLflow artifact not found, skipping: %s", artifact)


def log_model_safe(
    model: Any,
    artifact_path: str,
    force_enable: bool | None = None,
) -> None:
    """Log a scikit-learn compatible model when MLflow is enabled."""

    mlflow = setup_mlflow(force_enable)
    if mlflow is None or model is None:
        return
    try:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception as exc:
        logger.warning("MLflow model logging failed for %s: %s", artifact_path, exc)


def log_training_run(
    *,
    task_type: str,
    model_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifact_paths: list[str | Path] | None = None,
    model: Any = None,
    force_enable: bool | None = None,
) -> bool:
    """Log one model training run if MLflow is enabled.

    Returns True when a run was logged and False when tracking was disabled.
    """

    if not is_mlflow_enabled(force_enable):
        return False

    with start_run(run_name=f"{task_type}-{model_name}", force_enable=force_enable):
        log_params_safe({"task_type": task_type, "model_name": model_name}, force_enable)
        log_params_safe(params, force_enable)
        log_metrics_safe(metrics, force_enable)
        for path in artifact_paths or []:
            log_artifact_safe(path, artifact_path=task_type, force_enable=force_enable)
        if model is not None:
            safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
            log_model_safe(model, artifact_path=f"models/{safe_name}", force_enable=force_enable)
    return True
