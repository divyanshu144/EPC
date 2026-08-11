"""Environment-based settings for optional platform services."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ew_housing_energy_impact.paths import repo_root


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def default_database_url() -> str:
    """Return a safe local SQLite default for optional analytics experiments."""

    db_path = repo_root() / "reports" / "artifacts" / "ewhei_analytics.db"
    return f"sqlite:///{db_path}"


def default_mlflow_tracking_uri() -> str:
    """Return a local file-backed MLflow tracking URI."""

    mlruns_path = repo_root() / "mlruns"
    return f"file://{mlruns_path}"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    database_url: str
    enable_database: bool
    enable_mlflow: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    epc_bearer_token: str


def get_settings() -> Settings:
    """Load settings from environment variables without hardcoded secrets."""

    return Settings(
        database_url=os.getenv("DATABASE_URL", default_database_url()),
        enable_database=_env_bool("ENABLE_DATABASE", default=False),
        enable_mlflow=_env_bool("ENABLE_MLFLOW", default=False),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", default_mlflow_tracking_uri()),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "epc-housing-energy"),
        epc_bearer_token=os.getenv("EPC_BEARER_TOKEN", ""),
    )
