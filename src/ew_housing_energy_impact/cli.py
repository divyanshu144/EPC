"""Command-line interface for the EPC report pipeline."""

from __future__ import annotations

import argparse
import sys

from ew_housing_energy_impact.paths import repo_root

# The shared `config` module and the `scripts.pipeline.*` modules live at the
# repo root, which is NOT part of the installed (src-layout) package. Put the
# repo root on sys.path *before* importing them so the installed `ewhei` console
# script works from any working directory, not only when launched from the repo
# root. This must run before the `validation` and `config` imports below, which
# transitively pull in `scripts.pipeline.*`.
_ROOT = repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config import ARTIFACTS_DIR, REGISTRY_PATH
from ew_housing_energy_impact.logging_utils import setup_logger
from ew_housing_energy_impact.manifest import write_manifest
from ew_housing_energy_impact.registry import register_artifact
from ew_housing_energy_impact.validation import validate_imputed_csv, write_validation_report


def cmd_run_all(mode: str = "la") -> None:
    from scripts.pipeline import (
        build_features,
        clean_epc,
        epc_download_merge_fast,
        report_eda,
        run_did,
        train_models,
    )

    epc_download_merge_fast.main(mode=mode)
    clean_epc.main()
    build_features.main()
    report_eda.main()
    run_did.main()
    train_models.main()


def cmd_validate() -> None:
    root = repo_root()
    data_path = (
        root / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
    )
    report_path = ARTIFACTS_DIR / "validation_report.json"
    result = validate_imputed_csv(data_path)
    write_validation_report(result, report_path)
    register_artifact(REGISTRY_PATH, "validation_report", report_path, {"ok": result.ok})
    logger = setup_logger()
    if result.ok:
        logger.info("Validation OK. Report written to %s", report_path)
    else:
        logger.warning("Validation issues found. Report written to %s", report_path)


def cmd_manifest() -> None:
    root = repo_root()
    data_path = (
        root / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
    )
    out_path = ARTIFACTS_DIR / "data_manifest.json"
    df = pd.read_csv(data_path)
    write_manifest(df, out_path)
    register_artifact(REGISTRY_PATH, "manifest", out_path, {"type": "data_manifest"})
    logger = setup_logger()
    logger.info("Manifest written to %s", out_path)


def cmd_orchestrate(
    mode: str = "la",
    skip_download: bool = False,
    skip_eda: bool = False,
    skip_policy: bool = False,
    skip_training: bool = False,
    skip_quality: bool = False,
    skip_monitoring: bool = False,
    dry_run: bool = False,
) -> None:
    from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

    result = epc_pipeline_flow(
        mode=mode,
        skip_download=skip_download,
        skip_eda=skip_eda,
        skip_policy=skip_policy,
        skip_training=skip_training,
        skip_quality=skip_quality,
        skip_monitoring=skip_monitoring,
        dry_run=dry_run,
    )
    setup_logger().info("Orchestration result: %s", result)


def cmd_db_init() -> None:
    from ew_housing_energy_impact.db.engine import create_tables, get_engine
    from ew_housing_energy_impact.settings import get_settings

    settings = get_settings()
    create_tables(get_engine(settings.database_url))
    setup_logger().info("Database tables created at %s", settings.database_url)


def cmd_db_load(dry_run: bool = False) -> None:
    from ew_housing_energy_impact.db.loaders import refresh_analytics_database

    result = refresh_analytics_database(dry_run=dry_run)
    setup_logger().info("Database load result: %s", result)


def cmd_mlflow_info() -> None:
    from ew_housing_energy_impact.ml.tracking import is_mlflow_enabled
    from ew_housing_energy_impact.settings import get_settings

    settings = get_settings()
    logger = setup_logger()
    logger.info("MLflow enabled: %s", is_mlflow_enabled())
    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)
    logger.info("MLflow experiment: %s", settings.mlflow_experiment_name)


def cmd_monitor(
    reference: str | None = None,
    current: str | None = None,
    output: str | None = None,
    dry_run: bool = False,
) -> None:
    from ew_housing_energy_impact.monitoring.reporting import run_monitoring

    result = run_monitoring(
        reference_path=reference,
        current_path=current,
        output_path=output,
        dry_run=dry_run,
    )
    setup_logger().info("Monitoring result: %s", result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ewhei", description="EPC report pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    _MODE_HELP = (
        "'la' (default): download only SW LA zips (~200 MB). "
        "'year': full annual E&W bulk zips (~6.7 GB)."
    )
    run_all = sub.add_parser("run-all", help="Run the full report pipeline")
    run_all.add_argument("--mode", choices=["la", "year"], default="la", help=_MODE_HELP)
    dl = sub.add_parser("download", help="Download and merge EPC data")
    dl.add_argument("--mode", choices=["la", "year"], default="la", help=_MODE_HELP)
    sub.add_parser("clean", help="Clean EPC data for South West")
    sub.add_parser("features", help="Build features and policy periods")
    sub.add_parser("eda", help="Generate EDA figures and tables")
    sub.add_parser("fe", help="Run fixed-effects and clustering analysis")
    train = sub.add_parser("train", help="Run predictive modelling")
    train.add_argument(
        "--track",
        action="store_true",
        help="Force-enable MLflow tracking for this training run",
    )
    sub.add_parser("quality", help="Generate data quality HTML report")
    sub.add_parser("validate", help="Validate imputed dataset and write report")
    sub.add_parser("manifest", help="Write data manifest")
    monitor = sub.add_parser("monitor", help="Run lightweight drift monitoring")
    monitor.add_argument("--reference", help="Reference feature CSV path")
    monitor.add_argument("--current", help="Current/latest feature CSV path")
    monitor.add_argument(
        "--output",
        default=None,
        help="JSON report output path; Markdown is written beside it",
    )
    monitor.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned monitoring files without reading datasets",
    )
    sub.add_parser("mlflow-info", help="Show optional MLflow tracking configuration")
    sub.add_parser("db-init", help="Create optional analytical database tables")
    db_load = sub.add_parser("db-load", help="Load generated summaries into the analytical DB")
    db_load.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned table loads without modifying the database",
    )
    orchestrate = sub.add_parser("orchestrate", help="Run the Prefect-orchestrated pipeline")
    orchestrate.add_argument("--mode", choices=["la", "year"], default="la", help=_MODE_HELP)
    orchestrate.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip EPC API download/merge and use existing merged data",
    )
    orchestrate.add_argument("--skip-eda", action="store_true", help="Skip EDA outputs")
    orchestrate.add_argument(
        "--skip-policy", action="store_true", help="Skip fixed-effects policy models"
    )
    orchestrate.add_argument("--skip-training", action="store_true", help="Skip ML training")
    orchestrate.add_argument(
        "--skip-quality", action="store_true", help="Skip HTML quality report generation"
    )
    orchestrate.add_argument(
        "--skip-monitoring", action="store_true", help="Skip drift monitoring reports"
    )
    orchestrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned Prefect task order without running pipeline steps",
    )

    args = parser.parse_args()

    if args.cmd == "run-all":
        cmd_run_all(mode=args.mode)
    elif args.cmd == "download":
        from scripts.pipeline import epc_download_merge_fast

        epc_download_merge_fast.main(mode=args.mode)
    elif args.cmd == "clean":
        from scripts.pipeline import clean_epc

        clean_epc.main()
    elif args.cmd == "features":
        from scripts.pipeline import build_features

        build_features.main()
    elif args.cmd == "eda":
        from scripts.pipeline import report_eda

        report_eda.main()
    elif args.cmd == "fe":
        from scripts.pipeline import run_did

        run_did.main()
    elif args.cmd == "train":
        from scripts.pipeline import train_models

        train_models.main(track=args.track or None)
    elif args.cmd == "quality":
        from scripts.pipeline import quality_report

        quality_report.main()
    elif args.cmd == "validate":
        cmd_validate()
    elif args.cmd == "manifest":
        cmd_manifest()
    elif args.cmd == "monitor":
        cmd_monitor(
            reference=args.reference,
            current=args.current,
            output=args.output,
            dry_run=args.dry_run,
        )
    elif args.cmd == "orchestrate":
        cmd_orchestrate(
            mode=args.mode,
            skip_download=args.skip_download,
            skip_eda=args.skip_eda,
            skip_policy=args.skip_policy,
            skip_training=args.skip_training,
            skip_quality=args.skip_quality,
            skip_monitoring=args.skip_monitoring,
            dry_run=args.dry_run,
        )
    elif args.cmd == "db-init":
        cmd_db_init()
    elif args.cmd == "db-load":
        cmd_db_load(dry_run=args.dry_run)
    elif args.cmd == "mlflow-info":
        cmd_mlflow_info()


if __name__ == "__main__":
    main()
