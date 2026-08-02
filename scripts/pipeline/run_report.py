#!/usr/bin/env python3
"""Run the full report pipeline in order."""

from __future__ import annotations

from ew_housing_energy_impact.cli import cmd_manifest, cmd_validate
from scripts.pipeline import (
    build_features,
    clean_epc,
    epc_download_merge_fast,
    quality_report,
    report_eda,
    run_did,
    train_models,
)


def main() -> None:
    epc_download_merge_fast.main()
    clean_epc.main()
    build_features.main()
    report_eda.main()
    run_did.main()
    train_models.main()
    quality_report.main()
    cmd_validate()
    cmd_manifest()


if __name__ == "__main__":
    main()
