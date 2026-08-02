"""Airflow compatibility note.

This repository now uses Prefect for local-friendly orchestration. The production
workflow entry point is:

    ewhei orchestrate

This file intentionally does not define an Airflow DAG. It remains only to make
the previous placeholder explicit and avoid implying that Airflow orchestration is
implemented.
"""

AIRFLOW_DAG_IMPLEMENTED = False
ORCHESTRATOR = "prefect"
