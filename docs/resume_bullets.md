# Resume And Interview Packaging

## One-Line Project Summary

Production-style EPC housing-energy analytics platform with reproducible data
pipelines, policy modelling, ML training, orchestration, optional Postgres/MLflow,
monitoring, FastAPI, Streamlit, and CI.

## Data Engineer Bullets

- Built a reproducible Python data pipeline for UK EPC bulk data, including API
  ingestion, controlled schema extraction, chunked cleaning, geography rollups,
  validation, manifests, and artifact registry hashes.
- Added production-style platform layers around an analytics workflow with
  Prefect orchestration, optional PostgreSQL analytical summary tables, Docker
  services, FastAPI serving, and CI quality gates.

## Data Scientist Bullets

- Developed fixed-effects policy-period models for EPC efficiency, CO2 emissions,
  and below-C outcomes, with authority controls, housing features, and
  small-cluster bootstrap inference.
- Built predictive EPC regression and below-C classification models with
  leakage-safe property-level train/test splitting, feature importance, SHAP
  outputs, and model cards.

## ML Engineer Bullets

- Integrated optional MLflow experiment tracking for regression/classification
  models, logging parameters, metrics, artifacts, and supported model objects
  without breaking local file-based training.
- Added lightweight data/model monitoring that compares reference/current feature
  datasets, writes drift reports, and returns retraining recommendations without
  automatic retraining or deployment.

## Analytics Engineer Bullets

- Designed optional SQLAlchemy/PostgreSQL analytical tables for authority-year
  summaries, policy-period summaries, validation runs, artifact registry entries,
  and model metrics while preserving CSV artifacts as fallback.
- Expanded FastAPI into a fallback-friendly analytics service for trends,
  authorities, validation, artifacts, model metadata, drift reports, and
  retraining status.

## 30-Second Interview Explanation

"This project is a production-style EPC housing-energy analytics platform. It
takes UK Domestic EPC bulk data, cleans and standardises it, rolls lower-tier
authorities into stable South West analysis groups, builds policy and housing
features, runs fixed-effects policy-period analysis, trains predictive models,
validates outputs, tracks artifacts, monitors drift, and serves results through
FastAPI and Streamlit. I added Prefect, optional Postgres, optional MLflow, CI,
and demo-ready docs while keeping local execution lightweight."

## 90-Second Interview Explanation

"The project started as a dissertation analysis and I upgraded it into a
production-style data/ML platform. The data engineering challenge is that EPC data
is large, event-based, and geographically messy. I built ingestion and cleaning
steps that download lower-tier/unitary EPC bulk files, keep a controlled schema,
preserve raw authority identifiers, and roll records into 14 stable South West
analysis authorities.

On the analytics side, the pipeline creates policy periods, below-C flags, log
floor area, new-build share, and other modelling features. It runs descriptive
analysis, fixed-effects policy-period models, event-study style outputs, and
predictive ML for EPC score and below-C classification. The ML split is at
property level to reduce leakage from repeated certificates.

To make it production-style, I added Prefect orchestration, validation reports,
manifests, artifact hashes, optional PostgreSQL serving tables, optional MLflow
tracking, lightweight drift monitoring, FastAPI endpoints, Streamlit, Docker
services, and CI checks. I kept optional services optional so tests and demos do
not require full EPC data, live Postgres, or live MLflow."

## Hardest Technical Challenges

- Mapping EPC lower-tier/unitary files into stable South West analysis
  authorities without losing traceability to raw labels.
- Avoiding ML leakage from repeated EPC certificates for the same property.
- Keeping heavyweight EPC downloads out of CI while still testing high-risk logic.
- Adding platform layers without rewriting or destabilising the core analytics
  pipeline.
- Keeping policy modelling language honest and not overstating causal claims.

## Mistakes Avoided

- Using county council codes directly for EPC bulk downloads.
- Random row-level ML splitting when repeated properties exist.
- Making Postgres, MLflow, or full EPC data mandatory for tests.
- Claiming drift checks are statistical proof or automatic retraining triggers.
- Exposing local absolute paths unnecessarily through API responses.

## Trade-Offs Made

- CSV/report artifacts remain the source of truth; Postgres is an optional serving
  layer rather than a full warehouse.
- Prefect wraps existing scripts instead of rewriting transformations into a new
  framework.
- Drift monitoring uses transparent thresholds rather than heavier monitoring
  services.
- FastAPI remains a single file for now because the endpoint surface is still
  small enough to keep low-risk.
