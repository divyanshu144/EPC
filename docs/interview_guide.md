# Interview Guide

## 30-Second Explanation

This project is a reproducible data/ML pipeline for UK Domestic EPC data in South
West England. It downloads EPC bulk data, cleans and rolls raw local authority
records into stable analysis authorities, engineers policy and housing features,
generates EDA outputs, runs fixed-effects policy-period models, trains predictive
models, validates outputs, tracks artifacts, and exposes results through Streamlit
and FastAPI.

## 90-Second Explanation

The project started as a dissertation analysis and has been structured into a
production-oriented analytics pipeline. The raw EPC data is large and
geographically messy, so the first challenge is reliable ingestion and cleaning.
The pipeline downloads domestic EPC bulk files, keeps a controlled schema, filters
South West records, preserves raw authority identifiers, and rolls lower-tier or
unitary authorities into 14 stable analysis authorities.

After cleaning, the pipeline builds modelling features such as policy periods,
below-C flags, log floor area, missingness indicators, and new-build share. It then
generates descriptive analysis, fixed-effects policy-period models, event-study
style year coefficients, heterogeneity views, local-authority clustering, and
predictive ML models for EPC score and below-C classification. The ML split is done
at property level to avoid leakage from repeated EPC certificates.

The platform also includes validation reports, data manifests, an artifact
registry with hashes, tests for the highest-risk logic, CI quality checks,
Prefect orchestration, optional PostgreSQL summary tables, optional MLflow
tracking, drift monitoring, a Streamlit dashboard, and expanded FastAPI
endpoints.

## Hardest Technical Challenges

- **Geography rollup:** EPC bulk files are published for lower-tier/unitary
  authorities, while the analysis needs 14 stable South West authority groups.
  Using county council codes would miss data, so the pipeline downloads lower-level
  files and rolls them up consistently.
- **Repeated-property leakage:** A property can have multiple EPC certificates.
  Random row-level splitting can put the same dwelling in train and test sets, so
  the modelling split uses `BUILDING_REFERENCE_NUMBER`.
- **Large file handling:** The ingestion and cleaning steps use chunked reads and
  reduced schemas to avoid unnecessary memory pressure.
- **Small-cluster inference:** There are only 14 analysis authorities, so naive
  clustered standard errors are weak. The policy models report wild cluster
  bootstrap p-values for policy-period terms.
- **Reproducibility:** The project writes validation reports, manifests, model
  cards, and registry entries so outputs are traceable rather than ad hoc.

## Honest Limitations

- EPC data is lodgement/event based, not a complete annual stock census.
- The fixed-effects policy-period models are observational and should not be sold
  as definitive causal DiD.
- Policy periods are annual approximations.
- CSV/report artifacts remain the source of truth; PostgreSQL stores optional
  analytical summaries and metadata, not the full raw warehouse.
- MLflow tracking is optional and model registry promotion is not implemented.
- Drift monitoring is threshold based and recommends retraining consideration;
  it does not retrain or deploy models automatically.
- The API is demo-ready but still lacks authentication, pagination, and a
  prediction endpoint.

## Best Resume Bullets

- Built a reproducible Python data/ML pipeline for 2008-2025 UK EPC housing energy
  data, covering ingestion, cleaning, feature engineering, modelling, validation,
  dashboarding, and API access.
- Designed a geography rollup layer that maps lower-tier/unitary EPC authorities
  into 14 stable South West analysis authorities while preserving raw identifiers.
- Implemented leakage-safe model evaluation by splitting repeated EPC certificates
  at property level using `BUILDING_REFERENCE_NUMBER`.
- Developed fixed-effects policy-period models with small-cluster bootstrap
  inference, plus event-study style outputs and heterogeneity analysis.
- Added model training, feature importance, SHAP explainability, model cards,
  artifact hashes, and validation manifests for reproducible ML reporting.
- Hardened developer workflow with Ruff linting, formatting checks, pytest,
  coverage support, Make targets, and GitHub Actions CI.
- Added Prefect orchestration, optional PostgreSQL analytical summaries, optional
  MLflow tracking, drift monitoring, and expanded FastAPI endpoints while keeping
  tests lightweight.

## Skills Demonstrated

- Python data engineering
- pandas chunked processing
- data validation and reproducibility
- analytical modelling with fixed effects
- scikit-learn pipelines and model evaluation
- explainability with feature importance and SHAP
- FastAPI and Streamlit serving layers
- Docker-based local services
- CI/CD quality gates
- technical documentation and architecture communication
