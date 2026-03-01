# CLAUDE.md — EPC South West Analysis Pipeline

## Project overview

Reproducible academic pipeline for the dissertation:
**"The Impact of Energy Policy on Housing Energy Performance in the South West of England (2008–2025)"**

Analyses ~1.8 M domestic EPC lodgements across 14 South West local authorities using:
- Fixed-effects panel OLS (`linearmodels.PanelOLS`, indexed on `[LOCAL_AUTHORITY, YEAR]`)
- Wild cluster bootstrap for inference (14 clusters — too few for standard asymptotic clustered SEs)
- Year-by-year event study alongside five blunt policy-period dummies
- Gradient-boosted regression (XGBoost) with SHAP explainability
- PCA + KMeans clustering of LA random effects

The package is installed as `ew-housing-energy-impact` (editable). The CLI entry point is `ewhei`.

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # installs the ewhei CLI
```

Create a `.env` file in the project root:
```
EPC_EMAIL=your@email.com
EPC_API_KEY=your_key_here
```

Credentials are obtained from https://epc.opendatacommunities.org (free registration).

---

## Running the pipeline

### Full pipeline (one command)
```bash
ewhei run-all
```

### Step by step (recommended during development)
```bash
ewhei download          # LA mode: ~200 MB (14 SW zips). Use --mode year for full ~6.7 GB national
ewhei clean             # Filter to SW LAs, standardise fields, derive YEAR
ewhei features          # Impute, add LOG_FLOOR_AREA, POLICY_PERIOD, NEW_BUILD_SHARE
ewhei eda               # EDA figures and tables -> reports/
ewhei fe                # Fixed-effects models + event study + clustering -> reports/artifacts/
ewhei train             # GBR predictive model + SHAP -> reports/artifacts/
ewhei validate          # Schema / range checks -> reports/artifacts/validation_report.json
ewhei manifest          # Data manifest -> reports/artifacts/data_manifest.json
```

### Make shortcuts
```bash
make install    # pip install -r requirements.txt && pip install -e .
make test       # pytest
make validate   # ewhei validate
```

---

## Data paths (defined in `config.py`)

| Variable | Path |
|---|---|
| `RAW_BULK_DIR` | `data/raw/ew_bulk/` — downloaded `.zip` files |
| `RAW_EXTRACT_DIR` | `data/raw/ew_extracted/` — unzipped certificate folders |
| `EPC_MERGED_CSV` | `data/processed/ew_epc_core.csv` — merged national CSV |
| `EPC_CLEAN_SW_DIR` | `data/processed/ew_epc_core_clean_sw/` — SW-filtered outputs |
| `ARTIFACTS_DIR` | `reports/artifacts/` — tables, figures, registry |
| `REGISTRY_PATH` | `reports/artifacts/registry.jsonl` — artifact provenance log |

Key processed files:
- `ew_epc_core_clean_sw.csv` — cleaned SW records
- `ew_epc_core_clean_sw_imputed.csv` — imputed + feature-engineered, used by all models

**Always import `ARTIFACTS_DIR` and `REGISTRY_PATH` from `config.py`. Never redefine them inline.**

---

## Code map

```
config.py                              # All shared paths and lookup maps
scripts/pipeline/
  schemas.py                           # CORE_COLS, CORE_DTYPES, CLEAN_KEEP_COLS
  policy.py                            # policy_period(), POLICY_ORDER, _BOUNDARIES
  epc_download_merge_fast.py           # Download (LA mode default) + merge
  clean_epc.py                         # SW filter, standardise, derive YEAR
  build_features.py                    # Impute, LOG_FLOOR_AREA, NEW_BUILD_SHARE, POLICY_PERIOD
  report_eda.py                        # EDA figures and summary tables
  run_did.py                           # FE models, WCB, event study, clustering
  train_models.py                      # XGBoost + SHAP + PCA clustering
  quality_report.py                    # HTML data quality report
src/ew_housing_energy_impact/
  cli.py                               # ewhei CLI (argparse, delegates to scripts)
  validation.py                        # Schema/range validation
  manifest.py                          # Data manifest writer
  registry.py                          # Artifact registry (JSONL append)
  logging_utils.py                     # Structured logger setup
  paths.py                             # repo_root() helper
```

---

## Policy periods (defined in `scripts/pipeline/policy.py`)

| Period | Years | Notes |
|---|---|---|
| Pre-GreenDeal | 2008–2012 | Baseline |
| GreenDeal-ECO1 | 2013–**2015** | ECO2 launched April 2015 but 2015 stays here — annual data cannot resolve within-year transitions |
| ECO2 | 2016–2018 | |
| MEES | 2019–2020 | MEES enforcement April 2018; 2019–2020 = first full compliance cycle |
| Post-Strategy | 2021–2025 | |

**Do not move 2015 to ECO2.** The boundary is deliberate and documented in the policy.py module docstring.

---

## Statistical design decisions

### Why wild cluster bootstrap (not standard clustered SEs)
14 local authorities = 14 clusters. Cameron, Gelbach & Miller (2008) show clustered SEs are unreliable below ~30 clusters. `run_did.py` runs WCB with B=999 Rademacher weights alongside the standard clustered SEs; both appear in output tables.

### Event study vs policy periods
`BASE_RHS` uses `C(POLICY_PERIOD)` (five blunt dummies). `EVENT_STUDY_RHS` uses `C(YEAR)` with 2008 as the implicit reference. Both are estimated; the event study produces annotated year-by-year plots (`figure_6_5/6/7_event_study_*.png`).

### NEW_BUILD_SHARE covariate
Fraction of EPC lodgements flagged `TRANSACTION_TYPE == 'new dwelling'` per LA-year. Controls for the compositional shift that mechanically raises mean EPC scores in high-construction years. Computed in `build_features.py:add_new_build_share()` and included in both `BASE_RHS` and `EVENT_STUDY_RHS`.

### ML train/test split
Split is by `BUILDING_REFERENCE_NUMBER` (property level), not by row. This prevents the same physical property appearing in both train and test sets — a data leakage risk when the same address is re-lodged across years.

---

## Testing

```bash
pytest                  # runs tests/test_policy.py and tests/test_validation.py
```

Tests cover `policy_period()` boundary cases and the imputed CSV schema validation. Run tests before and after any changes to `policy.py` or `validation.py`.

---

## Download modes

| Mode | Data | Size | When to use |
|---|---|---|---|
| `--mode la` (default) | 14 SW LA zips only | ~200 MB | Development, disk-constrained |
| `--mode year` | 18 annual national bulk zips | ~6.7 GB compressed / ~50 GB unzipped | Full national replication |

The disk on this machine has limited free space (~26 GB available at project start). Always use LA mode unless the full national dataset is explicitly needed.

---

## Docker

```bash
make build      # docker compose build
make up         # docker compose up (notebook + API + Streamlit)
make down       # docker compose down
make bash       # shell into ew-notebook container
make logs       # tail ew-notebook logs
```

Services defined in `docker-compose.yml`: `ew-notebook`, `ew-api` (FastAPI, `Dockerfile.api`), `ew-streamlit` (`Dockerfile.streamlit`).

---

## Key conventions

- **No `sys.path` hacks.** The package is installed editably; imports work without path manipulation.
- **All paths from `config.py`.** Never hardcode directory paths in pipeline scripts.
- **All artifacts registered.** Every output file should call `register_artifact(REGISTRY_PATH, ...)` immediately after writing.
- **Chunked CSV reads.** Large CSVs (merged core, clean SW) are read with `chunksize=1_000_000` to avoid OOM.
- **No redundant writes.** Each intermediate CSV is written exactly once per pipeline run.


## Common Gotchas
- Never redefine `ARTIFACTS_DIR` or `REGISTRY_PATH` inline — always import from `config.py`
- Never use `--mode year` unless explicitly asked — disk is limited (~26 GB)
- Always run `pytest` before and after changes to `policy.py` or `validation.py`
- Never move 2015 into ECO2 period — boundary is deliberate


## Environment
- Python 3.11
- Installed as editable package: `pip install -e .`
- CLI entry point: `ewhei`