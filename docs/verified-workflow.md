# Verified benchmark and research serving

Run from the repository root. Preserve completed directories; each benchmark
requires a fresh output directory. `PYTHONPATH=src:.` exposes the source package
and repository pipeline modules without installing unrelated application tools.

## Train

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
python -m ew_housing_energy_impact.cli train \
  --release data/releases/epc-sw-2008-2025-v1 \
  --output reports/benchmarks/your-new-run \
  --strategy property --train-cap 200000 --validation-cap 50000 --bootstrap 500
```

Use `--strategy chronological` for the separate 2025 holdout. The new `train`
command always tracks actual runs in local MLflow. The old implementation is
available as `legacy-train`; historical `run-all` and Prefect orchestration
still use that legacy workflow and must not be cited as the verified benchmark.

## Verify and reproduce

```sh
PYTHONPATH=src:. python -m scripts.pipeline.verify_benchmark \
  reports/benchmarks/your-new-run \
  --release data/releases/epc-sw-2008-2025-v1 \
  --output reports/evidence/your-new-run-verification.json
```

Create an isolated environment and install `requirements-benchmark.txt`. This
pins direct numerical/reporting dependencies; the actual transitive environment
is captured in each run. Python 3.13 may compile pandas and SHAP from source.
Rerun training with the isolated interpreter and a fresh output directory, then:

```sh
PYTHONPATH=src:. python -m scripts.pipeline.compare_benchmarks \
  reports/benchmarks/property-v1 reports/benchmarks/property-reproduce-v1 \
  --release data/releases/epc-sw-2008-2025-v1 \
  --output reports/evidence/reproduction-comparison.json
```

Comparison independently verifies both runs, requires identical data/split/CV/
code/parameter identities and selected models, and checks metrics and predictions
with relative and absolute tolerance 1e-8. Failure must remain visible; do not
silently loosen tolerances or replace the original benchmark.

## Research prediction API

```sh
EPC_MODEL_RUN=reports/benchmarks/property-v1 \
EPC_MODEL_VERIFICATION=reports/evidence/benchmark-property-v1-verification.json \
PYTHONPATH=src:. python -m uvicorn ew_housing_energy_impact.ml.serving:create_app \
  --factory --host 127.0.0.1 --port 8001
```

`GET /health` returns the loaded run ID; `GET /model` returns model provenance
and fitted category labels. `POST /predict` accepts up to 100 properties:

```json
{"properties":[{"property_type":"House","age_band":"England and Wales: 1983-1990","tenure":"Owner-occupied","main_fuel":"gas","floor_area":90}]}
```

Floor area is in square metres, optional, and restricted to the release cohort's
20–1,000 range. Other inputs are optional standardised categorical labels from
`/model`. Responses include score, below-C probability, classification threshold
result, missing inputs and unseen categories. Unknown categories use the trained
encoder's fallback; this is not evidence of reliable extrapolation. Regression
outputs are not clipped to EPC bounds, preserving the fitted model's behaviour.

The service loads only operator-configured trusted local joblib files and checks
model/protocol hashes before deserialization. An API caller cannot provide model
paths. Hashes detect changes; they do not authenticate an untrusted model author.

This service is a tested local research interface. No public deployment,
authentication, sustained-load performance, regulatory suitability, adoption or
energy-saving outcomes have been established. The local smoke measurement uses
in-process TestClient requests; it is not network or production latency.
