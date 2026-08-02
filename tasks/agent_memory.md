# Agent Memory

Durable structured reference for this project. Unlike `lessons.md` (chronological),
this file is organised by topic and kept current.

**Locked decisions are binding.** Honour them even if the code appears to suggest
otherwise — if the code contradicts a locked decision, that is a bug to report, not a
licence to change the decision.

---

## Architecture Decisions

### LOCKED — Policy period boundaries
Five periods defined in `scripts/pipeline/policy.py`. **2015 belongs to GreenDeal-ECO1,
not ECO2**, even though ECO2 launched April 2015 — annual data cannot resolve
within-year transitions. Do not "fix" this.

### LOCKED — Wild cluster bootstrap for inference
14 local authorities = 14 clusters. Cameron, Gelbach & Miller (2008): clustered SEs are
unreliable below ~30 clusters. `run_did.py` runs WCB (B=999, Rademacher weights)
*alongside* standard clustered SEs; both appear in output tables. Do not drop either.

### LOCKED — ML split by `BUILDING_REFERENCE_NUMBER`
Train/test split is at property level, not row level. The same physical property is
re-lodged across years; a row-level split leaks it into both sides.

### LOCKED — Optional services degrade, never raise
Database and MLflow are default-off (`ENABLE_DATABASE`, `ENABLE_MLFLOW`). Every call site
falls back to CSV/artifacts when the service is absent. This is why the suite passes with
no Postgres and no MLflow server. Preserve the fallback in any new call site.

### Prefect, not Airflow
`dags/etl_epc_pipeline.py` is a deliberate tombstone (`AIRFLOW_DAG_IMPLEMENTED = False`)
left in place so the previous placeholder cannot be mistaken for a real DAG. The flow is
`orchestration/flows.py:epc_pipeline_flow` — 11 tasks, each individually skippable.

### Analysis framing is intentionally conservative
Fixed-effects policy-period models, **not** definitive causal DiD claims. Keep the hedged
language in reports and README; don't upgrade it to causal phrasing.

---

## Known Gotchas

| Gotcha | Detail |
|---|---|
| Green tests ≠ working CLI | `conftest.py` patches `sys.path`; verify with `make smoke` |
| Repo move breaks `ewhei` | Editable install stores an absolute path; re-run `pip install -e .` |
| Python version split | CI = 3.11, local = Anaconda 3.13.5. CI is the authority |
| Stale knowledge graph | Check node count / commit before trusting it for impact analysis |
| Disk is tight (~26 GB) | Never `--mode year` (~6.7 GB compressed, ~50 GB unzipped) unless asked |
| `ARTIFACTS_DIR` / `REGISTRY_PATH` | Import from `config.py`; never redefine inline |
| `E402` is ignored on purpose | Three sanctioned `sys.path` inserts reach root-level `config`/`scripts` |

---

## Solved Problems

### `ModuleNotFoundError` despite `pip show` succeeding — 2026-08-02
Stale editable-install path after the repo moved to `All Projects/`. Diagnose with
`pip show -f <pkg> | grep "Editable project location"`. Fix with `pip install -e .`
from the current directory. Full write-up in `lessons.md`.

---

## Useful Patterns

### Verification baseline (run before claiming done)
```bash
make ci      # ruff check + ruff format --check + pytest  -> expect 68 passed
make smoke   # ewhei --help + orchestrate/monitor/db-load --dry-run
```

### Dry-run first
`orchestrate`, `monitor`, and `db-load` all support `--dry-run`, which prints the planned
tasks/paths without touching data. Use it to confirm wiring before a real run — cheap, and
it catches path mistakes before a multi-minute pipeline does.

### Settings access
Read optional-service config through `settings.get_settings()`, never `os.getenv` directly.
Keeps defaults in one place and keeps secrets out of the code.

### Artifact provenance
Every written output calls `register_artifact(REGISTRY_PATH, ...)` immediately after the
write, appending to `reports/artifacts/registry.jsonl`.
