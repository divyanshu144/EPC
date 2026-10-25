# EPC South West — Technical Deep Dive

Written 2026-08-12 by reading the code at commit `3d81db2`, not the README. Where the
documentation and the code disagree, this document follows the code and says so.

Read this before an interview. It is organised so you can defend the design, not just
describe it.

---

## 1. Problem and purpose

### The question

England and Wales require an Energy Performance Certificate whenever a dwelling is built,
sold or let. Each certificate records an efficiency score (1–100), a band (A–G), estimated
CO₂ emissions, floor area, heating fuel, tenure and construction age. Between 2008 and 2025
the UK ran a sequence of housing-energy policies — Green Deal, ECO1/ECO2, Minimum Energy
Efficiency Standards, the Heat and Buildings Strategy.

The project asks: **did measured domestic energy performance in South West England change
across those policy periods, and by how much?**

### Who it is for

Originally a dissertation. The repository has since been rebuilt as a data platform, so the
audience is now two: the analysis itself, and a demonstration of how you would operate that
analysis as production data engineering.

### Input and output

**Input.** Bulk EPC certificate data from the government's open-data service. Each archive
holds `certificates.csv` (93 columns) and `recommendations.csv` (7 columns), verified in
`docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md` §0 against nine real
monthly archives.

**Output.** Coefficient tables (`reports/tables/policy_effects_*.csv`), event-study tables and
plots, trained models with SHAP explanations, a validation report, a data manifest, an
artifact registry, and a FastAPI/Streamlit surface over the results.

**Run status.** On 2026-08-12 the pipeline was run on real data for the first time in this
working copy: 1,376,757 national certificates from nine 2025 monthly archives, filtered to
125,111 South West records, producing 23 artifacts under `reports/`. **The econometrics
(`ewhei fe`) could not run** — the available data covers one year, so the policy-period model
is unidentified. Real model metrics and the full caveats are in §7.

---

## 2. Architecture

### Control flow

The single entry point is the `ewhei` console script, declared in `pyproject.toml` under
`[project.scripts]` and implemented in `src/ew_housing_energy_impact/cli.py`. It is a plain
`argparse` CLI: each subcommand maps to a `cmd_*` function that imports its dependencies
lazily inside the function body.

That laziness is deliberate and worth being able to explain. Importing `run_did` pulls in
`linearmodels`, `matplotlib` and `seaborn`; importing `train_models` pulls in `xgboost` and
`shap`. Doing that at module scope would make `ewhei --help` take seconds and would make the
CLI fail to start if any heavy optional dependency were missing. Lazy imports keep the
command surface cheap.

```
ewhei <cmd>
   │
   ├── download ──▶ scripts/pipeline/epc_download_merge_fast.py
   ├── clean    ──▶ scripts/pipeline/clean_epc.py
   ├── features ──▶ scripts/pipeline/build_features.py
   ├── eda      ──▶ scripts/pipeline/report_eda.py
   ├── fe       ──▶ scripts/pipeline/run_did.py          ← the econometrics
   ├── train    ──▶ scripts/pipeline/train_models.py     ← the ML
   ├── validate ──▶ src/.../validation.py
   ├── ingest   ──▶ src/.../ingestion/pipeline.py        ← the warehouse path
   ├── monitor  ──▶ src/.../monitoring/reporting.py
   ├── db-load  ──▶ src/.../db/loaders.py
   └── orchestrate ─▶ src/.../orchestration/flows.py     ← Prefect wrapper
```

### The two-layer split, and why it exists

There is a real architectural seam here, and it is historical rather than designed:

- **`scripts/pipeline/`** holds the original analysis: download, clean, features, EDA,
  econometrics, ML. These are procedural modules with a `main()`, reading and writing CSVs.
- **`src/ew_housing_energy_impact/`** is the installed package: CLI, validation, registry,
  manifest, plus the newer subsystems (`db/`, `ml/`, `monitoring/`, `orchestration/`,
  `ingestion/`). These are library code with injected dependencies and unit tests.

The seam has a cost. `scripts/` is not part of the src-layout package, so anything importing
it needs the repo root on `sys.path`. That is why `cli.py:16-18` inserts the repo root before
importing `config`, why `tests/conftest.py` does the same, and why `pyproject.toml` ignores
ruff's `E402` (module-level import not at top of file). Three sanctioned `sys.path` inserts,
documented in `CLAUDE.md` as a hard limit.

**Be honest about this in an interview:** it is incidental structure, not a designed
boundary. The newer code is better-factored than the older code because it was written later
and under test, not because a layering decision was made up front.

### How components communicate

Almost entirely **through the filesystem**. `clean_epc.py` writes a CSV that
`build_features.py` reads; `build_features.py` writes the imputed CSV that `run_did.py`,
`train_models.py`, `api.py` and `app.py` all read independently by path.

That is simple and debuggable — every stage's output is inspectable — but it means there is
no schema contract between stages beyond convention, and every consumer hardcodes the same
path. `api.py:14-16` and `app.py:8-10` each rebuild that path from `Path(__file__)` rather
than importing it from `config.py`, which is a genuine inconsistency with the project's own
stated rule that all paths come from `config.py`.

The one exception is the artifact **registry**: every stage calls
`register_artifact(REGISTRY_PATH, ...)` after writing, appending a JSON line to
`reports/artifacts/registry.jsonl` (`src/ew_housing_energy_impact/registry.py`). That gives
provenance across stages without coupling them.

---

## 3. Stage-by-stage walkthrough

### 3.1 Download — `scripts/pipeline/epc_download_merge_fast.py`, `ingestion/epc_source.py`

**In:** EPC API credentials. **Out:** zip archives in `data/raw/ew_bulk/`, extracted into
`data/raw/ew_extracted/<period>/`.

The download machinery lives in `ingestion/epc_source.py` and is re-exported by the legacy
module for backward compatibility. It is more careful than it first appears:

- `download_one_file` writes to a `.part` file and `os.replace`s on success, so an interrupted
  download never leaves a truncated file that later looks like a valid cache.
- `_is_valid_zip` opens the archive and reads its central directory. A truncated zip has no
  valid end-of-central-directory record, so this catches corruption without paying the
  full-CRC cost of `testzip()` on a large file.
- A cached archive that fails validation is deleted and re-downloaded.
- `extract_all_zips` deletes and raises on `BadZipFile` rather than warning and continuing.

**This stage currently cannot run.** `epc_source.py:102` and
`epc_download_merge_fast.py:56` both raise `NotImplementedError`. The EPC service moved to
GOV.UK: the old host 301-redirects to an HTML landing page, so `raise_for_status()` passes
and `.json()` fails at character 0. Auth also changed from Basic to Bearer, and per-authority
archives no longer exist. The guard is deliberate — it replaces a confusing `JSONDecodeError`
three frames deep with a clear refusal. Full account in `tasks/lessons.md` and spec §0.

**If removed:** nothing downstream has input.

### 3.2 Clean — `scripts/pipeline/clean_epc.py`

**In:** merged national CSV. **Out:** `ew_epc_core_clean_sw.csv`.

Read in chunks (`chunksize=CHUNK`, `clean_epc.py:139`) because the national file does not fit
comfortably in memory. Each chunk passes through `clean_chunk`, which:

1. **Rolls lower-tier authorities up to 14 analysis authorities** via
   `roll_up_to_analysis_authority` (`clean_epc.py:52`). This is the single most important
   transformation in the project and is explained in §5.
2. Normalises construction age bands (`norm_age_band`).
3. Maps fuel and tenure to standardised categories using `FUEL_MAP` and `TENURE_MAP` in
   `config.py`.
4. Drops rows outside the South West.

**If removed:** the panel would be keyed on ~30 districts rather than 14 authorities, and the
fixed-effects entity dimension would be wrong.

### 3.3 Features — `scripts/pipeline/build_features.py`

**In:** cleaned SW CSV. **Out:** `ew_epc_core_clean_sw_imputed.csv` — the file every model
reads.

Four things happen:

- **`ensure_year`** derives `YEAR` from `LODGEMENT_DATE`.
- **`impute_missing`** (`build_features.py:42`) fills categoricals with `"Unknown"` and
  numerics with the column median, and — importantly — writes a `<col>_MISSING` indicator
  column first (`build_features.py:65`). Imputation without an indicator destroys the
  information that a value was absent; this keeps it.
- **`add_new_build_share`** (`build_features.py:74`) computes, per LA-year, the fraction of
  certificates where `TRANSACTION_TYPE` is `"new dwelling"`. This is a control variable, and
  the reason for it is subtle enough to be worth rehearsing: new-build dwellings score far
  higher than the existing stock, so a year with heavy construction raises the LA-year mean
  score *mechanically*, with no retrofit having occurred. Without this control the
  policy-period coefficients absorb that compositional shift and are biased upward. The
  comment at `run_did.py:53-57` states exactly this.
- **`add_derived`** creates `LOG_FLOOR_AREA` (`np.log`), `POLICY_PERIOD` via
  `policy.policy_period`, `EPC_BAND_SCORE` from `BAND_TO_SCORE`, `BELOW_C`, `HAS_MAINS_GAS`,
  `IS_ELECTRIC_HEAT` and `EFFICIENCY_GAP`.

**If removed:** `run_did.py` and `train_models.py` both fail — they reference
`POLICY_PERIOD`, `LOG_FLOOR_AREA` and `NEW_BUILD_SHARE` directly.

### 3.4 Fixed-effects models — `scripts/pipeline/run_did.py`

This is the analytical heart and the part most likely to be probed.

**In:** the imputed CSV. **Out:** `reports/tables/policy_effects_{epc,co2,below_c}.csv`,
`event_study_*.csv`, and figures.

`prepare_panel` (`run_did.py:104`) sets a MultiIndex on `[LOCAL_AUTHORITY, YEAR]` — the panel
structure `linearmodels.PanelOLS` requires. Note the index has **duplicate (LA, YEAR) pairs**,
because the unit of observation is the certificate, not the LA-year. This matters later.

`fit_fe` (`run_did.py:113`) fits:

```
dep ~ 1 + C(POLICY_PERIOD) + C(PROPERTY_TYPE) + C(AGE_BAND_STD) + C(TENURE_STD)
        + C(MAIN_FUEL_STD) + LOG_FLOOR_AREA + NEW_BUILD_SHARE + EntityEffects
```

with `cov_type="clustered"` clustered on local authority. `EntityEffects` absorbs
time-invariant LA characteristics — geography, housing stock age profile, local policy
capacity — so the coefficients are identified from *within-authority* variation over time.

Three dependent variables are modelled: `CURRENT_ENERGY_EFFICIENCY` (continuous),
`CO2_EMISSIONS_CURRENT` (continuous) and `BELOW_C_INT` (binary, fitted as a linear
probability model).

**If removed:** the project has no policy estimates at all.

### 3.5 Wild cluster bootstrap — `run_did.py:124-229`

The most defensible piece of statistics in the repository, and the thing to lead with.

**The problem.** Clustered standard errors are asymptotic in the *number of clusters*. There
are 14 local authorities. Cameron, Gelbach & Miller (2008) show clustered SEs over-reject
badly below roughly 30 clusters — you get false positives.

**The implementation.** For each of B=999 iterations: draw one Rademacher weight (+1/−1) per
*cluster*, apply it to every observation in that authority, form `y* = fitted + w·resid`,
refit, and recentre the bootstrap t-statistic on the full-sample estimate:

```
t* = (beta*_b − beta_hat) / se*_b
```

The recentring is essential and the docstring says why: without it the bootstrap statistics
cluster around the observed estimate rather than the null, and p-values drift toward 0.5.

The p-value uses `(1 + #{|t*| ≥ |t_obs|}) / (n_boot + 1)` (`run_did.py:227`), so it can never
be exactly zero — an honest finite-sample convention.

**Two implementation details worth knowing**, because they show real engineering:

1. Because the panel index has duplicate (LA, YEAR) pairs, label-based assignment is
   impossible. The code reconstructs the estimation sample **positionally** with a boolean
   mask over the non-NaN rows of `_BASE_RHS_COLS` (`run_did.py:173-180`), and raises loudly if
   the mask size does not match the residual count. That guard exists because `PanelOLS` drops
   rows with NaN in any model term, so the estimation sample can be shorter than the panel.
2. A single working frame is reused across iterations, overwriting only the dependent column
   (`run_did.py:194-209`), to avoid copying a ~1.8M-row panel 999 times.

**A genuine methodological critique you should be ready for.** The docstring at
`run_did.py:147-148` describes this as imposing the null "on the statistic, not on the
resampling DGP" — that is the **unrestricted** wild cluster bootstrap (WCB-U). MacKinnon &
Webb's work generally finds the **restricted** variant (WCB-R), which imposes the null when
generating bootstrap samples, has better size properties with very few clusters. The code
picks WCB-U and documents that it is doing so, which is honest, but if an interviewer knows
this literature they may push. The right answer is that it is a defensible choice, clearly
labelled, and that WCB-R would be the natural robustness check.

### 3.6 Event study — `run_did.py:268-334`

`EVENT_STUDY_RHS` replaces `C(POLICY_PERIOD)` with `C(YEAR)`, giving one coefficient per
year with the earliest year as the implicit reference (`ref_year` is computed as the panel
minimum, `run_did.py:644`).

The value is diagnostic. Policy-period dummies assume the effect switches on at a boundary
and stays flat. Year dummies show *when* trends actually move, which lets you see whether a
"policy effect" is really a pre-existing trend that started before the policy did. Plots
annotate the period boundaries from `_PERIOD_BREAKS` (`run_did.py:90`).

### 3.7 Heterogeneity splits and clustering — `run_did.py:690-731, 507`

The model is refit on subsamples: mains-gas versus non-gas, owner-occupied versus private
rented. `clustering_from_random_effects` then runs PCA and KMeans over per-authority
estimated effects to group authorities with similar trajectories.

**Uncertain:** I read the signature and call sites of `clustering_from_random_effects` but not
its full body. To describe how many components or clusters it uses I would need to read
`run_did.py:507-585`.

### 3.8 ML models — `scripts/pipeline/train_models.py`

**In:** the imputed CSV. **Out:** metric tables, SHAP figures, serialised models, model cards.

`sample_data` (`train_models.py:73`) does the thing most portfolio projects get wrong, and it
is worth explaining carefully because it is a real credibility point.

The same physical dwelling can be certified several times across years. A row-level random
split would put the same property in both train and test, and the model would be scored partly
on properties it had memorised. So the split is on **`BUILDING_REFERENCE_NUMBER`**: 80% of
unique properties to train, 20% to test (`train_models.py:113-122`).

Crucially, if that column is absent the function **raises** rather than silently falling back
(`train_models.py:91-99`); the row-level split requires an explicit
`allow_row_level_split=True` and emits a `UserWarning`. That is a fail-loud default over a
correctness-critical detail.

Only *after* the property split is the training set down-sampled to `TARGET_N = 200_000`
rows, stratified on `BELOW_C_INT` (`train_models.py:124-132`). The test set is not
down-sampled. Sampling after the split preserves the leakage guarantee.

Models: `Ridge`, `Lasso`, `LinearRegression`, `GradientBoosting` for regression;
`LogisticRegression` and `XGBClassifier` for classification. `shap.TreeExplainer` on the
gradient-boosted model produces `figure_5_2_shap_summary.png`.

**If removed:** the analysis loses its predictive component, but the econometrics stand alone.

### 3.9 Validation — `src/ew_housing_energy_impact/validation.py`

Range and coverage checks, not schema inference:

| Check | Bound | Line |
|---|---|---|
| Required columns present | — | `:56` |
| `YEAR` within expected range | — | `:63` |
| `CURRENT_ENERGY_EFFICIENCY` | `[1, 100]` | `:67` |
| `CO2_EMISSIONS_CURRENT` | `[0, 30]` | `:73` |
| `TOTAL_FLOOR_AREA` | `[20, 1000]` | `:79` |
| All 14 analysis authorities present | — | `:22` |

Results are written as JSON and registered. These bounds are **hand-chosen domain
assumptions**, not derived from the data, and the file does not explain where they came from.

### 3.10 Warehouse ingestion — `src/ew_housing_energy_impact/ingestion/`

The newest subsystem, built this month and reviewed task by task. Four modules:

- **`epc_source.py`** — download, validation, extraction, period parsing, source enumeration.
- **`s3_upload.py`** — uploads to stable, period-keyed object paths, skipping files whose
  content hash is unchanged.
- **`load.py`** — generates raw-table DDL from the CSV header, runs `TRUNCATE` then
  `COPY INTO`, and reconciles row counts. Contains **zero storage references**, which is why
  the pending Azure migration barely touches it.
- **`pipeline.py`** — wiring, stage preflight, post-load authority coverage.

Three design decisions here are worth being able to defend:

**Raw tables are all `STRING` and take every column.** EPC ships `NO DATA!`, `INVALID!`,
empty numerics and inconsistent dates. Typing at load would *reject rows*, which is exactly
what a faithful bronze copy must not do. Casting happens downstream where a failed cast
becomes a testable NULL rather than a lost row.

**The load is a full refresh (`TRUNCATE` then `COPY INTO`, with `FORCE = TRUE`).** Snowflake's
`COPY` load metadata expires after 64 days, so relying on skip-already-loaded is correct for
64 days and then silently duplicates rows. Since EPC archives are complete period snapshots,
a full refresh is both correct and deterministic. `FORCE = TRUE` removes any dependence on the
metadata semantics — safe because the target is already empty.

**Columns bind by name, not position.** Every column is `STRING`, so a reordered export would
load "successfully" with postcodes in the rating column and nothing objecting. `PARSE_HEADER`
plus `MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE` prevents that.

**Row counting parses rather than counting newlines** (`load.py:local_row_count`). Free-text
columns can contain newlines inside quoted values; a line count would exceed the true record
count and fail reconciliation on a load that was actually correct.

### 3.11 Serving — `api.py`, `app.py`

`api.py` exposes 11 FastAPI routes (`/health`, `/trends`, `/authorities`, `/policy-summary`,
`/drift/latest`, `/model/metadata`, …). Its notable property is **graceful degradation**: it
reads from the analytical database when available and falls back to CSV/artifact files
otherwise, returning a typed `EmptyStateResponse` rather than a 500 when neither exists.

`app.py` is a 94-line Streamlit dashboard reading the same artifacts.

### 3.12 Planned but not implemented — the Azure migration

Object storage was decided to move from AWS S3 to Azure ADLS Gen2. The design is written and
approved (spec §0b, decisions D17–D22) and a four-task implementation plan exists
(`docs/superpowers/plans/2026-08-12-azure-blob-storage.md`), but **no Azure code has been
written**. `ingestion/s3_upload.py` is still what ships.

The driver was a skills decision rather than a technical one — Azure is the platform to
demonstrate — and the timing argument is the interesting part: a Snowflake account is pinned
to one cloud and region at creation and cannot be moved, so switching before signing up costs
a few hours of code, while switching afterwards costs a new Snowflake account.

The blast radius was measured rather than estimated. `git grep` returns 17 files mentioning
AWS, but only one contains cloud-specific *logic*: `s3_upload.py`, 82 lines. `load.py` — which
holds the DDL generation, `COPY INTO`, reconciliation, header diffing and quote-aware counting
— has **zero storage references**, and `epc_source.py` does not appear in the grep at all. The
reconciliation design, the post-load authority check and by-name column binding all carry over
untouched.

That portability was not foresight. It falls out of D12 — inject clients, never construct
them — which was chosen for testability and happens to make the storage layer swappable.
Worth saying plainly if asked, rather than claiming credit for planning it.

Four things genuinely change:

| Concern | S3 | Azure |
|---|---|---|
| Content comparison | MD5 stored in object metadata, because S3's ETag stops equalling the file MD5 above the 8 MB multipart threshold and certificates are ~122 MB | Native `content_md5` blob property — the workaround disappears |
| Snowflake trust | IAM role assumption with an external ID, guarding the confused-deputy problem | Entra ID consent plus a Storage Blob Data Reader role assignment; no external ID, so that explanation no longer applies |
| Uploader auth | boto3 credential chain | `DefaultAzureCredential`, so no long-lived secret enters the repo |
| Identifier hygiene | `${AWS_ACCOUNT_ID}` rendered from the environment | `${AZURE_TENANT_ID}` on the same mechanism |

Blocked on an Azure subscription: signing in is not enough, and resource creation fails
without one.

**A related piece of history worth knowing.** A real 12-digit AWS account id was committed in
the storage-integration DDL and repeated in the spec and plan. Because the repository is
public and nothing had been pushed, the commits were rewritten to purge it — only the
unpublished range, so the already-published commit stayed untouched and `origin/main` remained
an ancestor, meaning a normal push still works with no force. The pre-rewrite state is kept on
tag `backup/pre-purge-20260812-090954`. Every commit SHA after that point changed, which is
why `HANDOFF.md` tells readers to match commits by subject rather than hash.

---

## 4. Concepts you need to be able to explain

**Panel fixed effects (`linearmodels.PanelOLS`).** Data indexed by entity and time.
`EntityEffects` absorbs everything time-invariant about each authority, so estimates come from
within-authority change. *Why here:* authorities differ enormously in housing stock and
geography, and those differences would otherwise confound the policy comparison.
*Alternatives:* pooled OLS ignores the structure and is biased by between-authority
differences; random effects is more efficient but assumes effects uncorrelated with the
regressors, which is implausible here; true difference-in-differences needs a control group
that never received the policy, which does not exist for national policy.

**Wild cluster bootstrap.** Resample by flipping the sign of each cluster's residuals rather
than resampling rows, then refit many times to build an empirical distribution of the test
statistic. *Why here:* 14 clusters. *Alternatives:* asymptotic clustered SEs over-reject
below ~30 clusters; cluster-robust with a small-sample correction helps but not enough; a
pairs cluster bootstrap is possible but performs worse with few clusters.

**Rademacher weights.** Each cluster gets +1 or −1 with equal probability. With 14 clusters
there are only 2¹⁴ = 16,384 distinct weight vectors, which is a real constraint on how fine
the p-value grid can be — worth mentioning, because B=999 is well inside that space but the
discreteness is a known limitation of Rademacher weights with few clusters.

**Linear probability model.** `BELOW_C_INT` is binary but fitted by OLS. *Why:* coefficients
are directly interpretable as percentage-point changes, and fixed effects in logit suffer the
incidental-parameters problem. *Cost:* fitted values can fall outside [0,1] and errors are
heteroskedastic by construction.

**SHAP (`shap.TreeExplainer`).** Game-theoretic attribution of a prediction to its features,
with per-observation values rather than a single global importance. *Alternatives:* built-in
tree importance is biased toward high-cardinality features; permutation importance is global
only.

**Total variation distance** (`monitoring/drift.py:49`). Half the sum of absolute differences
between two categorical distributions; 0 identical, 1 disjoint. *Why:* bounded, interpretable,
no distributional assumptions. *Alternatives:* PSI is the industry convention but its
thresholds are folklore; KL divergence is unbounded and undefined on zero-probability
categories.

**Snowflake storage integration.** A warehouse-side object holding a cloud trust
relationship, so no credentials appear in SQL. On AWS this is IAM role assumption with an
external ID; the external ID exists to prevent the *confused deputy* problem, where a third
party trusted by many customers is tricked into accessing a bucket on behalf of the wrong
one.

**Medallion architecture.** Bronze (raw, faithful), silver (cleaned), gold (business-level).
The value is that bronze is re-derivable ground truth: if cleaning logic is wrong, you fix
and rerun rather than re-ingesting.

---

## 5. Data handling

### The geography rollup — the most important transformation

EPC publishes at lower-tier and unitary authority level. The analysis needs 14 stable South
West authorities. `config.py:46` defines `SOUTH_WEST_EPC_AUTHORITIES`, mapping each GSS code
to the analysis authority it rolls up to, and `analysis_authority_for` (`config.py:285`)
resolves by code with a label fallback.

This is genuinely hard because **local government reorganised during the study period**:
Bournemouth, Poole and Christchurch became BCP in 2019; Dorset districts merged into a
unitary; Somerset districts became a unitary in 2023. `tests/test_geography.py` parametrises
19 code/label pairs covering exactly these transitions.

Two invariants are enforced by test: only `E06`/`E07` codes are used, never `E10` county codes
(counties have no EPC bulk file), and there must be exactly 14 analysis authorities.

Commit `5319ee9` fixed a bug where district rows were being dropped entirely. Any dataset
produced before that commit carries the bug — which is why `data copy/processed/` is flagged
as unusable in spec §0.

### Missing data

`impute_missing` (`build_features.py:42`) fills categoricals with `"Unknown"` and numerics
with the column median, **after** writing a `<col>_MISSING` indicator. The indicator is what
makes this defensible: median imputation alone would let the model treat an imputed value as
observed.

**Assumption baked in:** missingness is treated as informative but its *mechanism* is never
modelled. If EPC assessors systematically omit floor area for a certain property type, the
median fill plus indicator captures that a value was missing, not why.

### Deduplication

**Uncertain.** I found no explicit deduplication step in `clean_epc.py` or
`build_features.py` — no `drop_duplicates` call in either. The same dwelling legitimately
appears multiple times (one row per certificate), and the ML split handles that correctly via
`BUILDING_REFERENCE_NUMBER`. But whether *exact duplicate certificates* exist in the source
and are removed anywhere, I could not confirm. To check I would grep the full pipeline for
`drop_duplicates` and inspect `LMK_KEY` uniqueness on a real extract.

### Policy period boundaries — a deliberate, documented choice

`scripts/pipeline/policy.py` maps year to period: Pre-GreenDeal 2008–2012, GreenDeal-ECO1
2013–**2015**, ECO2 2016–2018, MEES 2019–2020, Post-Strategy 2021–2025.

ECO2 actually launched in April 2015, so 2015 arguably belongs to ECO2. It is deliberately
left in GreenDeal-ECO1 because the data is annual and cannot resolve a within-year
transition. `CLAUDE.md` marks this as a locked decision. **This is a good interview answer:**
it shows you understood that a mid-year policy change cannot be cleanly assigned at annual
granularity, and chose one side explicitly rather than silently.

---

## 6. Engineering decisions and trade-offs

**Filesystem-as-interface.** Simple and debuggable; no schema contract and duplicated path
construction. `api.py` and `app.py` rebuild paths instead of importing `config.py`, violating
the project's own rule.

**Chunked reads everywhere.** Necessary at national scale; makes any operation needing a
global view (deduplication, global statistics) awkward.

**Optional services default off.** `settings.py` gates the database and MLflow behind
`ENABLE_DATABASE` / `ENABLE_MLFLOW`, and every call site degrades to CSV. This is why the
suite runs with no Postgres and no MLflow. Cost: two code paths, and the fallback path is the
one actually exercised by tests.

**Prefect wrapping existing scripts.** `orchestration/tasks.py` wraps each `main()` in a
`@task`. Cheap to add; the tasks are coarse, so a failure mid-`run_did` restarts the whole
stage.

**Airflow is a tombstone.** `dags/etl_epc_pipeline.py` deliberately implements no DAG and sets
`AIRFLOW_DAG_IMPLEMENTED = False`, so a placeholder cannot be mistaken for working
orchestration. Honest, and worth mentioning.

### Known limitations and debt

1. **The download path does not work.** Guarded with `NotImplementedError`; migration to the
   GOV.UK Bearer API is unstarted.
2. **Nothing has run against real cloud services.** All 129 tests are offline — mocked HTTP,
   fake object-store client, fake warehouse cursor.
3. **`--mode la` / `--mode year` are vestigial.** Still accepted by three subcommands
   (`cli.py:231,233,280`); the help text now says so, but they change nothing.
4. **Reconciliation scope mismatch.** `COPY INTO` loads every period ever staged while
   expected counts come from local disk, so pruning local extracts breaks reconciliation.
   `assert_local_covers_stage` now detects it, but the underlying mismatch is unresolved.
5. **The `scripts/` vs `src/` seam** and its three `sys.path` inserts.
6. **Validation bounds are unexplained.** `[20, 1000]` m² is plausible but undocumented.
7. **`a15b3c7` was never code-reviewed** — the `db/`, `ml/`, `monitoring/`, `orchestration/`
   subsystems landed on `main` unreviewed. `HANDOFF.md` records this.

### What I would change with more time

Move the analysis modules into the package and delete the `sys.path` inserts. Introduce an
explicit schema contract between stages (pandera or dbt tests) instead of convention. Add
WCB-R as a robustness check alongside WCB-U. Derive validation bounds from the data and
document them. Finish the warehouse migration so the econometrics read from the warehouse
rather than a CSV.

---

## 7. Honest metrics audit

**This section matters most for an interview. Read it before quoting any number.**

### Numbers the code and artifacts actually support

| Claim | Evidence |
|---|---|
| 129 tests pass; ruff clean; `make smoke` green | Reproducible now: `make ci && make smoke` |
| 14 South West analysis authorities | `config.py:249-275`, enforced by `tests/test_geography.py:34` |
| 19 code/label rollup cases covered | `tests/test_geography.py:39-68` |
| `certificates.csv` has 93 columns; `recommendations.csv` has 7; headers byte-identical across 9 monthly archives | Spec §0, verified by hashing real archives |
| Wild cluster bootstrap uses B=999, Rademacher weights, seed 42 | `run_did.py:127-128` |
| ML training down-samples to 200,000 rows after a property-level split | `train_models.py:57,124-132` |
| Drift thresholds: numeric 0.20, missingness 0.10, categorical 0.20, count 0.30, authority coverage 0.10 | `monitoring/drift.py:17-21` |
| Validation bounds: efficiency [1,100], CO₂ [0,30], floor area [20,1000] | `validation.py:67,73,79` |

### Real results — produced 2026-08-12 by an actual pipeline run

The pipeline was run end to end on **real EPC data** (nine national monthly archives,
2025-01 to 2025-09), bypassing the guarded download by extracting the local archives directly.
These numbers are reproducible from `reports/`.

**Scale, evidenced by the run itself:**

| Fact | Value | Evidence |
|---|---|---|
| National rows merged | 1,376,757 | `data/processed/ew_epc_core.csv`, 396 MB |
| South West rows after rollup | 126,368 (9.2%) | `ewhei clean` output |
| Rows in the modelling dataset | 125,111, 43 columns | `reports/artifacts/validation_report.json` |
| Analysis authorities present | **14 of 14** | same file, `geography.present` |
| Validation issues | **zero** | same file, `"ok": true` |

**Regression — predicting `CURRENT_ENERGY_EFFICIENCY`** (`reports/tables/regression_model_performance.csv`):

| Model | R² test | RMSE test | MAE test |
|---|---|---|---|
| MLP | **0.532** | 9.07 | 6.10 |
| MLP (tuned) | 0.529 | 9.10 | 6.12 |
| GBR (tuned) | 0.519 | 9.19 | 6.16 |
| GradientBoosting | 0.502 | 9.36 | 6.28 |
| RandomForest | 0.483 | 9.53 | 6.37 |
| Ridge | 0.461 | 9.73 | 6.69 |
| LinearRegression | 0.461 | 9.73 | 6.69 |
| Lasso | 0.460 | 9.75 | 6.69 |

**Classification — predicting `BELOW_C`** (`reports/tables/classification_model_performance.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | **0.763** | 0.704 | 0.729 | 0.716 | **0.849** |
| GradientBoosting | 0.754 | 0.695 | 0.715 | 0.705 | 0.839 |
| LogisticRegression | 0.754 | 0.713 | 0.672 | 0.692 | 0.836 |
| RandomForest | 0.746 | 0.694 | 0.681 | 0.688 | 0.822 |

Also produced: SHAP summary, GBR feature importance, XGB confusion/ROC, five EDA figures,
three serialised models, three model cards, 23 artifacts total.

### Three findings from the run that you should be ready to discuss

**1. `LinearRegression` was numerically broken — diagnosed and fixed (`162a80a`).**
The original run reported R² of −1.13 × 10¹⁵ with RMSE 4.45 × 10⁸, while `r2_train` was a
perfectly normal 0.479. That asymmetry was the clue: the *fit* was fine, the *extrapolation*
was not.

`OneHotEncoder` ran without a dropped level, so every categorical block summed to 1 and the
blocks were mutually collinear and collinear with the intercept — the design matrix measured
**rank 44 of 47 columns**. `LinearRegression` solves such a system by least squares and
returns the minimum-norm solution, whose coefficients reached **3.4 × 10¹¹** with an intercept
of −2.2 × 10¹¹. Those enormous values cancel *exactly* on any row where each block sums to 1.

**Exactly 3 of 25,032 test rows** carried an `AGE_BAND_STD` value unseen in training. Under
`handle_unknown="ignore"` their block encodes as all zeros, breaking the cancellation. Those
same 3 rows were the only ones predicting above 1000 — all at 4.07 × 10¹⁰ against actuals of
52, 77 and 87. **Three rows in twenty-five thousand destroyed the metric.**

Ridge and Lasso were unaffected *only* because regularisation suppresses the null-space
directions; the design was equally degenerate for them. That is why the fix is at the encoder
(`drop="first"`, giving full rank 43 of 43) rather than dropping the offending model.

Post-fix, LinearRegression reports **R² 0.4726 against Ridge's 0.4726** — near-identical,
exactly as theory predicts once the design is well conditioned. Other models moved by less
than 0.005. `tests/test_preprocessor.py` adds 7 regression tests that reproduce the failure at
4.8 × 10¹³ before the fix.

This is strong interview material precisely *because* it was found by running the thing: it is
a real numerical-linear-algebra failure, the diagnosis is evidence-based, and the fix is at the
root cause rather than the symptom.

**2. Construction age bands were never normalised, and EDA mislabelled 9,737 rows — both
fixed (`3351434`).**

EPC records construction age two ways: as a band, and for many newer certificates as a bare
year. `norm_age_band` only rewrote two "onwards" strings and passed everything else through,
leaving **40 levels of which 27 were bare years covering 13,338 rows (10.7%)**. The bulk —
12,367 rows for 2022–2025 — had no band at all, because the published vocabulary stops at
"2012 onwards".

Separately, `simplify_age_band` tested `"1900" in s` before `"1900-1929"`, making the
1900-1929 branch unreachable: **9,737 rows were plotted as "Pre-1900"** in every EDA figure.

Both now derive from one ordered `AGE_BANDS` table with numeric bounds and an open-ended
"2022 onwards" band. Result: **40 levels → 13**, no level below 4,225 rows, and **zero test
rows carrying a category unseen in training — down from 3**. Those 3 were exactly the rows
that triggered finding 1, so this removes that bug's upstream cause.

**The cost is real and worth stating:** linear-model R² fell ~0.011 (Ridge 0.473 → 0.461).
Collapsing an exact build year into a band discards granularity. That granularity existed for
only 10.7% of rows — the feature's resolution varied row by row, which is its own problem —
but the loss is not free. Retaining it properly would mean a separate numeric
construction-year feature with a missing indicator.

**3. RandomForest overfits visibly.** R² of 0.657 on train against 0.486 on test, the widest
train/test gap in the table. The gradient-boosted and MLP models generalise better.

**4. The fixed-effects analysis cannot run on this data, and fails opaquely.** Verified by
running it: `ewhei fe` dies with `TypeError: only length-1 arrays can be converted to Python
scalars` inside `linearmodels/shared/linalg.py:27`. The cause is that with a single year and a
single policy period, `C(POLICY_PERIOD)` collapses to one level, so the design matrix contains
more than one all-ones column and `has_constant` cannot resolve a single constant index.

That is correct behaviour in the sense that the model is unidentified — but **the pipeline has
no guard for insufficient temporal variation**, so the user sees a library-internal `TypeError`
rather than "this analysis needs more than one policy period". That is a real defect and a fair
thing to volunteer as something you would fix.

### Claims still NOT substantiated

- **Any policy-period coefficient, p-value, or event-study estimate.** The econometrics — the
  dissertation's headline result — have not been produced. They need 2008–2025, which requires
  the backfill.
- **"~1.8M EPC records analysed."** The actual run analysed **125,111** South West rows from
  1,376,757 national. The 1.8M figure refers to a full 2008–2025 extract that does not exist
  here.
- **The wild cluster bootstrap has never executed on real data.** Its correctness is supported
  by code review only.
- The values in `examples/demo_data/` remain synthetic fixtures. Interestingly they were
  *plausible* — the fixture said GradientBoosting R² 0.51 against a real 0.503, and XGBoost
  ROC-AUC 0.84 against a real 0.849 — but they were invented, and the real ones above are what
  you should quote.

### How to talk about this now

You can say, accurately: *"I ran it end to end on 1.38M real EPC certificates, filtered to
125,111 South West records across all 14 analysis authorities with zero validation issues.
Best regression R² was 0.532 predicting efficiency score; XGBoost reached 0.849 ROC-AUC
classifying below-band-C properties, using a property-level split to prevent leakage. The
panel econometrics need the full 2008–2025 history, which I haven't backfilled yet."*

Every number in that sentence is in `reports/`.

---

## 7b. Defects found by reviewing the unreviewed subsystems

The `db/`, `ml/`, `monitoring/`, `orchestration/` and FastAPI code — roughly 2,400 lines —
reached `main` on 2 August without ever being code-reviewed. Reviewing it produced nine
findings. Six were fixed (`dccaf1c`); three minor ones were left.

**Five of the six were silent: the system reported success while producing wrong output.**

### The two that mattered

**A NaN local authority was written to the database as the literal string `'nan'`**
(`db/loaders.py`). `groupby(..., dropna=False)` keeps NaN groups and `str(NaN)` yields
`"nan"`, so a missing authority became a row in the serving table and would have surfaced in
`/authorities`, in `/trends`, and on the Streamlit dashboard as though it were a real local
authority. Verified by running it: two rows in, two rows out, one with `authority='nan'`.
The same defect existed a second time in `load_policy_period_summary`, found by grepping for
other `groupby` calls rather than assuming the first was unique.

**Drift detection was blind to every proportion column** (`monitoring/drift.py`). Relative
shift used `max(abs(ref_mean), 1.0)` as its denominator, and that `1.0` floor turned a
*relative* shift into an *absolute* one for any column with a mean below 1 — which is every
rate in the project, including `BELOW_C`, one of the two model targets. `BELOW_C` moving
0.40 → 0.55 is a 37.5% change; it reported 0.150 against a 0.20 threshold and raised no
alert. The component whose entire job is noticing that data changed could not notice.

### The other four

| Defect | File | Failure mode |
|---|---|---|
| NaN year → `cannot convert float NaN to integer`, naming no column or row | `db/loaders.py` | Loud but uninformative |
| Prefect no-op fallback never announced itself, so a broken install ran as a plain sequential loop with no retries or observability and still reported success | `orchestration/tasks.py` | Silent |
| `pd.read_csv(usecols=[])` returns a `(0,0)` frame, so a mistyped column name surfaced as "no data" rather than an error | `api.py` | Silent |
| A 56 MB CSV header was re-read on every call, with no caching | `api.py` | Degrades under load |

Fixes: grouping keys are validated up front and fail naming the column and affected row
count; relative shift divides by the actual mean, falling back to absolute only at a zero
baseline; `PREFECT_AVAILABLE` is exposed and carried in the flow result; a total column miss
raises naming missing and available columns; column lookups are cached on `(mtime, size)` so
a rewritten file still invalidates.

Regression tests live in `tests/test_db_loader_integrity.py`, `tests/test_drift_sensitivity.py`
and `tests/test_api_and_orchestration_robustness.py` — twenty tests, each confirmed failing
before its fix.

### Left unfixed, deliberately

A hardcoded `0.10` class-balance threshold in `retraining.py` while its siblings are
parameters; `setup_mlflow()` re-running `set_tracking_uri` and `set_experiment` on every
logged metric; and `get_trends`/`get_policy_summary` being unbounded while `get_artifacts`
takes a limit. None can produce wrong output.

One suspicion was investigated and **dismissed**: `recommend_retraining`'s new-data-volume
trigger looked like it would fire permanently on a growing dataset, but reference and current
default to the same file, so it only fires on a genuine increase. That is the intended
behaviour, not a defect.

---

## 7c. What actually found the defects

Across the whole period, fifteen defects were found. Their discovery methods:

| Method | Defects |
|---|---|
| Reading the code | 7 |
| Code review by someone who did not write it | 4 |
| **Running the pipeline on real data** | 2 |
| Spec self-review | 1 |
| Plan pre-flight scan | 1 |
| **The passing test suite** | **0** |

**None were caught by tests.** The suite was green at every commit, including the commits
that introduced and shipped each defect.

The drift bug shows why, and it is worth being able to explain. The existing test moved
`BELOW_C_INT` from 0.0 → 1.0 — the most extreme change possible. Under both the correct and
the broken formula that yields 1.000 and alerts. **The test input has zero power to
discriminate between working and broken code.** A realistic 0.40 → 0.55 separates them
immediately. Likewise `tests/test_db.py` had three tests on clean hand-built fixtures and not
one NaN, so the `'nan'` corruption could never have failed them.

The pattern: fixtures were cleaner and more extreme than reality. Real data contained things
nobody would think to write into a fixture — 27 distinct bare-year age bands, three test rows
carrying a category unseen in training, embedded newlines in roughly 28% of lines, and a
one-hot encoding that was rank-deficient by exactly three. Extreme values feel like rigorous
testing but are often precisely where a bug is invisible; the interesting failures live in the
messy middle.

An attempt to quantify this by measuring how many tests exercise edge-case input returned 37%
for the original suite against 38% for the tests added later — no meaningful difference. The
metric was too crude to support the claim, so it is recorded here as a failed measurement
rather than dressed up.

**The honest summary for an interview:** the test suite was green through fifteen defects,
five of which silently corrupted output. What found them was running on real data and having
the code read by someone who had not written it.

---

## 8. Uncertainties

Things I could not confirm from reading, and what I would check:

1. **Deduplication.** No `drop_duplicates` found in the cleaning or feature stages. Would need
   to grep the full pipeline and check `LMK_KEY` uniqueness on a real extract.
2. **`clustering_from_random_effects` internals** (`run_did.py:507-585`) — I read the
   signature and call sites, not the body. Number of PCA components and KMeans clusters
   unknown.
3. **Whether the event-study reference year is always 2008.** It is computed as the panel
   minimum (`run_did.py:644`), so it depends on the data actually loaded.
4. **`report_eda.py` and `quality_report.py` outputs** — I mapped their existence but did not
   read them in detail.
5. **Real EPC sentinel-value frequency.** The design treats `NO DATA!` and `INVALID!` as
   meaningful, but I have not measured how often they occur.
