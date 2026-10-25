# Resume Numbers — EPC South West Project

Verified against the codebase on 2026-09-17 (branch `chore/remove-video-md`, commit `047c533`).
Every number below has a file/function pointer so you can re-check it before using it.
Where the project's own docs disagree with each other, both are noted — verify before quoting.

---

## 1. Scale

| Number | Where |
|---|---|
| **241 tests**, all passing (verified live via `pytest --collect-only`) | `tests/*.py` (22 files) — **note:** `CLAUDE.md` and `PROJECT_DEEPDIVE.md` both say "129 tests." That's stale; 241 is the current count. Recount before quoting. |
| 1,376,757 national EPC rows merged in a real run | `PROJECT_DEEPDIVE.md` §7 / `reports/artifacts/validation_report.json`, produced by `epc_download_merge_fast.py` |
| 126,368 SW rows after geography rollup (9.2% of national) → 125,111 rows / 43 cols in the final modelling dataset | same run, `ewhei clean` + `ewhei features` output |
| 14 of 14 analysis authorities present, **zero validation issues** on that run | `validation_report.json`, checked by `validation.py` |
| ML training downsamples to **200,000 rows** after a property-level train/test split | `scripts/pipeline/train_models.py:57,124-132` |
| `certificates.csv` = 93 columns, `recommendations.csv` = 7 columns, byte-identical headers across 9 real monthly archives | `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md` §0 |
| Wild cluster bootstrap: B=999 replications, Rademacher weights, seed=42 | `scripts/pipeline/run_did.py:127-128` |
| National archive ≈2.92 GB compressed, one file, split by period, regenerated monthly | `CLAUDE.md`, `epc_source.py` |
| Certificate file ≈122 MB, hashed in streaming blocks (not loaded whole) for S3 dedup | `src/ew_housing_energy_impact/ingestion/s3_upload.py:27,62` |

**⚠️ Do not put "1.8M records analyzed" on your resume.** `README.md:3` and `CLAUDE.md:8` both say
"~1.8M," but `PROJECT_DEEPDIVE.md:693-694` explicitly audits this and says it's not substantiated:
the actual run analyzed **125,111** SW rows from **1,376,757** national rows. The 1.8M figure
refers to a hypothetical full 2008–2025 extract that has never actually been downloaded/run.
This is the project's own internal audit calling out its own marketing copy — use the real
numbers above.

---

## 2. Baseline + Improvement

**Model comparison (real run, not synthetic fixtures)** —
`reports/tables/regression_model_performance.csv`, `classification_model_performance.csv`:
- Best regression R² (MLP): **0.532** test vs. linear baseline (Ridge/LinearRegression): **0.461** test
- XGBoost ROC-AUC: **0.849** vs. LogisticRegression baseline: **0.836**
- RandomForest overfit, worth citing honestly: **train R²=0.657 vs. test R²=0.486** — the gap was
  identified and reported rather than just quoting the train number.

**A real bug with before/after numbers** (`train_models.py` preprocessing, `tests/test_preprocessor.py`):
LinearRegression before fix reported **R² = −1.13×10¹⁵**, RMSE 4.45×10⁸ (broken — 3 of 25,032 test
rows had an unseen category, breaking a collinearity cancellation in a rank-44-of-47 design
matrix, producing coefficients of 3.4×10¹¹). After fixing the encoder (`drop="first"`):
**R² = 0.4726**, matching Ridge's 0.4726 almost exactly. 7 regression tests added.

**Age-band cleanup, with an honestly-reported cost** (`build_features.py`): 40 category levels →
13; unseen-category test rows 3 → 0. But Ridge R² **fell 0.473 → 0.461** as a real cost of that
simplification — the project's own doc says "the cost is real and worth stating."

**⚠️ `fast_wcb.py` speedup — do not quote a number here.** `scripts/pipeline/fast_wcb.py` is
untracked (`git status` shows `??`, never committed), has no test file, isn't wired into
`run_did.py` yet, and its own docstring references a `compare_against_reference` function that
doesn't actually exist in the file. It documents the *original* approach's cost (~20s/replication
on a 2.58M-row panel; separately `run_did.py:643` says ~19s/fit on a 2.68M-row panel → B=999×3
DVs ≈ 15.7 hours — note these two docstrings don't even agree on row count or B). There is no
measured runtime for the fast version. If you want a legitimate "Nx faster" resume line, benchmark
it first.

---

## 3. Cost

- Snowflake chosen **over the free Postgres already running in `docker-compose.yml`**, as an
  explicit accepted trade-off: **~$400 for a 30-day trial**, after which ingestion stops working.
  `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md`, decision D2 / risk R1.
- No LLM/token spend anywhere in this codebase — **not found** (this isn't an LLM app; don't
  invent an AI-cost angle).
- One caching decision, framed as reliability not $-cost: `api.py` was re-reading a 56 MB CSV
  header on every single API call; fixed by caching column lookups keyed on `(mtime, size)`. Real
  perf fix, not a dollar saving.

---

## 4. Reliability / Failure Handling

No retry/backoff library anywhere (grepped for tenacity/retry, found nothing) — just a flat
**120s request timeout** (`epc_source.py:32`) and idempotent MD5-based skip-unchanged uploads.
Be precise that "reliability" here means data-integrity guards, not network retries.

Six real, fixed defects (`PROJECT_DEEPDIVE.md` §7b, commit `dccaf1c`, **20 new regression tests**,
each confirmed failing before its fix):

1. **NaN local authority silently written to prod as the literal string `'nan'`** —
   `groupby(dropna=False)` + `str(NaN)`, found identically in two places in `db/loaders.py`;
   would've surfaced as a fake authority on `/authorities`, `/trends`, and the Streamlit dashboard.
2. **Drift monitor blind to every proportion column** — `monitoring/drift.py` used
   `max(abs(ref_mean), 1.0)` as a denominator, silently converting relative shift to absolute for
   any mean-below-1 column. Concretely: `BELOW_C` moving **0.40 → 0.55 (a real 37.5% change)** was
   reported as **0.150 against a 0.20 alert threshold** — no alert fired on the component whose
   entire job is catching drift.
3. Prefect install failure silently degraded to a sequential loop with no retries/observability
   while still reporting success — fixed by exposing `PREFECT_AVAILABLE` in the flow result
   (`orchestration/tasks.py`).
4. `pd.read_csv(usecols=[])` on a mistyped column silently returned a `(0,0)` frame, presenting
   as "no data" instead of an error (`api.py`).
5. Warehouse row-count **reconciliation** checks an independent local count against Snowflake's
   `COPY INTO` result in both directions (missing vs. unexpected kinds) and raises on any
   mismatch — deliberately not trusting Snowflake's own `rows_loaded`, which "would be Snowflake
   checking itself" (`ingestion/load.py:reconcile()`).
6. The EPC API silently 301-redirected and returned HTML with a 200 status, so
   `raise_for_status()` passed cleanly before `.json()` died — root-caused and fixed by switching
   Basic→Bearer auth (`tasks/lessons.md`, 2026-08-11 entry).

---

## 5. One Architecture Decision

**Wild cluster bootstrap instead of standard clustered standard errors**,
`scripts/pipeline/run_did.py:124-229` (function `wild_cluster_bootstrap_pvalues`).

The pitch to a skeptical engineer: clustered SEs are asymptotically justified in the *number of
clusters*, and this panel has 14 (South West local authorities). Cameron, Gelbach & Miller (2008)
show clustered SEs over-reject badly below ~30 clusters — i.e., the standard approach here would
produce false-positive policy effects. Instead: draw a Rademacher (+1/−1) weight per cluster,
refit B=999 times, and **recenter** the bootstrap statistic on the full-sample estimate
(`t* = (β*_b − β̂)/se*_b`) — the docstring explains that skipping recentering drifts p-values
toward 0.5. The p-value formula `(1+count)/(B+1)` guarantees it's never exactly zero.

Two things that show it's real engineering, not textbook copy-paste:
- The panel index has duplicate (LA, YEAR) pairs, so label-based row alignment is impossible; the
  code reconstructs the estimation sample **positionally** via a boolean mask and raises loudly if
  the mask size doesn't match the residual count (`run_did.py:173-180`).
- A single working DataFrame is reused across all 999 iterations, overwriting only the
  dependent-variable column, specifically to avoid copying a ~1.8M-row panel 999 times
  (`run_did.py:194-209,243`).

What makes it *defensible under pushback*: the code picks the **unrestricted** variant (WCB-U)
rather than the **restricted** one (WCB-R) that MacKinnon & Webb's literature generally favors for
very-few-cluster settings — and says so explicitly in its own docstring instead of hiding the
limitation. That's the difference between "I used a bootstrap" and "I understood the tradeoff
space and can defend the specific choice."
