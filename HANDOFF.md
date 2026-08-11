# HANDOFF

Written so a cold-start session can resume without re-deriving context.
Update at every checkpoint: task complete · milestone done · stop signalled · blocker ·
context ~70% · 30+ min since last update.

**Last updated:** 2026-08-11

---

## Current State

**Warehouse migration Phase A: implementation COMPLETE and reviewed. Never run end to end
against real AWS/Snowflake/EPC services — that is the next action for whoever has the
credentials.**

- **Branch:** `feat/warehouse-ingestion`, not pushed, not merged.
- **Spec:** `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md` — approved,
  amended 2026-08-11 (see spec §0) after the EPC API moved to GOV.UK mid-implementation.
- **Docs:** `docs/ingestion.md` — what it does, how to run it, S3 layout, the design rationale
  (why raw is all-STRING, why sentinels are preserved, why columns bind by name, why the load
  is a full refresh, why row counting parses instead of counting lines), and a failure-mode
  table keyed to spec §6.
- Tasks 1–6 delivered: `src/ew_housing_energy_impact/ingestion/` (`epc_source.py`,
  `s3_upload.py`, `load.py`, `pipeline.py`), `snowflake/` (`01_setup.sql`,
  `02_storage_integration.sql`, `03_stage_and_format.sql`, `README.md`), and the `ewhei ingest
  [--dry-run] [--allow-schema-change]` CLI command.
- Task 7 (this one) is documentation and handoff only — no production code touched.

The plan is a four-phase migration off the pandas/Prefect pipeline onto S3 → Snowflake →
dbt → Airflow, with an AI lane redesigned around EPC's actual data (EPC has no free text,
so sentiment analysis and review-RAG were dropped as inapplicable). Phase A (this work) is
ingestion only. Phases B (dbt), C (Airflow cutover), D (AI) each need their own spec first —
none of those specs exist yet.

Read the spec before doing anything else — it records why each decision was made, including
the ones that went against recommendation, and §0 records the mid-implementation API move
that reshaped several of those decisions.

### What "complete" does and does not mean here

Phase A code is complete and has been through a review-and-fix round (see
`.superpowers/sdd/2026-08-11-epc-warehouse-ingestion/task-6-report.md`). All 121 tests pass
offline — mocked HTTP, fake boto3 client, fake Snowflake cursor. **None of it has been run
against a real AWS account, a real Snowflake account, or a real bearer token.** Offline tests
prove the logic is internally consistent: `preflight_stage` raises on an empty stage,
`reconcile` raises on a count mismatch, `assert_authority_coverage` raises on a missing
authority. They do not prove `MATCH_BY_COLUMN_NAME` behaves as assumed against a real
Snowflake table, that the storage integration's IAM trust actually authenticates, or that
reconciliation matches on a real 2.92 GB EPC file rather than a small hand-built fixture. Say
this plainly to whoever picks this up next — do not let "121 passing" read as "verified
end to end."

### Earlier state (still true)

Onboarding complete, merged, and pushed.

- **Branch:** `main`, in sync with `origin/main` at `65f506b`.
  `chore/onboarding-conventions` was fast-forward merged and the local branch deleted.
  The remote branch `origin/chore/onboarding-conventions` still exists and is now
  redundant — safe to delete.
  - `a15b3c7` platform layers (the pre-existing uncommitted build-out)
  - `8e66fdd` working conventions + session-tracking scaffolding
  - `65f506b` HANDOFF sync
- **Excluded from the repo on purpose:** `claude_onboarding.md` is still untracked.
  It is a personal operating-manual prompt, not project code, and the repo is **public**.
  Add it deliberately if you want it published.
- `CLAUDE.md` extended with operating rules, the five new subsystems, and corrected
  environment/Docker/testing facts.
- Scaffolding created: `tasks/todo.md`, `tasks/lessons.md`, `tasks/agent_memory.md`,
  `HANDOFF.md`, `docs/plans/`, `docs/specs/`.
- **Fixed:** the editable install was pointing at `/Users/divyanshu/Desktop/EPC`, a path that
  no longer exists, so `ewhei` was dead. Re-ran `pip install -e .`; CLI verified working.

**The working tree carries substantial uncommitted work that predates this session** —
24 modified files, 10 new docs, and five new subsystems (`db/`, `ml/`, `monitoring/`,
`orchestration/`, `settings.py`). This was authored before onboarding and has not been
reviewed or committed. It is all green against the quality gate.

---

## Next Action

**Phase A (ingestion) is done. Phase B (dbt) is next, and needs its own spec before any
code.** In order:

1. **Human action, blocking:** obtain `EPC_BEARER_TOKEN` from the account page on
   `get-energy-performance-data.communities.gov.uk`. Nothing that touches the real API can
   run without it — not `ewhei download`, not the manual verification in
   `.superpowers/sdd/2026-08-11-epc-warehouse-ingestion/task-7-brief.md`.
2. **Manual verification of Phase A** (needs the token plus AWS/Snowflake credentials —
   deliberately outside CI, see that same brief): create the S3 bucket, apply
   `snowflake/01`–`03`, run `ewhei ingest --dry-run` then `ewhei ingest` twice and confirm
   row counts don't double the second time. Nobody has done this yet.
3. **2008–2024 backfill (R8/R9, carried forward, still open).** Only 2025-01…09 monthly
   archives exist locally, in `data copy/`. The full backfill needs the bearer token plus the
   API's yearly endpoint (`domestic-<year>.zip`, ~2.92 GB compressed per the national split,
   15–25 GB estimated unzipped for the full range) and is otherwise unscoped — R9 is still
   open.
4. **Known trap for the next person who runs `ewhei download`:** `download_one_file` (in
   `ingestion/epc_source.py`, re-exported by `epc_download_merge_fast.py`), plus
   `list_available_files` and `pick_year_from_name` (defined directly in
   `epc_download_merge_fast.py`), still hardcode the dead `epc.opendatacommunities.org` host.
   Nothing in Phase A exercises `ewhei download` end to end, so this was never forced to
   surface. See `tasks/lessons.md`, 2026-08-11 entry.
5. Once 1–3 are done, invoke `writing-plans` for **Phase B (dbt staging + marts)** against a
   new spec — none exists yet. Phase B replaces `clean_epc.py`, `build_features.py`, and
   `validation.py`'s job, not the CLI commands themselves (Phase C does that cutover).
   Phase C (Airflow cutover) and Phase D (AI lane) each need their own spec in turn — do not
   skip ahead to either.

**Carry-over risk, unchanged from before:** the platform build-out in `a15b3c7` (on `main`)
has still never been code-reviewed. It passes lint and tests, which is not the same as
having been read. Retrospective review of `db/`, `ml/`, `monitoring/`, `orchestration/` is
still the highest-value task if anything in those subsystems misbehaves.

---

## In-Flight Files

None — no edit is half-finished. Task 7 (this session) touched only
`docs/ingestion.md` (new), `CLAUDE.md`, `HANDOFF.md`, `tasks/todo.md`, `tasks/lessons.md` —
documentation only, no production code.

---

## Open Questions

1. **Retrospective review of `a15b3c7`** — merged unreviewed (see Next Action)
2. **Python version split** — local is Anaconda 3.13.5, CI pins 3.11. Leave as-is and treat
   CI as authoritative, or align local to 3.11 via a dedicated venv? Currently no failures
   trace to this, but it is a latent divergence. Relevant again for Phase B: R3 flags that
   `dbt-core` has historically lagged the newest Python.
3. **Stale knowledge graph** — the `code-review-graph` index was 4 subsystems behind as of
   the last check (built at `96ce4e3`). It has not been rebuilt since; check its
   last-indexed commit against `git log -1` before trusting it for Phase B impact analysis.
4. **Pre-existing dependency conflicts** — `fastmcp` wants `websockets>=15.0.1` (have 13.1),
   `typer-slim` wants `typer>=0.23.1` (have 0.17.5). Unrelated to this project and harmless
   to the suite, but they surface on every `pip install`.
5. **Phase A has never run against real services.** Every one of the 121 tests is offline.
   None of it proves `MATCH_BY_COLUMN_NAME` behaves as assumed, that the storage integration
   authenticates, or that reconciliation matches on real EPC files. See "What 'complete'
   does and does not mean here" above and `docs/ingestion.md`.

---

## Verification Baseline

Last run 2026-08-11, end of Task 6's review-fix round — all green:

```
ruff check .            -> All checks passed!
ruff format --check .   -> 61 files already formatted
python -m pytest        -> 121 passed, 948 warnings in 4.04s
make smoke              -> ewhei --help + 4 dry-run paths OK (added: ingest --dry-run)
```

Reproduce with `make ci` then `make smoke`. The 948 warnings are pre-existing deprecation
noise from `shap`, `pydantic.v1`, and `prefect` — not regressions. Test count grew from 68
(2026-08-02 baseline) to 121 across Tasks 1–6: `test_epc_source.py`, `test_s3_upload.py`,
`test_load.py`, `test_ingestion_pipeline.py`, `test_cli.py` are new; all offline.

**Note:** a green `pytest` does not prove the CLI works — `tests/conftest.py` patches
`sys.path`. Always pair it with `make smoke`.
