# HANDOFF

Written so a cold-start session can resume without re-deriving context.
Update at every checkpoint: task complete · milestone done · stop signalled · blocker ·
context ~70% · 30+ min since last update.

**Last updated:** 2026-08-02

---

## Current State

**Warehouse migration Phase A: design approved, implementation PAUSED at user's request
(2026-08-02). Do not auto-resume — wait for the user to pick it up.**

- **Branch:** `feat/warehouse-ingestion`, one commit (`794dabe`), **not pushed, not merged.**
- **Spec:** `docs/superpowers/specs/2026-08-02-epc-warehouse-ingestion-design.md` — approved.
- **Nothing implemented.** No code written, no data downloaded, no cloud resources created.
- Next step when resumed is `writing-plans`, not code.

The plan is a four-phase migration off the pandas/Prefect pipeline onto S3 → Snowflake →
dbt → Airflow, with an AI lane redesigned around EPC's actual data (EPC has no free text,
so sentiment analysis and review-RAG were dropped as inapplicable). Phase A is ingestion
only. Phases B (dbt), C (Airflow cutover), D (AI) each need their own spec first.

Read the spec before doing anything — it records why each decision was made, including the
ones that went against recommendation.

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

**Paused. Nothing to do until the user resumes.**

When they do, in this order:

1. Confirm the four blockers in `tasks/todo.md` are cleared — chiefly `.env` with
   `EPC_EMAIL` / `EPC_API_KEY`, since `data/` is empty and nothing runs without it.
2. Invoke `writing-plans` against the approved spec. Do not skip to code.
3. Snowflake account must be on **AWS / eu-west-2** to match the S3 bucket, or the storage
   integration pays cross-region egress.

**Carry-over risk:** the platform build-out in `a15b3c7` is now on `main` but has **never
been code-reviewed**. It passes lint and tests, which is not the same as having been read.
It landed on an explicit merge instruction, ahead of any review pass. A retrospective
review of `db/`, `ml/`, `monitoring/`, and `orchestration/` is the highest-value next task
if anything in those subsystems starts misbehaving.

---

## In-Flight Files

None — no edit is half-finished.

Everything modified this session is complete: `CLAUDE.md`, `tasks/*`, `HANDOFF.md`,
`docs/plans/README.md`, `docs/specs/README.md`.

---

## Open Questions

1. **Retrospective review of `a15b3c7`** — merged unreviewed (see Next Action)
2. **Python version split** — local is Anaconda 3.13.5, CI pins 3.11. Leave as-is and treat
   CI as authoritative, or align local to 3.11 via a dedicated venv? Currently no failures
   trace to this, but it is a latent divergence.
3. **Stale knowledge graph** — the `code-review-graph` index is 4 subsystems behind
   (built at `96ce4e3`, 75 nodes / 12 files). Rebuild before relying on impact analysis.
4. **Pre-existing dependency conflicts** — `fastmcp` wants `websockets>=15.0.1` (have 13.1),
   `typer-slim` wants `typer>=0.23.1` (have 0.17.5). Unrelated to this project and harmless
   to the suite, but they surface on every `pip install`.

---

## Verification Baseline

Last run 2026-08-02, after the editable reinstall — all green:

```
ruff check .            -> All checks passed!
ruff format --check .   -> 51 files already formatted
python -m pytest        -> 68 passed, 948 warnings in 4.81s
make smoke              -> ewhei --help + 3 dry-run paths OK
```

Reproduce with `make ci` then `make smoke`. The 948 warnings are pre-existing deprecation
noise from `shap`, `pydantic.v1`, and `prefect` — not regressions.

**Note:** a green `pytest` does not prove the CLI works — `tests/conftest.py` patches
`sys.path`. Always pair it with `make smoke`.
