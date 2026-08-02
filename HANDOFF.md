# HANDOFF

Written so a cold-start session can resume without re-deriving context.
Update at every checkpoint: task complete · milestone done · stop signalled · blocker ·
context ~70% · 30+ min since last update.

**Last updated:** 2026-08-02

---

## Current State

Onboarding complete. No feature work in progress.

- **Branch:** `chore/onboarding-conventions` (branched from `main`, nothing committed yet)
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

Nothing is queued. The open decision, when work resumes: **what to do with the large
uncommitted change set on `main`.** It is coherent and passing, but it is one enormous
unreviewed blob. Options are to review and split it into themed commits, or commit it
wholesale as a single "platform build-out" commit. This needs a decision before any new
feature work, otherwise new changes get tangled into it.

---

## In-Flight Files

None — no edit is half-finished.

Everything modified this session is complete: `CLAUDE.md`, `tasks/*`, `HANDOFF.md`,
`docs/plans/README.md`, `docs/specs/README.md`.

---

## Open Questions

1. **The uncommitted change set** — split into reviewed commits, or land as one? (above)
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
