# Lessons

Chronological log. After ANY correction or non-obvious discovery, append an entry:
**what broke** / **root cause** / **what to do next time**. Newest at the bottom.

---

## 2026-08-02 — `ewhei` broken by a repo move; tests hid it

**What broke.** Every `ewhei` invocation failed:

```
File "/opt/anaconda3/bin/ewhei", line 5, in <module>
    from ew_housing_energy_impact.cli import main
ModuleNotFoundError: No module named 'ew_housing_energy_impact'
```

`pip show ew-housing-energy-impact` still reported the package as installed, and
`ewhei` was still on `PATH`, so nothing looked wrong until the CLI was actually run.

**Root cause.** The editable install records an absolute source path:

```
Editable project location: /Users/divyanshu/Desktop/EPC          <- no longer exists
Actual project:            /Users/divyanshu/Desktop/All Projects/EPC
```

The repo was moved into `All Projects/` after `pip install -e .`. The
`__editable__.ew_housing_energy_impact-0.1.0.pth` finder kept pointing at the old
location, so the dist-info survived while the import path did not.

**Why it stayed hidden.** Three separate mechanisms mask it:
- `tests/conftest.py` inserts repo root + `src/` onto `sys.path` → all 68 tests pass.
- Makefile targets (`smoke`, `demo-dry-run`, `validate`) hardcode `PYTHONPATH=src:.`.
- CI installs fresh on a clean runner, so it never reproduces.

So the entire documented entry point was dead while every quality gate stayed green.

**What to do next time.**
- After moving or renaming the repo directory, re-run `pip install -e .` immediately.
- Never treat a green `pytest` as proof the CLI works. `make smoke` is the CLI check.
- When a `ModuleNotFoundError` contradicts `pip show`, check
  `pip show -f <pkg> | grep "Editable project location"` before anything else.
- Reinstalling pulled in ~20 transitive packages (prefect 2.20.25, sqlalchemy 2.0.35,
  psycopg 3.3.4) and reported pre-existing conflicts (`fastmcp` wants websockets>=15.0.1,
  have 13.1; `typer-slim` wants typer>=0.23.1, have 0.17.5). Those conflicts are unrelated
  to this project and did not affect the suite — re-verified 68 passed afterwards.

---

## 2026-08-02 — Code knowledge graph was stale enough to mislead

**What broke.** Nothing yet, but the `code-review-graph` MCP reported 75 nodes across
12 files, built at commit `96ce4e3`. The working tree has ~40 source files.

**Root cause.** The graph was built before the `db/`, `ml/`, `monitoring/`, and
`orchestration/` subsystems were added, and those changes are still uncommitted.

**What to do next time.** Check `list_graph_stats_tool` (node count, last-updated commit)
against `git log -1` before trusting the graph for impact analysis. A stale graph does not
error — it silently returns an answer that omits whole subsystems, which is worse than a
failed query. Rebuild, or fall back to Grep, when the commit doesn't match.
