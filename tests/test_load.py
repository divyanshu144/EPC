"""Offline tests for raw DDL, loading, and quote-aware reconciliation."""

from __future__ import annotations

import pytest

from ew_housing_energy_impact.ingestion import load


class FakeCursor:
    """Records statements and replays canned results."""

    def __init__(self, results: list | None = None):
        self.statements: list[str] = []
        self._results = list(results or [])
        self._last = None

    def execute(self, sql: str):
        self.statements.append(sql)
        self._last = self._results.pop(0) if self._results else None
        return self

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last or []


def test_read_header_returns_columns(tmp_path):
    p = tmp_path / "certificates.csv"
    p.write_text("LMK_KEY,POSTCODE,TENURE\nA,B,C\n")
    assert load.read_header(p) == ["LMK_KEY", "POSTCODE", "TENURE"]


def test_generate_raw_ddl_types_every_column_as_string():
    ddl = load.generate_raw_ddl("epc.raw.certificates", ["LMK_KEY", "TOTAL_FLOOR_AREA"])
    assert "CREATE TABLE IF NOT EXISTS epc.raw.certificates" in ddl
    assert '"LMK_KEY" STRING' in ddl
    assert '"TOTAL_FLOOR_AREA" STRING' in ddl
    assert "NUMBER" not in ddl and "FLOAT" not in ddl


def test_generate_raw_ddl_rejects_empty_columns():
    with pytest.raises(ValueError, match="no columns"):
        load.generate_raw_ddl("epc.raw.certificates", [])


def test_generate_raw_ddl_rejects_embedded_double_quote():
    """A quote in a header value must not break out of the quoted identifier."""
    with pytest.raises(ValueError, match="position 1"):
        load.generate_raw_ddl("epc.raw.certificates", ["LMK_KEY", 'BAD"COL'])


def test_generate_raw_ddl_rejects_semicolon():
    """A semicolon in a header value must not be interpolated into DDL unescaped."""
    with pytest.raises(ValueError, match="position 0"):
        load.generate_raw_ddl("epc.raw.certificates", ["BAD;DROP TABLE X"])


def test_generate_raw_ddl_accepts_normal_epc_style_names():
    ddl = load.generate_raw_ddl(
        "epc.raw.certificates", ["LMK_KEY", "TOTAL_FLOOR_AREA", "CURRENT ENERGY RATING"]
    )
    assert '"LMK_KEY" STRING' in ddl
    assert '"CURRENT ENERGY RATING" STRING' in ddl


def test_copy_into_binds_by_name_and_aborts_on_error():
    sql = load.copy_into_sql("epc.raw.certificates", "certificates")
    assert "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE" in sql
    assert "ON_ERROR = ABORT_STATEMENT" in sql
    assert "FORCE = TRUE" in sql
    assert "CONTINUE" not in sql
    assert "SKIP_HEADER" not in sql
    assert "@epc.raw.epc_raw_stage/certificates/" in sql


def test_local_row_count_excludes_header(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\na,b\nc,d\n")
    assert load.local_row_count(p) == 2


def test_local_row_count_handles_header_only(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\n")
    assert load.local_row_count(p) == 0


def test_local_row_count_respects_quoted_embedded_newline(tmp_path):
    """Regression test for counting newlines instead of CSV records.

    D6 keeps every column including free-text *_DESCRIPTION fields, which can
    carry newlines inside quoted values. A naive line count reports 3 here, and
    reconciliation would then fail a load that was actually correct.
    """
    p = tmp_path / "c.csv"
    p.write_text(
        "LMK_KEY,WALLS_DESCRIPTION\n"
        '1,"Cavity wall, as built,\ninsulated (assumed)"\n'
        '2,"Solid brick"\n'
    )
    assert load.local_row_count(p) == 2


def test_local_row_count_handles_crlf_with_quoted_newline(tmp_path):
    p = tmp_path / "c.csv"
    p.write_bytes(b'A,B\r\n1,"x\r\ny"\r\n2,"z"\r\n')
    assert load.local_row_count(p) == 2


def test_local_row_count_skips_trailing_blank_line(tmp_path):
    """Regression test: a stray trailing blank line must not inflate the count.

    Government CSV exports do carry trailing blank lines. csv.reader yields an
    empty row for one, and a naive `sum(1 for _ in reader)` would count it as a
    record — the same failure class the quoted-newline handling exists to
    prevent: a false reconciliation failure on a load that was correct.
    """
    p = tmp_path / "c.csv"
    p.write_text("H1,H2\na,b\nc,d\n\n")
    assert load.local_row_count(p) == 2


def test_table_columns_queries_information_schema():
    """Regression test: SHOW COLUMNS returns table_name first, not column_name.

    Reading row[0] off `SHOW COLUMNS IN TABLE` would return N copies of the
    table's own name instead of real column names, silently breaking
    header_diff. INFORMATION_SCHEMA.COLUMNS has an unambiguous COLUMN_NAME
    field; pin the emitted SQL so the intent stays correct even though a fake
    cursor can't validate the real result shape.
    """
    cursor = FakeCursor(results=[[("A",), ("B",)]])
    cols = load.table_columns(cursor, "epc.raw.certificates")
    assert cols == ["A", "B"]
    sql = cursor.statements[0]
    assert "INFORMATION_SCHEMA.COLUMNS" in sql
    assert "COLUMN_NAME" in sql
    assert "CERTIFICATES" in sql


def test_header_diff_detects_added_and_removed():
    diff = load.header_diff(["A", "B"], ["B", "C"])
    assert diff == {"added": ["C"], "removed": ["A"]}


def test_header_diff_is_empty_when_identical_ignoring_order():
    assert load.header_diff(["A", "B"], ["B", "A"]) == {"added": [], "removed": []}


def test_header_diff_ignores_case_only_changes():
    """COPY INTO matches columns case-insensitively, so a pure recase of a
    column name must not be reported as simultaneously added and removed."""
    assert load.header_diff(["postcode"], ["POSTCODE"]) == {"added": [], "removed": []}


def test_load_kind_fails_on_header_change():
    cursor = FakeCursor(results=[None, [("A",), ("B",)]])
    with pytest.raises(ValueError, match="Header changed"):
        load.load_kind(cursor, "certificates", ["A", "C"])


def test_load_kind_allows_header_change_when_overridden():
    cursor = FakeCursor(results=[None, [("A",), ("B",)], None, None, None, (7,)])
    assert load.load_kind(cursor, "certificates", ["A", "C"], allow_schema_change=True) == 7


def test_load_kind_truncates_before_copying():
    cursor = FakeCursor(results=[None, [("A",)], None, None, (5,)])
    rows = load.load_kind(cursor, "certificates", ["A"])

    joined = " ".join(cursor.statements)
    assert "TRUNCATE TABLE epc.raw.certificates" in joined
    truncate_at = next(i for i, s in enumerate(cursor.statements) if "TRUNCATE" in s)
    copy_at = next(i for i, s in enumerate(cursor.statements) if "COPY INTO" in s)
    assert truncate_at < copy_at
    assert rows == 5


def test_reconcile_passes_on_match():
    load.reconcile({"certificates": 10}, {"certificates": 10})


def test_reconcile_raises_on_mismatch():
    with pytest.raises(ValueError, match="certificates"):
        load.reconcile({"certificates": 9}, {"certificates": 10})


def test_reconcile_raises_on_unexpected_actual_kind():
    """Regression test: an extra kind in `actual` must not pass silently.

    Walking only `expected.items()` never notices a kind that appears in
    `actual` but not in `expected`, which contradicts the "match exactly"
    promise in the docstring.
    """
    with pytest.raises(ValueError, match="recommendations"):
        load.reconcile({"certificates": 100, "recommendations": 987654}, {"certificates": 100})


def test_reconcile_raises_on_missing_expected_kind():
    with pytest.raises(ValueError, match="certificates"):
        load.reconcile({}, {"certificates": 100})
