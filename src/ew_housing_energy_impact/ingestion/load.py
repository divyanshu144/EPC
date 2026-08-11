"""Create raw tables, load them from the stage, and prove the load."""

from __future__ import annotations

import csv
from pathlib import Path

DATABASE = "epc"
RAW_SCHEMA = "raw"
STAGE = "epc.raw.epc_raw_stage"
FILE_FORMAT = "epc.raw.csv_format"


def raw_table(kind: str) -> str:
    """Fully qualified raw table name for a source kind."""
    return f"{DATABASE}.{RAW_SCHEMA}.{kind}"


def read_header(path: Path) -> list[str]:
    """First row of a CSV, as a list of column names."""
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle))


def generate_raw_ddl(table: str, columns: list[str]) -> str:
    """CREATE TABLE with every column typed STRING.

    Generated rather than hand-written because raw takes every column EPC ships
    (93 for certificates) and the list is only knowable from the data.

    Raw is a bronze layer: EPC ships 'NO DATA!', 'INVALID!', empty numerics and
    inconsistent dates, so typing here would reject rows at load — exactly what
    a faithful copy of source must not do. Casting happens in staging with
    TRY_TO_*, where a failed cast becomes a testable NULL rather than a lost row.
    """
    if not columns:
        raise ValueError(f"Cannot generate DDL for {table}: no columns found in header")
    body = ",\n".join(f'  "{name}" STRING' for name in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n{body}\n)"


def copy_into_sql(table: str, kind: str) -> str:
    """COPY INTO binding columns by name, failing loudly on any malformed row.

    MATCH_BY_COLUMN_NAME requires PARSE_HEADER on the file format. Positional
    binding is unsafe here: every column is STRING, so a reordered export would
    load successfully with values in the wrong columns.
    """
    return (
        f"COPY INTO {table}\n"
        f"FROM @{STAGE}/{kind}/\n"
        f"FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')\n"
        f"MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE\n"
        f"PATTERN = '.*{kind}[.]csv'\n"
        f"ON_ERROR = ABORT_STATEMENT"
    )


def local_row_count(path: Path) -> int:
    """Data rows in a CSV, excluding the header.

    Parses rather than counting newlines. Free-text columns can contain newlines
    inside quoted values, and a line count would then exceed the true record
    count and fail reconciliation on a load that was correct. `newline=""` is
    required: without it, universal-newline translation breaks the csv module's
    own quoted-newline handling.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def table_columns(cursor, table: str) -> list[str]:
    """Column names of an existing table, empty if it does not exist yet."""
    rows = cursor.execute(f"SHOW COLUMNS IN TABLE {table}").fetchall()
    return [row[0] for row in rows] if rows else []


def header_diff(existing: list[str], incoming: list[str]) -> dict[str, list[str]]:
    """Set difference between an existing table's columns and a file's header."""
    return {
        "added": sorted(set(incoming) - set(existing)),
        "removed": sorted(set(existing) - set(incoming)),
    }


def load_kind(
    cursor,
    kind: str,
    columns: list[str],
    allow_schema_change: bool = False,
) -> int:
    """Create if absent, truncate, load one raw table. Returns its row count.

    TRUNCATE precedes COPY INTO deliberately. Snowflake's COPY load metadata
    expires after 64 days, so relying on skip-already-loaded is correct for 64
    days and silently duplicates rows afterwards. EPC archives are complete
    snapshots per period, so a full refresh is both correct and deterministic.
    TRUNCATE also clears the load metadata, so COPY reloads without FORCE.

    A header change fails rather than being absorbed silently: on a full-refresh
    pipeline a dropped source column would vanish from raw and resurface later as
    a confusing staging error.
    """
    table = raw_table(kind)
    cursor.execute(generate_raw_ddl(table, columns))

    existing = table_columns(cursor, table)
    if existing:
        diff = header_diff(existing, columns)
        if diff["added"] or diff["removed"]:
            if not allow_schema_change:
                raise ValueError(
                    f"Header changed for {table} — added={diff['added']} "
                    f"removed={diff['removed']}. Re-run with --allow-schema-change "
                    f"to accept this deliberately."
                )
            # CREATE TABLE IF NOT EXISTS is a no-op against a table that already
            # exists, so an accepted schema change must replace the table
            # definition outright or the new/dropped columns never take effect.
            replace_ddl = generate_raw_ddl(table, columns).replace(
                "CREATE TABLE IF NOT EXISTS", "CREATE OR REPLACE TABLE", 1
            )
            cursor.execute(replace_ddl)

    cursor.execute(f"TRUNCATE TABLE {table}")
    cursor.execute(copy_into_sql(table, kind))
    row = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def reconcile(actual: dict[str, int], expected: dict[str, int]) -> None:
    """Fail unless warehouse counts match local counts exactly.

    A zero exit from COPY INTO is not evidence of a correct load. This compares
    an independent local count against the warehouse; Snowflake's own
    rows_loaded would be Snowflake checking itself and could not detect a
    truncated upload.
    """
    mismatches = [
        f"{kind}: warehouse={actual.get(kind)} local={count}"
        for kind, count in expected.items()
        if actual.get(kind) != count
    ]
    if mismatches:
        raise ValueError("Row count reconciliation failed — " + "; ".join(mismatches))
