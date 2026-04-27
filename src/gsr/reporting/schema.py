from __future__ import annotations

import sqlite3


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


def require_tables(conn: sqlite3.Connection, tables: list[str]) -> None:
    missing = [t for t in tables if not table_exists(conn, t)]
    if missing:
        raise RuntimeError(
            f"Missing required table(s): {', '.join(missing)}. "
            "Did you run the required pipeline steps (extract/retrieve/verify)?"
        )