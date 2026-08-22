"""SQLite connection management.

One place that knows how to open the database. Everything else asks for a
connection from here so the row factory and pragmas are consistent.

    from src.db.connection import get_connection, query_df

    with get_connection() as con:
        row = con.execute("SELECT ...").fetchone()

    df = query_df("SELECT * FROM v_return_eligibility LIMIT 5")
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config.settings import settings


def _configure(con: sqlite3.Connection) -> sqlite3.Connection:
    """Apply consistent settings to every connection."""
    con.row_factory = sqlite3.Row          # rows behave like dicts
    con.execute("PRAGMA foreign_keys = ON")
    return con


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context-managed connection. Commits on success, rolls back on error.

    Usage:
        with get_connection() as con:
            con.execute("INSERT ...")
    """
    path = db_path or settings.db_path
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run: python scripts/setup_database.py"
        )
    con = _configure(sqlite3.connect(path))
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def query_df(sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. The workhorse for analytics."""
    with get_connection() as con:
        return pd.read_sql_query(sql, con, params=params)


def query_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    """Run a SELECT and return the first row as a dict, or None."""
    with get_connection() as con:
        row = con.execute(sql, params).fetchone()
        return dict(row) if row else None


def query_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a SELECT and return all rows as a list of dicts."""
    with get_connection() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def table_exists(name: str) -> bool:
    """True if a table or view with this name exists."""
    with get_connection() as con:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
            (name,),
        ).fetchone()
        return row is not None


def list_objects() -> pd.DataFrame:
    """List all tables and views with row counts. Useful for a sanity check."""
    with get_connection() as con:
        objs = con.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
        rows = []
        for o in objs:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {o['name']}").fetchone()[0]
            except sqlite3.Error:
                n = None
            rows.append({"name": o["name"], "type": o["type"], "rows": n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(list_objects().to_string(index=False))
