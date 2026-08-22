"""One-command database setup.

Builds pacify.db from the data assets:
  1. verifies the seeded database exists
  2. imports ticket history into tickets_raw
  3. applies the business-logic views
  4. runs a smoke test against the deterministic edge cases

Run from the project root:
    python scripts/setup_database.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

# make `src` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402

TICKET_COLUMNS = [
    "ticket_id", "created_at", "intent", "subtopic", "sentiment", "priority",
    "resolved_by", "status", "confidence", "latency_seconds", "tokens_used",
    "region", "channel", "feedback",
]


def import_tickets(con: sqlite3.Connection) -> int:
    """Load ticket_history.csv into tickets_raw."""
    csv_path = settings.tickets_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    con.execute("DROP TABLE IF EXISTS tickets_raw")
    cols = ", ".join(
        f"{c} {'REAL' if c in ('confidence', 'latency_seconds') else 'INTEGER' if c == 'tokens_used' else 'TEXT'}"
        for c in TICKET_COLUMNS
    )
    con.execute(f"CREATE TABLE tickets_raw ({cols})")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        placeholders = ",".join("?" * len(TICKET_COLUMNS))
        con.executemany(f"INSERT INTO tickets_raw VALUES ({placeholders})", reader)

    con.execute("CREATE INDEX IF NOT EXISTS idx_tr_date   ON tickets_raw(created_at)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tr_intent ON tickets_raw(intent)")
    return con.execute("SELECT COUNT(*) FROM tickets_raw").fetchone()[0]


def apply_views(con: sqlite3.Connection) -> list[str]:
    """Run every .sql file in sql/ in filename order.

    Fails loudly if nothing was applied. The original version returned an empty
    list when sql/ was missing or unreadable, which produced a database with
    tables but no views - and the failure only surfaced much later as
    "no such table: v_order_detail" from whatever code touched a view first.
    """
    if not settings.sql_dir.exists():
        raise FileNotFoundError(
            f"sql/ not found at {settings.sql_dir}. The view definitions live "
            f"there and the database is unusable without them.")

    sql_files = sorted(settings.sql_dir.glob("*.sql"))
    applied = []
    for path in sql_files:
        if "analytics" in path.name:
            continue  # analytics is a reference library, not DDL
        con.executescript(path.read_text(encoding="utf-8"))
        applied.append(path.name)

    if not applied:
        raise RuntimeError(
            f"no DDL files found in {settings.sql_dir}. Expected "
            f"01_business_logic_views.sql.")

    views = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='view'")]
    expected = {"v_order_detail", "v_return_eligibility", "v_warranty_status",
                "v_refund_quote", "v_customer_contact_history"}
    missing = expected - set(views)
    if missing:
        raise RuntimeError(
            f"views were not created: {sorted(missing)}. "
            f"Check {settings.sql_dir / '01_business_logic_views.sql'}.")

    return applied


def smoke_test(con: sqlite3.Connection) -> bool:
    """Verify the views against known edge cases. Fails loudly if wrong."""
    con.row_factory = sqlite3.Row
    expected = {
        "PAC-2026-12345": ("eligible", 14),   # day 12 of 14
        "PAC-2026-12347": ("expired", 14),    # day 15
        "PAC-2026-12348": ("eligible", 30),   # sealed, day 22
        "PAC-2026-12354": ("eligible", 14),   # EU override
        "PAC-2026-12368": ("expired", 7),     # bulk, day 9
    }
    ok = True
    for oid, (want_state, want_window) in expected.items():
        row = con.execute(
            "SELECT eligibility, window_days FROM v_return_eligibility WHERE order_id = ?",
            (oid,),
        ).fetchone()
        if row is None:
            print(f"  FAIL {oid}: not found")
            ok = False
        elif row["eligibility"] != want_state or row["window_days"] != want_window:
            print(f"  FAIL {oid}: got {row['eligibility']}/{row['window_days']}, "
                  f"want {want_state}/{want_window}")
            ok = False
        else:
            print(f"  ok   {oid}  {row['eligibility']:9s} window={row['window_days']}d")

    # EU must carry no restocking fee
    eu = con.execute(
        "SELECT restocking_fee FROM v_refund_quote WHERE order_id = 'PAC-2026-12354'"
    ).fetchone()
    if eu and eu["restocking_fee"] == 0:
        print("  ok   PAC-2026-12354  EU restocking fee waived")
    else:
        print("  FAIL EU restocking fee not waived")
        ok = False
    return ok


def main() -> int:
    print(f"database: {settings.db_path}")
    if not settings.db_path.exists():
        print("ERROR: pacify.db not found. Copy the data assets into data/ first.")
        return 1

    with sqlite3.connect(settings.db_path) as con:
        n = import_tickets(con)
        print(f"imported tickets_raw: {n:,} rows")

        applied = apply_views(con)
        print(f"applied SQL: {', '.join(applied)}")

        views = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")]
        print(f"views: {', '.join(views)}")

        print("\nsmoke test:")
        ok = smoke_test(con)
        con.commit()

    print("\nsetup complete" if ok else "\nSETUP FAILED - views are incorrect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
