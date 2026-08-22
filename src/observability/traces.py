"""Trace persistence.

One record per request, written at the boundary. The admin dashboard is a
read-only view over this table, which is why the schema is fixed here rather
than derived from whatever the UI happens to need.

WHY THIS EXISTS SEPARATELY FROM THE AGENT
-----------------------------------------
The agent returns a decision; it does not know or care that anything is being
recorded. Keeping persistence out of the agent means the agent stays testable
without a database, and the trace schema can change without touching the
decision logic.

PRIVACY
-------
Message text is redacted through the guardrail layer before it is written. A
customer who pastes a card number into chat should not have it sitting in the
trace table forever.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.settings import settings

TRACE_DB = settings.data_dir / "db" / "traces.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id            TEXT PRIMARY KEY,
    session_id          TEXT,
    created_at          TEXT,

    question            TEXT,
    answer              TEXT,
    has_image           INTEGER DEFAULT 0,
    image_contributed   INTEGER DEFAULT 0,
    image_error_code    TEXT,

    intent              TEXT,
    intent_margin       REAL,
    sentiment           TEXT,
    urgency             TEXT,

    resolution_status   TEXT,
    escalation_required INTEGER,
    escalation_reason   TEXT,
    confidence          REAL,
    ticket_id           TEXT,

    actions_taken       TEXT,
    citations           TEXT,
    caveats             TEXT,
    guardrail_rules     TEXT,
    guardrail_severity  TEXT,

    max_bm25            REAL DEFAULT 0,
    n_chunks            INTEGER DEFAULT 0,
    retrieval_failed    INTEGER DEFAULT 0,
    n_tools             INTEGER DEFAULT 0,
    n_citations         INTEGER DEFAULT 0,

    steps               INTEGER,
    latency_ms          REAL,
    feedback            TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_intent  ON traces(intent);
CREATE INDEX IF NOT EXISTS idx_traces_status  ON traces(resolution_status);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
"""


def _connect() -> sqlite3.Connection:
    TRACE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(TRACE_DB)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


@dataclass
class Trace:
    """One request, as stored."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = "default"
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data: dict[str, Any] = field(default_factory=dict)


def record(decision: Any, session_id: str = "default",
           question: str | None = None, redact: bool = True) -> str:
    """Persist one agent decision. Returns the trace id.

    Never raises: a logging failure must not take down a customer request.
    """
    try:
        from src.guardrails.policy import ENGINE

        d = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        text = question if question is not None else d.get("question", "")
        if redact and text:
            text = ENGINE.redact(str(text))

        g = d.get("guardrails") or {}
        rules, severity = [], "info"
        for stage in g.values():
            if isinstance(stage, dict):
                rules.extend(stage.get("rules_fired", []) or [])
                if stage.get("severity") and stage["severity"] != "info":
                    severity = stage["severity"]

        row = {
            "trace_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": str(text)[:1000],
            "answer": str(d.get("answer", ""))[:3000],
            "has_image": int(bool(d.get("has_image"))),
            "image_contributed": int(bool(d.get("image_contributed"))),
            "image_error_code": d.get("image_error_code"),
            "intent": d.get("intent"),
            "intent_margin": d.get("intent_margin"),
            "sentiment": d.get("sentiment"),
            "urgency": d.get("urgency"),
            "resolution_status": d.get("resolution_status"),
            "escalation_required": int(bool(d.get("escalation_required"))),
            "escalation_reason": d.get("escalation_reason"),
            "confidence": d.get("confidence"),
            "ticket_id": d.get("ticket_id"),
            "actions_taken": json.dumps(d.get("actions_taken") or []),
            "citations": json.dumps(d.get("citations") or []),
            "caveats": json.dumps(d.get("caveats") or []),
            "guardrail_rules": json.dumps(sorted(set(rules))),
            "guardrail_severity": severity,
            "max_bm25": d.get("max_bm25", 0.0) or 0.0,
            "n_chunks": d.get("n_chunks", 0) or 0,
            "retrieval_failed": int(bool(d.get("retrieval_failed"))),
            "n_tools": len(d.get("actions_taken") or []),
            "n_citations": len(d.get("citations") or []),
            "steps": d.get("steps", 0),
            "latency_ms": d.get("latency_ms", 0.0),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with _connect() as con:
            con.execute(f"INSERT INTO traces ({cols}) VALUES ({marks})",
                        tuple(row.values()))
        return row["trace_id"]
    except Exception:
        return ""


def set_feedback(trace_id: str, value: str) -> bool:
    """Record a thumbs up/down against a trace."""
    if value not in ("up", "down", ""):
        return False
    try:
        with _connect() as con:
            con.execute("UPDATE traces SET feedback = ? WHERE trace_id = ?",
                        (value, trace_id))
        return True
    except Exception:
        return False


def load(limit: int = 500, session_id: str | None = None):
    """Recent traces as a DataFrame. Empty frame when nothing is logged yet."""
    import pandas as pd

    try:
        with _connect() as con:
            sql = "SELECT * FROM traces"
            params: tuple = ()
            if session_id:
                sql += " WHERE session_id = ?"
                params = (session_id,)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params = params + (limit,)
            return pd.read_sql_query(sql, con, params=params)
    except Exception:
        return pd.DataFrame()


def count() -> int:
    try:
        with _connect() as con:
            return con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    except Exception:
        return 0


def clear() -> None:
    with _connect() as con:
        con.execute("DELETE FROM traces")


if __name__ == "__main__":
    from src.agent.loop import SupportAgent

    agent = SupportAgent()
    for q in ["How many dead pixels before replacement?",
              "Where is my order PAC-2026-12345?",
              "Do you offer student discounts?"]:
        tid = record(agent.handle(q), session_id="demo", question=q)
        print(f"  {tid}  {q}")
    print(f"\ntotal traces: {count()}")
    print(load(5)[["created_at", "intent", "resolution_status", "latency_ms"]]
          .to_string(index=False))
