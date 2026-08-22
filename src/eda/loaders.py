"""Canonical loaders for every PacifyIQ data asset.

One place that knows how to read each file, with dtypes and NA handling made
explicit rather than left to pandas inference. Notebooks and analysis modules
import from here so they cannot disagree about what the data is.

    from src.eda.loaders import load_tickets, load_intent_train

    tickets = load_tickets()          # parsed timestamps, derived date parts
    train   = load_intent_train()     # text + intent
"""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config.settings import settings

# Ordered by real-world volume, from the taxonomy in docs/PROBLEM.md.
INTENT_ORDER = [
    "order_tracking",
    "return_policy_question",
    "return_refund_request",
    "product_information",
    "shipping_delivery",
    "technical_support",
    "warranty_claim",
    "payment_issue",
    "account_management",
    "complaint",
    "out_of_scope",
]

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
PRIORITY_ORDER = ["low", "medium", "high"]


# =====================================================================
# Ticket history
# =====================================================================

@lru_cache(maxsize=1)
def load_tickets() -> pd.DataFrame:
    """Simulated 6-month support ticket history.

    NOTE: `feedback` is blank for most rows in the source CSV. pandas reads
    blanks as NaN, so we normalise to the empty string and add an explicit
    `has_feedback` flag rather than letting NaN propagate silently.
    """
    df = pd.read_csv(
        settings.tickets_csv,
        dtype={
            "ticket_id": "string",
            "intent": "category",
            "subtopic": "category",
            "sentiment": "category",
            "priority": "category",
            "resolved_by": "category",
            "status": "category",
            "region": "category",
            "channel": "category",
        },
        keep_default_na=False,
        na_values=[],
    )
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["confidence"] = pd.to_numeric(df["confidence"])
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"])
    df["tokens_used"] = pd.to_numeric(df["tokens_used"])

    # derived time parts, used repeatedly downstream
    df["date"] = df["created_at"].dt.date
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.day_name()
    df["week"] = df["created_at"].dt.to_period("W").astype(str)
    df["days_ago"] = (df["created_at"].max() - df["created_at"]).dt.days

    df["feedback"] = df["feedback"].fillna("").astype(str)
    df["has_feedback"] = df["feedback"].str.len() > 0
    df["escalated"] = df["resolved_by"] == "human"

    df["intent"] = df["intent"].cat.set_categories(
        [c for c in INTENT_ORDER if c in df["intent"].cat.categories]
    )
    return df


# =====================================================================
# Intent classification data
# =====================================================================

@lru_cache(maxsize=1)
def load_intent_train() -> pd.DataFrame:
    """Template-generated training set. 2 columns: text, intent."""
    df = pd.read_csv(settings.intents_dir / "train.csv", dtype="string")
    df["intent"] = pd.Categorical(
        df["intent"], categories=[c for c in INTENT_ORDER], ordered=False
    )
    df["split"] = "train"
    return df


@lru_cache(maxsize=1)
def load_intent_test() -> pd.DataFrame:
    """Hand-authored hard test set. Carries a secondary intent for compound
    messages and a note explaining why each case is difficult."""
    df = pd.read_csv(
        settings.intents_dir / "test_hard.csv",
        dtype="string",
        keep_default_na=False,
        na_values=[],
    )
    df["secondary_intent"] = df["secondary_intent"].fillna("").astype(str)
    df["is_compound"] = df["secondary_intent"].str.len() > 0
    df["primary_intent"] = pd.Categorical(
        df["primary_intent"], categories=[c for c in INTENT_ORDER]
    )
    df["split"] = "test"
    # rename for a common schema with train
    return df.rename(columns={"primary_intent": "intent"})


def load_intent_combined() -> pd.DataFrame:
    """Train and test stacked, for drift comparison. Common columns only."""
    tr = load_intent_train()[["text", "intent", "split"]]
    te = load_intent_test()[["text", "intent", "split"]]
    return pd.concat([tr, te], ignore_index=True)


# =====================================================================
# Operational database
# =====================================================================

def _db(sql: str) -> pd.DataFrame:
    with sqlite3.connect(settings.db_path) as con:
        return pd.read_sql_query(sql, con)


@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    """Denormalised orders from v_order_detail."""
    df = _db("SELECT * FROM v_order_detail")
    for c in ("order_date", "dispatch_date", "delivery_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_return_eligibility() -> pd.DataFrame:
    return _db("SELECT * FROM v_return_eligibility")


@lru_cache(maxsize=1)
def load_customers() -> pd.DataFrame:
    return _db("SELECT * FROM customers")


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    return _db("SELECT * FROM products")


# =====================================================================
# Knowledge corpus
# =====================================================================

@lru_cache(maxsize=1)
def load_corpus() -> pd.DataFrame:
    """Extract raw text page by page from every PDF in the corpus.

    Returns one row per page so section- and page-level statistics are
    available before any chunking decision is made.
    """
    import pdfplumber

    rows = []
    for path in sorted(settings.documents_dir.rglob("*.pdf")):
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                rows.append(
                    {
                        "doc": path.stem,
                        "path": str(path.relative_to(settings.documents_dir)),
                        "is_manual": "manuals" in str(path),
                        "page": i,
                        "text": text,
                        "n_chars": len(text),
                        "n_words": len(text.split()),
                    }
                )
    return pd.DataFrame(rows)


# =====================================================================
# Evaluation sets
# =====================================================================

def load_eval(name: str) -> dict:
    """Load one evaluation set by filename stem, e.g. 'retrieval_eval'."""
    path = settings.eval_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_evals() -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(settings.eval_dir.glob("*.json"))
    }


# =====================================================================
# Convenience
# =====================================================================

def figures_dir() -> Path:
    """Where EDA plots are written."""
    d = settings.root / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir() -> Path:
    """Where derived tables are written for reuse by later phases."""
    d = settings.data_dir / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    for name, fn in [
        ("tickets", load_tickets),
        ("intent train", load_intent_train),
        ("intent test", load_intent_test),
        ("orders", load_orders),
        ("customers", load_customers),
        ("products", load_products),
    ]:
        df = fn()
        print(f"{name:16s} {df.shape[0]:6,} rows x {df.shape[1]:2d} cols")
    ev = load_all_evals()
    total = sum(len(v.get("cases", [])) for v in ev.values())
    print(f"{'eval sets':16s} {len(ev):6,} files, {total} cases")
