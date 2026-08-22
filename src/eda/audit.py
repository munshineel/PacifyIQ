"""PHASE 2 — Dataset audit.

Structural and quality checks on every asset. Answers the question "what is
actually in this data and can I trust it" before any analysis or modelling.

Nothing here assumes column names or distributions; everything is derived
from the files as they exist on disk.

    python -m src.eda.audit          # full audit to stdout
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.eda import loaders

# ---------------------------------------------------------------- PII
# Patterns for detecting personal data that should not appear in a
# portfolio dataset. Order IDs are business references, not PII.
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_in": re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"),
    "card_like": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "aadhaar_like": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "url": re.compile(r"https?://\S+"),
}

ORDER_ID = re.compile(r"\bPAC[-]?2026[-]?\d{5}\b|\b#?\d{5}\b", re.I)


@dataclass
class AuditResult:
    """Findings for one dataset."""

    name: str
    rows: int
    cols: int
    schema: pd.DataFrame
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        out = [f"\n{'=' * 72}", f"AUDIT: {self.name}", "=" * 72]
        out.append(f"shape: {self.rows:,} rows x {self.cols} columns\n")
        out.append(self.schema.to_string(index=False))
        if self.notes:
            out.append("\nnotes:")
            out += [f"  - {n}" for n in self.notes]
        if self.issues:
            out.append("\nISSUES:")
            out += [f"  ! {i}" for i in self.issues]
        else:
            out.append("\nno structural issues found")
        return "\n".join(out)


# =====================================================================
# Generic profiling
# =====================================================================

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, nulls, cardinality, and a sample value."""
    rows = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        sample = s.dropna().iloc[0] if s.notna().any() else None
        if isinstance(sample, str) and len(sample) > 34:
            sample = sample[:31] + "..."
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "nulls": n_null,
                "null_pct": round(100 * n_null / max(len(df), 1), 1),
                "unique": n_unique,
                "card_pct": round(100 * n_unique / max(len(df), 1), 1),
                "sample": sample,
            }
        )
    return pd.DataFrame(rows)


def detect_pii(series: pd.Series) -> dict[str, int]:
    """Count PII pattern matches in a text column."""
    text = series.dropna().astype(str)
    return {k: int(text.str.contains(p, regex=True).sum()) for k, p in PII_PATTERNS.items()}


def numeric_outliers(s: pd.Series, k: float = 1.5) -> dict[str, Any]:
    """Tukey fence outlier count for a numeric column."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return {
        "q1": round(float(q1), 3),
        "q3": round(float(q3), 3),
        "lower_fence": round(float(lo), 3),
        "upper_fence": round(float(hi), 3),
        "n_outliers": int(((s < lo) | (s > hi)).sum()),
        "min": round(float(s.min()), 3),
        "max": round(float(s.max()), 3),
    }


# =====================================================================
# Per-dataset audits
# =====================================================================

def audit_tickets() -> AuditResult:
    df = loaders.load_tickets()
    res = AuditResult("tickets/ticket_history.csv", len(df), df.shape[1], profile_columns(df))

    # --- duplicates
    dup_ids = int(df["ticket_id"].duplicated().sum())
    if dup_ids:
        res.issues.append(f"{dup_ids} duplicate ticket_id values")
    else:
        res.notes.append("ticket_id is unique across all rows (primary key holds)")

    # exact duplicates ignoring the id
    content_cols = [c for c in df.columns if c not in ("ticket_id", "date", "hour",
                                                       "weekday", "week", "days_ago",
                                                       "has_feedback", "escalated")]
    dup_rows = int(df.duplicated(subset=content_cols).sum())
    res.notes.append(
        f"{dup_rows:,} rows share an identical attribute combination "
        f"({100 * dup_rows / len(df):.1f}%) - expected given only 14 categorical fields"
    )

    # --- missingness
    res.notes.append(
        f"feedback blank on {(~df['has_feedback']).sum():,} rows "
        f"({100 * (~df['has_feedback']).mean():.1f}%) - by design, most tickets are unrated"
    )

    # --- timestamps
    res.extras["date_min"] = str(df["created_at"].min())
    res.extras["date_max"] = str(df["created_at"].max())
    span = (df["created_at"].max() - df["created_at"].min()).days
    res.notes.append(f"timespan {span} days, {df['date'].nunique()} distinct dates")

    out_of_hours = int(((df["hour"] < 9) | (df["hour"] > 20)).sum())
    if out_of_hours:
        res.issues.append(f"{out_of_hours} tickets outside stated 09:00-21:00 support hours")
    else:
        res.notes.append("all timestamps fall inside stated support hours 09:00-21:00")

    sundays = int((df["weekday"] == "Sunday").sum())
    res.notes.append(f"Sunday tickets: {sundays} (policy says no Sunday support)")

    # --- numeric ranges
    for col in ("confidence", "latency_seconds", "tokens_used"):
        res.extras[f"outliers_{col}"] = numeric_outliers(df[col])
    bad_conf = int(((df["confidence"] < 0) | (df["confidence"] > 1)).sum())
    if bad_conf:
        res.issues.append(f"{bad_conf} confidence values outside [0, 1]")
    neg_lat = int((df["latency_seconds"] <= 0).sum())
    if neg_lat:
        res.issues.append(f"{neg_lat} non-positive latency values")

    # --- categorical integrity
    res.extras["intent_counts"] = df["intent"].value_counts().to_dict()
    res.extras["subtopics_per_intent"] = (
        df.groupby("intent", observed=True)["subtopic"].nunique().to_dict()
    )

    # is subtopic nested within intent, or does it cross over?
    crossover = (
        df.groupby("subtopic", observed=True)["intent"].nunique().pipe(lambda s: s[s > 1])
    )
    if len(crossover):
        res.issues.append(f"{len(crossover)} subtopics appear under more than one intent")
    else:
        res.notes.append("subtopic is strictly nested within intent (clean hierarchy)")

    # --- leakage check
    # confidence is generated from resolved_by, so it trivially predicts it.
    conf_ai = df.loc[~df["escalated"], "confidence"].mean()
    conf_hu = df.loc[df["escalated"], "confidence"].mean()
    res.extras["confidence_by_outcome"] = {"ai": round(conf_ai, 3), "human": round(conf_hu, 3)}
    res.issues.append(
        f"LEAKAGE: confidence separates outcomes almost perfectly "
        f"(ai={conf_ai:.3f} vs human={conf_hu:.3f}). It was generated FROM resolved_by, "
        f"so it must never be used as a feature to predict escalation."
    )

    # --- no free text
    res.notes.append(
        "no free-text column: this is aggregate operational metadata, not message content. "
        "Message-level text analysis must use intents/*.csv"
    )
    res.notes.append("no image/screenshot column present in any tabular asset")
    return res


def audit_intent_train() -> AuditResult:
    df = loaders.load_intent_train()
    res = AuditResult("intents/train.csv", len(df), df.shape[1], profile_columns(df))

    exact = int(df["text"].duplicated().sum())
    ci = int(df["text"].str.lower().str.strip().duplicated().sum())
    res.notes.append(f"exact duplicate texts: {exact}")
    if ci > exact:
        res.issues.append(
            f"{ci} case-insensitive duplicates ({ci - exact} differ only by casing/whitespace) "
            f"- these will inflate CV scores unless deduplicated before splitting"
        )

    # same text under two different labels = label noise
    conflicts = (
        df.groupby(df["text"].str.lower().str.strip())["intent"]
        .nunique()
        .pipe(lambda s: s[s > 1])
    )
    if len(conflicts):
        res.issues.append(f"{len(conflicts)} texts carry more than one intent label")
    else:
        res.notes.append("no label conflicts: each text maps to exactly one intent")

    empty = int((df["text"].str.strip().str.len() == 0).sum())
    if empty:
        res.issues.append(f"{empty} empty messages")
    short = int((df["text"].str.split().str.len() <= 2).sum())
    res.notes.append(f"very short messages (<=2 words): {short}")

    counts = df["intent"].value_counts()
    res.extras["class_counts"] = counts.to_dict()
    res.extras["imbalance_ratio"] = round(counts.max() / counts.min(), 1)
    res.notes.append(
        f"class imbalance {counts.max()}:{counts.min()} = "
        f"{counts.max() / counts.min():.1f}x - use macro-F1, not accuracy"
    )

    pii = detect_pii(df["text"])
    res.extras["pii"] = pii
    hits = {k: v for k, v in pii.items() if v}
    if hits:
        res.issues.append(f"possible PII patterns: {hits}")
    else:
        res.notes.append("no PII patterns detected (emails, phones, cards, IFSC)")

    order_refs = int(df["text"].str.contains(ORDER_ID, regex=True).sum())
    res.notes.append(f"{order_refs} messages contain an order reference")
    return res


def audit_intent_test() -> AuditResult:
    df = loaders.load_intent_test()
    res = AuditResult("intents/test_hard.csv", len(df), df.shape[1], profile_columns(df))

    train = loaders.load_intent_train()
    overlap = set(df["text"].str.lower().str.strip()) & set(
        train["text"].str.lower().str.strip()
    )
    if overlap:
        res.issues.append(f"LEAKAGE: {len(overlap)} test texts also appear in train")
    else:
        res.notes.append("no text overlap with train.csv - the split is clean")

    res.notes.append(
        f"compound messages (two intents present): {df['is_compound'].sum()} "
        f"({100 * df['is_compound'].mean():.0f}%) - single-label scoring will "
        f"undercount performance on these"
    )
    res.extras["class_counts"] = df["intent"].value_counts().to_dict()

    tags = {
        "IMAGE-dependent": df["note"].str.contains("IMAGE").sum(),
        "planted defect": df["note"].str.contains("DEFECT").sum(),
        "boundary case": df["note"].str.contains("BOUNDARY").sum(),
        "security": df["note"].str.contains("SECURITY").sum(),
        "hallucination bait": df["note"].str.contains("HALLUCINATION").sum(),
    }
    res.extras["case_tags"] = tags
    res.notes.append("annotated case types: " + ", ".join(f"{k}={v}" for k, v in tags.items()))

    missing = set(loaders.INTENT_ORDER) - set(df["intent"].dropna().unique())
    if missing:
        res.issues.append(f"intents absent from test set: {sorted(missing)}")
    else:
        res.notes.append("all 11 intents represented")

    counts = df["intent"].value_counts()
    thin = counts[counts < 8]
    if len(thin):
        res.issues.append(
            f"{len(thin)} classes have <8 test examples ({dict(thin)}) - "
            f"per-class F1 on these has wide confidence intervals"
        )
    return res


def audit_orders() -> AuditResult:
    df = loaders.load_orders()
    res = AuditResult("db/pacify.db :: orders", len(df), df.shape[1], profile_columns(df))

    dup = int(df["order_id"].duplicated().sum())
    res.notes.append("order_id unique" if not dup else f"{dup} duplicate order_ids")

    undelivered = df["delivery_date"].isna().sum()
    res.notes.append(
        f"{undelivered:,} orders have no delivery_date "
        f"({100 * undelivered / len(df):.0f}%) - correct for in-transit/cancelled states"
    )

    # referential sanity: delivery must not precede dispatch
    both = df.dropna(subset=["dispatch_date", "delivery_date"])
    bad = int((both["delivery_date"] < both["dispatch_date"]).sum())
    if bad:
        res.issues.append(f"{bad} orders delivered before dispatch")
    else:
        res.notes.append("no orders delivered before dispatch (date ordering holds)")

    res.extras["status_counts"] = df["status"].value_counts().to_dict()
    res.extras["region_counts"] = df["region"].value_counts().to_dict()
    res.extras["payment_counts"] = df["payment_method"].value_counts().to_dict()

    elig = loaders.load_return_eligibility()
    res.extras["eligibility_counts"] = elig["eligibility"].value_counts().to_dict()
    boundary = int(elig["days_remaining"].between(-2, 2).sum())
    res.notes.append(f"{boundary} orders sit within +/-2 days of their return-window boundary")
    return res


def audit_corpus() -> AuditResult:
    df = loaders.load_corpus()
    res = AuditResult("documents/*.pdf", len(df), df.shape[1], profile_columns(df))

    res.notes.append(f"{df['doc'].nunique()} documents, {len(df)} pages total")
    empty = int((df["n_chars"] == 0).sum())
    if empty:
        res.issues.append(f"{empty} pages extract no text (scanned or image-only?)")
    else:
        res.notes.append("every page extracts text successfully")

    res.extras["pages_per_doc"] = df.groupby("doc")["page"].max().to_dict()
    res.extras["words_per_page"] = numeric_outliers(df["n_words"])
    res.notes.append(
        f"words per page: median {df['n_words'].median():.0f}, "
        f"range {df['n_words'].min()}-{df['n_words'].max()}"
    )
    return res


# =====================================================================
# Runner
# =====================================================================

def run_all() -> list[AuditResult]:
    return [
        audit_tickets(),
        audit_intent_train(),
        audit_intent_test(),
        audit_orders(),
        audit_corpus(),
    ]


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    results = run_all()
    for r in results:
        print(r.report())

    n_issues = sum(len(r.issues) for r in results)
    print(f"\n{'=' * 72}\nAUDIT COMPLETE: {len(results)} datasets, {n_issues} issues flagged\n{'=' * 72}")
