"""Support Intelligence.

Analytics over the support interactions **this system produced** — not over a
ticket table, and not over customers, products or revenue.

SCOPE, AND WHY IT IS NARROW
---------------------------
`src/analytics/metrics.py` (Phase 2) analyses `ticket_history.csv`: volume by
channel, refund exposure, warranty funnels. That is business analytics, and it
belongs to a different question.

This module answers one question instead:

    Is the AI doing its job, and where is it failing?

Every metric here is about the assistant's own behaviour — what it understood,
what evidence it found, which tools it used, when it declined, and where the
failures cluster. Nothing here reports sales, customers or product performance,
because those are not support operations.

DATA SOURCE
-----------
`data/db/traces.db` — one row per real request handled by the system. These are
genuine outputs, not simulated labels: the intent came from the classifier, the
confidence from the pipeline, the escalation reason from the actual gate that
fired. That distinction matters when reading anything below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.observability import traces

# Retrieval below this BM25 is treated as "nothing usable found" - the
# threshold calibrated in Phase 7 and re-swept in Phase 10.
WEAK_RETRIEVAL = 7.0
LOW_CONFIDENCE = 0.50


def load_traces(days: int | None = None, limit: int = 20000) -> pd.DataFrame:
    """Load traces with list columns already parsed."""
    df = traces.load(limit=limit)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    if days:
        cutoff = df["created_at"].max() - pd.Timedelta(days=days)
        df = df[df["created_at"] >= cutoff]

    for col in ("actions_taken", "citations", "caveats", "guardrail_rules"):
        df[col] = df[col].apply(_parse_list)

    df["date"] = df["created_at"].dt.date
    df["hour"] = df["created_at"].dt.hour
    df["resolved"] = df["resolution_status"].str.startswith("resolved")
    df["low_confidence"] = (df["confidence"] < LOW_CONFIDENCE) & (~df["escalation_required"].astype(bool))
    return df.reset_index(drop=True)


def _parse_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v or "[]")
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _pct(numer: float, denom: float) -> float:
    return round(100 * numer / denom, 1) if denom else 0.0


# =====================================================================
# Headline
# =====================================================================

@dataclass
class SupportOverview:
    """The numbers an operations lead would look at first."""

    total_conversations: int = 0
    resolution_rate_pct: float = 0.0
    escalation_rate_pct: float = 0.0
    clarification_rate_pct: float = 0.0
    refusal_rate_pct: float = 0.0

    avg_confidence: float = 0.0
    low_confidence_pct: float = 0.0

    retrieval_failure_pct: float = 0.0
    avg_citations: float = 0.0
    uncited_answer_pct: float = 0.0

    screenshot_pct: float = 0.0
    screenshot_useful_pct: float = 0.0

    avg_tools_per_conversation: float = 0.0
    zero_tool_pct: float = 0.0

    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    guardrail_trigger_pct: float = 0.0
    thumbs_down_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def overview(df: pd.DataFrame) -> SupportOverview:
    if df.empty:
        return SupportOverview()

    n = len(df)
    answered = df[df["resolved"]]
    with_image = df[df["has_image"] == 1]

    return SupportOverview(
        total_conversations=n,
        resolution_rate_pct=_pct(df["resolved"].sum(), n),
        escalation_rate_pct=_pct(df["escalation_required"].sum(), n),
        clarification_rate_pct=_pct(
            (df["resolution_status"] == "needs_information").sum(), n),
        refusal_rate_pct=_pct((df["resolution_status"] == "refused").sum(), n),

        avg_confidence=round(float(df["confidence"].mean()), 3),
        low_confidence_pct=_pct(df["low_confidence"].sum(), n),

        retrieval_failure_pct=_pct(df["retrieval_failed"].sum(), n),
        avg_citations=round(float(answered["n_citations"].mean()), 2)
        if len(answered) else 0.0,
        uncited_answer_pct=_pct((answered["n_citations"] == 0).sum(),
                                len(answered)),

        screenshot_pct=_pct(len(with_image), n),
        screenshot_useful_pct=_pct(with_image["image_contributed"].sum(),
                                   len(with_image)) if len(with_image) else 0.0,

        avg_tools_per_conversation=round(float(df["n_tools"].mean()), 2),
        zero_tool_pct=_pct((df["n_tools"] == 0).sum(), n),

        median_latency_ms=round(float(df["latency_ms"].median()), 1),
        p95_latency_ms=round(float(df["latency_ms"].quantile(0.95)), 1),

        guardrail_trigger_pct=_pct((df["guardrail_severity"] != "info").sum(), n),
        thumbs_down_pct=_pct((df["feedback"] == "down").sum(), n),
    )


# =====================================================================
# Distributions
# =====================================================================

def intent_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Volume is not cost. An intent can be 20% of traffic and cause no work,
    or 2% and escalate almost every time."""
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("intent").agg(
        conversations=("trace_id", "size"),
        resolution_pct=("resolved", lambda s: round(100 * s.mean(), 1)),
        escalation_pct=("escalation_required", lambda s: round(100 * s.mean(), 1)),
        avg_confidence=("confidence", lambda s: round(s.mean(), 3)),
        retrieval_fail_pct=("retrieval_failed",
                            lambda s: round(100 * s.mean(), 1)),
        avg_tools=("n_tools", lambda s: round(s.mean(), 2)),
        median_latency_ms=("latency_ms", lambda s: round(s.median(), 1)),
    ).reset_index()
    g["share_pct"] = (100 * g["conversations"] / len(df)).round(1)
    # Escalation load: what fraction of ALL human handovers this intent causes.
    total_esc = df["escalation_required"].sum()
    esc_by = df.groupby("intent")["escalation_required"].sum()
    g["share_of_escalations_pct"] = g["intent"].map(
        lambda i: _pct(esc_by.get(i, 0), total_esc))
    return g.sort_values("conversations", ascending=False)


def sentiment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("sentiment").agg(
        conversations=("trace_id", "size"),
        escalation_pct=("escalation_required", lambda s: round(100 * s.mean(), 1)),
        avg_confidence=("confidence", lambda s: round(s.mean(), 3)),
    ).reset_index()
    g["share_pct"] = (100 * g["conversations"] / len(df)).round(1)
    return g.sort_values("conversations", ascending=False)


def urgency_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    order = {"high": 0, "medium": 1, "low": 2}
    g = df.groupby("urgency").agg(
        conversations=("trace_id", "size"),
        escalation_pct=("escalation_required", lambda s: round(100 * s.mean(), 1)),
        median_latency_ms=("latency_ms", lambda s: round(s.median(), 1)),
    ).reset_index()
    g["share_pct"] = (100 * g["conversations"] / len(df)).round(1)
    g["_o"] = g["urgency"].map(order).fillna(9)
    return g.sort_values("_o").drop(columns="_o")


# =====================================================================
# AI performance
# =====================================================================

def tool_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Which tools the agent actually reaches for.

    A tool that never fires is either unnecessary or unreachable, and both are
    worth knowing.
    """
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for t in r["actions_taken"]:
            rows.append({"tool": t, "resolved": r["resolved"],
                         "escalated": bool(r["escalation_required"]),
                         "confidence": r["confidence"]})
    if not rows:
        return pd.DataFrame()

    t = pd.DataFrame(rows)
    g = t.groupby("tool").agg(
        calls=("tool", "size"),
        resolution_pct=("resolved", lambda s: round(100 * s.mean(), 1)),
        escalation_pct=("escalated", lambda s: round(100 * s.mean(), 1)),
        avg_confidence=("confidence", lambda s: round(s.mean(), 3)),
    ).reset_index()
    g["conversations_pct"] = (100 * g["calls"] / len(df)).round(1)
    return g.sort_values("calls", ascending=False)


def unused_tools(df: pd.DataFrame) -> list[str]:
    """Registered tools that were never called."""
    try:
        from src.agent.tools import REGISTRY

        used = {t for row in df["actions_taken"] for t in row} if not df.empty else set()
        return sorted(set(REGISTRY) - used)
    except Exception:
        return []


def retrieval_health(df: pd.DataFrame) -> dict[str, Any]:
    """How often the knowledge base is consulted, and how often it delivers."""
    if df.empty:
        return {}

    consulted = df[df["n_chunks"] > 0]
    failed = df[df["retrieval_failed"] == 1]

    return {
        "consulted_pct": _pct(len(consulted), len(df)),
        "failure_pct": _pct(len(failed), len(df)),
        "failure_pct_of_consulted": _pct(len(failed), len(consulted)),
        "median_bm25_success": round(
            float(consulted[consulted["retrieval_failed"] == 0]["max_bm25"].median()), 2
        ) if len(consulted[consulted["retrieval_failed"] == 0]) else 0.0,
        "median_bm25_failure": round(float(failed["max_bm25"].median()), 2)
        if len(failed) else 0.0,
        "escalation_when_retrieval_fails_pct": _pct(
            failed["escalation_required"].sum(), len(failed)) if len(failed) else 0.0,
        "escalation_when_retrieval_ok_pct": _pct(
            df[df["retrieval_failed"] == 0]["escalation_required"].sum(),
            len(df[df["retrieval_failed"] == 0])),
    }


def failed_retrievals(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """The questions the knowledge base could not answer.

    This is the single most actionable table here: each row is a candidate
    documentation gap.
    """
    if df.empty:
        return pd.DataFrame()
    f = df[df["retrieval_failed"] == 1]
    if f.empty:
        return pd.DataFrame()
    return (f[["created_at", "question", "intent", "max_bm25",
               "resolution_status", "escalation_reason"]]
            .sort_values("max_bm25")
            .head(limit)
            .reset_index(drop=True))


def low_confidence_cases(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Answers given with weak conviction. These reached a customer."""
    if df.empty:
        return pd.DataFrame()
    lc = df[df["low_confidence"]]
    if lc.empty:
        return pd.DataFrame()
    return (lc[["created_at", "question", "intent", "confidence", "n_citations",
                "max_bm25", "resolution_status"]]
            .sort_values("confidence")
            .head(limit)
            .reset_index(drop=True))


def screenshot_usage(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    img = df[df["has_image"] == 1]
    if img.empty:
        return {"conversations_with_image": 0}

    useful = img[img["image_contributed"] == 1]
    return {
        "conversations_with_image": len(img),
        "share_pct": _pct(len(img), len(df)),
        "contributed_pct": _pct(len(useful), len(img)),
        "code_extracted_pct": _pct(img["image_error_code"].notna().sum(), len(img)),
        "resolution_pct_with_image": _pct(img["resolved"].sum(), len(img)),
        "resolution_pct_without": _pct(
            df[df["has_image"] == 0]["resolved"].sum(),
            len(df[df["has_image"] == 0])),
        "top_codes": img["image_error_code"].dropna().value_counts()
        .head(8).to_dict(),
    }


def escalation_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Why humans are being pulled in. The reason matters more than the rate.

    A high escalation rate driven by `mutating_action_requires_approval` is the
    system working as designed. The same rate driven by
    `no_supporting_documentation` is a documentation gap.
    """
    if df.empty:
        return pd.DataFrame()
    esc = df[df["escalation_required"] == 1]
    if esc.empty:
        return pd.DataFrame()

    g = esc.groupby("escalation_reason").agg(
        count=("trace_id", "size"),
        avg_confidence=("confidence", lambda s: round(s.mean(), 3)),
        top_intent=("intent", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()
    g["share_pct"] = (100 * g["count"] / len(esc)).round(1)

    # By design vs by failure. The distinction is what makes this table useful.
    BY_DESIGN = {
        "mutating_action_requires_approval", "identity_verification_required",
        "legal_or_chargeback_threat", "relationship_issue_requires_human",
        "unauthorised_action", "unauthorised_commitment_requested",
    }
    g["category"] = g["escalation_reason"].map(
        lambda r: "by design" if r in BY_DESIGN else "capability gap")
    return g.sort_values("count", ascending=False)


def unresolved_clusters(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Recurring questions the system could not resolve.

    Grouped on a normalised question stem, so near-identical phrasings collapse
    into one row. Frequency is what makes a gap worth fixing.
    """
    if df.empty:
        return pd.DataFrame()

    un = df[(~df["resolved"]) | (df["feedback"] == "down")].copy()
    if un.empty:
        return pd.DataFrame()

    import re

    STOP = {"the", "a", "an", "my", "i", "is", "do", "can", "you", "to", "of",
            "for", "it", "and", "what", "how", "in", "on", "does"}

    def stem(q: str) -> str:
        words = [w for w in re.findall(r"[a-z]+", str(q).lower())
                 if w not in STOP and len(w) > 2]
        return " ".join(sorted(words)[:4])

    un["cluster"] = un["question"].apply(stem)
    g = un.groupby("cluster").agg(
        occurrences=("trace_id", "size"),
        example=("question", "first"),
        intent=("intent", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
        top_reason=("escalation_reason",
                    lambda s: s.dropna().mode().iloc[0]
                    if len(s.dropna().mode()) else "-"),
        avg_bm25=("max_bm25", lambda s: round(s.mean(), 2)),
    ).reset_index()
    return (g[g["occurrences"] >= 1]
            .sort_values("occurrences", ascending=False)
            .head(limit)
            .drop(columns="cluster")
            .reset_index(drop=True))


def guardrail_activity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = [{"rule": r, "severity": row["guardrail_severity"]}
            for _, row in df.iterrows() for r in row["guardrail_rules"]]
    if not rows:
        return pd.DataFrame()
    g = pd.DataFrame(rows).groupby(["rule", "severity"]).size().reset_index(
        name="count")
    return g.sort_values("count", ascending=False)


# =====================================================================
# Trends
# =====================================================================

def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("date").agg(
        conversations=("trace_id", "size"),
        resolution_pct=("resolved", lambda s: round(100 * s.mean(), 1)),
        escalation_pct=("escalation_required", lambda s: round(100 * s.mean(), 1)),
        avg_confidence=("confidence", lambda s: round(s.mean(), 3)),
        retrieval_fail_pct=("retrieval_failed", lambda s: round(100 * s.mean(), 1)),
    ).reset_index()
    return g.sort_values("date")


@dataclass
class EmergingIssue:
    """A support problem that grew. Stated as a sentence, because a lift ratio
    on its own does not tell an operator what to do."""

    topic: str
    intent: str
    recent_count: int
    baseline_rate: float
    lift: float
    signal: str
    headline: str
    escalation_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


TOPIC_TERMS = {
    "login / account access": ("login", "log in", "sign in", "password",
                               "locked out", "otp", "access my account",
                               "can't log", "cannot log", "verification code"),
    "payment failure": ("payment failed", "card declined", "payment isn't",
                        "payment not", "charged twice", "double charged",
                        "transaction failed", "pay-4", "pay-5", "gateway"),
    "refund delay": ("refund", "money back", "not received my refund",
                     "still waiting"),
    "delivery delay": ("where is my order", "not arrived", "late", "delayed",
                       "still not delivered", "tracking"),
    "display fault": ("screen", "display", "monitor", "dead pixel", "flicker",
                      "black screen", "err-dp", "dsp-"),
    "battery / charging": ("battery", "charging", "won't charge", "charger",
                           "bat-"),
    "overheating": ("hot", "overheat", "thermal", "fan", "thrm-"),
    "connectivity": ("wifi", "wi-fi", "network", "bluetooth", "wifi-"),
    "boot failure": ("won't turn on", "will not turn on", "won't boot",
                     "blue screen", "crashed", "sys-0x"),
    "return eligibility": ("can i return", "return window", "returnable",
                           "restocking"),
    "warranty claim": ("warranty", "still covered", "under warranty"),
}


def _topic_of(text: str) -> str | None:
    low = str(text).lower()
    for topic, terms in TOPIC_TERMS.items():
        if any(t in low for t in terms):
            return topic
    return None


def emerging_issues(df: pd.DataFrame, recent_days: int = 7,
                    baseline_days: int = 28, min_recent: int = 3,
                    lift_threshold: float = 1.8) -> list[EmergingIssue]:
    """Detect support topics growing faster than their own baseline.

    Compares a trailing window against the preceding one, normalised per day so
    unequal window lengths do not create a false signal. A topic absent from the
    baseline is reported as NEW rather than as an infinite lift.
    """
    if df.empty or "created_at" not in df:
        return []

    d = df.copy()
    d["topic"] = d["question"].apply(_topic_of)
    d = d[d["topic"].notna()]
    if d.empty:
        return []

    latest = d["created_at"].max()
    recent_start = latest - pd.Timedelta(days=recent_days)
    base_start = recent_start - pd.Timedelta(days=baseline_days)

    recent = d[d["created_at"] >= recent_start]
    baseline = d[(d["created_at"] >= base_start) & (d["created_at"] < recent_start)]

    if recent.empty:
        return []

    out: list[EmergingIssue] = []
    for topic, grp in recent.groupby("topic"):
        n_recent = len(grp)
        if n_recent < min_recent:
            continue

        recent_rate = n_recent / recent_days
        base_n = len(baseline[baseline["topic"] == topic])
        base_rate = base_n / baseline_days if baseline_days else 0.0

        intent = grp["intent"].mode().iloc[0] if len(grp["intent"].mode()) else ""
        esc = _pct(grp["escalation_required"].sum(), n_recent)

        if base_rate == 0:
            signal = "NEW"
            lift = float("inf")
            headline = (f"{topic.capitalize()} issues appeared this week "
                        f"({n_recent} conversations) with no prior history.")
        else:
            lift = recent_rate / base_rate
            if lift >= lift_threshold * 1.6:
                signal = "SPIKE"
            elif lift >= lift_threshold:
                signal = "ELEVATED"
            else:
                continue
            headline = (f"{topic.capitalize()} issues are running "
                        f"{lift:.1f}x their usual rate this week "
                        f"({n_recent} conversations vs "
                        f"{base_rate * recent_days:.0f} expected).")

        if esc >= 50:
            headline += f" {esc:.0f}% required a human."

        out.append(EmergingIssue(
            topic=topic, intent=intent, recent_count=n_recent,
            baseline_rate=round(base_rate, 3),
            lift=round(lift, 2) if np.isfinite(lift) else 999.0,
            signal=signal, headline=headline, escalation_pct=esc))

    return sorted(out, key=lambda e: (-e.recent_count, e.topic))


def topic_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Most common support topics, regardless of outcome.

    Topic is matched on vocabulary rather than the predicted intent, because
    intent is an 11-way label and "battery" and "overheating" both land in
    technical_support while needing different fixes.
    """
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["topic"] = d["question"].apply(_topic_of)
    d = d[d["topic"].notna()]
    if d.empty:
        return pd.DataFrame()

    g = d.groupby("topic").agg(
        conversations=("trace_id", "size"),
        resolution_pct=("resolved", lambda s: round(100 * s.mean(), 1)),
        escalation_pct=("escalation_required", lambda s: round(100 * s.mean(), 1)),
        retrieval_fail_pct=("retrieval_failed", lambda s: round(100 * s.mean(), 1)),
    ).reset_index()
    g["share_pct"] = (100 * g["conversations"] / len(df)).round(1)
    return g.sort_values("conversations", ascending=False)


def escalation_headline(df: pd.DataFrame) -> str | None:
    """One sentence naming the largest source of human workload."""
    if df.empty:
        return None
    esc = df[df["escalation_required"] == 1]
    if len(esc) < 3:
        return None
    top = esc["intent"].value_counts()
    if top.empty:
        return None
    intent, count = top.index[0], int(top.iloc[0])
    share = _pct(count, len(esc))
    return (f"{intent.replace('_', ' ').capitalize()} accounts for the largest "
            f"share of escalations ({share:.0f}%, {count} of {len(esc)}).")


def documentation_gap_headline(df: pd.DataFrame) -> str | None:
    """One sentence on the biggest knowledge-base gap."""
    if df.empty:
        return None
    failed = df[df["retrieval_failed"] == 1]
    if len(failed) < 3:
        return None
    top = failed["intent"].value_counts()
    intent, count = top.index[0], int(top.iloc[0])
    return (f"The knowledge base returned nothing usable for {len(failed)} "
            f"conversations ({_pct(len(failed), len(df)):.0f}%), most often on "
            f"{intent.replace('_', ' ')} ({count}).")


def headlines(df: pd.DataFrame) -> list[str]:
    """Plain-language findings, for the top of the dashboard."""
    out = []
    for e in emerging_issues(df):
        out.append(e.headline)
    for fn in (escalation_headline, documentation_gap_headline):
        h = fn(df)
        if h:
            out.append(h)
    return out


if __name__ == "__main__":
    df = load_traces()
    if df.empty:
        print("No traces yet. Run: python scripts/simulate_support_traffic.py")
        raise SystemExit(0)

    o = overview(df)
    print(f"conversations   {o.total_conversations}")
    print(f"resolution      {o.resolution_rate_pct}%")
    print(f"escalation      {o.escalation_rate_pct}%")
    print(f"retrieval fail  {o.retrieval_failure_pct}%")
    print(f"low confidence  {o.low_confidence_pct}%")
    print("\nheadlines:")
    for h in headlines(df):
        print(f"  - {h}")
