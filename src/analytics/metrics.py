"""Dashboard analytics.

Every function returns a pandas DataFrame ready to hand to Streamlit. SQL is
kept simple and inline; the composition, parameterisation and post-processing
happen in Python.

    from src.analytics.metrics import overview, emerging_issues

    print(overview())
    print(emerging_issues(recent_days=7, baseline_days=28))

Phases 1-13 read `tickets_raw` (simulated history). Once the app logs real
traces, point TICKET_TABLE at the trace table — the queries are unchanged,
which is why the trace schema should mirror these columns.
"""
from __future__ import annotations

import pandas as pd

from src.db.connection import query_df

TICKET_TABLE = "tickets_raw"

# Rough Groq blended rate, USD per million tokens. Adjust to the live figure.
COST_PER_MILLION_TOKENS = 0.30


# =====================================================================
# Overview
# =====================================================================

def overview() -> pd.DataFrame:
    """Headline operational metrics — the tiles at the top of the dashboard."""
    df = query_df(f"""
        SELECT
            COUNT(*)                                                AS total_tickets,
            SUM(resolved_by = 'ai')                                 AS ai_resolved,
            SUM(resolved_by = 'human')                              AS escalated,
            AVG(latency_seconds)                                    AS avg_latency_s,
            AVG(confidence)                                         AS avg_confidence,
            SUM(tokens_used)                                        AS total_tokens,
            SUM(sentiment = 'negative')                             AS negative_tickets,
            SUM(feedback = 'up')                                    AS thumbs_up,
            SUM(feedback = 'down')                                  AS thumbs_down
        FROM {TICKET_TABLE}
    """)

    # derived metrics in Python — easier to read and to test than nested SQL
    total = df.at[0, "total_tickets"]
    rated = df.at[0, "thumbs_up"] + df.at[0, "thumbs_down"]
    df["deflection_rate_pct"] = round(100 * df["ai_resolved"] / total, 1)
    df["escalation_rate_pct"] = round(100 * df["escalated"] / total, 1)
    df["negative_pct"] = round(100 * df["negative_tickets"] / total, 1)
    df["thumbs_down_pct"] = round(100 * df["thumbs_down"] / rated, 1) if rated else None
    df["est_cost_usd"] = round(df["total_tokens"] / 1e6 * COST_PER_MILLION_TOKENS, 2)
    df["avg_latency_s"] = df["avg_latency_s"].round(2)
    df["avg_confidence"] = df["avg_confidence"].round(3)
    return df


def overview_dict() -> dict:
    """Overview as a plain dict — convenient for st.metric() calls."""
    return overview().iloc[0].to_dict()


# =====================================================================
# Distributions
# =====================================================================

def intent_distribution() -> pd.DataFrame:
    """Volume, deflection and sentiment per intent."""
    df = query_df(f"""
        SELECT
            intent,
            COUNT(*)                        AS tickets,
            SUM(resolved_by = 'ai')         AS ai_resolved,
            SUM(sentiment = 'negative')     AS negative,
            AVG(confidence)                 AS avg_confidence,
            AVG(latency_seconds)            AS avg_latency_s
        FROM {TICKET_TABLE}
        GROUP BY intent
        ORDER BY tickets DESC
    """)
    total = df["tickets"].sum()
    df["share_pct"] = (100 * df["tickets"] / total).round(1)
    df["deflection_pct"] = (100 * df["ai_resolved"] / df["tickets"]).round(1)
    df["negative_pct"] = (100 * df["negative"] / df["tickets"]).round(1)
    df["avg_confidence"] = df["avg_confidence"].round(3)
    df["avg_latency_s"] = df["avg_latency_s"].round(2)
    return df[["intent", "tickets", "share_pct", "deflection_pct",
               "negative_pct", "avg_confidence", "avg_latency_s"]]


def sentiment_by_week() -> pd.DataFrame:
    """Weekly sentiment split with a 4-week rolling average of negatives."""
    df = query_df(f"""
        SELECT
            STRFTIME('%Y-W%W', created_at)  AS week,
            COUNT(*)                        AS tickets,
            SUM(sentiment = 'positive')     AS positive,
            SUM(sentiment = 'neutral')      AS neutral,
            SUM(sentiment = 'negative')     AS negative
        FROM {TICKET_TABLE}
        GROUP BY week
        ORDER BY week
    """)
    for col in ["positive", "neutral", "negative"]:
        df[f"{col}_pct"] = (100 * df[col] / df["tickets"]).round(1)
    # rolling window in pandas rather than a SQL window function
    df["negative_ma4w"] = df["negative_pct"].rolling(4, min_periods=1).mean().round(1)
    return df


def channel_region_breakdown() -> pd.DataFrame:
    """Deflection and latency split by region and channel."""
    df = query_df(f"""
        SELECT
            region, channel,
            COUNT(*)                    AS tickets,
            SUM(resolved_by = 'ai')     AS ai_resolved,
            AVG(latency_seconds)        AS avg_latency_s
        FROM {TICKET_TABLE}
        GROUP BY region, channel
        ORDER BY tickets DESC
    """)
    df["deflection_pct"] = (100 * df["ai_resolved"] / df["tickets"]).round(1)
    df["avg_latency_s"] = df["avg_latency_s"].round(2)
    return df.drop(columns=["ai_resolved"])


# =====================================================================
# Trends
# =====================================================================

def daily_volume(days: int = 30) -> pd.DataFrame:
    """Daily ticket counts with a 7-day moving average and week-over-week delta.

    The rolling calculation is pandas rather than a SQL window function —
    clearer, and it keeps the SQL to a simple GROUP BY.
    """
    df = query_df(f"""
        SELECT DATE(created_at) AS date, COUNT(*) AS tickets
        FROM {TICKET_TABLE}
        GROUP BY date
        ORDER BY date
    """)
    df["date"] = pd.to_datetime(df["date"])
    df["ma_7d"] = df["tickets"].rolling(7, min_periods=1).mean().round(1)
    df["wow_change"] = df["tickets"] - df["tickets"].shift(7)
    df["wow_pct"] = (100 * df["wow_change"] / df["tickets"].shift(7)).round(1)
    return df.tail(days).reset_index(drop=True)


def emerging_issues(
    recent_days: int = 7,
    baseline_days: int = 28,
    min_recent: int = 5,
    spike_threshold: float = 2.5,
    elevated_threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect intent/subtopic pairs rising above their recent baseline.

    Compares a trailing window against a prior baseline window, normalised to
    a per-day rate so the window lengths can differ.

    Validate against data/tickets/PLANTED_TRENDS.md:
      T1 account_management / login failure  -> must appear as SPIKE
      T2 technical_support / display issue   -> must appear as ELEVATED
      T4 seasonal shipping rise              -> must NOT be flagged
    """
    df = query_df(f"""
        SELECT
            intent,
            subtopic,
            DATE(created_at) AS date
        FROM {TICKET_TABLE}
    """)
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    recent_start = max_date - pd.Timedelta(days=recent_days)
    baseline_start = max_date - pd.Timedelta(days=recent_days + baseline_days)

    recent = df[df["date"] > recent_start]
    baseline = df[(df["date"] > baseline_start) & (df["date"] <= recent_start)]

    r = recent.groupby(["intent", "subtopic"]).size().rename("recent_n")
    b = baseline.groupby(["intent", "subtopic"]).size().rename("baseline_n")

    out = pd.concat([r, b], axis=1).fillna(0).reset_index()
    out["recent_per_day"] = (out["recent_n"] / recent_days).round(2)
    out["baseline_per_day"] = (out["baseline_n"] / baseline_days).round(2)
    out["lift"] = (
        out["recent_per_day"] / out["baseline_per_day"].replace(0, pd.NA)
    ).round(2)

    def classify(row) -> str:
        if row["baseline_per_day"] == 0 and row["recent_per_day"] > 1:
            return "NEW ISSUE"
        if pd.isna(row["lift"]):
            return "normal"
        if row["lift"] >= spike_threshold:
            return "SPIKE"
        if row["lift"] >= elevated_threshold:
            return "ELEVATED"
        return "normal"

    out["signal"] = out.apply(classify, axis=1)
    out = out[out["recent_n"] >= min_recent]
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


# =====================================================================
# Escalation and calibration
# =====================================================================

def escalation_analysis(min_tickets: int = 30) -> pd.DataFrame:
    """What drives handoffs — intent crossed with sentiment."""
    df = query_df(f"""
        SELECT
            intent, sentiment,
            COUNT(*)                                                    AS tickets,
            SUM(resolved_by = 'human')                                  AS escalated,
            AVG(CASE WHEN resolved_by = 'human' THEN confidence END)    AS conf_escalated,
            AVG(CASE WHEN resolved_by = 'ai'    THEN confidence END)    AS conf_resolved
        FROM {TICKET_TABLE}
        GROUP BY intent, sentiment
    """)
    df = df[df["tickets"] >= min_tickets].copy()
    df["escalation_pct"] = (100 * df["escalated"] / df["tickets"]).round(1)
    df["conf_escalated"] = df["conf_escalated"].round(3)
    df["conf_resolved"] = df["conf_resolved"].round(3)
    return df.sort_values("escalation_pct", ascending=False).reset_index(drop=True)


def confidence_calibration(bins: int = 10) -> pd.DataFrame:
    """Does reported confidence predict a good outcome?

    Buckets confidence and compares thumbs-down rate per bucket. A flat curve
    means the confidence signal is uninformative, which is itself a finding
    and the starting point for the Phase 12 calibration work.
    """
    df = query_df(f"""
        SELECT confidence, feedback, resolved_by
        FROM {TICKET_TABLE}
        WHERE feedback <> ''
    """)
    df["bucket"] = (df["confidence"] * bins).astype(int).clip(0, bins - 1) / bins

    out = (
        df.groupby("bucket")
        .agg(
            n=("feedback", "size"),
            thumbs_up=("feedback", lambda s: (s == "up").sum()),
            thumbs_down=("feedback", lambda s: (s == "down").sum()),
        )
        .reset_index()
    )
    out["down_rate_pct"] = (100 * out["thumbs_down"] / out["n"]).round(1)
    return out


# =====================================================================
# Operational
# =====================================================================

def expiring_return_windows(days_ahead: int = 3, limit: int = 25) -> pd.DataFrame:
    """Orders whose return window closes soon. Reads the eligibility view."""
    return query_df(
        """
        SELECT order_id, product_name, region, quantity, is_opened,
               delivery_date, days_since_delivery, window_days,
               days_remaining, window_basis
        FROM v_return_eligibility
        WHERE eligibility = 'eligible'
          AND days_remaining BETWEEN 0 AND ?
        ORDER BY days_remaining, order_id
        LIMIT ?
        """,
        (days_ahead, limit),
    )


def refund_exposure() -> pd.DataFrame:
    """Worst-case refund liability if every eligible order were returned."""
    df = query_df("""
        SELECT payment_method,
               COUNT(*)                     AS eligible_orders,
               SUM(refund_change_of_mind)   AS exposure,
               AVG(refund_change_of_mind)   AS avg_refund,
               SUM(restocking_fee)          AS restocking_recovered
        FROM v_refund_quote
        WHERE eligibility = 'eligible'
        GROUP BY payment_method
    """)
    df = df.sort_values("exposure", ascending=False).reset_index(drop=True)
    df["exposure_rank"] = df.index + 1
    for col in ["exposure", "avg_refund", "restocking_recovered"]:
        df[col] = df[col].round(0)
    return df


def warranty_funnel() -> pd.DataFrame:
    """Claim rates by brand and administering route."""
    df = query_df("""
        SELECT w.warranty_route,
               w.brand,
               COUNT(DISTINCT w.order_id)                       AS orders,
               SUM(w.warranty_state = 'in_warranty')            AS in_warranty,
               COUNT(wc.claim_id)                               AS claims_raised,
               SUM(wc.outcome = 'not_covered')                  AS rejected
        FROM v_warranty_status w
        LEFT JOIN warranty_claims wc ON wc.order_id = w.order_id
        GROUP BY w.warranty_route, w.brand
        ORDER BY orders DESC
    """)
    df["claim_rate_pct"] = (
        100 * df["claims_raised"] / df["in_warranty"].replace(0, pd.NA)
    ).round(2)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=" * 70)
    print("OVERVIEW")
    print("=" * 70)
    o = overview_dict()
    print(f"  total tickets     {o['total_tickets']:,}")
    print(f"  deflection rate   {o['deflection_rate_pct']}%")
    print(f"  escalation rate   {o['escalation_rate_pct']}%")
    print(f"  avg latency       {o['avg_latency_s']}s")
    print(f"  est cost          ${o['est_cost_usd']}")

    print("\n" + "=" * 70)
    print("EMERGING ISSUES (validate vs PLANTED_TRENDS.md)")
    print("=" * 70)
    print(emerging_issues().head(8).to_string(index=False))
