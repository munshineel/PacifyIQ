"""Conversation history — real logged requests, not simulated data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui  # noqa: E402
from src.ui import service as svc  # noqa: E402

st.set_page_config(page_title="History · PacifyIQ", page_icon="🗂️",
                   layout="wide")
ui.inject_css()

ui.header("Conversation history",
          "Every request handled by this installation. Message text is "
          "redacted before it is stored.")

df = svc.live_traces(limit=1000)

if df.empty:
    st.info("No conversations yet.")
    st.caption("Ask something on the **Customer Support** page and it will "
               "appear here.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversations", len(df))
resolved = df["resolution_status"].str.startswith("resolved").sum()
c2.metric("Resolved", f"{resolved} ({resolved / len(df):.0%})")
c3.metric("Escalated", int(df["escalation_required"].sum()))
c4.metric("Median latency", f"{df['latency_ms'].median():.0f} ms")

st.divider()

f1, f2, f3 = st.columns(3)
with f1:
    statuses = st.multiselect(
        "Outcome", sorted(df["resolution_status"].dropna().unique()),
        default=list(sorted(df["resolution_status"].dropna().unique())))
with f2:
    intents = st.multiselect("Intent", sorted(df["intent"].dropna().unique()))
with f3:
    only_flagged = st.checkbox("Only escalated, refused or thumbs-down")

view = df[df["resolution_status"].isin(statuses)]
if intents:
    view = view[view["intent"].isin(intents)]
if only_flagged:
    view = view[(view["escalation_required"] == 1)
                | (view["resolution_status"] == "refused")
                | (view["feedback"] == "down")]

st.caption(f"{len(view)} of {len(df)} conversations")

STATUS_ICON = {"resolved": "✅", "resolved_with_caveat": "✅",
               "needs_information": "❓", "escalated": "🔺", "refused": "🚫"}

for _, row in view.head(60).iterrows():
    icon = STATUS_ICON.get(row["resolution_status"], "•")
    fb = {"up": " 👍", "down": " 👎"}.get(row.get("feedback", ""), "")
    label = (f"{icon} {row['created_at']} — {str(row['question'])[:70]}"
             f"{'…' if len(str(row['question'])) > 70 else ''}{fb}")

    with st.expander(label):
        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"**Intent**  \n{row.get('intent') or '—'}")
        c2.caption(f"**Sentiment**  \n{row.get('sentiment') or '—'}")
        c3.caption(f"**Urgency**  \n{row.get('urgency') or '—'}")
        c4.caption(f"**Confidence**  \n{(row.get('confidence') or 0):.0%}")

        st.markdown("**Question**")
        st.markdown(f"> {row['question']}")
        st.markdown("**Answer**")
        st.markdown(str(row["answer"])[:1200] or "_none_")

        if row["escalation_required"]:
            st.warning(f"Escalated — "
                       f"{str(row.get('escalation_reason', '')).replace('_', ' ')}")

        try:
            actions = json.loads(row.get("actions_taken") or "[]")
            citations = json.loads(row.get("citations") or "[]")
            rules = json.loads(row.get("guardrail_rules") or "[]")
        except Exception:
            actions, citations, rules = [], [], []

        if actions:
            st.caption("**Actions** — " + ", ".join(actions))
        if citations:
            st.caption("**Sources** — " + "; ".join(citations))
        if rules:
            st.caption("**Safety rules triggered** — " + ", ".join(rules))
        st.caption(f"{row['steps']} step(s) · {row['latency_ms']:.0f} ms · "
                   f"trace `{row['trace_id']}`")

st.divider()
st.download_button(
    "Download history (CSV)",
    view.to_csv(index=False).encode("utf-8"),
    file_name="pacifyiq_history.csv", mime="text/csv")
