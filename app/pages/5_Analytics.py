"""Support Intelligence.

AI and support-operations analytics over the system's own conversations.
Deliberately NOT a business dashboard: no sales, no products, no customer
segments. The question here is "is the assistant doing its job, and where is it
failing".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui  # noqa: E402
from src.ui import service as svc  # noqa: E402

st.set_page_config(page_title="Support Intelligence · PacifyIQ", page_icon="📊",
                   layout="wide")
ui.inject_css()

ui.header("Support Intelligence",
          "How the assistant is performing, and where it is failing. Measured "
          "over its own conversations.")

data = svc.support_intelligence()
if not data.get("ok"):
    st.info(data.get("error", "No conversations logged yet."))
    st.caption("Handle a few requests on the Customer Support page, or "
               "generate a realistic workload:")
    st.code("python scripts/simulate_support_traffic.py --days 35 --per-day 14",
            language="bash")
    st.stop()

o = data["overview"]
df = data["frame"]

st.caption(f"{o['total_conversations']} conversations · "
           f"{data['date_range']}")

# ==================================================================
# What changed
# ==================================================================
headlines = data["headlines"]
if headlines:
    st.markdown("#### What changed")
    for h in headlines:
        st.markdown(f'<div class="pq-card pq-card-accent">{h}</div>',
                    unsafe_allow_html=True)
    st.write("")

tabs = st.tabs(["Operations", "AI performance", "Where it fails", "Trends"])

# ==================================================================
# 1. Operations
# ==================================================================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations", o["total_conversations"])
    c2.metric("Resolved without a human", f"{o['resolution_rate_pct']}%")
    c3.metric("Escalated", f"{o['escalation_rate_pct']}%")
    c4.metric("Asked for clarification", f"{o['clarification_rate_pct']}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average confidence", f"{o['avg_confidence']:.2f}")
    c2.metric("Refused", f"{o['refusal_rate_pct']}%")
    c3.metric("Median response", f"{o['median_latency_ms']:.0f} ms")
    c4.metric("95th percentile", f"{o['p95_latency_ms']:.0f} ms")

    st.divider()
    st.markdown("#### Intent mix — volume is not workload")
    st.caption("An intent can be a fifth of traffic and cause no work, or a "
               "twentieth and escalate every time. The last two columns are "
               "the ones that matter operationally.")

    intents = data["intents"]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.bar_chart(intents.set_index("intent")["conversations"], height=340)
    with c2:
        st.dataframe(
            intents[["intent", "share_pct", "resolution_pct",
                     "escalation_pct", "share_of_escalations_pct"]]
            .rename(columns={"share_pct": "% traffic",
                             "resolution_pct": "% resolved",
                             "escalation_pct": "% escalated",
                             "share_of_escalations_pct": "% of all handovers"}),
            use_container_width=True, hide_index=True, height=340)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Sentiment")
        s = data["sentiment"]
        st.dataframe(
            s[["sentiment", "conversations", "share_pct", "escalation_pct"]]
            .rename(columns={"share_pct": "% of traffic",
                             "escalation_pct": "% escalated"}),
            use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Urgency")
        u = data["urgency"]
        st.dataframe(
            u[["urgency", "conversations", "share_pct", "escalation_pct"]]
            .rename(columns={"share_pct": "% of traffic",
                             "escalation_pct": "% escalated"}),
            use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Most common issues")
    st.caption("Topics detected across all conversations, resolved or not.")
    topics = data["topics"]
    if len(topics):
        st.dataframe(topics, use_container_width=True, hide_index=True)
    else:
        st.caption("No recognised topics yet.")

# ==================================================================
# 2. AI performance
# ==================================================================
with tabs[1]:
    st.markdown("#### Knowledge base")
    r = data["retrieval"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations using it", f"{r.get('consulted_pct', 0)}%")
    c2.metric("Returned nothing usable", f"{r.get('failure_pct', 0)}%")
    c3.metric("Median score when it works",
              f"{r.get('median_bm25_success', 0):.1f}")
    c4.metric("Median score when it fails",
              f"{r.get('median_bm25_failure', 0):.1f}")

    fail_esc = r.get("escalation_when_retrieval_fails_pct", 0)
    ok_esc = r.get("escalation_when_retrieval_ok_pct", 0)
    ui.note(
        f"When retrieval finds nothing, <b>{fail_esc:.0f}%</b> of conversations "
        f"reach a human. When it succeeds, only <b>{ok_esc:.0f}%</b> do. "
        f"Retrieval quality is the single largest driver of human workload in "
        f"this system.")

    st.divider()
    st.markdown("#### Tool usage")
    st.caption("Which tools the agent actually reaches for. A tool that never "
               "fires is either unnecessary or unreachable.")
    tools = data["tools"]
    c1, c2 = st.columns([3, 2])
    with c1:
        if len(tools):
            st.dataframe(
                tools[["tool", "calls", "conversations_pct", "resolution_pct",
                       "escalation_pct"]]
                .rename(columns={"conversations_pct": "% of conversations",
                                 "resolution_pct": "% resolved",
                                 "escalation_pct": "% escalated"}),
                use_container_width=True, hide_index=True)
    with c2:
        st.metric("Average tools per conversation",
                  f"{o['avg_tools_per_conversation']:.2f}")
        st.caption("Of 13 available. An agent calling everything would "
                   "average 13.")
        st.metric("Answered with no tools", f"{o['zero_tool_pct']}%")
        unused = data["unused_tools"]
        if unused:
            st.caption("**Never called:** " + ", ".join(unused))
            st.caption("The tier-3 tools should never fire autonomously; the "
                       "rest indicate workloads this traffic did not contain.")

    st.divider()
    st.markdown("#### Screenshot contribution")
    sc = data["screenshots"]
    if sc.get("conversations_with_image"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("With a screenshot", f"{sc['share_pct']}%")
        c2.metric("Screenshot was usable", f"{sc['contributed_pct']}%")
        c3.metric("Resolved with image", f"{sc['resolution_pct_with_image']}%")
        c4.metric("Resolved without", f"{sc['resolution_pct_without']}%")

        lift = sc["resolution_pct_with_image"] - sc["resolution_pct_without"]
        if lift > 0:
            ui.note(f"Conversations with a readable screenshot resolve "
                    f"<b>{lift:.0f} percentage points</b> more often. The image "
                    f"usually contains an error code that the customer's own "
                    f"words do not.")
        if sc.get("top_codes"):
            st.markdown("**Error codes seen in screenshots**")
            st.bar_chart(pd.Series(sc["top_codes"]), height=220)
    else:
        st.caption("No screenshots in this period.")

    st.divider()
    st.markdown("#### Safety rules triggered")
    g = data["guardrails"]
    if len(g):
        st.dataframe(g, use_container_width=True, hide_index=True)
        st.caption(f"{o['guardrail_trigger_pct']}% of conversations triggered "
                   f"at least one rule.")
    else:
        st.caption("No safety rules triggered in this period.")

# ==================================================================
# 3. Where it fails
# ==================================================================
with tabs[2]:
    st.markdown("#### Why humans get involved")
    st.caption("The reason matters more than the rate. Escalating a refund is "
               "the system working as designed; escalating because no "
               "documentation was found is a gap.")

    esc = data["escalations"]
    if len(esc):
        by_design = esc[esc["category"] == "by design"]["count"].sum()
        gaps = esc[esc["category"] == "capability gap"]["count"].sum()
        total = by_design + gaps

        c1, c2 = st.columns(2)
        c1.metric("By design", f"{by_design} ({100 * by_design / total:.0f}%)")
        c2.metric("Capability gap", f"{gaps} ({100 * gaps / total:.0f}%)")

        st.dataframe(
            esc[["escalation_reason", "count", "share_pct", "category",
                 "top_intent"]]
            .rename(columns={"share_pct": "% of escalations"}),
            use_container_width=True, hide_index=True)

        if gaps > by_design:
            ui.note(
                "Most handovers are <b>capability gaps</b>, not policy "
                "requirements. That is a documentation and retrieval problem "
                "rather than a safety one, and it is the highest-value thing "
                "to fix.")

    st.divider()
    st.markdown("#### Failed retrievals — the documentation gap list")
    st.caption("Questions where the knowledge base returned nothing usable. "
               "Each row is a candidate for a new FAQ entry.")
    fr = data["failed_retrievals"]
    if len(fr):
        st.dataframe(fr, use_container_width=True, hide_index=True, height=320)
        st.download_button("Download gap list (CSV)",
                           fr.to_csv(index=False).encode("utf-8"),
                           "documentation_gaps.csv", "text/csv")
    else:
        st.success("No failed retrievals in this period.")

    st.divider()
    st.markdown("#### Low-confidence answers")
    st.caption("Answers given to customers with weak conviction. These did not "
               "escalate, so nobody reviewed them.")
    lc = data["low_confidence"]
    if len(lc):
        st.dataframe(lc, use_container_width=True, hide_index=True, height=280)
    else:
        st.success("No low-confidence answers in this period.")

    st.divider()
    st.markdown("#### Recurring unresolved issues")
    st.caption("Near-identical questions the system could not resolve, grouped. "
               "Frequency is what makes a gap worth fixing.")
    un = data["unresolved"]
    if len(un):
        st.dataframe(un, use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing recurring.")

    if o["thumbs_down_pct"]:
        st.divider()
        st.metric("Marked unhelpful by customers", f"{o['thumbs_down_pct']}%")

# ==================================================================
# 4. Trends
# ==================================================================
with tabs[3]:
    trend = data["daily"]
    if len(trend) < 2:
        st.info("Not enough history for trends yet.")
    else:
        st.markdown("#### Conversation volume")
        st.line_chart(trend.set_index("date")["conversations"], height=240)

        st.markdown("#### Resolution and escalation")
        st.line_chart(
            trend.set_index("date")[["resolution_pct", "escalation_pct"]],
            height=240)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average confidence")
            st.line_chart(trend.set_index("date")["avg_confidence"], height=220)
        with c2:
            st.markdown("#### Retrieval failure rate")
            st.line_chart(trend.set_index("date")["retrieval_fail_pct"],
                          height=220)

        st.divider()
        st.markdown("#### Emerging issues")
        st.caption("Trailing 7 days against the preceding 28, normalised per "
                   "day so unequal windows do not create a false signal.")

        issues = data["emerging"]
        if issues:
            for e in issues:
                colour = {"NEW": "red", "SPIKE": "red"}.get(e["signal"], "orange")
                st.markdown(
                    f'{ui.badge(e["signal"], colour)} **{e["topic"]}** — '
                    f'{e["recent_count"]} conversations, '
                    f'{e["escalation_pct"]:.0f}% escalated',
                    unsafe_allow_html=True)
                st.caption(e["headline"])
        else:
            st.success("No topic is growing beyond its normal rate.")

st.divider()
st.caption(
    "Messages in this installation are synthetic, but every measurement above "
    "is the system's real behaviour: the intent came from the classifier, the "
    "retrieval scores from the index, the escalation reason from the gate that "
    "actually fired.")
