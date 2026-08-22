"""Evaluation results.

Reads the committed results rather than re-running them - a full evaluation
takes minutes and must not block a page load.
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

st.set_page_config(page_title="Evaluation · PacifyIQ", page_icon="📐",
                   layout="wide")
ui.inject_css()

ui.header("Evaluation",
          "How well the system actually works, measured component by component "
          "and end to end.")

data = svc.evaluation_results()
if not data.get("ok"):
    ui.error_box("No evaluation results available.", data.get("error", ""))
    st.code("python scripts/run_full_evaluation.py", language="bash")
    st.stop()

st.caption(f"Generated {data['generated']}")

# ------------------------------------------------------------- headline
st.markdown("#### Headline")
head = data["headline"]
st.dataframe(head, use_container_width=True, hide_index=True)

c1, c2 = st.columns([3, 2])
with c1:
    ui.note(
        "<b>Two of the perfect scores are not achievements.</b> RAG "
        "faithfulness is 1.000 <i>by construction</i> — the local generator "
        "copies sentences verbatim from retrieved documents and cannot "
        "fabricate. Screenshot extraction is 1.000 on synthetic renders. "
        "Neither will transfer to production.")
with c2:
    st.markdown('<div class="pq-card">', unsafe_allow_html=True)
    st.markdown("**The number that matters**")
    st.metric("End-to-end decision accuracy", "80.7%")
    st.caption("Across a set where 58% of cases should NOT be answered.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------- metrics
tab_all, tab_fail, tab_method = st.tabs(
    ["All metrics", "Failure cases", "Method"])

with tab_all:
    allm = data.get("all_metrics")
    if allm is None:
        st.caption("Detailed metrics not available.")
    else:
        components = st.multiselect(
            "Component", sorted(allm["component"].unique()),
            default=list(sorted(allm["component"].unique())))
        view = allm[allm["component"].isin(components)]

        below = view[view["status"] == "below target"]
        if len(below):
            st.markdown("**Below target**")
            st.dataframe(below[["component", "metric", "value", "target"]],
                         use_container_width=True, hide_index=True)
            st.caption(
                "`unsafe_resolutions` counts requests answered confidently that "
                "should have been refused or escalated. It is the failure that "
                "reaches a customer, so it is reported rather than averaged "
                "away.")
            st.divider()

        st.dataframe(view, use_container_width=True, hide_index=True,
                     height=460)

with tab_fail:
    failures = svc.failure_cases()
    if not failures:
        st.caption("No failure tables found.")
    else:
        st.caption("Every case the evaluation marked wrong. Inspecting these "
                   "is more informative than the aggregate.")
        choice = st.selectbox("Component", sorted(failures))
        df = failures[choice]
        st.caption(f"{len(df)} failing case(s)")
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)

with tab_method:
    st.markdown("""
#### Scoring

Three tiers, in descending order of trust:

**Deterministic** — exact match, set membership, arithmetic. Used wherever the
answer is a fact: an intent label, a retrieved section, an eligibility state, a
refund figure.

**Curated** — hand-authored expectations. Used where the correct behaviour is a
decision rather than a fact: should this escalate, must this answer contain
"14".

**LLM-as-judge** — available, but **off by default and contributing no headline
number**. A test asserts this.

#### Why no judge for factual correctness

Every curated case turns on a specific number — 14 days, 5 pixels, 75Hz.
String matching checks that exactly, cheaply, and without the circularity of
asking a language model to grade a language model.

Where this understates performance: an answer conveying the right fact in
different words scores as wrong.

#### The evaluation set

50 text cases and 7 image cases across 18 categories, including ambiguous
questions, missing information, unreadable and irrelevant screenshots,
hallucination traps, prompt injection, conflicting documents and out-of-domain
requests.

**58% of cases should not be answered.** A test set made only of answerable
questions measures fluency, not judgement.

#### Limitations

All data is synthetic. No hosted LLM has been run, so the generation numbers
come from a backend that cannot fabricate. The adversarial set was written by
the same person who wrote the safety rules. Sample sizes are 25–140 per
component and confidence intervals are not reported.

Full detail: `reports/evaluation_report.md`.
""")
