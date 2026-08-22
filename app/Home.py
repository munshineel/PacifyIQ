"""PacifyIQ — entry point.

The UI layer contains NO business logic. Every page talks to
`src.ui.service`, which is the only module that reaches into the backend.
`src/` never imports streamlit; a test in tests/test_architecture.py asserts
the direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui  # noqa: E402
from src.ui import service as svc  # noqa: E402

st.set_page_config(page_title="PacifyIQ", page_icon="🎧", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

st.sidebar.markdown("### PacifyIQ")
st.sidebar.caption("Customer support intelligence")
st.sidebar.divider()
st.sidebar.caption(
    "All data is synthetic. Pacify Electronics Pvt. Ltd. is a fictional "
    "retailer created for this project.")

ui.header(
    "PacifyIQ",
    "A grounded customer-support assistant: it answers from published policy, "
    "cites its sources, and hands over to a colleague when it should not "
    "answer.")

# ------------------------------------------------------------- bootstrap
# On a fresh deployment the derived artifacts do not exist yet. Building them
# takes ~30s once; committing them would mean shipping duplicated state that
# can silently disagree with the corpus that produced it.
if not svc.system_ready():
    import scripts.ensure_artifacts as bootstrap

    if bootstrap.missing():
        st.info("First run on this server - preparing the knowledge base. "
                "This takes about half a minute and happens only once.")
        box = st.empty()
        with st.spinner("Building..."):
            ok, log = bootstrap.build(progress=lambda m: box.caption(m))
        box.empty()
        if ok:
            st.success("Ready.")
            st.rerun()
        else:
            st.error("Setup failed.")
            st.code(chr(10).join(log[-6:]))
            st.stop()

# ---------------------------------------------------------------- status
status = svc.check_system()
ready = svc.system_ready()

if not ready:
    st.error("Some required components are missing. The assistant will not "
             "run until these are set up.")

cols = st.columns(3)
for i, c in enumerate(status):
    with cols[i % 3]:
        st.markdown(
            f'<div class="pq-card"><div class="pq-value">{c.icon} '
            f'<b>{c.name}</b></div>'
            f'<div class="pq-muted">{c.detail}</div></div>',
            unsafe_allow_html=True)

if not ready:
    st.markdown("#### Setup")
    st.code(
        "python scripts/setup_database.py\n"
        "python scripts/build_index.py\n"
        "python scripts/train_intent_classifier.py",
        language="bash")
    st.stop()

st.divider()

# ------------------------------------------------------------ what it does
st.markdown("#### What this system does")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("""
Every request follows the same path, and every stage is visible in the answer:

**Understanding** → intent, sentiment and urgency, computed locally in a few
milliseconds.

**Evidence** → policy documents retrieved from a 13-document corpus, plus order
records, plus anything readable in an attached screenshot.

**Actions** → the assistant selects tools based on the request. It does not call
everything, and it cannot perform actions that move money or change an account.

**Result** → a grounded answer with citations, a request for more information,
or a handover to a colleague.
""")

with c2:
    st.markdown('<div class="pq-card">', unsafe_allow_html=True)
    st.markdown("**Measured performance**")
    st.caption("From the evaluation suite, not a demo.")
    m = [("End-to-end decision accuracy", "81%"),
         ("Retrieval Recall@5", "0.883"),
         ("Escalation accuracy", "0.933"),
         ("Adversarial inputs handled", "100%")]
    for label, value in m:
        st.markdown(f"**{value}** &nbsp;<span class='pq-muted'>{label}</span>",
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("58% of the evaluation set consists of questions the system "
               "should *not* answer.")

st.divider()

# ---------------------------------------------------------------- honesty
st.markdown("#### What it does not do")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Never moves money**")
    st.caption("Refunds, cancellations and account changes always require a "
               "human. This is enforced in code, not requested in a prompt.")
with c2:
    st.markdown("**Never invents policy**")
    st.caption("Answers come from retrieved documents. When the documentation "
               "does not cover something, it says so.")
with c3:
    st.markdown("**Not production-safe**")
    st.caption("The safety checks reduce risk; they do not eliminate it. See "
               "reports/safety_report.md for the honest limitations.")

st.divider()
n = svc.trace_count()
st.caption(f"{n} conversation{'s' if n != 1 else ''} logged this installation. "
           f"Use **Customer Support** in the sidebar to start.")
