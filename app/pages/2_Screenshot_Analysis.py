"""Standalone screenshot analysis.

Shows what the vision layer extracts before it reaches the rest of the
pipeline, including the distinction between what was READ and what was
INFERRED.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui  # noqa: E402
from src.ui import service as svc  # noqa: E402

st.set_page_config(page_title="Screenshots · PacifyIQ", page_icon="🖼️",
                   layout="wide")
ui.inject_css()

ui.header("Screenshot analysis",
          "Upload an error message or a photo of the problem. The system "
          "reports what it can actually read, and says so when it cannot.")

upload = st.file_uploader(
    "Image", type=["png", "jpg", "jpeg", "webp", "gif"],
    help=f"Up to {svc.MAX_UPLOAD_MB} MB")
context = st.text_input("What were you doing? (optional)",
                        placeholder="e.g. trying to pay for my order")

if upload is None:
    st.info("Upload a screenshot to see what the system extracts from it.")
    st.markdown("#### What it looks for")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Error codes**")
    c1.caption("Codes like `PAY-402` or `ERR-DP-0x004` are matched against the "
               "troubleshooting guide.")
    c2.markdown("**Interface context**")
    c2.caption("Which screen this is — checkout, monitor menu, system "
               "notification.")
    c3.markdown("**Readability**")
    c3.caption("If the image is blurred or dark, it reports that rather than "
               "guessing at the text.")
    st.stop()

data = upload.getvalue()
c1, c2 = st.columns([1, 1])
with c1:
    st.image(data, caption=upload.name, use_container_width=True)

with c2:
    with st.spinner("Reading the image…"):
        res = svc.analyse_screenshot(data, upload.name, context)

    if not res.ok:
        ui.error_box(res.error, res.error_hint)
        v = res.validation or {}
        if v:
            with st.expander("Technical detail"):
                st.json({k: v.get(k) for k in
                         ("format", "width", "height", "size_bytes", "status")})
        st.stop()

    a = res.analysis
    ev = a.get("evidence", {})

    if not a.get("is_useful"):
        st.warning(f"**No usable information found** — {a.get('reason', '')}")
    else:
        st.success("Information extracted")

    ui.flow_label("What was found")

    code = a.get("error_code")
    level = ev.get("error_code", "unknown")
    if code:
        colour = {"visible": "green", "inferred": "orange"}.get(level, "grey")
        st.markdown(f"**Error code** &nbsp; `{code}` &nbsp;"
                    + ui.badge(level, colour), unsafe_allow_html=True)
        if level == "inferred":
            st.caption(
                "Read with low confidence. It is passed on as uncertain and is "
                "not used to search the knowledge base — a misread code would "
                "point at the wrong fix.")
    else:
        st.markdown("**Error code** &nbsp;" + ui.badge("not found", "grey"),
                    unsafe_allow_html=True)

    for label, key in [("Screen type", "image_type"),
                       ("Error message", "visible_error"),
                       ("Interface context", "ui_context")]:
        value = a.get(key)
        if value:
            lvl = ev.get(key, "unknown")
            colour = {"visible": "green", "inferred": "orange"}.get(lvl, "grey")
            st.markdown(f"**{label}** &nbsp; {value} &nbsp;"
                        + ui.badge(lvl, colour), unsafe_allow_html=True)

    obs = a.get("relevant_observations") or []
    if obs:
        st.markdown("**Also noticed**")
        for o in obs:
            st.caption(f"• {o}")

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Confidence", f"{a.get('confidence', 0):.0%}")
c2.metric("Text recognition", f"{a.get('ocr_confidence', 0):.0%}")
c3.metric("Time", f"{a.get('latency_ms', 0):.0f} ms")

v = res.validation or {}
warnings = v.get("warnings") or []
if warnings:
    st.markdown("#### Image quality")
    for w in warnings:
        ui.note(w)

with st.expander("How evidence levels work"):
    st.markdown("""
Every observation carries one of three levels, and the pipeline treats them
differently:

- **visible** — read directly and clearly. Safe to use for searching the
  knowledge base.
- **inferred** — deduced, or recovered from a misread. Passed on as uncertain,
  but **never** used to steer the search.
- **unknown** — could not be determined. Reported as absent.

A model asked "what is the error code?" on a blurred screenshot will produce a
plausible-looking code. That is worse than returning nothing, because it sends
the customer to the wrong fix. Hence the distinction.
""")

if a.get("is_useful") and code:
    st.divider()
    if st.button("Ask the assistant about this", type="primary"):
        st.session_state.prefill = context or f"I'm seeing error {code}"
        st.switch_page("pages/1_Support_Agent.py")
