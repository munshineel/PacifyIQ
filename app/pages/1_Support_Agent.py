"""Customer support agent — the main interface."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import components as ui  # noqa: E402
from src.ui import service as svc  # noqa: E402

st.set_page_config(page_title="Support · PacifyIQ", page_icon="🎧",
                   layout="wide")
ui.inject_css()

ui.header("Customer support",
          "Describe the problem in your own words. Attach a screenshot if you "
          "have one.")

if not svc.system_ready():
    ui.error_box("The assistant is not fully set up.",
                 "See the Home page for the setup checklist.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = uuid.uuid4().hex[:8]

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Try an example")
    for group, questions in svc.sample_questions().items():
        with st.expander(group):
            for q in questions:
                if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
                    st.session_state.prefill = q
                    st.rerun()

    st.divider()
    st.markdown("### Optional context")
    st.caption("Providing an order reference gives a more specific answer.")
    order_id = st.text_input("Order reference", placeholder="PAC-2026-12345")
    customer_id = st.text_input("Customer ID", placeholder="CUS-10001")

    st.divider()
    if st.session_state.history and st.button("Clear this conversation",
                                              use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------- input
with st.form("support", clear_on_submit=False):
    text = st.text_area(
        "Your question",
        value=st.session_state.pop("prefill", ""),
        height=110,
        placeholder="e.g. My monitor keeps going black and I don't know why",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        upload = st.file_uploader(
            "Screenshot (optional)",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            help="PNG or JPEG, up to 10 MB. A screenshot is usually clearer "
                 "than a photograph of a screen.")
    with c2:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Send", type="primary",
                                          use_container_width=True)

# ------------------------------------------------------------- process
if submitted:
    image_path = None
    upload_error = None

    if upload is not None:
        data = upload.getvalue()
        check = svc.analyse_screenshot(data, upload.name, text or "")
        if not check.ok:
            upload_error = (check.error, check.error_hint)
        else:
            suffix = Path(upload.name).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            image_path = tmp.name

    if upload_error:
        ui.error_box(*upload_error)
        st.caption("You can still send your question without the screenshot.")
    else:
        with st.spinner("Checking policies and order records…"):
            result = svc.ask(
                text, image_path=image_path,
                order_id=order_id or None, customer_id=customer_id or None,
                session_id=st.session_state.session_id)
        st.session_state.history.append(
            {"question": text, "result": result,
             "image": upload.name if upload else None})

# ------------------------------------------------------------- display
if st.session_state.history:
    st.divider()
    for i, turn in enumerate(reversed(st.session_state.history)):
        latest = i == 0
        result = turn["result"]

        with st.container():
            st.markdown('<div class="pq-flow">Customer issue</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**{turn['question']}**")
            if turn.get("image"):
                st.caption(f"📎 {turn['image']}")

            if latest:
                ui.render_result(result)
                ui.feedback_buttons(result)
            else:
                ui.status_banner(result)
                with st.expander("Show details"):
                    ui.render_result(result)
        st.divider()
else:
    st.info("Ask a question above, or pick an example from the sidebar.")
