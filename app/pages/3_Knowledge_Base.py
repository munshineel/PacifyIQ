"""Knowledge base browser and search."""
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

st.set_page_config(page_title="Knowledge · PacifyIQ", page_icon="📚",
                   layout="wide")
ui.inject_css()

ui.header("Knowledge base",
          "Every answer is grounded in these documents. Nothing is generated "
          "from the model's own knowledge.")

ui.simulated_banner(
    "Pacify Electronics Pvt. Ltd. is a fictional retailer. These policies were "
    "written for this project and do not describe any real company.")

tab_search, tab_docs = st.tabs(["Search", "Documents"])

with tab_search:
    query = st.text_input("Search the knowledge base",
                          placeholder="e.g. restocking fee on opened laptops")
    top_k = st.slider("Passages to return", 3, 10, 5)

    if query:
        with st.spinner("Searching…"):
            res = svc.search_knowledge(query, top_k=top_k)

        if not res.get("ok"):
            ui.error_box(res.get("error", "Search failed."))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Passages", res.get("n_results", 0))
            c2.metric("Best keyword score", f"{res.get('max_bm25', 0):.1f}")
            c3.metric("Best semantic score", f"{res.get('max_cosine', 0):.2f}")

            if res.get("has_conflict"):
                ui.note("The retrieved passages span different policy versions "
                        "or regions. In a live request this triggers a handover "
                        "rather than an answer.")

            st.divider()
            for i, ch in enumerate(res.get("chunks", []), start=1):
                with st.container():
                    head = f"**{i}. {ch['citation']}**"
                    if ch.get("version") == "archived":
                        head += " " + ui.badge("superseded", "red")
                    if ch.get("region") and ch["region"] != "all":
                        head += " " + ui.badge(ch["region"], "orange")
                    st.markdown(head, unsafe_allow_html=True)
                    st.caption(f"score {ch.get('score', 0):.4f}")
                    st.markdown(
                        f'<div class="pq-card">{ch["text"][:900]}'
                        f'{"…" if len(ch["text"]) > 900 else ""}</div>',
                        unsafe_allow_html=True)
    else:
        st.caption("Search across policies, FAQs, troubleshooting guides and "
                   "product manuals.")

with tab_docs:
    docs = svc.list_documents()
    if not docs:
        ui.error_box("No documents found.",
                     "Check that data/documents/ contains the corpus.")
    else:
        df = pd.DataFrame(docs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(df))
        c2.metric("Policies", int((df["type"] == "policy").sum()))
        c3.metric("Superseded", int((df["version"] == "archived").sum()))

        st.caption("One document is deliberately superseded. Retrieval filters "
                   "it out by default, because a shorter, less qualified policy "
                   "is often a *closer* text match than the current one.")

        types = st.multiselect("Filter by type", sorted(df["type"].unique()),
                               default=list(sorted(df["type"].unique())))
        view = df[df["type"].isin(types)]
        st.dataframe(view, use_container_width=True, hide_index=True)
