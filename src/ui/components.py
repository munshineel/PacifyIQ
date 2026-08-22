"""Shared UI components.

Presentation only. These render what the service layer returns; they contain no
business logic and never call the backend directly.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------
# Palette. Kept in one place so the four pages stay visually consistent.
# ---------------------------------------------------------------------
NAVY = "#1a2b45"
SLATE = "#4a5560"
MUTED = "#7a828a"
LINE = "#dde1e6"
GREEN = "#1e7a4c"
AMBER = "#b5730f"
RED = "#b03a2e"
GREY = "#6b7280"

STATUS_STYLE = {
    "green": (GREEN, "#e8f4ee"),
    "orange": (AMBER, "#fdf3e3"),
    "red": (RED, "#fbeae8"),
    "grey": (GREY, "#f1f3f5"),
}


def inject_css() -> None:
    """One stylesheet for the whole app."""
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
          h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
          .pq-badge {{
            display:inline-block; padding:3px 11px; border-radius:11px;
            font-size:0.78rem; font-weight:600; margin-right:6px;
          }}
          .pq-card {{
            border:1px solid {LINE}; border-radius:8px; padding:14px 16px;
            margin-bottom:10px; background:#fff;
          }}
          .pq-card-accent {{ border-left:4px solid {NAVY}; }}
          .pq-label {{
            font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
            color:{MUTED}; font-weight:700; margin-bottom:3px;
          }}
          .pq-value {{ font-size:0.94rem; color:#22262b; }}
          .pq-muted {{ color:{MUTED}; font-size:0.84rem; }}
          .pq-flow {{
            font-size:0.72rem; letter-spacing:0.09em; color:{MUTED};
            text-transform:uppercase; font-weight:700;
            border-bottom:1px solid {LINE}; padding-bottom:5px;
            margin:18px 0 10px 0;
          }}
          .pq-note {{
            background:#fdf9ec; border-left:3px solid {AMBER};
            padding:9px 13px; border-radius:4px; font-size:0.87rem;
            margin-bottom:8px; color:#5c4708;
          }}
          .pq-sim {{
            background:#eef1f5; border:1px dashed {SLATE};
            padding:9px 13px; border-radius:5px; font-size:0.84rem;
            color:{SLATE}; margin-bottom:14px;
          }}
          div[data-testid="stMetricValue"] {{ font-size:1.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, colour: str = "grey") -> str:
    fg, bg = STATUS_STYLE.get(colour, STATUS_STYLE["grey"])
    return (f'<span class="pq-badge" style="color:{fg};background:{bg};">'
            f'{text}</span>')


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="pq-muted">{subtitle}</div>',
                    unsafe_allow_html=True)
    st.write("")


def flow_label(text: str) -> None:
    """Section marker matching the documented request flow."""
    st.markdown(f'<div class="pq-flow">{text}</div>', unsafe_allow_html=True)


def field(label: str, value: Any) -> None:
    st.markdown(
        f'<div class="pq-label">{label}</div>'
        f'<div class="pq-value">{value}</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="pq-note">{text}</div>', unsafe_allow_html=True)


def simulated_banner(detail: str) -> None:
    """Marks synthetic data wherever it appears.

    Non-negotiable: presenting simulated volume as real operational data would
    misrepresent the system.
    """
    st.markdown(
        f'<div class="pq-sim"><b>Simulated data.</b> {detail}</div>',
        unsafe_allow_html=True,
    )


def error_box(message: str, hint: str = "") -> None:
    st.error(message)
    if hint:
        st.caption(hint)


# =====================================================================
# Result rendering
# =====================================================================

SENTIMENT_COLOUR = {"negative": "red", "neutral": "grey", "positive": "green"}
URGENCY_COLOUR = {"high": "red", "medium": "orange", "low": "grey"}


def status_banner(result) -> None:
    """Headline outcome with a visual indicator."""
    fg, bg = STATUS_STYLE.get(result.status_colour, STATUS_STYLE["grey"])
    extra = ""
    if result.low_confidence:
        extra = badge("Low confidence", "orange")
    st.markdown(
        f'<div class="pq-card pq-card-accent" '
        f'style="border-left-color:{fg};background:{bg};">'
        f'<span style="font-size:1.05rem;font-weight:600;color:{fg};">'
        f'{result.status_icon} {result.status_label}</span> {extra}</div>',
        unsafe_allow_html=True,
    )


def understanding_panel(result) -> None:
    flow_label("Understanding")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        field("Intent", result.intent.replace("_", " ") or "—")
    with c2:
        st.markdown('<div class="pq-label">Sentiment</div>',
                    unsafe_allow_html=True)
        st.markdown(badge(result.sentiment or "—",
                          SENTIMENT_COLOUR.get(result.sentiment, "grey")),
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="pq-label">Urgency</div>',
                    unsafe_allow_html=True)
        st.markdown(badge(result.urgency or "—",
                          URGENCY_COLOUR.get(result.urgency, "grey")),
                    unsafe_allow_html=True)
    with c4:
        field("Confidence", f"{result.confidence:.0%}")


def evidence_panel(result) -> None:
    flow_label("Evidence")
    if result.image:
        img = result.image
        code = img.get("error_code")
        with st.container():
            if code:
                lvl = img.get("evidence", "unknown")
                colour = {"visible": "green", "inferred": "orange"}.get(lvl, "grey")
                st.markdown(
                    f'**Screenshot** — error code `{code}` '
                    f'{badge(lvl, colour)}', unsafe_allow_html=True)
                if lvl == "inferred":
                    st.caption(
                        "Read with low confidence, so it was not used to search "
                        "the knowledge base.")
            else:
                st.markdown("**Screenshot** — no error information found")

    if not result.sources:
        st.caption("No policy documents were needed for this answer.")
        return

    for s in result.sources:
        cols = st.columns([5, 2, 2])
        with cols[0]:
            st.markdown(f"**{s.get('title', '')}**")
            st.caption(s["citation"])
        with cols[1]:
            st.caption(s.get("type", ""))
        with cols[2]:
            if s.get("version") == "archived":
                st.markdown(badge("superseded", "red"), unsafe_allow_html=True)
            else:
                st.caption(s.get("topic", ""))


def action_panel(result) -> None:
    flow_label("Actions taken")
    if not result.trajectory and not result.actions:
        st.caption("No tools were needed.")
    else:
        rows = result.trajectory or [{"tool": a, "args": {}, "status": ""}
                                     for a in result.actions]
        for step in rows:
            args = ", ".join(f"{k}={v}" for k, v in (step.get("args") or {}).items())
            ok = step.get("status") in ("ok", "")
            icon = "✓" if ok else "✗"
            st.markdown(
                f"`{icon}` **{step['tool']}**"
                + (f" &nbsp;<span class='pq-muted'>{args}</span>" if args else ""),
                unsafe_allow_html=True,
            )
    if result.tools_skipped:
        with st.expander(f"{len(result.tools_skipped)} tool(s) considered and "
                         f"skipped"):
            for s in result.tools_skipped:
                st.caption(f"**{s.get('tool')}** — {s.get('reason')}")


def result_panel(result) -> None:
    flow_label("Result")
    st.markdown(result.answer or "_No answer produced._")

    for c in result.caveats:
        note(c)

    if result.missing_information:
        st.info("Still needed: " + ", ".join(
            m.replace("_", " ") for m in result.missing_information))

    if result.escalated:
        flow_label("Escalation")
        reason = (result.escalation_reason or "").replace("_", " ")
        st.warning(f"**Handed to a colleague** — {reason}")
        if result.ticket_id:
            st.caption(f"Reference: `{result.ticket_id}`")


def metadata_panel(result) -> None:
    """Concise decision metadata. Deliberately NOT chain of thought - what was
    done and why, at the level of actions and evidence."""
    with st.expander("Decision details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", result.steps)
        c2.metric("Tools", len(result.actions))
        c3.metric("Sources", len(result.citations))
        c4.metric("Latency", f"{result.latency_ms:.0f} ms")

        if result.intent_margin:
            st.caption(f"Intent margin {result.intent_margin:.2f} — the gap to "
                       f"the runner-up class. Below 0.15 the message may carry "
                       f"more than one intent.")

        rules = []
        for stage in (result.guardrails or {}).values():
            if isinstance(stage, dict):
                rules.extend(stage.get("rules_fired", []) or [])
        if rules:
            st.caption("Safety rules triggered: " + ", ".join(sorted(set(rules))))
        else:
            st.caption("Safety checks: no rules triggered.")


def render_result(result) -> None:
    """The full request flow, in the documented order."""
    if not result.ok:
        error_box(result.error, result.error_hint)
        return

    status_banner(result)
    understanding_panel(result)
    evidence_panel(result)
    action_panel(result)
    result_panel(result)
    metadata_panel(result)


def feedback_buttons(result, key_prefix: str = "fb") -> None:
    if not result.ok or not result.trace_id:
        return
    from src.ui import service as svc

    c1, c2, _ = st.columns([1, 1, 8])
    if c1.button("👍", key=f"{key_prefix}_up_{result.trace_id}",
                 help="This was helpful"):
        svc.record_feedback(result.trace_id, "up")
        st.toast("Thanks for the feedback.")
    if c2.button("👎", key=f"{key_prefix}_down_{result.trace_id}",
                 help="This was not helpful"):
        svc.record_feedback(result.trace_id, "down")
        st.toast("Thanks — this has been flagged for review.")
