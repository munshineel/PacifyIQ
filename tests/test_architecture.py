"""Architecture tests (Phase 13).

The layering claims in the README are only worth making if something enforces
them. These tests fail the moment a boundary is crossed.
"""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP = ROOT / "app"


def _imports(path: Path) -> set[str]:
    """Top-level module names imported by a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]


# ============================================== streamlit stays out of src
def test_backend_never_imports_streamlit():
    """If a backend module grew a `st.spinner`, the whole backend would become
    untestable without a browser."""
    offenders = [
        str(p.relative_to(ROOT)) for p in _py_files(SRC)
        if "src/ui" not in str(p).replace("\\", "/")
        and any(m.split(".")[0] == "streamlit" for m in _imports(p))
    ]
    assert not offenders, f"streamlit imported in backend: {offenders}"


def test_ui_components_are_presentation_only():
    """components.py may use streamlit, but must not reach into the backend
    itself - that is the service layer's job."""
    imports = _imports(SRC / "ui" / "components.py")
    forbidden = [m for m in imports
                 if m.startswith(("src.agent", "src.rag", "src.knowledge",
                                  "src.guardrails", "src.evaluation",
                                  "src.multimodal"))]
    assert not forbidden, f"components.py reaches into the backend: {forbidden}"


# ================================================ pages go through service
def test_pages_only_import_the_ui_layer():
    """Pages must not import the agent, retriever or guardrails directly."""
    if not APP.exists():
        pytest.skip("app/ not present")
    offenders = {}
    for page in _py_files(APP):
        bad = [m for m in _imports(page)
               if m.startswith("src.") and not m.startswith("src.ui")]
        if bad:
            offenders[page.name] = bad
    assert not offenders, f"pages bypass the service layer: {offenders}"


def test_every_page_exists():
    expected = ["Home.py", "1_Support_Agent.py", "2_Screenshot_Analysis.py",
                "3_Knowledge_Base.py", "4_Conversation_History.py",
                "5_Analytics.py", "6_Evaluation.py"]
    present = {p.name for p in _py_files(APP)}
    missing = [e for e in expected if e not in present]
    assert not missing, f"missing pages: {missing}"


def test_all_pages_parse():
    for page in _py_files(APP):
        ast.parse(page.read_text(encoding="utf-8"))


# ============================================ guardrails stay independent
def test_guardrails_do_not_import_what_they_veto():
    """Guardrails must be able to veto the agent and the RAG pipeline, so they
    cannot depend on them."""
    for f in _py_files(SRC / "guardrails"):
        bad = [m for m in _imports(f)
               if m.startswith(("src.agent", "src.rag"))]
        assert not bad, f"{f.name} imports {bad}"


def test_evaluation_is_not_imported_by_runtime_code():
    """The evaluation framework is a measurement tool, not a runtime
    dependency."""
    offenders = []
    for f in _py_files(SRC):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith(("src/evaluation", "src/ui")):
            continue
        if any(m.startswith("src.evaluation") for m in _imports(f)):
            offenders.append(rel)
    assert not offenders, f"runtime code imports evaluation: {offenders}"


# ==================================================== service behaviour
def test_service_handles_empty_input():
    from src.ui import service as svc

    r = svc.ask("", log=False)
    assert not r.ok
    assert "type your question" in r.error.lower()


def test_service_handles_absurdly_long_input():
    from src.ui import service as svc

    r = svc.ask("x" * 6000, log=False)
    assert not r.ok


def test_service_rejects_oversized_upload():
    from src.ui import service as svc

    r = svc.analyse_screenshot(b"x" * (11 * 1024 * 1024), "big.png")
    assert not r.ok
    assert "MB" in r.error


def test_service_rejects_empty_upload():
    from src.ui import service as svc

    assert not svc.analyse_screenshot(b"", "empty.png").ok


def test_service_rejects_corrupt_upload():
    from src.ui import service as svc

    r = svc.analyse_screenshot(b"not an image at all, definitely not", "x.png")
    assert not r.ok
    assert r.error_hint


def test_every_error_carries_a_hint():
    """A customer-facing surface must never show a bare failure."""
    from src.ui import service as svc

    for r in [svc.ask("", log=False),
              svc.analyse_screenshot(b"", "e.png"),
              svc.analyse_screenshot(b"junk", "e.png")]:
        assert r.error
        assert getattr(r, "error_hint", "")


def test_system_check_reports_every_component():
    from src.ui import service as svc

    names = {c.name for c in svc.check_system()}
    for expected in ["Knowledge index", "Intent classifier",
                     "Operational database", "Knowledge corpus"]:
        assert expected in names


# ======================================================= result contract
@pytest.mark.skipif(
    not (ROOT / "data" / "index" / "vectors.npy").exists(),
    reason="run scripts/build_index.py first")
def test_query_result_exposes_the_documented_flow():
    """The UI renders understanding -> evidence -> action -> result ->
    escalation. Every field that flow needs must be present."""
    from src.ui import service as svc

    r = svc.ask("How many dead pixels before replacement?", log=False)
    assert r.ok
    for attr in ["intent", "sentiment", "urgency", "confidence", "sources",
                 "actions", "trajectory", "answer", "resolution_status",
                 "escalated", "escalation_reason", "status_label",
                 "status_icon", "status_colour"]:
        assert hasattr(r, attr), f"QueryResult is missing {attr}"


@pytest.mark.skipif(
    not (ROOT / "data" / "index" / "vectors.npy").exists(),
    reason="run scripts/build_index.py first")
def test_result_never_exposes_chain_of_thought():
    from src.ui import service as svc

    r = svc.ask("Can I return order PAC-2026-12345?", log=False)
    blob = str(r.__dict__).lower()
    for leak in ["let me think", "step 1:", "first, i will", "reasoning:",
                 "my thought"]:
        assert leak not in blob


@pytest.mark.skipif(
    not (ROOT / "data" / "index" / "vectors.npy").exists(),
    reason="run scripts/build_index.py first")
def test_status_indicators_cover_every_outcome():
    from src.ui import service as svc

    seen = set()
    for q in ["How many dead pixels before replacement?",
              "Where is my order?",
              "I want to return PAC-2026-12345 and get a refund",
              "Who won the cricket match?"]:
        r = svc.ask(q, log=False)
        seen.add(r.resolution_status)
        assert r.status_icon and r.status_label
        assert r.status_colour in ("green", "orange", "red", "grey")
    assert len(seen) >= 3, f"outcomes not differentiated: {seen}"


# ============================================================ traces
def test_traces_survive_a_bad_decision_object():
    """A logging failure must never take down a customer request."""
    from src.observability import traces

    assert traces.record(object()) == ""


def test_trace_redaction_is_applied():
    from src.observability import traces

    tid = traces.record(
        type("D", (), {"to_dict": lambda self: {
            "answer": "ok", "resolution_status": "resolved"}})(),
        session_id="test_redaction",
        question="my card is 4111 1111 1111 1111")
    assert tid
    df = traces.load(limit=5, session_id="test_redaction")
    assert not df.empty
    assert "4111 1111 1111 1111" not in df.iloc[0]["question"]
