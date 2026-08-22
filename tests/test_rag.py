"""Tests for the LLM and RAG layers (Phase 7).

None of these require a network call. The pipeline is built against an
interface, and the local extractive backend plus a scripted backend cover every
path including failure handling.
"""
import pytest

from src.config.settings import settings
from src.llm.client import (LocalExtractiveLLM, Message, ScriptedLLM,
                            available_backends, count_tokens, get_llm)
from src.llm.structured import parse_citation, parse_response, repair_json
from src.rag import evaluation as rev
from src.rag.abstention import Decision, decide
from src.rag.citations import check_grounding, verify_citations
from src.rag.context import assemble, budget_report
from src.rag.generator import build_pipeline
from src.rag.prompts import VERSIONS, get_prompt

pytestmark = pytest.mark.rag

needs_index = pytest.mark.skipif(
    not (settings.index_dir / "vectors.npy").exists(),
    reason="run scripts/build_index.py first",
)


# ------------------------------------------------------------- client
def test_local_backend_always_available():
    """The whole point of the abstraction: no key, no network, still runs."""
    assert available_backends()["local"] is True


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_llm("gpt5")


def test_local_llm_returns_structured_json():
    llm = LocalExtractiveLLM()
    prompt = ("CONTEXT:\n\n[1] SOURCE: POL-RET-002, p.1, S2\n"
              "Opened consumer electronics may be returned within 14 calendar "
              "days of delivery.\n\nQUESTION: How long to return an opened laptop?")
    r = llm.ask("system", prompt)
    parsed = parse_response(r.text)
    assert parsed.parse_ok
    assert "14 calendar days" in parsed.answer


def test_local_llm_does_not_echo_the_question():
    """Regression: the context parser ran past the QUESTION marker and the
    question text was selected as an answer sentence."""
    llm = LocalExtractiveLLM()
    prompt = ("CONTEXT:\n\n[1] SOURCE: X-YZ-001, p.1, S1\n"
              "The warranty period is 24 months from delivery.\n\n"
              "QUESTION: How long is the warranty on a laptop?")
    parsed = parse_response(llm.ask("s", prompt).text)
    assert "QUESTION:" not in parsed.answer


def test_local_llm_abstains_without_context():
    parsed = parse_response(
        LocalExtractiveLLM().ask("s", "QUESTION: anything at all?").text
    )
    assert parsed.needs_escalation


def test_scripted_backend_records_calls():
    llm = ScriptedLLM(['{"answer":"x","citations":[],"confidence":0.5,'
                       '"needs_escalation":false}'])
    llm.ask("sys", "user")
    assert len(llm.calls) == 1
    assert llm.calls[0][0].role == "system"


def test_token_accounting_is_populated():
    r = LocalExtractiveLLM().ask("system prompt here", "QUESTION: test?")
    assert r.prompt_tokens > 0
    assert r.total_tokens == r.prompt_tokens + r.completion_tokens


# --------------------------------------------------------- structured
@pytest.mark.parametrize("text,label", [
    ('{"answer":"a","citations":[],"confidence":0.5,"needs_escalation":false}', "clean"),
    ('```json\n{"answer":"a","citations":[],"confidence":0.5,'
     '"needs_escalation":false}\n```', "fenced"),
    ('Here: {"answer":"a","citations":[],"confidence":0.5,'
     '"needs_escalation":false} done', "prose wrapped"),
    ('{"answer":"a","citations":[],"confidence":0.5,}', "trailing comma"),
    ("{'answer':'a','citations':[],'confidence':0.5}", "single quotes"),
])
def test_json_repair_handles_common_malformations(text, label):
    data, _ = repair_json(text)
    assert data is not None, f"failed to repair: {label}"
    assert data["answer"] == "a"


def test_unparseable_output_escalates_rather_than_raising():
    """Raising would take down the request; escalating hands it to a human."""
    r = parse_response("total garbage {{{ not json")
    assert not r.parse_ok
    assert r.needs_escalation
    assert r.escalation_reason == "unparseable_model_output"


def test_confidence_out_of_range_is_clipped_and_flagged():
    r = parse_response('{"answer":"a","citations":[],"confidence":1.9,'
                       '"needs_escalation":false}')
    assert r.confidence == 1.0
    assert any("outside" in e for e in r.parse_errors)


@pytest.mark.parametrize("raw,ref,section", [
    ("POL-RET-002, p.1, S2", "POL-RET-002", "S2"),
    ("POL-REF-001 p.3 S7", "POL-REF-001", "S7"),
    ("MAN-PB14-001", "MAN-PB14-001", None),
])
def test_citation_parsing(raw, ref, section):
    c = parse_citation(raw)
    assert c.doc_ref == ref
    assert c.section == section


def test_abstention_detected_from_answer_text():
    r = parse_response('{"answer":"I don\'t have documentation covering that.",'
                       '"citations":[],"confidence":0.0,"needs_escalation":true}')
    assert r.is_abstention


# ------------------------------------------------------------ prompts
def test_all_prompt_versions_render():
    for v in VERSIONS:
        p = get_prompt(v)
        out = p.render(context="CTX", question="Q?")
        assert "CTX" in out and "Q?" in out


def test_unknown_prompt_version_raises():
    with pytest.raises(ValueError):
        get_prompt("v99")


def test_prompt_versions_increase_in_specificity():
    assert (count_tokens(get_prompt("v1").system)
            < count_tokens(get_prompt("v2").system)
            < count_tokens(get_prompt("v3").system))


def test_v3_covers_the_behaviours_it_claims():
    s = get_prompt("v3").system.lower()
    for behaviour in ["conflict", "escalat", "cite", "never invent"]:
        assert behaviour in s


# ----------------------------------------------------------- grounding
def test_verified_citation_is_marked():
    from src.llm.structured import Citation

    n, fab = verify_citations([Citation("POL-RET-002", 1, "S2")],
                              ["POL-RET-002, p.1, S2"])
    assert n == 1 and not fab


def test_fabricated_citation_is_caught():
    from src.llm.structured import Citation

    n, fab = verify_citations([Citation("POL-XYZ-999", 7, "S3")],
                              ["POL-RET-002, p.1, S2"])
    assert n == 0 and len(fab) == 1


CTX = ("[1] SOURCE: POL-RET-002, p.1, S2\nOpened consumer electronics may be "
       "returned within 14 calendar days of delivery.")
CTX_CITES = ["POL-RET-002, p.1, S2"]


def test_grounded_answer_passes():
    r = parse_response('{"answer":"You have 14 calendar days.",'
                       '"citations":["POL-RET-002, p.1, S2"],"confidence":0.9,'
                       '"needs_escalation":false}')
    assert check_grounding(r, CTX, CTX_CITES).is_grounded


def test_fabricated_number_is_caught():
    """Every planted hallucination trap in the corpus is numeric, so a number
    absent from the context is unambiguous evidence of fabrication."""
    r = parse_response('{"answer":"You have 45 days.",'
                       '"citations":["POL-RET-002, p.1, S2"],"confidence":0.9,'
                       '"needs_escalation":false}')
    g = check_grounding(r, CTX, CTX_CITES)
    assert not g.is_grounded
    assert "45" in g.unsupported_numbers


def test_uncited_answer_is_not_grounded():
    r = parse_response('{"answer":"14 days.","citations":[],"confidence":0.9,'
                       '"needs_escalation":false}')
    assert not check_grounding(r, CTX, CTX_CITES).is_grounded


# ------------------------------------------------------------ context
@needs_index
def _pipe(**kw):
    return build_pipeline(backend="local", **kw)


@needs_index
def test_context_carries_provenance():
    p = _pipe()
    res = p.retriever.retrieve("how long to return an opened laptop", top_k=5)
    ctx = assemble(res)
    assert ctx.n_chunks > 0
    assert all("SOURCE:" in ctx.text for _ in [1])
    assert len(ctx.citations) == ctx.n_chunks


@needs_index
def test_context_respects_token_budget():
    p = _pipe()
    res = p.retriever.retrieve("return policy", top_k=5)
    ctx = assemble(res, max_tokens=200)
    assert ctx.n_tokens <= 260, "budget exceeded"
    assert ctx.dropped > 0


def test_budget_report_adds_up():
    b = budget_report(300, 700, 20, max_response=512, window=8192)
    assert b["total_input"] == 1020
    assert b["headroom"] == 8192 - 1020 - 512


# ---------------------------------------------------------- abstention
@needs_index
def test_abstains_on_unanswerable_question():
    p = _pipe()
    r = p.answer("Do you offer student discounts?")
    assert r.trace.decision == "abstain"
    assert r.escalated
    assert r.trace.completion_tokens == 0, "model was called despite abstaining"


@needs_index
def test_answers_a_well_supported_question():
    p = _pipe()
    r = p.answer("How long do I have to return an opened laptop?")
    assert r.trace.decision in ("answer", "caveat")
    assert r.response.citations


@needs_index
def test_abstention_gate_runs_before_generation():
    """The gate exists to stop unanswerable questions reaching the model at
    all, which removes a class of hallucination rather than detecting it."""
    p = _pipe()
    r = p.answer("What are your carbon emissions per shipment?")
    assert r.trace.llm_backend == "none"


@needs_index
def test_version_conflict_escalates():
    p = _pipe()
    res = p.retriever.retrieve("what is your return policy", top_k=6,
                               include_archived=True)
    ctx = assemble(res)
    d = decide(res, ctx)
    assert d.decision == Decision.ESCALATE


# ------------------------------------------------------------ pipeline
@needs_index
def test_trace_is_complete():
    """The trace is the dashboard's data source, so every field must populate."""
    r = _pipe().answer("How long do I have to return an opened laptop?")
    t = r.trace
    for f in ["question", "answer", "decision", "retrieval_strategy",
              "prompt_version", "llm_backend"]:
        assert getattr(t, f), f"trace.{f} is empty"
    assert t.latency_ms > 0
    assert t.max_bm25 > 0
    assert isinstance(t.to_dict(), dict)


@needs_index
def test_ungrounded_answer_forces_escalation():
    """An answer that fails the grounding check is never shown as-is."""
    from src.rag.generator import RAGPipeline

    bad = ScriptedLLM(['{"answer":"You have 45 calendar days to return it.",'
                       '"citations":["POL-XYZ-999, p.9, S9"],"confidence":0.95,'
                       '"needs_escalation":false}'])
    p = _pipe()
    pipe = RAGPipeline(p.retriever, bad, prompt_version="v3")
    r = pipe.answer("How long do I have to return an opened laptop?")
    assert r.escalated
    assert not r.grounding.is_grounded
    assert r.response.confidence <= 0.3


@needs_index
def test_pipeline_survives_malformed_model_output():
    from src.rag.generator import RAGPipeline

    p = _pipe()
    pipe = RAGPipeline(p.retriever, ScriptedLLM(["not json at all {{{"]),
                       prompt_version="v3")
    r = pipe.answer("How long do I have to return an opened laptop?")
    assert r.escalated
    assert not r.trace.parse_ok


# ---------------------------------------------------------- evaluation
@needs_index
def test_abstention_rate_meets_threshold():
    res = rev.evaluate_abstention(_pipe())
    s = rev.summarize_abstention(res)
    assert s["abstention_rate"] >= 0.60, f"fell to {s['abstention_rate']}"


@needs_index
def test_false_abstention_stays_low():
    """A system that refuses everything scores perfectly on abstention and is
    useless, so the cost side is measured too."""
    fa = rev.evaluate_false_abstention(_pipe(), n=40)
    assert fa["false_abstention_rate"] <= 0.30


@needs_index
def test_extractive_backend_cannot_hallucinate():
    """Every word it emits came verbatim from retrieved context, so
    faithfulness is guaranteed by construction. This is the floor an LLM is
    measured against."""
    res = rev.evaluate_generation(_pipe())
    s = rev.summarize_generation(res)
    assert s["faithfulness"] == 1.0
    assert s["hallucination_rate"] == 0.0


@needs_index
def test_answers_carry_citations():
    res = rev.evaluate_generation(_pipe())
    s = rev.summarize_generation(res)
    assert s["answers_with_citation"] >= 0.85
    assert s["citation_accuracy"] >= 0.95
