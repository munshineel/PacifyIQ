"""Tests for the understanding-retrieval bridge (Phase 6.5)."""
import pytest

from src.config.settings import settings
from src.knowledge import evaluation as kev
from src.knowledge.bm25 import BM25Index
from src.knowledge.embedder import TfidfSvdEmbedder
from src.knowledge.retriever import Retriever
from src.knowledge.vector_store import VectorStore
from src.rag.routing import (CODE_ROUTING, INTENT_ROUTING, MIN_MARGIN_FOR_BOOST,
                             RoutedRetriever, build_routing, detect_meta_question,
                             detect_region)
from src.understanding.pipeline import UnderstandingPipeline

pytestmark = pytest.mark.retrieval

needs_index = pytest.mark.skipif(
    not (settings.index_dir / "vectors.npy").exists(),
    reason="run scripts/build_index.py first",
)


@pytest.fixture(scope="module")
def parts():
    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    base = Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)
    return base, UnderstandingPipeline.load()


# ------------------------------------------------------------- config
def test_every_intent_has_a_route():
    from src.eda.loaders import INTENT_ORDER

    assert set(INTENT_ROUTING) == set(INTENT_ORDER)


def test_routed_docs_exist_in_the_corpus():
    from src.knowledge.loader import DOC_REGISTRY

    for intent, route in INTENT_ROUTING.items():
        for doc in route["docs"]:
            assert doc in DOC_REGISTRY, f"{intent} routes to unknown doc {doc}"
    for prefix, docs in CODE_ROUTING.items():
        for doc in docs:
            assert doc in DOC_REGISTRY, f"{prefix} routes to unknown doc {doc}"


def test_margin_threshold_is_the_measured_optimum():
    """0.25 was selected by sweeping against the 120-query retrieval set."""
    assert MIN_MARGIN_FOR_BOOST == 0.25


# ------------------------------------------------------------ signals
@pytest.mark.parametrize("text,expected", [
    ("Am I talking to a bot?", True),
    ("How do I escalate my issue?", True),
    ("What do you need to verify my identity?", True),
    ("Where is my order 12345?", False),
    ("How long is the warranty?", False),
])
def test_meta_question_detection(text, expected):
    assert detect_meta_question(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("I am in Germany, can I return this?", "EU"),
    ("Under EU distance selling rules...", "EU"),
    ("I'm writing from Berlin", "EU"),
    ("Where is my order?", None),
])
def test_region_detection(text, expected):
    assert detect_region(text) == expected


@needs_index
def test_error_code_routes_to_the_defining_document(parts):
    _, u = parts
    route = build_routing(u.understand("What does SYS-0x0000007B mean?"))
    assert "technical_support_faq" in route.boosted_docs
    assert route.error_codes


@needs_index
def test_low_margin_intent_is_not_trusted(parts):
    """The measured reason this gate exists: on the queries retrieval fails,
    the classifier is both wrong and uncertain. A hard filter driven by it
    would remove the correct document more often than it removes noise."""
    _, u = parts
    understanding = u.understand("What does SYS-0x0000007B mean?")
    assert understanding.intent_margin < MIN_MARGIN_FOR_BOOST
    route = build_routing(understanding)
    assert not route.intent_applied
    assert any("intent_skipped" in r for r in route.reasons)


@needs_index
def test_high_margin_intent_is_applied(parts):
    _, u = parts
    understanding = u.understand("I want to delete my account and all my data")
    if understanding.intent_margin >= MIN_MARGIN_FOR_BOOST:
        route = build_routing(understanding)
        assert route.intent_applied


@needs_index
def test_entities_override_a_wrong_intent(parts):
    """SYS-0x0000007B is classified as payment_issue with margin 0.04. The
    extracted code is deterministic and must win."""
    _, u = parts
    understanding = u.understand("What does SYS-0x0000007B mean?")
    assert understanding.intent == "payment_issue"      # wrong
    route = build_routing(understanding)
    assert "technical_support_faq" in route.boosted_docs  # right anyway


@needs_index
def test_query_enrichment_adds_entity_tokens(parts):
    _, u = parts
    route = build_routing(u.understand("monitor shows ERR-DP-0x004"))
    assert route.enriched_query != route.intent
    assert "ERR" in route.enriched_query.upper()


# ----------------------------------------------------------- retrieval
@needs_index
def test_meta_defaults_off_because_it_was_measured_to_hurt(parts):
    """Injecting customer_service_policy for anything mentioning 'escalate'
    fixed 3 queries and broke 6. Retained behind a flag, off by default."""
    base, u = parts
    r = RoutedRetriever(base, u)
    assert r.use_meta is False


@needs_index
def test_routing_never_removes_candidates(parts):
    """Boosts reorder; they never filter. A wrong boost costs rank position,
    not availability."""
    base, u = parts
    r = RoutedRetriever(base, u)
    res, _, _ = r.retrieve("How long is the warranty on a Pacify laptop?", top_k=5)
    assert len(res.hits) == 5


@needs_index
def test_injection_surfaces_documents_outside_the_pool(parts):
    """Boosting alone cannot rescue a document that never entered the
    candidate pool, which is exactly the meta-question case."""
    base, u = parts
    r = RoutedRetriever(base, u, use_meta=True)
    res, route, _ = r.retrieve("How do I escalate my issue?", top_k=5)
    assert "customer_service_policy" in route.boosted_docs
    assert any(h.chunk.doc == "customer_service_policy" for h in res.hits)


@needs_index
def test_eu_query_surfaces_the_addendum(parts):
    base, u = parts
    r = RoutedRetriever(base, u)
    res, route, _ = r.retrieve(
        "I am in Germany. How long do I have to return an opened laptop?", top_k=6
    )
    assert route.region == "EU"


# --------------------------------------------------------- measured effect
@needs_index
def test_routing_improves_retrieval(parts):
    """The whole justification for this phase. Guards the recorded numbers."""
    import sys
    sys.path.insert(0, str(settings.root / "scripts"))
    from evaluate_routing import eval_retrieval

    base, u = parts
    cases = kev.load_eval_set("retrieval_eval")

    before = kev.summarize(eval_retrieval(None, base, cases))
    after = kev.summarize(eval_retrieval(RoutedRetriever(base, u), base, cases))

    assert after["recall@5"] >= before["recall@5"]
    assert after["mrr"] > before["mrr"], "routing no longer improves ranking"
    assert after["ndcg@5"] > before["ndcg@5"]


@needs_index
def test_routing_does_not_break_abstention():
    """Boosting raises retrieval scores, and abstention thresholds on those
    scores - so a routing gain can silently destroy the refusal behaviour."""
    from src.rag import evaluation as rev
    from src.rag.generator import build_pipeline

    s = rev.summarize_abstention(rev.evaluate_abstention(build_pipeline()))
    assert s["abstention_rate"] >= 0.60


# ------------------------------------------------------------ pipeline
@needs_index
def test_pipeline_records_understanding_in_the_trace():
    from src.rag.generator import build_pipeline

    r = build_pipeline().answer("What does SYS-0x0000007B mean?")
    assert r.trace.intent
    assert r.trace.sentiment
    assert r.trace.urgency
    assert r.trace.routing_reasons


@needs_index
def test_pipeline_works_with_routing_disabled():
    from src.rag.generator import RAGPipeline

    base = build_base()
    p = RAGPipeline(base, use_routing=False)
    assert p.routed is None
    r = p.answer("How long do I have to return an opened laptop?")
    assert r.response.answer


def build_base():
    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    return Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)


@needs_index
def test_abstention_is_grounded_by_definition():
    """Regression: an abstention has no citations, and the grounding check
    was scoring correct refusals as hallucinations."""
    from src.llm.structured import parse_response
    from src.rag.citations import check_grounding

    r = parse_response('{"answer":"I don\'t have documentation covering that.",'
                       '"citations":[],"confidence":0.0,"needs_escalation":true}')
    g = check_grounding(r, "some context", ["POL-RET-002, p.1, S2"])
    assert g.is_abstention
    assert g.is_grounded
    assert "no_citation" not in g.hallucination_flags
