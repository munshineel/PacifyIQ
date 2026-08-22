"""Tests for the knowledge base layer (Phase 5/6)."""
import numpy as np
import pytest

from src.config.settings import settings
from src.knowledge import evaluation as ev
from src.knowledge.bm25 import BM25Index, tokenize
from src.knowledge.chunker import (build_chunks, chunk_stats, clean,
                                   count_tokens, split_sections)
from src.knowledge.embedder import get_embedder
from src.knowledge.loader import DOC_REGISTRY, corpus_summary, find_sections, load_corpus
from src.knowledge.retriever import Retriever
from src.knowledge.vector_store import VectorStore

pytestmark = pytest.mark.retrieval

INDEX = settings.index_dir / "vectors.npy"
needs_index = pytest.mark.skipif(
    not INDEX.exists(), reason="run scripts/build_index.py first"
)


# ------------------------------------------------------------- loading
def test_corpus_loads_all_documents():
    s = corpus_summary(load_corpus())
    assert s["n_documents"] == 13
    assert s["n_pages"] == 47
    assert s["empty_pages"] == 0


def test_every_document_is_registered():
    """An unregistered document loses its version and region metadata, which
    would silently defeat the archived-content filter."""
    docs = {p.doc for p in load_corpus()}
    assert docs <= set(DOC_REGISTRY), f"unregistered: {docs - set(DOC_REGISTRY)}"


def test_archived_document_is_marked():
    pages = {p.doc: p for p in load_corpus()}
    assert pages["return_policy_v1_ARCHIVED"].version == "archived"
    assert pages["return_policy_v2"].version == "current"


def test_eu_addendum_is_region_tagged():
    pages = {p.doc: p for p in load_corpus()}
    assert pages["eu_regional_addendum"].region == "EU"


def test_section_ids_extracted():
    assert find_sections("S1. Scope\nS1.1 text\nS2. Windows") == ["S1", "S2"]


def test_cross_reference_does_not_swallow_next_heading():
    """Regression: `\\s+` in the heading regex matched newlines, so a line
    ending '...see POL-WAR-001 S10.' consumed the following heading and
    mislabelled an entire section."""
    text = "S2.6 Raise a claim under POL-WAR-001 S10.\nS3. Display problems\nS3.1 body"
    ids = [s[0] for s in split_sections(text)]
    assert "S3" in ids


# ------------------------------------------------------------ cleaning
def test_cleaner_strips_repeated_footers():
    t = "S1. Scope\nbody text\nPacify Electronics Pvt. Ltd.\nPage 3"
    out = clean(t)
    assert "Pacify Electronics Pvt. Ltd." not in out
    assert "body text" in out


def test_cleaner_rejoins_hyphenated_line_breaks():
    assert "warranty" in clean("war-\nranty period")


# ------------------------------------------------------------ chunking
def test_section_chunking_preserves_provenance():
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    assert len(chunks) > 100
    for c in chunks:
        assert c.doc and c.page >= 1 and c.doc_ref
        assert c.citation.startswith(c.doc_ref)


def test_chunk_ids_are_unique_and_deterministic():
    a = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    b = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    assert len({c.chunk_id for c in a}) == len(a)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_section_coverage_is_high():
    """A chunk with no section cannot be cited precisely."""
    s = chunk_stats(build_chunks(load_corpus(), strategy="section", max_tokens=200))
    assert s["section_coverage_pct"] > 90


def test_both_chunking_strategies_produce_sections():
    for strat in ("section", "fixed"):
        s = chunk_stats(build_chunks(load_corpus(), strategy=strat, max_tokens=200))
        assert s["section_coverage_pct"] > 80, f"{strat} lost section attribution"


def test_unknown_chunking_strategy_raises():
    with pytest.raises(ValueError):
        build_chunks(load_corpus(), strategy="nonsense")


def test_token_counting_is_sane():
    assert 3 <= count_tokens("the quick brown fox jumps") <= 10


# ----------------------------------------------------------- embedding
def test_embeddings_are_l2_normalised():
    """Cosine similarity is computed as a plain dot product, which is only
    correct if vectors are unit length."""
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    e = get_embedder("tfidf_svd", dim=64).fit([c.text for c in chunks])
    V = e.encode([c.text for c in chunks])
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic():
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)[:40]
    texts = [c.text for c in chunks]
    a = get_embedder("tfidf_svd", dim=32, seed=42).fit(texts).encode(texts)
    b = get_embedder("tfidf_svd", dim=32, seed=42).fit(texts).encode(texts)
    assert np.allclose(a, b)


def test_unknown_embedding_backend_raises():
    with pytest.raises(ValueError):
        get_embedder("word2vec")


# --------------------------------------------------------------- bm25
def test_bm25_tokenizer_preserves_error_codes():
    toks = tokenize("getting ERR-DP-0x004 on my monitor")
    assert "err-dp-0x004" in toks


def test_bm25_finds_exact_identifiers():
    """EDA finding 7e: codes fragment under subword tokenisation, so dense
    retrieval is weak on them and BM25 is the fix."""
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    bm = BM25Index(chunks)
    for code, doc in [("ERR-DP-0x004", "vision27"), ("PAY-402", "payment"),
                      ("THRM-88", None)]:
        hits = bm.search(code, top_k=3)
        assert hits, f"no BM25 hit for {code}"
        assert code.lower() in hits[0].chunk.text.lower()


# -------------------------------------------------------- vector store
def test_vector_store_rejects_mismatched_input():
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    with pytest.raises(ValueError):
        VectorStore(chunks, np.zeros((len(chunks) - 1, 8), dtype=np.float32))


def test_faiss_matches_brute_force_exactly():
    """FAISS is offered as an alternative backend, not a different answer."""
    faiss = pytest.importorskip("faiss")
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    e = get_embedder("tfidf_svd", dim=64).fit([c.text for c in chunks])
    V = e.encode([c.text for c in chunks])
    q = e.encode_one("how long do I have to return an opened laptop")

    a = VectorStore(chunks, V, backend="numpy").search(q, top_k=5)
    b = VectorStore(chunks, V, backend="faiss").search(q, top_k=5)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


# ---------------------------------------------------------- retrieval
@needs_index
def _retriever(strategy="rrf_w", **kw):
    from src.knowledge.embedder import TfidfSvdEmbedder

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    return Retriever(store, emb, BM25Index(store.chunks), strategy=strategy, **kw)


@needs_index
def test_archived_content_is_excluded_by_default():
    """DEFECT-02: the superseded v1 policy is a CLOSER lexical match to a naive
    query than the current v2, so similarity alone ranks it first. Only
    metadata filtering fixes this."""
    r = _retriever()
    default = r.retrieve("what is your return policy", top_k=5)
    assert all(h.chunk.is_current for h in default.hits)

    unfiltered = r.retrieve("what is your return policy", top_k=5, include_archived=True)
    assert any(not h.chunk.is_current for h in unfiltered.hits), \
        "v1 no longer surfaces - the version-preference test has stopped being meaningful"


@needs_index
def test_filters_apply_before_ranking():
    r = _retriever()
    res = r.retrieve("warranty period", top_k=5, doc_type="policy")
    assert res.hits
    assert all(h.chunk.doc_type == "policy" for h in res.hits)


@needs_index
def test_unknown_strategy_raises():
    store = VectorStore.load(settings.index_dir)
    from src.knowledge.embedder import TfidfSvdEmbedder
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    with pytest.raises(ValueError):
        Retriever(store, emb, strategy="magic")


@needs_index
def test_conflict_flag_fires_on_mixed_evidence():
    r = _retriever()
    res = r.retrieve("what is your return policy", top_k=6, include_archived=True)
    assert res.has_conflict()


# --------------------------------------------------------- evaluation
def test_gold_labels_resolve_to_real_sections():
    """Gold references (doc, section), not chunk IDs, so it survives every
    chunking ablation. That only works if the sections actually exist."""
    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    available = {(c.doc, c.section) for c in chunks}
    missing = set()
    for case in ev.load_eval_set():
        for g in case["gold_sections"]:
            if (g["doc"], g["section"]) not in available:
                missing.add((g["doc"], g["section"]))
    assert len(missing) <= 3, f"unresolvable gold labels: {sorted(missing)}"


def test_metric_functions_behave():
    gold = {("a", "S1"), ("b", "S2")}
    retrieved = [("x", "S9"), ("a", "S1"), ("b", "S2")]
    assert ev.recall_at_k(retrieved, gold, 1) == 0.0
    assert ev.recall_at_k(retrieved, gold, 2) == 1.0
    assert ev.coverage_at_k(retrieved, gold, 3) == 1.0
    assert ev.coverage_at_k(retrieved, gold, 2) == 0.5
    assert ev.reciprocal_rank(retrieved, gold) == 0.5
    assert 0.0 < ev.ndcg_at_k(retrieved, gold, 3) <= 1.0


def test_ndcg_rewards_higher_ranking():
    gold = {("a", "S1")}
    high = ev.ndcg_at_k([("a", "S1"), ("x", "S9")], gold, 5)
    low = ev.ndcg_at_k([("x", "S9"), ("a", "S1")], gold, 5)
    assert high > low


@needs_index
def test_retrieval_quality_meets_threshold():
    """Guards the recorded result. A drop means something regressed."""
    r = _retriever()
    summ = ev.summarize(ev.evaluate_all(r, ev.load_eval_set(), strategy="rrf_w"))
    assert summ["recall@5"] >= 0.80, f"recall@5 fell to {summ['recall@5']}"
    assert summ["mrr"] >= 0.55, f"MRR fell to {summ['mrr']}"


@needs_index
def test_authority_weighting_helps():
    """Measured improvement, not an assumption: encoding the corpus's own
    stated precedence (policy governs over FAQ) lifted recall@5 by +0.067."""
    cases = ev.load_eval_set()
    off = ev.summarize(ev.evaluate_all(
        _retriever("hybrid", authority_weighting=False), cases, strategy="hybrid"))
    on = ev.summarize(ev.evaluate_all(
        _retriever("hybrid", authority_weighting=True), cases, strategy="hybrid"))
    assert on["recall@5"] > off["recall@5"]
    assert on["mrr"] > off["mrr"]


@needs_index
def test_lexical_queries_are_handled():
    """The 8 bare-error-code queries are what hybrid retrieval exists for."""
    r = _retriever()
    cases = [c for c in ev.load_eval_set() if c.get("type") == "lexical"]
    res = ev.evaluate_all(r, cases, strategy="rrf_w")
    assert ev.summarize(res)["recall@5"] >= 0.75


@needs_index
def test_index_fits_deployment_budget():
    total_mb = sum(f.stat().st_size for f in settings.index_dir.glob("*")) / 1024 / 1024
    assert total_mb < 100, f"index is {total_mb:.1f} MB"
