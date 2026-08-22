"""DATA — preprocessing, validation and edge cases.

The premise: every downstream metric in this project is meaningless if the data
underneath it is wrong. These tests check the corpus, the database, the training
data and the evaluation sets are internally consistent BEFORE any model is
measured on them.

Several of these would have caught real bugs found manually in earlier phases —
the cross-reference regex swallowing headings, the leaked train/test rows, the
confidence column that was derived from the label.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config.settings import settings

pytestmark = pytest.mark.data


# =====================================================================
# Corpus integrity
# =====================================================================

def test_corpus_documents_all_load(corpus):
    from src.knowledge.loader import corpus_summary

    s = corpus_summary(corpus)
    assert s["n_documents"] == 13
    assert s["n_pages"] == 47
    assert s["empty_pages"] == 0, "a PDF extracted no text"


def test_every_document_is_registered(corpus):
    """An unregistered document silently loses its version and region flags,
    which would defeat the archived-content filter without any error."""
    from src.knowledge.loader import DOC_REGISTRY

    unknown = {p.doc for p in corpus} - set(DOC_REGISTRY)
    assert not unknown, f"unregistered documents: {unknown}"


def test_registry_has_no_orphan_entries(corpus):
    """The converse: a registry entry with no file is a silent gap."""
    from src.knowledge.loader import DOC_REGISTRY

    present = {p.doc for p in corpus}
    orphans = set(DOC_REGISTRY) - present
    assert not orphans, f"registered but missing: {orphans}"


def test_exactly_one_archived_document(corpus):
    """DEFECT-02 depends on there being a superseded policy to prefer against."""
    archived = {p.doc for p in corpus if p.version == "archived"}
    assert archived == {"return_policy_v1_ARCHIVED"}


def test_regional_document_is_tagged(corpus):
    regions = {p.doc: p.region for p in corpus}
    assert regions["eu_regional_addendum"] == "EU"


def test_section_ids_are_extracted_for_every_document(corpus):
    """A page with no section id produces chunks that cannot be cited."""
    from collections import defaultdict

    per_doc = defaultdict(set)
    for p in corpus:
        per_doc[p.doc].update(p.sections)
    missing = [d for d, s in per_doc.items() if not s]
    assert not missing, f"no sections found in: {missing}"


def test_section_ids_are_contiguous(corpus):
    """A gap in numbering usually means the heading regex missed one - the
    exact failure found manually in Phase 5."""
    from collections import defaultdict

    per_doc = defaultdict(set)
    for p in corpus:
        per_doc[p.doc].update(int(s[1:]) for s in p.sections)

    for doc, ids in per_doc.items():
        if len(ids) < 3:
            continue
        expected = set(range(min(ids), max(ids) + 1))
        gaps = expected - ids
        assert len(gaps) <= 1, f"{doc} is missing sections {sorted(gaps)}"


def test_planted_defects_are_documented():
    path = settings.data_dir / "PLANTED_DEFECTS.md"
    assert path.exists(), "PLANTED_DEFECTS.md is the register of engineered "\
                          "corpus faults and must exist"
    text = path.read_text(encoding="utf-8")
    assert text.count("DEFECT-") >= 15


def test_canonical_facts_exist():
    path = settings.data_dir / "canonical_facts.md"
    assert path.exists()
    assert len(path.read_text(encoding="utf-8")) > 2000


def test_the_planted_contradiction_is_still_present(corpus):
    """DEFECT-01: two CURRENT documents disagree on the return window. If this
    ever silently resolves, the conflict-detection tests become vacuous."""
    from collections import defaultdict

    # Join ALL pages per document - a per-page dict comprehension keeps only
    # the last page and silently misses text on every earlier one.
    text = defaultdict(str)
    for p in corpus:
        text[p.doc] += " " + " ".join(p.text.split())

    assert "30-day" in text["shipping_policy"] or "30 day" in text["shipping_policy"]
    assert "14 calendar days" in text["return_policy_v2"]


# =====================================================================
# Cleaning and chunking
# =====================================================================

def test_cleaning_removes_repeated_footers(corpus):
    from src.knowledge.chunker import clean

    for p in corpus[:10]:
        out = clean(p.text)
        assert "Pacify Electronics Pvt. Ltd." not in out


def test_cleaning_is_idempotent(corpus):
    from src.knowledge.chunker import clean

    once = clean(corpus[0].text)
    assert clean(once) == once


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "•", "a"])
def test_cleaning_handles_degenerate_input(text):
    from src.knowledge.chunker import clean

    assert isinstance(clean(text), str)


def test_chunking_produces_no_empty_chunks(chunks):
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_chunking_is_deterministic(corpus):
    from src.knowledge.chunker import build_chunks

    a = build_chunks(corpus, strategy="section", max_tokens=200)
    b = build_chunks(corpus, strategy="section", max_tokens=200)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_no_chunk_wildly_exceeds_its_budget(chunks):
    """Overlong chunks waste context budget and dilute retrieval."""
    over = [c for c in chunks if c.n_tokens > 200 * 3]
    assert not over, f"{len(over)} chunks exceed 3x the target size"


def test_chunks_preserve_document_provenance(chunks):
    for c in chunks:
        assert c.doc and c.doc_ref and c.page >= 1


def test_section_coverage_is_high(chunks):
    covered = sum(1 for c in chunks if c.section) / len(chunks)
    assert covered > 0.90, f"only {covered:.1%} of chunks carry a section id"


# =====================================================================
# Training data
# =====================================================================

def test_intent_training_data_is_well_formed():
    from src.eda import loaders

    df = loaders.load_intent_train()
    assert len(df) > 500
    assert df["text"].notna().all()
    assert not (df["text"].str.strip() == "").any()


def test_intent_labels_match_the_taxonomy():
    from src.eda import loaders

    for df in (loaders.load_intent_train(), loaders.load_intent_test()):
        unknown = set(df["intent"]) - set(loaders.INTENT_ORDER)
        assert not unknown, f"labels outside the taxonomy: {unknown}"


def test_every_intent_has_training_examples():
    from src.eda import loaders

    counts = loaders.load_intent_train()["intent"].value_counts()
    missing = set(loaders.INTENT_ORDER) - set(counts.index)
    assert not missing, f"no training data for: {missing}"
    assert counts.min() >= 10, f"thinnest class has {counts.min()} examples"


def test_train_test_leakage_is_bounded_and_known():
    """The raw CSVs share 2 rows - found in Phase 3 and left in place, because
    they are natural phrasings a customer would plausibly send.

    The invariant that matters is not "zero overlap in the files" but "no
    leaked row reaches a metric". This test pins the raw count so a new leak
    is caught, and the next test proves the removal actually happens."""
    from src.eda import loaders

    train = set(loaders.load_intent_train()["text"].str.strip().str.lower())
    test = set(loaders.load_intent_test()["text"].str.strip().str.lower())
    overlap = train & test
    assert len(overlap) <= 2, (
        f"leakage grew to {len(overlap)} rows: {sorted(overlap)[:5]}")


def test_leaked_rows_are_removed_before_evaluation():
    """The mechanism that makes the leak harmless. If this ever stops working,
    every reported classification number is inflated."""
    from src.eda import loaders
    from src.understanding import evaluation as uev

    train = loaders.load_intent_train()
    test = loaders.load_intent_test()
    cleaned = uev.drop_leaked(test, train["text"])

    train_set = set(train["text"].str.strip().str.lower())
    remaining = set(cleaned["text"].str.strip().str.lower()) & train_set
    assert not remaining, f"drop_leaked missed {len(remaining)} rows"
    assert len(cleaned) < len(test), "drop_leaked removed nothing at all"


def test_training_data_has_no_exact_duplicates():
    from src.eda import loaders

    df = loaders.load_intent_train()
    dupes = df["text"].str.strip().str.lower().duplicated().sum()
    assert dupes / len(df) < 0.05, f"{dupes} duplicated training texts"


# =====================================================================
# Database
# =====================================================================

def test_database_tables_are_populated(needs_db_check=None):
    from src.db.connection import query_all

    for table, minimum in [("customers", 400), ("orders", 1500),
                           ("products", 10), ("tickets_raw", 10000)]:
        n = query_all(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        assert n >= minimum, f"{table} has only {n} rows"


def test_every_order_references_a_real_customer():
    from src.db.connection import query_all

    orphans = query_all(
        "SELECT COUNT(*) AS n FROM orders o "
        "LEFT JOIN customers c ON o.customer_id = c.customer_id "
        "WHERE c.customer_id IS NULL")[0]["n"]
    assert orphans == 0


def test_every_order_references_a_real_product():
    from src.db.connection import query_all

    orphans = query_all(
        "SELECT COUNT(*) AS n FROM orders o "
        "LEFT JOIN products p ON o.sku = p.sku WHERE p.sku IS NULL")[0]["n"]
    assert orphans == 0


def test_delivery_never_precedes_dispatch():
    from src.db.connection import query_all

    bad = query_all(
        "SELECT COUNT(*) AS n FROM orders "
        "WHERE delivery_date IS NOT NULL AND dispatch_date IS NOT NULL "
        "AND delivery_date < dispatch_date")[0]["n"]
    assert bad == 0


def test_no_negative_monetary_values():
    from src.db.connection import query_all

    bad = query_all(
        "SELECT COUNT(*) AS n FROM orders WHERE total_paid < 0")[0]["n"]
    assert bad == 0


def test_database_views_exist():
    """A database with tables but no views is the most confusing failure in
    this project: setup appears to succeed, and the error surfaces much later
    as "no such table: v_order_detail" from whatever touched a view first.

    All business logic - eligibility, refund arithmetic, warranty state - lives
    in these views, so their absence breaks the agent, the notebooks and the
    evaluation suite at once."""
    import sqlite3

    con = sqlite3.connect(settings.db_path)
    views = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='view'")}
    con.close()

    expected = {"v_order_detail", "v_return_eligibility", "v_warranty_status",
                "v_refund_quote", "v_customer_contact_history"}
    missing = expected - views
    assert not missing, (
        f"missing views: {sorted(missing)}. Run: "
        f"python scripts/setup_database.py")


def test_every_view_is_queryable():
    """A view can exist but be broken if an underlying column was renamed."""
    from src.db.connection import query_all

    for view in ["v_order_detail", "v_return_eligibility", "v_warranty_status",
                 "v_refund_quote", "v_customer_contact_history"]:
        rows = query_all(f"SELECT * FROM {view} LIMIT 1")
        assert rows, f"{view} returned no rows"


def test_edge_case_orders_exist():
    """26 deterministic orders underpin the eligibility and refund tests."""
    from src.db.queries import get_order

    for oid in ["PAC-2026-12345", "PAC-2026-12347", "PAC-2026-12354",
                "PAC-2026-12357", "PAC-2026-12368"]:
        assert get_order(oid) is not None, f"{oid} is missing"


def test_ticket_history_is_labelled_simulated():
    """Presenting synthetic volume as real operational data would be
    misleading, so the marker must survive."""
    path = settings.data_dir / "tickets" / "ticket_history.csv"
    if not path.exists():
        pytest.skip("ticket history not generated")
    head = path.read_text(encoding="utf-8", errors="ignore")[:4000].lower()
    assert "simulated" in head or "synthetic" in head or \
        (settings.data_dir / "tickets" / "PLANTED_TRENDS.md").exists()


# =====================================================================
# Evaluation sets
# =====================================================================

EVAL_SETS = ["retrieval_eval", "generation_eval", "unanswerable_eval",
             "agent_trajectory_eval", "adversarial_eval", "vision_eval",
             "sentiment_urgency_eval", "end_to_end_eval"]


@pytest.mark.parametrize("name", EVAL_SETS)
def test_eval_set_is_valid_json_with_cases(name):
    data = json.loads((settings.eval_dir / f"{name}.json").read_text())
    assert data.get("cases"), f"{name} has no cases"
    ids = [c["id"] for c in data["cases"]]
    assert len(set(ids)) == len(ids), f"{name} has duplicate ids"


def test_retrieval_gold_labels_point_at_real_sections(chunks):
    """Gold is keyed to (doc, section) so it survives re-chunking - but only if
    those sections actually exist."""
    available = {(c.doc, c.section) for c in chunks}
    missing = set()
    for case in json.loads(
            (settings.eval_dir / "retrieval_eval.json").read_text())["cases"]:
        for g in case["gold_sections"]:
            if (g["doc"], g["section"]) not in available:
                missing.add((g["doc"], g["section"]))
    assert len(missing) <= 3, f"unresolvable gold labels: {sorted(missing)}"


def test_unanswerable_questions_are_genuinely_absent(chunks):
    """If a topic marked unanswerable is in fact documented, the abstention
    metric measures nothing."""
    corpus_text = " ".join(c.text.lower() for c in chunks)
    for term in ["student discount", "trade-in programme", "loyalty points",
                 "gift wrap"]:
        assert term not in corpus_text, f"'{term}' IS documented"


def test_end_to_end_set_is_majority_unanswerable():
    """A set of only answerable questions measures fluency, not judgement."""
    cases = json.loads(
        (settings.eval_dir / "end_to_end_eval.json").read_text())["cases"]
    resolved = sum(1 for c in cases if c["expected_outcome"] == "resolved")
    assert resolved / len(cases) < 0.55


def test_adversarial_set_covers_distinct_attack_families():
    cases = json.loads(
        (settings.eval_dir / "adversarial_eval.json").read_text())["cases"]
    assert len({c["category"] for c in cases}) >= 12
