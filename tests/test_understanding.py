"""Tests for the understanding layer (Phase 4)."""
import json
from pathlib import Path

import pytest

from src.eda import loaders
from src.understanding import entities as ent
from src.understanding import evaluation as ev
from src.understanding import sentiment as sent
from src.understanding.preprocessing import TextPreprocessor, mask_entities, normalize

pytestmark = pytest.mark.classification

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "intent_classifier.joblib"
needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason="run scripts/train_intent_classifier.py first"
)


# ------------------------------------------------------- preprocessing
@pytest.mark.parametrize("raw,expected_token", [
    ("where is my order PAC-2026-12345", "orderref"),
    ("status of #12347", "orderref"),
    ("order 12345 status", "orderref"),
    ("getting PAY-402 at checkout", "errorcode"),
    ("monitor shows ERR-DP-0x004", "errorcode"),
    ("refund of Rs 64,900", "moneyamount"),
])
def test_masking_replaces_identifiers(raw, expected_token):
    assert expected_token in mask_entities(raw)


def test_masking_removes_the_specific_value():
    """The entity TYPE is informative; the VALUE is noise."""
    a = mask_entities("where is order PAC-2026-12345")
    b = mask_entities("where is order PAC-2026-99999")
    assert a == b


def test_normalize_collapses_repeated_characters():
    assert normalize("waaaarrraaanty") == "waarraanty"


def test_preprocessor_is_sklearn_compatible():
    p = TextPreprocessor(mask=True)
    out = p.fit(["a"]).transform(["order 12345"])
    assert isinstance(out, list) and len(out) == 1


def test_masking_can_be_disabled_for_ablation():
    on = TextPreprocessor(mask=True).transform(["order 12345"])[0]
    off = TextPreprocessor(mask=False).transform(["order 12345"])[0]
    assert on != off
    assert "12345" in off


# ------------------------------------------------------------ entities
@pytest.mark.parametrize("raw,expected", [
    ("where is my order PAC-2026-12345", "PAC-2026-12345"),
    ("status of #12347 please", "PAC-2026-12347"),
    ("order 12345 kahan hai", "PAC-2026-12345"),
    ("pac-2026-12350 status", "PAC-2026-12350"),
])
def test_order_id_extraction_and_normalisation(raw, expected):
    """Argument extraction is the top agent failure mode, so it is
    deterministic rather than model-based - a regex cannot hallucinate."""
    assert expected in ent.extract(raw).order_ids


def test_error_code_extraction():
    e = ent.extract("getting PAY-402 and ERR-DP-0x004")
    assert "PAY-402" in e.error_codes
    assert any("ERR-DP" in c for c in e.error_codes)


def test_product_extraction_prefers_longest_match():
    e = ent.extract("my northwind ultra 15 is faulty")
    assert "northwind ultra 15" in e.products


def test_entities_empty_on_plain_greeting():
    assert ent.extract("hello").is_empty()


# ----------------------------------------------------------- sentiment
def test_anger_detected_as_negative():
    r = sent.score_sentiment("this is the THIRD time, i'm furious", intent="complaint")
    assert r.label == "negative"
    assert r.matched_negative


def test_gratitude_detected_as_positive():
    r = sent.score_sentiment("thanks, that was really helpful")
    assert r.label == "positive"


def test_neutral_query_is_not_scored_positive():
    """Regression: the intent prior once pushed neutral messages positive.
    A 27% negative base rate does not imply 73% positive."""
    for text, intent in [
        ("where is my order 12345", "order_tracking"),
        ("what is your return policy", "return_policy_question"),
        ("do you ship to germany", "shipping_delivery"),
    ]:
        assert sent.score_sentiment(text, intent=intent).label != "positive"


def test_negation_is_handled():
    r = sent.score_sentiment("no complaints at all, just a question")
    assert r.label != "negative"


def test_intent_prior_only_pushes_negative():
    high = sent.score_sentiment("update on this", intent="complaint").score
    low = sent.score_sentiment("update on this", intent="product_information").score
    assert high < low
    assert low <= 0.25, "a low negative base rate must not produce a positive label"


def test_sentiment_is_explainable():
    r = sent.score_sentiment("worst service ever", intent="complaint")
    assert "negative terms" in r.explain()


# ------------------------------------------------------------- urgency
def test_legal_threat_forces_high_urgency():
    """POL-CS-001 S3.4(d): a legal threat is a hard escalation trigger and
    must not be diluted by the weighted blend."""
    r = sent.score_urgency("i'm taking you to consumer court", intent="complaint")
    assert r.label == "high"
    assert r.signals["legal_threat"] > 0


def test_chargeback_threat_forces_high_urgency():
    assert sent.score_urgency("i'm doing a chargeback", intent="complaint").label == "high"


def test_plain_query_is_low_urgency():
    assert sent.score_urgency("what is your return policy",
                              intent="return_policy_question").label == "low"


def test_urgency_signals_are_reported_separately():
    r = sent.score_urgency("third time asking, urgent", intent="complaint",
                           sentiment="negative")
    assert set(r.signals) == {"lexical", "intent_prior", "sentiment",
                              "repeat_contact", "legal_threat"}


# ---------------------------------------------------------- evaluation
def test_group_split_shares_no_templates():
    """The core methodological fix: a random split leaked 180 template
    skeletons between halves, tying nine models at 0.9929."""
    df = loaders.load_intent_train()
    tr, va = ev.make_group_splits(df)
    shared = set(tr["text"].map(ev.template_skeleton)) & set(
        va["text"].map(ev.template_skeleton))
    assert len(shared) == 0


def test_random_split_does_leak_templates():
    """Documents the failure mode the group split exists to fix."""
    df = loaders.load_intent_train()
    tr, va = ev.make_splits(df)
    shared = set(tr["text"].map(ev.template_skeleton)) & set(
        va["text"].map(ev.template_skeleton))
    assert len(shared) > 50


def test_group_split_covers_all_classes():
    df = loaders.load_intent_train()
    _, va = ev.make_group_splits(df)
    assert va["intent"].nunique() == 11


def test_leakage_detection_finds_known_overlap():
    tr = loaders.load_intent_train()
    te = loaders.load_intent_test()
    assert len(ev.check_leakage(tr["text"], te["text"])) == 2


def test_drop_leaked_removes_exactly_those_rows():
    tr = loaders.load_intent_train()
    te = loaders.load_intent_test()
    clean = ev.drop_leaked(te, tr["text"])
    assert len(clean) == len(te) - 2
    assert not ev.check_leakage(tr["text"], clean["text"])


def test_evaluate_returns_all_required_metrics():
    r = ev.evaluate("t", "s", ["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
    for attr in ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
                 "macro_recall", "balanced_accuracy"]:
        assert 0.0 <= getattr(r, attr) <= 1.0
    assert r.confusion.shape == (2, 2)


# --------------------------------------------------------- trained model
@needs_model
def test_model_beats_majority_baseline_substantially():
    from src.understanding.pipeline import UnderstandingPipeline

    up = UnderstandingPipeline.load()
    te = ev.drop_leaked(loaders.load_intent_test(), loaders.load_intent_train()["text"])
    preds = [up.understand(t).intent for t in te["text"]]
    r = ev.evaluate("model", "test", te["intent"].astype(str), preds, loaders.INTENT_ORDER)
    assert r.macro_f1 > 0.50, "far below the recorded 0.61 - model regressed"
    assert r.macro_f1 > 10 * 0.0154, "no better than the majority baseline"


@needs_model
def test_reported_metrics_match_the_artifact():
    meta = json.loads((ROOT / "models" / "intent_classifier_metadata.json").read_text())
    assert meta["test_macro_f1"] > 0.55
    assert meta["validation_macro_f1"] > 0.90
    assert "group-aware" in meta["selection_metric"]
    assert len(meta["leaked_rows_removed"]) == 2


@needs_model
def test_pipeline_is_fast_enough_to_run_before_the_llm():
    from src.understanding.pipeline import UnderstandingPipeline

    up = UnderstandingPipeline.load()
    up.understand("warmup")
    times = [up.understand("where is my order 12345").latency_ms for _ in range(20)]
    assert sum(times) / len(times) < 50, "too slow to justify over an LLM call"


@needs_model
def test_multi_intent_flag_fires_on_compound_messages():
    from src.understanding.pipeline import UnderstandingPipeline

    up = UnderstandingPipeline.load()
    compound = up.understand("Where is my order and can I return it if it arrives tomorrow?")
    simple = up.understand("what is your return policy")
    assert compound.intent_margin < simple.intent_margin


@needs_model
def test_understanding_object_is_complete():
    from src.understanding.pipeline import UnderstandingPipeline

    u = UnderstandingPipeline.load().understand("my laptop is damaged, order 12345")
    assert u.intent in loaders.INTENT_ORDER
    assert u.sentiment in {"positive", "neutral", "negative"}
    assert u.urgency in {"low", "medium", "high"}
    assert "PAC-2026-12345" in u.entities.order_ids
    assert len(u.intent_top3) == 3
    assert isinstance(u.to_dict(), dict)


@needs_model
def test_model_artifact_fits_deployment_budget():
    """Streamlit Community Cloud allows ~1GB total."""
    assert MODEL.stat().st_size / 1024 < 2000, "model too large for the target"
