"""Tests for the EDA layer.

Loaders must be stable (Phase 4+ depends on them) and the audit must keep
flagging the known data defects so they cannot silently disappear.
"""
import pandas as pd
import pytest

from src.eda import audit, loaders, text_stats

pytestmark = pytest.mark.data


# ------------------------------------------------------------- loaders
def test_tickets_load_with_correct_shape():
    df = loaders.load_tickets()
    assert len(df) == 11_905
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df["ticket_id"].is_unique


def test_tickets_have_derived_columns():
    df = loaders.load_tickets()
    for col in ["date", "hour", "weekday", "week", "days_ago", "escalated", "has_feedback"]:
        assert col in df.columns


def test_feedback_blanks_are_empty_string_not_nan():
    """Blank feedback must not become NaN, or downstream string ops break."""
    df = loaders.load_tickets()
    assert df["feedback"].isna().sum() == 0
    assert (df["feedback"] == "").sum() > 0


def test_intent_splits_have_expected_size():
    assert len(loaders.load_intent_train()) == 2_200
    assert len(loaders.load_intent_test()) == 142


def test_all_eleven_intents_present_in_both_splits():
    for df in (loaders.load_intent_train(), loaders.load_intent_test()):
        assert set(df["intent"].dropna().unique()) == set(loaders.INTENT_ORDER)


def test_corpus_extracts_text_from_every_page():
    df = loaders.load_corpus()
    assert df["doc"].nunique() == 13
    assert len(df) == 47
    assert (df["n_chars"] > 0).all(), "a page extracted no text"


# ---------------------------------------------------------- known defects
def test_audit_still_flags_confidence_leakage():
    """A1: confidence was generated from the outcome. If this stops firing,
    someone has changed the data and the leakage warning must be revisited."""
    res = audit.audit_tickets()
    assert any("LEAKAGE" in i for i in res.issues)
    conf = res.extras["confidence_by_outcome"]
    assert conf["ai"] - conf["human"] > 0.25


def test_audit_still_flags_train_test_overlap():
    """A2: two texts appear in both splits. Remove them in Phase 4 and this
    test should be inverted to assert zero overlap."""
    res = audit.audit_intent_test()
    assert any("LEAKAGE" in i for i in res.issues)


def test_sunday_tickets_contradict_policy():
    """A3: documented generator defect, kept deliberately."""
    df = loaders.load_tickets()
    assert (df["weekday"] == "Sunday").sum() > 0


def test_no_pii_in_message_text():
    """No emails, phones, card numbers or IFSC codes anywhere."""
    for texts in (loaders.load_intent_train()["text"], loaders.load_intent_test()["text"]):
        pii = audit.detect_pii(texts)
        for key in ("email", "phone_in", "card_like", "aadhaar_like", "ifsc"):
            assert pii[key] == 0, f"{key} detected in message text"


def test_no_label_conflicts_in_train():
    df = loaders.load_intent_train()
    conflicts = df.groupby(df["text"].str.lower().str.strip())["intent"].nunique()
    assert (conflicts > 1).sum() == 0


def test_subtopic_nested_within_intent():
    df = loaders.load_tickets()
    crossover = df.groupby("subtopic", observed=True)["intent"].nunique()
    assert (crossover > 1).sum() == 0


# ------------------------------------------------------------ text stats
def test_class_imbalance_is_ten_to_one():
    c = loaders.load_intent_train()["intent"].value_counts()
    assert 9.0 < c.max() / c.min() < 11.0


def test_length_features_produce_expected_columns():
    df = text_stats.length_features(loaders.load_intent_train()["text"].head(50))
    for col in ["n_chars", "n_words", "n_sentences", "has_order_ref", "has_error_code"]:
        assert col in df.columns
    assert (df["n_words"] > 0).all()


def test_drift_between_splits_is_substantial():
    """Q6: the train/test gap is the headline Phase 4 finding, so guard it."""
    d = text_stats.vocabulary_drift(
        loaders.load_intent_train()["text"], loaders.load_intent_test()["text"]
    )
    assert d["oov_token_rate_pct"] > 15, "test set is no longer meaningfully harder"


def test_train_is_templated_and_test_is_not():
    tr = text_stats.template_score(loaders.load_intent_train()["text"])
    te = text_stats.template_score(loaders.load_intent_test()["text"])
    assert tr["template_rate_pct"] > 20
    assert te["template_rate_pct"] < 5


def test_order_tracking_and_refund_are_most_confusable():
    """7a: order references drive the top overlap. Masking them should
    reduce this - which is the Phase 4 ablation."""
    tr = loaders.load_intent_train()
    ov = text_stats.class_lexical_overlap(tr["text"], tr["intent"])
    top = ov.iloc[0]
    assert {top["intent_a"], top["intent_b"]} == {"order_tracking", "return_refund_request"}


def test_corpus_is_small_enough_that_exact_search_suffices():
    """7c: documents the corpus-size finding that changed the chunking plan."""
    corpus = loaders.load_corpus()
    total_words = corpus["n_words"].sum()
    assert total_words < 30_000
    est_chunks_512 = total_words * 1.3 / 512
    assert est_chunks_512 < 100, "corpus grew - revisit the ANN phase decision"


@pytest.mark.parametrize("name,expected", [
    ("retrieval_eval", 120), ("generation_eval", 25), ("unanswerable_eval", 40),
    ("agent_trajectory_eval", 30), ("multiturn_eval", 25),
    ("adversarial_eval", 30), ("vision_eval", 25),
])
def test_eval_sets_have_expected_case_counts(name, expected):
    assert len(loaders.load_eval(name)["cases"]) == expected


def test_eval_gold_sections_reference_real_documents():
    corpus_docs = set(loaders.load_corpus()["doc"])
    referenced = {
        g["doc"]
        for c in loaders.load_eval("retrieval_eval")["cases"]
        for g in c["gold_sections"]
    }
    assert referenced <= corpus_docs, f"missing: {referenced - corpus_docs}"
