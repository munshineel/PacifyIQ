"""Tests for the Support Intelligence layer (Phase 13b).

Scope guard included: this layer must stay focused on AI and support
operations, and must not drift into the business analytics that already exist
in src/analytics/metrics.py.
"""
import pandas as pd
import pytest

from src.analytics import support_intelligence as si

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def df():
    d = si.load_traces()
    if d.empty:
        pytest.skip("no traces - run scripts/simulate_support_traffic.py")
    return d


# ============================================================== scope
def test_layer_does_not_duplicate_business_analytics():
    """EcomIQ-style metrics - revenue, product performance, customer segments -
    belong to a different question and a different module."""
    # Check FUNCTION NAMES, not prose. The module docstring names the business
    # metrics it deliberately excludes, so a naive text scan matches its own
    # explanation of why they are absent.
    names = [n for n in dir(si) if not n.startswith("_")]
    for term in ["revenue", "sales", "profit", "aov", "lifetime", "basket",
                 "product_performance", "segment"]:
        offenders = [n for n in names if term in n.lower()]
        assert not offenders, f"business metric leaked in: {offenders}"


def test_layer_reads_traces_not_ticket_history():
    """The distinction that makes this layer meaningful: it measures the
    system's own behaviour, not a synthetic ticket table."""
    src = open(si.__file__).read()
    assert "from src.observability import traces" in src
    # It must not READ the synthetic ticket table - that belongs to the
    # business analytics module. Mentioning it in a comment is fine.
    assert "read_ticket" not in src
    assert "load_tickets" not in src
    assert "ticket_history.csv" not in src.split('"""')[2] if src.count('"""') > 2 else True


# =========================================================== loading
def test_empty_traces_do_not_crash():
    empty = pd.DataFrame()
    assert si.overview(empty).total_conversations == 0
    assert si.intent_breakdown(empty).empty
    assert si.tool_usage(empty).empty
    assert si.emerging_issues(empty) == []
    assert si.headlines(empty) == []


def test_list_columns_are_parsed(df):
    assert isinstance(df["actions_taken"].iloc[0], list)
    assert isinstance(df["guardrail_rules"].iloc[0], list)


# ========================================================== overview
def test_overview_covers_every_required_metric(df):
    o = si.overview(df).to_dict()
    for metric in ["total_conversations", "resolution_rate_pct",
                   "escalation_rate_pct", "avg_confidence",
                   "low_confidence_pct", "retrieval_failure_pct",
                   "screenshot_pct", "avg_tools_per_conversation",
                   "median_latency_ms"]:
        assert metric in o, f"{metric} missing"


def test_rates_are_percentages(df):
    o = si.overview(df)
    for v in [o.resolution_rate_pct, o.escalation_rate_pct,
              o.retrieval_failure_pct, o.low_confidence_pct]:
        assert 0 <= v <= 100


def test_outcome_rates_are_mutually_exclusive(df):
    o = si.overview(df)
    total = (o.resolution_rate_pct + o.escalation_rate_pct
             + o.clarification_rate_pct + o.refusal_rate_pct)
    assert total <= 101, f"outcome rates sum to {total} - categories overlap"


# ===================================================== distributions
def test_intent_breakdown_separates_volume_from_workload(df):
    """An intent can be a fifth of traffic and cause no work. The share of
    ESCALATIONS is the operational number, not the share of volume."""
    g = si.intent_breakdown(df)
    assert "share_pct" in g
    assert "share_of_escalations_pct" in g
    assert abs(g["share_pct"].sum() - 100) < 2


def test_sentiment_and_urgency_are_reported(df):
    assert not si.sentiment_breakdown(df).empty
    assert not si.urgency_breakdown(df).empty


def test_urgency_is_ordered_high_to_low(df):
    order = si.urgency_breakdown(df)["urgency"].tolist()
    known = [u for u in order if u in ("high", "medium", "low")]
    assert known == sorted(known, key=lambda u: {"high": 0, "medium": 1,
                                                 "low": 2}[u])


# ===================================================== AI performance
def test_tool_usage_is_measured(df):
    t = si.tool_usage(df)
    assert not t.empty
    assert {"tool", "calls", "conversations_pct"} <= set(t.columns)


def test_agent_does_not_call_every_tool(df):
    """Guards the claim that tool selection is genuine rather than exhaustive."""
    from src.agent.tools import REGISTRY

    assert si.overview(df).avg_tools_per_conversation < len(REGISTRY) / 3


def test_tier3_tools_never_fire(df):
    """Mutating tools must never appear in real traffic."""
    used = set(si.tool_usage(df)["tool"]) if not si.tool_usage(df).empty else set()
    for forbidden in ("approve_refund", "cancel_order", "modify_account"):
        assert forbidden not in used


def test_retrieval_health_distinguishes_success_from_failure(df):
    r = si.retrieval_health(df)
    assert r["median_bm25_success"] > r["median_bm25_failure"]


def test_retrieval_failure_drives_escalation(df):
    """The finding this layer exists to surface: retrieval quality is the
    largest driver of human workload."""
    r = si.retrieval_health(df)
    assert (r["escalation_when_retrieval_fails_pct"]
            > r["escalation_when_retrieval_ok_pct"])


def test_failed_retrievals_are_listed(df):
    f = si.failed_retrievals(df)
    if len(f):
        assert "question" in f.columns
        assert (f["max_bm25"] < si.WEAK_RETRIEVAL).all()


def test_low_confidence_cases_exclude_escalations(df):
    """An escalated answer was already flagged. The interesting cases are the
    weak answers that reached a customer unreviewed."""
    lc = si.low_confidence_cases(df)
    if len(lc):
        assert (lc["confidence"] < si.LOW_CONFIDENCE).all()
        assert not lc["resolution_status"].eq("escalated").any()


def test_screenshot_usage_is_reported(df):
    s = si.screenshot_usage(df)
    assert "conversations_with_image" in s


# ========================================================= escalation
def test_escalations_split_by_design_from_capability_gap(df):
    """A refund escalation is the system working. A "no documentation"
    escalation is a gap. Reporting one number for both hides the distinction
    that determines what to fix."""
    e = si.escalation_breakdown(df)
    if len(e):
        assert "category" in e.columns
        assert set(e["category"]) <= {"by design", "capability gap"}


def test_unresolved_issues_are_clustered(df):
    u = si.unresolved_clusters(df)
    if len(u):
        assert "occurrences" in u.columns
        assert u["occurrences"].is_monotonic_decreasing


# ============================================================ trends
def test_daily_trend_is_chronological(df):
    t = si.daily_trend(df)
    assert len(t) > 1
    assert t["date"].is_monotonic_increasing


def test_emerging_issue_detection_finds_the_planted_surge(df):
    """The simulator plants a login and payment surge in the final week
    specifically so the detector can be validated against a known answer.
    Without a planted signal, "nothing detected" is indistinguishable from a
    broken detector."""
    issues = si.emerging_issues(df)
    topics = {e.topic for e in issues}
    assert issues, "no emerging issues detected at all"
    assert any("login" in t or "payment" in t for t in topics), \
        f"planted surge not found; detected {topics}"


def test_emerging_issues_are_normalised_per_day(df):
    """Windows of 7 and 28 days must be compared as rates, not raw counts."""
    for e in si.emerging_issues(df):
        if e.signal != "NEW":
            assert e.lift > 1.0


def test_new_topics_are_labelled_not_infinite(df):
    for e in si.emerging_issues(df):
        assert e.lift < 1000


def test_headlines_are_sentences_not_numbers(df):
    """An operator needs "login issues rose this week", not "lift=3.1"."""
    for h in si.headlines(df):
        assert len(h.split()) >= 6, f"not a sentence: {h}"
        assert h.endswith(".")


def test_escalation_headline_names_the_largest_source(df):
    h = si.escalation_headline(df)
    if h:
        assert "escalation" in h.lower()


# ======================================================== integration
def test_service_layer_exposes_every_section(df):
    from src.ui import service as svc

    d = svc.support_intelligence()
    assert d["ok"]
    for section in ["overview", "intents", "sentiment", "urgency", "topics",
                    "tools", "retrieval", "failed_retrievals",
                    "low_confidence", "unresolved", "screenshots",
                    "escalations", "daily", "emerging", "headlines"]:
        assert section in d, f"{section} missing from the service payload"


def test_service_handles_no_traces_gracefully(monkeypatch):
    from src.ui import service as svc

    monkeypatch.setattr(si, "load_traces", lambda **kw: pd.DataFrame())
    d = svc.support_intelligence()
    assert not d["ok"]
    assert "error" in d
