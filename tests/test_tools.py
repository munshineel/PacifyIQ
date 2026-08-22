"""TOOLS — contract, invalid parameters, failures and conflicting outputs.

A tool is a boundary. Everything crossing it must be typed, and every failure
must be representable in the return value rather than as an exception - the
agent has to route on it.
"""
from __future__ import annotations

import pytest

from src.agent.tools import REGISTRY, Tier, ToolStatus, call_tool

pytestmark = pytest.mark.tools


# =====================================================================
# Contract
# =====================================================================

def test_every_tool_returns_a_tool_result():
    """No tool may return a bare value, a dict, or None."""
    from src.agent.tools import ToolResult

    for name, spec in REGISTRY.items():
        if spec.tier == Tier.MUTATING:
            continue
        args = {a: "PAC-2026-12345" if "order" in a else "test"
                for a in spec.required_args}
        r = call_tool(name, **args)
        assert isinstance(r, ToolResult), f"{name} returned {type(r)}"


def test_every_tool_result_carries_its_status():
    for name in REGISTRY:
        r = call_tool(name)
        assert r.status is not None
        assert isinstance(r.ok, bool)


def test_tool_data_is_always_a_mapping():
    """The agent reads fields off `.data`. A non-dict payload fails silently
    at the point of use rather than at the point of return."""
    for name, spec in REGISTRY.items():
        if spec.tier == Tier.MUTATING:
            continue
        args = {a: "PAC-2026-12345" if "order" in a else "return policy"
                for a in spec.required_args}
        r = call_tool(name, **args)
        assert isinstance(r.data, dict), f"{name}.data is {type(r.data)}"


def test_tools_are_declared_with_a_tier():
    for name, spec in REGISTRY.items():
        assert spec.tier in (Tier.READ_ONLY, Tier.CREATES_RECORD,
                             Tier.MUTATING), name


# =====================================================================
# Valid calls
# =====================================================================

@pytest.mark.parametrize("order_id", [
    "PAC-2026-12345", "pac-2026-12345", "12345", "#12345", " PAC-2026-12345 ",
])
def test_order_reference_normalisation(order_id):
    """Argument extraction is the most common tool-calling failure. Here it is
    deterministic - a regex cannot hallucinate an order id."""
    r = call_tool("get_order", order_id=order_id)
    assert r.ok
    assert r.data["order_id"] == "PAC-2026-12345"


def test_policy_check_returns_deterministic_eligibility():
    assert call_tool("check_policy",
                     order_id="PAC-2026-12345").data["eligibility"] == "eligible"
    assert call_tool("check_policy",
                     order_id="PAC-2026-12347").data["eligibility"] == "expired"


def test_knowledge_search_returns_ranked_chunks():
    r = call_tool("search_knowledge_base", query="restocking fee")
    assert r.ok
    chunks = r.data["chunks"]
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True), "results are not ranked"


def test_knowledge_chunks_carry_filtering_metadata():
    """Region and version are needed downstream to tell a genuine contradiction
    from an inapplicable regional variant."""
    r = call_tool("search_knowledge_base", query="return window")
    for c in r.data["chunks"]:
        assert "region" in c and "version" in c and "citation" in c


# =====================================================================
# Invalid parameters
# =====================================================================

def test_unknown_tool_name_returns_a_result():
    r = call_tool("teleport_the_customer")
    assert not r.ok
    assert r.status in (ToolStatus.ERROR, ToolStatus.INVALID_ARGS)


def test_missing_required_argument_is_reported():
    r = call_tool("get_order")
    assert not r.ok
    assert r.status in (ToolStatus.INVALID_ARGS, ToolStatus.NOT_FOUND)


@pytest.mark.parametrize("bad", ["", "   ", "not-an-order", "'; DROP TABLE orders;--",
                                 "PAC-9999-99999", "🎧", "0", "-1"])
def test_malformed_order_ids_do_not_raise(bad):
    r = call_tool("get_order", order_id=bad)
    assert not r.ok
    assert r.message


def test_sql_injection_through_an_argument_is_inert():
    """Parameterised queries, verified rather than assumed."""
    from src.db.connection import query_all

    call_tool("get_order", order_id="x'; DROP TABLE orders; --")
    n = query_all("SELECT COUNT(*) AS n FROM orders")[0]["n"]
    assert n > 1000, "the orders table was damaged"


def test_unexpected_keyword_argument_does_not_crash():
    r = call_tool("get_order", order_id="PAC-2026-12345", nonsense=True)
    assert r is not None


@pytest.mark.parametrize("query", ["", "   ", "a", "?" * 200, "🎧" * 50])
def test_degenerate_search_queries_do_not_raise(query):
    r = call_tool("search_knowledge_base", query=query)
    assert r is not None
    assert isinstance(r.data, dict)


# =====================================================================
# Failures
# =====================================================================

def test_not_found_is_distinct_from_error():
    """A missing order is a normal outcome; a broken tool is not. Collapsing
    them would make the agent escalate typos."""
    missing = call_tool("get_order", order_id="PAC-2026-99999")
    assert missing.status == ToolStatus.NOT_FOUND
    assert not missing.ok


def test_a_raising_tool_is_caught_and_typed():
    """A tool failure must not take down the request."""
    from src.agent.tools import ToolResult, ToolStatus as TS

    def explode(**kw):
        raise RuntimeError("upstream is down")

    original = REGISTRY["get_order"].fn
    REGISTRY["get_order"].fn = explode
    try:
        r = call_tool("get_order", order_id="PAC-2026-12345")
        assert isinstance(r, ToolResult)
        assert r.status == TS.ERROR
        assert "upstream is down" in r.message
    finally:
        REGISTRY["get_order"].fn = original


def test_tool_failure_leaves_the_registry_usable():
    r = call_tool("get_order", order_id="PAC-2026-12345")
    assert r.ok, "the registry did not recover after an injected failure"


# =====================================================================
# Conflicting outputs
# =====================================================================

def test_eligibility_boolean_agrees_with_the_label():
    """Found by mutation testing: every test asserted the `eligibility` STRING,
    so inverting the `eligible` BOOLEAN broke nothing. The agent and the UI both
    branch on the boolean, so an inversion would have told customers the
    opposite of the truth with the whole suite green."""
    for oid, expected in [("PAC-2026-12345", True), ("PAC-2026-12347", False),
                          ("PAC-2026-12354", True), ("PAC-2026-12368", False)]:
        d = call_tool("check_policy", order_id=oid).data
        assert d["eligible"] is expected, f"{oid} eligible={d['eligible']}"
        assert d["eligible"] == (d["eligibility"] == "eligible"), (
            f"{oid}: boolean {d['eligible']} contradicts label "
            f"{d['eligibility']!r}")


def test_warranty_coverage_boolean_agrees_with_state():
    """Same class of bug on the warranty branch."""
    for oid in ["PAC-2026-12356", "PAC-2026-12345"]:
        d = call_tool("check_policy", order_id=oid, policy="warranty").data
        assert d["covered"] == (d["state"] == "in_warranty"), (
            f"{oid}: covered={d['covered']} state={d['state']!r}")


def test_expired_order_reports_zero_days_remaining():
    """Internal consistency: eligibility and the day count must agree, or the
    generated answer will contradict itself."""
    r = call_tool("check_policy", order_id="PAC-2026-12347")
    assert r.data["eligibility"] == "expired"
    assert r.data["days_remaining"] <= 0


def test_eligible_order_reports_positive_days_remaining():
    r = call_tool("check_policy", order_id="PAC-2026-12345")
    assert r.data["eligibility"] == "eligible"
    assert r.data["days_remaining"] > 0


def test_refund_never_exceeds_the_price_paid():
    r = call_tool("check_policy", order_id="PAC-2026-12345")
    assert r.data["refund_if_change_of_mind"] <= r.data["price_paid"]
    assert r.data["refund_if_defective"] <= r.data["price_paid"]


def test_defective_refund_is_never_worse_than_change_of_mind():
    """A customer with a broken product must not be paid less than one who
    simply changed their mind - that would be a policy inversion."""
    for oid in ["PAC-2026-12345", "PAC-2026-12354", "PAC-2026-12368"]:
        d = call_tool("check_policy", order_id=oid).data
        if d.get("refund_if_defective") is None:
            continue
        assert d["refund_if_defective"] >= d["refund_if_change_of_mind"], oid


def test_two_tools_agree_on_the_same_order():
    """get_order and check_policy read the same underlying record. If they ever
    disagree, the answer assembled from both is incoherent."""
    order = call_tool("get_order", order_id="PAC-2026-12345").data
    policy = call_tool("check_policy", order_id="PAC-2026-12345").data
    assert order["order_id"] == policy["order_id"]
    assert order["region"] == policy["region"]
    assert bool(order["is_opened"]) == bool(policy["is_opened"])


def test_repeated_calls_return_identical_results():
    """Non-determinism here would make every downstream metric unstable."""
    for tool, kw in [("get_order", {"order_id": "PAC-2026-12345"}),
                     ("check_policy", {"order_id": "PAC-2026-12345"}),
                     ("check_payment", {"order_id": "PAC-2026-12345"})]:
        a, b = call_tool(tool, **kw), call_tool(tool, **kw)
        assert a.data == b.data, f"{tool} is not deterministic"


# =====================================================================
# Tier enforcement
# =====================================================================

@pytest.mark.parametrize("tool", ["approve_refund", "cancel_order",
                                  "modify_account"])
def test_mutating_tools_are_refused(tool):
    """A prompt instruction not to issue refunds is a request. A code path that
    cannot issue one is a guarantee."""
    r = call_tool(tool, order_id="PAC-2026-12345", customer_id="CUS-10001")
    assert not r.ok
    assert r.status == ToolStatus.REFUSED


def test_mutating_tools_are_refused_regardless_of_arguments():
    for kwargs in [{}, {"order_id": "PAC-2026-12345"},
                   {"order_id": "PAC-2026-12345", "amount": 1},
                   {"approved": True}, {"force": True}, {"admin": True}]:
        assert not call_tool("approve_refund", **kwargs).ok


def test_mock_tools_are_labelled():
    """A simulated fact must never be mistaken for a real one downstream."""
    for tool in ["check_payment", "check_subscription"]:
        r = call_tool(tool, order_id="PAC-2026-12345")
        assert r.data.get("_source") == "mock", f"{tool} is not labelled"


def test_refund_figures_state_that_they_are_not_disbursed():
    r = call_tool("check_policy", order_id="PAC-2026-12345")
    assert "approval" in str(r.data.get("disbursement", "")).lower()
