"""PHASE 10 — Agent scenario evaluation.

Ten realistic customer scenarios, each targeting a distinct decision path, plus
the trajectory evaluation set.

The question this answers is not "does it produce fluent text" but:
does it select the right tools, stop at the right point, and escalate when it
must?

    python scripts/evaluate_agent.py
    python scripts/evaluate_agent.py --scenarios-only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import Resolution, SupportAgent  # noqa: E402
from src.agent.tools import REGISTRY, Tier, call_tool  # noqa: E402
from src.config.settings import settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SHOTS = settings.eval_dir / "screenshots"


@dataclass
class Scenario:
    """One expected decision path."""

    n: int
    label: str
    text: str
    image: str | None = None
    order_id: str | None = None
    expect_status: list[str] = field(default_factory=list)
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    expect_escalation: bool | None = None
    expect_reason: str | None = None
    max_tools: int | None = None
    note: str = ""


SCENARIOS = [
    Scenario(
        1, "Simple FAQ",
        "What is your return policy for opened laptops?",
        expect_status=["resolved", "resolved_with_caveat"],
        expect_tools=["search_knowledge_base"],
        forbid_tools=["get_order", "check_payment", "check_policy"],
        expect_escalation=False, max_tools=2,
        note="informational: no order in play, so no operational tool should run",
    ),
    Scenario(
        2, "Billing issue",
        "I was charged twice for order PAC-2026-12364",
        expect_status=["resolved", "resolved_with_caveat", "escalated"],
        expect_tools=["get_order", "check_payment", "search_knowledge_base"],
        note="needs live payment state AND the policy on double charges",
    ),
    Scenario(
        3, "Refund request",
        "I want to return order PAC-2026-12345 and get my money back",
        expect_status=["escalated"],
        expect_tools=["get_order", "check_policy"],
        forbid_tools=["approve_refund"],
        expect_escalation=True,
        expect_reason="mutating_action_requires_approval",
        note="tier 3: eligibility and amount computed, but the refund is never issued",
    ),
    Scenario(
        4, "Technical problem",
        "My Pacify ProBook 14 won't turn on, order PAC-2026-12356",
        expect_status=["resolved", "resolved_with_caveat", "escalated"],
        expect_tools=["search_knowledge_base"],
        note="symptom vocabulary must override the order_tracking pull",
    ),
    Scenario(
        5, "Screenshot technical problem",
        "my payment isn't working, see attached",
        image=str(SHOTS / "V003_PAY_402.png"),
        expect_status=["resolved", "resolved_with_caveat", "escalated"],
        expect_tools=["analyze_screenshot", "search_knowledge_base"],
        note="the error code exists only in the image",
    ),
    Scenario(
        6, "Missing information",
        "where is my order?",
        expect_status=["needs_information"],
        forbid_tools=["get_order"],
        expect_escalation=False,
        note="must ask rather than call get_order with no id",
    ),
    Scenario(
        7, "Conflicting information",
        "How long do I have to return my laptop? Your site says 30 days.",
        expect_status=["resolved", "resolved_with_caveat", "escalated"],
        expect_tools=["search_knowledge_base"],
        note="DEFECT-01: two current documents disagree",
    ),
    Scenario(
        8, "Unsupported request",
        "can you write me a python script to sort a list",
        expect_status=["refused", "escalated"],
        forbid_tools=["get_order", "check_policy", "check_payment"],
        note="out of scope: nothing to retrieve, nothing to act on",
    ),
    Scenario(
        9, "High-risk / ambiguous",
        "change the bank account for my refund to a different one",
        expect_status=["escalated"],
        forbid_tools=["modify_account"],
        expect_escalation=True,
        expect_reason="identity_verification_required",
        note="POL-CS-001 S6: identity cannot be verified in chat",
    ),
    Scenario(
        10, "Requires human escalation",
        "This is the THIRD time. Refund me today or I'm going to consumer court.",
        expect_status=["escalated"],
        expect_escalation=True,
        expect_reason="legal_or_chargeback_threat",
        max_tools=2,
        note="hard trigger: escalate before spending any tool calls on policy",
    ),
]


def run_scenarios(agent: SupportAgent) -> tuple[pd.DataFrame, list[dict]]:
    rows, details = [], []
    for s in SCENARIOS:
        d = agent.handle(s.text, image_path=s.image, order_id=s.order_id)

        checks: list[tuple[str, bool]] = []
        if s.expect_status:
            checks.append(("status", d.resolution_status in s.expect_status))
        for t in s.expect_tools:
            checks.append((f"used:{t}", t in d.actions_taken))
        for t in s.forbid_tools:
            checks.append((f"avoided:{t}", t not in d.actions_taken))
        if s.expect_escalation is not None:
            checks.append(("escalation", d.escalation_required == s.expect_escalation))
        if s.expect_reason:
            checks.append(("reason", d.escalation_reason == s.expect_reason))
        if s.max_tools is not None:
            checks.append((f"<= {s.max_tools} tools",
                           len([a for a in d.actions_taken
                                if a != "escalate_to_human"]) <= s.max_tools))

        failed = [name for name, ok in checks if not ok]
        rows.append({
            "n": s.n, "scenario": s.label,
            "status": d.resolution_status,
            "tools": len(d.actions_taken),
            "escalated": d.escalation_required,
            "reason": d.escalation_reason or "-",
            "conf": round(d.confidence, 2),
            "ms": round(d.latency_ms),
            "pass": not failed,
            "failed_checks": ", ".join(failed),
        })
        details.append({"scenario": s.label, "text": s.text, "note": s.note,
                        "decision": d.to_dict()})
    return pd.DataFrame(rows), details


# The trajectory eval set was authored in Phase 0, before these tools existed,
# and names capabilities rather than implementations. Mapping the historical
# names onto the shipped tools is honest - the agent is calling the right
# capability - but the mapping is stated here rather than hidden.
TOOL_ALIASES = {
    "get_order_status": "get_order",
    "check_return_eligibility": "check_policy",
    "check_warranty_status": "check_policy",
    "search_company_policy": "search_knowledge_base",
    "get_customer_details": "get_customer",
    "get_payment_status": "check_payment",
    "get_product_details": "search_products",
    "create_return_request": "escalate_to_human",   # tier 3 -> handoff
}


def _canon(tool: str) -> str:
    return TOOL_ALIASES.get(tool, tool)


def run_trajectories(agent: SupportAgent) -> pd.DataFrame:
    """Score tool selection and argument extraction against the eval set."""
    cases = json.loads(
        (settings.eval_dir / "agent_trajectory_eval.json").read_text()
    )["cases"]

    rows = []
    for c in cases:
        d = agent.handle(c["user_message"])
        expected_tools = [_canon(t["tool"]) for t in c["expected_tools"]]
        expected_args = {
            t["tool"]: t["args"] for t in c["expected_tools"] if t.get("args")
        }
        actual = [a for a in d.actions_taken]

        # tool selection: did the required tools get called
        needed = list(dict.fromkeys(t for t in expected_tools if t in REGISTRY))
        hit = sum(1 for t in needed if t in actual)
        selection = hit / len(needed) if needed else (1.0 if not actual else 0.0)

        # argument extraction: did an order id get normalised correctly
        arg_ok = None
        want_oid = next(
            (v.get("order_id") for v in expected_args.values()
             if isinstance(v, dict) and v.get("order_id")), None
        )
        if want_oid:
            arg_ok = any(
                want_oid in json.dumps(a) for a in [d.to_dict()]
            )

        rows.append({
            "id": c["id"],
            "message": c["user_message"][:44],
            "expected": ",".join(expected_tools) or "-",
            "actual": ",".join(actual) or "-",
            "selection": round(selection, 3),
            "escalation_expected": c["expected_escalation"],
            "escalation_actual": d.escalation_required,
            "escalation_ok": c["expected_escalation"] == d.escalation_required,
            "arg_ok": arg_ok,
            "note": c["note"][:38],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-only", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)

    print("=" * 90)
    print("AGENT EVALUATION")
    print("=" * 90)
    print(f"  {len(REGISTRY)} tools registered "
          f"({sum(1 for s in REGISTRY.values() if s.tier <= Tier.CREATES_RECORD)} "
          f"autonomous, "
          f"{sum(1 for s in REGISTRY.values() if s.tier == Tier.MUTATING)} "
          f"requiring human approval)")

    agent = SupportAgent()

    # =================================================================
    print("\n" + "=" * 90)
    print("1. TEN SCENARIOS")
    print("=" * 90)
    scen, details = run_scenarios(agent)
    print(scen.to_string(index=False))
    print(f"\n  passed {scen['pass'].sum()}/{len(scen)}")
    scen.to_csv(RESULTS / "agent_scenarios.csv", index=False)
    (RESULTS / "agent_scenario_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    print("\n  tool selection is genuinely differentiated:")
    print(f"    tools called per scenario: "
          f"min={scen['tools'].min()}  max={scen['tools'].max()}  "
          f"mean={scen['tools'].mean():.1f}")
    print("    -> an agent that called every tool every time would show max=13")

    # =================================================================
    print("\n" + "=" * 90)
    print("2. TIER-3 ENFORCEMENT")
    print("=" * 90)
    blocked = []
    for name in ("approve_refund", "cancel_order", "modify_account"):
        r = call_tool(name, order_id="PAC-2026-12345", customer_id="CUS-10000")
        blocked.append({"tool": name, "status": r.status.value,
                        "message": r.message[:66]})
    print(pd.DataFrame(blocked).to_string(index=False))
    print("\n  enforced in code, not requested in a prompt: the agent has no")
    print("  path that can mint an approval token.")

    # =================================================================
    if not args.scenarios_only:
        print("\n" + "=" * 90)
        print("3. TRAJECTORY EVALUATION (30 cases)")
        print("=" * 90)
        traj = run_trajectories(agent)
        traj.to_csv(RESULTS / "agent_trajectories.csv", index=False)

        sel = traj["selection"].mean()
        esc = traj["escalation_ok"].mean()
        argd = traj[traj["arg_ok"].notna()]
        print(f"  tool selection accuracy      {sel:.3f}")
        print(f"  escalation decision accuracy {esc:.3f}")
        if len(argd):
            print(f"  argument extraction accuracy {argd['arg_ok'].mean():.3f} "
                  f"({len(argd)} cases with an order id)")

        print("\n  cases where escalation was wrong:")
        bad = traj[~traj["escalation_ok"]]
        if len(bad):
            print(bad[["id", "message", "escalation_expected",
                       "escalation_actual", "note"]].to_string(index=False))
        else:
            print("    none")

        print("\n  weakest tool selection:")
        print(traj.nsmallest(8, "selection")[
            ["id", "message", "expected", "actual", "selection"]
        ].to_string(index=False))

    # =================================================================
    print("\n" + "=" * 90)
    print("4. WORKED EXAMPLE — decision metadata, not chain of thought")
    print("=" * 90)
    d = agent.handle("I want to return order PAC-2026-12345 and get my money back")
    meta = {k: v for k, v in d.to_dict().items() if k != "answer"}
    print(json.dumps(meta, indent=2, default=str))
    print(f"\n  ANSWER: {d.answer}")

    # =================================================================
    summary = {
        "n_tools": len(REGISTRY),
        "scenarios_passed": int(scen["pass"].sum()),
        "scenarios_total": len(scen),
        "tools_per_scenario": {
            "min": int(scen["tools"].min()), "max": int(scen["tools"].max()),
            "mean": round(float(scen["tools"].mean()), 2),
        },
        "tier3_blocked": len(blocked),
    }
    if not args.scenarios_only:
        summary.update({
            "tool_selection_accuracy": round(float(traj["selection"].mean()), 4),
            "escalation_accuracy": round(float(traj["escalation_ok"].mean()), 4),
        })
    (RESULTS / "agent_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 90)
    print("AGENT EVALUATION COMPLETE")
    print(f"  scenarios passed   {summary['scenarios_passed']}/{summary['scenarios_total']}")
    if not args.scenarios_only:
        print(f"  tool selection     {summary['tool_selection_accuracy']:.3f}")
        print(f"  escalation         {summary['escalation_accuracy']:.3f}")
    print(f"  tools per request  {summary['tools_per_scenario']}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
