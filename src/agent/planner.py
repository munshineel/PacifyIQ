"""Tool planning.

The agent selects tools from **explicit rules**, not from a model guessing.
Two reasons, both measured elsewhere in this project:

1. The intent classifier is wrong *and* uncertain on hard queries (Phase 6.5),
   so a plan derived purely from predicted intent would be unreliable. Entities
   are deterministic and take precedence.
2. An LLM asked "which tools do you need?" tends to call everything available.
   The brief for this phase is explicitly that the agent must not do that.

The planner produces a `Plan`: an ordered list of steps with a stated reason and
a stated precondition. Steps whose precondition fails are skipped, not attempted
- which is how the agent avoids calling `get_order` when no order id exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.agent.tools import REGISTRY, Tier
from src.understanding.schema import Understanding


@dataclass
class PlannedStep:
    tool: str
    args: dict[str, Any]
    reason: str
    requires: list[str] = field(default_factory=list)   # state keys that must exist
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    steps: list[PlannedStep] = field(default_factory=list)
    intent: str = ""
    needs_knowledge: bool = False
    needs_order_data: bool = False
    needs_image: bool = False
    missing_info: list[str] = field(default_factory=list)
    escalate_immediately: bool = False
    escalation_reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "steps": [s.to_dict() for s in self.steps],
            "needs_knowledge": self.needs_knowledge,
            "needs_order_data": self.needs_order_data,
            "needs_image": self.needs_image,
            "missing_info": self.missing_info,
            "escalate_immediately": self.escalate_immediately,
            "escalation_reason": self.escalation_reason,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------
# Per-intent requirements.
#
#   knowledge      does answering need policy documentation
#   order_data     does answering need this customer's specific order
#   policy_check   which eligibility computation, if any
#   always_escalate this intent cannot be resolved autonomously
# ---------------------------------------------------------------------
INTENT_PLAN = {
    "order_tracking": {
        "knowledge": False, "order_data": True, "policy_check": None,
        "note": "live order state, not policy - retrieving the shipping SLA "
                "does not tell the customer where their parcel is",
    },
    "return_policy_question": {
        "knowledge": True, "order_data": False, "policy_check": None,
        "note": "informational: no specific order in play",
    },
    "return_refund_request": {
        "knowledge": True, "order_data": True, "policy_check": "return",
        "always_escalate": True,
        "note": "transactional and money-moving: tier 3",
    },
    "warranty_claim": {
        "knowledge": True, "order_data": True, "policy_check": "warranty",
        "note": "coverage depends on brand, age and fault category",
    },
    "shipping_delivery": {
        "knowledge": True, "order_data": False, "policy_check": None,
        "note": "policy unless a specific parcel is named",
    },
    "product_information": {
        "knowledge": True, "order_data": False, "policy_check": None,
        "products": True,
        "note": "specifications come from the corpus; stock from the catalogue",
    },
    "technical_support": {
        "knowledge": True, "order_data": False, "policy_check": None,
        "note": "troubleshooting guide; order data only if a warranty question "
                "emerges",
    },
    "payment_issue": {
        "knowledge": True, "order_data": True, "policy_check": None,
        "payment": True,
        "note": "must distinguish a refund from a failed-payment reversal",
    },
    "account_management": {
        "knowledge": True, "order_data": False, "policy_check": None,
        "always_escalate": True,
        "note": "identity cannot be verified in chat (POL-CS-001 S6)",
    },
    "complaint": {
        "knowledge": False, "order_data": False, "policy_check": None,
        "always_escalate": True,
        "note": "a complaint is a relationship problem; citing policy makes it worse",
    },
    "out_of_scope": {
        "knowledge": False, "order_data": False, "policy_check": None,
        "note": "nothing to retrieve, nothing to act on",
    },
}

# Phrases that force escalation regardless of intent or confidence.
LEGAL_THREAT = ("consumer court", "consumer forum", "lawyer", "legal action",
                "chargeback", "sue you", "small claims", "ombudsman")
IDENTITY_SENSITIVE = ("change my email", "change my registered", "delete my account",
                      "close my account", "reset my password", "new bank account",
                      "different account", "send my data", "export my data",
                      "someone hacked", "someone else logged")


def build_plan(
    understanding: Understanding,
    image_path: str | None = None,
    known_order_id: str | None = None,
) -> Plan:
    """Decide which tools this request needs. Nothing is called speculatively."""
    text_low = understanding.text.lower()
    intent = _correct_intent(understanding, text_low)
    rule = INTENT_PLAN.get(intent, INTENT_PLAN["out_of_scope"])

    plan = Plan(intent=intent)
    if rule.get("note"):
        plan.notes.append(rule["note"])

    order_id = known_order_id or (
        understanding.entities.order_ids[0]
        if understanding.entities.order_ids else None
    )

    # ---- entities override the informational/transactional split -----
    # `return_policy_question` is defined as "no specific order in play", so it
    # plans no order lookup. But "Is PAC-2026-12354 returnable?" names an order,
    # and answering it from generic policy is wrong twice over: the customer's
    # window may already have closed, and their region may override the base
    # rule entirely (the EU addendum).
    #
    # This is the Phase 6.5 principle applied to planning - an extracted entity
    # is deterministic and outranks a probabilistic intent label.
    ORDER_SPECIFIC = {"return_policy_question", "shipping_delivery",
                      "product_information", "technical_support"}
    if order_id and intent in ORDER_SPECIFIC:
        rule = dict(rule)
        rule["order_data"] = True
        if intent == "return_policy_question":
            rule["policy_check"] = "warranty" if "warrant" in text_low else "return"
        plan.notes.append(
            "an order reference was supplied, so the answer is order-specific "
            "rather than general policy"
        )

    # ---- hard escalation triggers, checked before any tool -----------
    if any(t in text_low for t in LEGAL_THREAT):
        plan.escalate_immediately = True
        plan.escalation_reason = "legal_or_chargeback_threat"
        plan.notes.append("POL-CS-001 S3.4(d): legal threat is a hard trigger")
    elif any(t in text_low for t in IDENTITY_SENSITIVE):
        plan.escalate_immediately = True
        plan.escalation_reason = "identity_verification_required"
        plan.notes.append("POL-CS-001 S6: identity cannot be verified in chat")

    # ---- image first: it may supply the entity everything else needs --
    if image_path:
        plan.needs_image = True
        plan.steps.append(PlannedStep(
            tool="analyze_screenshot",
            args={"image_path": image_path, "user_text": understanding.text},
            reason="a screenshot was attached and may contain an error code",
        ))

    # ---- order-specific data -----------------------------------------
    wants_order = rule["order_data"] or (
        order_id is not None and intent in
        {"shipping_delivery", "technical_support", "product_information"}
    )
    if wants_order:
        if order_id:
            plan.needs_order_data = True
            plan.steps.append(PlannedStep(
                tool="get_order", args={"order_id": order_id},
                reason=f"{intent} requires the state of a specific order",
            ))
        else:
            plan.missing_info.append("order_id")
            plan.notes.append(
                "order reference not provided - cannot look up order state"
            )

    # ---- eligibility computation --------------------------------------
    policy = rule.get("policy_check")
    if policy and order_id:
        plan.steps.append(PlannedStep(
            tool="check_policy", args={"order_id": order_id, "policy": policy},
            reason=f"{policy} eligibility is a deterministic computation, "
                   f"not a judgement",
            requires=["get_order"],
        ))

    # a refund request also needs the amount computed
    if intent == "return_refund_request" and order_id:
        plan.steps.append(PlannedStep(
            tool="check_policy", args={"order_id": order_id, "policy": "refund"},
            reason="refund amount must be computed in SQL, never by the model",
            requires=["get_order"],
        ))

    # ---- payment state -------------------------------------------------
    if rule.get("payment") and order_id:
        plan.steps.append(PlannedStep(
            tool="check_payment", args={"order_id": order_id},
            reason="distinguishes a refund in progress from a failed-payment "
                   "reversal (DEFECT-08)",
            requires=["get_order"],
        ))

    # ---- subscription / extended warranty -------------------------------
    if order_id and any(t in text_low for t in
                        ("pacifycare", "care+", "extended warranty",
                         "subscription", "extend my warranty", "add warranty")):
        plan.steps.append(PlannedStep(
            tool="check_subscription", args={"order_id": order_id},
            reason="customer asked about extended cover",
            requires=["get_order"],
        ))

    # ---- product catalogue ----------------------------------------------
    if rule.get("products") and (
        understanding.entities.products
        or any(t in text_low for t in ("in stock", "available", "price", "cost",
                                       "how much"))
    ):
        term = (understanding.entities.products[0]
                if understanding.entities.products else "")
        plan.steps.append(PlannedStep(
            tool="search_products", args={"query": term},
            reason="stock and price are live catalogue data, not documentation",
            optional=True,
        ))

    # ---- explicit ticket request ------------------------------------------
    # "create a ticket" / "raise a ticket" is a direct instruction to take a
    # Tier-2 action. Answering it with a policy quote would be obtuse.
    if any(t in text_low for t in ("create a ticket", "raise a ticket",
                                   "open a ticket", "log a ticket",
                                   "file a ticket", "create a case")):
        plan.steps.append(PlannedStep(
            tool="create_support_ticket",
            args={"summary": understanding.text, "intent": intent,
                  "order_id": understanding.entities.order_ids[0]
                  if understanding.entities.order_ids else "",
                  "priority": "high" if understanding.urgency == "high" else "medium"},
            reason="the customer explicitly asked for a ticket to be opened",
        ))

    # ---- knowledge base --------------------------------------------------
    if rule["knowledge"]:
        plan.needs_knowledge = True
        plan.steps.append(PlannedStep(
            tool="search_knowledge_base",
            args={"query": understanding.text,
                  "region": "EU" if _looks_eu(text_low) else ""},
            reason="the answer must be grounded in published policy",
        ))

    # ---- always-escalate intents -----------------------------------------
    # One refinement: `return_refund_request` covers both "can I return this?"
    # and "return this for me". Only the second is a mutating action. Treating
    # an eligibility QUESTION as a refund REQUEST escalates cases the agent can
    # answer completely from check_policy, which is both unhelpful and wasteful
    # of a human agent's time.
    #
    # The distinction is grammatical, not semantic, so it is detectable with
    # rules rather than a model: an imperative or a stated desire signals a
    # request; an interrogative signals a question.
    if intent == "return_refund_request" and _is_eligibility_question(text_low):
        rule = dict(rule)
        rule["always_escalate"] = False
        plan.notes.append(
            "phrased as an eligibility question, not a request to act - "
            "answering from policy rather than escalating"
        )

    if rule.get("always_escalate") and not plan.escalate_immediately:
        plan.escalate_immediately = True
        plan.escalation_reason = {
            "return_refund_request": "mutating_action_requires_approval",
            "account_management": "identity_verification_required",
            "complaint": "relationship_issue_requires_human",
        }.get(intent, "policy_requires_human")

    return plan


# Phrasing that marks a REQUEST to perform a return or refund, as opposed to a
# question about whether one is possible.
ACTION_REQUEST_TERMS = (
    "i want", "i'd like", "i would like", "please refund", "refund my",
    "refund me", "return my", "send someone", "arrange", "process my",
    "initiate", "start a return", "give me my money", "money back",
    "cancel and refund", "chahiye", "kar do", "karna hai",
)
QUESTION_OPENERS = (
    "can i", "could i", "am i", "is it", "do i", "would i", "what if",
    "is there", "how do i", "how long", "what is", "whats", "what's",
    "am i able", "are we able", "is my",
)


def _is_eligibility_question(text_low: str) -> bool:
    """True when the customer is ASKING about a return rather than requesting one."""
    if any(t in text_low for t in ACTION_REQUEST_TERMS):
        return False
    return text_low.strip().startswith(QUESTION_OPENERS) or text_low.rstrip().endswith("?")


# Symptom vocabulary that overrides a low-confidence intent. Measured in
# Phase 6.5: the classifier is both wrong and uncertain on exactly these
# queries, and mentioning an order number pulls them toward order_tracking.
SYMPTOM_TERMS = (
    "won't turn on", "wont turn on", "will not turn on", "not turning on",
    "won't boot", "wont boot", "crashing", "crashes", "keeps crashing",
    "screen", "display", "flicker", "overheat", "not charging",
    "won't charge", "wont charge", "no sound", "keyboard", "wifi", "wi-fi",
    "error code", "blue screen", "freezes", "broken", "faulty", "not working",
)
TRACKING_TERMS = ("where is", "when will", "arrive", "delivery", "track",
                  "shipped", "dispatch", "kahan", "kab")

INTENT_OVERRIDE_MARGIN = 0.25


def _correct_intent(understanding: Understanding, text_low: str) -> str:
    """Override a low-confidence intent using deterministic symptom vocabulary.

    Mentioning an order number pulls almost anything toward `order_tracking`,
    because 77% of training examples for that intent contain one. A message
    describing a fault is a technical-support request whether or not it names
    an order.
    """
    intent = understanding.intent
    if understanding.intent_margin >= INTENT_OVERRIDE_MARGIN:
        return intent

    has_symptom = any(t in text_low for t in SYMPTOM_TERMS)
    has_tracking = any(t in text_low for t in TRACKING_TERMS)

    if has_symptom and not has_tracking and intent in (
        "order_tracking", "shipping_delivery", "product_information",
        "payment_issue",
    ):
        return "technical_support"

    # The converse: a message asking where a parcel is needs order data even
    # when the classifier labelled it as a policy question. Compound messages
    # trigger this often - "where is my order and can I return it" carries two
    # intents, and the classifier can only name one.
    if has_tracking and not has_symptom and intent in (
        "return_policy_question", "product_information", "out_of_scope",
    ):
        return "order_tracking"
    return intent


def _looks_eu(text_low: str) -> bool:
    return any(t in text_low for t in (
        "eu ", "european", "germany", "france", "netherlands", "ireland",
        "spain", "italy", "berlin", "statutory withdrawal", "distance selling",
    ))


if __name__ == "__main__":
    from src.understanding.pipeline import UnderstandingPipeline

    up = UnderstandingPipeline.load()
    cases = [
        "What is your return policy?",
        "Where is my order PAC-2026-12345?",
        "I want to return order PAC-2026-12345 and get a refund",
        "My laptop won't turn on, order PAC-2026-12356",
        "I was charged twice for order PAC-2026-12364",
        "Delete my account",
        "I'm taking you to consumer court",
        "Is the Phone X Pro in stock?",
        "my payment isn't working",
    ]
    for c in cases:
        p = build_plan(up.understand(c))
        print(f"\n{c}")
        print(f"  intent={p.intent}  escalate={p.escalate_immediately}"
              f"{' (' + p.escalation_reason + ')' if p.escalation_reason else ''}")
        print(f"  tools: {p.tool_names or '[]'}")
        if p.missing_info:
            print(f"  missing: {p.missing_info}")
