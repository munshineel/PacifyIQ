"""Agent tools.

Nine tools over the real operational database and knowledge base. Every one
returns a structured `ToolResult`, never free text, so the agent reasons over
typed fields rather than parsing prose.

TIER ENFORCEMENT
----------------
Tools are grouped by blast radius, and the tier is enforced **in code**, not in
the prompt:

    TIER 1  read-only            autonomous
    TIER 2  creates a record     autonomous, reversible
    TIER 3  moves money or       NEVER autonomous - requires human approval
            changes an account     at ANY confidence

A prompt instruction not to issue refunds is a request. A code path that cannot
issue a refund is a guarantee. Tier 3 tools raise unless an explicit approval
token is supplied, which the agent has no way to produce.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable

from src.config.settings import settings
from src.db import queries as q
from src.db.connection import query_all, query_one


class Tier(IntEnum):
    READ_ONLY = 1
    CREATES_RECORD = 2
    MUTATING = 3


class ToolStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    INVALID_ARGS = "invalid_args"
    REFUSED = "refused"          # tier violation or policy block
    ERROR = "error"


@dataclass
class ToolResult:
    """Structured output. The agent never sees free text from a tool."""

    tool: str
    status: ToolStatus
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    tier: int = 1
    latency_ms: float = 0.0
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def summary(self) -> str:
        if self.ok:
            keys = ", ".join(list(self.data)[:5])
            return f"{self.tool} -> ok ({keys})"
        return f"{self.tool} -> {self.status.value}: {self.message}"

    def as_evidence(self) -> str:
        """Render for the generation prompt."""
        if not self.ok:
            return f"[TOOL {self.tool}] {self.status.value}: {self.message}"
        lines = [f"[TOOL {self.tool}]"]
        for k, v in self.data.items():
            if isinstance(v, (dict, list)):
                continue
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# =====================================================================
# Tool definitions
# =====================================================================

@dataclass
class ToolSpec:
    name: str
    tier: Tier
    description: str
    required_args: list[str]
    optional_args: list[str] = field(default_factory=list)
    fn: Callable[..., ToolResult] | None = None

    def schema(self) -> dict[str, Any]:
        """JSON schema, for a function-calling model."""
        props = {a: {"type": "string"} for a in self.required_args + self.optional_args}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": self.required_args,
            },
        }


def _timed(fn):
    def wrapper(*a, **kw) -> ToolResult:
        t0 = time.perf_counter()
        try:
            r = fn(*a, **kw)
        except Exception as e:  # a tool failure must not take down the request
            r = ToolResult(
                tool=getattr(fn, "__name__", "unknown"),
                status=ToolStatus.ERROR,
                message=f"{type(e).__name__}: {e}",
            )
        r.latency_ms = (time.perf_counter() - t0) * 1000
        return r
    return wrapper


# ---------------------------------------------------------------------
# TIER 1 — read only
# ---------------------------------------------------------------------

def refund_breakdown(order_id: str) -> dict:
    """Refund breakdown for an order, or an empty dict if not applicable.

    Refund figures travel with return eligibility because they are the same
    question for the customer: "can I return this, and what do I get back".
    Computed by the SQL waterfall, never by the model - a wrong refund figure
    stated fluently is the worst failure mode in this product.
    """
    quote = q.calculate_refund(order_id)
    if quote is None:
        return {}
    return {
        "price_paid": quote.price_paid,
        "restocking_fee": quote.restocking_fee,
        "return_shipping": quote.return_shipping,
        "refund_if_change_of_mind": quote.refund_change_of_mind,
        "refund_if_defective": quote.refund_defective,
        "store_credit_alternative": quote.store_credit_change_of_mind,
        "refund_timeline": quote.refund_timeline,
        "refund_caveat": quote.caveat,
        "disbursement": "requires human approval",
    }



@_timed
def get_customer(customer_id: str = "", email: str = "") -> ToolResult:
    """Customer record. Identity verification is enforced upstream."""
    if not customer_id and not email:
        return ToolResult("get_customer", ToolStatus.INVALID_ARGS,
                          message="customer_id or email required")
    if customer_id:
        row = query_one("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    else:
        row = query_one("SELECT * FROM customers WHERE email = ?", (email,))
    if not row:
        return ToolResult("get_customer", ToolStatus.NOT_FOUND,
                          message="no customer with that identifier",
                          args={"customer_id": customer_id, "email": email})
    return ToolResult("get_customer", ToolStatus.OK, data={
        "customer_id": row["customer_id"], "name": row["name"],
        "region": row["region"], "country": row["country"],
        "is_business": bool(row["is_business"]),
        "total_orders": row["total_orders"],
        "customer_since": row["created_at"],
    }, args={"customer_id": customer_id, "email": email})


@_timed
def get_order(order_id: str) -> ToolResult:
    """Order status, dates, product and payment method."""
    if not order_id:
        return ToolResult("get_order", ToolStatus.INVALID_ARGS,
                          message="order_id required")
    o = q.get_order(order_id)
    if o is None:
        return ToolResult("get_order", ToolStatus.NOT_FOUND,
                          message=f"no order found for {order_id!r}",
                          args={"order_id": order_id})
    return ToolResult("get_order", ToolStatus.OK, data={
        "order_id": o.order_id, "product": o.product_name, "brand": o.brand,
        "category": o.category, "quantity": o.quantity,
        "status": o.status, "order_date": o.order_date,
        "dispatch_date": o.dispatch_date, "delivery_date": o.delivery_date,
        "days_since_delivery": o.days_since_delivery,
        "payment_method": o.payment_method, "total_paid": o.total_paid,
        "is_opened": bool(o.is_opened), "region": o.region,
        "tracking_ref": o.tracking_ref,
    }, args={"order_id": order_id})


@_timed
def check_payment(order_id: str) -> ToolResult:
    """Payment and refund state for an order.

    Distinguishes a refund (order completed, then returned) from a failed-payment
    reversal (no order was created) - the ambiguity planted as DEFECT-08.
    """
    if not order_id:
        return ToolResult("check_payment", ToolStatus.INVALID_ARGS,
                          message="order_id required")
    o = q.get_order(order_id)
    if o is None:
        return ToolResult("check_payment", ToolStatus.NOT_FOUND,
                          message=f"no order found for {order_id!r}. If no order "
                                  f"exists, any debit is a failed-payment reversal, "
                                  f"not a refund.",
                          args={"order_id": order_id})

    ret = query_one(
        "SELECT * FROM returns WHERE order_id = ? ORDER BY requested_date DESC LIMIT 1",
        (q.normalise_order_id(order_id),),
    )
    timelines = {
        "upi": "3-5 business days", "credit_card": "5-7 business days",
        "debit_card": "5-7 business days", "net_banking": "5-7 business days",
        "emi": "7-14 business days (principal only)",
        "cod": "7-10 business days (bank details required)",
    }
    data = {
        "order_id": o.order_id, "payment_method": o.payment_method,
        "total_paid": o.total_paid, "order_status": o.status,
        "is_no_cost_emi": bool(o.is_no_cost_emi),
        # Simulated: no payment gateway exists. Labelled so a simulated fact
        # can never be mistaken for a real one downstream.
        "_source": "mock",
        "expected_refund_timeline": timelines.get(o.payment_method, "unknown"),
        "refund_exists": ret is not None,
    }
    if ret:
        data.update({
            "refund_status": ret["refund_status"],
            "refund_amount": ret["refund_amount"],
            "refund_initiated": ret["refund_initiated_date"],
            "restocking_fee_applied": ret["restocking_fee"],
        })
    return ToolResult("check_payment", ToolStatus.OK, data=data,
                      args={"order_id": order_id})


@_timed
def check_subscription(order_id: str) -> ToolResult:
    """PacifyCare+ extended-warranty status.

    Pacify sells no recurring subscription; the equivalent recurring-entitlement
    product is PacifyCare+, purchasable within 30 days of delivery (POL-WAR-001
    S9.2). Inventing a subscription product would contradict the corpus.
    """
    if not order_id:
        return ToolResult("check_subscription", ToolStatus.INVALID_ARGS,
                          message="order_id required")
    w = q.check_warranty_status(order_id)
    if w is None:
        return ToolResult("check_subscription", ToolStatus.NOT_FOUND,
                          message=f"no order found for {order_id!r}",
                          args={"order_id": order_id})
    return ToolResult("check_subscription", ToolStatus.OK, data={
        "order_id": w.order_id, "product": w.product_name,
        "plan": "PacifyCare+ (extended warranty)",
        "_source": "mock",
        "base_warranty_months": w.warranty_months,
        "months_since_delivery": w.months_since_delivery,
        "care_plus_purchasable": bool(w.care_plus_purchasable),
        "purchase_window": "within 30 days of delivery (POL-WAR-001 S9.2)",
        "note": ("still within the purchase window"
                 if w.care_plus_purchasable else
                 "purchase window has closed; PacifyCare+ cannot be added now"),
    }, args={"order_id": order_id})


@_timed
def check_policy(order_id: str, policy: str = "return") -> ToolResult:
    """Eligibility under a named policy. Computed in SQL, never by a model.

    `policy` is one of: return, warranty, refund.
    """
    if not order_id:
        return ToolResult("check_policy", ToolStatus.INVALID_ARGS,
                          message="order_id required")
    policy = (policy or "return").lower()

    if policy in ("return", "returns"):
        e = q.check_return_eligibility(order_id)
        if e is None:
            return ToolResult("check_policy", ToolStatus.NOT_FOUND,
                              message=f"no order found for {order_id!r}",
                              args={"order_id": order_id, "policy": policy})
        return ToolResult("check_policy", ToolStatus.OK, data={
            "policy": "return", "order_id": e.order_id,
            "eligible": e.is_eligible, "eligibility": e.eligibility,
            "window_days": e.window_days,
            "days_since_delivery": e.days_since_delivery,
            "days_remaining": e.days_remaining,
            "basis": e.window_basis, "remedy_path": e.remedy_path,
            "region": e.region, "is_opened": bool(e.is_opened),
            # Refund figures travel with eligibility because they are the same
            # question for the customer: "can I return this, and what do I get
            # back". Computed by the SQL waterfall, never by the model - a
            # wrong refund figure stated fluently is the worst failure here.
            **refund_breakdown(order_id),
        }, args={"order_id": order_id, "policy": policy})

    if policy in ("warranty", "warranties"):
        w = q.check_warranty_status(order_id)
        if w is None:
            return ToolResult("check_policy", ToolStatus.NOT_FOUND,
                              message=f"no order found for {order_id!r}",
                              args={"order_id": order_id, "policy": policy})
        return ToolResult("check_policy", ToolStatus.OK, data={
            "policy": "warranty", "order_id": w.order_id,
            "covered": w.is_covered, "state": w.warranty_state,
            "warranty_months": w.warranty_months,
            "months_since_delivery": w.months_since_delivery,
            "days_remaining": w.days_remaining,
            "administered_by": w.warranty_route,
            "routing_note": w.routing_note,
            "brand": w.brand,
        }, args={"order_id": order_id, "policy": policy})

    if policy in ("refund", "refunds"):
        r = q.calculate_refund(order_id)
        if r is None:
            return ToolResult("check_policy", ToolStatus.NOT_FOUND,
                              message=f"no order found for {order_id!r}",
                              args={"order_id": order_id, "policy": policy})
        return ToolResult("check_policy", ToolStatus.OK, data={
            "policy": "refund", "order_id": r.order_id,
            "eligibility": r.eligibility, "price_paid": r.price_paid,
            "restocking_fee": r.restocking_fee,
            "return_shipping": r.return_shipping,
            "refund_change_of_mind": r.refund_change_of_mind,
            "refund_defective": r.refund_defective,
            "store_credit": r.store_credit_change_of_mind,
            "timeline": r.refund_timeline, "caveat": r.caveat,
            "breakdown": r.explain(),
        }, args={"order_id": order_id, "policy": policy})

    return ToolResult("check_policy", ToolStatus.INVALID_ARGS,
                      message=f"unknown policy {policy!r}; expected "
                              f"return, warranty or refund")


@_timed
def search_products(query: str = "", category: str = "",
                    in_stock_only: bool = False) -> ToolResult:
    """Product catalogue. Specifications come from the knowledge base, not here."""
    rows = q.search_products(query=query, category=category or None,
                             in_stock_only=in_stock_only)
    if not rows:
        return ToolResult("search_products", ToolStatus.NOT_FOUND,
                          message=f"no products matched {query!r}",
                          args={"query": query, "category": category})
    return ToolResult("search_products", ToolStatus.OK, data={
        "n_results": len(rows),
        "products": [
            {"sku": r["sku"], "name": r["name"], "price": r["price"],
             "brand": r["brand"], "in_stock": bool(r["in_stock"]),
             "warranty_months": r["warranty_months"]}
            for r in rows[:6]
        ],
    }, args={"query": query, "category": category})


@_timed
def search_knowledge_base(query: str, top_k: int = 5,
                          region: str = "") -> ToolResult:
    """Retrieval over the policy corpus. Wraps the Phase 6/6.5 retriever."""
    if not query:
        return ToolResult("search_knowledge_base", ToolStatus.INVALID_ARGS,
                          message="query required")
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.retriever import Retriever
    from src.knowledge.vector_store import VectorStore

    global _RETRIEVER
    if _RETRIEVER is None:
        store = VectorStore.load(settings.index_dir)
        emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
        _RETRIEVER = Retriever(store, emb, BM25Index(store.chunks),
                               strategy="rrf_w", top_k=top_k)

    res = _RETRIEVER.retrieve(query, top_k=top_k, region=region or None)
    return ToolResult("search_knowledge_base", ToolStatus.OK, data={
        "n_results": len(res.hits),
        "max_bm25": round(res.max_bm25_score, 2),
        "max_cosine": round(res.max_dense_score, 4),
        "has_conflict": res.has_conflict(),
        "chunks": [
            {"citation": h.chunk.citation, "doc": h.chunk.doc,
             "section": h.chunk.section, "text": h.chunk.text,
             "score": round(h.score, 4), "version": h.chunk.version,
             # region is needed downstream to tell a genuine contradiction from
             # a regional variant that does not apply to this customer
             "region": h.chunk.region, "doc_type": h.chunk.doc_type}
            for h in res.hits
        ],
    }, args={"query": query, "top_k": top_k, "region": region})


_RETRIEVER = None


@_timed
def analyze_screenshot(image_path: str, user_text: str = "",
                       backend: str = "local_ocr") -> ToolResult:
    """Vision analysis. Wraps the Phase 8 multimodal layer."""
    if not image_path:
        return ToolResult("analyze_screenshot", ToolStatus.INVALID_ARGS,
                          message="image_path required")
    from src.multimodal.vision import analyze_image

    a = analyze_image(image_path, user_text=user_text, backend=backend)
    if not a.is_useful:
        return ToolResult("analyze_screenshot", ToolStatus.NOT_FOUND,
                          message=a.reason or "image contained no usable information",
                          data=a.to_dict(), args={"image_path": str(image_path)})
    return ToolResult("analyze_screenshot", ToolStatus.OK, data=a.to_dict(),
                      args={"image_path": str(image_path)})


# ---------------------------------------------------------------------
# TIER 2 — creates a record, reversible
# ---------------------------------------------------------------------

@_timed
def create_support_ticket(summary: str, intent: str = "", customer_id: str = "",
                          order_id: str = "", priority: str = "medium",
                          sentiment: str = "neutral") -> ToolResult:
    """Open a support ticket. Autonomous: creating a ticket is reversible and
    the alternative is losing the customer's context."""
    if not summary:
        return ToolResult("create_support_ticket", ToolStatus.INVALID_ARGS,
                          message="summary required", tier=2)
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    from src.db.connection import get_connection

    with get_connection() as con:
        con.execute(
            "INSERT INTO tickets (ticket_id, customer_id, order_id, created_at, "
            "intent, sentiment, priority, status, resolved_by, summary) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ticket_id, customer_id or None,
             q.normalise_order_id(order_id) if order_id else None,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             intent or "unknown", sentiment, priority, "open", "pending",
             summary[:500]),
        )
    return ToolResult("create_support_ticket", ToolStatus.OK, data={
        "ticket_id": ticket_id, "status": "open", "priority": priority,
        "summary": summary[:200],
    }, tier=2, args={"summary": summary[:80], "order_id": order_id})


@_timed
def escalate_to_human(reason: str, context: str = "", intent: str = "",
                      order_id: str = "", customer_id: str = "",
                      priority: str = "high") -> ToolResult:
    """Hand off to a human agent, carrying the context already gathered.

    The handoff package matters: 'transferring you' is a phone tree. Passing
    the order, the policy position and the steps already taken is deflection.
    """
    if not reason:
        return ToolResult("escalate_to_human", ToolStatus.INVALID_ARGS,
                          message="reason required", tier=2)
    ticket = create_support_ticket(
        summary=f"[ESCALATED] {reason}. {context}"[:500],
        intent=intent, customer_id=customer_id, order_id=order_id,
        priority=priority, sentiment="negative",
    )
    return ToolResult("escalate_to_human", ToolStatus.OK, data={
        "escalated": True, "reason": reason,
        "ticket_id": ticket.data.get("ticket_id"),
        "queue": "L2_specialist" if priority == "high" else "L1_support",
        "context_handed_over": context[:300],
    }, tier=2, args={"reason": reason, "order_id": order_id})


# ---------------------------------------------------------------------
# TIER 3 — mutating. Never autonomous.
# ---------------------------------------------------------------------

APPROVAL_TOKEN_PREFIX = "HUMAN-APPROVED-"


def _tier3_guard(tool: str, approval_token: str | None) -> ToolResult | None:
    """Refuse unless a human approval token is present.

    The agent has no code path that can mint one. This is the difference
    between a prompt instruction and a guarantee.
    """
    if not approval_token or not str(approval_token).startswith(APPROVAL_TOKEN_PREFIX):
        return ToolResult(
            tool, ToolStatus.REFUSED, tier=3,
            message=(f"{tool} is a tier-3 action and requires human approval. "
                     f"The assistant cannot authorise it at any confidence level."),
        )
    return None


@_timed
def approve_refund(order_id: str, amount: float = 0.0,
                   approval_token: str | None = None) -> ToolResult:
    """TIER 3. Refused without human approval."""
    blocked = _tier3_guard("approve_refund", approval_token)
    if blocked:
        return blocked
    return ToolResult("approve_refund", ToolStatus.OK, tier=3, data={
        "order_id": q.normalise_order_id(order_id), "amount": amount,
        "approved_by": approval_token,
    })


@_timed
def cancel_order(order_id: str, approval_token: str | None = None) -> ToolResult:
    """TIER 3. Refused without human approval."""
    blocked = _tier3_guard("cancel_order", approval_token)
    if blocked:
        return blocked
    return ToolResult("cancel_order", ToolStatus.OK, tier=3,
                      data={"order_id": q.normalise_order_id(order_id),
                            "status": "cancelled"})


@_timed
def modify_account(customer_id: str, field_name: str = "", value: str = "",
                   approval_token: str | None = None) -> ToolResult:
    """TIER 3. Refused without human approval AND identity verification."""
    blocked = _tier3_guard("modify_account", approval_token)
    if blocked:
        return blocked
    return ToolResult("modify_account", ToolStatus.OK, tier=3,
                      data={"customer_id": customer_id, "field": field_name})


# =====================================================================
# Registry
# =====================================================================

REGISTRY: dict[str, ToolSpec] = {
    s.name: s for s in [
        ToolSpec("get_customer", Tier.READ_ONLY,
                 "Look up a customer by id or email.",
                 [], ["customer_id", "email"], get_customer),
        ToolSpec("get_order", Tier.READ_ONLY,
                 "Order status, dates, product, payment method.",
                 ["order_id"], [], get_order),
        ToolSpec("check_payment", Tier.READ_ONLY,
                 "Payment and refund state; distinguishes a refund from a "
                 "failed-payment reversal.",
                 ["order_id"], [], check_payment),
        ToolSpec("check_subscription", Tier.READ_ONLY,
                 "PacifyCare+ extended warranty status and purchase window.",
                 ["order_id"], [], check_subscription),
        ToolSpec("check_policy", Tier.READ_ONLY,
                 "Eligibility under the return, warranty or refund policy. "
                 "Computed in SQL, not by a model.",
                 ["order_id"], ["policy"], check_policy),
        ToolSpec("search_products", Tier.READ_ONLY,
                 "Product catalogue: price, stock, brand, warranty length.",
                 [], ["query", "category", "in_stock_only"], search_products),
        ToolSpec("search_knowledge_base", Tier.READ_ONLY,
                 "Retrieve policy documentation.",
                 ["query"], ["top_k", "region"], search_knowledge_base),
        ToolSpec("analyze_screenshot", Tier.READ_ONLY,
                 "Extract error codes and UI context from a customer screenshot.",
                 ["image_path"], ["user_text"], analyze_screenshot),
        ToolSpec("create_support_ticket", Tier.CREATES_RECORD,
                 "Open a support ticket.",
                 ["summary"], ["intent", "customer_id", "order_id", "priority"],
                 create_support_ticket),
        ToolSpec("escalate_to_human", Tier.CREATES_RECORD,
                 "Hand off to a human agent with the gathered context.",
                 ["reason"], ["context", "intent", "order_id", "priority"],
                 escalate_to_human),
        ToolSpec("approve_refund", Tier.MUTATING,
                 "TIER 3. Issue a refund. Requires human approval.",
                 ["order_id"], ["amount", "approval_token"], approve_refund),
        ToolSpec("cancel_order", Tier.MUTATING,
                 "TIER 3. Cancel an order. Requires human approval.",
                 ["order_id"], ["approval_token"], cancel_order),
        ToolSpec("modify_account", Tier.MUTATING,
                 "TIER 3. Change account details. Requires human approval.",
                 ["customer_id"], ["field_name", "value", "approval_token"],
                 modify_account),
    ]
}

AUTONOMOUS_TOOLS = [n for n, s in REGISTRY.items() if s.tier <= Tier.CREATES_RECORD]
TIER3_TOOLS = [n for n, s in REGISTRY.items() if s.tier == Tier.MUTATING]


def call_tool(name: str, **kwargs) -> ToolResult:
    """Dispatch by name, enforcing the tier boundary."""
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult(name, ToolStatus.INVALID_ARGS,
                          message=f"unknown tool {name!r}")
    if spec.tier == Tier.MUTATING and not kwargs.get("approval_token"):
        return ToolResult(
            name, ToolStatus.REFUSED, tier=3,
            message=(f"{name} is a tier-3 action. The assistant cannot "
                     f"authorise it; a human must approve."),
            args={k: v for k, v in kwargs.items() if k != "approval_token"},
        )
    # Validate arguments here rather than relying on TypeError from the call.
    # A missing order id is a routing mistake the agent can recover from; an
    # unhandled TypeError is not.
    missing = [a for a in spec.required_args if not kwargs.get(a)]
    if missing:
        return ToolResult(
            name, ToolStatus.INVALID_ARGS, tier=int(spec.tier),
            message=f"missing required argument(s): {', '.join(missing)}",
            data={"missing": missing}, args=kwargs)

    try:
        result = spec.fn(**kwargs)
    except TypeError as e:
        # Wrong or unexpected keyword. Callers pass extra context freely, so
        # this must degrade rather than crash.
        return ToolResult(
            name, ToolStatus.INVALID_ARGS, tier=int(spec.tier),
            message=f"invalid arguments: {e}", args=kwargs)
    except Exception as e:
        # A failing tool is a normal event in a support system - an upstream
        # service is down, a file is locked. It must be representable in the
        # return value so the agent can route on it, never an exception that
        # takes down the customer's request.
        return ToolResult(
            name, ToolStatus.ERROR, tier=int(spec.tier),
            message=f"{type(e).__name__}: {e}", args=kwargs)

    if not isinstance(result, ToolResult):
        return ToolResult(
            name, ToolStatus.ERROR, tier=int(spec.tier),
            message=f"tool returned {type(result).__name__}, expected ToolResult")

    result.tier = int(spec.tier)
    return result


def tool_schemas(tier_max: Tier = Tier.CREATES_RECORD) -> list[dict[str, Any]]:
    """Schemas for the tools an agent is allowed to call."""
    return [s.schema() for s in REGISTRY.values() if s.tier <= tier_max]


if __name__ == "__main__":
    import json

    print(f"{len(REGISTRY)} tools: {len(AUTONOMOUS_TOOLS)} autonomous, "
          f"{len(TIER3_TOOLS)} requiring approval\n")
    for name, spec in REGISTRY.items():
        print(f"  T{int(spec.tier)}  {name:24s} {spec.description[:56]}")

    print("\n--- sample calls ---")
    for name, kw in [
        ("get_order", {"order_id": "12345"}),
        ("check_policy", {"order_id": "PAC-2026-12345", "policy": "return"}),
        ("check_payment", {"order_id": "PAC-2026-12364"}),
        ("check_subscription", {"order_id": "PAC-2026-12345"}),
        ("get_order", {"order_id": "PAC-2026-99999"}),
        ("approve_refund", {"order_id": "PAC-2026-12345", "amount": 57960}),
    ]:
        r = call_tool(name, **kw)
        print(f"\n{r.summary()}")
        if r.ok:
            print("   " + json.dumps(
                {k: v for k, v in list(r.data.items())[:5]}, default=str))
