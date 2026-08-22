"""Order, return, warranty and refund lookups.

These are the functions the agent tools will call in Phase 10. Each is a thin
Python wrapper over a SQL view, returning a typed dataclass rather than a raw
row so downstream code (and the LLM prompt) gets stable field names.

The business rules live in sql/01_business_logic_views.sql, not here. This
module is the Python surface over them.

    from src.db.queries import get_order, check_return_eligibility

    order = get_order("PAC-2026-12345")
    elig  = check_return_eligibility("PAC-2026-12345")
    print(elig.eligibility, elig.days_remaining, elig.window_basis)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from src.db.connection import query_one, query_all


# =====================================================================
# Result types
# =====================================================================

@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    customer_name: str
    region: str
    country: str
    product_name: str
    sku: str
    category: str
    brand: str
    quantity: int
    unit_price: float
    total_paid: float
    payment_method: str
    is_no_cost_emi: int
    order_date: str
    dispatch_date: str | None
    delivery_date: str | None
    status: str
    shipping_method: str
    is_opened: int
    tracking_ref: str | None
    days_since_delivery: int | None
    days_since_dispatch: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnEligibility:
    order_id: str
    eligibility: str          # eligible | expired | not_delivered | not_applicable | already_in_progress
    window_days: int
    days_since_delivery: int | None
    days_remaining: int
    window_basis: str         # doubles as the policy citation
    remedy_path: str | None   # doa_window_open | grey_zone_return_or_warranty | warranty_only
    region: str
    product_name: str
    is_opened: int
    quantity: int

    @property
    def is_eligible(self) -> bool:
        return self.eligibility == "eligible"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WarrantyStatus:
    order_id: str
    product_name: str
    brand: str
    warranty_months: int
    months_since_delivery: float | None
    warranty_state: str        # in_warranty | expired | not_delivered
    days_remaining: int | None
    warranty_route: str        # pacify_administered | manufacturer_administered
    routing_note: str
    care_plus_purchasable: int

    @property
    def is_covered(self) -> bool:
        return self.warranty_state == "in_warranty"

    @property
    def handled_by_pacify(self) -> bool:
        return self.warranty_route == "pacify_administered"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefundQuote:
    order_id: str
    eligibility: str
    region: str
    payment_method: str
    price_paid: float
    restocking_fee: float
    return_shipping: float
    original_shipping: float
    refund_change_of_mind: float
    refund_defective: float
    store_credit_change_of_mind: float
    refund_timeline: str
    caveat: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def explain(self, reason: str = "change_of_mind") -> str:
        """Human-readable breakdown. Feed this to the LLM to narrate, so the
        model explains a computed figure rather than deriving one itself."""
        if reason in ("defective", "damaged_in_transit", "wrong_item"):
            return (
                f"Item price paid: Rs {self.price_paid:,.0f}\n"
                f"Restocking fee: waived (defective/wrong item)\n"
                f"Return shipping: free (Pacify arranged)\n"
                f"Original shipping refunded: Rs {self.original_shipping:,.0f}\n"
                f"Refund payable: Rs {self.refund_defective:,.0f}\n"
                f"Timeline: {self.refund_timeline}"
            )
        return (
            f"Item price paid: Rs {self.price_paid:,.0f}\n"
            f"Less restocking fee: Rs {self.restocking_fee:,.0f}\n"
            f"Less return shipping: Rs {self.return_shipping:,.0f}\n"
            f"Refund payable: Rs {self.refund_change_of_mind:,.0f}\n"
            f"Store credit alternative: Rs {self.store_credit_change_of_mind:,.0f} (+5%)\n"
            f"Timeline: {self.refund_timeline}"
            + (f"\nNote: {self.caveat}" if self.caveat else "")
        )


# =====================================================================
# Lookups
# =====================================================================

def normalise_order_id(raw: str) -> str:
    """Turn what a customer types into a canonical order ID.

    Customers write '12345', '#12347', 'pac-2026-12350', 'PAC202612345'.
    Argument-extraction errors are the most common agent failure mode, so
    normalisation happens in code rather than being left to the model.
    """
    s = "".join(ch for ch in raw.strip() if ch.isalnum()).upper()
    if s.startswith("PAC"):
        digits = s[3:]
        if digits.startswith("2026"):
            digits = digits[4:]
        return f"PAC-2026-{digits}"
    if s.isdigit():
        return f"PAC-2026-{s}"
    return raw.strip().upper()


def get_order(order_id: str) -> Order | None:
    """Full order detail, or None if not found."""
    oid = normalise_order_id(order_id)
    row = query_one(
        """
        SELECT order_id, customer_id, customer_name, region, country,
               product_name, sku, category, brand, quantity, unit_price,
               total_paid, payment_method, is_no_cost_emi, order_date,
               dispatch_date, delivery_date, status, shipping_method,
               is_opened, tracking_ref, days_since_delivery, days_since_dispatch
        FROM v_order_detail
        WHERE order_id = ?
        """,
        (oid,),
    )
    return Order(**row) if row else None


def check_return_eligibility(order_id: str) -> ReturnEligibility | None:
    """Return-window decision. Rules live in v_return_eligibility."""
    oid = normalise_order_id(order_id)
    row = query_one(
        """
        SELECT order_id, eligibility, window_days, days_since_delivery,
               days_remaining, window_basis, remedy_path, region,
               product_name, is_opened, quantity
        FROM v_return_eligibility
        WHERE order_id = ?
        """,
        (oid,),
    )
    return ReturnEligibility(**row) if row else None


def check_warranty_status(order_id: str) -> WarrantyStatus | None:
    """Warranty coverage and which party administers the claim."""
    oid = normalise_order_id(order_id)
    row = query_one(
        """
        SELECT order_id, product_name, brand, warranty_months,
               months_since_delivery, warranty_state, days_remaining,
               warranty_route, routing_note, care_plus_purchasable
        FROM v_warranty_status
        WHERE order_id = ?
        """,
        (oid,),
    )
    return WarrantyStatus(**row) if row else None


def calculate_refund(order_id: str) -> RefundQuote | None:
    """Refund figures from the policy fee waterfall. Computed in SQL, never
    by the LLM — a wrong refund figure stated fluently is the worst failure
    mode in this product."""
    oid = normalise_order_id(order_id)
    row = query_one(
        """
        SELECT order_id, eligibility, region, payment_method, price_paid,
               restocking_fee, return_shipping, original_shipping,
               refund_change_of_mind, refund_defective,
               store_credit_change_of_mind, refund_timeline, caveat
        FROM v_refund_quote
        WHERE order_id = ?
        """,
        (oid,),
    )
    return RefundQuote(**row) if row else None


def search_products(
    query: str = "", category: str | None = None, in_stock_only: bool = False
) -> list[dict[str, Any]]:
    """Product catalogue search. Specs come from the corpus, not from here."""
    sql = """
        SELECT sku, name, category, brand, price, warranty_months,
               in_stock, is_refurbished
        FROM products
        WHERE 1 = 1
    """
    params: list[Any] = []
    if query:
        sql += " AND LOWER(name) LIKE ?"
        params.append(f"%{query.lower()}%")
    if category:
        sql += " AND category = ?"
        params.append(category)
    if in_stock_only:
        sql += " AND in_stock = 1"
    sql += " ORDER BY price DESC"
    return query_all(sql, tuple(params))


def get_customer_orders(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent orders for a customer. Gated by identity verification upstream."""
    return query_all(
        """
        SELECT order_id, product_name, order_date, delivery_date,
               status, total_paid
        FROM v_order_detail
        WHERE customer_id = ?
        ORDER BY order_date DESC
        LIMIT ?
        """,
        (customer_id, limit),
    )


if __name__ == "__main__":
    for oid in ["PAC-2026-12345", "12347", "#12354", "pac-2026-12357"]:
        o = get_order(oid)
        e = check_return_eligibility(oid)
        print(f"\n{oid!r} -> {normalise_order_id(oid)}")
        if o and e:
            print(f"  {o.product_name} [{o.region}] {o.status}")
            print(f"  {e.eligibility}, {e.days_remaining}d remaining, {e.window_basis}")
