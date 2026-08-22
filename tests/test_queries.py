"""Tests for the data layer.

These verify the SQL views produce correct answers on the deterministic
edge-case orders. No LLM in the loop — business logic is tested in isolation.
"""
import pytest

from src.db.queries import (
    calculate_refund,
    check_return_eligibility,
    check_warranty_status,
    get_order,
    normalise_order_id,
    search_products,
)

pytestmark = pytest.mark.data


# ---------------------------------------------------------------- ids
@pytest.mark.parametrize("raw,expected", [
    ("PAC-2026-12345", "PAC-2026-12345"),
    ("12345",          "PAC-2026-12345"),
    ("#12345",         "PAC-2026-12345"),
    ("pac-2026-12345", "PAC-2026-12345"),
    ("PAC202612345",   "PAC-2026-12345"),
    ("  12345  ",      "PAC-2026-12345"),
])
def test_order_id_normalisation(raw, expected):
    """Argument extraction is the top agent failure mode, so it lives in code."""
    assert normalise_order_id(raw) == expected


# --------------------------------------------------- return eligibility
@pytest.mark.parametrize("order_id,eligibility,window_days", [
    ("PAC-2026-12345", "eligible", 14),   # day 12 of 14, opened
    ("PAC-2026-12346", "eligible", 14),   # day 14, last day
    ("PAC-2026-12347", "expired",  14),   # day 15, one day late
    ("PAC-2026-12348", "eligible", 30),   # sealed, day 22
    ("PAC-2026-12349", "expired",  30),   # sealed, day 31
    ("PAC-2026-12367", "eligible",  7),   # bulk 6 units, day 5
    ("PAC-2026-12368", "expired",   7),   # bulk 8 units, day 9
    ("PAC-2026-12369", "eligible", 30),   # accessory, day 25
])
def test_return_windows(order_id, eligibility, window_days):
    e = check_return_eligibility(order_id)
    assert e is not None
    assert e.eligibility == eligibility
    assert e.window_days == window_days


def test_boundary_last_day_is_still_eligible():
    """Day 14 of a 14-day window must be inclusive."""
    e = check_return_eligibility("PAC-2026-12346")
    assert e.days_since_delivery == 14
    assert e.is_eligible


def test_boundary_one_day_late_expires():
    e = check_return_eligibility("PAC-2026-12347")
    assert e.days_since_delivery == 15
    assert not e.is_eligible


# ------------------------------------------------------- EU override
def test_eu_gets_14_day_window_even_when_opened():
    """POL-EU-001 S2 overrides the opened/sealed distinction."""
    e = check_return_eligibility("PAC-2026-12354")
    assert e.region == "EU"
    assert e.is_opened == 1
    assert e.window_days == 14
    assert "POL-EU-001" in e.window_basis


def test_eu_pays_no_restocking_fee():
    """POL-EU-001 S4.1. Same product and day count as an Indian order,
    but zero fees. The jurisdictional override is enforced in the view."""
    eu = calculate_refund("PAC-2026-12354")
    assert eu.restocking_fee == 0
    assert eu.return_shipping == 0
    assert eu.refund_change_of_mind == eu.price_paid


def test_india_does_pay_restocking_fee():
    """Contrast case for the EU test above."""
    india = calculate_refund("PAC-2026-12345")
    assert india.restocking_fee > 0
    assert india.return_shipping == 450


# --------------------------------------------------- refund waterfall
def test_refund_waterfall_arithmetic():
    """refund = price - restocking - return_shipping (POL-REF-001 S4.1)."""
    q = calculate_refund("PAC-2026-12345")
    expected = q.price_paid - q.restocking_fee - q.return_shipping
    assert q.refund_change_of_mind == pytest.approx(expected, abs=0.01)


def test_restocking_is_ten_percent_of_price_paid():
    """S4.2: calculated on price actually paid, not list price."""
    q = calculate_refund("PAC-2026-12345")
    assert q.restocking_fee == pytest.approx(q.price_paid * 0.10, abs=1)


def test_store_credit_carries_five_percent_bonus():
    q = calculate_refund("PAC-2026-12345")
    assert q.store_credit_change_of_mind == pytest.approx(
        q.refund_change_of_mind * 1.05, abs=1
    )


def test_defective_return_waives_fees_and_refunds_shipping():
    q = calculate_refund("PAC-2026-12345")
    assert q.refund_defective == pytest.approx(
        q.price_paid + q.original_shipping, abs=0.01
    )
    assert q.refund_defective > q.refund_change_of_mind


def test_accessory_has_no_restocking_fee():
    q = calculate_refund("PAC-2026-12369")
    assert q.restocking_fee == 0


def test_no_cost_emi_carries_a_caveat():
    """DEFECT-07: the customer must be told the refund is of the
    discounted amount, not the list price."""
    q = calculate_refund("PAC-2026-12352")
    assert q.caveat is not None
    assert "discounted" in q.caveat.lower()


def test_standard_emi_caveat_mentions_principal():
    """DEFECT-06: bank interest is not refunded."""
    q = calculate_refund("PAC-2026-12353")
    assert q.caveat is not None
    assert "principal" in q.caveat.lower()


# ------------------------------------------------------------ warranty
def test_pacify_brand_is_administered_by_pacify():
    w = check_warranty_status("PAC-2026-12356")
    assert w.brand == "Pacify"
    assert w.warranty_months == 24
    assert w.is_covered
    assert w.handled_by_pacify


def test_third_party_brand_routes_to_manufacturer():
    """DEFECT-09: same symptom, same age, different route."""
    w = check_warranty_status("PAC-2026-12357")
    assert w.brand == "Northwind"
    assert w.warranty_months == 12
    assert w.is_covered
    assert not w.handled_by_pacify
    assert "manufacturer" in w.routing_note.lower()


def test_expired_warranty_detected():
    w = check_warranty_status("PAC-2026-12358")  # 25 months, 24-month cover
    assert not w.is_covered


def test_accessory_warranty_is_six_months():
    w = check_warranty_status("PAC-2026-12359")  # 7 months old
    assert w.warranty_months == 6
    assert not w.is_covered


def test_refurbished_warranty_is_six_months():
    w = check_warranty_status("PAC-2026-12360")  # 5 months old
    assert w.warranty_months == 6
    assert w.is_covered


# ----------------------------------------------------- remedy paths
def test_doa_window_flagged_within_48h():
    e = check_return_eligibility("PAC-2026-12350")
    assert e.remedy_path == "doa_window_open"


def test_grey_zone_flagged_between_doa_and_window_close():
    """DEFECT-12: day 8 is past the 48h DOA window but inside the
    14-day return window, so both remedies are arguable."""
    e = check_return_eligibility("PAC-2026-12351")
    assert e.days_since_delivery == 8
    assert e.remedy_path == "grey_zone_return_or_warranty"


# ------------------------------------------------------ misc lookups
def test_missing_order_returns_none():
    """Tool-failure path: must degrade gracefully, not raise."""
    assert get_order("PAC-2026-99999") is None
    assert check_return_eligibility("PAC-2026-99999") is None


def test_undelivered_order_is_not_returnable():
    e = check_return_eligibility("PAC-2026-12362")  # in transit
    assert e.eligibility == "not_delivered"


def test_product_search_filters_by_stock():
    all_products = search_products()
    in_stock = search_products(in_stock_only=True)
    assert len(in_stock) < len(all_products)
    assert all(p["in_stock"] == 1 for p in in_stock)


def test_product_search_by_name():
    results = search_products("probook")
    assert len(results) >= 3
    assert all("ProBook" in p["name"] for p in results)
