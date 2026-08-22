-- =====================================================================
-- PacifyIQ — Business logic views
-- =====================================================================
-- Return and warranty eligibility encoded as SQL rather than Python.
--
-- WHY SQL: eligibility is a deterministic function of stored facts
-- (delivery date, opened flag, category, quantity, region). Putting it in
-- SQL means one definition, testable in isolation, and the LLM never
-- computes it. Reproduces canonical_facts.md sections 1, 3 and 7.
--
-- Requires SQLite 3.25+ for window functions.
-- =====================================================================

DROP VIEW IF EXISTS v_return_eligibility;
DROP VIEW IF EXISTS v_warranty_status;
DROP VIEW IF EXISTS v_refund_quote;
DROP VIEW IF EXISTS v_order_detail;
DROP VIEW IF EXISTS v_customer_contact_history;

-- ---------------------------------------------------------------------
-- v_order_detail — denormalised order view. Joins the three tables every
-- tool needs, so tool queries stay single-statement.
-- ---------------------------------------------------------------------
CREATE VIEW v_order_detail AS
SELECT
    o.order_id,
    o.customer_id,
    c.name              AS customer_name,
    c.region,
    c.country,
    c.is_business,
    o.sku,
    p.name              AS product_name,
    p.category,
    p.brand,
    p.warranty_months,
    p.restocking_pct,
    p.is_refurbished,
    o.quantity,
    o.unit_price,
    o.discount,
    o.shipping_charge,
    o.total_paid,
    o.payment_method,
    o.emi_tenure,
    o.is_no_cost_emi,
    o.order_date,
    o.dispatch_date,
    o.delivery_date,
    o.status,
    o.shipping_method,
    o.is_opened,
    o.tracking_ref,
    -- date arithmetic: days elapsed since delivery
    CASE WHEN o.delivery_date IS NULL THEN NULL
         ELSE CAST(julianday('2026-08-21') - julianday(o.delivery_date) AS INTEGER)
    END                 AS days_since_delivery,
    CASE WHEN o.dispatch_date IS NULL THEN NULL
         ELSE CAST(julianday('2026-08-21') - julianday(o.dispatch_date) AS INTEGER)
    END                 AS days_since_dispatch,
    -- brand routing (canonical S3 / DEFECT-09)
    CASE WHEN p.brand = 'Pacify' THEN 'pacify_administered'
         ELSE 'manufacturer_administered'
    END                 AS warranty_route
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.sku         = o.sku;


-- ---------------------------------------------------------------------
-- v_return_eligibility — the return window decision.
--
-- Window precedence (canonical S1, S7):
--   1. EU customers      -> 14 days, opened or not   (POL-EU-001 S2)
--   2. Bulk (qty >= 5)   -> 7 days                   (POL-RET-002 S13)
--   3. Refurbished       -> 14 days                  (POL-RET-002 S2)
--   4. Accessories       -> 30 days                  (POL-RET-002 S2)
--   5. Opened            -> 14 days                  (POL-RET-002 S2)
--   6. Sealed            -> 30 days                  (POL-RET-002 S2)
-- ---------------------------------------------------------------------
CREATE VIEW v_return_eligibility AS
WITH windowed AS (
    SELECT
        d.*,
        CASE
            WHEN d.region      = 'EU'        THEN 14
            WHEN d.quantity   >= 5           THEN 7
            WHEN d.is_refurbished = 1        THEN 14
            WHEN d.category   = 'accessory'  THEN 30
            WHEN d.is_opened  = 1            THEN 14
            ELSE 30
        END AS window_days,
        CASE
            WHEN d.region      = 'EU'        THEN 'POL-EU-001 S2 (statutory, overrides base policy)'
            WHEN d.quantity   >= 5           THEN 'POL-RET-002 S13 (bulk order)'
            WHEN d.is_refurbished = 1        THEN 'POL-RET-002 S2 (refurbished)'
            WHEN d.category   = 'accessory'  THEN 'POL-RET-002 S2 (accessory)'
            WHEN d.is_opened  = 1            THEN 'POL-RET-002 S2 (opened electronics)'
            ELSE 'POL-RET-002 S2 (sealed)'
        END AS window_basis
    FROM v_order_detail d
)
SELECT
    order_id, customer_id, region, product_name, category, brand,
    quantity, is_opened, delivery_date, days_since_delivery,
    window_days, window_basis, status,
    (window_days - COALESCE(days_since_delivery, 0)) AS days_remaining,
    CASE
        WHEN delivery_date IS NULL              THEN 'not_delivered'
        WHEN status IN ('cancelled','returned')  THEN 'not_applicable'
        WHEN status = 'refund_in_progress'       THEN 'already_in_progress'
        WHEN days_since_delivery <= window_days  THEN 'eligible'
        ELSE 'expired'
    END AS eligibility,
    -- the 48h DOA window and the grey zone (DEFECT-12)
    CASE
        WHEN days_since_delivery IS NULL         THEN NULL
        WHEN days_since_delivery <= 2             THEN 'doa_window_open'
        WHEN days_since_delivery <= window_days   THEN 'grey_zone_return_or_warranty'
        ELSE 'warranty_only'
    END AS remedy_path
FROM windowed;


-- ---------------------------------------------------------------------
-- v_warranty_status — coverage period and administering party.
-- ---------------------------------------------------------------------
CREATE VIEW v_warranty_status AS
SELECT
    d.order_id, d.customer_id, d.product_name, d.brand, d.category,
    d.warranty_months, d.is_refurbished, d.delivery_date,
    d.days_since_delivery,
    ROUND(d.days_since_delivery / 30.44, 1) AS months_since_delivery,
    d.warranty_route,
    CASE
        WHEN d.delivery_date IS NULL THEN 'not_delivered'
        WHEN d.days_since_delivery <= d.warranty_months * 30.44 THEN 'in_warranty'
        ELSE 'expired'
    END AS warranty_state,
    CAST(d.warranty_months * 30.44 - d.days_since_delivery AS INTEGER) AS days_remaining,
    -- PacifyCare+ purchasable only within 30 days of delivery (POL-WAR-001 S9.2)
    CASE WHEN d.days_since_delivery <= 30 THEN 1 ELSE 0 END AS care_plus_purchasable,
    CASE WHEN d.brand = 'Pacify'
         THEN 'Pacify handles end to end (POL-WAR-001 S8.2)'
         ELSE 'Manufacturer service network; Pacify facilitates only (POL-WAR-001 S8.3)'
    END AS routing_note
FROM v_order_detail d;


-- ---------------------------------------------------------------------
-- v_refund_quote — the fee waterfall from POL-REF-001 S4.1.
--
--   refund = price_paid
--          - restocking_fee   (10% if opened & change-of-mind & not accessory & not EU)
--          - return_shipping  (450 if change-of-mind)
--          + original_shipping (only if defective / wrong item / DOA)
--
-- Computed here rather than by the LLM. Arithmetic belongs in the data layer.
-- ---------------------------------------------------------------------
CREATE VIEW v_refund_quote AS
WITH base AS (
    SELECT
        e.order_id, e.eligibility, e.remedy_path, e.region,
        d.unit_price, d.quantity, d.shipping_charge, d.is_opened,
        d.category, d.restocking_pct, d.payment_method, d.is_no_cost_emi,
        d.unit_price * d.quantity AS price_paid
    FROM v_return_eligibility e
    JOIN v_order_detail d USING (order_id)
),
fees AS (
    SELECT
        *,
        -- restocking: waived for EU (POL-EU-001 S4.1), accessories, and sealed items
        CASE
            WHEN region = 'EU'          THEN 0
            WHEN category = 'accessory' THEN 0
            WHEN is_opened = 0          THEN 0
            ELSE ROUND(price_paid * restocking_pct, 2)
        END AS restocking_fee,
        -- return shipping: 450 change-of-mind, EU pays actual cost, 0 if defective
        CASE WHEN region = 'EU' THEN 0 ELSE 450 END AS return_shipping_change_of_mind
    FROM base
)
SELECT
    order_id, eligibility, remedy_path, region, payment_method,
    is_no_cost_emi, price_paid, restocking_fee,
    return_shipping_change_of_mind AS return_shipping,
    shipping_charge AS original_shipping,
    -- scenario A: customer changed their mind
    ROUND(price_paid - restocking_fee - return_shipping_change_of_mind, 2)
        AS refund_change_of_mind,
    -- scenario B: item defective / DOA / wrong item — no fees, shipping refunded
    ROUND(price_paid + shipping_charge, 2)
        AS refund_defective,
    -- scenario C: store credit, +5% bonus (POL-REF-001 S6.2)
    ROUND((price_paid - restocking_fee - return_shipping_change_of_mind) * 1.05, 2)
        AS store_credit_change_of_mind,
    CASE payment_method
        WHEN 'upi'          THEN '3-5 business days'
        WHEN 'credit_card'  THEN '5-7 business days'
        WHEN 'debit_card'   THEN '5-7 business days'
        WHEN 'net_banking'  THEN '5-7 business days'
        WHEN 'emi'          THEN '7-14 business days (principal only)'
        WHEN 'cod'          THEN '7-10 business days (bank details required)'
    END AS refund_timeline,
    CASE WHEN is_no_cost_emi = 1
         THEN 'Refund is of the discounted amount actually paid, not list price (POL-REF-001 S7.3)'
         WHEN payment_method = 'emi'
         THEN 'Principal only; bank interest not refunded (POL-REF-001 S7.2)'
         ELSE NULL
    END AS caveat
FROM fees;


-- ---------------------------------------------------------------------
-- v_customer_contact_history — repeat-contact detection.
-- Three or more contacts on one matter is an escalation trigger
-- (POL-CS-001 S3.4(c)). Uses a window function over ticket history.
-- ---------------------------------------------------------------------
CREATE VIEW v_customer_contact_history AS
SELECT
    t.customer_id,
    t.ticket_id,
    t.order_id,
    t.created_at,
    t.intent,
    t.sentiment,
    COUNT(*)      OVER (PARTITION BY t.customer_id, t.intent)               AS contacts_on_intent,
    ROW_NUMBER()  OVER (PARTITION BY t.customer_id, t.intent
                        ORDER BY t.created_at)                              AS contact_seq,
    LAG(t.created_at) OVER (PARTITION BY t.customer_id
                            ORDER BY t.created_at)                          AS prev_contact_at,
    CASE WHEN COUNT(*) OVER (PARTITION BY t.customer_id, t.intent) >= 3
         THEN 1 ELSE 0 END                                                  AS escalation_trigger
FROM tickets t;
