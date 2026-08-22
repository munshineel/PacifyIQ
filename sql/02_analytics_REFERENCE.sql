-- =====================================================================
-- PacifyIQ — Analytics queries for the admin dashboard
-- =====================================================================
-- Every dashboard number is produced by SQL against ticket history and
-- trace logs, not computed in pandas. Window functions do the trend work.
--
-- Attach the ticket history first:
--   .mode csv
--   .import tickets/ticket_history.csv tickets_raw
-- =====================================================================


-- =====================================================================
-- Q1. Overview tile — headline operational metrics
-- =====================================================================
SELECT
    COUNT(*)                                                   AS total_tickets,
    SUM(resolved_by = 'ai')                                    AS ai_resolved,
    ROUND(100.0 * SUM(resolved_by = 'ai') / COUNT(*), 1)       AS deflection_rate_pct,
    ROUND(100.0 * SUM(resolved_by = 'human') / COUNT(*), 1)    AS escalation_rate_pct,
    ROUND(AVG(latency_seconds), 2)                             AS avg_latency_s,
    ROUND(AVG(confidence), 3)                                  AS avg_confidence,
    SUM(tokens_used)                                           AS total_tokens,
    ROUND(SUM(tokens_used) / 1e6 * 0.30, 2)                    AS est_cost_usd,
    ROUND(100.0 * SUM(feedback = 'down')
          / NULLIF(SUM(feedback <> ''), 0), 1)                 AS thumbs_down_pct
FROM tickets_raw;


-- =====================================================================
-- Q2. Intent distribution with deflection and sentiment per intent
-- =====================================================================
SELECT
    intent,
    COUNT(*)                                                   AS tickets,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)         AS share_pct,
    ROUND(100.0 * SUM(resolved_by = 'ai') / COUNT(*), 1)       AS deflection_pct,
    ROUND(100.0 * SUM(sentiment = 'negative') / COUNT(*), 1)   AS negative_pct,
    ROUND(AVG(confidence), 3)                                  AS avg_confidence,
    ROUND(AVG(latency_seconds), 2)                             AS avg_latency_s
FROM tickets_raw
GROUP BY intent
ORDER BY tickets DESC;


-- =====================================================================
-- Q3. EMERGING ISSUE DETECTION
-- Compares the trailing 7-day rate against the prior 28-day baseline
-- per intent+subtopic. Validate against tickets/PLANTED_TRENDS.md —
-- T1, T2 and T3 must surface; T4 is seasonal and should be filtered.
-- =====================================================================
WITH bounds AS (
    SELECT MAX(DATE(created_at)) AS max_d FROM tickets_raw
),
windows AS (
    SELECT
        t.intent,
        t.subtopic,
        CASE
            WHEN DATE(t.created_at) >  DATE(b.max_d, '-7 days')  THEN 'recent'
            WHEN DATE(t.created_at) >  DATE(b.max_d, '-35 days') THEN 'baseline'
            ELSE 'older'
        END AS bucket
    FROM tickets_raw t CROSS JOIN bounds b
),
counts AS (
    SELECT
        intent, subtopic,
        SUM(bucket = 'recent')   AS recent_n,
        SUM(bucket = 'baseline') AS baseline_n
    FROM windows
    WHERE bucket <> 'older'
    GROUP BY intent, subtopic
),
rates AS (
    SELECT
        intent, subtopic, recent_n, baseline_n,
        -- normalise: recent window is 7 days, baseline is 28
        1.0 * recent_n   / 7.0  AS recent_per_day,
        1.0 * baseline_n / 28.0 AS baseline_per_day
    FROM counts
)
SELECT
    intent,
    subtopic,
    recent_n,
    ROUND(recent_per_day, 2)    AS recent_per_day,
    ROUND(baseline_per_day, 2)  AS baseline_per_day,
    ROUND(recent_per_day / NULLIF(baseline_per_day, 0), 2) AS lift,
    CASE
        WHEN baseline_per_day = 0 AND recent_per_day > 1 THEN 'NEW ISSUE'
        WHEN recent_per_day / NULLIF(baseline_per_day, 0) >= 2.5 THEN 'SPIKE'
        WHEN recent_per_day / NULLIF(baseline_per_day, 0) >= 1.5 THEN 'ELEVATED'
        ELSE 'normal'
    END AS signal
FROM rates
WHERE recent_n >= 5                       -- suppress low-volume noise
ORDER BY lift DESC
LIMIT 15;


-- =====================================================================
-- Q4. Daily volume with 7-day moving average
-- Window function over an ordered frame — the standard way to separate
-- signal from weekday seasonality.
-- =====================================================================
WITH daily AS (
    SELECT DATE(created_at) AS d, COUNT(*) AS n
    FROM tickets_raw
    GROUP BY 1
)
SELECT
    d,
    n,
    ROUND(AVG(n) OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1)
        AS ma_7d,
    n - LAG(n, 7) OVER (ORDER BY d) AS wow_change,
    ROUND(100.0 * (n - LAG(n, 7) OVER (ORDER BY d))
          / NULLIF(LAG(n, 7) OVER (ORDER BY d), 0), 1) AS wow_pct
FROM daily
ORDER BY d DESC
LIMIT 30;


-- =====================================================================
-- Q5. Sentiment trend by week, with running share of negative
-- =====================================================================
SELECT
    STRFTIME('%Y-W%W', created_at)                              AS week,
    COUNT(*)                                                    AS tickets,
    ROUND(100.0 * SUM(sentiment = 'positive') / COUNT(*), 1)    AS positive_pct,
    ROUND(100.0 * SUM(sentiment = 'neutral')  / COUNT(*), 1)    AS neutral_pct,
    ROUND(100.0 * SUM(sentiment = 'negative') / COUNT(*), 1)    AS negative_pct,
    ROUND(AVG(100.0 * SUM(sentiment = 'negative') / COUNT(*))
          OVER (ORDER BY STRFTIME('%Y-W%W', created_at)
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 1)   AS negative_ma4w
FROM tickets_raw
GROUP BY week
ORDER BY week;


-- =====================================================================
-- Q6. Escalation analysis — what drives handoffs
-- =====================================================================
SELECT
    intent,
    sentiment,
    COUNT(*)                                                 AS tickets,
    SUM(resolved_by = 'human')                               AS escalated,
    ROUND(100.0 * SUM(resolved_by = 'human') / COUNT(*), 1)  AS escalation_pct,
    ROUND(AVG(CASE WHEN resolved_by = 'human' THEN confidence END), 3)
        AS avg_conf_escalated,
    ROUND(AVG(CASE WHEN resolved_by = 'ai'    THEN confidence END), 3)
        AS avg_conf_resolved
FROM tickets_raw
GROUP BY intent, sentiment
HAVING tickets >= 30
ORDER BY escalation_pct DESC
LIMIT 20;


-- =====================================================================
-- Q7. Confidence calibration buckets
-- Does reported confidence actually predict a good outcome? Compare
-- confidence deciles against thumbs-down rate. If the curve is flat,
-- the confidence signal is uninformative — which is itself a finding.
-- =====================================================================
WITH bucketed AS (
    SELECT
        CAST(confidence * 10 AS INTEGER) AS decile,
        feedback,
        resolved_by
    FROM tickets_raw
    WHERE feedback <> ''
)
SELECT
    decile / 10.0                                            AS confidence_floor,
    COUNT(*)                                                 AS n,
    SUM(feedback = 'up')                                     AS thumbs_up,
    SUM(feedback = 'down')                                   AS thumbs_down,
    ROUND(100.0 * SUM(feedback = 'down') / COUNT(*), 1)      AS down_rate_pct
FROM bucketed
GROUP BY decile
ORDER BY decile;


-- =====================================================================
-- Q8. Channel and region breakdown
-- =====================================================================
SELECT
    region,
    channel,
    COUNT(*)                                                 AS tickets,
    ROUND(100.0 * SUM(resolved_by = 'ai') / COUNT(*), 1)     AS deflection_pct,
    ROUND(AVG(latency_seconds), 2)                           AS avg_latency_s
FROM tickets_raw
GROUP BY region, channel
ORDER BY tickets DESC;


-- =====================================================================
-- Q9. OPERATIONAL — orders approaching the end of their return window
-- Joins the eligibility view; the kind of query a real support desk runs.
-- =====================================================================
SELECT
    order_id, product_name, region, quantity, is_opened,
    delivery_date, days_since_delivery, window_days, days_remaining,
    window_basis
FROM v_return_eligibility
WHERE eligibility = 'eligible'
  AND days_remaining BETWEEN 0 AND 3
ORDER BY days_remaining, order_id
LIMIT 25;


-- =====================================================================
-- Q10. OPERATIONAL — refund exposure by payment method
-- Aggregation over the refund quote view, ranked with a window function.
-- =====================================================================
SELECT
    payment_method,
    COUNT(*)                                                    AS eligible_orders,
    ROUND(SUM(refund_change_of_mind), 0)                        AS exposure_if_all_returned,
    ROUND(AVG(refund_change_of_mind), 0)                        AS avg_refund,
    ROUND(SUM(restocking_fee), 0)                               AS restocking_recovered,
    RANK() OVER (ORDER BY SUM(refund_change_of_mind) DESC)      AS exposure_rank
FROM v_refund_quote
WHERE eligibility = 'eligible'
GROUP BY payment_method
ORDER BY exposure_rank;


-- =====================================================================
-- Q11. Warranty claim funnel by brand route
-- =====================================================================
SELECT
    w.warranty_route,
    w.brand,
    COUNT(DISTINCT w.order_id)                                  AS orders,
    SUM(w.warranty_state = 'in_warranty')                       AS in_warranty,
    COUNT(wc.claim_id)                                          AS claims_raised,
    ROUND(100.0 * COUNT(wc.claim_id)
          / NULLIF(SUM(w.warranty_state = 'in_warranty'), 0), 2) AS claim_rate_pct,
    SUM(wc.outcome = 'not_covered')                             AS rejected
FROM v_warranty_status w
LEFT JOIN warranty_claims wc ON wc.order_id = w.order_id
GROUP BY w.warranty_route, w.brand
ORDER BY orders DESC;


-- =====================================================================
-- Q12. Repeat-contact customers — escalation candidates
-- =====================================================================
SELECT
    customer_id,
    intent,
    contacts_on_intent,
    MIN(created_at) AS first_contact,
    MAX(created_at) AS latest_contact
FROM v_customer_contact_history
WHERE escalation_trigger = 1
GROUP BY customer_id, intent
ORDER BY contacts_on_intent DESC
LIMIT 20;
