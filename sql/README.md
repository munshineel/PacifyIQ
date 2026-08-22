# PacifyIQ — SQL layer

Business logic and analytics live in SQL rather than Python. Not for the sake of
using SQL, but because both are genuinely better expressed declaratively.

## Files

| File | What it does |
|---|---|
| `01_business_logic_views.sql` | 5 views encoding return/warranty eligibility and the refund waterfall |
| `02_analytics.sql` | 12 dashboard queries — aggregation, window functions, trend detection |

Apply with:
```python
con.executescript(open("sql/01_business_logic_views.sql").read())
```

## Why the eligibility logic is SQL

Return eligibility is a deterministic function of stored facts: delivery date,
opened flag, product category, quantity, customer region. It is not a judgement
call and it is not language work.

Three consequences:

1. **One definition.** The tool layer, the eval harness, and the dashboard all
   read `v_return_eligibility`. They cannot drift apart.
2. **The LLM never computes it.** `calculate_refund()` returns a number the
   model explains rather than derives. LLM arithmetic is unreliable, and a wrong
   refund figure stated fluently is the worst failure mode in the product.
3. **Testable in isolation.** The views are verified against the 27
   deterministic edge-case orders without any model in the loop.

## Views

**`v_order_detail`** — denormalised join across orders, customers, products.
Adds `days_since_delivery` via `julianday()` arithmetic and `warranty_route`
from brand. Keeps every tool query to a single statement.

**`v_return_eligibility`** — the window decision, with precedence:

```
EU customer      -> 14 days (statutory, overrides everything)
bulk (qty >= 5)  -> 7 days
refurbished      -> 14 days
accessory        -> 30 days
opened           -> 14 days
sealed           -> 30 days
```

Returns `eligibility`, `days_remaining`, `window_basis` (the citation), and
`remedy_path` — which encodes the DOA / grey-zone / warranty-only bands from
DEFECT-12.

**`v_warranty_status`** — coverage period, months elapsed, and the
Pacify-vs-manufacturer routing split (DEFECT-09).

**`v_refund_quote`** — the POL-REF-001 S4.1 waterfall as a CTE chain. Emits
three scenarios (change of mind, defective, store credit), the payment-method
timeline, and a `caveat` field carrying the EMI qualifications.

**`v_customer_contact_history`** — repeat-contact detection via
`COUNT() OVER (PARTITION BY customer_id, intent)`. Three or more contacts on one
matter is an escalation trigger under POL-CS-001 S3.4(c).

## Verified behaviour

| Order | Expected | View output |
|---|---|---|
| PAC-2026-12345 | day 12 of 14, eligible | eligible, 2 days remaining |
| PAC-2026-12346 | day 14 of 14, last day | eligible, 0 days remaining |
| PAC-2026-12347 | day 15, expired | expired |
| PAC-2026-12348 | sealed, day 22 of 30 | eligible, 8 days remaining |
| PAC-2026-12354 | EU, opened, day 10 | eligible on 14-day EU window, **no restocking fee** |
| PAC-2026-12357 | Northwind, 8 months | in_warranty, **manufacturer_administered** |
| PAC-2026-12367 | bulk 6 units, day 5 of 7 | eligible |
| PAC-2026-12368 | bulk 8 units, day 9 of 7 | expired |

Note the EU case: same product, same day count, same opened flag as an Indian
order, but zero restocking fee and zero return shipping. The jurisdictional
override is enforced in the view, not left to the prompt.

Note also that refund figures use **price actually paid**, not list price, so an
order carrying a promotional discount quotes lower than the canonical worked
example. That is POL-REF-001 S4.2 behaving correctly.

## Analytics

`02_analytics.sql` is what the dashboard runs. Highlights:

**Q3 — emerging issue detection.** Compares a trailing 7-day rate against a
prior 28-day baseline per intent+subtopic, normalised per day, with a
volume floor to suppress noise. Validate against `tickets/PLANTED_TRENDS.md`:
T1, T2 and T3 must surface; T4 is seasonal and should not.

**Q4 — daily volume with a 7-day moving average** using
`AVG() OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`, plus
week-over-week deltas via `LAG(n, 7)`. Separates real movement from weekday
seasonality.

**Q7 — confidence calibration.** Buckets confidence into deciles and compares
thumbs-down rate per bucket. If the curve is flat, the confidence signal is
uninformative — which is itself a reportable finding and feeds Phase 12.

**Q10 — refund exposure by payment method**, ranked with `RANK() OVER`. The kind
of query a finance team would actually ask for.

## Techniques used

CTEs (including chained), window functions (`COUNT`/`AVG`/`RANK`/`ROW_NUMBER`/
`LAG` with `PARTITION BY` and explicit frames), multi-table joins, `LEFT JOIN`
with `NULLIF` guards, date arithmetic (`julianday`, `STRFTIME`, `DATE(..., '-N days')`),
conditional aggregation, views as a business-logic layer, and indexing on the
columns the dashboard filters by.
