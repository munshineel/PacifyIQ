# Planted trends in ticket_history.csv

*** ticket_history.csv is SIMULATED data. Label it as such in the UI and README. ***

The emerging-issue detector on the admin dashboard must find these. If it does
not, the detector is wrong. If it finds trends not listed here, they are noise.

| ID | Trend | Window | Magnitude | Dominant subtopic |
|---|---|---|---|---|
| T1 | account_management spike | last 7 days | 4.5x baseline | login failure (78%) |
| T2 | technical_support ramp | last 21 days, rising | up to 2.4x | display issue (55%) |
| T3 | payment_issue incident | 38-41 days ago, 3-day burst | 6x baseline | payment failed amount debited (80%) |
| T4 | shipping/tracking/complaint rise | 60-95 days ago | 1.5-2.0x | festival season |
| T5 | return_policy_question elevated | last 90 days | 1.35x | policy v2 took effect |

T1 is the headline case: a sharp, recent, narrow spike. T2 tests detection of a
gradual ramp rather than a step change. T3 tests detection of a historical
incident that has already resolved. T4 tests seasonality that should NOT be
flagged as an emerging issue. T5 is a slow drift that a naive week-over-week
detector will miss entirely.

A good detector distinguishes T1/T2/T3 (real) from T4 (seasonal) and catches
T5 only with a longer baseline window.
