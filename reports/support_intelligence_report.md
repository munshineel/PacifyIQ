# PacifyIQ — Support Intelligence

**Phase 13b.** AI and support-operations analytics over the system's own
conversations.

```bash
python scripts/simulate_support_traffic.py --days 35 --per-day 14
streamlit run app/Home.py     # → Support Intelligence
```

---

## 1. Scope — and what this deliberately is not

`src/analytics/metrics.py` (Phase 2) analyses `ticket_history.csv`: volume by
channel, refund exposure, warranty funnels, expiring return windows. That is
**business** analytics.

This layer answers a different question:

> **Is the AI doing its job, and where is it failing?**

Every metric is about the assistant's own behaviour — what it understood, what
evidence it found, which tools it reached for, when it declined, and where the
failures cluster. Nothing here reports sales, products, customer segments or
revenue.

**A test enforces the boundary.** `test_layer_does_not_duplicate_business_analytics`
scans the module's public function names for business terms and fails if any
appear. (The first version scanned the file text and failed on the docstring
explaining which metrics were excluded — a scope guard has to check the code,
not the prose about the code.)

---

## 2. The data is real behaviour on a synthetic workload

This distinction matters for reading every number below.

| | |
|---|---|
| **Messages** | Synthetic — drawn from the evaluation sets and paraphrase templates |
| **Everything measured about them** | **Real** — the intent came from the classifier, the retrieval scores from the index, the escalation reason from the gate that actually fired |

`scripts/simulate_support_traffic.py` runs real requests through the real agent
and records genuine outputs. It is not a fabricated analytics table.

**A trend is planted deliberately** — login and payment issues rise over the
final week — so the emerging-issue detector can be validated against a known
answer. Without a planted signal, *"no issues detected"* is indistinguishable
from a broken detector.

---

## 3. What the dashboard found

377 conversations over 35 days:

```
Login / account access issues appeared this week (14 conversations) with no
prior history. 100% required a human.

Payment failure issues are running 3.1x their usual rate this week
(14 conversations vs 4 expected).

Return policy question accounts for the largest share of escalations
(22%, 29 of 132).

The knowledge base returned nothing usable for 99 conversations (26%), most
often on return policy question (30).
```

The first two are the planted surge, found. The last two were not planted — they
are genuine findings about how the system behaves.

---

## 4. Headline metrics

| | |
|---|---|
| Conversations | 377 |
| Resolved without a human | **56.2%** |
| Escalated | 35.0% |
| Asked for clarification | 5.6% |
| Refused | 3.2% |
| Average confidence | 0.46 |
| **Retrieval returned nothing usable** | **26.3%** |
| Conversations with a screenshot | 8.8% |
| Average tools per conversation | **1.9** of 13 available |
| Median response | 24 ms |

---

## 5. Three findings worth acting on

### 5.1 Retrieval quality is the largest driver of human workload

| | Escalation rate |
|---|---|
| When retrieval **fails** | **82.8%** |
| When retrieval **succeeds** | **18.0%** |

Median BM25 is 11.96 when retrieval works and 5.22 when it does not — a clean
separation, which is why the 7.0 threshold holds.

**This reframes the escalation rate.** 35% looks like a policy-driven number
until you see that retrieval failure quadruples it. The fix is documentation,
not model capability.

### 5.2 Most escalations are capability gaps, not policy requirements

| Reason | Count | Share | Category |
|---|---|---|---|
| `no_supporting_documentation` | 76 | **57.6%** | **capability gap** |
| `identity_verification_required` | 22 | 16.7% | by design |
| `mutating_action_requires_approval` | 22 | 16.7% | by design |
| `relationship_issue_requires_human` | 12 | 9.1% | by design |

Splitting escalations into **by design** vs **capability gap** is the single
most useful thing this dashboard does. A refund escalation is the system working
correctly. A *"no documentation found"* escalation is a gap someone can close.

**42% by design, 58% gap.** Reporting one escalation number would have hidden
that entirely.

### 5.3 Screenshots resolve 31 points more often

| | Resolution rate |
|---|---|
| With a readable screenshot | **84.8%** |
| Without | 53.5% |

Screenshot contribution was **100%** — every image analysed yielded a usable
error code. The image contains something the customer's own words do not, which
is exactly the Phase 8 ablation showing up in operational data.

---

## 6. Volume is not workload

| Intent | % traffic | % resolved | % escalated | **% of all handovers** |
|---|---|---|---|---|
| return_policy_question | 18.8 | 59.2 | 40.8 | **22.0** |
| payment_issue | 17.8 | 62.7 | 37.3 | 18.9 |
| warranty_claim | 13.8 | **100.0** | 0.0 | **0.0** |
| return_refund_request | 11.7 | 50.0 | 50.0 | 16.7 |
| product_information | 8.5 | 53.1 | 46.9 | 11.4 |
| order_tracking | 5.0 | 52.6 | 0.0 | **0.0** |

`warranty_claim` is 13.8% of traffic and causes **zero** human work.
`return_refund_request` is 11.7% and escalates half the time — by design, since
refunds are Tier 3.

The last column is the operational one. Ranking by volume alone would put effort
in the wrong place.

---

## 7. Tool usage

| Tool | Calls | % of conversations | % resolved |
|---|---|---|---|
| `search_knowledge_base` | 320 | 84.9 | 65.0 |
| `escalate_to_human` | 132 | 35.0 | 0.0 |
| `check_policy` | 70 | 18.6 | 37.1 |
| `get_order` | 48 | 12.7 | 54.2 |
| `analyze_screenshot` | 33 | 8.8 | **84.8** |
| `search_products` | 8 | 2.1 | 100.0 |

**Never called:** `approve_refund`, `cancel_order`, `modify_account` (all
Tier 3 — correct, they must never fire autonomously), plus `check_payment`,
`check_subscription`, `create_support_ticket`, `get_customer`.

Mean **1.9 tools per conversation** of 13 available. An agent calling everything
would average 13. A test asserts this stays below one-third.

**A tool that never fires is either unnecessary or unreachable**, and both are
worth knowing. Four non-Tier-3 tools sitting idle suggests this workload does
not exercise them, not that they are broken — but on real traffic that
distinction would be worth chasing.

---

## 8. Sentiment and urgency

| Sentiment | Conversations | Escalation | Avg confidence |
|---|---|---|---|
| neutral | 248 (65.8%) | 34.7% | 0.478 |
| negative | 97 (25.7%) | 39.2% | 0.442 |

| Urgency | Conversations | Escalation |
|---|---|---|
| **high** | 18 (4.8%) | **66.7%** |
| medium | 79 (21.0%) | 32.9% |
| low | 248 (65.8%) | 34.7% |

High-urgency conversations escalate at nearly twice the rate of low-urgency
ones, which is the urgency signal doing its job rather than adding noise.

The 8.5% with blank sentiment are guardrail refusals — blocked at the input
stage before understanding runs. That is correct behaviour, but it means
sentiment percentages are over all conversations, not over analysed ones.

---

## 9. Emerging-issue detection

Trailing 7 days against the preceding 28, **normalised per day** so unequal
window lengths cannot create a false signal.

| Signal | Meaning |
|---|---|
| **NEW** | Absent from the baseline entirely — reported as new rather than as infinite lift |
| **SPIKE** | ≥ 2.9× the baseline rate |
| **ELEVATED** | ≥ 1.8× |

Topics are matched on **vocabulary, not predicted intent**, because intent is an
11-way label — "battery" and "overheating" both land in `technical_support` while
needing entirely different fixes.

**Headlines are sentences, not ratios.** An operator needs *"login issues
appeared this week and all of them required a human"*, not `lift=3.1`. A test
asserts every headline is at least six words and ends in a full stop.

---

## 10. Failure surfaces

Three tables, each pointing at a different kind of fix:

| Table | What it is | Action |
|---|---|---|
| **Failed retrievals** | Questions where the knowledge base returned nothing usable, sorted worst-first, downloadable as CSV | Write documentation |
| **Low-confidence answers** | Answers given with weak conviction that **did not** escalate — so nobody reviewed them | Manual review |
| **Recurring unresolved** | Near-identical unresolved questions, clustered on a normalised stem | Prioritise by frequency |

The low-confidence table is the one most likely to be overlooked in a real
deployment: these answers reached customers and no gate flagged them.

---

## 11. Bugs found while building this

**Screenshots reported 0% usage** while `analyze_screenshot` showed 33 calls.
The agent never set `has_image` on its decision — only the RAG pipeline's
multimodal entry point did. The tool firing and the image being *useful* are
different things, and the metric needed both.

**Sentiment and urgency were null on every trace.** The agent computed them
during understanding and dropped them before returning. Two of the distributions
this phase was asked for could not have been produced.

**Retrieval failure was not measurable.** Added `max_bm25`, `n_chunks` and
`retrieval_failed` to the decision and the trace schema, with `retrieval_failed`
defined specifically as *"the knowledge base was consulted and returned nothing
usable"* — not consulting it at all is a different thing, and conflating them
would have inflated the failure rate.

---

## 12. Honest limitations

1. **The workload is synthetic.** Real customers phrase things in ways no
   template generates. Volumes, topic mix and the surge are all authored.
2. **The surge is planted**, so detecting it validates the detector, not the
   system's ability to find genuinely unexpected problems.
3. **Topic matching is keyword-based.** A new problem described in unanticipated
   vocabulary falls into no topic and is invisible to §9.
4. **377 conversations over 35 days** is roughly 11 per day. Daily rates are
   noisy at that volume and no confidence intervals are reported.
5. **No cost tracking.** The local backend is free, so token cost per
   conversation is not measured. It would matter immediately with a hosted
   model.
6. **Unresolved clustering is a word-overlap stem**, not semantic. Two phrasings
   of the same problem sharing no vocabulary land in separate clusters.
7. **Feedback is simulated** — 12% of conversations, randomly assigned. Real
   thumbs-down data would be far more informative and far sparser.

---

## 13. Artifacts

| Path | Contents |
|---|---|
| `src/analytics/support_intelligence.py` | All metrics, trend detection, headline generation |
| `src/observability/traces.py` | Trace schema and persistence |
| `scripts/simulate_support_traffic.py` | Generates real agent behaviour on a synthetic workload |
| `app/pages/5_Analytics.py` | Four-tab dashboard |
| `tests/test_support_intelligence.py` | 28 tests including the scope guard |

**28 Phase 13b tests. 430 across the project.**

---

## Established

```
Real conversations → operations · AI performance · failure surfaces · trends
```

**Resolution 56.2%, escalation 35.0% — of which 58% are capability gaps rather
than policy requirements.** Retrieval failure quadruples the escalation rate,
screenshots lift resolution by 31 points, and the emerging-issue detector found
the planted surge and reported it as a sentence an operator can act on.
