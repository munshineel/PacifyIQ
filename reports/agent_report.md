# PacifyIQ — Agentic Support System

**Phase 10.** Explicit tools, explicit decision logic, explicit state. Runs
offline.

```bash
python scripts/evaluate_agent.py
```

---

## 1. Results

| Metric | Value |
|---|---|
| Scenarios passed | **10 / 10** |
| Escalation decision accuracy | **0.967** |
| Argument extraction accuracy | **1.000** |
| Tool selection accuracy | **0.883** |
| Tools called per request | **min 0 · max 5 · mean 2.0** |
| Tier-3 actions executed autonomously | **0** |

Tools available: 13. An agent that fired everything would show max 13 and mean
13. The spread from 0 to 5 is what "selects tools based on the issue" looks like
when measured.

---

## 2. What makes this an agent rather than a chatbot

**1. It selects tools from explicit rules.** A plan is built before anything
runs, from intent + extracted entities + deterministic vocabulary. `Hello` calls
nothing. `Can I return order PAC-2026-12345?` calls four.

**2. It has state that changes what happens next.** An order that turns out not
to exist stops the plan and asks for a correction. A policy check returning
`expired` removes the need for approval.

**3. It has stop conditions and cannot loop.** Five reasons to stop:
`plan_complete`, `sufficient_evidence`, `missing_precondition`,
`hard_escalation`, `max_steps`. A test asserts termination on adversarial input
including `"?"` and `"aaaaaaa"`.

**4. Tier 3 is blocked in code.** A prompt instruction not to issue refunds is a
request. A code path that cannot issue one is a guarantee.

**5. It returns decision metadata, not reasoning prose.** No chain of thought is
exposed; a test scans the output for narration leakage.

---

## 3. Tools

Thirteen tools in three tiers. **The tier is the security model, enforced at the
registry — not in the prompt.**

| Tier | Tools | Autonomy |
|---|---|---|
| **1** read-only | `get_order` · `get_customer` · `check_policy` · `check_payment` · `check_subscription` · `search_products` · `search_knowledge_base` · `analyze_screenshot` | autonomous |
| **2** creates a record | `create_support_ticket` · `escalate_to_human` | autonomous, reversible |
| **3** mutating | `approve_refund` · `cancel_order` · `modify_account` | **never autonomous** |

Tier 3 escalates at **any** confidence. An agent 99% certain a ₹64,900 refund is
warranted still does not get to issue it. Confidence gates *answers*; tier gates
*actions*. Conflating them is the mistake this design prevents.

### Three sources of truth, kept distinct

| Source | Tools | Why it matters |
|---|---|---|
| **SQL views** | order, customer, policy, refund | Same logic serves the tool, the eval harness and the dashboard — they cannot drift |
| **Corpus** | knowledge search | Phase 6 retrieval, unchanged |
| **Mock APIs** | payment gateway, subscriptions | No such system exists. Deterministic, seeded per order, and every payload carries `"_source": "mock"` |

That last label matters. A simulated fact must never be mistaken for a real one
three layers downstream, and a test enforces it.

### Refund figures are computed, never generated

`check_policy` returns the full waterfall — price paid, restocking fee, return
shipping, payable amount, timeline — from the SQL views, alongside
`"disbursement": "requires human approval"`. The model explains a number it did
not derive. A wrong refund figure stated fluently is the worst failure mode here.

---

## 4. Workflow

```
INPUT (text + optional screenshot + optional context)
  ↓
UNDERSTAND        intent · sentiment · urgency · entities        (~3 ms, local)
  ↓
HARD TRIGGERS     legal threat · identity-sensitive → escalate immediately
  ↓
PLAN              which tools, why, and what each requires
  ↓
EXECUTE           skip steps whose preconditions failed
  ↓
OBSERVE           re-plan once if a tool revealed new information
  ↓
ASSEMBLE          tool results + retrieved policy + image evidence
  ↓
GENERATE          grounded answer
  ↓
GATE              evidence · grounding · conflict · tier
  ↓
RESOLVE · NEEDS_INFORMATION · REFUSED · ESCALATED
```

The hard triggers run **before** any tool. A legal threat escalates in ≤2
actions — researching a customer's case before handing over a chargeback threat
wastes time on both sides.

---

## 5. Ten scenarios

| # | Scenario | Status | Tools | Escalated | Reason |
|---|---|---|---|---|---|
| 1 | Simple FAQ | resolved (caveat) | 1 | — | |
| 2 | Billing issue | resolved (caveat) | 3 | — | |
| 3 | Refund request | **escalated** | 5 | ✓ | `mutating_action_requires_approval` |
| 4 | Technical problem | resolved | 2 | — | |
| 5 | Screenshot technical 🖼️ | resolved | 3 | — | |
| 6 | Missing information | **needs_information** | 0 | — | asks for order reference |
| 7 | Conflicting information | resolved | 1 | — | |
| 8 | Unsupported request | **refused** | 0 | — | |
| 9 | High-risk / ambiguous | **escalated** | 2 | ✓ | `identity_verification_required` |
| 10 | Requires escalation | **escalated** | 1 | ✓ | `legal_or_chargeback_threat` |

Scenario 3 calls five tools and escalates anyway — it gathers the order, the
eligibility, the refund figure and the policy basis, then hands a human a
one-click decision. That is the difference between deflection and a phone tree.

---

## 6. Decision metadata — no chain of thought

```json
{
  "intent": "return_refund_request",
  "actions_taken": ["get_order", "check_policy", "search_knowledge_base",
                    "escalate_to_human"],
  "evidence_used": ["get_order", "check_policy", "knowledge_base(5 chunks)"],
  "confidence": 0.0,
  "resolution_status": "escalated",
  "escalation_required": true,
  "escalation_reason": "mutating_action_requires_approval",
  "trajectory": [
    {"tool": "get_order", "args": {"order_id": "PAC-2026-12345"}, "status": "ok"},
    {"tool": "check_policy", "args": {"order_id": "PAC-2026-12345",
                                      "policy": "return"}, "status": "ok"}
  ],
  "tools_considered": [...], "tools_skipped": [...],
  "stop_reason": "plan_complete", "steps": 4, "replans": 0
}
```

`trajectory` was added because `actions_taken` alone cannot answer *"did it
extract the order ID correctly?"* — which is the most common tool-calling
failure in practice, and which the evaluation was silently unable to measure.

---

## 7. Eight bugs found and fixed

Every one was found by measurement, not inspection.

### 7.1 Double-gating — the worst of them

The RAG abstention gate thresholds on **retrieval** score. That is the right
signal when policy documentation is the only evidence, and the wrong signal once
a tool has answered: no policy document discusses one specific parcel.

*"Where is my order PAC-2026-12345?"* escalated **after `get_order` succeeded**.

Fixed by making the agent override the RAG gate when it holds independent tool
evidence, and defer to it otherwise. A grounding failure is never overridden — a
fabricated figure is the one thing tool evidence cannot excuse.

### 7.2 `result_for` hid failures

It returned only successful calls, making a not-found order indistinguishable
from never having called the tool. A failure **is** information.

### 7.3 Unknown orders escalated

A reference matching nothing is almost always a typo. Now `needs_information`
with a format hint — what a person would do first.

### 7.4 Eligibility questions treated as action requests

*"Can I return order X?"* asks whether it is possible. *"Return X for me"* asks
for the action. Only the second is mutating. Detected grammatically —
interrogative opener versus stated desire — because the distinction is
grammatical, not semantic.

### 7.5 Tier-3 escalation when the action was impossible

If policy says the window closed, there is nothing to approve. Escalating an
ineligible refund queues a human to deliver a "no" the agent could already give,
with the citation attached.

### 7.6 Entity did not override intent in planning

`return_policy_question` is defined as *"no specific order in play"* and plans no
order lookup. But *"Is PAC-2026-12354 returnable?"* names an order, and answering
from generic policy is wrong twice over — the window may have closed, and the
region may override the base rule entirely.

This is the Phase 6.5 principle applied to planning: **an extracted entity is
deterministic and outranks a probabilistic intent label.**

### 7.7 False regional conflicts

*"Can I return PAC-2026-12368?"* escalated on `conflicting_documentation` because
retrieval surfaced the EU addendum alongside base policy — for an Indian order.

Phase 7's conflict detector is region-agnostic because it has no order context.
The agent **does**, so it can resolve what retrieval alone cannot. Where the
region is unknown the conflict stands, because guessing which jurisdiction
applies is exactly the wrong call.

**The genuine contradiction still escalates.** *"Is there a 30 day satisfaction
guarantee?"* (DEFECT-01, two current documents disagreeing) → escalated.
Regression-tested in both directions.

### 7.8 Compound messages lost their tracking intent

*"Where is my order and can I return it?"* was classified `return_policy_question`
at margin 0.11 and never planned an order lookup. Added the converse of the
existing symptom override: tracking vocabulary corrects a low-confidence intent
toward `order_tracking`.

---

## 8. Measured movement

| Metric | Before | After |
|---|---|---|
| Escalation decision accuracy | 0.500 | **0.967** |
| Argument extraction accuracy | 0.000 | **1.000** |
| Tool selection accuracy | 0.400 | **0.883** |

**Argument extraction was a measurement bug, not a behaviour bug.** The metric
searched a field that did not exist. `Order 12345` → `PAC-2026-12345` was always
correct; nothing could see it.

**Tool selection 0.400 → 0.883 is partly a naming mismatch.** The trajectory eval
set was authored in Phase 0, before implementation, and names capabilities
(`get_order_status`, `check_return_eligibility`) rather than the shipped tools
(`get_order`, `check_policy`). The evaluator now carries an explicit alias map —
stated in the code, not hidden. The remaining gain came from bugs 7.6 and 7.8.

---

## 9. ⚠️ One threshold deliberately not tuned

*"What is your return policy?"* still escalates. BM25 scores it **5.22**, below
the 7.0 gate, because *return* and *policy* appear in nearly every document —
low IDF on the most obvious query in the corpus.

Tempting to drop the threshold to 5.0. I swept it against all 160 evaluation
questions instead:

| Threshold | Answerable kept | Unanswerable blocked | Balanced |
|---|---|---|---|
| 5.0 | 0.958 | 0.175 | 0.168 |
| 6.0 | 0.933 | 0.400 | 0.373 |
| **7.0** | 0.842 | 0.700 | **0.589** |
| 7.5 | 0.817 | 0.750 | 0.613 |

Lowering it rescues a handful of queries at a large cost in refusals. **Kept
7.0.** This is Phase 7's documented 13.3% false-abstention rate surfacing at the
agent layer, not a new defect — and fixing it needs a better evidence signal,
not a looser one.

---

## 10. Honest limitations

1. **The planner is rule-based, not learned.** `INTENT_PLAN` encodes which tools
   each intent needs. That is deliberate — Phase 6.5 measured that trusting the
   classifier unconditionally is *worse* than ignoring it — but it does not
   generalise to intents nobody anticipated.
2. **Conflict detection is structural, not semantic.** It fires on version and
   region diversity among retrieved chunks, not on actual contradiction. It
   catches DEFECT-01 partly by luck. A genuine same-version contradiction with no
   metadata difference would pass.
3. **Two tools are simulated.** No payment gateway or subscription service
   exists. Labelled, deterministic, and clearly bounded — but simulated.
4. **Generation is the local extractive backend.** Every answer is assembled from
   retrieved sentences. The `groq` path is wired and untested.
5. **`replans` is almost always 0.** The re-plan loop exists and is exercised by
   tool failures, but the ten scenarios rarely trigger it — so it is less tested
   than the rest.
6. **Single-turn.** No conversation memory; each request is independent.

---

## 11. Artifacts

| Path | Contents |
|---|---|
| `src/agent/tools.py` | 13 tools, 3 tiers, registry, tier enforcement |
| `src/agent/planner.py` | Rule-based planning, intent correction, preconditions |
| `src/agent/loop.py` | State, execution, gates, decision metadata |
| `scripts/evaluate_agent.py` | 10 scenarios · tier enforcement · 30-case trajectory eval |
| `tests/test_agent.py` | 39 tests, all offline |
| `reports/results/agent_scenarios.csv` | Per-scenario outcomes |
| `reports/results/agent_trajectories.csv` | Per-case tool selection and escalation |

**39 Phase 10 tests. 291 across the project.**

---

## Established

```
INPUT → INTENT → PLAN → EXECUTE → OBSERVE → ASSEMBLE
      → GENERATE → GATE → RESOLVE | ESCALATE
```

Explicit tools, explicit state, explicit stop conditions, and a security model
enforced in code. **Escalation accuracy 0.967, argument extraction 1.000, mean
2.0 tools per request from a pool of 13.**
