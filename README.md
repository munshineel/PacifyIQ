# PacifyIQ

AI customer support intelligence and agent platform.

Phase 0 complete: typed config, database layer, analytics layer, tests.

---

## Setup (VSCode)

```bash
git clone <your-repo> && cd pacifyiq
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # add your Groq key
python scripts/setup_database.py   # builds views, imports tickets, smoke test
pytest                             # 47 tests should pass
```

In VSCode: open the folder, `Ctrl+Shift+P` -> **Python: Select Interpreter** ->
pick `.venv`. Recommended extensions are prompted automatically from
`.vscode/extensions.json`.

`PYTHONPATH` is set to the workspace root in `.vscode/settings.json`, so
`from src.db.queries import ...` resolves without any packaging step.

---

## Layout

```
pacifyiq/
├── .vscode/                    settings, launch configs, extensions
├── src/
│   ├── config/settings.py      typed config (pydantic-settings)
│   ├── db/
│   │   ├── connection.py       connection management, query helpers
│   │   └── queries.py          order/return/warranty/refund lookups
│   └── analytics/metrics.py    dashboard analytics -> DataFrames
├── sql/
│   └── 01_business_logic_views.sql   5 views, applied once at setup
├── scripts/setup_database.py   one-command setup
├── tests/                      47 tests, no LLM required
└── data/                       corpus, db, eval sets, intents, tickets
```

---

## Where the logic lives

| Concern | Location | Why |
|---|---|---|
| Return / warranty eligibility | SQL views | Deterministic function of stored facts. One definition shared by tools, evals and dashboard. |
| Refund arithmetic | SQL views | LLM arithmetic is unreliable, and a wrong refund figure stated fluently is the worst failure mode in this product. |
| Order ID normalisation | Python (`queries.py`) | Argument extraction is the top agent failure mode. `"12345"`, `"#12345"` and `"pac-2026-12345"` all resolve in code, not in the prompt. |
| Analytics, trends, rolling windows | Python (`metrics.py`) | Easier to read, parameterise and test than nested SQL. |
| Configuration | `settings.py` | One typed object. Never read `os.environ` directly. |

The SQL stays deliberately simple: `SELECT`, `JOIN`, `CASE`, one CTE chain. All
composition and post-processing is pandas.

---

## Usage

**Order lookups** — these become the agent tools in Phase 10:

```python
from src.db.queries import check_return_eligibility, calculate_refund

e = check_return_eligibility("12345")          # accepts any customer format
print(e.eligibility)                            # 'eligible'
print(e.days_remaining)                         # 2
print(e.window_basis)                           # 'POL-RET-002 S2 (opened electronics)'

q = calculate_refund("PAC-2026-12345")
print(q.explain())                              # full breakdown to hand to the LLM
```

**Analytics** — these become the dashboard pages in Phase 14:

```python
from src.analytics.metrics import overview_dict, emerging_issues

o = overview_dict()                             # deflection %, latency, cost
df = emerging_issues(recent_days=7, baseline_days=28)
```

Run any module directly to see it work:

```bash
python -m src.db.queries
python -m src.analytics.metrics
python -m src.config.settings
```

---

## Tests

47 tests, all against real data, none requiring an LLM or network:

- **Order ID normalisation** — 6 input formats
- **Return windows** — 8 boundary cases including day 14 (eligible) and day 15 (expired)
- **EU override** — same product and day count as an Indian order, but zero fees
- **Refund waterfall** — arithmetic verified against the policy formula
- **Warranty routing** — Pacify vs third-party administration
- **Emerging-issue detection** — validated against trends deliberately planted in the ticket history

```bash
pytest                          # all
pytest tests/test_queries.py -v # data layer only
pytest -k "eu_"                 # EU override cases
```

---

## Data

See `data/README.md` for the full description. All data is synthetic; Pacify
Electronics Pvt. Ltd. is fictional. Read `data/canonical_facts.md` and
`data/PLANTED_DEFECTS.md` before touching the corpus.

---

## Phases 2 & 3 — Dataset audit and EDA (complete)

```bash
python scripts/run_audit.py --save   # Phase 2: structural + quality audit
python scripts/run_eda.py            # Phase 3: figures + processed outputs
jupyter notebook notebooks/          # the three analysis notebooks
```

Full write-up: **[`reports/eda_findings.md`](reports/eda_findings.md)**

### Headline findings

| # | Finding | Consequence |
|---|---|---|
| 🔴 | `confidence` in ticket history was generated **from** `resolved_by` (0.815 vs 0.461) | Never usable as an escalation feature |
| 🔴 | Corpus is **16,208 words → ~41 chunks at 512 tokens**, not the ~2,000 assumed | Chunk at 128–256; ANN phase needs rethinking |
| 🔴 | 2 texts leak between train and test | Remove before Phase 4 |
| 🟠 | Train is 26.9% templated; test is 0%. OOV token rate 22.6% | Expect a large macro-F1 drop — the headline Phase 4 result |
| 🟠 | Top class overlap (`order_tracking` ↔ `return_refund_request`, J=0.25) is **entirely order-reference tokens** | Mask order IDs before vectorising; ablate it |
| 🟠 | Class imbalance 10×; majority baseline 18.2% | Macro-F1, balanced weights, stratified splits |
| 🟠 | Sentiment→escalation is r=0.305 overall but +23pp within `order_tracking` and +5pp within `complaint` | Sentiment as an **intent-conditional** signal, not a trigger |
| 🟠 | Saturday 52%, Sunday 23% of a weekday | Trend detector must compare whole weeks |

16 decisions are recorded in `reports/eda_findings.md`, each traced to the finding that
produced it. Phase 4 hyperparameters follow from them rather than from convention.

### Figures

`reports/figures/` — 8 figures, each answering one question:

1. Intent distribution vs escalation rate
2. Sentiment and priority composition
3. Class imbalance and train/test shift
4. Text characteristics and vocabulary coverage
5. Lexical overlap heatmap
6. Temporal patterns
7. Corpus structure
8. Confidence leakage

### Modules

```
src/eda/loaders.py      canonical loaders for every asset
src/eda/audit.py        Phase 2 — profiling, duplicates, PII, leakage
src/eda/text_stats.py   length, n-grams, vocabulary, drift, class overlap
src/eda/plots.py        8 figure functions
```

Processed outputs land in `data/processed/` for Phase 4.

---

## Next


**Phase 1** — Groq client with retries, structured-output helper, token counter,
and the trace schema every dashboard metric will read from.

---

## Troubleshooting notebooks

**`ModuleNotFoundError: No module named 'src'`**

The kernel's working directory is the workspace root rather than `notebooks/`.
Run once:

```bash
python scripts/fix_notebooks.py
```

This replaces the bootstrap cell with one that walks upward to find the project
root, so it works from either location.

**Wrong interpreter**

Add this to any cell to check:

```python
import sys; print(sys.executable)
```

It must point at `.venv`. If not, register the kernel and re-select it:

```bash
pip install ipykernel
python -m ipykernel install --user --name pacifyiq --display-name "Python (PacifyIQ)"
```

Then in VSCode: kernel picker (top right) → Select Another Kernel → Python
Environments → the one showing `.venv`.

---

## Phase 4 — NLP classification baseline (complete)

```bash
python scripts/train_intent_classifier.py     # comparison, ablations, artifact
python scripts/evaluate_sentiment_urgency.py  # rule-based scorers
```

Full write-up: **[`reports/classification_report.md`](reports/classification_report.md)**

### What the data supports

| Task | Approach | Why |
|---|---|---|
| Intent | Full supervised ML | 2,200 labelled messages |
| Sentiment | Rule-based lexicon + intent prior | **No text carries sentiment labels** |
| Urgency | Rule-based, 5 weighted signals | **No text carries urgency labels** |

`ticket_history.csv` has sentiment and priority but no message text; the intent
CSVs have text but no sentiment. Supervised text→sentiment is not possible here.

### The methodological finding

A random stratified split **tied nine models at macro-F1 0.9929** — the metric
could not discriminate, and selecting on it picked one of the weakest models on
held-out data. Cause: 26.9% of training rows are template-generated, and a random
split put 180 template skeletons on both sides.

Fixed with **group-aware splitting by template skeleton** (0 shared templates),
then **repeated group splits across 5 seeds** to break the remaining tie. The test
set is never consulted for selection.

### Results

**Selected: TF-IDF word(1,2) + char_wb(3,5) union → LinearSVC**

| | Validation | Test |
|---|---|---|
| Macro F1 | 0.9940 | **0.6107** |
| Weighted F1 | 0.9953 | 0.6248 |
| Accuracy | 0.9954 | 0.6357 |

257 KB · 2.45 ms inference · 39.3% test error rate

The 0.383 gap was **predicted from EDA before training**: 26.9%→0% template rate,
22.6% OOV tokens, 6→8 median words, 0%→42% compound messages. Character n-grams
close about a third of it.

| Sentiment (n=65) | | Urgency (n=65) | |
|---|---|---|---|
| Macro F1 | 0.808 | Macro F1 | 0.737 |
| Intent-prior gain | **+0.055** | Within one level | 0.954 |

⚠️ Sentiment/urgency annotations are LLM-authored, single-annotator. Indicative
only — see the limitation note in the report.

### Negative result

Masking order references was predicted by EDA to reduce the top class confusion.
**It did not** — neutral to −0.037 on test. Two reasons: `min_df=2` already dropped
most order IDs as hapax, and the *presence* of an order reference is itself
informative (77% of `order_tracking` vs under 20% elsewhere). Masking is retained
for format robustness, but the hypothesis is reported as refuted.

### Why this ships instead of an LLM

2.45 ms vs ~300 ms, zero cost, deterministic, 257 KB, no network. More importantly:
it is a **fixed measurement instrument** for the dashboard's intent trends, and the
only understanding signal not produced by the LLM — which matters for Phase 11,
where escalation needs genuinely independent evidence.

---

## Phases 5 & 6 — Knowledge base and retrieval (complete)

```bash
python scripts/build_index.py
python scripts/evaluate_retrieval.py --ablate
jupyter notebook notebooks/04_retrieval_evaluation.ipynb
```

Full write-up: **[`reports/knowledge_base_report.md`](reports/knowledge_base_report.md)**

### Pipeline

```
PDF → load → clean → chunk → embed → store → retrieve
```

13 documents · 47 pages · 16,208 words → **200 chunks**, each carrying
`doc / doc_ref / page / section / topic / version / region / product`.

### Retrieval results (120 queries, no LLM)

| | |
|---|---|
| **Recall@5** | **0.875** |
| Recall@10 | 0.942 |
| Coverage@5 | 0.793 |
| MRR | 0.665 |
| Queries answered | 87.5% |

| Strategy | Recall@5 | MRR |
|---|---|---|
| **rrf_w** (weighted RRF) | **0.875** | 0.665 |
| hybrid | 0.850 | 0.670 |
| dense | 0.800 | 0.585 |
| bm25 | 0.758 | 0.586 |

### Three measured findings

**Section chunking beats fixed windows at every size** — 0.800 vs 0.542 recall@5 at
512 tokens. Policy documents are written in clauses; a clause is an answer, and a
fixed window that splits one destroys it.

**FAISS is 2.5× slower than NumPy at 200 vectors** (0.127 ms vs 0.050 ms, identical
results). Exact search is sufficient at this corpus size, so no approximate index
ships. A measurement, not an assumption.

**Authority weighting, derived from failure analysis.** 14 of 24 initial failures had
the correct policy section at rank 6–10, displaced by an FAQ restatement. The corpus
itself states *"where it differs from a policy document, the policy document
governs"* — encoding that lifted recall@5 by +0.067 and MRR by +0.095.

### Planted defects handled

| Defect | Behaviour |
|---|---|
| Archived v1 outranks current v2 on similarity | metadata filtering, not embedding quality |
| EU addendum overrides base policy | region filter |
| Error codes fragment under tokenisation | BM25 half of the hybrid |
| Contradiction between two current documents | `has_conflict()` flag — resolving is Phase 7 |

### Modules

```
src/knowledge/loader.py       PDF → pages + document registry
src/knowledge/chunker.py      cleaning + section/fixed chunking
src/knowledge/embedder.py     tfidf_svd (local) | groq (API)
src/knowledge/vector_store.py NumPy brute force | FAISS
src/knowledge/bm25.py         lexical retrieval
src/knowledge/retriever.py    4 strategies + filtering + authority
src/knowledge/evaluation.py   Recall@K, Coverage@K, MRR, nDCG
```

Index is **8.3 MB**, committed rather than built at startup.

---

## Phase 7 — RAG pipeline (complete, no API required)

```bash
python scripts/evaluate_rag.py                  # local backend, no key
python scripts/evaluate_rag.py --backend groq   # with your key
```

Full write-up: **[`reports/rag_report.md`](reports/rag_report.md)**

### Built against an interface, not an SDK

| Backend | Requires |
|---|---|
| **`local`** — extractive generator, selects sentences from retrieved context | nothing |
| `groq` — hosted completion, retries, JSON mode | API key |
| `scripted` — fixed responses for failure-path tests | nothing |

The local backend is a genuine RAG baseline with one property no LLM has:
**faithfulness is 1.0 by construction**, because every word came verbatim from a
retrieved document. That makes it the floor an LLM is measured against — any
hallucination is then attributable to the model.

All 41 Phase 7 tests run offline.

### Pipeline

```
question → retrieve → assemble → ABSTENTION GATE → generate
        → parse+repair → verify grounding → answer | escalate → trace
```

The gate sits **before** generation, so an unanswerable question never reaches
the model. Verifiable in the trace: `llm_backend="none"`, `completion_tokens=0`.

### The counterintuitive finding

Measuring 120 answerable against 40 unanswerable questions:

| Signal | Answerable | Unanswerable | Separation |
|---|---|---|---|
| **BM25 max** | 13.12 | 6.21 | **2.1×** |
| Dense cosine | 0.572 | 0.507 | 1.1× |
| RRF fused | 0.0156 | 0.0164 | **inverted** |

**For abstention the lexical signal beats the semantic one, and the fused score
you rank with is the wrong thing to threshold on.** A question the corpus doesn't
cover contains rare vocabulary; BM25 notices, embeddings smooth it away.

### Results

| | |
|---|---|
| Abstention (40 unanswerable) | **0.700** |
| False abstention (60 answerable) | 0.133 |
| Balanced score | 0.607 |
| Faithfulness | 1.000 *(by construction)* |
| Citation accuracy | 1.000 |
| Correctness | 0.240 *(extractive floor)* |

### Hallucination detection — 5 types, all blocked

Fabricated numbers · fabricated citations · fabricated error codes · missing
citations · low context overlap. An ungrounded answer forces escalation and caps
confidence at 0.3.

⚠️ **Prompt comparison returned a null result** — v1/v2/v3 identical, because the
extractive backend doesn't follow instructions. The harness is built and wired;
the measurement is blocked on an LLM. Reported rather than omitted.

---

## Phase 6.5 — Understanding-retrieval bridge (complete)

```bash
python scripts/evaluate_routing.py
```

Full write-up: **[`reports/routing_report.md`](reports/routing_report.md)**

Connects the Phase 4 classifier and entity extractor to the Phase 6 retriever.

### The design constraint, found by measuring first

The obvious design — *classify the intent, filter retrieval to that topic* —
fails. On the queries retrieval finds hard, the classifier is **both wrong and
uncertain**:

| Query | Predicted | Margin | Correct? |
|---|---|---|---|
| *"Am I talking to a bot?"* | complaint | 0.103 | ✗ |
| *"What does SYS-0x0000007B mean?"* | payment_issue | 0.036 | ✗ |
| *"My laptop will not turn on"* | warranty_claim | 0.036 | ✗ |

The failure modes are correlated — both struggle with unusual phrasing — so
stacking them compounds rather than cancels. A hard filter would remove the
correct document more often than it removes noise.

Confirmed by sweep: **trusting the classifier unconditionally scores recall@5
0.808, worse than ignoring it entirely at 0.867.** Optimum is margin ≥ 0.25.

### Results

| Metric | Before | After | Δ |
|---|---|---|---|
| Recall@1 | 0.542 | 0.575 | +0.033 |
| Recall@3 | 0.742 | 0.817 | **+0.075** |
| Recall@5 | 0.875 | 0.883 | +0.008 |
| **MRR** | 0.664 | **0.706** | **+0.042** |
| **nDCG@5** | 0.726 | **0.782** | **+0.056** |
| Abstention | 0.700 | 0.700 | 0.000 |

The gain is in **ranking, not coverage** — routing moves the right document up
rather than finding documents retrieval missed.

### ⚠️ One component measurably hurts

Meta-question routing, built specifically to fix *"Am I talking to a bot?"*,
**fixed 3 queries and broke 6**. Recall@5 0.875 → 0.858. The trigger is too
loose: any query containing "escalate" or "verify" injected support-policy
chunks that displaced correct evidence.

`use_meta` defaults to `False`. Retained behind a flag with the measurement
recorded rather than quietly dropped.

### Bug found on the way

Faithfulness briefly read 0.96 — not a hallucination, but the grounding check
**scoring correct abstentions as ungrounded** because a refusal carries no
citation. An abstention asserts nothing, so there is nothing to support. The
metric was inverted exactly where it mattered most.

---

## Phase 8 — Multimodal screenshot analysis (complete, no API required)

```bash
python scripts/data_generation/gen_screenshots.py
python scripts/evaluate_vision.py
```

Full write-up: **[`reports/multimodal_report.md`](reports/multimodal_report.md)**

### Headline

| Metric | Text only | Text + vision | Δ |
|---|---|---|---|
| **Recall@5** | 0.720 | **1.000** | **+0.280** |
| **MRR** | 0.503 | **0.780** | **+0.277** |
| Max BM25 | 6.8 | 38.1 | +31.3 |

Error-code extraction **100% (25/25)**. **7 rescued, 0 broken. Zero invented
codes** across 8 unreadable images.

### Runs offline

Tesseract is available, so the vision backend is **real OCR, not a simulation** —
the ablation is a measured result. OCR also has a property hosted vision models
lack: **per-word confidence**, which makes visible/inferred/unknown measurable
rather than self-reported.

### Visible / inferred / unknown

| Level | Effect on retrieval |
|---|---|
| **VISIBLE** (84%) | enters the query |
| **INFERRED** (12%) | passed to the model, labelled; **never steers retrieval** |
| **UNKNOWN** (4%) | reported as absent |

A misread code that steers retrieval sends the customer to the wrong fix — worse
than no image at all. `ERR-DP-0x004` was OCR'd as `ERR-DP-@x004`; the repair
layer resolves it against the canonical registry but marks it INFERRED, so it
reaches the model as labelled evidence and stays out of the query.

### Fusion, not parallel analysis

The image is fused **into the query**, so understanding, routing, retrieval and
generation all see one enriched request:

```
text:      "my payment isn't working"
enriched:  "my payment isn't working PAY-402 payment checkout transaction"
```

### Why the delta is so large

All 7 rescued cases sat **below the BM25 7.0 abstention threshold** on text
alone. Six would have been correctly refused as unanswerable — the customer's
words genuinely didn't contain enough. The screenshot is what makes them
answerable.

### A measurement bug worth recording

Blur detection first used PIL `FIND_EDGES`: mild blur scored 15.2, severe 14.8 —
no separation, and separation is the only thing the metric exists for. A real
Laplacian variance gives 1296 / 3.7 / 0.5. Regression-tested.

---

## Phase 10 — Agentic support system (complete, no API required)

```bash
python scripts/evaluate_agent.py
```

Full write-up: **[`reports/agent_report.md`](reports/agent_report.md)**

### Results

| Metric | Value |
|---|---|
| Scenarios passed | **10 / 10** |
| Escalation decision accuracy | **0.967** |
| Argument extraction accuracy | **1.000** |
| Tool selection accuracy | **0.883** |
| Tools per request | **min 0 · max 5 · mean 2.0** (of 13 available) |
| Tier-3 actions executed autonomously | **0** |

An agent that fired every tool would show mean 13. The spread from 0 to 5 is
what tool *selection* looks like when measured.

### Tools — 13, in 3 tiers

| Tier | Tools | Autonomy |
|---|---|---|
| **1** read-only | `get_order` · `get_customer` · `check_policy` · `check_payment` · `check_subscription` · `search_products` · `search_knowledge_base` · `analyze_screenshot` | autonomous |
| **2** creates record | `create_support_ticket` · `escalate_to_human` | autonomous, reversible |
| **3** mutating | `approve_refund` · `cancel_order` · `modify_account` | **never autonomous** |

Tier is enforced at the registry, not in the prompt. A prompt instruction not to
issue refunds is a request; a code path that cannot issue one is a guarantee.

Mock tools (payment gateway, subscriptions) carry `"_source": "mock"` so a
simulated fact cannot be mistaken for a real one downstream.

### Eight bugs found by measurement

The worst: **double-gating.** The RAG abstention gate thresholds on *retrieval*
score, which is the wrong signal once a tool has answered — no policy document
discusses one specific parcel. `"Where is my order PAC-2026-12345?"` escalated
*after* `get_order` succeeded.

Others: failures hidden by `result_for` · unknown orders escalating instead of
asking · *"Can I return X?"* treated as *"Return X for me"* · Tier-3 escalation
when policy already said no · entity not overriding intent in planning · EU
addendum read as a contradiction on an Indian order · compound messages losing
their tracking intent.

| Metric | Before | After |
|---|---|---|
| Escalation accuracy | 0.500 | **0.967** |
| Argument extraction | 0.000 | **1.000** |
| Tool selection | 0.400 | **0.883** |

Argument extraction was a *measurement* bug — the metric searched a field that
didn't exist. `Order 12345` → `PAC-2026-12345` was always correct.

### One threshold deliberately not tuned

`"What is your return policy?"` scores BM25 5.22 and gets refused, because
*return* and *policy* appear in nearly every document. Swept 5.0–7.5 against all
160 eval questions: dropping to 5.0 rescues it but blocks only 17.5% of
unanswerable questions versus 70% at 7.0. Kept 7.0 and logged it as Phase 7's
inherited false-abstention rate.

---

## Phase 11 — Safety and guardrails (complete, no API required)

```bash
python scripts/evaluate_guardrails.py
```

Full write-up: **[`reports/safety_report.md`](reports/safety_report.md)**

> ⚠️ **This system is not production-safe.** These checks reduce risk; they do
> not eliminate it. 100% detection on 30 attacks written by the same person who
> wrote the rules measures imagination, not security. See §9 of the report.

### Results

| Metric | Value |
|---|---|
| Adversarial detection (30 attacks) | **100.0%** |
| False positives (295 benign messages) | **1.0%** |
| Balanced score | **0.990** |
| End-to-end through the agent | **9 / 9** |

Both sides reported deliberately — a rule that blocks everything scores 100% on
detection and destroys the product.

### 21 rules across 4 stages

```
INPUT ──► EVIDENCE ──► ACTION ──► OUTPUT
```

| Stage | Catches |
|---|---|
| **input** | injection · role override · fabricated authority · prompt/schema extraction · data exfiltration · SQL · indirection · hypothetical framing · false premise · identity-sensitive requests · PII · out-of-domain |
| **evidence** | no evidence · weak evidence · version conflict · regional ambiguity · low confidence · tool failures · invalid tool output |
| **action** | mutating tools, at any confidence |
| **output** | fabricated numbers · invented error codes · fabricated citations · forbidden commitments · internal leakage |

### Why rules, not prompts

`src/guardrails/` imports neither `src/agent` nor `src/rag` — guardrails must be
able to **veto** those layers, so they cannot depend on them. A test asserts the
import direction.

*"Never approve refunds"* in a prompt is a request. A code path that cannot
approve refunds is a guarantee.

### Three decisions worth defending

**Refusals are deliberately vague.** A refusal naming the rule it tripped is a
free oracle. A test asserts the customer message contains no rule names.

**Questions about a process ≠ requests to perform it.** *"How do I delete my
account?"* is answerable from policy; *"Delete my account"* needs verified
identity. Without this, the guardrail refused questions the documentation
answers — an over-broad rule is its own failure mode, and harder to spot than a
gap. This dropped false positives from 3.0% to 1.0%.

**Forbidden claims are the last line for the tier model.** Tier 3 stops the
assistant *performing* a refund. It doesn't stop it *writing* "your refund has
been approved" — and a customer will act on that.

### Image-borne injection

Extracted image text is screened with the same rules as typed input. An
instruction rendered into a PNG is still an instruction once OCR reads it.

---

## Phase 12 — Full evaluation

```bash
python scripts/run_full_evaluation.py
```

Full write-up: **[`reports/evaluation_report.md`](reports/evaluation_report.md)**

### How well does PacifyIQ actually work?

**Right decision on 81% of requests; right decision *and* right fact on 75%** —
across a curated set where **58% of cases should not be answered at all**.

| # | Component | Headline metric | Value | n | Scoring |
|---|---|---|---|---|---|
| 1 | Intent classification | macro-F1 | 0.611 | 140 | deterministic |
| 2 | Sentiment / urgency | macro-F1 | 0.808 | 65 | curated |
| 3 | Retrieval | Recall@5 | 0.883 | 120 | deterministic |
| 4 | RAG answer quality | faithfulness | 1.000 ⚠️ | 25 | deterministic |
| 5 | Screenshot understanding | extraction accuracy | 1.000 ⚠️ | 25 | deterministic |
| 6/7 | Agent tools | selection recall | 0.917 | 30 | curated |
| 8 | Groundedness | detection rate | 1.000 | 11 | curated |
| 9 | Escalation decisions | accuracy | 0.933 | 30 | curated |
| **10** | **End-to-end resolution** | **outcome accuracy** | **0.807** | 57 | curated |

⚠️ **Two 1.000s are not achievements.** Faithfulness is 1.0 *by construction* —
the extractive backend copies sentences verbatim and cannot fabricate.
Screenshot extraction is 1.0 on synthetic renders. Neither transfers.

**19/19 targets met** across components 1–9. Both metrics below target are on
end-to-end.

### The number worth leading with

**`unsafe_resolutions = 5`** — five times in 57, the system answered confidently
when it should have refused. That's the failure that reaches a customer, and the
report names all five rather than averaging them away:

- **3 under-specification failures** — deliberately ambiguous corpus cases
  (DEFECT-04, -08, -12) that the system resolves instead of surfacing
- **2 "topic covered, fact absent"** — the manual exists and matches every term
  except the one that matters

Neither is a tunable threshold. Both need capabilities the system doesn't have.

### By category

100% on **injection · escalation · conflict · mutating actions · out-of-domain ·
eligibility**. Weakest on **ambiguous (0.500)** and **meta questions (0.000)** —
the honest shape of the problem.

The meta cluster is a known limitation: *"Am I talking to a bot?"* scores BM25
**0.00** against the policy that answers it. Phase 6.5 built a fix and measured
that it **fixed 3 queries and broke 6**, so it ships disabled.

### Scoring — no LLM judge

Deterministic where the answer is a fact; curated where it's a decision. **No
headline metric depends on an LLM judge**, and a test asserts it. Every curated
case turns on a specific number, which string matching checks exactly and without
the circularity of grading a language model with a language model.

### Curated dataset

`data/eval/end_to_end_eval.json` — 50 text + 7 image cases across 18 categories
including ambiguous questions, missing information, misleading and irrelevant
screenshots, hallucination traps, prompt injection, conflicting documents and
out-of-domain requests.

---

## Phase 13 — Streamlit interface

```bash
pip install streamlit
streamlit run app/Home.py
```

Seven pages: **Home · Support Agent · Screenshot Analysis · Knowledge Base ·
Conversation History · Analytics · Evaluation**

### Architecture — enforced, not just claimed

```
app/pages/*  →  src/ui/service.py  →  src/agent, src/rag, src/knowledge, …
             →  src/ui/components.py   (presentation only)
```

`src/` never imports `streamlit`. Pages never import the agent, retriever,
guardrails or evaluation framework directly. **`tests/test_architecture.py`
asserts both directions by parsing the import graph** — 19 tests that fail the
moment a boundary is crossed, including that guardrails never import what they
veto and evaluation is never a runtime dependency.

### The response flow is the UI

Every answer renders in the documented order, with visual indicators:

```
CUSTOMER ISSUE → UNDERSTANDING (intent · sentiment · urgency)
               → EVIDENCE (sources · screenshot observations)
               → ACTIONS (tools called, with arguments)
               → RESULT (answer · ✅ resolved / ❓ needs clarification /
                         🔺 escalated / 🚫 refused / low-confidence flag)
               → ESCALATION (reason and reference, when applicable)
```

**No chain of thought is exposed.** The "Decision details" panel shows steps,
tools, sources, latency, intent margin and which safety rules fired — what was
done and why, at the level of actions and evidence.

### Error handling in one place

Every failure mode is caught in the service layer and returned as a typed
result, so no page can show a traceback: empty input, oversized message,
oversized upload, corrupt image, unsupported format, missing index, unset API
key, tool errors, model errors. **A test asserts every error carries a hint**
telling the user what to do next.

### Trace persistence

`src/observability/traces.py` writes one row per request to `data/db/traces.db`.
Message text is **redacted through the guardrail layer before storage** — a
customer who pastes a card number into chat should not have it sitting in the
trace table. History and the live Analytics tab read from it; 👍/👎 writes back.

### ⚠️ Simulated data is labelled in the UI

The Analytics page separates **Simulated history** (11,905 synthetic tickets with
planted trends) from **This installation** (real logged requests). The synthetic
tab carries a visible banner. Presenting generated volume as real operational
data would misrepresent the system.

---

## Phase 13b — Support Intelligence

```bash
python scripts/simulate_support_traffic.py --days 35 --per-day 14
streamlit run app/Home.py     # → Support Intelligence
```

Full write-up: **[`reports/support_intelligence_report.md`](reports/support_intelligence_report.md)**

AI and support-operations analytics over the system's **own conversations** —
not a business dashboard. No sales, products, customer segments or revenue.
A test scans the module's public functions and fails if a business metric
appears.

### Real behaviour on a synthetic workload

The **messages** are synthetic. Everything measured about them is **real**: the
intent came from the classifier, the retrieval scores from the index, the
escalation reason from the gate that actually fired.

### What the dashboard found

> Login / account access issues appeared this week (14 conversations) with no
> prior history. 100% required a human.
>
> Payment failure issues are running 3.1× their usual rate this week.
>
> The knowledge base returned nothing usable for 99 conversations (26%).

The first two are a planted surge, found. The last was not planted.

### Three findings worth acting on

**Retrieval quality drives human workload.** Escalation is **82.8%** when
retrieval fails versus **18.0%** when it succeeds. That reframes the 35%
escalation rate as a documentation problem, not a model-capability one.

**58% of escalations are capability gaps, not policy.** Splitting handovers into
*by design* (refunds, identity, legal threats) versus *capability gap* (no
documentation found) is the most useful thing here — reporting one number would
hide it entirely.

**Screenshots resolve 31 points more often** — 84.8% vs 53.5%. The Phase 8
ablation showing up in operational data.

### Volume is not workload

`warranty_claim` is 13.8% of traffic and causes **zero** human work.
`return_refund_request` is 11.7% and escalates half the time — by design, since
refunds are Tier 3. Ranking by volume alone puts effort in the wrong place.

### Three failure surfaces

**Failed retrievals** (downloadable — each row is a documentation gap) ·
**Low-confidence answers** that did *not* escalate, so nobody reviewed them ·
**Recurring unresolved issues**, clustered by frequency.

### Three bugs this surfaced

Screenshots reported 0% usage while the tool showed 33 calls — the agent never
set `has_image`. Sentiment and urgency were null on every trace — computed
during understanding, then dropped. Retrieval failure wasn't measurable at all
until `max_bm25` and `retrieval_failed` were added to the schema.

---

## Phase 14 — Testing strategy

```bash
pytest                                   # 572 tests, ~4.5 min
python scripts/verify_test_suite.py      # prove the suite catches real bugs
```

Full write-up: **[`docs/TESTING.md`](docs/TESTING.md)**

### The suite is verified, not just green

A passing test suite proves nothing on its own — it might assert things true no
matter what the code does. So `scripts/verify_test_suite.py` introduces **ten
deliberate bugs** one at a time and checks the tests **fail**.

**Mutation score: 10/10.**

| Mutation | Caught |
|---|---|
| Tier-3 tools become callable | ✅ |
| Superseded policy cited as current | ✅ |
| Abstention disabled | ✅ |
| Prompt injection undetected | ✅ |
| Fabricated figures pass grounding | ✅ |
| Order normalisation broken | ✅ |
| Eligibility reported backwards | ✅ |
| Conflict detection off | ✅ |
| Codes invented on unreadable images | ✅ |
| Tool errors crash the request | ✅ |

**The first run scored 89%** — `eligibility_inverted` survived. Every test
asserted the eligibility *string*; nothing asserted the *boolean* the agent and
UI actually branch on. Inverting it would have told customers the opposite of
the truth about returning their laptop, with the whole suite green. Two tests
closed it.

### Coverage by category

```bash
pytest -m data            # 149    pytest -m tools        #  46
pytest -m classification  #  39    pytest -m agent        #  39
pytest -m retrieval       #  58    pytest -m guardrails   #  63
pytest -m rag             #  41    pytest -m ui           #  19
pytest -m vision          #  41    pytest -m integration  #  77
```

### Three real bugs the new tests found

**Footer regex missed the manual layout.** Policies split footers across lines;
manuals put reference, date, company and page on one line. Every manual page
kept its footer, leaking the company name into chunks.

**`call_tool` crashed on any tool exception.** An upstream failure would have
taken down the customer's request rather than degrading the answer.

**Eligibility boolean was untested** — found by mutation, described above.

### What is not tested

No hosted LLM, no load or concurrency testing, no browser testing, no
property-based generation, and line coverage is deliberately **not** reported —
mutation score measures whether tests *detect changes*, which is the stronger
property.

---

## Setup, run and deploy

Full guide with PowerShell commands: **[`docs/SETUP_AND_DEPLOY.md`](docs/SETUP_AND_DEPLOY.md)**

### Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

python scripts\setup_database.py
python scripts\build_index.py
python scripts\train_intent_classifier.py

python scripts\verify_setup.py      # one command that checks everything
streamlit run app\Home.py
```

`verify_setup.py` checks packages, artifacts and configuration, then runs three
real requests — a normal question, a refund (must escalate), and a prompt
injection (must refuse). It ends with `READY` or prints the exact commands to
fix what's missing.

### Deploy to Streamlit Cloud

The repo ships everything Cloud needs: `requirements.txt` (no `torch`, fits the
~1 GB limit), `packages.txt` (installs Tesseract), and `.streamlit/config.toml`.

`data/index/`, `models/` and `pacify.db` are **committed deliberately** —
Streamlit Cloud has no build step, so the app must ship with a prebuilt index or
it cannot start. Total repo ~22 MB.
