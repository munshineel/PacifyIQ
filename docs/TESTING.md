# PacifyIQ — Testing Strategy

**572 tests. 100% mutation score.**

```bash
pytest                                   # everything (~4.5 min)
python scripts/verify_test_suite.py      # prove the suite catches real bugs
```

---

## 1. The problem with a green test suite

A passing suite proves nothing on its own. It might be asserting things that are
true no matter what the code does — that a function returns a dict, that a list
has a length, that a string is a string.

So this project answers a harder question directly:

> **If I deliberately break the system, does the suite notice?**

`scripts/verify_test_suite.py` introduces ten real bugs one at a time, runs the
relevant tests, and checks they **fail**. A mutation that survives is a hole:
code that could break in production with every test still green.

| Mutation | What it breaks | Caught |
|---|---|---|
| `tier_bypass` | Tier-3 tools become callable | ✅ |
| `archived_policy_leaks` | Superseded policy cited as current | ✅ |
| `abstention_disabled` | System stops refusing what it can't answer | ✅ |
| `injection_undetected` | Prompt injection stops being detected | ✅ |
| `grounding_check_off` | Fabricated figures pass unchecked | ✅ |
| `order_normalisation_broken` | Bare order numbers stop resolving | ✅ |
| `eligibility_inverted` | Return eligibility reported backwards | ✅ |
| `conflict_detection_off` | Contradictory policies stop surfacing | ✅ |
| `vision_invents_codes` | Codes reported on unreadable images | ✅ |
| `tool_errors_uncaught` | A failing tool crashes the request | ✅ |

**Mutation score: 10/10.**

### This found a real gap

The first run scored **89%**. `eligibility_inverted` **survived** — every test
asserted the `eligibility` *string* (`"eligible"` / `"expired"`), and nothing
asserted the `eligible` *boolean*. The agent and the UI both branch on the
boolean.

Inverting it would have told customers the opposite of the truth about whether
they could return their laptop, **with the entire suite green.** Two tests were
added and the score went to 100%.

That is the argument for mutation testing in one example.

---

## 2. Running the tests

### Everything

```bash
pytest                    # 572 tests, ~4.5 minutes
pytest -x                 # stop at the first failure
pytest -n auto            # parallel (needs pytest-xdist)
```

### By category

```bash
pytest -m data            # 149  datasets, preprocessing, integrity
pytest -m classification  #  39  intent, sentiment, urgency
pytest -m retrieval       #  58  chunking, embedding, search
pytest -m rag             #  41  grounded generation, abstention
pytest -m vision          #  41  screenshots
pytest -m tools           #  46  tool contract and execution
pytest -m agent           #  39  planning, execution, stopping
pytest -m guardrails      #  63  safety at every stage
pytest -m ui              #  19  service layer, error handling
pytest -m integration     #  77  full pipeline
```

```bash
pytest -m "not slow"                  # skip the slow ones
pytest -m "guardrails or agent"       # combine
```

### One file, one test

```bash
pytest tests/test_tools.py
pytest tests/test_tools.py::test_sql_injection_through_an_argument_is_inert
pytest -k "injection"                 # by name
```

### Mutation testing

```bash
python scripts/verify_test_suite.py
python scripts/verify_test_suite.py --list
python scripts/verify_test_suite.py --mutation tier_bypass
```

### Prerequisites

Tests that need built artifacts **skip** rather than fail, so a fresh clone
reports skips instead of a wall of red. To run everything:

```bash
python scripts/setup_database.py
python scripts/build_index.py
python scripts/train_intent_classifier.py
python scripts/data_generation/gen_screenshots.py     # vision tests
python scripts/simulate_support_traffic.py            # analytics tests
```

`tests/test_data.py` checks those artifacts are correct and **fails loudly** if
the build is broken — so a missing artifact skips, but a *wrong* one does not.

---

## 3. Coverage by category

### DATA — 149 tests

| Area | What is checked |
|---|---|
| Corpus integrity | All 13 documents load, 47 pages, no empty extractions, every document registered, no orphan registry entries |
| Section extraction | Every document yields sections; numbering is contiguous (a gap usually means the heading regex missed one) |
| Planted defects | The contradiction, the archived policy and the regional addendum are all still present |
| Preprocessing | Footer removal, idempotence, degenerate input (empty, whitespace, single bullet) |
| Chunking | No empty chunks, unique IDs, deterministic, no chunk exceeds 3× budget, >90% carry a section |
| Training data | Labels within the taxonomy, every class populated, duplicates bounded, **leakage bounded and provably removed** |
| Database | Referential integrity, no delivery before dispatch, no negative money, 26 edge-case orders present |
| Evaluation sets | Valid JSON, unique IDs, gold labels resolve to real sections, **unanswerable topics genuinely absent** |

Two of these are worth calling out:

**`test_unanswerable_questions_are_genuinely_absent`** — if a topic marked
unanswerable turned out to be documented, the abstention metric would be
measuring nothing.

**`test_leaked_rows_are_removed_before_evaluation`** — the raw CSVs share 2 rows.
Rather than pretend otherwise, one test pins the raw count and another proves
the removal mechanism works. If it stops working, every classification number is
inflated.

### CLASSIFICATION — 39 tests

Correct predictions on the hand-authored hard set, malformed inputs (empty,
5000 chars, emoji-only, control characters, non-Latin script), unknown-class
handling, deterministic output, and calibration.

### RETRIEVAL — 58 tests

Relevant queries, irrelevant queries, no-result queries, exact-identifier
queries where dense retrieval is weak, metadata filtering, FAISS-vs-NumPy
equivalence, and a threshold guard on recall@5.

### RAG — 41 tests

Grounded answers, insufficient evidence, five classes of hallucination attempt,
five classes of malformed model output with repair, and abstention measured on
**both sides** — a system that refuses everything scores 1.0 on abstention and
must not pass.

### VISION — 41 tests

Valid screenshots, invalid formats, corrupt files, blurry (mild vs severe),
blank, noise, irrelevant, oversized, and low-resolution. The central test:
**never invents a code on an unreadable image**, plus its converse — a
poor-quality but legible image must still be read, because being uselessly
conservative is also a failure.

### TOOLS — 46 tests

| Area | Examples |
|---|---|
| Contract | Every tool returns a `ToolResult`; `.data` is always a mapping |
| Valid calls | Order normalisation across 5 formats; results are ranked |
| Invalid parameters | 8 malformed order IDs, degenerate queries, unexpected kwargs |
| Security | **SQL injection verified inert by counting rows afterwards**, not assumed |
| Failures | `NOT_FOUND` distinct from `ERROR`; a raising tool is caught and typed |
| Conflicting outputs | Eligibility agrees with days-remaining; refund never exceeds price; **defective refund never worse than change-of-mind**; two tools agree on the same order |
| Determinism | Repeated calls return identical results |

### AGENT — 39 tests

Correct tool selection, **unnecessary tool avoidance** (mean 1.9 of 13 available;
a test asserts it stays below one-third), escalation decisions, and termination
under adversarial input including `"?"`, `"aaaa"` and 40 emoji.

### GUARDRAILS — 63 tests

All 30 planted attacks, false-positive rate on 295 benign messages, image-borne
injection, PII redaction that preserves order references, unauthorised actions
at any confidence, and **refusal messages that don't name the rule** — a refusal
naming its trigger is a free oracle for an attacker.

### UI — 19 tests

Empty input, oversized input, oversized upload, corrupt upload, missing
configuration, and — enforced by parsing the import graph — that `src/` never
imports `streamlit` and pages never bypass the service layer.

### INTEGRATION — 77 tests

The seams, where most of this project's real bugs lived:

- Entities extracted during understanding arrive at tools as normalised arguments
- Tool facts answer questions without policy documents *(regression: the RAG gate escalated a question a tool had already answered)*
- Understanding signals survive to the decision *(regression: sentiment and urgency were computed then dropped)*
- Conflict detection survives the tool boundary *(regression: the KB tool dropped `region`, blinding the resolver)*
- The pipeline survives a broken tool, a corrupt image, and every malformed input
- Repeated identical requests are stable

---

## 4. Design decisions

### Skip, don't fail, on missing artifacts

A fresh clone reports `N skipped`, not a wall of red. But `test_data.py` fails
loudly if an artifact exists and is *wrong* — the distinction between "not built"
and "built incorrectly".

### Session-scoped fixtures

Loading the index and classifier costs ~2s each. Function scope would push the
suite past twenty minutes and nobody would run it.

### Shared malformed-input fixtures

Eleven degenerate inputs — empty, whitespace, single char, 5000 chars, emoji,
control characters, SQL, HTML, Devanagari, extreme repetition — parametrised
across every entry point. One definition, many suites.

### Test the claim, not a constant

An early integration test asserted `max_bm25 > 20` after a screenshot. It failed
at 13.37 — which was still a real lift over the ~5 baseline. The assertion was
rewritten to compare *with image* against *without image*, which is the actual
claim.

### Regressions are named

Tests that guard a specific past bug say so in the docstring. Six months later
that comment is the difference between "why is this here?" and "don't remove
this".

---

## 5. What is not tested

Stated plainly, because a coverage claim without exclusions is not a coverage
claim:

1. **No hosted LLM.** Everything runs on the local extractive backend. The Groq
   path is exercised only through the scripted-response backend.
2. **No load or concurrency testing.** Single-threaded throughout. Race
   conditions in the trace writer would not be caught.
3. **No browser testing.** The Streamlit *service layer* is tested exhaustively;
   rendering is verified only by booting the app and checking HTTP 200.
4. **No property-based testing.** Malformed inputs are enumerated by hand, not
   generated. Hypothesis would find inputs nobody thought of.
5. **Line coverage is not measured.** Mutation score is reported instead — it
   measures whether tests *detect changes*, which is the stronger property. Line
   coverage counts lines executed, not assertions that matter.
6. **Adversarial tests were written by the same person as the rules.** 100%
   detection measures imagination, not security.
7. **All data is synthetic.** Tests verify internal consistency, not
   correspondence with the real world.

---

## 6. Continuous integration

```bash
# fast gate — every commit, ~90s
pytest -m "not slow and not integration"

# full gate — every pull request, ~5 min
pytest

# nightly — the expensive question
pytest && python scripts/verify_test_suite.py
```

Mutation testing takes ~20 minutes because it runs a subset of the suite ten
times. It belongs on a schedule, not on the commit path.

---

## 7. Files

| Path | Tests | Category |
|---|---|---|
| `tests/conftest.py` | — | Fixtures, skip markers, malformed inputs |
| `tests/test_data.py` | 48 | data |
| `tests/test_eda.py` · `test_queries.py` · `test_analytics.py` · `test_support_intelligence.py` | 101 | data |
| `tests/test_understanding.py` | 39 | classification |
| `tests/test_knowledge.py` · `test_routing.py` | 58 | retrieval |
| `tests/test_rag.py` | 41 | rag |
| `tests/test_multimodal.py` | 41 | vision |
| `tests/test_tools.py` | 46 | tools |
| `tests/test_agent.py` | 39 | agent |
| `tests/test_guardrails.py` | 63 | guardrails |
| `tests/test_architecture.py` | 19 | ui |
| `tests/test_integration.py` · `test_evaluation.py` | 77 | integration |
| `scripts/verify_test_suite.py` | 10 mutations | — |

---

## Summary

**572 tests across ten categories, and a mutation harness proving they catch
real bugs.**

The suite has found genuine defects during construction: a footer regex that
missed the manual layout, `call_tool` crashing on any tool exception, and an
eligibility boolean that could be inverted without a single test noticing.

That last one is the point. **The tests are not there to be green.**
