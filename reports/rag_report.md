# PacifyIQ — RAG Pipeline

**Phase 7.** Grounded generation, citation enforcement, abstention, conflict
handling — built and measured **without a hosted LLM**.

```bash
python scripts/evaluate_rag.py              # local backend, no key needed
python scripts/evaluate_rag.py --backend groq   # with your key
```

---

## 1. Built without an API — deliberately

Everything in a RAG system except token generation can be built and measured
without a model: context assembly, citation verification, abstention thresholds,
conflict detection, grounding checks, prompt versioning, token budgeting.

The pipeline is built against an **interface** rather than an SDK, with three
backends:

| Backend | What | Requires |
|---|---|---|
| **`local`** | extractive generator — selects the sentences from retrieved context most relevant to the question and attaches citations | nothing |
| `groq` | hosted chat completion, retries, JSON mode | API key |
| `scripted` | fixed responses, for testing failure paths | nothing |

**The local backend is not a stub.** It is a genuine RAG baseline — the kind of
system that shipped before instruction-tuned models existed — and it produces
real numbers an LLM can be measured against. It also has one property no LLM has:

> Every word it emits came verbatim from a retrieved document, so **faithfulness
> is 1.0 by construction**. That makes it a useful floor: any hallucination an
> LLM introduces is attributable to the LLM, not to the retrieval or the prompt.

Swapping to Groq is one flag. All 41 Phase 7 tests run offline and in CI.

---

## 2. Pipeline

```
question
  → retrieve                 (Phase 6: rrf_w, recall@5 0.875)
  → assemble context         token budget, position-aware ordering
  → ABSTENTION GATE          ← before generation, not after
  → generate                 (any backend)
  → parse + repair           5 malformation classes handled
  → verify grounding         citations + numbers + codes
  → answer or escalate
  → one trace record
```

### The gate placement is the key design decision

The abstention gate sits **before** generation. A question with no supporting
evidence never reaches the model at all.

That removes a class of hallucination rather than trying to detect it
afterwards, and it is verifiable: on an unanswerable question the trace records
`llm_backend = "none"` and `completion_tokens = 0`. There is a test asserting
exactly that.

### Context ordering

Models attend most reliably to the start and end of context and least reliably
to the middle. Chunks are interleaved so rank order `[1,2,3,4,5]` becomes
`[1,3,5,4,2]` — strongest evidence at both edges, weakest buried in the middle.
Free to apply, and it costs nothing if the effect is small.

---

## 3. Abstention — thresholds derived, not guessed

The most important behaviour in a grounded support system is knowing when *not*
to answer. A fluent wrong answer about a return window creates contractual
exposure; an honest refusal does not.

### The finding that set the threshold

Measuring 120 answerable against 40 unanswerable questions:

| Signal | Answerable (median) | Unanswerable (median) | Separation |
|---|---|---|---|
| **BM25 max** | **13.12** | **6.21** | **2.1×** |
| Dense cosine max | 0.572 | 0.507 | 1.1× |
| RRF fused score | 0.0156 | 0.0164 | **inverted** |

**BM25 separates roughly twice as well as dense cosine, and RRF is worse than
useless.** Fusion compresses everything into a ~0.015–0.017 band, and an
unanswerable question can score *higher* than an answerable one.

The reason is intuitive once measured: a question about a topic the corpus does
not cover contains **rare vocabulary that appears nowhere**. A lexical scorer
notices that directly. An embedding smooths it away — "student discount" is
semantically near "payment terms", so cosine stays comfortably high.

> **This is a genuinely counterintuitive result and worth stating in an
> interview:** for abstention, the *lexical* signal beats the *semantic* one,
> and the fused score you rank with is the wrong thing to threshold on.

Thresholds: abstain below BM25 7.0, caveat below 10.0.

### Results

| Metric | Value |
|---|---|
| **Abstention rate** (40 unanswerable) | **0.700** |
| False abstention (60 answerable) | 0.133 |
| **Balanced score** (true × (1 − false)) | **0.607** |
| Mean BM25 when abstaining | 5.52 |
| Mean BM25 when answering | 10.71 |

**Both sides are reported deliberately.** A system that refuses everything scores
1.00 on abstention and is useless; the balanced score makes that visible — it
would be 0.

### Where abstention fails

**12 of 40 wrongly answered.** They split into two clear groups:

| Group | n | Example | Why |
|---|---|---|---|
| **Spec questions about real products** | 5 | *"How much VRAM does the ProBook 16 have?"* | The manual exists and matches every term except the one that matters. High lexical overlap, missing fact. |
| **Plausible policy extensions** | 7 | *"Can I upgrade PacifyCare+ from 12 to 24 months?"* (BM25 20.5) | The corpus discusses PacifyCare+ extensively; it just never addresses upgrading. |

Both are the **hardest possible abstention cases**: the topic is covered, the
specific fact is not. No retrieval-score threshold can separate them, because
retrieval is behaving correctly — it found the right document. Detecting this
requires checking whether the retrieved text actually *contains* an answer, which
is a generation-side check, not a retrieval-side one.

**8 of 60 falsely refused**, and they cluster tightly: *"Am I talking to a bot?"*
(BM25 0.00), *"How do I escalate my issue?"*, *"What are your support hours?"*.
All are **meta-questions about the support system itself**, answered by
`customer_service_policy` but phrased with vocabulary that appears nowhere in it.
Phase 6's failure analysis flagged the same cluster. The fix is
intent-conditional routing using the Phase 4 classifier, not a threshold change.

---

## 4. Grounding verification

A citation the model emits is a claim, not a fact. Two checks run on every
answer:

**1. Do cited sources exist and were they in the context?** Matched on document
reference and section, ignoring page (a section spans pages; a page mismatch is
a formatting slip, not a fabrication).

**2. Is the content supported?** Deliberately **lexical, not LLM-judged.** An LLM
judging LLM output has correlated failure modes. More importantly, every planted
hallucination trap in this corpus is **numeric** — 144Hz vs 75Hz, IP68 vs IP53,
Rs 57,960 — and a number in the answer that appears nowhere in the context is
unambiguous evidence of fabrication. Cheap, deterministic, and it catches the
case that matters.

Five failure types, all caught:

| Type | Example | Detected by |
|---|---|---|
| Fabricated number | *"You have 45 days"* | number absent from context |
| Fabricated citation | `POL-XYZ-999` | not in supplied sources |
| Fabricated error code | `THRM-88` | code absent from context |
| Missing citation | claim with no source | citation count zero |
| Low overlap | fluent but unrelated | token overlap |

**An ungrounded answer is never shown as-is.** It forces escalation and caps
confidence at 0.3. A test asserts this using a scripted backend that returns a
fabricated 45-day figure with an invented citation.

---

## 5. Structured output

Three layers, because each fails differently:

1. Ask for JSON in the prompt — usually works
2. Provider JSON mode — guarantees syntax, not schema
3. **Parse, repair, validate** — the only layer that guarantees fields

Even with provider JSON mode, layer 3 is mandatory: schema-valid output can be
semantically wrong. A citation pointing at a page that does not exist is perfectly
well-formed JSON.

Five malformation classes repaired, all tested:

| Malformation | Fix |
|---|---|
| ` ```json ` fences | strip fence |
| Prose wrapper (*"Sure! Here you go: {...} Hope that helps."*) | extract object |
| Trailing comma | remove |
| Python dict syntax (single quotes) | convert keys **and values** |
| Confidence out of `[0,1]` | clip and flag |

**Unparseable output escalates rather than raising.** Raising would take down the
request; escalating hands it to a human with the context already gathered.

---

## 6. Generation results (local extractive backend)

| Metric | Value |
|---|---|
| Correctness (exact fact match) | 0.240 |
| Partial credit | 0.580 |
| **Faithfulness** | **1.000** |
| **Hallucination rate** | **0.000** |
| Citation accuracy | 1.000 |
| Citation recall | 0.587 |
| Answers carrying a citation | 0.920 |

**Read these as a floor, not a result.** Faithfulness of 1.0 and hallucination of
0.0 are guaranteed by construction — an extractive generator cannot fabricate.
Correctness of 0.24 is what you get from sentence selection with no paraphrasing
or reasoning.

The **gap between 0.58 partial and 0.24 exact** is the shape of the limitation:
the extractive backend usually retrieves a sentence containing part of the answer
but cannot synthesise across sentences. *"How much do I get back on a Rs 64,900
laptop?"* needs the fee rate, the shipping charge, and the waterfall combined into
one figure. Extraction returns three sentences; it cannot compute Rs 57,960.

**This is exactly the gap an LLM should close, and it is now measurable.**

---

## 7. ⚠️ Prompt comparison — measured, and the result is null

| Prompt | Correctness | Faithfulness | Citation acc. | Abstention |
|---|---|---|---|---|
| v1 (minimal) | 0.24 | 1.0 | 1.0 | 0.70 |
| v2 (+ schema, rules) | 0.24 | 1.0 | 1.0 | 0.70 |
| v3 (+ conflicts, authority) | 0.24 | 1.0 | 1.0 | 0.70 |

**Identical across all three, to two decimal places.**

This is not a finding about prompts. It is a finding about the **measurement
apparatus**: the extractive backend does not follow instructions, so prompt
content cannot affect its output. The harness is built, versioned and wired to
the eval sets — but it **cannot produce a signal without an instruction-following
model.**

Reporting a null result here rather than quietly omitting the comparison is the
honest move. The harness is ready; the measurement is blocked on the LLM, and
running `scripts/evaluate_rag.py --backend groq` will produce it.

---

## 8. Honest limitations

1. **No LLM has been run.** Every generation number comes from the extractive
   backend. Correctness will rise substantially with an instruction-following
   model; faithfulness will almost certainly *fall* from 1.0, because a model
   that can paraphrase can also fabricate. **The interesting comparison has not
   been made.**
2. **Prompt engineering is unmeasured** — see §7.
3. **Grounding is lexical.** It catches numeric and citation fabrication reliably.
   It does not catch a paraphrase that reverses meaning while reusing the same
   vocabulary — *"you may not return opened items"* passes a token-overlap check
   against context stating the opposite. That needs entailment checking.
4. **Correctness scoring is string matching**, not semantic. An answer conveying
   the right fact in different words scores as wrong. This *understates*
   correctness and was chosen over an LLM judge because judging LLM output with
   an LLM is circular.
5. **12 of 40 abstention failures are structurally unfixable by retrieval
   thresholds** — see §3.
6. **The corpus is small** (16,208 words), so all numbers are optimistic relative
   to production.

---

## 9. What to run with your key

```bash
python scripts/evaluate_rag.py --backend groq
```

Produces the four comparisons this report is missing:

| Comparison | Expected direction |
|---|---|
| Correctness: extractive 0.24 → LLM | should rise sharply — synthesis is what extraction cannot do |
| Faithfulness: 1.00 → LLM | **should fall.** The delta is the hallucination rate attributable to the model |
| Prompt v1 → v2 → v3 | should separate; if not, the prompts need work |
| Abstention: 0.70 → LLM | may improve — a model can notice the retrieved text lacks the fact |

Cost is a few cents. Paste the output and I'll write it into this report properly.

---

## 10. Artifacts

| Path | Contents |
|---|---|
| `src/llm/client.py` | 3 backends behind one interface, token accounting |
| `src/llm/structured.py` | Parse, repair, validate; citation extraction |
| `src/rag/prompts.py` | 3 versioned prompts with change notes |
| `src/rag/context.py` | Assembly, budgeting, position-aware ordering |
| `src/rag/abstention.py` | The gate, thresholds derived from measurement |
| `src/rag/citations.py` | Verification and grounding checks |
| `src/rag/generator.py` | Pipeline + trace schema |
| `src/rag/evaluation.py` | Correctness, faithfulness, abstention metrics |
| `reports/results/rag_summary.json` | Headline numbers |
| `reports/results/rag_failures.csv` | 19 incorrect cases with diagnosis |
| `reports/results/rag_abstention_per_case.csv` | All 40, with BM25 scores |
| `reports/results/rag_prompt_comparison.csv` | The null result |

**41 Phase 7 tests, all offline.** 185 across the project.

---

## Established

```
question → retrieve → assemble → GATE → generate → verify → answer | escalate
```

Grounded answers with verified citations, an abstention path that refuses 70% of
unanswerable questions while wrongly refusing only 13% of answerable ones, and
five classes of hallucination detected and blocked — **with no network call
anywhere in the pipeline.**

## Next

**Phase 8 — multimodal.** 🖼️ Screenshot generation for the 25 vision cases, error-code
extraction, and the text-only vs text+vision retrieval ablation.
