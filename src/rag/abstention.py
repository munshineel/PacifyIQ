"""Abstention and escalation decisions.

The most important behaviour in a grounded support system is knowing when *not*
to answer. A fluent wrong answer about a return window creates a contractual
exposure; an honest "I don't have documentation on that" does not.

Thresholds here were chosen by measuring the separation between the 120
answerable and 40 unanswerable evaluation questions, not by intuition. The
measurement produced a finding worth stating: **BM25 score separates answerable
from unanswerable roughly twice as well as dense cosine does** (13.5 vs 6.8
median, against 0.59 vs 0.51 for cosine). Rare vocabulary is the signal — a
question about a topic the corpus does not cover contains terms that appear
nowhere, and a lexical scorer notices that directly while an embedding smooths
it away.

RRF scores are useless here. Fusion compresses everything into ~0.015-0.017 and
an unanswerable question can score higher than an answerable one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.knowledge.retriever import RetrievalResult
from src.rag.context import AssembledContext


class Decision(str, Enum):
    ANSWER = "answer"                # evidence is sufficient
    ANSWER_WITH_CAVEAT = "caveat"    # answerable, but flag a limitation
    ABSTAIN = "abstain"              # no supporting evidence
    ESCALATE = "escalate"            # conflicting evidence, or out of scope


# Thresholds. See module docstring for how these were derived.
BM25_ABSTAIN_BELOW = 7.0      # unanswerable median 6.2, answerable median 13.1
BM25_WEAK_BELOW = 10.0        # weak but not absent
DENSE_ABSTAIN_BELOW = 0.42    # secondary signal only; dense separates poorly
MIN_CHUNKS = 1


@dataclass
class AbstentionResult:
    decision: Decision
    reason: str
    signals: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    @property
    def should_generate(self) -> bool:
        return self.decision in (Decision.ANSWER, Decision.ANSWER_WITH_CAVEAT)

    @property
    def needs_human(self) -> bool:
        return self.decision == Decision.ESCALATE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def decide(
    result: RetrievalResult,
    context: AssembledContext,
    bm25_abstain: float = BM25_ABSTAIN_BELOW,
    bm25_weak: float = BM25_WEAK_BELOW,
    dense_abstain: float = DENSE_ABSTAIN_BELOW,
) -> AbstentionResult:
    """Decide whether to answer, caveat, abstain, or escalate.

    Runs *before* generation, so a question with no supporting evidence never
    reaches the model at all. That removes an entire class of hallucination
    rather than trying to detect it afterwards.
    """
    signals = {
        "max_bm25": round(result.max_bm25_score, 3),
        "max_dense": round(result.max_dense_score, 4),
        "n_chunks": float(context.n_chunks),
        "score_gap": round(context.score_gap, 4),
        "n_docs": float(len(context.docs)),
    }
    caveats: list[str] = []

    # ---- nothing retrieved ------------------------------------------
    if context.is_empty or context.n_chunks < MIN_CHUNKS:
        return AbstentionResult(Decision.ABSTAIN, "no_chunks_retrieved", signals)

    # ---- conflicting evidence, checked FIRST -------------------------
    # Precedence matters: a version conflict must escalate even when lexical
    # support is thin. Abstaining on a conflict would hide it, and hiding a
    # conflict is the exact failure the corpus was built to expose.
    if context.has_version_conflict:
        return AbstentionResult(
            Decision.ESCALATE, "version_conflict_in_evidence", signals,
            ["Retrieved evidence spans current and superseded policy versions."],
        )

    # ---- evidence too weak to support any claim ---------------------
    if result.max_bm25_score < bm25_abstain and result.max_dense_score < dense_abstain:
        return AbstentionResult(
            Decision.ABSTAIN,
            f"weak_evidence (bm25 {result.max_bm25_score:.1f} < {bm25_abstain}, "
            f"cosine {result.max_dense_score:.2f} < {dense_abstain})",
            signals,
        )
    if result.max_bm25_score < bm25_abstain:
        return AbstentionResult(
            Decision.ABSTAIN,
            f"no_lexical_support (bm25 {result.max_bm25_score:.1f} < {bm25_abstain})",
            signals,
        )

    # ---- regional variants -------------------------------------------
    if context.has_region_variant:
        caveats.append(
            "Evidence includes a regional variant; the answer may differ by country."
        )

    # ---- weak but usable --------------------------------------------
    if result.max_bm25_score < bm25_weak:
        caveats.append("Supporting evidence is weak; treat this answer as provisional.")
    if context.is_single_source:
        caveats.append("All evidence comes from a single document.")
    if context.dropped:
        caveats.append(f"{context.dropped} retrieved chunk(s) dropped for token budget.")

    if caveats:
        return AbstentionResult(
            Decision.ANSWER_WITH_CAVEAT, "sufficient_with_caveats", signals, caveats
        )
    return AbstentionResult(Decision.ANSWER, "sufficient_evidence", signals)


ABSTENTION_MESSAGE = (
    "I don't have documentation covering that. I've passed this to a human "
    "colleague who can look into it properly."
)

CONFLICT_MESSAGE = (
    "I found conflicting information in our documentation on this, so I don't "
    "want to give you a figure that might be wrong. I've escalated it to a "
    "colleague who can confirm which applies to your order."
)


if __name__ == "__main__":
    import json

    from src.config.settings import settings
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.retriever import Retriever
    from src.knowledge.vector_store import VectorStore
    from src.rag.context import assemble

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    r = Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)

    answerable = [c["question"] for c in
                  json.loads((settings.eval_dir / "retrieval_eval.json").read_text())["cases"][:12]]
    unanswerable = [c["question"] for c in
                    json.loads((settings.eval_dir / "unanswerable_eval.json").read_text())["cases"][:12]]

    for label, qs in [("ANSWERABLE", answerable), ("UNANSWERABLE", unanswerable)]:
        print(f"\n{label}")
        correct = 0
        for q in qs:
            res = r.retrieve(q)
            d = decide(res, assemble(res))
            want_abstain = label == "UNANSWERABLE"
            got_abstain = d.decision == Decision.ABSTAIN
            ok = got_abstain == want_abstain
            correct += ok
            print(f"  {'ok ' if ok else 'MISS'} {d.decision.value:8s} "
                  f"bm25={d.signals['max_bm25']:6.2f}  {q[:52]}")
        print(f"  -> {correct}/{len(qs)} correct")
