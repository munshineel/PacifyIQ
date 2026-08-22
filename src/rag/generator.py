"""The RAG pipeline.

    question
      -> retrieve
      -> assemble context
      -> abstention gate        (before generation, not after)
      -> generate
      -> parse + repair
      -> verify grounding
      -> final answer or escalation

The abstention gate sits *before* generation deliberately. A question with no
supporting evidence never reaches the model, which removes a class of
hallucination rather than trying to detect it afterwards.

Every stage emits into one trace record. The dashboard is a read-only view over
those traces, so the schema is fixed here rather than bolted on later.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.retriever import Retriever
from src.llm.client import LLMClient, Message, count_tokens, get_llm
from src.llm.structured import SupportResponse, parse_response
from src.rag.abstention import (ABSTENTION_MESSAGE, CONFLICT_MESSAGE,
                                AbstentionResult, Decision, decide)
from src.rag.citations import GroundingReport, check_grounding
from src.rag.context import AssembledContext, assemble, budget_report
from src.rag.prompts import get_prompt


@dataclass
class RAGTrace:
    """One request, fully accounted for. This is the dashboard's data source."""

    question: str
    answer: str
    decision: str
    escalated: bool

    prompt_version: str = ""
    llm_backend: str = ""
    llm_model: str = ""

    retrieval_strategy: str = ""
    n_chunks: int = 0
    max_bm25: float = 0.0
    max_dense: float = 0.0

    has_image: bool = False
    image_contributed: bool = False
    image_error_code: str | None = None
    image_evidence_level: str = ""
    image_terms: list[str] = field(default_factory=list)

    intent: str = ""
    intent_margin: float = 0.0
    sentiment: str = ""
    urgency: str = ""
    routing_reasons: list[str] = field(default_factory=list)

    citations: list[str] = field(default_factory=list)
    n_citations: int = 0
    citation_accuracy: float = 0.0
    is_grounded: bool = True
    hallucination_flags: list[str] = field(default_factory=list)

    confidence: float = 0.0
    caveats: list[str] = field(default_factory=list)
    parse_ok: bool = True
    parse_repaired: bool = False

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d

    def summary(self) -> str:
        bits = [
            f"decision={self.decision}",
            f"chunks={self.n_chunks}",
            f"bm25={self.max_bm25:.1f}",
            f"cites={self.n_citations}",
            f"grounded={self.is_grounded}",
            f"{self.latency_ms:.0f}ms",
            f"{self.total_tokens}tok",
        ]
        if self.hallucination_flags:
            bits.append(f"FLAGS={','.join(self.hallucination_flags)}")
        return " | ".join(bits)


@dataclass
class RAGResult:
    """Everything one question produced."""

    question: str
    response: SupportResponse
    trace: RAGTrace
    abstention: AbstentionResult
    context: AssembledContext
    grounding: GroundingReport | None = None
    fused: Any = None

    @property
    def answer(self) -> str:
        return self.response.answer

    @property
    def escalated(self) -> bool:
        return self.trace.escalated

    def display(self) -> str:
        lines = [self.response.answer]
        if self.response.citations:
            lines.append("")
            lines.append("Sources:")
            for c in self.response.citations:
                mark = "" if c.verified else "  [unverified]"
                lines.append(f"  - {c}{mark}")
        if self.abstention.caveats:
            lines.append("")
            for cav in self.abstention.caveats:
                lines.append(f"Note: {cav}")
        if self.escalated:
            lines.append("")
            lines.append("This has been passed to a human colleague.")
        return "\n".join(lines)


class RAGPipeline:
    """Retrieve, decide, generate, verify."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None = None,
        prompt_version: str = "v3",
        top_k: int = 5,
        max_context_tokens: int = 1800,
        max_response_tokens: int = 512,
        temperature: float = 0.1,
        verify_grounding: bool = True,
        use_routing: bool = True,
    ):
        self.retriever = retriever
        # Understanding-driven routing. Measured to lift recall@5 0.875 -> 0.883
        # and MRR 0.664 -> 0.706; see reports/routing_report.md.
        self.routed = None
        if use_routing:
            try:
                from src.rag.routing import RoutedRetriever

                self.routed = RoutedRetriever(retriever)
            except FileNotFoundError:
                # no trained classifier yet - fall back to plain retrieval
                self.routed = None
        self.llm = llm or get_llm("local")
        self.prompt = get_prompt(prompt_version)
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.verify_grounding = verify_grounding

    # ---------------------------------------------------------------
    def answer_multimodal(
        self,
        text: str,
        image_path=None,
        image_filename: str | None = None,
        context=None,
        vision_backend: str = "local_ocr",
        **retrieve_kw,
    ) -> RAGResult:
        """Entry point for a request that may carry a screenshot.

        The image is fused INTO the query rather than analysed separately, so
        everything downstream - understanding, routing, retrieval, generation -
        sees one enriched request. See src/multimodal/fusion.py.
        """
        from src.multimodal.fusion import CustomerContext, MultimodalRequest, fuse

        fused = fuse(
            MultimodalRequest(
                text=text, image_path=image_path, image_filename=image_filename,
                context=context or CustomerContext(),
            ),
            vision_backend=vision_backend,
        )
        result = self.answer(
            fused.enriched_query,
            _display_question=text,
            _image_evidence=fused.evidence_text,
            **retrieve_kw,
        )

        t = result.trace
        t.has_image = fused.has_image
        t.image_contributed = fused.image_contributed
        t.image_terms = fused.image_terms
        if fused.image:
            t.image_error_code = fused.image.error_code
            t.image_evidence_level = fused.image.evidence.get(
                "error_code", "unknown"
            ).value if fused.image.evidence.get("error_code") else "unknown"
        result.fused = fused
        return result

    def answer(self, question: str, _display_question: str | None = None,
               _image_evidence: str = "", **retrieve_kw) -> RAGResult:
        t0 = time.perf_counter()

        # 1. retrieve, with understanding-driven routing when available
        understanding = None
        route = None
        if self.routed is not None:
            retrieval, route, understanding = self.routed.retrieve(
                question, top_k=self.top_k, **retrieve_kw
            )
        else:
            retrieval = self.retriever.retrieve(question, top_k=self.top_k, **retrieve_kw)

        # 2. assemble
        context = assemble(
            retrieval, max_tokens=self.max_context_tokens, max_chunks=self.top_k
        )

        # 3. abstention gate - before the model sees anything
        gate = decide(retrieval, context)

        if not gate.should_generate:
                return self._short_circuit(
                question, retrieval, context, gate, t0, understanding, route
            )

        # 4. generate
        # Image observations are prepended to the retrieved context so the
        # model reasons over text and image evidence together, with the
        # evidence levels attached to each image claim.
        ctx_text = (
            f"{_image_evidence}\n\n{context.text}" if _image_evidence
            else context.text
        )
        user = self.prompt.render(
            context=ctx_text, question=_display_question or question
        )
        messages = [Message("system", self.prompt.system), Message("user", user)]
        llm_out = self.llm.complete(
            messages, temperature=self.temperature, max_tokens=self.max_response_tokens
        )

        # 5. parse and repair
        response = parse_response(llm_out.text)

        # 6. verify grounding
        grounding = None
        if self.verify_grounding:
            grounding = check_grounding(response, context.text, context.citations)
            if not grounding.is_grounded:
                # An ungrounded answer is not shown as-is. Numeric fabrication
                # is the failure this system exists to prevent.
                response.needs_escalation = True
                response.escalation_reason = (
                    response.escalation_reason or "failed_grounding_check"
                )
                response.confidence = min(response.confidence, 0.3)

        escalated = bool(response.needs_escalation or gate.needs_human)

        trace = RAGTrace(
            question=_display_question or question,
            answer=response.answer,
            decision=gate.decision.value,
            escalated=escalated,
            prompt_version=self.prompt.version,
            llm_backend=llm_out.backend,
            llm_model=llm_out.model,
            retrieval_strategy=retrieval.strategy,
            n_chunks=context.n_chunks,
            max_bm25=round(retrieval.max_bm25_score, 3),
            max_dense=round(retrieval.max_dense_score, 4),
            citations=[str(c) for c in response.citations],
            n_citations=len(response.citations),
            citation_accuracy=round(grounding.citation_accuracy, 3) if grounding else 0.0,
            is_grounded=grounding.is_grounded if grounding else True,
            hallucination_flags=grounding.hallucination_flags if grounding else [],
            confidence=response.confidence,
            caveats=gate.caveats,
            parse_ok=response.parse_ok,
            parse_repaired=response.repaired,
            prompt_tokens=llm_out.prompt_tokens,
            completion_tokens=llm_out.completion_tokens,
            intent=understanding.intent if understanding else "",
            intent_margin=understanding.intent_margin if understanding else 0.0,
            sentiment=understanding.sentiment if understanding else "",
            urgency=understanding.urgency if understanding else "",
            routing_reasons=route.reasons if route else [],
            latency_ms=(time.perf_counter() - t0) * 1000,
            budget=budget_report(
                count_tokens(self.prompt.system), context.n_tokens,
                count_tokens(question), self.max_response_tokens,
            ),
        )
        return RAGResult(question, response, trace, gate, context, grounding)

    # ---------------------------------------------------------------
    def _short_circuit(self, question, retrieval, context, gate, t0,
                       understanding=None, route=None) -> RAGResult:
        """Abstain or escalate without calling the model at all."""
        message = (
            CONFLICT_MESSAGE if gate.decision == Decision.ESCALATE
            else ABSTENTION_MESSAGE
        )
        response = SupportResponse(
            answer=message,
            citations=[],
            confidence=0.0,
            needs_escalation=True,
            escalation_reason=gate.reason,
        )
        trace = RAGTrace(
            question=question,
            answer=message,
            decision=gate.decision.value,
            escalated=True,
            prompt_version=self.prompt.version,
            llm_backend="none",
            llm_model="none",
            retrieval_strategy=retrieval.strategy,
            n_chunks=context.n_chunks,
            max_bm25=round(retrieval.max_bm25_score, 3),
            max_dense=round(retrieval.max_dense_score, 4),
            confidence=0.0,
            caveats=gate.caveats,
            intent=understanding.intent if understanding else "",
            intent_margin=understanding.intent_margin if understanding else 0.0,
            sentiment=understanding.sentiment if understanding else "",
            urgency=understanding.urgency if understanding else "",
            routing_reasons=route.reasons if route else [],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return RAGResult(question, response, trace, gate, context, None)

    # ---------------------------------------------------------------
    def explain(self, question: str, **kw) -> str:
        r = self.answer(question, **kw)
        lines = [
            f"QUESTION  {question}",
            f"DECISION  {r.abstention.decision.value}  ({r.abstention.reason})",
            f"SIGNALS   {r.abstention.signals}",
            "",
            "CONTEXT   " + ", ".join(r.context.citations) if r.context.citations else "CONTEXT   none",
            "",
            "ANSWER",
            "  " + r.response.answer.replace("\n", "\n  "),
        ]
        if r.response.citations:
            lines.append("")
            lines.append("CITATIONS")
            for c in r.response.citations:
                lines.append(f"  {c}  verified={c.verified}")
        if r.grounding and r.grounding.hallucination_flags:
            lines.append(f"\nFLAGS     {r.grounding.hallucination_flags}")
        lines.append(f"\nTRACE     {r.trace.summary()}")
        return "\n".join(lines)


def build_pipeline(backend: str = "local", prompt_version: str = "v3",
                   strategy: str = "rrf_w", top_k: int = 5) -> RAGPipeline:
    """Convenience constructor that loads the committed index."""
    from src.config.settings import settings
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.vector_store import VectorStore

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    retriever = Retriever(store, emb, BM25Index(store.chunks),
                          strategy=strategy, top_k=top_k)
    return RAGPipeline(retriever, get_llm(backend), prompt_version=prompt_version,
                       top_k=top_k)


if __name__ == "__main__":
    pipe = build_pipeline()
    for q in [
        "How long do I have to return an opened laptop?",
        "How many dead pixels before you replace the screen?",
        "Do you offer student discounts?",
        "Does the Pacify Vision 27 support 144Hz?",
    ]:
        print(pipe.explain(q))
        print("\n" + "=" * 76 + "\n")
