"""The agent loop.

    INPUT
      -> understand (intent, sentiment, urgency, entities)
      -> plan       (which tools, and why)
      -> execute    (skip steps whose preconditions failed)
      -> observe    (re-plan once if a tool revealed new information)
      -> assemble   (tool results + retrieved policy + image evidence)
      -> generate   (grounded answer)
      -> gate       (confidence, grounding, tier, escalation triggers)
      -> RESOLVE or ESCALATE

WHAT MAKES THIS AN AGENT RATHER THAN A CHATBOT
----------------------------------------------
1. It selects tools from explicit rules, and calls between 0 and 5 of them
   depending on the request. It never calls everything.
2. It has state. A tool result changes what happens next - an order that turns
   out to be undeliverable changes the plan.
3. It has stop conditions. It stops when evidence is sufficient, when a step
   cannot proceed, or at a hard iteration cap. It cannot loop.
4. Tier 3 actions are blocked in code, not requested in a prompt.
5. It returns structured decision metadata, not reasoning prose.

NO CHAIN OF THOUGHT IS EXPOSED. `AgentDecision` carries what was done and why
at the level of actions and evidence, not the model's internal narration.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.agent.planner import Plan, PlannedStep, build_plan
from src.agent.tools import REGISTRY, ToolResult, ToolStatus, call_tool
from src.understanding.pipeline import UnderstandingPipeline
from src.guardrails.policy import ENGINE, SafetyRecord
from src.understanding.schema import Understanding


class Resolution(str, Enum):
    RESOLVED = "resolved"                  # answered from evidence
    RESOLVED_WITH_CAVEAT = "resolved_with_caveat"
    NEEDS_INFORMATION = "needs_information"  # cannot proceed without input
    ESCALATED = "escalated"
    REFUSED = "refused"                    # out of scope or blocked


class StopReason(str, Enum):
    PLAN_COMPLETE = "plan_complete"
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    MISSING_PRECONDITION = "missing_precondition"
    HARD_ESCALATION = "hard_escalation"
    MAX_STEPS = "max_steps"
    TOOL_FAILURES = "repeated_tool_failures"


@dataclass
class AgentState:
    """Explicit, inspectable state. Every field is written by a named stage."""

    text: str
    image_path: str | None = None
    customer_id: str | None = None
    order_id: str | None = None
    region: str | None = None

    understanding: Understanding | None = None
    plan: Plan | None = None

    executed: list[ToolResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failures: int = 0
    replans: int = 0

    knowledge_chunks: list[dict[str, Any]] = field(default_factory=list)
    image_analysis: dict[str, Any] | None = None
    max_bm25: float = 0.0
    has_conflict: bool = False

    stop_reason: StopReason | None = None

    def tool_names(self) -> list[str]:
        return [r.tool for r in self.executed]

    def context_text(self) -> str:
        """All retrieved text, for the output grounding check."""
        parts = [c.get("text", "") for c in self.knowledge_chunks]
        parts += [str(r.data) for r in self.executed if r.ok]
        return " ".join(parts)

    def available_citations(self) -> list[str]:
        return [c.get("citation", "") for c in self.knowledge_chunks]

    def result_for(self, tool: str, ok_only: bool = True) -> ToolResult | None:
        """Most recent result for a tool.

        `ok_only=False` is needed to reason about FAILURES - a not-found order
        is information, and a lookup for it that returns None is indistinguishable
        from never having called the tool.
        """
        for r in reversed(self.executed):
            if r.tool == tool and (r.ok or not ok_only):
                return r
        return None

    def succeeded(self, tool: str) -> bool:
        return self.result_for(tool) is not None


@dataclass
class AgentDecision:
    """The structured output. Deliberately not reasoning prose."""

    intent: str
    actions_taken: list[str] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    resolution_status: str = Resolution.ESCALATED.value
    escalation_required: bool = False
    escalation_reason: str | None = None

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    # execution metadata
    tools_considered: list[str] = field(default_factory=list)
    tools_skipped: list[dict[str, str]] = field(default_factory=list)
    tier3_blocked: list[str] = field(default_factory=list)
    stop_reason: str = ""
    steps: int = 0
    # Auditable trajectory: which tool ran, with which arguments, and what
    # came back. `actions_taken` alone cannot answer "did it extract the order
    # id correctly", which is the most common tool-calling failure in practice.
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Guardrail findings. Recorded on every request, not only when they fire,
    # so "nothing fired" is distinguishable from "nothing was checked".
    guardrails: dict[str, Any] = field(default_factory=dict)
    # Retrieval strength, surfaced so the analytics layer can distinguish
    # "no documents matched" from "documents matched but the answer was wrong".
    max_bm25: float = 0.0
    n_chunks: int = 0
    retrieval_failed: bool = False
    has_image: bool = False
    image_contributed: bool = False
    image_error_code: str | None = None
    # Understanding signals, carried onto the decision so the analytics layer
    # can report sentiment and urgency distributions without re-running the
    # classifier over stored text.
    intent_margin: float = 0.0
    sentiment: str = ""
    urgency: str = ""
    replans: int = 0
    latency_ms: float = 0.0
    ticket_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        bits = [
            f"intent={self.intent}",
            f"status={self.resolution_status}",
            f"actions={len(self.actions_taken)}",
            f"conf={self.confidence:.2f}",
            f"{self.latency_ms:.0f}ms",
        ]
        if self.escalation_required:
            bits.append(f"ESCALATED({self.escalation_reason})")
        return " | ".join(bits)


MAX_STEPS = 8
MAX_FAILURES = 2
MIN_BM25_FOR_ANSWER = 7.0


class SupportAgent:
    """Plan, act, observe, decide."""

    def __init__(
        self,
        pipeline=None,
        understander: UnderstandingPipeline | None = None,
        max_steps: int = MAX_STEPS,
        vision_backend: str = "local_ocr",
    ):
        from src.rag.generator import build_pipeline

        self.pipeline = pipeline or build_pipeline()
        self.understander = understander or UnderstandingPipeline.load()
        self.max_steps = max_steps
        self.vision_backend = vision_backend

    # =================================================================
    def handle(
        self,
        text: str,
        image_path: str | None = None,
        customer_id: str | None = None,
        order_id: str | None = None,
    ) -> AgentDecision:
        t0 = time.perf_counter()
        state = AgentState(text=text, image_path=image_path,
                           customer_id=customer_id, order_id=order_id)
        safety = SafetyRecord()

        # ---- 0. input guardrails --------------------------------------
        # Runs before understanding, retrieval or tools. An input designed to
        # manipulate the system should not reach the system.
        safety.input = ENGINE.screen_input(text)
        if safety.input.blocked:
            return self._guardrail_stop(safety, t0, escalate=False)
        if safety.input.must_escalate:
            return self._guardrail_stop(safety, t0, escalate=True, state=state)

        # ---- 1. understand -------------------------------------------
        state.understanding = self.understander.understand(text)
        if not state.order_id and state.understanding.entities.order_ids:
            state.order_id = state.understanding.entities.order_ids[0]

        # ---- 2. plan --------------------------------------------------
        state.plan = build_plan(
            state.understanding, image_path=image_path,
            known_order_id=state.order_id,
        )

        # ---- 3. execute ------------------------------------------------
        self._execute(state)

        # ---- 4. observe and re-plan once -------------------------------
        self._maybe_replan(state)

        # ---- 5. decide --------------------------------------------------
        return self._decide(state, t0)

    # =================================================================
    def _execute(self, state: AgentState) -> None:
        """Run planned steps, skipping any whose precondition failed."""
        for step in state.plan.steps:
            if len(state.executed) >= self.max_steps:
                state.stop_reason = StopReason.MAX_STEPS
                return
            if state.failures >= MAX_FAILURES:
                state.stop_reason = StopReason.TOOL_FAILURES
                return

            unmet = [r for r in step.requires if not state.succeeded(r)]
            if unmet:
                state.skipped.append(
                    (step.tool, f"precondition not met: {', '.join(unmet)}")
                )
                continue

            result = call_tool(step.tool, **step.args)
            state.executed.append(result)

            if not result.ok:
                if result.status in (ToolStatus.NOT_FOUND, ToolStatus.ERROR):
                    state.failures += 1
                continue

            self._absorb(state, result)

        state.stop_reason = state.stop_reason or StopReason.PLAN_COMPLETE

    # -----------------------------------------------------------------
    def _absorb(self, state: AgentState, result: ToolResult) -> None:
        """Fold a tool result into state. This is what makes it an agent:
        the result of one call changes what the next call knows."""
        if result.tool == "search_knowledge_base":
            state.knowledge_chunks = result.data.get("chunks", [])
            state.max_bm25 = max(state.max_bm25, result.data.get("max_bm25", 0.0))
            state.has_conflict = state.has_conflict or result.data.get(
                "has_conflict", False)

        elif result.tool == "analyze_screenshot":
            state.image_analysis = result.data
            # A code read from the image may be the only usable entity in the
            # request - promote it so downstream retrieval can use it.
            if result.data.get("error_code") and \
                    result.data.get("evidence", {}).get("error_code") == "visible":
                state.understanding.entities.error_codes.insert(
                    0, result.data["error_code"])

        elif result.tool == "get_order":
            state.order_id = result.data.get("order_id", state.order_id)
            state.region = result.data.get("region", state.region)

    # -----------------------------------------------------------------
    def _maybe_replan(self, state: AgentState) -> None:
        """One re-plan, when a tool revealed something the original plan
        could not have known. Capped at one so the agent cannot loop."""
        if state.replans >= 1:
            return

        # A screenshot supplied an error code, and no knowledge search ran with it.
        code = None
        if state.image_analysis and state.image_analysis.get("error_code"):
            ev = state.image_analysis.get("evidence", {}).get("error_code")
            if ev == "visible":
                code = state.image_analysis["error_code"]

        if code and not state.succeeded("search_knowledge_base"):
            state.replans += 1
            result = call_tool(
                "search_knowledge_base",
                query=f"{state.text} {code}",
                region=state.region or "",
            )
            state.executed.append(result)
            if result.ok:
                self._absorb(state, result)
            return

        # The knowledge search was weak but an image code exists: search again
        # with the code, which is exactly the Phase 8 finding.
        if code and state.max_bm25 < MIN_BM25_FOR_ANSWER:
            state.replans += 1
            result = call_tool("search_knowledge_base",
                               query=f"{state.text} {code}")
            state.executed.append(result)
            if result.ok:
                self._absorb(state, result)

    # =================================================================
    def _guardrail_stop(self, safety: SafetyRecord, t0: float,
                        escalate: bool, state: AgentState | None = None
                        ) -> AgentDecision:
        """Terminate on an input-stage guardrail finding.

        A BLOCK refuses without creating a case: a prompt-injection attempt is
        not a support request, and queueing it wastes an agent's time. An
        ESCALATE hands over, because the customer may have a real problem
        underneath - an account change request is genuine, it just cannot be
        actioned in chat.
        """
        finding = safety.primary()
        d = AgentDecision(
            intent="guardrail",
            actions_taken=[],
            resolution_status=(Resolution.ESCALATED.value if escalate
                               else Resolution.REFUSED.value),
            escalation_required=escalate,
            escalation_reason=finding.rule if finding else "guardrail",
            answer=(finding.customer_message if finding and finding.customer_message
                    else "I can't help with that request."),
            confidence=0.0,
            stop_reason=StopReason.HARD_ESCALATION.value,
            guardrails={"input": safety.input.to_dict() if safety.input else {}},
        )
        if escalate and state is not None:
            rec = call_tool(
                "escalate_to_human",
                reason=d.escalation_reason,
                context=ENGINE.redact(state.text)[:400],
                priority="high",
            )
            if rec.ok:
                d.actions_taken = ["escalate_to_human"]
                d.ticket_id = rec.data.get("escalation_id") or rec.data.get("ticket_id")
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    def _decide(self, state: AgentState, t0: float) -> AgentDecision:
        u, plan = state.understanding, state.plan

        d = AgentDecision(
            intent=plan.intent,
            actions_taken=state.tool_names(),
            tools_considered=plan.tool_names,
            tools_skipped=[{"tool": t, "reason": r} for t, r in state.skipped],
            missing_information=list(plan.missing_info),
            stop_reason=(state.stop_reason or StopReason.PLAN_COMPLETE).value,
            steps=len(state.executed),
            replans=state.replans,
            trajectory=[
                {
                    "tool": r.tool,
                    "args": {k: v for k, v in (getattr(r, "args", {}) or {}).items()
                             if v not in (None, "")},
                    "status": r.status.value if hasattr(r.status, "value")
                    else str(r.status),
                }
                for r in state.executed
            ],
        )
        d.evidence_used = self._evidence_labels(state)
        d.max_bm25 = round(state.max_bm25, 3)
        if state.understanding is not None:
            d.intent_margin = round(state.understanding.intent_margin, 4)
            d.sentiment = state.understanding.sentiment
            d.urgency = state.understanding.urgency

        # Image outcome, surfaced for analytics. The screenshot tool firing is
        # not the same as the screenshot being useful, and the difference is
        # exactly what the vision-usage metric needs to report.
        d.has_image = bool(state.image_path)
        img = state.result_for("analyze_screenshot")
        if img is not None and img.ok:
            d.image_error_code = img.data.get("error_code")
            d.image_contributed = bool(
                img.data.get("is_useful") or img.data.get("error_code"))
        d.n_chunks = len(state.knowledge_chunks)
        # A retrieval failure is specifically: the knowledge base was consulted
        # and returned nothing usable. Not consulting it at all is a different
        # thing, and conflating them would inflate the failure rate.
        kb_called = state.result_for("search_knowledge_base") is not None
        d.retrieval_failed = bool(kb_called and state.max_bm25 < 7.0)

        # Surface any record the agent created, whatever path produced it, so
        # the customer gets a reference they can quote back.
        for tool in ("create_support_ticket", "escalate_to_human"):
            rec = state.result_for(tool)
            if rec is not None and rec.ok:
                d.ticket_id = (rec.data.get("ticket_id")
                               or rec.data.get("escalation_id") or d.ticket_id)

        # ---- hard escalation --------------------------------------------
        # A mutating action escalates so a human can approve it. But if policy
        # already says the action is not available - the return window closed -
        # there is nothing to approve. Escalating an ineligible refund wastes a
        # human's time and leaves the customer waiting for a "no" that the
        # agent can already give them, with the citation attached.
        policy_result = state.result_for("check_policy")
        blocked_by_policy = (
            plan.escalation_reason == "mutating_action_requires_approval"
            and policy_result is not None
            and policy_result.data.get("eligibility") in ("expired", "not_applicable")
        )
        if blocked_by_policy:
            plan.escalate_immediately = False
            d.caveats.append(
                "Outside the policy window, so no approval is required - the "
                "outcome is determined by policy."
            )

        if plan.escalate_immediately:
            return self._escalate(state, d, plan.escalation_reason, t0)

        # ---- missing information -----------------------------------------
        if plan.missing_info and not state.knowledge_chunks:
            d.resolution_status = Resolution.NEEDS_INFORMATION.value
            d.answer = self._ask_for(plan.missing_info)
            d.confidence = 0.0
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        # ---- out of scope --------------------------------------------------
        if plan.intent == "out_of_scope" and not state.knowledge_chunks:
            d.resolution_status = Resolution.REFUSED.value
            d.answer = (
                "I can help with orders, returns, warranty, shipping, payments "
                "and technical problems with Pacify products. I'm not able to "
                "help with that one."
            )
            d.confidence = 0.6
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        # ---- conflicting evidence -------------------------------------------
        # A genuine contradiction between two CURRENT documents must escalate
        # (DEFECT-01: the 30-day guarantee versus the 14-day window). But a
        # regional addendum is not a contradiction when the customer's region
        # is known - the EU rules simply do not apply to an Indian order.
        #
        # Phase 7's detector is region-agnostic because it has no order context.
        # The agent does, so it can resolve what retrieval alone cannot. Where
        # the region is unknown the conflict stands, because guessing which
        # jurisdiction applies is exactly the wrong call.
        if state.has_conflict and self._conflict_is_regional_only(state):
            state.has_conflict = False
            d.caveats.append(
                "A regional variant was retrieved but does not apply to this "
                "order's region."
            )

        if state.has_conflict:
            return self._escalate(state, d, "conflicting_documentation", t0)

        # ---- generate ---------------------------------------------------------
        rag = self._generate(state)
        d.answer = rag.response.answer
        d.citations = [str(c) for c in rag.response.citations]
        d.confidence = rag.response.confidence

        # tool-derived facts are deterministic, so their presence raises
        # confidence in a way a fluent sentence does not
        if state.succeeded("check_policy"):
            d.confidence = min(0.95, d.confidence + 0.15)

        # ---- escalation gate ---------------------------------------------
        # The RAG abstention gate thresholds on RETRIEVAL score. That is the
        # right signal when policy documentation is the only evidence, and the
        # wrong signal in two situations the agent creates:
        #
        #   1. A tool already answered the question. "Where is my order" is
        #      answered by get_order; no policy document discusses that parcel.
        #      Escalating there would abandon a lookup the agent completed.
        #
        #   2. Retrieval found the right passage but scored it low. BM25 gives
        #      low IDF to terms that appear in every document - "return" and
        #      "policy" are cross-referenced throughout the corpus, so
        #      "what is your return policy" scores 5.2 despite matching well.
        #      This is Phase 7's documented 13.3% false-abstention rate.
        #
        # So the agent overrides the RAG gate when it holds independent
        # evidence, and defers to it otherwise.
        tool_facts = [
            r for r in state.executed
            if r.ok and r.data and r.tool not in
            ("search_knowledge_base", "analyze_screenshot")
        ]
        # Retrieval ALWAYS returns its top-k, so the presence of chunks proves
        # nothing. Quality is the signal, and BM25 is the measured one:
        # Phase 7 found it separates answerable from unanswerable roughly twice
        # as well as cosine (13.1 vs 6.2 median).
        kb = state.result_for("search_knowledge_base")
        # 7.0 is Phase 7's calibrated threshold. Re-swept here against the
        # full 120 answerable / 40 unanswerable sets before reusing it:
        #
        #     threshold   answerable kept   unanswerable blocked   balanced
        #         5.0            0.958              0.175            0.168
        #         7.0            0.842              0.700            0.589
        #         7.5            0.817              0.750            0.613
        #
        # Lowering it to rescue individual queries would have been overfitting
        # to a handful of cases at a large cost in refusals.
        kb_passages = bool(
            kb and kb.ok
            and kb.data.get("chunks")
            and kb.data.get("max_bm25", 0) >= 7.0
        )

        rag_wants_escalation = rag.trace.decision == "abstain" or rag.escalated
        has_own_evidence = bool(tool_facts) or kb_passages

        # An order reference that matches nothing is not a failure needing a
        # human - it is almost always a typo. Asking for a correction resolves
        # far more of these than a handoff does, and it is what a person would
        # do first.
        order_lookup = state.result_for("get_order", ok_only=False)
        if (order_lookup is not None and not order_lookup.ok
                and not tool_facts and not kb_passages):
            d.resolution_status = Resolution.NEEDS_INFORMATION.value
            d.missing_information = ["order_id"]
            d.answer = (
                f"I couldn't find an order matching that reference. Could you "
                f"check it and send it again? Order references look like "
                f"PAC-2026-12345."
            )
            d.confidence = 0.0
            d.stop_reason = StopReason.MISSING_PRECONDITION.value
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        # 2. A Tier-2 record IS the resolution. "Create a ticket" is completed
        # by creating the ticket; it does not additionally require a policy
        # document to justify it.
        created = state.result_for("create_support_ticket")
        if created is not None and created.ok and not has_own_evidence:
            d.resolution_status = Resolution.RESOLVED.value
            d.ticket_id = created.data.get("ticket_id")
            d.answer = (
                f"I've opened ticket {d.ticket_id} for this. A colleague will "
                f"respond within {created.data.get('first_response_target', '24 hours')}."
            )
            d.confidence = 0.70
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        if rag_wants_escalation and not has_own_evidence:
            reason = (
                "no_supporting_documentation" if rag.trace.decision == "abstain"
                else (rag.response.escalation_reason or "generation_gate")
            )
            return self._escalate(state, d, reason, t0,
                                  answer=rag.response.answer)

        # A grounding failure is never overridden - a fabricated figure is the
        # one thing tool evidence cannot excuse.
        if not rag.trace.is_grounded:
            return self._escalate(state, d, "failed_grounding_check", t0,
                                  answer=rag.response.answer)

        if rag_wants_escalation and has_own_evidence:
            if tool_facts and not kb_passages:
                d.caveats.append(
                    "Answered from order records; no policy document was needed."
                )
            else:
                d.caveats.append(
                    "Supporting documentation matched weakly; treat as provisional."
                )
            d.confidence = max(d.confidence, 0.45)

        # ---- output guardrails -------------------------------------------
        # An independent check on what the model actually said. The RAG layer
        # runs its own grounding check; this one also catches forbidden
        # commitments and internal leakage, which grounding does not.
        # Validate against what the MODEL was given, not what the agent's own
        # KB tool retrieved. The agent and the RAG pipeline currently retrieve
        # independently, so their chunk sets can differ - checking the answer
        # against the agent's set flagged valid citations as fabricated.
        #
        # (That double retrieval is itself a design smell, noted in the report:
        # one retrieval passed through to generation would be better.)
        model_citations = list(rag.context.citations) + state.available_citations()
        out_verdict = ENGINE.screen_output(
            rag.response.answer,
            context=(rag.context.text or "") + " " + state.context_text(),
            cited=[str(c) for c in rag.response.citations],
            available_citations=model_citations,
            is_abstention=rag.response.is_abstention,
        )
        d.guardrails = {"output": out_verdict.to_dict()}
        if out_verdict.must_escalate:
            reason = out_verdict.primary.rule
            return self._escalate(state, d, reason, t0,
                                  answer=out_verdict.customer_message()
                                  or rag.response.answer)

        d.caveats = list(rag.abstention.caveats) + out_verdict.caveats()
        d.resolution_status = (
            Resolution.RESOLVED_WITH_CAVEAT.value if d.caveats
            else Resolution.RESOLVED.value
        )
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    # -----------------------------------------------------------------
    @staticmethod
    def _conflict_is_regional_only(state: AgentState) -> bool:
        """True when the only source of conflict is an inapplicable region.

        Two independent ways a regional flag turns out to be noise:

        1. RELEVANCE - the addendum was retrieved but ranks below the top 3,
           meaning the question is not about returns at all. "What does
           ERR-DP-0x004 mean?" is a display-cable question; the EU withdrawal
           addendum appearing at rank 5 says nothing about it.

        2. JURISDICTION - the customer's region is known, so the addendum
           either governs (EU) or does not apply (elsewhere). Either way the
           ambiguity is resolved. Only an UNKNOWN region leaves a real
           conflict, because guessing which jurisdiction applies is exactly the
           wrong call.
        """
        kb = state.result_for("search_knowledge_base")
        if not kb:
            return False
        chunks = kb.data.get("chunks", [])
        docs = {c.get("doc") for c in chunks}
        regional_docs = {"eu_regional_addendum"}

        if not (docs & regional_docs):
            return False        # the conflict came from somewhere else

        # Anything other than a regional document disagreeing is a real
        # conflict and must not be resolved here.
        versions = {
            c.get("version", "current") for c in chunks
            if c.get("doc") not in regional_docs
        }
        if len(versions) > 1:
            return False

        # 1. relevance
        regional_ranks = [
            i for i, c in enumerate(chunks) if c.get("doc") in regional_docs
        ]
        if regional_ranks and min(regional_ranks) >= 3:
            return True

        # 2. jurisdiction
        order = state.result_for("get_order")
        region = (order.data.get("region") if order else None) or state.region
        if region is None:
            from src.rag.routing import detect_region

            region = detect_region(state.text.lower())
        return region is not None

    def _generate(self, state: AgentState):
        """Feed tool results and image evidence into the RAG pipeline."""
        blocks = []
        if state.image_analysis:
            from src.multimodal.vision import ImageAnalysis  # noqa: F401
            img = state.result_for("analyze_screenshot")
            if img:
                blocks.append(self._image_block(img.data))
        for r in state.executed:
            if r.ok and r.tool not in ("search_knowledge_base", "analyze_screenshot"):
                blocks.append(r.as_evidence())

        query = state.text
        codes = state.understanding.entities.error_codes
        if codes:
            query = f"{state.text} {codes[0]}"

        return self.pipeline.answer(
            query,
            _display_question=state.text,
            _image_evidence="\n\n".join(blocks),
        )

    @staticmethod
    def _image_block(data: dict[str, Any]) -> str:
        ev = data.get("evidence", {})
        lines = ["[IMAGE EVIDENCE]"]
        for f in ("image_type", "error_code", "visible_error", "ui_context"):
            if data.get(f):
                lines.append(f"  {f}: {data[f]} ({ev.get(f, 'unknown')})")
        lines.append("  Treat 'inferred' as uncertain and 'unknown' as absent.")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    def _escalate(self, state: AgentState, d: AgentDecision, reason: str,
                  t0: float, answer: str = "") -> AgentDecision:
        """Hand off with context. Creating the ticket is a tier-2 action and
        is performed autonomously - losing the context would be worse."""
        context = self._handoff_context(state)
        result = call_tool(
            "escalate_to_human", reason=reason, context=context,
            intent=d.intent, order_id=state.order_id or "",
            customer_id=state.customer_id or "",
            priority="high" if state.understanding.urgency == "high" else "medium",
        )
        if result.ok:
            d.ticket_id = result.data.get("ticket_id")
            if "escalate_to_human" not in d.actions_taken:
                d.actions_taken.append("escalate_to_human")
                state.executed.append(result)

        d.escalation_required = True
        d.escalation_reason = reason
        d.resolution_status = Resolution.ESCALATED.value
        d.confidence = min(d.confidence, 0.4)
        d.answer = answer or self._escalation_message(reason, d.ticket_id)
        d.steps = len(state.executed)
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    @staticmethod
    def _handoff_context(state: AgentState) -> str:
        bits = [f"intent={state.understanding.intent}",
                f"sentiment={state.understanding.sentiment}",
                f"urgency={state.understanding.urgency}"]
        if state.order_id:
            bits.append(f"order={state.order_id}")
        ok = [r.tool for r in state.executed if r.ok]
        if ok:
            bits.append(f"checked={','.join(ok)}")
        pol = state.result_for("check_policy")
        if pol:
            bits.append(f"policy={pol.data.get('eligibility') or pol.data.get('state')}")
        return " | ".join(bits)

    @staticmethod
    def _escalation_message(reason: str, ticket_id: str | None) -> str:
        base = {
            "legal_or_chargeback_threat":
                "I've passed this straight to a colleague who can deal with it "
                "properly.",
            "identity_verification_required":
                "For account changes I need to verify your identity, which I "
                "can't do here. A colleague will pick this up.",
            "mutating_action_requires_approval":
                "I've gathered everything needed and passed it to a colleague "
                "who can authorise this.",
            "conflicting_documentation":
                "Our documentation gives two different answers on this, so I "
                "don't want to quote a figure that might be wrong. A colleague "
                "will confirm which applies.",
            "no_supporting_documentation":
                "I don't have documentation covering that. I've passed it to a "
                "colleague.",
            "relationship_issue_requires_human":
                "I'm sorry this has been frustrating. I've passed it to a "
                "colleague rather than give you another policy quote.",
        }.get(reason, "I've passed this to a colleague who can help.")
        return f"{base} Reference: {ticket_id}." if ticket_id else base

    @staticmethod
    def _ask_for(missing: list[str]) -> str:
        labels = {"order_id": "your order reference (it looks like PAC-2026-12345)",
                  "email": "the email address on your account"}
        asked = " and ".join(labels.get(m, m) for m in missing)
        return f"I can look that up - could you give me {asked}?"

    @staticmethod
    def _evidence_labels(state: AgentState) -> list[str]:
        ev = []
        if state.knowledge_chunks:
            ev.append(f"knowledge_base({len(state.knowledge_chunks)} chunks)")
        if state.image_analysis and state.image_analysis.get("is_useful"):
            ev.append("screenshot")
        for r in state.executed:
            if r.ok and r.tool not in ("search_knowledge_base",
                                       "analyze_screenshot",
                                       "escalate_to_human"):
                ev.append(r.tool)
        return ev


if __name__ == "__main__":
    import json

    agent = SupportAgent()
    for text, img in [
        ("What is your return policy?", None),
        ("Where is my order PAC-2026-12345?", None),
        ("I want to return order PAC-2026-12345 and get a refund", None),
        ("I'm taking you to consumer court over this", None),
        ("Do you offer student discounts?", None),
    ]:
        d = agent.handle(text, image_path=img)
        print(f"\n{'=' * 74}\n{text}")
        print(json.dumps({k: v for k, v in d.to_dict().items()
                          if k not in ("answer",)}, indent=2, default=str)[:900])
        print(f"ANSWER: {d.answer[:180]}")
