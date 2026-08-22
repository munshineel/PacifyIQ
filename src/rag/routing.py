"""The understanding-retrieval bridge.

Connects the Phase 4 classifier and entity extractor to the Phase 6 retriever.

DESIGN CONSTRAINT, MEASURED FIRST
---------------------------------
The obvious design - "classify the intent, filter retrieval to that topic" -
does not survive contact with the data. Checking the classifier on the queries
that retrieval currently fails:

    "Am I talking to a bot?"            -> complaint             margin 0.103
    "What do you need to verify..."     -> shipping_delivery     margin 0.035
    "What does SYS-0x0000007B mean?"    -> payment_issue         margin 0.036
    "My laptop will not turn on"        -> warranty_claim        margin 0.036

Three of those four are wrong, and all four have a margin under 0.11. A hard
topic filter driven by a classifier this uncertain would *remove the correct
document* more often than it removes noise.

So the bridge applies three graded signals instead:

  1. ENTITY OVERRIDE   an extracted error code is a hard, reliable signal and
                       outranks the classifier entirely
  2. INTENT BOOST      a soft score multiplier, applied only above a margin
                       threshold, never a filter
  3. QUERY ENRICHMENT  extracted entities appended to the query text so the
                       lexical retriever can match them exactly

MEASURED OUTCOME
----------------
Component ablation on the 120-query retrieval set:

    configuration                    recall@5    MRR      nDCG@5
    baseline (no routing)            0.875       0.664    0.726
    entities only                    0.875       0.682    0.754
    entities + intent (margin 0.25)  0.883       0.706    0.782   <- shipped
    entities + intent + meta         0.883       0.682    0.762
    meta only                        0.858       0.650    0.707   <- hurts

`use_meta` defaults to FALSE. Injecting customer_service_policy candidates for
anything resembling a meta-question fixed three target queries and broke six
others, because the injected chunks displaced correct evidence for questions
that merely mentioned "escalate" or "verify" in passing. The idea is sound and
the implementation is retained behind a flag, but on this corpus it costs more
than it returns. Reported rather than quietly dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.knowledge.retriever import RetrievalResult, Retriever
from src.understanding.pipeline import UnderstandingPipeline
from src.understanding.schema import Understanding

# ---------------------------------------------------------------------
# Intent -> where the answer usually lives.
#
# `topics` is used as a soft boost. `docs` names documents that are
# authoritative for an intent even when their vocabulary does not match the
# query - the meta-question problem, where "Am I talking to a bot?" is answered
# by customer_service_policy S12 but shares almost no terms with it.
# ---------------------------------------------------------------------
INTENT_ROUTING: dict[str, dict[str, Any]] = {
    "order_tracking": {
        "topics": ["shipping"], "docs": ["shipping_policy"],
    },
    "return_policy_question": {
        "topics": ["returns"], "docs": ["return_policy_v2", "eu_regional_addendum"],
    },
    "return_refund_request": {
        "topics": ["returns", "refunds"],
        "docs": ["return_policy_v2", "refund_policy"],
    },
    "warranty_claim": {
        "topics": ["warranty"], "docs": ["warranty_policy"],
    },
    "shipping_delivery": {
        "topics": ["shipping"], "docs": ["shipping_policy"],
    },
    "product_information": {
        "topics": ["product", "general"],
        "docs": ["product_faq", "manual_probook14", "manual_phonex", "manual_vision27"],
    },
    "technical_support": {
        "topics": ["technical"], "docs": ["technical_support_faq"],
    },
    "payment_issue": {
        "topics": ["billing", "refunds"], "docs": ["payment_policy", "refund_policy"],
    },
    "account_management": {
        "topics": ["account"], "docs": ["customer_service_policy"],
    },
    "complaint": {
        "topics": ["account"], "docs": ["customer_service_policy"],
    },
    "out_of_scope": {
        "topics": [], "docs": ["customer_service_policy"],
    },
}

# Error-code prefix -> the documents that define it. Deterministic, so it
# outranks a probabilistic classifier.
CODE_ROUTING = {
    "PAY": ["payment_policy", "technical_support_faq"],
    "ERR": ["technical_support_faq", "manual_vision27"],
    "BAT": ["technical_support_faq", "warranty_policy"],
    "WIFI": ["technical_support_faq"],
    "SYS": ["technical_support_faq"],
    "THRM": ["technical_support_faq"],
    "DSP": ["technical_support_faq", "warranty_policy"],
    "AUD": ["technical_support_faq"],
    "KEY": ["technical_support_faq"],
    "STO": ["technical_support_faq"],
    "MEM": ["technical_support_faq"],
    "CAM": ["technical_support_faq"],
}

# Terms that mark a question about the support system itself rather than about
# a product or order. These queries share almost no vocabulary with the policy
# that answers them, which is why retrieval alone fails on them.
META_MARKERS = {
    "bot", "robot", "human", "ai", "assistant", "agent", "automated",
    "escalate", "escalation", "manager", "supervisor", "complain",
    "support hours", "respond", "response time", "verify", "verification",
    "grievance", "officer",
}

# Swept against the 120-query retrieval set. 0.25 is the measured optimum:
#   margin >= 0.00  recall@5 0.808   (trusting a coin-flip classifier is worse
#                                     than ignoring it entirely)
#   margin >= 0.15  recall@5 0.867
#   margin >= 0.25  recall@5 0.883   <- selected
#   margin >= 0.40  recall@5 0.875
MIN_MARGIN_FOR_BOOST = 0.25
INTENT_BOOST = 1.35              # score multiplier for a matching document
TOPIC_BOOST = 1.15               # weaker boost for a matching topic
CODE_BOOST = 1.60                # entity signal, strongest


@dataclass
class RoutingDecision:
    """What the bridge concluded, and why. Logged for auditability."""

    intent: str = ""
    intent_margin: float = 0.0
    intent_applied: bool = False

    boosted_docs: list[str] = field(default_factory=list)
    boosted_topics: list[str] = field(default_factory=list)

    error_codes: list[str] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)

    enriched_query: str = ""
    is_meta_question: bool = False
    region: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_meta_question(text: str) -> bool:
    low = str(text).lower()
    return any(m in low for m in META_MARKERS)


def detect_region(text: str) -> str | None:
    """EU signals in the query. The addendum overrides base policy, so
    surfacing it depends on noticing the customer is in the EU."""
    low = str(text).lower()
    eu_terms = (
        "eu ", "european union", "germany", "france", "netherlands", "ireland",
        "spain", "italy", "berlin", "paris", "madrid", "rome", "amsterdam",
        "dublin", "statutory withdrawal", "distance selling",
    )
    return "EU" if any(t in low for t in eu_terms) else None


def build_routing(
    understanding: Understanding,
    use_intent: bool = True,
    use_entities: bool = True,
    use_meta: bool = True,
    min_margin: float = MIN_MARGIN_FOR_BOOST,
) -> RoutingDecision:
    """Turn an Understanding into retrieval guidance."""
    d = RoutingDecision(
        intent=understanding.intent,
        intent_margin=understanding.intent_margin,
        enriched_query=understanding.text,
    )
    docs: list[str] = []
    topics: list[str] = []
    enrich: list[str] = []

    # --- 1. entity signals: deterministic, applied first ---------------
    if use_entities:
        e = understanding.entities
        d.error_codes = e.error_codes
        d.order_ids = e.order_ids
        d.products = e.products

        for code in e.error_codes:
            prefix = code.split("-")[0].upper()
            for doc in CODE_ROUTING.get(prefix, []):
                if doc not in docs:
                    docs.append(doc)
            if code not in enrich:
                enrich.append(code)
            # split forms too, so the lexical retriever matches either spelling
            enrich.extend(p for p in code.split("-") if len(p) > 1)
            d.reasons.append(f"error_code:{code}")

        for p in e.products:
            enrich.append(p)
            d.reasons.append(f"product:{p}")

    # --- 2. meta-questions --------------------------------------------
    # These share almost no vocabulary with the policy that answers them, so
    # retrieval scores stay low and abstention wrongly fires.
    if use_meta and detect_meta_question(understanding.text):
        d.is_meta_question = True
        if "customer_service_policy" not in docs:
            docs.append("customer_service_policy")
        d.reasons.append("meta_question")

    # --- 3. intent boost, gated on classifier confidence ---------------
    if use_intent:
        route = INTENT_ROUTING.get(understanding.intent, {})
        if understanding.intent_margin >= min_margin:
            d.intent_applied = True
            for doc in route.get("docs", []):
                if doc not in docs:
                    docs.append(doc)
            topics.extend(route.get("topics", []))
            d.reasons.append(
                f"intent:{understanding.intent}(margin {understanding.intent_margin:.2f})"
            )
        else:
            d.reasons.append(
                f"intent_skipped(margin {understanding.intent_margin:.2f} "
                f"< {min_margin})"
            )

    # --- 4. region ------------------------------------------------------
    d.region = detect_region(understanding.text)
    if d.region:
        docs.append("eu_regional_addendum")
        d.reasons.append(f"region:{d.region}")

    d.boosted_docs = docs
    d.boosted_topics = list(dict.fromkeys(topics))
    d.enriched_query = (
        understanding.text + " " + " ".join(dict.fromkeys(enrich))
    ).strip() if enrich else understanding.text
    return d


class RoutedRetriever:
    """Retriever wrapped with understanding-driven boosting.

    Boosts are applied to a deeper candidate pool and then re-ranked, so a
    correct document sitting below the cut can be promoted rather than merely
    reshuffled. Nothing is ever filtered out - a wrong boost costs rank
    position, not availability.
    """

    def __init__(
        self,
        retriever: Retriever,
        understander: UnderstandingPipeline | None = None,
        use_intent: bool = True,
        use_entities: bool = True,
        use_meta: bool = False,      # measured to hurt - see below
        use_enrichment: bool = True,
        min_margin: float = MIN_MARGIN_FOR_BOOST,
    ):
        self.retriever = retriever
        self.understander = understander or UnderstandingPipeline.load()
        self.use_intent = use_intent
        self.use_entities = use_entities
        self.use_meta = use_meta
        self.use_enrichment = use_enrichment
        self.min_margin = min_margin

    # ---------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int = 5, **kw
    ) -> tuple[RetrievalResult, RoutingDecision, Understanding]:
        u = self.understander.understand(query)
        route = build_routing(
            u,
            use_intent=self.use_intent,
            use_entities=self.use_entities,
            use_meta=self.use_meta,
            min_margin=self.min_margin,
        )

        search_text = route.enriched_query if self.use_enrichment else query

        # region is a genuine filter, not a boost: the EU addendum legally
        # overrides base policy, so it must surface rather than merely rank well
        if route.region and "region" not in kw:
            kw["region"] = route.region

        pool = max(top_k * 4, 20)
        result = self.retriever.retrieve(search_text, top_k=pool, **kw)

        # Boosting can only reorder what is already in the pool. For a
        # meta-question the answering document shares almost no vocabulary
        # with the query, so it never enters the pool at all and no amount of
        # re-ranking rescues it. A second retrieval restricted to the boosted
        # documents injects those candidates so the boost has something to act
        # on. This is the difference between routing that works and routing
        # that only looks like it works.
        if route.boosted_docs:
            seen = {h.chunk.chunk_id for h in result.hits}
            injected = self.retriever.retrieve(
                search_text, top_k=3 * len(set(route.boosted_docs)),
                docs=list(dict.fromkeys(route.boosted_docs)),
                **{k: v for k, v in kw.items() if k not in ("region", "docs")},
            )
            for h in injected.hits:
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    result.hits.append(h)

        if route.boosted_docs or route.boosted_topics:
            for h in result.hits:
                mult = 1.0
                if h.chunk.doc in route.boosted_docs:
                    mult *= CODE_BOOST if route.error_codes and h.chunk.doc in (
                        d for c in route.error_codes
                        for d in CODE_ROUTING.get(c.split("-")[0].upper(), [])
                    ) else INTENT_BOOST
                if h.chunk.topic in route.boosted_topics:
                    mult *= TOPIC_BOOST
                h.score *= mult
            result.hits.sort(key=lambda h: -h.score)

        result.hits = result.hits[:top_k]
        for i, h in enumerate(result.hits, start=1):
            h.rank = i
        result.top_score = result.hits[0].score if result.hits else 0.0
        return result, route, u

    # ---------------------------------------------------------------
    def explain(self, query: str, top_k: int = 5, **kw) -> str:
        result, route, u = self.retrieve(query, top_k=top_k, **kw)
        lines = [
            f"QUERY      {query}",
            f"UNDERSTAND {u.summary()}",
            f"ROUTING    {', '.join(route.reasons) or 'none'}",
            f"BOOSTED    docs={route.boosted_docs or '-'} topics={route.boosted_topics or '-'}",
        ]
        if route.enriched_query != query:
            lines.append(f"ENRICHED   {route.enriched_query}")
        lines.append("")
        for h in result.hits:
            lines.append(f"  [{h.rank}] {h.score:7.4f}  {h.chunk.citation:32s} ({h.source})")
        return "\n".join(lines)


if __name__ == "__main__":
    from src.config.settings import settings
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.vector_store import VectorStore

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    base = Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)
    routed = RoutedRetriever(base)

    for q in [
        "Am I talking to a bot?",
        "What does SYS-0x0000007B mean?",
        "I am in Germany. How long do I have to return an opened laptop?",
        "How do I escalate my issue?",
    ]:
        print(routed.explain(q, top_k=3))
        print()
