"""Output schema for the understanding layer.

`Understanding` is the single object the router and agent consume. Keeping it
a typed dataclass rather than a dict means downstream code cannot silently
depend on a key that stops being produced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Entities:
    """Structured items extracted from the message."""

    order_ids: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(asdict(self).values())


@dataclass
class Understanding:
    """Everything the fast local layer knows about one message.

    Produced before any LLM call, in roughly 10ms. The router uses `intent`
    and `entities`; the escalation policy uses `sentiment` and `urgency`.
    """

    text: str
    intent: str
    intent_confidence: float
    intent_margin: float                       # gap to the runner-up class
    intent_top3: list[tuple[str, float]] = field(default_factory=list)

    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    sentiment_explain: str = ""

    urgency: str = "low"
    urgency_score: float = 0.0
    urgency_explain: str = ""

    entities: Entities = field(default_factory=Entities)

    is_multi_intent: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entities"] = asdict(self.entities)
        return d

    def summary(self) -> str:
        parts = [
            f"intent={self.intent} ({self.intent_confidence:.2f}, margin {self.intent_margin:.2f})",
            f"sentiment={self.sentiment}",
            f"urgency={self.urgency}",
        ]
        if self.entities.order_ids:
            parts.append(f"order={self.entities.order_ids[0]}")
        if self.entities.error_codes:
            parts.append(f"code={self.entities.error_codes[0]}")
        if self.is_multi_intent:
            parts.append("MULTI-INTENT")
        return " | ".join(parts)
