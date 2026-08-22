"""LLM client.

Provider-agnostic interface with three backends:

    local        deterministic extractive generator, no network
    groq         hosted chat completion
    scripted     fixed responses, for testing failure paths

WHY A LOCAL BACKEND EXISTS. Everything in a RAG system except token generation
can be built and measured without a model: context assembly, citation
enforcement, abstention thresholds, contradiction detection, faithfulness
scoring. Building against an interface rather than an SDK means those parts are
testable offline and in CI, and swapping providers is a config change.

The `local` backend is not a stub. It is an extractive generator that selects
the sentences from retrieved context most relevant to the question and attaches
citations. That is a genuine RAG baseline - the kind of system that shipped
before instruction-tuned models - and it produces real numbers to compare an
LLM against.
"""
from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import settings


# =====================================================================
# Types
# =====================================================================

@dataclass
class Message:
    role: str          # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """One completion, with the accounting the trace log needs."""

    text: str
    backend: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "finish_reason": self.finish_reason,
        }


# =====================================================================
# Token counting
# =====================================================================

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return int(len(str(text).split()) * 1.3)


def count_messages(messages: list[Message]) -> int:
    """Approximate prompt size, including per-message role overhead."""
    return sum(count_tokens(m.content) + 4 for m in messages)


# =====================================================================
# Base
# =====================================================================

class LLMClient(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, messages: list[Message], temperature: float = 0.1,
                 max_tokens: int = 512, **kw) -> LLMResponse:
        ...

    def ask(self, system: str, user: str, **kw) -> LLMResponse:
        return self.complete(
            [Message("system", system), Message("user", user)], **kw
        )


# =====================================================================
# Local extractive generator
# =====================================================================

SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "of",
    "to", "in", "for", "on", "at", "by", "with", "from", "as", "that", "this",
    "it", "its", "i", "you", "my", "your", "we", "our", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "how", "what", "when",
    "where", "which", "who", "if", "not", "have", "has", "had", "there",
}


def _terms(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", str(text).lower())
        if w not in STOP and len(w) > 2
    }


class LocalExtractiveLLM(LLMClient):
    """Answers by selecting the most relevant sentences from the context.

    It cannot paraphrase or reason, which is the point: every word it emits
    came verbatim from a retrieved document, so faithfulness is guaranteed by
    construction. It gives the pipeline a floor to measure a real LLM against.
    """

    name = "local"

    def __init__(self, model: str = "extractive-v1", max_sentences: int = 4):
        self.model = model
        self.max_sentences = max_sentences

    def complete(self, messages: list[Message], temperature: float = 0.1,
                 max_tokens: int = 512, **kw) -> LLMResponse:
        t0 = time.perf_counter()
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        question = self._extract_question(user)
        context_blocks = self._extract_context(user)

        if not context_blocks:
            text = self._json_answer(
                "I don't have documentation covering that.", [], 0.0, True,
                "no_context",
            )
        else:
            text = self._answer(question, context_blocks)

        return LLMResponse(
            text=text,
            backend=self.name,
            model=self.model,
            prompt_tokens=count_messages(messages),
            completion_tokens=count_tokens(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ---------------------------------------------------------------
    @staticmethod
    def _extract_question(user: str) -> str:
        m = re.search(r"QUESTION:\s*(.+?)(?:\n\n|\Z)", user, re.S)
        return m.group(1).strip() if m else user.strip()

    @staticmethod
    def _extract_context(user: str) -> list[tuple[str, str]]:
        """Pull (citation, text) pairs out of the rendered prompt."""
        # Stop at the QUESTION marker, otherwise the question text is parsed
        # as part of the final context block and gets echoed into the answer.
        body = re.split(r"\n\s*QUESTION:", user)[0]
        blocks = []
        for m in re.finditer(
            r"\[(\d+)\]\s*SOURCE:\s*(.+?)\n(.*?)(?=\n\[\d+\]\s*SOURCE:|\Z)",
            body, re.S,
        ):
            blocks.append((m.group(2).strip(), m.group(3).strip()))
        return blocks

    # ---------------------------------------------------------------
    def _answer(self, question: str, blocks: list[tuple[str, str]]) -> str:
        q_terms = _terms(question)

        scored: list[tuple[float, str, str]] = []
        for citation, body in blocks:
            for sent in SENT_SPLIT.split(body):
                sent = " ".join(sent.split())
                if len(sent) < 25:
                    continue
                s_terms = _terms(sent)
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms)
                if overlap == 0:
                    continue
                # normalise so long sentences do not win on volume alone
                score = overlap / (len(q_terms) ** 0.5 * len(s_terms) ** 0.25)
                scored.append((score, sent, citation))

        scored.sort(key=lambda x: -x[0])
        picked = scored[: self.max_sentences]

        if not picked:
            return self._json_answer(
                "The retrieved documentation does not address that question.",
                [], 0.15, True, "no_relevant_sentence",
            )

        answer = " ".join(s for _, s, _ in picked)
        citations = list(dict.fromkeys(c for _, _, c in picked))
        confidence = min(0.85, 0.35 + 0.12 * len(picked))
        return self._json_answer(answer, citations, confidence, False, None)

    @staticmethod
    def _json_answer(answer: str, citations: list[str], confidence: float,
                     escalate: bool, reason: str | None) -> str:
        import json

        return json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "confidence": round(confidence, 2),
                "needs_escalation": escalate,
                "escalation_reason": reason,
            },
            ensure_ascii=False,
        )


# =====================================================================
# Groq
# =====================================================================

class GroqLLM(LLMClient):
    """Hosted chat completion. Requires PACIFYIQ_GROQ_API_KEY."""

    name = "groq"

    def __init__(self, model: str | None = None, max_retries: int | None = None,
                 timeout: int | None = None):
        self.model = model or settings.llm_model
        self.max_retries = max_retries or settings.llm_max_retries
        self.timeout = timeout or settings.llm_timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is None:
            key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "PACIFYIQ_GROQ_API_KEY is not set. Use the local backend:\n"
                    "  get_llm('local')"
                )
            from groq import Groq

            self._client = Groq(api_key=key, timeout=self.timeout)
        return self._client

    def complete(self, messages: list[Message], temperature: float = 0.1,
                 max_tokens: int = 512, json_mode: bool = True, **kw) -> LLMResponse:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = client.chat.completions.create(**payload)
                choice = r.choices[0]
                return LLMResponse(
                    text=choice.message.content or "",
                    backend=self.name,
                    model=self.model,
                    prompt_tokens=getattr(r.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(r.usage, "completion_tokens", 0),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    finish_reason=choice.finish_reason or "stop",
                    raw=r,
                )
            except Exception as e:  # network, rate limit, transient 5xx
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq failed after {self.max_retries} attempts: {last_error}")


# =====================================================================
# Scripted (testing)
# =====================================================================

class ScriptedLLM(LLMClient):
    """Returns queued responses in order. Used to test failure handling:
    malformed JSON, fabricated citations, refusals."""

    name = "scripted"

    def __init__(self, responses: list[str], model: str = "scripted"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], **kw) -> LLMResponse:
        self.calls.append(messages)
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(
            text=text, backend=self.name, model=self.model,
            prompt_tokens=count_messages(messages),
            completion_tokens=count_tokens(text),
        )


# =====================================================================
BACKENDS = {"local": LocalExtractiveLLM, "groq": GroqLLM, "scripted": ScriptedLLM}


def get_llm(backend: str = "local", **kw) -> LLMClient:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected {list(BACKENDS)}")
    return BACKENDS[backend](**kw)


def available_backends() -> dict[str, bool]:
    """Which backends can actually run right now."""
    return {
        "local": True,
        "scripted": True,
        "groq": bool(settings.groq_api_key),
    }


if __name__ == "__main__":
    print("available backends:", available_backends())
    llm = get_llm("local")
    prompt = """CONTEXT:

[1] SOURCE: POL-RET-002, p.1, S2
S2. Return windows S2.1 The applicable return window is determined by product
category and by the condition of the item. Opened consumer electronics may be
returned within 14 calendar days of delivery. Sealed items may be returned
within 30 calendar days.

QUESTION: How long do I have to return an opened laptop?"""
    r = llm.ask("You are a support assistant.", prompt)
    print("\nresponse:", r.text)
    print("accounting:", r.to_dict())
