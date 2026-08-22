"""Cleaning and chunking.

Two chunking strategies, both preserving the (doc, page, section) triple that
citations and evaluation gold labels depend on:

- `section`: split on section boundaries, then subdivide anything oversized.
  Sections are the natural semantic unit of a policy document — a clause is an
  answer, and splitting mid-clause destroys it.
- `fixed`: fixed token windows with overlap. The conventional baseline, kept so
  the choice can be ablated rather than asserted.

EDA finding 7c drives the size: the corpus is 16,208 words (~21,000 tokens), so
512-token chunks would yield only ~41 chunks and make retrieval metrics
meaningless. Default is 200 tokens.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from src.knowledge.loader import Page

# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------
# Footers appear in two layouts. Policies break them across lines; manuals put
# the reference, effective date, company name and page number on ONE line. The
# original alternation only matched the split form, so every manual page kept
# its footer and the company name leaked into chunks.
RE_FOOTER = re.compile(
    r"^\s*(?:"
    r"[A-Z]{3}-[A-Z0-9]{2,5}-\d{3}\s*\|.*?(?:Effective|Last updated).*?"
    r"(?:Pacify Electronics[^\n]*)?(?:Page\s*\d+)?"
    r"|Pacify Electronics Pvt\.?\s*Ltd\.?(?:\s*Page\s*\d+)?"
    r"|Page\s*\d+(?:\s*of\s*\d+)?"
    r")\s*$",
    re.M | re.I,
)
# Belt and braces: the company name can also appear mid-line after a date.
RE_FOOTER_INLINE = re.compile(
    r"\s*Pacify Electronics Pvt\.?\s*Ltd\.?\s*(?:Page\s*\d+)?\s*$",
    re.M | re.I,
)
RE_MULTISPACE = re.compile(r"[ \t]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")
RE_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
RE_BULLET = re.compile(r"^\s*[•▪◦]\s*", re.M)


def clean(text: str) -> str:
    """Strip PDF artefacts without destroying structure.

    Footers repeat on every page and would otherwise appear in every chunk,
    inflating similarity between unrelated chunks of the same document.
    """
    t = RE_FOOTER.sub("", text)
    t = RE_FOOTER_INLINE.sub("", t)
    t = RE_HYPHEN_BREAK.sub(r"\1\2", t)      # rejoin words split across lines
    t = RE_BULLET.sub("- ", t)
    t = RE_MULTISPACE.sub(" ", t)
    t = RE_MULTINEWLINE.sub("\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - fallback when tiktoken is unavailable
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------
@dataclass
class Chunk:
    """One retrievable unit, carrying everything a citation needs."""

    chunk_id: str
    text: str
    doc: str
    page: int
    section: str | None
    section_title: str | None
    title: str
    doc_ref: str
    doc_type: str
    topic: str
    version: str
    region: str
    product: str | None
    n_tokens: int
    n_chars: int
    strategy: str
    position: int
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        sec = f", {self.section}" if self.section else ""
        return f"{self.doc_ref}, p.{self.page}{sec}"

    @property
    def is_current(self) -> bool:
        return self.version == "current"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def preview(self, n: int = 110) -> str:
        t = " ".join(self.text.split())
        return t[:n] + ("..." if len(t) > n else "")


def _chunk_id(doc: str, page: int, section: str | None, position: int, text: str) -> str:
    """Deterministic ID. Same input always produces the same ID, so an index
    rebuild is reproducible and diffable."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    sec = section or "S0"
    return f"{doc}::{sec}::p{page}::{position:03d}::{h}"


# ---------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------
# `[ \t]+` not `\s+`: \s matches newlines, so a line ending in a cross-reference
# such as "...see POL-WAR-001 S10." would swallow the heading on the NEXT line
# and mislabel an entire section. Found by inspecting chunk output.
RE_SECTION_HEAD = re.compile(r"^(S\d+)\.?[ \t]+([^\n]{2,90})$", re.M)


def split_sections(text: str) -> list[tuple[str | None, str | None, str]]:
    """Split page text at top-level section headings.

    Returns (section_id, section_title, body) triples. Text before the first
    heading is returned with section_id None.
    """
    matches = list(RE_SECTION_HEAD.finditer(text))
    if not matches:
        return [(None, None, text)]

    out: list[tuple[str | None, str | None, str]] = []
    if matches[0].start() > 0:
        head = text[: matches[0].start()].strip()
        if head:
            out.append((None, None, head))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start(): end].strip()
        if body:
            out.append((m.group(1), m.group(2).strip(), body))
    return out


def _split_oversized(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Break an oversized block on sentence boundaries where possible."""
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.;:])\s+", text)
    parts, current, cur_tok = [], [], 0

    for s in sentences:
        st = count_tokens(s)
        if cur_tok + st > max_tokens and current:
            parts.append(" ".join(current))
            # carry the tail forward as overlap so a clause split across two
            # chunks is still answerable from either
            keep, kept = [], 0
            for prev in reversed(current):
                pt = count_tokens(prev)
                if kept + pt > overlap:
                    break
                keep.insert(0, prev)
                kept += pt
            current, cur_tok = keep, kept
        current.append(s)
        cur_tok += st

    if current:
        parts.append(" ".join(current))
    return [p for p in parts if p.strip()]


def chunk_by_section(
    pages: Iterable[Page], max_tokens: int = 200, overlap: int = 40,
    min_tokens: int = 20,
) -> list[Chunk]:
    """Split at section boundaries, subdividing anything oversized."""
    chunks: list[Chunk] = []
    # A section frequently continues across a page break. Carrying the last
    # seen heading forward keeps that continuation attributable instead of
    # leaving ~20% of chunks with no section and therefore no citation.
    carry: dict[str, tuple[str | None, str | None]] = {}

    for page in pages:
        text = clean(page.text)
        if not text:
            continue
        pos = 0
        for sec_id, sec_title, body in split_sections(text):
            if sec_id is None:
                sec_id, sec_title = carry.get(page.doc, (None, None))
            else:
                carry[page.doc] = (sec_id, sec_title)
            for part in _split_oversized(body, max_tokens, overlap):
                n_tok = count_tokens(part)
                if n_tok < min_tokens:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(page.doc, page.page, sec_id, pos, part),
                        text=part,
                        doc=page.doc,
                        page=page.page,
                        section=sec_id,
                        section_title=sec_title,
                        title=page.title,
                        doc_ref=page.doc_ref,
                        doc_type=page.doc_type,
                        topic=page.topic,
                        version=page.version,
                        region=page.region,
                        product=page.product,
                        n_tokens=n_tok,
                        n_chars=len(part),
                        strategy=f"section_{max_tokens}",
                        position=pos,
                    )
                )
                pos += 1
    return chunks


def chunk_fixed(
    pages: Iterable[Page], max_tokens: int = 200, overlap: int = 40,
    min_tokens: int = 20,
) -> list[Chunk]:
    """Fixed-size windows with overlap. The conventional baseline.

    Section attribution is best-effort: whichever section heading most recently
    appeared before the window. That is exactly the weakness this strategy has
    and the ablation is meant to expose.
    """
    chunks: list[Chunk] = []
    carry_fixed: dict[str, str] = {}
    for page in pages:
        text = clean(page.text)
        if not text:
            continue
        # Record where each heading sits in the word stream BEFORE splitting,
        # because joining words with spaces destroys the line breaks the
        # heading regex depends on.
        heading_at: list[tuple[int, str, str]] = []
        seen = 0
        for line in text.split("\n"):
            m = RE_SECTION_HEAD.match(line.strip())
            if m:
                heading_at.append((seen, m.group(1), m.group(2).strip()))
            seen += len(line.split())

        words = text.split()
        step = max(1, int((max_tokens - overlap) / 1.3))
        window = max(1, int(max_tokens / 1.3))

        pos = 0
        for start in range(0, len(words), step):
            part = " ".join(words[start: start + window])
            n_tok = count_tokens(part)
            if n_tok < min_tokens:
                continue
            # the last heading that begins at or before the middle of the window
            midpoint = start + window // 2
            prior = [h for h in heading_at if h[0] <= midpoint]
            sec_id = prior[-1][1] if prior else carry_fixed.get(page.doc)
            sec_title = prior[-1][2] if prior else None
            if sec_id:
                carry_fixed[page.doc] = sec_id

            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(page.doc, page.page, sec_id, pos, part),
                    text=part, doc=page.doc, page=page.page,
                    section=sec_id, section_title=sec_title,
                    title=page.title, doc_ref=page.doc_ref, doc_type=page.doc_type,
                    topic=page.topic, version=page.version, region=page.region,
                    product=page.product, n_tokens=n_tok, n_chars=len(part),
                    strategy=f"fixed_{max_tokens}", position=pos,
                )
            )
            pos += 1
            if start + window >= len(words):
                break
    return chunks


CHUNKERS = {"section": chunk_by_section, "fixed": chunk_fixed}


def build_chunks(
    pages: Iterable[Page], strategy: str = "section", max_tokens: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {list(CHUNKERS)}")
    return CHUNKERS[strategy](pages, max_tokens=max_tokens, overlap=overlap)


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    import numpy as np
    from collections import Counter

    toks = np.array([c.n_tokens for c in chunks])
    return {
        "n_chunks": len(chunks),
        "tokens_total": int(toks.sum()),
        "tokens_mean": round(float(toks.mean()), 1),
        "tokens_median": int(np.median(toks)),
        "tokens_min": int(toks.min()),
        "tokens_max": int(toks.max()),
        "chunks_with_section": sum(1 for c in chunks if c.section),
        "section_coverage_pct": round(
            100 * sum(1 for c in chunks if c.section) / max(len(chunks), 1), 1
        ),
        "unique_ids": len({c.chunk_id for c in chunks}),
        "by_doc_type": dict(Counter(c.doc_type for c in chunks)),
        "archived_chunks": sum(1 for c in chunks if not c.is_current),
    }


if __name__ == "__main__":
    from src.knowledge.loader import load_corpus

    pages = load_corpus()
    print(f"{'strategy':22s} {'chunks':>7s} {'mean tok':>9s} {'median':>7s} "
          f"{'max':>5s} {'sec cov':>8s}")
    print("-" * 64)
    for strat in ("section", "fixed"):
        for size in (128, 200, 256, 512):
            cs = build_chunks(pages, strategy=strat, max_tokens=size,
                              overlap=int(size * 0.2))
            s = chunk_stats(cs)
            print(f"{strat + '_' + str(size):22s} {s['n_chunks']:7d} "
                  f"{s['tokens_mean']:9.1f} {s['tokens_median']:7d} "
                  f"{s['tokens_max']:5d} {s['section_coverage_pct']:7.1f}%")
