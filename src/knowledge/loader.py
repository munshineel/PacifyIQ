"""Document loading with provenance.

Every unit of text carries the document, page and section it came from, because
citations are a hard requirement: the system must be able to say "return_policy_v2,
page 3, section S2" and have that be verifiable.

Loads from PDF (the shipped corpus) with a markdown fallback for the sources in
data/_source/.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import settings

# ---------------------------------------------------------------------
# Document classification. Drives metadata filtering in the retriever:
# archived documents must never be cited as current policy.
# ---------------------------------------------------------------------
DOC_REGISTRY: dict[str, dict[str, Any]] = {
    "return_policy_v2": {
        "title": "Returns Policy", "ref": "POL-RET-002", "type": "policy",
        "topic": "returns", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "return_policy_v1_ARCHIVED": {
        "title": "Returns Policy (superseded)", "ref": "POL-RET-001", "type": "policy",
        "topic": "returns", "version": "archived", "effective": "2024-03-01",
        "region": "all",
    },
    "refund_policy": {
        "title": "Refunds Policy", "ref": "POL-REF-001", "type": "policy",
        "topic": "refunds", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "warranty_policy": {
        "title": "Warranty Policy", "ref": "POL-WAR-001", "type": "policy",
        "topic": "warranty", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "shipping_policy": {
        "title": "Shipping and Delivery Policy", "ref": "POL-SHP-001", "type": "policy",
        "topic": "shipping", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "payment_policy": {
        "title": "Payments Policy", "ref": "POL-PAY-001", "type": "policy",
        "topic": "billing", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "customer_service_policy": {
        "title": "Customer Service Policy", "ref": "POL-CS-001", "type": "policy",
        "topic": "account", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "eu_regional_addendum": {
        "title": "EU Regional Addendum", "ref": "POL-EU-001", "type": "policy",
        "topic": "returns", "version": "current", "effective": "2026-01-15",
        "region": "EU",
    },
    "product_faq": {
        "title": "Frequently Asked Questions", "ref": "FAQ-PRD-001", "type": "faq",
        "topic": "general", "version": "current", "effective": "2026-01-15",
        "region": "all",
    },
    "technical_support_faq": {
        "title": "Technical Support Guide", "ref": "FAQ-TEC-001",
        "type": "troubleshooting", "topic": "technical", "version": "current",
        "effective": "2026-01-15", "region": "all",
    },
    "manual_probook14": {
        "title": "Pacify ProBook 14 User Manual", "ref": "MAN-PB14-001",
        "type": "manual", "topic": "product", "version": "current",
        "effective": "2026-01-15", "region": "all", "product": "Pacify ProBook 14",
    },
    "manual_phonex": {
        "title": "Pacify Phone X User Manual", "ref": "MAN-PPX-001",
        "type": "manual", "topic": "product", "version": "current",
        "effective": "2026-01-15", "region": "all", "product": "Pacify Phone X",
    },
    "manual_vision27": {
        "title": "Pacify Vision 27 User Manual", "ref": "MAN-PV27-001",
        "type": "manual", "topic": "product", "version": "current",
        "effective": "2026-01-15", "region": "all", "product": "Pacify Vision 27",
    },
}

# Section headings look like "S1. Scope" or "S2.1 Return windows".
RE_SECTION = re.compile(r"^(S\d+)(?:\.(\d+))?\.?\s+(.{0,90})$", re.M)
RE_SECTION_INLINE = re.compile(r"\b(S\d+)\.(\d+)\b")


@dataclass
class Page:
    """One page of one document, with everything needed for a citation."""

    doc: str
    page: int
    text: str
    title: str
    doc_ref: str
    doc_type: str
    topic: str
    version: str
    region: str
    effective: str
    product: str | None = None
    sections: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.doc_ref}, p.{self.page}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _meta(stem: str) -> dict[str, Any]:
    """Registry lookup with a safe default for unregistered documents."""
    return DOC_REGISTRY.get(
        stem,
        {
            "title": stem.replace("_", " ").title(),
            "ref": stem.upper(),
            "type": "unknown",
            "topic": "general",
            "version": "current",
            "effective": "unknown",
            "region": "all",
        },
    )


def find_sections(text: str) -> list[str]:
    """Top-level section IDs appearing on a page.

    Evaluation gold labels reference (doc, section) pairs rather than chunk IDs,
    because chunk IDs change with every chunking ablation. Section IDs are the
    stable join key, so extracting them correctly is load-bearing.
    """
    ids = []
    for m in RE_SECTION.finditer(text):
        sid = m.group(1)
        if sid not in ids:
            ids.append(sid)
    for m in RE_SECTION_INLINE.finditer(text):
        sid = m.group(1)
        if sid not in ids:
            ids.append(sid)
    return sorted(ids, key=lambda s: int(s[1:]))


def load_pdf(path: Path) -> list[Page]:
    import pdfplumber

    stem = path.stem
    m = _meta(stem)
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            pages.append(
                Page(
                    doc=stem,
                    page=i,
                    text=text,
                    title=m["title"],
                    doc_ref=m["ref"],
                    doc_type=m["type"],
                    topic=m["topic"],
                    version=m["version"],
                    region=m["region"],
                    effective=m["effective"],
                    product=m.get("product"),
                    sections=find_sections(text),
                )
            )
    return pages


def load_markdown(path: Path) -> list[Page]:
    """Fallback loader for data/_source/*.md — one synthetic page per file."""
    stem = path.stem
    m = _meta(stem)
    text = path.read_text(encoding="utf-8")
    return [
        Page(
            doc=stem, page=1, text=text, title=m["title"], doc_ref=m["ref"],
            doc_type=m["type"], topic=m["topic"], version=m["version"],
            region=m["region"], effective=m["effective"], product=m.get("product"),
            sections=find_sections(text),
        )
    ]


def load_corpus(directory: Path | None = None) -> list[Page]:
    """Load every document in the knowledge base."""
    directory = Path(directory or settings.documents_dir)
    pages: list[Page] = []
    for path in sorted(directory.rglob("*.pdf")):
        pages.extend(load_pdf(path))
    if not pages:  # fall back to markdown sources
        for path in sorted((settings.data_dir / "_source").glob("*.md")):
            pages.extend(load_markdown(path))
    return pages


def corpus_summary(pages: list[Page]) -> dict[str, Any]:
    from collections import Counter

    docs = {p.doc for p in pages}
    return {
        "n_documents": len(docs),
        "n_pages": len(pages),
        "n_words": sum(len(p.text.split()) for p in pages),
        "n_chars": sum(len(p.text) for p in pages),
        "by_type": dict(Counter(p.doc_type for p in pages)),
        "by_topic": dict(Counter(p.topic for p in pages)),
        "by_version": dict(Counter(p.version for p in pages)),
        "regional": sorted({p.doc for p in pages if p.region != "all"}),
        "empty_pages": sum(1 for p in pages if not p.text.strip()),
    }


if __name__ == "__main__":
    pages = load_corpus()
    s = corpus_summary(pages)
    print(f"documents  {s['n_documents']}")
    print(f"pages      {s['n_pages']}")
    print(f"words      {s['n_words']:,}")
    print(f"by type    {s['by_type']}")
    print(f"by topic   {s['by_topic']}")
    print(f"by version {s['by_version']}")
    print(f"EU-only    {s['regional']}")
    print(f"empty      {s['empty_pages']}")
    print("\nsections found per document:")
    from collections import defaultdict

    per = defaultdict(set)
    for p in pages:
        per[p.doc].update(p.sections)
    for d in sorted(per):
        ids = sorted(per[d], key=lambda x: int(x[1:]))
        print(f"  {d:32s} {len(ids):2d}  {', '.join(ids[:14])}")
