"""PHASE 5/6 — Build the knowledge base index.

Loads documents, cleans, chunks, embeds, and writes a committed index that the
app loads at startup rather than rebuilding.

    python scripts/build_index.py
    python scripts/build_index.py --strategy fixed --chunk-size 256
    python scripts/build_index.py --backend groq        # needs an API key
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.knowledge.chunker import build_chunks, chunk_stats  # noqa: E402
from src.knowledge.embedder import TfidfSvdEmbedder, get_embedder  # noqa: E402
from src.knowledge.loader import corpus_summary, load_corpus  # noqa: E402
from src.knowledge.vector_store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="section", choices=["section", "fixed"])
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--backend", default="tfidf_svd", choices=["tfidf_svd", "groq"])
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else settings.index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("1. LOAD")
    print("=" * 72)
    t0 = time.perf_counter()
    pages = load_corpus()
    summary = corpus_summary(pages)
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    if summary["empty_pages"]:
        print(f"  WARNING: {summary['empty_pages']} pages extracted no text")

    print("\n" + "=" * 72)
    print("2. CHUNK")
    print("=" * 72)
    chunks = build_chunks(
        pages, strategy=args.strategy, max_tokens=args.chunk_size, overlap=args.overlap
    )
    cstats = chunk_stats(chunks)
    for k, v in cstats.items():
        print(f"  {k:22s} {v}")
    if cstats["unique_ids"] != cstats["n_chunks"]:
        print("  WARNING: duplicate chunk IDs detected")

    print("\n" + "=" * 72)
    print("3. EMBED")
    print("=" * 72)
    texts = [c.text for c in chunks]
    if args.backend == "tfidf_svd":
        emb = get_embedder("tfidf_svd", dim=args.dim).fit(texts)
        print(f"  backend            tfidf_svd (local, offline)")
        print(f"  explained variance {emb.explained_variance:.3f}")
    else:
        emb = get_embedder("groq")
        print(f"  backend            groq ({emb.model})")

    t1 = time.perf_counter()
    vectors = emb.encode(texts)
    print(f"  dimension          {emb.dim}")
    print(f"  matrix             {vectors.shape}  {vectors.nbytes / 1024:.1f} KB")
    print(f"  encode time        {time.perf_counter() - t1:.2f}s")

    print("\n" + "=" * 72)
    print("4. STORE")
    print("=" * 72)
    store = VectorStore(chunks, vectors)
    store.save(out_dir)
    if isinstance(emb, TfidfSvdEmbedder):
        emb.save(out_dir / "embedder.pkl")

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "backend": args.backend,
        "embedding_dim": emb.dim,
        "corpus": summary,
        "chunks": cstats,
    }
    (out_dir / "index_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    for f in sorted(out_dir.glob("*")):
        print(f"  {f.name:24s} {f.stat().st_size / 1024:8.1f} KB")

    total_kb = sum(f.stat().st_size for f in out_dir.glob("*")) / 1024
    print(f"\n  total index size   {total_kb:.1f} KB")
    print(f"  build time         {time.perf_counter() - t0:.2f}s")

    print("\n" + "=" * 72)
    print(f"INDEX BUILT  |  {cstats['n_chunks']} chunks  ->  {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
