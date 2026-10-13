"""pgvector HNSW versus exact search: recall and latency as a function of `ef_search`.

Indexes a dataset's chunks in PostgreSQL (pgvector HNSW, m=16, ef_construction=64)
and in the exact in-memory store, then compares the top-10 neighbours of every
query. Recall@10 here means "share of the exact top-10 that HNSW also returns".

    RAG_DATABASE_URL=postgresql://rag:rag@127.0.0.1:5432/rag \
        uv run python benchmarks/hnsw_vs_exact.py --dataset scifact --embedder hashing
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from rag_engine.adapters.chunking.paragraph import ParagraphChunker
from rag_engine.adapters.vectorstores.memory import InMemoryVectorStore
from rag_engine.adapters.vectorstores.postgres import PgVectorStore, PostgresDatabase
from rag_engine.evaluation.cli import build_embedders, load_dataset
from rag_engine.ports import Matrix, Vector

EF_SEARCH = (10, 20, 40, 80, 160)
K = 10


def query_plan(db: PostgresDatabase, query: Vector, ef_search: int) -> str:
    """Which access path PostgreSQL picks: the planner may prefer an exact scan on small tables."""
    with db.connection() as conn, conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        rows = conn.execute(
            "EXPLAIN SELECT id FROM bench_hnsw ORDER BY embedding <=> %s LIMIT 10", (query,)
        ).fetchall()
    plan = " ".join(str(r[0]) for r in rows)
    return "HNSW index scan" if "embedding_hnsw" in plan else "sequential scan + sort (exact)"


def percentiles(values: list[float]) -> tuple[float, float]:
    q = statistics.quantiles(values, n=100, method="inclusive")
    return round(q[49], 2), round(q[94], 2)


def main() -> None:  # noqa: PLR0915 - linear benchmark script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nimbus")
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "fastembed"])
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dsn = os.environ.get("RAG_DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5432/rag")

    dataset = load_dataset(args.dataset, args.cache / "datasets")
    ((name, embedder),) = build_embedders([args.embedder]).items()
    chunks = [c for d in dataset.documents for c in ParagraphChunker(400, 40).split(d)]

    cache = args.cache / "embeddings" / f"{dataset.name}-{name}.npy"
    if cache.exists():
        matrix: Matrix = np.load(cache)
    else:
        matrix = embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, matrix)
    queries = [embedder.embed_query(q) for q in dataset.queries.values()]

    exact = InMemoryVectorStore(embedder.dimension)
    exact.upsert(chunks, matrix)
    exact_ms: list[float] = []
    truth: list[set[str]] = []
    for q in queries:
        start = time.perf_counter()
        truth.append({h.chunk.id for h in exact.search(q, K)})
        exact_ms.append((time.perf_counter() - start) * 1000)

    db = PostgresDatabase(dsn, embedder.dimension, table="bench_hnsw")
    with db.connection() as conn:
        conn.execute("TRUNCATE bench_hnsw")
    start = time.perf_counter()
    PgVectorStore(db).upsert(chunks, matrix)  # the HNSW index is maintained on insert
    build_s = time.perf_counter() - start

    rows = []
    p50, p95 = percentiles(exact_ms)
    rows.append(
        {"method": "exact (numpy, in-memory)", "recall_at_10": 1.0, "p50_ms": p50, "p95_ms": p95}
    )
    for ef in EF_SEARCH:
        store = PgVectorStore(db, ef_search=ef)
        for q in queries[:20]:  # warm-up
            store.search(q, K)
        latencies, recalls = [], []
        for q, expected in zip(queries, truth, strict=True):
            start = time.perf_counter()
            got = {h.chunk.id for h in store.search(q, K)}
            latencies.append((time.perf_counter() - start) * 1000)
            recalls.append(len(got & expected) / len(expected))
        p50, p95 = percentiles(latencies)
        rows.append({"method": f"pgvector, hnsw.ef_search={ef}",
                     "plan": query_plan(db, queries[0], ef),
                     "recall_at_10": round(statistics.fmean(recalls), 4),
                     "p50_ms": p50, "p95_ms": p95})  # fmt: skip
    with db.connection() as conn:
        conn.execute("DROP TABLE bench_hnsw")
    db.close()

    print(f"{dataset.name}: {len(chunks)} chunks, {len(queries)} queries, embedder {name} "
          f"({embedder.dimension}-d). Insert + HNSW build: {build_s:.1f} s\n")  # fmt: skip
    print("| Method | Recall@10 vs exact | p50 ms | p95 ms |\n|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['method']} | {r['recall_at_10']:.3f} | {r['p50_ms']} | {r['p95_ms']} |")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "dataset": dataset.name, "chunks": len(chunks), "queries": len(queries),
            "embedder": name, "dimension": embedder.dimension, "build_seconds": round(build_s, 1),
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "results": rows,
        }, indent=2) + "\n", encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    main()
