"""Run retrieval configurations against a dataset and aggregate metrics and latencies.

The corpus is indexed once per embedder; keyword, dense and hybrid retrievers
are then evaluated on the same indexes, so differences come from the retrieval
strategy alone.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from rag_engine.adapters.chunking.paragraph import ParagraphChunker
from rag_engine.adapters.keyword.bm25 import BM25Index
from rag_engine.adapters.vectorstores.memory import InMemoryVectorStore
from rag_engine.application.ingestion import IngestionService
from rag_engine.application.retrieval import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
    deduplicate_by_document,
)
from rag_engine.evaluation.datasets import Dataset
from rag_engine.evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from rag_engine.ports import Embedder, Retriever, VectorStore

type StoreFactory = Callable[[Embedder], VectorStore]


@dataclass(frozen=True)
class RunResult:
    config: str
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    mrr_at_10: float
    hit_at_5: float
    query_p50_ms: float
    query_p95_ms: float
    queries: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class IndexStats:
    embedder: str
    documents: int
    chunks: int
    seconds: float


def evaluate_retriever(name: str, retriever: Retriever, dataset: Dataset) -> RunResult:
    scores: dict[str, list[float]] = {m: [] for m in ("ndcg", "r10", "r100", "mrr", "hit5")}
    latencies: list[float] = []
    for query_id, query in dataset.queries.items():
        qrels = dataset.qrels[query_id]
        start = time.perf_counter()
        # Over-fetch chunks, then keep the best chunk per document (qrels are per document).
        hits = deduplicate_by_document(retriever.retrieve(query, 300), 100)
        latencies.append((time.perf_counter() - start) * 1000)
        ranking = [h.chunk.document_id for h in hits]
        scores["ndcg"].append(ndcg_at_k(ranking, qrels, 10))
        scores["r10"].append(recall_at_k(ranking, qrels, 10))
        scores["r100"].append(recall_at_k(ranking, qrels, 100))
        scores["mrr"].append(mrr_at_k(ranking, qrels, 10))
        scores["hit5"].append(hit_at_k(ranking, qrels, 5))

    def mean(key: str) -> float:
        return round(statistics.fmean(scores[key]), 4)

    p50, p95 = _percentiles(latencies)
    return RunResult(
        config=name,
        ndcg_at_10=mean("ndcg"),
        recall_at_10=mean("r10"),
        recall_at_100=mean("r100"),
        mrr_at_10=mean("mrr"),
        hit_at_5=mean("hit5"),
        query_p50_ms=p50,
        query_p95_ms=p95,
        queries=len(latencies),
    )


def _percentiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:  # noqa: PLR2004
        only = round(values[0], 2) if values else 0.0
        return only, only
    q = statistics.quantiles(values, n=100, method="inclusive")
    return round(q[49], 2), round(q[94], 2)


def run(
    dataset: Dataset,
    embedders: dict[str, Embedder],
    *,
    chunk_max_words: int = 400,
    store_factory: StoreFactory | None = None,
    store_label: str = "",
) -> tuple[list[RunResult], list[IndexStats]]:
    """Evaluate BM25 once, then dense and hybrid retrieval for each embedder."""
    make_store: StoreFactory = store_factory or (lambda e: InMemoryVectorStore(e.dimension))
    chunker = ParagraphChunker(max_words=chunk_max_words, overlap_words=40)
    results: list[RunResult] = []
    stats: list[IndexStats] = []
    bm25: BM25Index | None = None
    suffix = f" ({store_label})" if store_label else ""

    for embedder_name, embedder in embedders.items():
        store = make_store(embedder)
        keyword = BM25Index()
        start = time.perf_counter()
        report = IngestionService(chunker, embedder, store, keyword).ingest(dataset.documents)
        stats.append(
            IndexStats(embedder_name, report.documents, report.chunks,
                       round(time.perf_counter() - start, 2))
        )  # fmt: skip
        if bm25 is None:
            bm25 = keyword
            results.append(evaluate_retriever("BM25", KeywordRetriever(bm25), dataset))
        dense = DenseRetriever(embedder, store)
        results.append(evaluate_retriever(f"Dense: {embedder_name}{suffix}", dense, dataset))
        hybrid = HybridRetriever([KeywordRetriever(bm25), dense])
        results.append(
            evaluate_retriever(f"Hybrid RRF: BM25 + {embedder_name}{suffix}", hybrid, dataset)
        )
    return results, stats
