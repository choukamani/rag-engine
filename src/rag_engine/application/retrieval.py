"""Retrieval strategies. All implement the `Retriever` port, so they are interchangeable."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_engine.domain import Chunk, ScoredChunk
from rag_engine.ports import Embedder, KeywordIndex, Retriever, VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._store.search(self._embedder.embed_query(query), k)


class KeywordRetriever:
    def __init__(self, index: KeywordIndex) -> None:
        self._index = index

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._index.search(query, k)


class HybridRetriever:
    """Composite retriever fusing ranked lists with Reciprocal Rank Fusion.

    RRF(d) = sum_i w_i / (rrf_k + rank_i(d)). It uses ranks only, so BM25 scores
    (unbounded) and cosine similarities (in [-1, 1]) need no calibration.
    `rrf_k = 60` is the constant from Cormack et al. (2009). Each child retriever
    is asked for `candidates` results so that documents ranked low by one
    retriever but high by the other can still surface.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("one weight per retriever")
        self._retrievers = list(retrievers)
        self._weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        self._rrf_k = rrf_k
        self._candidates = candidates

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        fused: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        depth = max(k, self._candidates)
        for retriever, weight in zip(self._retrievers, self._weights, strict=True):
            for rank, hit in enumerate(retriever.retrieve(query, depth), start=1):
                fused[hit.chunk.id] += weight / (self._rrf_k + rank)
                chunks.setdefault(hit.chunk.id, hit.chunk)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [ScoredChunk(chunks[cid], score, "rrf") for cid, score in ranked]


def deduplicate_by_document(hits: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Keep the best chunk per document: evaluation qrels are document-level."""
    seen: set[str] = set()
    result: list[ScoredChunk] = []
    for hit in hits:
        if hit.chunk.document_id not in seen:
            seen.add(hit.chunk.document_id)
            result.append(hit)
            if len(result) == k:
                break
    return result
