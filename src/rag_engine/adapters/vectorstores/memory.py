"""Exact (brute-force) cosine search over an in-memory matrix.

Exact search is the reference the approximate pgvector HNSW index is compared
against. Rows are appended in blocks and deleted with tombstones, so ingestion
stays linear; the matrix is compacted lazily when a search needs it.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from rag_engine.domain import Chunk, ScoredChunk
from rag_engine.ports import Matrix, Vector


class InMemoryVectorStore:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._chunks: list[Chunk | None] = []  # None marks a deleted row
        self._rows: dict[str, int] = {}
        self._rows_by_document: defaultdict[str, set[int]] = defaultdict(set)
        self._pending: list[Matrix] = []
        self._matrix: Matrix = np.zeros((0, dimension), dtype=np.float32)
        self._lock = threading.RLock()

    def upsert(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        if embeddings.shape[0] and embeddings.shape[1] != self._dimension:
            raise ValueError(f"expected {self._dimension}-d embeddings")
        with self._lock:
            for chunk in chunks:
                if chunk.id in self._rows:
                    self._delete_row(self._rows[chunk.id])
            start = len(self._chunks)
            for offset, chunk in enumerate(chunks):
                self._chunks.append(chunk)
                self._rows[chunk.id] = start + offset
                self._rows_by_document[chunk.document_id].add(start + offset)
            self._pending.append(embeddings.astype(np.float32))

    def _delete_row(self, row: int) -> None:
        chunk = self._chunks[row]
        if chunk is None:
            return
        self._chunks[row] = None
        del self._rows[chunk.id]
        self._rows_by_document[chunk.document_id].discard(row)

    def _materialize(self) -> Matrix:
        if self._pending:
            self._matrix = np.vstack([self._matrix, *self._pending])
            self._pending.clear()
        return self._matrix

    def search(self, query: Vector, k: int) -> list[ScoredChunk]:
        with self._lock:
            if not self._rows or k <= 0:
                return []
            scores = self._materialize() @ query.astype(np.float32)
            alive = np.fromiter((c is not None for c in self._chunks), bool, len(self._chunks))
            scores[~alive] = -np.inf
            k = min(k, len(self._rows))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [
                ScoredChunk(chunk, float(scores[i]), "dense")
                for i in top
                if (chunk := self._chunks[i]) is not None
            ]

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            rows = self._rows_by_document.pop(document_id, set())
            for row in list(rows):
                self._delete_row(row)
            return len(rows)

    def count(self) -> int:
        return len(self._rows)
