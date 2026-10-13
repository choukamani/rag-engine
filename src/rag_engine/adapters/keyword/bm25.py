"""Okapi BM25 over an inverted index, implemented from scratch.

score(q, d) = sum over query terms t of
    IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)) (the Lucene variant,
always positive). Defaults k1 = 1.2, b = 0.75 match Lucene/Elasticsearch.
Only postings of the query terms are visited, so search cost grows with the
number of matching chunks, not with the corpus size.
"""

from __future__ import annotations

import heapq
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence

from rag_engine.domain import Chunk, ScoredChunk
from rag_engine.text import tokenize


class BM25Index:
    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._lengths: dict[str, int] = {}
        self._postings: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._by_document: defaultdict[str, set[str]] = defaultdict(set)
        self._total_length = 0
        self._lock = threading.RLock()

    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.id in self._chunks:
                    self._remove(chunk.id)
                terms = Counter(tokenize(f"{chunk.title} {chunk.text}"))
                self._chunks[chunk.id] = chunk
                self._by_document[chunk.document_id].add(chunk.id)
                self._lengths[chunk.id] = sum(terms.values())
                self._total_length += self._lengths[chunk.id]
                for term, tf in terms.items():
                    self._postings[term][chunk.id] = tf

    def _remove(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id)
        siblings = self._by_document.get(chunk.document_id)
        if siblings is not None:
            siblings.discard(chunk_id)
        self._total_length -= self._lengths.pop(chunk_id)
        for term in set(tokenize(f"{chunk.title} {chunk.text}")):
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(chunk_id, None)
                if not postings:
                    del self._postings[term]

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = self._by_document.pop(document_id, set())
            for chunk_id in list(ids):
                self._remove(chunk_id)
            return len(ids)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            n = len(self._chunks)
            if n == 0 or k <= 0:
                return []
            avgdl = self._total_length / n
            scores: defaultdict[str, float] = defaultdict(float)
            for term in set(tokenize(query)):
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = len(postings)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for chunk_id, tf in postings.items():
                    norm = self.k1 * (1 - self.b + self.b * self._lengths[chunk_id] / avgdl)
                    scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + norm)
            best = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], item[0]))
            return [ScoredChunk(self._chunks[cid], score, "bm25") for cid, score in best]

    def __len__(self) -> int:
        return len(self._chunks)
