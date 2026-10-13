"""Ingestion use case: chunk, embed in batches, write to the vector store and keyword index."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice

from rag_engine.domain import Chunk, Document
from rag_engine.ports import Chunker, Embedder, KeywordIndex, VectorStore


@dataclass(frozen=True, slots=True)
class IngestionReport:
    documents: int
    chunks: int


def _batched(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


class IngestionService:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        *,
        batch_size: int = 64,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._keyword_index = keyword_index
        self._batch_size = batch_size

    def ingest(self, documents: Sequence[Document]) -> IngestionReport:
        """Idempotent: re-ingesting a document replaces its previous chunks."""
        total = 0
        for document in documents:
            self.delete(document.id)
            chunks = self._chunker.split(document)
            for batch in _batched(chunks, self._batch_size):
                embeddings = self._embedder.embed_documents([self._embed_text(c) for c in batch])
                self._store.upsert(batch, embeddings)
                self._keyword_index.add(batch)
            total += len(chunks)
        return IngestionReport(documents=len(documents), chunks=total)

    def delete(self, document_id: str) -> int:
        removed = self._store.delete_document(document_id)
        self._keyword_index.delete_document(document_id)
        return removed

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        # The title gives short chunks the context they lack on their own.
        return f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text
