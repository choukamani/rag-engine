"""Ports: the interfaces the application core depends on.

The core (`rag_engine.application`) imports only these protocols and the
domain model. Concrete technologies (pgvector, fastembed, OpenAI...) live in
`rag_engine.adapters` and are wired together in `rag_engine.container`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from rag_engine.domain import Chunk, Document, ScoredChunk

type Vector = npt.NDArray[np.float32]
type Matrix = npt.NDArray[np.float32]


@runtime_checkable
class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """One L2-normalised row per text."""
        ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None: ...

    def search(self, query: Vector, k: int) -> list[ScoredChunk]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...


@runtime_checkable
class KeywordIndex(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def delete_document(self, document_id: str) -> int: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


__all__ = [
    "LLM",
    "Chunker",
    "Embedder",
    "KeywordIndex",
    "Matrix",
    "Retriever",
    "Vector",
    "VectorStore",
]
