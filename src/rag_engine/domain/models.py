"""Domain model. Pure data, no I/O, no framework imports."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def make_id(document_id: str, position: int) -> str:
        return f"{document_id}#{position}"


class RetrievalMode(enum.StrEnum):
    KEYWORD = "keyword"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str
    """Which retriever produced the score (e.g. "bm25", "dense", "rrf")."""


@dataclass(frozen=True, slots=True)
class Citation:
    index: int
    document_id: str
    chunk_id: str
    title: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    contexts: tuple[ScoredChunk, ...]
    grounded: bool
    """False when the engine declined to answer because nothing relevant was retrieved."""
