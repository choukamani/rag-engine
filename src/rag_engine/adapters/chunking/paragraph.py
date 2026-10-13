"""Chunking strategy: pack whole paragraphs into windows of at most `max_words` words.

Paragraph boundaries are kept when possible (they carry meaning); a paragraph
longer than the window is split on sentence boundaries, then on words as a
last resort. Consecutive chunks share `overlap_words` words of context.
"""

from __future__ import annotations

import re
from typing import Final

from rag_engine.domain import Chunk, Document

_PARAGRAPH: Final = re.compile(r"\n\s*\n")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s+")


class ParagraphChunker:
    def __init__(self, max_words: int = 200, overlap_words: int = 30) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        if not 0 <= overlap_words < max_words:
            raise ValueError("overlap_words must be in [0, max_words)")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def split(self, document: Document) -> list[Chunk]:
        units = [u for p in _PARAGRAPH.split(document.text) for u in self._units(p)]
        windows: list[list[str]] = []
        current: list[str] = []
        for unit in units:
            words = unit.split()
            if current and len(current) + len(words) > self.max_words:
                windows.append(current)
                # Overlap only as much as still fits next to the incoming unit.
                keep = min(self.overlap_words, self.max_words - len(words))
                current = current[-keep:] if keep > 0 else []
            current = current + words
        if current:
            windows.append(current)
        return [
            Chunk(
                id=Chunk.make_id(document.id, position),
                document_id=document.id,
                text=" ".join(words),
                position=position,
                title=document.title,
                metadata=dict(document.metadata),
            )
            for position, words in enumerate(windows)
        ]

    def _units(self, paragraph: str) -> list[str]:
        """Return pieces no longer than the window: the paragraph, its sentences, or word slices."""
        if not paragraph.strip():
            return []
        if len(paragraph.split()) <= self.max_words:
            return [paragraph.strip()]
        pieces: list[str] = []
        for sentence in _SENTENCE.split(paragraph.strip()):
            words = sentence.split()
            step = self.max_words - self.overlap_words
            if len(words) <= self.max_words:
                pieces.append(sentence)
            else:
                pieces.extend(
                    " ".join(words[i : i + self.max_words]) for i in range(0, len(words), step)
                )
        return pieces
