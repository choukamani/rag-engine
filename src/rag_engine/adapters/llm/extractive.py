"""Deterministic offline stand-in for an LLM: answers by extracting the best-matching sentence.

It parses the numbered sources of the grounded prompt and returns the source
sentence sharing the most stemmed terms with the question, followed by its
citation. No API key, no network, same output every time, which is what tests,
CI and the demo need. It does not paraphrase or reason across sources.
"""

from __future__ import annotations

import re
from typing import Final

from rag_engine.application.answering import NO_ANSWER
from rag_engine.text import tokenize

_SOURCE: Final = re.compile(r"^\[(\d+)\] ?(.*)$")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM:
    def __init__(self, *, min_overlap: int = 1) -> None:
        self._min_overlap = min_overlap

    def generate(self, prompt: str) -> str:
        question = self._section(prompt, "Question:", "Answer:")
        sources = self._sources(self._section(prompt, "Sources:", "Question:"))
        terms = set(tokenize(question))
        best: tuple[int, int, str] | None = None
        for index, text in sources:
            for sentence in _SENTENCE.split(text):
                overlap = len(terms & set(tokenize(sentence)))
                if overlap >= self._min_overlap and (best is None or overlap > best[0]):
                    best = (overlap, index, sentence.strip())
        if best is None:
            return NO_ANSWER
        return f"{best[2]} [{best[1]}]"

    @staticmethod
    def _section(prompt: str, start: str, end: str) -> str:
        begin = prompt.rfind(start)
        if begin < 0:
            return ""
        begin += len(start)
        stop = prompt.find(end, begin)
        return prompt[begin : stop if stop >= 0 else None].strip()

    @staticmethod
    def _sources(block: str) -> list[tuple[int, str]]:
        sources: list[tuple[int, list[str]]] = []
        for line in block.splitlines():
            match = _SOURCE.match(line)
            if match:
                # The first line holds the title; keep it out of the answer text.
                sources.append((int(match.group(1)), []))
            elif sources and line.strip():
                sources[-1][1].append(line.strip())
        return [(i, " ".join(lines)) for i, lines in sources]
