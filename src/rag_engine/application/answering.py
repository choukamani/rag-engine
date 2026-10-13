"""Question-answering use case: retrieve, build a grounded prompt, generate, extract citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from rag_engine.domain import Answer, Citation, ScoredChunk
from rag_engine.ports import LLM, Retriever

NO_ANSWER: Final = "I don't know based on the provided documents."
_CITATION: Final = re.compile(r"\[(\d+)\]")

PROMPT_TEMPLATE: Final = """You answer questions using ONLY the numbered sources below.
Cite every statement with the source number in square brackets, e.g. [1].
If the sources do not contain the answer, reply exactly: "{no_answer}"

Sources:
{sources}

Question: {question}
Answer:"""


def build_prompt(question: str, contexts: Sequence[ScoredChunk]) -> str:
    sources = "\n\n".join(
        f"[{i}] {hit.chunk.title}\n{hit.chunk.text}".strip() for i, hit in enumerate(contexts, 1)
    )
    return PROMPT_TEMPLATE.format(no_answer=NO_ANSWER, sources=sources, question=question)


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str) -> Answer:
        contexts = self._retriever.retrieve(question, self._top_k)
        if self._min_score is not None:
            contexts = [c for c in contexts if c.score >= self._min_score]
        if not contexts:
            # Do not call the LLM without evidence: it would answer from memory.
            return Answer(NO_ANSWER, (), (), grounded=False)

        text = self._llm.generate(build_prompt(question, contexts)).strip()
        citations = self._citations(text, contexts)
        grounded = bool(citations) and NO_ANSWER not in text
        return Answer(text, citations, tuple(contexts), grounded=grounded)

    @staticmethod
    def _citations(text: str, contexts: Sequence[ScoredChunk]) -> tuple[Citation, ...]:
        """Keep only citation numbers that point to a source actually given to the model."""
        indexes = dict.fromkeys(int(n) for n in _CITATION.findall(text))
        return tuple(
            Citation(i, contexts[i - 1].chunk.document_id, contexts[i - 1].chunk.id,
                     contexts[i - 1].chunk.title)
            for i in indexes
            if 1 <= i <= len(contexts)
        )  # fmt: skip
