from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rag_engine.adapters.llm.extractive import ExtractiveLLM
from rag_engine.application.answering import NO_ANSWER, AnswerService, build_prompt
from rag_engine.application.retrieval import HybridRetriever, deduplicate_by_document
from rag_engine.domain import ScoredChunk
from rag_engine.ports import LLM, Retriever
from tests.conftest import chunk


@dataclass
class FixedRetriever:
    hits: list[ScoredChunk]
    calls: list[int] = field(default_factory=list)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        self.calls.append(k)
        return self.hits[:k]


@dataclass
class RecordingLLM:
    reply: str
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def hit(doc: str, score: float = 1.0, text: str = "", source: str = "x") -> ScoredChunk:
    return ScoredChunk(chunk(doc, text or doc), score, source)


class TestHybrid:
    def test_rrf_rewards_agreement(self) -> None:
        keyword = FixedRetriever([hit("a"), hit("b"), hit("c")])
        dense = FixedRetriever([hit("c"), hit("a"), hit("d")])
        result = HybridRetriever([keyword, dense]).retrieve("q", 4)
        assert [h.chunk.document_id for h in result] == ["a", "c", "b", "d"]
        assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert result[0].source == "rrf"

    def test_children_are_asked_for_candidates(self) -> None:
        child = FixedRetriever([hit("a")])
        HybridRetriever([child], candidates=50).retrieve("q", 5)
        HybridRetriever([child], candidates=10).retrieve("q", 20)
        assert child.calls == [50, 20]

    def test_weights(self) -> None:
        keyword = FixedRetriever([hit("a"), hit("b")])
        dense = FixedRetriever([hit("b"), hit("a")])
        result = HybridRetriever([keyword, dense], weights=[1.0, 3.0]).retrieve("q", 2)
        assert result[0].chunk.document_id == "b"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            HybridRetriever([])
        with pytest.raises(ValueError, match="one weight"):
            HybridRetriever([FixedRetriever([])], weights=[1.0, 2.0])

    def test_satisfies_port(self) -> None:
        assert isinstance(HybridRetriever([FixedRetriever([])]), Retriever)


def test_deduplicate_keeps_best_chunk_per_document() -> None:
    hits = [
        ScoredChunk(chunk("a", "1", 0), 0.9, "x"),
        ScoredChunk(chunk("a", "2", 1), 0.8, "x"),
        ScoredChunk(chunk("b", "3", 0), 0.7, "x"),
        ScoredChunk(chunk("c", "4", 0), 0.6, "x"),
    ]
    result = deduplicate_by_document(hits, 2)
    assert [(h.chunk.document_id, h.chunk.position) for h in result] == [("a", 0), ("b", 0)]


class TestAnswerService:
    def test_prompt_numbers_sources_and_states_the_rules(self) -> None:
        prompt = build_prompt("Why?", [hit("a", text="alpha"), hit("b", text="beta")])
        assert "[1] \nalpha" in prompt or "[1]" in prompt
        assert "[2]" in prompt
        assert "beta" in prompt
        assert NO_ANSWER in prompt
        assert prompt.rstrip().endswith("Question: Why?\nAnswer:")

    def test_valid_citations_are_extracted_and_invalid_ones_dropped(self) -> None:
        llm = RecordingLLM("Yes [2], also [1] and [7].")
        answer = AnswerService(FixedRetriever([hit("a"), hit("b")]), llm).answer("q")
        assert [c.document_id for c in answer.citations] == ["b", "a"]
        assert answer.grounded
        assert len(answer.contexts) == 2

    def test_no_context_means_no_llm_call(self) -> None:
        llm = RecordingLLM("should not be used")
        answer = AnswerService(FixedRetriever([]), llm).answer("q")
        assert answer.text == NO_ANSWER
        assert not answer.grounded
        assert llm.prompts == []

    def test_min_score_filters_weak_evidence(self) -> None:
        llm = RecordingLLM("x [1]")
        service = AnswerService(FixedRetriever([hit("a", 0.1)]), llm, min_score=0.5)
        assert not service.answer("q").grounded
        assert llm.prompts == []

    def test_uncited_or_refused_answers_are_not_grounded(self) -> None:
        retriever = FixedRetriever([hit("a")])
        assert not AnswerService(retriever, RecordingLLM("no citation")).answer("q").grounded
        refused = AnswerService(retriever, RecordingLLM(f"{NO_ANSWER} [1]")).answer("q")
        assert not refused.grounded


class TestExtractiveLLM:
    def test_satisfies_port(self) -> None:
        assert isinstance(ExtractiveLLM(), LLM)

    def test_extracts_best_sentence_with_citation(self) -> None:
        contexts = [
            hit("plans", text="Free includes 5 GB. Plus costs 9.99 euros per month."),
            hit("trash", text="Deleted files stay 30 days in the Trash."),
        ]
        reply = ExtractiveLLM().generate(build_prompt("How much does Plus cost?", contexts))
        assert reply == "Plus costs 9.99 euros per month. [1]"

    def test_refuses_without_overlap(self) -> None:
        prompt = build_prompt("quantum entanglement?", [hit("a", text="Files sync.")])
        assert ExtractiveLLM().generate(prompt) == NO_ANSWER

    def test_tolerates_prompts_without_sections(self) -> None:
        assert ExtractiveLLM().generate("hello") == NO_ANSWER
