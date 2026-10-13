from __future__ import annotations

import pytest

from rag_engine.adapters.chunking.paragraph import ParagraphChunker
from rag_engine.ports import Chunker
from rag_engine.text import tokenize
from tests.conftest import document


def test_tokenize_lowercases_stems_and_drops_stop_words() -> None:
    assert tokenize("The Files were DELETED, it's restoring!") == ["file", "delet", "restor"]


def test_tokenize_keeps_numbers_and_negations() -> None:
    assert tokenize("not 30 days") == ["not", "30", "day"]


def test_chunker_satisfies_port() -> None:
    assert isinstance(ParagraphChunker(), Chunker)


def test_short_document_is_one_chunk_with_metadata() -> None:
    doc = document("d1", "First paragraph.\n\nSecond paragraph.", title="T")
    chunks = ParagraphChunker(max_words=50).split(doc)
    assert len(chunks) == 1
    assert chunks[0].id == "d1#0"
    assert chunks[0].title == "T"
    assert chunks[0].text == "First paragraph. Second paragraph."


def test_paragraphs_are_packed_without_exceeding_the_window() -> None:
    paragraphs = [" ".join(f"p{p}w{i}" for i in range(30)) for p in range(5)]
    chunks = ParagraphChunker(max_words=70, overlap_words=0).split(
        document("d", "\n\n".join(paragraphs))
    )
    assert all(len(c.text.split()) <= 70 for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    # Paragraph boundaries are kept: no chunk starts mid-paragraph without overlap.
    assert all(c.text.split()[0].endswith("w0") for c in chunks)


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    text = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(40)) for p in range(3))
    chunks = ParagraphChunker(max_words=60, overlap_words=10).split(document("d", text))
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert second[:10] == first[-10:]


def test_long_paragraph_is_split_on_sentences_then_words() -> None:
    sentence = " ".join(f"w{i}" for i in range(15)) + "."
    giant = " ".join(f"x{i}" for i in range(120))
    chunks = ParagraphChunker(max_words=40, overlap_words=5).split(
        document("d", f"{sentence} {sentence} {sentence}\n\n{giant}")
    )
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert "x119" in chunks[-1].text


def test_empty_document_has_no_chunks() -> None:
    assert ParagraphChunker().split(document("d", "  \n\n  ")) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunker_settings(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must"):
        ParagraphChunker(size, overlap)
