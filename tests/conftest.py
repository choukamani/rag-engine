from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.config import Settings
from rag_engine.container import Container
from rag_engine.domain import Chunk, Document
from rag_engine.evaluation.datasets import Dataset, load_beir

NIMBUS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "nimbus"


@pytest.fixture(scope="session")
def nimbus() -> Dataset:
    return load_beir(NIMBUS, name="nimbus")


@pytest.fixture
def container() -> Container:
    return Container(Settings(_env_file=None))  # type: ignore[call-arg]


@pytest.fixture
def loaded(container: Container, nimbus: Dataset) -> Container:
    container.ingestion.ingest(nimbus.documents)
    return container


def chunk(doc: str, text: str, position: int = 0, title: str = "") -> Chunk:
    return Chunk(Chunk.make_id(doc, position), doc, text, position, title)


def document(doc_id: str, text: str, title: str = "") -> Document:
    return Document(doc_id, text, title)
