"""End-to-end through HTTP with the default offline adapters (hashing, memory, extractive)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.container import Container, UnknownProviderError
from rag_engine.evaluation.datasets import Dataset

pytestmark = pytest.mark.integration


@pytest.fixture
def client(container: Container, nimbus: Dataset) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        documents = [{"id": d.id, "title": d.title, "text": d.text} for d in nimbus.documents]
        response = test_client.post("/documents", json={"documents": documents})
        assert response.status_code == 201
        assert response.json() == {"documents": 31, "chunks": 31}
        yield test_client


@pytest.mark.parametrize("mode", ["keyword", "dense", "hybrid"])
def test_search_modes(client: TestClient, mode: str) -> None:
    response = client.post(
        "/search", json={"query": "How much does the Plus plan cost?", "k": 3, "mode": mode}
    )
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 3
    assert len({h["document_id"] for h in hits}) == 3
    assert hits[0]["document_id"] == "plans-pricing"


def test_ask_returns_a_grounded_cited_answer(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "How much does the Plus plan cost?"}).json()
    assert body["grounded"] is True
    assert "9.99 euros" in body["answer"]
    assert body["citations"][0]["document_id"] == "plans-pricing"
    assert body["latency_ms"] >= 0


def test_ask_declines_out_of_scope_questions(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "Quelle est la capitale du Pérou ?"}).json()
    assert body["grounded"] is False


def test_reingest_is_idempotent_and_delete_works(client: TestClient) -> None:
    doc = {"id": "plans-pricing", "title": "Plans", "text": "Plus now costs 12 euros."}
    client.post("/documents", json={"documents": [doc]})
    assert client.get("/health").json()["chunks"] == 31
    answer = client.post("/ask", json={"question": "How much does Plus cost?"}).json()
    assert "12 euros" in answer["answer"]
    assert client.delete("/documents/plans-pricing").json() == {"deleted_chunks": 1}
    assert client.get("/health").json()["chunks"] == 30


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 500}).status_code == 422
    assert client.post("/documents", json={"documents": []}).status_code == 422


def test_health_reports_providers(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok", "chunks": 31, "embedder": "hashing", "store": "memory", "llm": "extractive"
    }  # fmt: skip


def test_container_rejects_unknown_provider() -> None:
    settings = Settings.model_construct(embedder="word2vec")  # bypass validation on purpose
    with pytest.raises(UnknownProviderError, match="unknown embedder 'word2vec'"):
        _ = Container(settings).embedder


def test_container_wires_openai_compatible_providers() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        embedder="openai_compatible", llm="openai_compatible", embedding_dimension=8,
        api_key="sk-test",  # type: ignore[arg-type]
    )  # fmt: skip
    container = Container(settings)
    assert container.embedder.dimension == 8
    assert type(container.llm).__name__ == "OpenAICompatibleLLM"
