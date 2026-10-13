"""HTTP entry point (driving adapter). Thin: validates input, delegates to use cases."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from rag_engine.application.retrieval import deduplicate_by_document
from rag_engine.container import Container
from rag_engine.domain import Document, RetrievalMode, ScoredChunk


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    title: str = ""
    metadata: dict[str, str] = {}


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=1_000)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    k: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode | None = None


class Hit(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    text: str
    score: float
    source: str


class SearchResponse(BaseModel):
    hits: list[Hit]
    latency_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    mode: RetrievalMode | None = None


class CitationOut(BaseModel):
    index: int
    document_id: str
    title: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationOut]
    contexts: list[Hit]
    latency_ms: float


def _hit(h: ScoredChunk) -> Hit:
    c = h.chunk
    return Hit(
        document_id=c.document_id, chunk_id=c.id, title=c.title, text=c.text,
        score=round(h.score, 6), source=h.source,
    )  # fmt: skip


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container()
        yield

    app = FastAPI(
        title="rag-engine",
        version="0.1.0",
        description="RAG engine with swappable providers. Demonstration project.",
        lifespan=lifespan,
    )

    @app.post("/documents", status_code=201)
    def ingest(body: IngestRequest, c: ContainerDep) -> IngestResponse:
        docs = [Document(d.id, d.text, d.title, d.metadata) for d in body.documents]
        report = c.ingestion.ingest(docs)
        return IngestResponse(documents=report.documents, chunks=report.chunks)

    @app.delete("/documents/{document_id}")
    def delete(document_id: str, c: ContainerDep) -> dict[str, int]:
        return {"deleted_chunks": c.ingestion.delete(document_id)}

    @app.post("/search")
    def search(body: SearchRequest, c: ContainerDep) -> SearchResponse:
        start = time.perf_counter()
        hits = c.retriever(body.mode).retrieve(body.query, body.k * 3)
        hits = deduplicate_by_document(hits, body.k)
        latency = (time.perf_counter() - start) * 1000
        return SearchResponse(hits=[_hit(h) for h in hits], latency_ms=round(latency, 2))

    @app.post("/ask")
    def ask(body: AskRequest, c: ContainerDep) -> AskResponse:
        start = time.perf_counter()
        answer = c.answering(body.mode).answer(body.question)
        latency = (time.perf_counter() - start) * 1000
        return AskResponse(
            answer=answer.text,
            grounded=answer.grounded,
            citations=[CitationOut(index=x.index, document_id=x.document_id, title=x.title)
                       for x in answer.citations],
            contexts=[_hit(h) for h in answer.contexts],
            latency_ms=round(latency, 2),
        )  # fmt: skip

    @app.get("/health")
    def health(c: ContainerDep) -> dict[str, str | int]:
        s = c.settings
        return {
            "status": "ok",
            "chunks": c.vector_store.count(),
            "embedder": s.embedder,
            "store": s.store,
            "llm": s.llm,
        }

    return app


app = create_app()
