"""Composition root: the only module that knows every concrete adapter.

Each provider family has a registry mapping a configuration value to a factory.
Adding a provider means registering one factory; the application core and the
HTTP layer do not change (Open/Closed principle). Dependencies are injected
through constructors, so tests build the same services with in-memory adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

from rag_engine.adapters.chunking.paragraph import ParagraphChunker
from rag_engine.adapters.embeddings.hashing import HashingEmbedder
from rag_engine.adapters.keyword.bm25 import BM25Index
from rag_engine.adapters.llm.extractive import ExtractiveLLM
from rag_engine.adapters.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleEmbedder,
    OpenAICompatibleLLM,
)
from rag_engine.adapters.vectorstores.memory import InMemoryVectorStore
from rag_engine.application.answering import AnswerService
from rag_engine.application.ingestion import IngestionService
from rag_engine.application.retrieval import DenseRetriever, HybridRetriever, KeywordRetriever
from rag_engine.config import Settings
from rag_engine.domain import RetrievalMode
from rag_engine.ports import LLM, Embedder, KeywordIndex, Retriever, VectorStore


class UnknownProviderError(ValueError):
    pass


@dataclass
class Registry[T]:
    """Named factories for one port."""

    kind: str
    factories: dict[str, Callable[[Container], T]] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[Callable[[Container], T]], Callable[[Container], T]]:
        def decorator(factory: Callable[[Container], T]) -> Callable[[Container], T]:
            self.factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, container: Container) -> T:
        try:
            factory = self.factories[name]
        except KeyError:
            known = ", ".join(sorted(self.factories))
            raise UnknownProviderError(f"unknown {self.kind} '{name}' (known: {known})") from None
        return factory(container)


EMBEDDERS: Registry[Embedder] = Registry("embedder")
STORES: Registry[tuple[VectorStore, KeywordIndex]] = Registry("store")
LLMS: Registry[LLM] = Registry("llm")


@EMBEDDERS.register("hashing")
def _hashing(c: Container) -> Embedder:
    return HashingEmbedder(c.settings.hashing_dimension)


@EMBEDDERS.register("fastembed")
def _fastembed(c: Container) -> Embedder:  # pragma: no cover - needs a model download
    from rag_engine.adapters.embeddings.fastembed import FastEmbedEmbedder

    return FastEmbedEmbedder(c.settings.embedding_model)


@EMBEDDERS.register("openai_compatible")
def _openai_embedder(c: Container) -> Embedder:
    return OpenAICompatibleEmbedder(
        c.http_client, c.settings.embedding_model, c.settings.embedding_dimension
    )


@STORES.register("memory")
def _memory(c: Container) -> tuple[VectorStore, KeywordIndex]:
    return InMemoryVectorStore(c.embedder.dimension), BM25Index()


@STORES.register("pgvector")
def _pgvector(c: Container) -> tuple[VectorStore, KeywordIndex]:  # pragma: no cover - needs DB
    from rag_engine.adapters.vectorstores.postgres import (
        PgVectorStore,
        PostgresDatabase,
        PostgresFullTextIndex,
    )

    db = PostgresDatabase(c.settings.database_url, c.embedder.dimension)
    return PgVectorStore(db, ef_search=c.settings.hnsw_ef_search), PostgresFullTextIndex(db)


@LLMS.register("extractive")
def _extractive(c: Container) -> LLM:  # noqa: ARG001 - factory signature
    return ExtractiveLLM()


@LLMS.register("openai_compatible")
def _openai_llm(c: Container) -> LLM:
    return OpenAICompatibleLLM(c.http_client, c.settings.llm_model)


class Container:
    """Lazily builds each component once, from `Settings`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @cached_property
    def http_client(self) -> OpenAICompatibleClient:
        key = self.settings.api_key.get_secret_value() if self.settings.api_key else None
        return OpenAICompatibleClient(self.settings.api_base_url, key)

    @cached_property
    def embedder(self) -> Embedder:
        return EMBEDDERS.create(self.settings.embedder, self)

    @cached_property
    def _stores(self) -> tuple[VectorStore, KeywordIndex]:
        return STORES.create(self.settings.store, self)

    @property
    def vector_store(self) -> VectorStore:
        return self._stores[0]

    @property
    def keyword_index(self) -> KeywordIndex:
        return self._stores[1]

    @cached_property
    def llm(self) -> LLM:
        return LLMS.create(self.settings.llm, self)

    def retriever(self, mode: RetrievalMode | None = None) -> Retriever:
        mode = mode or self.settings.retrieval_mode
        dense = DenseRetriever(self.embedder, self.vector_store)
        keyword = KeywordRetriever(self.keyword_index)
        if mode is RetrievalMode.DENSE:
            return dense
        if mode is RetrievalMode.KEYWORD:
            return keyword
        return HybridRetriever([keyword, dense])

    @cached_property
    def ingestion(self) -> IngestionService:
        chunker = ParagraphChunker(self.settings.chunk_max_words, self.settings.chunk_overlap_words)
        return IngestionService(chunker, self.embedder, self.vector_store, self.keyword_index)

    def answering(self, mode: RetrievalMode | None = None) -> AnswerService:
        return AnswerService(self.retriever(mode), self.llm, top_k=self.settings.top_k)
