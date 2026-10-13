"""Typed configuration from environment variables (prefix `RAG_`) or a `.env` file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_engine.domain import RetrievalMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Providers: each value selects one adapter in the container's registries.
    embedder: Literal["hashing", "fastembed", "openai_compatible"] = "hashing"
    store: Literal["memory", "pgvector"] = "memory"
    llm: Literal["extractive", "openai_compatible"] = "extractive"

    # Retrieval
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=50)
    chunk_max_words: int = Field(default=200, ge=20)
    chunk_overlap_words: int = Field(default=30, ge=0)

    # Embeddings
    hashing_dimension: int = 512
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # PostgreSQL + pgvector
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    hnsw_ef_search: int = 40

    # OpenAI-compatible API (OpenAI, Mistral, Ollama /v1, vLLM...)
    api_base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
