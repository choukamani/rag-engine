# 2. Hexagonal architecture with an explicit composition root

- Status: accepted
- Date: 2026-10-13

## Context
RAG stacks change fast: the embedding model, the LLM provider and the vector database are each replaced several times in a product's life, often for cost or data-residency reasons. Frameworks such as LangChain make swapping easy but hide the retrieval logic behind deep abstractions, which makes it hard to measure, test and explain.

## Decision
- The core (`domain/`, `application/`) depends only on small `Protocol`s in `ports/`: `Chunker`, `Embedder`, `VectorStore`, `KeywordIndex`, `Retriever`, `LLM`.
- Every technology lives in `adapters/` and implements one port.
- `container.py` is the only module that imports concrete adapters. It maps configuration values to factories through one registry per port, and injects dependencies through constructors (no DI framework, no global state).
- The HTTP layer (`api.py`) is a thin driving adapter that only calls use cases.

## Consequences
- Swapping a provider is a configuration change (`RAG_EMBEDDER`, `RAG_STORE`, `RAG_LLM`). Adding one is one class plus one registered factory.
- Every use case is unit-tested with in-memory adapters; the PostgreSQL adapters are tested separately against a real database.
- More files than a single-script RAG, and some features of provider SDKs (streaming, tool calls) are not exposed until a port needs them.
