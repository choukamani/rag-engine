# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-13

### Added
- Ports for chunker, embedder, vector store, keyword index, retriever and LLM; composition root with provider registries.
- Adapters: paragraph chunker, hashing / fastembed / OpenAI-compatible embedders, in-memory exact and pgvector HNSW vector stores, BM25 and PostgreSQL full-text keyword indexes, extractive and OpenAI-compatible LLMs.
- Keyword, dense and hybrid (Reciprocal Rank Fusion) retrieval; grounded answering with validated citations and refusal without evidence.
- FastAPI service (`/documents`, `/search`, `/ask`, `/health`), Docker image and compose stack with PostgreSQL + pgvector.
- `rag-eval` CLI: nDCG@10, Recall@10/100, MRR@10, Hit@5 and latency on the bundled Nimbus set and BEIR SciFact.
- pgvector HNSW versus exact search benchmark; ADRs 1–5; CI with a pgvector service container.
