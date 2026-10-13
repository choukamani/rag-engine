# rag-engine: retrieval-augmented generation with swappable providers and a reproducible evaluation

[![CI](https://github.com/choukamani/rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/choukamani/rag-engine/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> **Demonstration project.** An original, generic implementation of RAG patterns, written for my portfolio. It is not a production system and contains no code or data from any employer.

A typed Python RAG engine built on **ports and adapters**. The LLM, the embedding model, the vector store and the keyword index are each one configuration value away from being replaced. It offers BM25, dense and hybrid (RRF) retrieval, answers with **validated citations**, and an evaluation on a **public benchmark** whose numbers anyone can reproduce.

## Problem

Most RAG prototypes are a chain of framework calls that nobody can measure: which retriever is better on *our* documents? Does hybrid search help? What does the approximate index cost in recall? And they are welded to one provider, although teams typically switch LLM or vector database several times (cost, data residency, quality).

`rag-engine` answers those questions with code that stays small enough to read:
- **Swappable providers** behind six ports: OpenAI, Mistral, Ollama or any OpenAI-compatible API; local ONNX embeddings; PostgreSQL + pgvector or in-memory.
- **Three retrieval strategies** evaluated on the same indexes, with standard IR metrics.
- **Grounded answers:** citations are checked against the sources actually given to the model, and the engine refuses to answer without evidence instead of calling the LLM blind.
- **Runs offline by default:** no API key is needed for the demo, the tests or the CI.

## Demo

```bash
make up            # or: make up ENGINE=podman  → PostgreSQL + pgvector, API, demo corpus loaded
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Can Nimbus employees read my files?"}'
```

Real response from the containerised stack (PostgreSQL + pgvector, offline adapters; `contexts` shortened):

```json
{
  "answer": "Vault files are encrypted on your device before upload (end-to-end encryption), so Nimbus staff and infrastructure cannot read them. [4]",
  "grounded": true,
  "citations": [{"index": 4, "document_id": "encryption", "title": "How your data is encrypted"}],
  "contexts": [{"document_id": "offline-mobile", "score": 0.032018, "source": "rrf"}, "..."]
}
```

The default answerer is extractive (it quotes the best source sentence). Set `RAG_LLM=openai_compatible` with an API key, or point `RAG_API_BASE_URL` at a local Ollama, to generate answers with a real model through the same prompt and citation checks. Interactive API docs are served at `/docs`.

## Quick start (3 commands)

```bash
git clone https://github.com/choukamani/rag-engine && cd rag-engine
cp .env.example .env
make up
```

Without containers: `uv sync --all-extras && uv run uvicorn rag_engine.api:app`, then `uv run python scripts/seed.py` (in-memory store by default).

## Architecture

```mermaid
flowchart LR
    subgraph Driving["Driving adapters"]
        API[FastAPI<br/>api.py]
        CLI[rag-eval CLI]
    end
    subgraph Core["Application core (no framework, no I/O)"]
        ING[IngestionService]
        ANS[AnswerService<br/>prompt · citations · refusal]
        RET{{Retriever port}}
        KW[KeywordRetriever]
        DN[DenseRetriever]
        HY[HybridRetriever<br/>RRF composite]
        HY --> KW & DN
        ANS --> RET
        RET -.implemented by.-> KW & DN & HY
    end
    subgraph Ports
        P1[[Chunker]] & P2[[Embedder]] & P3[[VectorStore]] & P4[[KeywordIndex]] & P5[[LLM]]
    end
    subgraph Adapters["Driven adapters"]
        A1[ParagraphChunker]
        A2[Hashing · fastembed bge-small · OpenAI-compatible]
        A3[In-memory exact · pgvector HNSW]
        A4[BM25 · PostgreSQL full-text]
        A5[Extractive · OpenAI / Mistral / Ollama]
    end
    API & CLI --> ING & ANS
    ING --> P1 & P2 & P3 & P4
    DN --> P2 & P3
    KW --> P4
    ANS --> P5
    P1 --- A1
    P2 --- A2
    P3 --- A3
    P4 --- A4
    P5 --- A5
    C[container.py<br/>composition root + registries] -. builds .-> Adapters
```

The core depends only on `Protocol`s in [`ports/`](src/rag_engine/ports/__init__.py). [`container.py`](src/rag_engine/container.py) is the single place that knows concrete classes ([ADR 2](docs/adr/0002-hexagonal-architecture.md)).

## Design patterns used

| Pattern | Where | Why |
|---|---|---|
| Ports & Adapters (hexagonal) | [`ports/`](src/rag_engine/ports/__init__.py), [`adapters/`](src/rag_engine/adapters/) | The core never imports pgvector, httpx or fastembed; every technology is replaceable. |
| Dependency injection + composition root | [`container.py`](src/rag_engine/container.py) | Constructor injection, wired in one place from typed settings; tests inject in-memory adapters. |
| Registry + Factory | `Registry[T]`, `EMBEDDERS` / `STORES` / `LLMS` in [`container.py`](src/rag_engine/container.py) | A provider is added by registering a factory; nothing else changes (Open/Closed). |
| Strategy | [`DenseRetriever`, `KeywordRetriever`](src/rag_engine/application/retrieval.py), [`ParagraphChunker`](src/rag_engine/adapters/chunking/paragraph.py) | Interchangeable retrieval and chunking algorithms behind one interface. |
| Composite | [`HybridRetriever`](src/rag_engine/application/retrieval.py) | A retriever made of retrievers, fused with RRF, usable wherever a retriever is. |
| Adapter | [`openai_compatible.py`](src/rag_engine/adapters/openai_compatible.py) | One adapter covers OpenAI, Mistral, Ollama and vLLM, which share the same HTTP contract. |
| Template (prompt) | [`build_prompt`](src/rag_engine/application/answering.py) | One grounded-prompt contract shared by every LLM, with numbered sources. |
| Repository | [`PgVectorStore`](src/rag_engine/adapters/vectorstores/postgres.py), [`InMemoryVectorStore`](src/rag_engine/adapters/vectorstores/memory.py) | Persistence hidden behind `upsert` / `search` / `delete_document`. |

## Measured results

Every number comes from scripts in this repository; raw JSON with environment details is in [`evaluation/results/`](evaluation/results/).

### Retrieval quality: BEIR SciFact (public benchmark)

5 183 scientific abstracts, 300 test questions with expert relevance judgements. Reproduce with `make eval-scifact` (downloads the dataset, verified by SHA-256, and the bge-small ONNX model).

| Retrieval | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 query |
|---|---:|---:|---:|---:|---:|
| BM25 (own implementation) | 0.690 | 0.821 | 0.921 | 0.654 | 9 ms |
| Dense, hashing embedder (no model) | 0.401 | 0.507 | 0.682 | 0.369 | 3 ms |
| Hybrid RRF, BM25 + hashing | 0.549 | 0.693 | 0.907 | 0.507 | 14 ms |
| Dense, bge-small-en-v1.5 (local ONNX) | 0.713 | 0.836 | 0.945 | 0.682 | 90 ms |
| **Hybrid RRF, BM25 + bge-small** | **0.728** | **0.861** | **0.968** | **0.693** | 115 ms |

What this shows:
- **The BM25 implementation is sound:** 0.690 nDCG@10, in line with the BM25 baseline published with BEIR (0.665 with Elasticsearch; the gap comes from the analyzer and multi-field indexing).
- **Hybrid wins when both retrievers are good:** BM25 + bge-small beats each alone, with +3.5 points of Recall@100 over dense only. That matters for RAG, because a document that is not retrieved cannot be cited.
- **Hybrid loses when one side is weak:** equal-weight RRF with the hashing embedder scores *below* BM25 alone. Fusion is not free; the evaluation, not intuition, should decide weights ([ADR 3](docs/adr/0003-hybrid-retrieval-with-rrf.md)).
- **Latency is dominated by the query embedding** (bge-small on CPU), not by the search itself.

Measured on a laptop CPU (Windows 11, Python 3.12), with the in-memory exact store.

### Bundled Nimbus set (offline, runs in CI)

31 help-center articles of a fictional product and 40 questions, written for this repository and mixing literal and paraphrased questions (`make eval`).

| Retrieval | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.815 | 0.950 | 0.786 |
| Dense, hashing | 0.657 | 0.838 | 0.607 |
| Hybrid RRF, BM25 + hashing | 0.758 | 0.925 | 0.720 |
| Dense, bge-small | **0.882** | **0.975** | **0.874** |
| Hybrid RRF, BM25 + bge-small | 0.867 | 0.950 | 0.847 |

On paraphrased questions ("Can I open my documents on the plane without wifi?") the neural embedder clearly beats lexical search. The set is small and author-written, so treat it as a regression test rather than a benchmark.

### pgvector HNSW versus exact search

SciFact chunks (5 297 × 512-d hashing vectors), 300 queries, top-10. Measured inside a Linux container on the same network as PostgreSQL 16 + pgvector 0.8.2 (`benchmarks/hnsw_vs_exact.py`). "Recall" is the share of the exact top-10 that the database returns.

| Method | Plan chosen by PostgreSQL | Recall@10 vs exact | p50 | p95 |
|---|---|---:|---:|---:|
| `SELECT 1` (round trip only) | - | - | 0.2 ms | 0.5 ms |
| Exact search, NumPy in memory | brute force | 1.000 | 4.7 ms | 18.0 ms |
| pgvector, `hnsw.ef_search=10` | HNSW index scan | 0.441 | 3.3 ms | 6.6 ms |
| pgvector, `hnsw.ef_search=20` | HNSW index scan | 0.587 | 2.9 ms | 6.6 ms |
| pgvector, `hnsw.ef_search=40` (default) | HNSW index scan | 0.725 | 3.7 ms | 5.9 ms |
| pgvector, `hnsw.ef_search=80` | **sequential scan + sort** | 1.000 | 42.7 ms | 77.5 ms |
| pgvector, `hnsw.ef_search=160` | **sequential scan + sort** | 1.000 | 36.0 ms | 67.6 ms |

Two lessons I did not expect before measuring:
1. **The planner can silently bypass the index.** On this 5 k-row table, from `ef_search=80` the estimated cost of the HNSW scan exceeds a sequential scan + sort, so PostgreSQL returns *exact* results about 10× slower. Always check plans with `EXPLAIN`. The benchmark records the plan for each setting.
2. **HNSW recall depends on the embedding geometry.** The hashing vectors are sparse and poorly clustered, a hard case for a proximity graph (0.73 recall at the default `ef_search`). Dense neural embeddings are the intended input; measuring them is on the roadmap (the run needs about 40 minutes of CPU embedding on a laptop).

Latencies come from a shared laptop (Windows 11 + WSL2, other containers running) and vary between runs. Earlier runs from the Windows host showed about 200 ms p95 spikes that disappeared inside Linux: they came from the Windows-to-WSL network relay, not from PostgreSQL.

**Tests:** 69 tests (unit, HTTP end-to-end, PostgreSQL + pgvector integration), **98 % branch coverage**, `mypy --strict` clean.

## Technical choices and trade-offs

| Choice | Rationale | Trade-off |
|---|---|---|
| Hexagonal core, no RAG framework ([ADR 2](docs/adr/0002-hexagonal-architecture.md)) | Every step is visible, testable and measurable; providers are swappable | More code than a LangChain one-liner; streaming and tool calls not exposed yet |
| PostgreSQL + pgvector, one table ([ADR 4](docs/adr/0004-postgresql-pgvector-single-table.md)) | One database to operate; chunk, vector and `tsvector` written atomically | `ts_rank_cd` is not BM25; very large scale may justify a dedicated engine |
| HNSW (m=16, ef_construction=64) | Sub-linear search, no training step, incremental inserts | Approximate; recall depends on `ef_search` (see benchmark) |
| Reciprocal Rank Fusion ([ADR 3](docs/adr/0003-hybrid-retrieval-with-rrf.md)) | No score calibration, no training | Can hurt when one retriever is weak |
| bge-small via fastembed (ONNX) | Free, local, about 130 MB, good quality per size | About 0.1 s per query embedding on a laptop CPU |
| Paragraph chunking (400 words, 40 overlap) | Keeps semantic units; titles prepended for context | No layout awareness (tables, PDFs) |
| Offline extractive answerer ([ADR 5](docs/adr/0005-offline-adapters-and-reproducible-evaluation.md)) | Demo and CI without keys; deterministic | Quotes, does not synthesise across sources |

## Known limitations

- Generation quality (faithfulness, answer relevance) is not evaluated yet. The metrics above measure retrieval.
- No re-ranker, no query rewriting, no metadata filtering in the API.
- No authentication or multi-tenant isolation on the API (see my `agent-platform` project for that concern).
- The PostgreSQL deployment uses full-text ranking, not BM25, so its keyword quality differs from the in-memory reference.
- Chunking is text-only; PDF/HTML parsing is left to the caller.

## Roadmap

- `Reranker` port with a cross-encoder adapter, evaluated on SciFact.
- HNSW recall/latency benchmark with bge-small embeddings and larger corpora (where the index, not a sequential scan, is chosen).
- LLM-as-judge faithfulness and citation-precision evaluation for real LLMs.
- Metadata filters pushed down to SQL; per-tenant tables or row-level security.
- Streaming answers (SSE) and OpenTelemetry spans per pipeline stage.

## Project layout

```
src/rag_engine/
├── domain/            # Document, Chunk, ScoredChunk, Answer, Citation
├── ports/             # Protocols: Chunker, Embedder, VectorStore, KeywordIndex, Retriever, LLM
├── application/       # ingestion, retrieval strategies (incl. RRF), grounded answering
├── adapters/          # chunking, embeddings, vector stores, keyword indexes, LLMs
├── container.py       # composition root + provider registries
├── config.py          # typed settings (RAG_* environment variables)
├── api.py             # FastAPI driving adapter
└── evaluation/        # BEIR loader, metrics, runner, rag-eval CLI
evaluation/            # Nimbus dataset source + results JSON
benchmarks/            # pgvector HNSW vs exact search
docs/adr/              # architecture decision records
```

## Development

```bash
make install                         # uv sync --all-extras + pre-commit hooks
docker compose up -d postgres        # optional, for the pgvector tests
RAG_TEST_DATABASE_URL=postgresql://rag:rag@127.0.0.1:5432/rag make check
make eval                            # Nimbus retrieval metrics
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, strict typing and all tests against a pgvector service container, publishes the Nimbus evaluation in the job summary, then builds the image, starts the stack and asks a question end to end.

---

## Résumé en français

**rag-engine** est un moteur RAG en Python typé, construit en architecture hexagonale. Le LLM (OpenAI, Mistral, Ollama…), le modèle d'embeddings (hashing, bge-small en ONNX local, API) et le stockage (mémoire ou PostgreSQL + pgvector HNSW) se remplacent par simple configuration, grâce à des ports, des adaptateurs et une racine de composition avec registres.

- **Recherche :** trois stratégies (BM25 implémenté à la main, dense, hybride par fusion RRF).
- **Réponses :** citations vérifiées, et refus de répondre sans preuve.
- **Démo et CI :** tournent sans clé API.
- **Résultats mesurés sur le benchmark public BEIR SciFact (nDCG@10) :**
  - BM25 0,690, cohérent avec la référence publiée ;
  - dense bge-small 0,713 ;
  - hybride 0,728, le meilleur ;
  - en revanche, fusionner BM25 avec un retriever faible dégrade le résultat, et l'évaluation le montre.
- **Benchmark pgvector :** il mesure le compromis rappel/latence de l'index HNSW et montre quand PostgreSQL bascule vers un scan exact.
- **Projet de démonstration :** réimplémentation originale et générique, sans code ni donnée d'un employeur.

## License

[MIT](LICENSE) © 2026 Amani Chouk
