# 5. Offline default adapters and a reproducible evaluation

- Status: accepted
- Date: 2026-10-13

## Context
A RAG system is only as good as its retrieval, and retrieval quality is invisible without a test set. Demos that need a paid API key are rarely run by reviewers.

## Decision
- Default adapters need no key and no network: a hashing embedder, an in-memory or PostgreSQL store, and an extractive answerer that quotes the best-matching source sentence with its citation.
- Retrieval is evaluated with standard IR metrics (nDCG@10, Recall@k, MRR@10) on two BEIR-format datasets:
  - **Nimbus**: 31 documents and 40 questions written for this repository, offline, run in CI;
  - **SciFact**: the public BEIR benchmark, downloaded on demand and verified by SHA-256.
- Results are written to versioned JSON files together with their environment.

## Consequences
- Anyone can reproduce every number in the README.
- SciFact lets the BM25 implementation be checked against published baselines.
- The Nimbus set is small and was written by the author, so it is a regression test, not a benchmark.
- The extractive answerer does not measure generation quality: answer faithfulness with a real LLM is out of scope for now (see roadmap).
