# 4. PostgreSQL + pgvector, one table for vectors and full-text

- Status: accepted
- Date: 2026-10-13

## Context
Most teams already run PostgreSQL. A dedicated vector database (Qdrant, Weaviate, Pinecone) adds a service to operate, secure and back up, plus a consistency problem between the document store and the vector index.

## Decision
- Use pgvector with an HNSW index (`vector_cosine_ops`, m = 16, ef_construction = 64, `hnsw.ef_search` set per query with `SET LOCAL`).
- Store chunk text, metadata, embedding and a generated `tsvector` column in **one table**, so a chunk and both of its indexes are written in one transaction and cannot drift apart.
- PostgreSQL full-text search (`ts_rank_cd`, English stemming, OR of the query terms) is the keyword adapter in this deployment. The in-memory BM25 remains the reference implementation in the evaluation.

## Consequences
- One backup, one access-control model, transactional updates, SQL filters on metadata.
- `ts_rank_cd` is not BM25, so keyword ranking quality differs between the memory and PostgreSQL deployments.
- HNSW is approximate and its recall depends on `ef_search`. The benchmark also shows that PostgreSQL's planner may switch to an exact sequential scan on small tables when `ef_search` is high. This is correct but changes the latency profile, so plans must be checked with `EXPLAIN`.
- Beyond tens of millions of vectors, or for multi-tenant sharding, a dedicated engine may become worth its operational cost.
