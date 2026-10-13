# 3. Hybrid retrieval with Reciprocal Rank Fusion

- Status: accepted
- Date: 2026-10-13

## Context
Keyword search (BM25) is precise on exact terms (product names, error codes such as "429"), while dense retrieval matches paraphrases. Their scores live on incompatible scales: BM25 is unbounded, cosine similarity lies in [-1, 1].

## Decision
Fuse ranked lists with Reciprocal Rank Fusion: `score(d) = Σ w_i / (60 + rank_i(d))`, over the top 50 candidates of each retriever. `HybridRetriever` is a Composite: it implements the same `Retriever` port as its children.

## Alternatives considered
- Linear combination of normalised scores: needs per-corpus calibration and is sensitive to score distributions.
- Learned re-ranking (cross-encoder): better quality but adds a model and roughly 10× the latency. Kept on the roadmap as a separate `Reranker` port.

## Consequences
No calibration and no training. The evaluation shows the limit: with equal weights, fusing a strong retriever with a weak one can score *below* the strong one alone (BM25 + hashing on SciFact). Weights are configurable, and the evaluation is what decides them.
