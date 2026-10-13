"""Ranking metrics with graded relevance (`qrels`: doc id -> relevance grade, > 0 is relevant).

Definitions follow the BEIR/TREC conventions, so results are comparable to
published numbers (nDCG@10 with linear gains, as computed by pytrec_eval).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    dcg = sum(
        qrels.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking[:k], start=1)
    )
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def hit_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    return 1.0 if any(qrels.get(d, 0) > 0 for d in ranking[:k]) else 0.0
