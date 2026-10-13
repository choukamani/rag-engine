"""Local neural embeddings with fastembed (ONNX Runtime, CPU), no API key needed.

Requires the `local-embeddings` extra. The model (about 130 MB for
bge-small-en-v1.5) is downloaded once into the fastembed cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rag_engine.ports import Matrix, Vector

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder:  # pragma: no cover - exercised by the evaluation, needs a model download
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        query_prefix: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from fastembed import TextEmbedding

        self._model: Any = TextEmbedding(model_name=model_name)
        self._batch_size = batch_size
        # BGE models are trained with an instruction prefix on queries only.
        self._query_prefix = (
            query_prefix if query_prefix is not None else
            (BGE_QUERY_PREFIX if "bge" in model_name.lower() else "")
        )  # fmt: skip
        self._dimension = int(self.embed_query("dimension probe").shape[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        rows = list(self._model.embed(list(texts), batch_size=self._batch_size))
        return _normalize(np.vstack(rows).astype(np.float32))

    def embed_query(self, text: str) -> Vector:
        row = next(iter(self._model.embed([self._query_prefix + text])))
        vector: Vector = _normalize(np.asarray(row, dtype=np.float32)[None, :])[0]
        return vector


def _normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: Matrix = (matrix / norms).astype(np.float32)
    return normalized
