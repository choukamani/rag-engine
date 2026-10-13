"""Dependency-free embedder based on the hashing trick (feature hashing).

Stemmed unigrams and bigrams are hashed into a fixed number of buckets with a
sign bit, weighted by sublinear term frequency, then L2-normalised. It needs no
model download and is fully deterministic, which makes it ideal for tests and
offline demos. It is a *lexical* representation: it does not capture synonyms,
which the evaluation makes visible.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from rag_engine.ports import Matrix, Vector
from rag_engine.text import tokenize


class HashingEmbedder:
    def __init__(self, dimension: int = 512, *, bigrams: bool = True) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._bigrams = bigrams

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features = Counter(tokens)
        if self._bigrams:
            features.update(f"{a} {b}" for a, b in pairwise(tokens))
        return features

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature, count in self._features(text).items():
            # crc32 is stable across processes, unlike hash() with PYTHONHASHSEED.
            digest = zlib.crc32(feature.encode())
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self._dimension] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)
