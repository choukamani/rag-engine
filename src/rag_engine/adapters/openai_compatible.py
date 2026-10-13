"""Adapters for any OpenAI-compatible HTTP API: OpenAI, Mistral, Ollama (`/v1`), vLLM...

One adapter covers several providers because they share the same
`/chat/completions` and `/embeddings` contracts; only `base_url`, the key and
the model name change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from rag_engine.ports import Matrix, Vector


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s, transport=transport
        )

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(path, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def close(self) -> None:
        self._http.close()


class OpenAICompatibleLLM:
    def __init__(
        self, client: OpenAICompatibleClient, model: str, *, temperature: float = 0.0
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        data = self._client.post(
            "/chat/completions",
            {
                "model": self._model,
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        content = data["choices"][0]["message"]["content"]
        return str(content or "")


class OpenAICompatibleEmbedder:
    def __init__(self, client: OpenAICompatibleClient, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        data = self._client.post("/embeddings", {"model": self._model, "input": list(texts)})
        rows = sorted(data["data"], key=lambda item: item["index"])
        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if matrix.shape[1] != self._dimension:
            raise ValueError(
                f"model returned {matrix.shape[1]} dimensions, configured {self._dimension}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized: Matrix = (matrix / norms).astype(np.float32)
        return normalized

    def embed_query(self, text: str) -> Vector:
        row: Vector = self.embed_documents([text])[0]
        return row
