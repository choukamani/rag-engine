from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import respx

from rag_engine.adapters.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleEmbedder,
    OpenAICompatibleLLM,
)

BASE = "https://llm.example/v1"


@respx.mock
def test_llm_sends_chat_completion_and_reads_content() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Hi [1]"}}]})
    )
    llm = OpenAICompatibleLLM(OpenAICompatibleClient(BASE, "sk-test"), "small-model")
    assert llm.generate("prompt") == "Hi [1]"
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "small-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "prompt"}]


@respx.mock
def test_null_content_becomes_empty_string() -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    )
    assert OpenAICompatibleLLM(OpenAICompatibleClient(BASE), "m").generate("p") == ""


@respx.mock
def test_http_errors_propagate() -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(429))
    client = OpenAICompatibleClient(BASE)
    with pytest.raises(httpx.HTTPStatusError):
        OpenAICompatibleLLM(client, "m").generate("p")
    client.close()


@respx.mock
def test_embedder_orders_by_index_and_normalises() -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}]},
        )
    )
    embedder = OpenAICompatibleEmbedder(OpenAICompatibleClient(BASE), "emb", dimension=2)
    matrix = embedder.embed_documents(["first", "second"])
    np.testing.assert_allclose(matrix, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
    assert embedder.dimension == 2
    assert embedder.embed_documents([]).shape == (0, 2)


@respx.mock
def test_embedder_detects_dimension_mismatch() -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3]}]})
    )
    embedder = OpenAICompatibleEmbedder(OpenAICompatibleClient(BASE), "emb", dimension=2)
    with pytest.raises(ValueError, match="3 dimensions"):
        embedder.embed_query("x")
