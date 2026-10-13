from __future__ import annotations

import math

import numpy as np
import pytest

from rag_engine.adapters.embeddings.hashing import HashingEmbedder
from rag_engine.adapters.keyword.bm25 import BM25Index
from rag_engine.adapters.vectorstores.memory import InMemoryVectorStore
from rag_engine.ports import Embedder, KeywordIndex, VectorStore
from tests.conftest import chunk


def test_adapters_satisfy_ports() -> None:
    assert isinstance(HashingEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(4), VectorStore)
    assert isinstance(BM25Index(), KeywordIndex)


class TestHashingEmbedder:
    def test_vectors_are_normalised_and_deterministic(self) -> None:
        e = HashingEmbedder(64)
        a = e.embed_query("restore deleted files")
        assert a.shape == (64,)
        assert np.linalg.norm(a) == pytest.approx(1.0, rel=1e-5)
        np.testing.assert_array_equal(a, HashingEmbedder(64).embed_query("restore deleted files"))

    def test_lexical_similarity(self) -> None:
        e = HashingEmbedder(256)
        q = e.embed_query("restore deleted files")
        close, far = e.embed_documents(["How to restore a deleted file", "Pricing of plans"])
        assert float(q @ close) > float(q @ far)

    def test_empty_inputs(self) -> None:
        e = HashingEmbedder(8)
        assert e.embed_documents([]).shape == (0, 8)
        assert not e.embed_query("the of and").any()  # only stop words

    def test_unigram_only_mode_and_validation(self) -> None:
        assert HashingEmbedder(16, bigrams=False).dimension == 16
        with pytest.raises(ValueError, match="positive"):
            HashingEmbedder(0)


class TestInMemoryVectorStore:
    def _store(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore(2)
        chunks = [chunk("a", "x"), chunk("b", "y"), chunk("c", "z")]
        vectors = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
        store.upsert(chunks, vectors)
        return store

    def test_search_orders_by_cosine(self) -> None:
        hits = self._store().search(np.array([1, 0], dtype=np.float32), 2)
        assert [h.chunk.document_id for h in hits] == ["a", "c"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].source == "dense"

    def test_upsert_replaces_existing_ids(self) -> None:
        store = self._store()
        store.upsert([chunk("a", "x2")], np.array([[0, 1]], dtype=np.float32))
        assert store.count() == 3
        top = store.search(np.array([0, 1], dtype=np.float32), 3)
        assert {h.chunk.text for h in top} >= {"x2"}
        assert "x" not in {h.chunk.text for h in top}

    def test_delete_document_and_k_bounds(self) -> None:
        store = self._store()
        assert store.delete_document("a") == 1
        assert store.delete_document("missing") == 0
        assert store.count() == 2
        hits = store.search(np.array([1, 0], dtype=np.float32), 10)
        assert [h.chunk.document_id for h in hits] == ["c", "b"]
        assert store.search(np.array([1, 0], dtype=np.float32), 0) == []
        assert InMemoryVectorStore(2).search(np.array([1, 0], dtype=np.float32), 3) == []

    def test_validation(self) -> None:
        store = InMemoryVectorStore(2)
        with pytest.raises(ValueError, match="same length"):
            store.upsert([chunk("a", "x")], np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="2-d"):
            store.upsert([chunk("a", "x")], np.zeros((1, 3), dtype=np.float32))


class TestBM25:
    def _index(self) -> BM25Index:
        index = BM25Index()
        index.add(
            [
                chunk("trash", "Deleted files stay in the trash for 30 days.", title="Trash"),
                chunk("price", "The Plus plan costs 9.99 euros per month.", title="Pricing"),
                chunk("sync", "The desktop app syncs files and folders.", title="Sync"),
            ]
        )
        return index

    def test_ranks_matching_chunk_first(self) -> None:
        hits = self._index().search("how long do deleted files stay?", 3)
        assert hits[0].chunk.document_id == "trash"
        assert hits[0].source == "bm25"
        assert all(h.score > 0 for h in hits)

    def test_score_matches_the_formula(self) -> None:
        index = BM25Index(k1=1.2, b=0.75)
        index.add([chunk("a", "apple banana"), chunk("b", "cherry")])
        # "appl": tf=1, df=1, N=2, |d|=2, avgdl=1.5
        idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
        expected = idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / 1.5))
        assert index.search("apple", 1)[0].score == pytest.approx(expected)

    def test_title_is_indexed(self) -> None:
        assert self._index().search("pricing", 1)[0].chunk.document_id == "price"

    def test_no_match_and_empty_index(self) -> None:
        assert self._index().search("quantum", 5) == []
        assert BM25Index().search("anything", 5) == []
        assert self._index().search("files", 0) == []

    def test_reindex_and_delete_keep_statistics_consistent(self) -> None:
        index = self._index()
        index.add([chunk("trash", "Completely different text about invoices.")])
        assert len(index) == 3
        assert index.search("invoices", 1)[0].chunk.document_id == "trash"
        assert (
            index.search("30 days", 3) == []
            or index.search("30 days", 3)[0].chunk.document_id != "trash"
        )
        assert index.delete_document("trash") == 1
        assert index.delete_document("trash") == 0
        assert len(index) == 2
        assert index.search("invoices", 3) == []
