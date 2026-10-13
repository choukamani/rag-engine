"""pgvector + PostgreSQL full-text adapters against a real database.

Runs when RAG_TEST_DATABASE_URL points to PostgreSQL with the pgvector
extension available (CI starts one as a service container); skipped otherwise.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from rag_engine.adapters.chunking.paragraph import ParagraphChunker
from rag_engine.adapters.embeddings.hashing import HashingEmbedder
from rag_engine.adapters.keyword.bm25 import BM25Index
from rag_engine.adapters.vectorstores.memory import InMemoryVectorStore
from rag_engine.application.ingestion import IngestionService
from rag_engine.application.retrieval import DenseRetriever, KeywordRetriever
from rag_engine.evaluation.datasets import Dataset
from rag_engine.ports import KeywordIndex, VectorStore
from tests.conftest import chunk

DSN = os.getenv("RAG_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.skipif(DSN is None, reason="RAG_TEST_DATABASE_URL not set"),
]


@pytest.fixture
def pg() -> Iterator[tuple[VectorStore, KeywordIndex]]:
    from rag_engine.adapters.vectorstores.postgres import (
        PgVectorStore,
        PostgresDatabase,
        PostgresFullTextIndex,
    )

    assert DSN is not None
    db = PostgresDatabase(DSN, 512, table=f"test_{uuid.uuid4().hex[:12]}")
    try:
        yield PgVectorStore(db, ef_search=100), PostgresFullTextIndex(db)
    finally:
        with db.connection() as conn:
            conn.execute(f"DROP TABLE {db.table}")
        db.close()


def test_adapters_satisfy_ports(pg: tuple[VectorStore, KeywordIndex]) -> None:
    store, keyword = pg
    assert isinstance(store, VectorStore)
    assert isinstance(keyword, KeywordIndex)


def test_dense_results_match_exact_search(
    pg: tuple[VectorStore, KeywordIndex], nimbus: Dataset
) -> None:
    store, keyword = pg
    embedder = HashingEmbedder(512)
    chunker = ParagraphChunker()
    IngestionService(chunker, embedder, store, keyword).ingest(nimbus.documents)
    exact = InMemoryVectorStore(512)
    IngestionService(chunker, embedder, exact, BM25Index()).ingest(nimbus.documents)

    assert store.count() == 31
    pg_dense, exact_dense = DenseRetriever(embedder, store), DenseRetriever(embedder, exact)
    for query in list(nimbus.queries.values())[:10]:
        expected = [h.chunk.id for h in exact_dense.retrieve(query, 5)]
        got = pg_dense.retrieve(query, 5)
        # On 31 rows with ef_search=100, HNSW must return the exact neighbours.
        assert [h.chunk.id for h in got] == expected
        assert got[0].source == "dense"


def test_full_text_search_and_deletion(pg: tuple[VectorStore, KeywordIndex]) -> None:
    store, keyword = pg
    chunks = [
        chunk("trash", "Deleted files stay in the trash for 30 days.", title="Trash"),
        chunk("price", "The Plus plan costs 9.99 euros per month.", title="Pricing"),
    ]
    store.upsert(chunks, HashingEmbedder(512).embed_documents([c.text for c in chunks]))
    keyword.add(chunks)  # no-op, the tsvector column is generated

    hits = KeywordRetriever(keyword).retrieve("how long are deleted files kept?", 5)
    assert hits[0].chunk.document_id == "trash"
    assert hits[0].source == "fts"
    assert keyword.search("!!!", 5) == []
    assert store.search(np.zeros(512, dtype=np.float32), 0) == []

    # Upsert replaces in place.
    store.upsert([chunk("price", "Now 12 euros.")], HashingEmbedder(512).embed_documents(["x"]))
    assert store.count() == 2
    assert store.delete_document("trash") == 1
    assert keyword.delete_document("trash") == 0
    assert keyword.search("deleted files", 5) == []


def test_mismatched_lengths_are_rejected(pg: tuple[VectorStore, KeywordIndex]) -> None:
    store, _ = pg
    with pytest.raises(ValueError, match="same length"):
        store.upsert([chunk("a", "x")], np.zeros((2, 512), dtype=np.float32))


def test_table_name_is_validated() -> None:
    from rag_engine.adapters.vectorstores.postgres import PostgresDatabase

    assert DSN is not None
    with pytest.raises(ValueError, match="identifier"):
        PostgresDatabase(DSN, 8, table="x; DROP TABLE users")
