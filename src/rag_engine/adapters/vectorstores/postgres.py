"""PostgreSQL adapters: pgvector (HNSW) for dense search, built-in full-text search for keywords.

Both views live in **one table**, so a chunk, its embedding and its `tsvector`
are written in a single transaction and can never drift apart. The keyword
column is a generated column, which is why `PostgresFullTextIndex.add` has
nothing to do (see ADR 0004).

Requires the `postgres` extra.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Final

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag_engine.domain import Chunk, ScoredChunk
from rag_engine.ports import Matrix, Vector

_WORD: Final = re.compile(r"[a-z0-9]+")

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS {table} (
    id          text PRIMARY KEY,
    document_id text NOT NULL,
    position    integer NOT NULL,
    title       text NOT NULL DEFAULT '',
    text        text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding   vector({dimension}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS {document_index} ON {table} (document_id);
CREATE INDEX IF NOT EXISTS {hnsw_index} ON {table}
    USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction});
CREATE INDEX IF NOT EXISTS {gin_index} ON {table} USING gin (tsv);
"""


class PostgresDatabase:
    """Connection pool + schema bootstrap shared by the two adapters."""

    def __init__(
        self,
        dsn: str,
        dimension: int,
        *,
        table: str = "rag_chunks",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        pool_size: int = 5,
    ) -> None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", table):
            raise ValueError("table name must be a lowercase SQL identifier")
        self.table = table
        self.dimension = dimension
        # The extension must exist before pooled connections register the vector type.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=pool_size, open=True, configure=self._configure
        )
        self._create_schema(hnsw_m, hnsw_ef_construction)

    @staticmethod
    def _configure(conn: Connection[Any]) -> None:
        conn.autocommit = True
        register_vector(conn)

    def _create_schema(self, m: int, ef_construction: int) -> None:
        t = self.table
        statement = sql.SQL(_SCHEMA).format(
            table=sql.Identifier(t),
            dimension=sql.Literal(self.dimension),
            document_index=sql.Identifier(f"{t}_document_idx"),
            hnsw_index=sql.Identifier(f"{t}_embedding_hnsw"),
            gin_index=sql.Identifier(f"{t}_tsv_gin"),
            m=sql.Literal(m),
            ef_construction=sql.Literal(ef_construction),
        )
        with self._pool.connection() as conn:
            conn.execute(statement)

    @contextmanager
    def connection(self) -> Iterator[Connection[Any]]:
        with self._pool.connection() as conn:
            yield conn

    def ident(self) -> sql.Identifier:
        return sql.Identifier(self.table)

    def close(self) -> None:
        self._pool.close()


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        text=row["text"],
        position=row["position"],
        title=row["title"],
        metadata=dict(row["metadata"]),
    )


class PgVectorStore:
    def __init__(self, db: PostgresDatabase, *, ef_search: int = 40) -> None:
        self._db = db
        self._ef_search = ef_search

    def upsert(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        query = sql.SQL(
            """INSERT INTO {t} (id, document_id, position, title, text, metadata, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                 document_id = EXCLUDED.document_id, position = EXCLUDED.position,
                 title = EXCLUDED.title, text = EXCLUDED.text,
                 metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"""
        ).format(t=self._db.ident())
        rows = [
            (c.id, c.document_id, c.position, c.title, c.text, json.dumps(c.metadata), e)
            for c, e in zip(chunks, embeddings.astype(np.float32), strict=True)
        ]
        with self._db.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(query, rows)

    def search(self, query: Vector, k: int) -> list[ScoredChunk]:
        if k <= 0:
            return []
        statement = sql.SQL(
            """SELECT id, document_id, position, title, text, metadata,
                      1 - (embedding <=> %(q)s) AS score
               FROM {t} ORDER BY embedding <=> %(q)s LIMIT %(k)s"""
        ).format(t=self._db.ident())
        with (
            self._db.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=dict_row) as cur,
        ):
            # Recall/latency knob of the HNSW index, scoped to this transaction.
            cur.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(self._ef_search))
            )
            cur.execute(statement, {"q": query.astype(np.float32), "k": k})
            rows = cur.fetchall()
        return [ScoredChunk(_row_to_chunk(r), float(r["score"]), "dense") for r in rows]

    def delete_document(self, document_id: str) -> int:
        statement = sql.SQL("DELETE FROM {t} WHERE document_id = %s").format(t=self._db.ident())
        with self._db.connection() as conn:
            return conn.execute(statement, (document_id,)).rowcount

    def count(self) -> int:
        with self._db.connection() as conn:
            row = conn.execute(sql.SQL("SELECT count(*) FROM {t}").format(t=self._db.ident()))
            result = row.fetchone()
        return int(result[0]) if result else 0


class PostgresFullTextIndex:
    """Keyword search with PostgreSQL full-text search (`ts_rank_cd`, English stemming).

    Note: `ts_rank_cd` is a cover-density ranking, not BM25. It is the pragmatic
    choice when the data already lives in PostgreSQL; the in-memory `BM25Index`
    is the reference implementation used by the evaluation.
    """

    def __init__(self, db: PostgresDatabase) -> None:
        self._db = db

    def add(self, chunks: Sequence[Chunk]) -> None:
        """No-op: `tsv` is a generated column, filled when the vector store writes the row."""

    def delete_document(self, document_id: str) -> int:  # noqa: ARG002
        """No-op: rows are deleted by the vector store; reports 0 to avoid double counting."""
        return 0

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        words = _WORD.findall(query.lower())
        if not words or k <= 0:
            return []
        # OR the words (natural-language questions rarely contain every term).
        ts_query = " | ".join(dict.fromkeys(words))
        statement = sql.SQL(
            """SELECT id, document_id, position, title, text, metadata,
                      ts_rank_cd(tsv, q, 1) AS score
               FROM {t}, to_tsquery('english', %s) AS q
               WHERE tsv @@ q ORDER BY score DESC, id LIMIT %s"""
        ).format(t=self._db.ident())
        with self._db.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(statement, (ts_query, k))
            rows = cur.fetchall()
        return [ScoredChunk(_row_to_chunk(r), float(r["score"]), "fts") for r in rows]
