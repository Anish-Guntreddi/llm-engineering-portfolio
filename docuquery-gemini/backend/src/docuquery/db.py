"""Vector store: pgvector-backed when DATABASE_URL is set, in-memory otherwise.

Both implement the same interface so ingestion / retrieval / eval logic is identical and
unit-testable without Docker. Every row is scoped by ``user_id`` — retrieval ALWAYS filters by
user, which is the IDOR control.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from docuquery.config import get_settings


@dataclass
class StoredChunk:
    id: str
    document_id: str
    user_id: str
    document_name: str
    text: str
    page_number: int
    section_title: str | None


@dataclass
class Document:
    id: str
    user_id: str
    name: str
    n_chunks: int


class VectorStore:
    def add_document(self, user_id: str, name: str) -> str: ...
    def add_chunks(self, document_id: str, user_id: str, document_name: str,
                   chunks: list, embeddings: np.ndarray) -> int: ...
    def search(self, user_id: str, query_vec: np.ndarray, k: int) -> list[tuple[StoredChunk, float]]: ...
    def list_documents(self, user_id: str) -> list[Document]: ...
    def get_chunk(self, user_id: str, chunk_id: str) -> StoredChunk | None: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store (tests / no Docker)
# ---------------------------------------------------------------------------
class InMemoryStore(VectorStore):
    backend = "memory"

    def __init__(self):
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, StoredChunk] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def add_document(self, user_id: str, name: str) -> str:
        did = str(uuid.uuid4())
        self._docs[did] = Document(did, user_id, name, 0)
        return did

    def add_chunks(self, document_id, user_id, document_name, chunks, embeddings) -> int:
        for ch, vec in zip(chunks, embeddings):
            cid = str(uuid.uuid4())
            self._chunks[cid] = StoredChunk(cid, document_id, user_id, document_name,
                                            ch.text, ch.page_number, ch.section_title)
            self._vecs[cid] = np.asarray(vec, dtype=np.float32)
        self._docs[document_id].n_chunks += len(chunks)
        return len(chunks)

    def search(self, user_id, query_vec, k):
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        scored = []
        for cid, ch in self._chunks.items():
            if ch.user_id != user_id:   # IDOR guard
                continue
            sim = float(np.dot(self._vecs[cid], q))
            scored.append((ch, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def list_documents(self, user_id):
        return [d for d in self._docs.values() if d.user_id == user_id]

    def get_chunk(self, user_id, chunk_id):
        ch = self._chunks.get(chunk_id)
        return ch if ch and ch.user_id == user_id else None

    def reset(self):
        self._docs.clear(); self._chunks.clear(); self._vecs.clear()


# ---------------------------------------------------------------------------
# pgvector store
# ---------------------------------------------------------------------------
class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._register = register_vector
        self._dsn = dsn
        self._dim = dim
        self._init_schema()

    def _conn(self):
        conn = self._psycopg.connect(self._dsn, autocommit=True)
        self._register(conn)
        return conn

    def _init_schema(self):
        with self._psycopg.connect(self._dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._register(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now())""")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_number INT NOT NULL,
                    section_title TEXT,
                    embedding vector({self._dim}))""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_user_idx ON chunks(user_id)")
            conn.execute("""CREATE INDEX IF NOT EXISTS chunks_embed_idx ON chunks
                            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)""")

    def add_document(self, user_id, name):
        did = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("INSERT INTO documents (id, user_id, name) VALUES (%s,%s,%s)",
                         (did, user_id, name))
        return did

    def add_chunks(self, document_id, user_id, document_name, chunks, embeddings):
        with self._conn() as conn, conn.cursor() as cur:
            for ch, vec in zip(chunks, embeddings):
                cur.execute(
                    """INSERT INTO chunks (id, document_id, user_id, document_name, text,
                       page_number, section_title, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (str(uuid.uuid4()), document_id, user_id, document_name, ch.text,
                     ch.page_number, ch.section_title, np.asarray(vec, dtype=np.float32)))
        return len(chunks)

    def search(self, user_id, query_vec, k):
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, document_id, user_id, document_name, text, page_number, section_title,
                          1 - (embedding <=> %s) AS sim
                   FROM chunks WHERE user_id = %s ORDER BY embedding <=> %s LIMIT %s""",
                (q, user_id, q, k)).fetchall()
        return [(StoredChunk(str(r[0]), str(r[1]), r[2], r[3], r[4], r[5], r[6]), float(r[7]))
                for r in rows]

    def list_documents(self, user_id):
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT d.id, d.user_id, d.name, count(c.id)
                   FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                   WHERE d.user_id = %s GROUP BY d.id ORDER BY d.created_at DESC""",
                (user_id,)).fetchall()
        return [Document(str(r[0]), r[1], r[2], int(r[3])) for r in rows]

    def get_chunk(self, user_id, chunk_id):
        with self._conn() as conn:
            r = conn.execute(
                """SELECT id, document_id, user_id, document_name, text, page_number, section_title
                   FROM chunks WHERE id = %s AND user_id = %s""", (chunk_id, user_id)).fetchone()
        return StoredChunk(str(r[0]), str(r[1]), r[2], r[3], r[4], r[5], r[6]) if r else None

    def reset(self):
        with self._conn() as conn:
            conn.execute("TRUNCATE chunks, documents CASCADE")


_STORE: VectorStore | None = None


def get_store(dim: int | None = None) -> VectorStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    s = get_settings()
    if s.database_url:
        if dim is None:
            from docuquery.providers.embeddings import get_embedding_provider

            prov = get_embedding_provider()
            prov.embed(["warmup"])  # ensure dim is known
            dim = prov.dim
        _STORE = PgVectorStore(s.database_url, dim)
    else:
        _STORE = InMemoryStore()
    return _STORE


def set_store(store: VectorStore) -> None:
    """Test hook to inject a store."""
    global _STORE
    _STORE = store
