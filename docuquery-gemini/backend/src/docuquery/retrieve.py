"""Retrieval: embed the question, return top-k chunks for THIS user (cosine)."""

from __future__ import annotations

from docuquery.db import StoredChunk, get_store
from docuquery.providers.embeddings import get_embedding_provider


def retrieve(user_id: str, question: str, k: int) -> list[tuple[StoredChunk, float]]:
    provider = get_embedding_provider()
    qvec = provider.embed([question])[0]
    store = get_store(dim=provider.dim)
    return store.search(user_id, qvec, k)
