"""Ingestion: validate upload -> extract pages -> clean -> chunk -> embed -> store (user-scoped)."""

from __future__ import annotations

from dataclasses import dataclass

from docuquery.chunking import chunk_pages, extract_pages
from docuquery.config import get_settings
from docuquery.db import get_store
from docuquery.providers.embeddings import get_embedding_provider


class UploadError(ValueError):
    pass


@dataclass
class IngestResult:
    document_id: str
    document_name: str
    n_pages: int
    n_chunks: int


def validate_upload(filename: str, data: bytes) -> None:
    s = get_settings()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in s.allowed_extensions:
        raise UploadError(f"unsupported file type '{ext}'; allowed: {s.allowed_extensions}")
    if len(data) == 0:
        raise UploadError("empty file")
    if len(data) > s.max_upload_bytes:
        raise UploadError(f"file too large ({len(data)} bytes > {s.max_upload_bytes})")


def ingest_document(user_id: str, filename: str, data: bytes) -> IngestResult:
    s = get_settings()
    validate_upload(filename, data)

    pages = extract_pages(filename, data)
    if len(pages) > s.max_pages:
        raise UploadError(f"too many pages ({len(pages)} > {s.max_pages})")

    chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise UploadError("no extractable text found in document")

    provider = get_embedding_provider()
    embeddings = provider.embed([c.text for c in chunks])
    store = get_store(dim=provider.dim)

    document_id = store.add_document(user_id, filename)
    n = store.add_chunks(document_id, user_id, filename, chunks, embeddings)
    return IngestResult(document_id, filename, len(pages), n)
