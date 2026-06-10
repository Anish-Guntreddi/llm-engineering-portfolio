"""Central configuration (env-driven). The Gemini key lives here, server-side only."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCUQUERY_", env_file=".env", extra="ignore")

    # Storage: if unset, the in-memory vector store is used (tests / no-docker).
    database_url: str | None = None

    # Extra CORS origins (comma-separated) for the deployed frontend, e.g. your Vercel domain.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Providers. Gemini is used iff a key is present; otherwise local fallbacks.
    gemini_api_key: str | None = None
    embedding_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embed_model: str = "models/text-embedding-004"
    gemini_chat_model: str = "models/gemini-1.5-flash"

    # Chunking
    chunk_size: int = 900          # characters
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5
    # Below this max cosine similarity, the system abstains ("I don't know").
    min_similarity: float = 0.25

    # Upload safety
    max_upload_bytes: int = 20 * 1024 * 1024   # 20 MB
    max_pages: int = 200
    allowed_extensions: tuple[str, ...] = (".pdf", ".md", ".txt")

    @property
    def use_gemini_embeddings(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def use_gemini_generation(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
