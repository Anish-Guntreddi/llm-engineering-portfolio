"""Embedding provider: local sentence-transformers by default, Gemini when a key is set.

Both return L2-normalized float vectors so cosine similarity is a dot product. The provider is
chosen once at process start and reported by /health, so retrieval is consistent across
ingestion, querying, and evaluation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from docuquery.config import get_settings


class EmbeddingProvider:
    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, normalized
        raise NotImplementedError


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers MiniLM — free, semantic, reproducible. Lazy-loads the model."""

    def __init__(self, model_name: str):
        self.name = f"local:{model_name.split('/')[-1]}"
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model
        self.name = f"gemini:{model.split('/')[-1]}"
        self.dim = 768

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            r = self._genai.embed_content(model=self._model, content=t, task_type="retrieval_document")
            out.append(r["embedding"])
        return _normalize(np.asarray(out, dtype=np.float32))


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    if s.use_gemini_embeddings:
        return GeminiEmbeddingProvider(s.gemini_api_key, s.gemini_embed_model)
    return LocalEmbeddingProvider(s.embedding_model_local)
