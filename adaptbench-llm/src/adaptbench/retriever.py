"""Few-shot retriever over the WorkflowLM TRAIN split (the RAG corpus).

Fairness: RAG retrieves only from the TRAIN examples — the same data the fine-tune learned from —
so the two adaptation methods share an identical knowledge source. A guard refuses any corpus row
whose input matches a test input (no leakage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


class FewShotRetriever:
    def __init__(self, train_path: Path, test_path: Path, embed_model: str):
        train = [json.loads(l) for l in train_path.open(encoding="utf-8") if l.strip()]
        test_inputs = {_norm(json.loads(l)["input"]) for l in test_path.open(encoding="utf-8") if l.strip()}
        # leakage guard: the retrieval corpus must not contain any test input
        self.corpus = [ex for ex in train if _norm(ex["input"]) not in test_inputs]
        assert len(self.corpus) == len(train), "retrieval corpus overlaps the test set (leakage)"

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(embed_model)
        self._emb = self._model.encode(
            [ex["input"] for ex in self.corpus], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def retrieve(self, query: str, k: int) -> list[dict]:
        q = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._emb @ q
        idx = np.argsort(-sims)[:k]
        return [self.corpus[i] for i in idx]
