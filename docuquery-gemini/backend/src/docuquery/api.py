"""FastAPI app for DocuQuery.

Endpoints (all document endpoints are scoped by the `X-User-Id` header — the IDOR control):
  GET  /health
  POST /upload         (multipart file)            -> ingestion result
  POST /query          {question, k?}              -> answer + citations + retrieved chunks
  GET  /documents                                  -> this user's documents
  POST /eval/run                                   -> ingest labeled corpus + compute RAG metrics

The Gemini key is never exposed by any endpoint; /health only reports which providers are active.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docuquery.answer import answer_question
from docuquery.config import get_settings
from docuquery.db import get_store
from docuquery.ingest import UploadError, ingest_document
from docuquery.providers.embeddings import get_embedding_provider
from docuquery.providers.llm import get_llm_provider

app = FastAPI(title="DocuQuery-Gemini", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def user_id(x_user_id: str | None = Header(default=None)) -> str:
    return x_user_id or "anonymous"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    k: int | None = None


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "gemini_enabled": bool(s.gemini_api_key),
        "embedding_provider": get_embedding_provider().name,
        "llm_provider": get_llm_provider().name,
        "store_backend": getattr(get_store(dim=get_embedding_provider().dim), "backend", "unknown"),
    }


@app.post("/upload")
async def upload(file: UploadFile, uid: str = Depends(user_id)) -> dict:
    data = await file.read()
    try:
        result = ingest_document(uid, file.filename or "upload", data)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(result)


@app.post("/query")
def query(req: QueryRequest, uid: str = Depends(user_id)) -> dict:
    res = answer_question(uid, req.question, req.k)
    return asdict(res)


@app.get("/documents")
def documents(uid: str = Depends(user_id)) -> dict:
    store = get_store(dim=get_embedding_provider().dim)
    return {"documents": [asdict(d) for d in store.list_documents(uid)]}


@app.post("/eval/run")
def eval_run() -> dict:
    """Ingest the labeled corpus for the eval user and compute RAG metrics."""
    from docuquery.eval.dataset import EVAL_USER, ingest_corpus
    from docuquery.eval.metrics import run_eval

    store = get_store(dim=get_embedding_provider().dim)
    # fresh corpus for a deterministic eval run
    docs = store.list_documents(EVAL_USER)
    if not docs:
        ingest_corpus(EVAL_USER)
    return run_eval(EVAL_USER)
