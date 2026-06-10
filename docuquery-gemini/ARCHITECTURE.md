# DocuQuery-Gemini — Scope & Architecture

> Folds `/office-hours` (scope) and `/autoplan` (CEO + design + eng) into one doc.

## 1. Reframed scope

A full-stack RAG app: upload technical docs (PDF/MD/TXT) → ask questions → get answers
**grounded in retrieved chunks, with citations (document + page)**, plus a **retrieval-evaluation
dashboard**. The differentiator vs "chat with PDF" is **citation traceability** (every answer
links to the exact chunks that produced it) and a **real eval page** (Recall@k, MRR, citation
accuracy, faithfulness).

**V1 cut-line (to avoid drowning):**
- ✅ in V1: ingestion, retrieval, grounded answers with citations, sources panel, eval dashboard,
  "I don't know" path, per-user document scoping.
- ✂️ cut from V1: full auth (passwords/sessions), reranking, multi-doc collections UI, streaming.
  User scoping is done with an `X-User-Id` header so the **IDOR** security concern is real and
  testable without building auth.

**"Done" =** upload a multi-page PDF, ask a question whose answer is on a known page, get a correct
answer whose citation points to the right document+page, see the retrieved chunks in a sources
panel, get "I don't know" when the docs can't answer, and see eval metrics render.

## 2. Providers (pluggable — runnable now, Gemini-powered with a key)

| Concern | Default (no key) | With `GEMINI_API_KEY` |
|---|---|---|
| Embeddings | local `sentence-transformers` (all-MiniLM-L6-v2) — free, semantic, reproducible | Gemini `text-embedding-004` |
| Generation | extractive grounded answer (stitches top chunks + citations) | Gemini `gemini-1.5-flash` |

The Gemini API key lives **only** server-side (FastAPI). The frontend never sees it; all model
calls go through the backend. Selected provider is reported by `/health`.

## 3. Data flow

```
upload (PDF/MD/TXT, user-scoped)
  └─ extract text (+ page numbers) → clean → chunk (overlapping, page-tagged)
       └─ embed chunks → store in pgvector  (chunk: id, document_id, user_id, text,
                                             page_number, section_title, embedding)
query (question, user-scoped)
  └─ embed question → top-k cosine over THIS USER's chunks → grounded prompt
       └─ LLM → answer + structured citations [{document, page, chunk_id, snippet}]
       └─ if top similarity < threshold → "I don't know" (honest abstention)
eval
  └─ labeled set {question, expected_answer, expected_doc, expected_page}
       └─ Recall@k, MRR, citation accuracy, faithfulness  → dashboard
```

**Citation integrity (the whole point):** citations are built from the *actual retrieved chunk
rows*, never re-derived from the answer text. Each citation carries the `chunk_id` that was in
the context, so the UI can prove the answer's sources are the chunks that were retrieved.

## 4. Storage

Postgres + `pgvector` (via docker-compose, `pgvector/pgvector:pg16`). One `documents` table and
one `chunks` table with an `vector` column + ivfflat index. A pure-Python in-memory vector store
(cosine over numpy) is the fallback used in tests / when `DATABASE_URL` is unset, behind the same
interface — so ingestion/retrieval/eval logic is identical and unit-testable without Docker.

## 5. Security posture (audited by `/cso`)

- **IDOR:** every chunk/document row carries `user_id`; every query filters by the request's
  `X-User-Id`. User A can never retrieve user B's chunks. (Tested.)
- **Malicious upload:** size cap, extension/MIME allow-list, page cap, text-only extraction
  (no macro/script execution).
- **Prompt injection:** retrieved document text is wrapped as untrusted context with an explicit
  instruction boundary; the system prompt tells the model to answer only from context and ignore
  instructions inside documents.
- **Key leakage:** Gemini key is server-side only; never returned by any endpoint; `.env` gitignored.

## 6. Repo layout

```
docker-compose.yml                 # postgres + pgvector
backend/
  src/docuquery/
    config.py  db.py  chunking.py
    providers/embeddings.py providers/llm.py
    ingest.py  retrieve.py  answer.py  api.py
    eval/metrics.py  eval/run_eval.py  eval/dataset.py
  tests/   Dockerfile   .env.example
frontend/                          # Next.js (App Router) + Tailwind
  app/(upload, chat, sources panel, eval page)
```

## 7. Deployment

Deploy-ready: backend `Dockerfile` (Render/Fly/Railway), Next.js for Vercel, env templates, and
a managed Postgres+pgvector. Actual external deploy needs the owner's cloud accounts; steps are
documented in the README. Locally verified via docker-compose.
