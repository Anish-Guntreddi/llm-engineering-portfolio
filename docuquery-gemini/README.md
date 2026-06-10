# DocuQuery-Gemini

**A full-stack RAG platform: upload technical docs, ask questions, get answers grounded in
retrieved chunks _with citations (document + page)_ — plus a real retrieval-evaluation dashboard.**

Stack: **Next.js + TypeScript + Tailwind** · **FastAPI** · **Postgres + pgvector** · **Gemini**
(embeddings + generation), with a **local fallback** so it runs with zero API cost.

> The differentiator over "chat with PDF": **citation traceability** (every answer links to the
> exact chunks that produced it) and a **retrieval-evaluation dashboard** (Recall@k, MRR, citation
> accuracy, faithfulness).

## Retrieval evaluation (local provider, labeled set)

Run `python -m docuquery.eval.run_eval` (or POST `/eval/run`, or the `/eval` page):

| Metric | Score |
|---|---|
| Recall@k (k=5) | **1.00** |
| MRR | **0.95** |
| Citation accuracy | **1.00** |
| Faithfulness (lexical proxy) | **1.00** |
| Abstention on unanswerable | **correct** |

Measured on a labeled set of questions tagged with their expected document + page, using the local
`all-MiniLM-L6-v2` embeddings. Faithfulness is a lexical proxy here; an LLM-judge faithfulness
path is used automatically when a Gemini key is configured.

## Providers — runs now, Gemini-powered with a key

| Concern | Default (no key) | With `DOCUQUERY_GEMINI_API_KEY` |
|---|---|---|
| Embeddings | local `all-MiniLM-L6-v2` (384-d, free, semantic) | Gemini `text-embedding-004` (768-d) |
| Generation | extractive grounded answer (stitches top chunks + citations) | Gemini `gemini-1.5-flash` |

The Gemini key is **server-side only** — the frontend never sees it and no endpoint returns it
(`/health` reports only which providers are active). See [SECURITY.md](SECURITY.md).

## How it works

```
upload (PDF/MD/TXT, user-scoped) → extract+page-tag → clean → overlapping chunks
   → embed → pgvector (chunk: id, document_id, user_id, text, page_number, section_title, embedding)
query → embed question → top-k cosine over THIS user's chunks → grounded prompt
   → Gemini/extractive answer + citations built from the ACTUAL retrieved chunk rows
   → abstain ("I don't know based on the provided documents") when top similarity < threshold
```

**Citation integrity:** citations are constructed from the retrieved chunk rows (each carries its
`chunk_id`, document, and page) — never re-derived from the answer text — so the UI can prove an
answer's sources are exactly the chunks that were retrieved. Verified by tests.

## Run it locally

```bash
# 1. Postgres + pgvector
docker compose up -d                       # exposes db on localhost:5433

# 2. Backend
cd backend
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -e ".[local,dev]"   # add ,gemini for the Gemini provider
uv pip install --python .venv "transformers==4.46.3"   # pin (sentence-transformers compat)
cp .env.example .env                        # set DOCUQUERY_DATABASE_URL; optionally the Gemini key
uvicorn docuquery.api:app --port 8000

# 3. Frontend
cd ../frontend
npm install
cp .env.local.example .env.local            # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                                 # http://localhost:3000

# 4. (optional) reproduce the eval numbers
cd ../backend && python -m docuquery.eval.run_eval
```

Without Docker/Postgres, leave `DOCUQUERY_DATABASE_URL` unset and the backend uses an in-memory
vector store (great for the test suite and a quick spin).

## Tests

```bash
cd backend && python -m pytest        # citation traceability, IDOR isolation, upload safety, abstention, eval
```

## Security

Audited for IDOR (cross-user document access), malicious uploads, prompt injection via document
content, and API-key leakage — see [SECURITY.md](SECURITY.md). V1 uses lightweight `X-User-Id`
scoping; the row-level access filter is the real boundary and is implemented + tested.

## Deploy (handoff)

This is deploy-ready but the actual push needs **your** cloud accounts:
- **Backend** → Render/Fly/Railway from `backend/Dockerfile`. Set `DOCUQUERY_DATABASE_URL`
  (managed Postgres with the `vector` extension) and optionally `DOCUQUERY_GEMINI_API_KEY`.
- **Frontend** → Vercel from `frontend/`. Set `NEXT_PUBLIC_API_URL` to the deployed backend URL.
- **DB** → any managed Postgres with `pgvector` (Supabase, Neon, RDS). The schema auto-creates on
  first connect.
See [DEPLOY.md](DEPLOY.md) for step-by-step.

## Layout

```
docker-compose.yml          # postgres + pgvector
backend/  src/docuquery/    # config, db (pgvector + in-memory), providers, chunking,
          ingest, retrieve, answer, api, eval/   + tests, Dockerfile, .env.example
frontend/                   # Next.js (App Router) + Tailwind: upload, chat, sources panel, eval
```
