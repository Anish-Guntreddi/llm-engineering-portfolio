# Deploying DocuQuery-Gemini

The app is deploy-ready; these steps require **your** cloud accounts (they can't be done on your
behalf). Verified locally via `docker compose` + `uvicorn` + `npm run dev`.

## 1. Database — managed Postgres with pgvector

Use Supabase, Neon, or RDS. Ensure the `vector` extension is available (it auto-creates on first
backend connect via `CREATE EXTENSION IF NOT EXISTS vector`). Grab the connection string.

## 2. Backend — Render / Fly / Railway

Build from `backend/Dockerfile`.

Environment variables:
- `DOCUQUERY_DATABASE_URL` = your managed Postgres URL (required for persistence)
- `DOCUQUERY_GEMINI_API_KEY` = your Gemini key (optional; omit to use the local provider)

Render example: New Web Service → "Build from Dockerfile" → root `backend/` → add the env vars →
deploy. The service listens on `$PORT`/8000 and exposes `/health`, `/upload`, `/query`,
`/documents`, `/eval/run`.

> Note: the Dockerfile installs **CPU** torch + `all-MiniLM-L6-v2` for the local embedding
> fallback. If you set a Gemini key, embeddings/generation use Gemini and the local model is just
> a fallback. For a leaner image when always using Gemini, you can drop the `local` extra + torch.

## 3. Frontend — Vercel

Import the repo, set the project root to `frontend/`. Environment variable:
- `NEXT_PUBLIC_API_URL` = the deployed backend URL (e.g. `https://docuquery-api.onrender.com`)

Deploy. Vercel auto-detects Next.js.

## 4. Verify in production

1. `GET https://<backend>/health` → `{status:"ok", store_backend:"pgvector", ...}`
2. In the UI: upload a multi-page PDF, ask a question whose answer is on a known page, confirm the
   answer + citation (document + page) are correct and the sources panel shows the retrieved chunks.
3. Ask an unanswerable question → expect "I don't know based on the provided documents".
4. Open `/eval` → run evaluation → metrics render.

## CORS

Set `DOCUQUERY_CORS_ORIGINS` on the backend to a comma-separated list including your Vercel domain,
e.g. `DOCUQUERY_CORS_ORIGINS=https://docuquery.vercel.app,http://localhost:3000`. Defaults to
localhost only.
