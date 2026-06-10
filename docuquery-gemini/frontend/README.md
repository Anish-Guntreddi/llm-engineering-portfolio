# DocuQuery-Gemini — Frontend

A Next.js (App Router) + TypeScript + Tailwind frontend for the DocuQuery-Gemini RAG application. It talks to the FastAPI backend (default `http://localhost:8000`).

## Features

- Upload documents (`.pdf`, `.md`, `.txt`) with drag-and-drop, live progress, and resulting page/chunk counts.
- Ask questions against your documents and see the rendered answer, including a clear "I don't know" abstention state.
- A **Sources** panel listing every retrieved chunk, with the cited chunks highlighted and tagged with their `[marker]` — making citation-to-chunk traceability obvious.
- An `/eval` page that runs the backend evaluation and renders headline metrics, an abstention badge, and a per-question table.
- A shared header showing live backend status (`embedding_provider`, `llm_provider`, `store_backend`, Gemini enabled) from `/health`.

## Requirements

- Node 18+ (tested with Node 22 / npm 10).
- The DocuQuery-Gemini FastAPI backend running and reachable.

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Configure the backend URL. Copy the example env file and adjust if needed:

   ```bash
   cp .env.local.example .env.local
   ```

   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

   If `NEXT_PUBLIC_API_URL` is not set, the app defaults to `http://localhost:8000`.

## Run

Development server (hot reload):

```bash
npm run dev
```

Then open http://localhost:3000.

Production build:

```bash
npm run build
npm run start
```

## How it works

- A per-browser user id is generated on first load (`user-<random>`), persisted in `localStorage` under `docuquery_user_id`, and sent on every request as the `X-User-Id` header. It is shown (truncated) in the header.
- All backend calls go through the typed client in [`lib/api.ts`](lib/api.ts).
- No backend secrets live in the frontend; only the public API URL is read from the environment.
