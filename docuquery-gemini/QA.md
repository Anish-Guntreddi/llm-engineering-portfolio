# DocuQuery-Gemini — QA Report

## Automated / verified

**Backend HTTP API** (exercised through the exact endpoints the frontend calls, via FastAPI
TestClient and against live pgvector):

| Check | Result |
|---|---|
| `GET /health` | ok; reports provider + store backend |
| `POST /upload` (multipart .md) | 200, returns page/chunk counts |
| `POST /upload` (.exe) | 400 `unsupported file type '.exe'` |
| `POST /query` (answerable) | grounded answer; **every citation chunk_id ∈ retrieved chunks** (traceability) |
| `POST /query` (unanswerable) | abstains: "I don't know based on the provided documents" |
| Cross-user query (different `X-User-Id`) | abstains, **0 chunks retrieved** (IDOR isolation) |
| `GET /documents` | lists only the caller's documents |
| `POST /eval/run` | Recall@k 1.0, MRR 0.95, citation_acc 1.0, faithfulness 1.0, abstention correct |
| Live **pgvector** path | page-level citation correct (vectordb_guide p.2); IDOR holds against the real DB |

**Frontend:** `npm run build` passes cleanly (no TS errors). Backend + frontend boot together;
the frontend serves the app shell and reads `NEXT_PUBLIC_API_URL`. 6 backend pytest cases pass.

## Manual browser checklist (the `/qa` flow)

Run `docker compose up -d`, backend `uvicorn docuquery.api:app --port 8000`, frontend
`npm run dev`, open http://localhost:3000:

1. Upload a multi-page PDF → wait for ingestion → see page/chunk counts; doc appears in sidebar.
2. Ask a question whose answer is on a known page → answer is correct **and** the citation points
   to the right document + page; open the sources panel → the highlighted (cited) chunks match.
3. Ask something the docs can't answer → "I don't know based on the provided documents".
4. Upload an empty/garbage file → friendly 400 error surfaced in the UI.
5. Open `/eval` → Run evaluation → metric cards + per-question table render.

## Notes
- Page-level citation correctness is proven by the eval corpus + live pgvector smoke (answers
  cited to pages 2/3). The upload→query→cite flow is proven via HTTP for `.md` (single page);
  multi-page PDF page-citation is the one item left to the manual browser pass above.
- Browser click-through automation isn't run in this environment; the API contract the UI depends
  on is fully exercised, and the UI builds and boots.
