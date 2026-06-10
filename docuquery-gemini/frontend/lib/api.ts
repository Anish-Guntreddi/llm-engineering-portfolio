// Typed API client for the DocuQuery-Gemini FastAPI backend.
// Injects the X-User-Id header and reads NEXT_PUBLIC_API_URL.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

const USER_ID_KEY = "docuquery_user_id";

/**
 * Returns a stable per-browser user id, generating + persisting one on first use.
 * Safe to call on the server (returns a transient id that is not persisted).
 */
export function getUserId(): string {
  if (typeof window === "undefined") {
    return "user-server";
  }
  let id = window.localStorage.getItem(USER_ID_KEY);
  if (!id) {
    const rand =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).slice(2, 10);
    id = `user-${rand}`;
    window.localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

function headers(extra?: Record<string, string>): HeadersInit {
  return {
    "X-User-Id": getUserId(),
    ...extra,
  };
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as { detail?: string };
    if (body?.detail) detail = body.detail;
  } catch {
    // non-JSON body; keep default message
  }
  throw new ApiError(detail, res.status);
}

// ---------------------------------------------------------------------------
// Types matching the backend contract
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  gemini_enabled: boolean;
  embedding_provider: string;
  llm_provider: string;
  store_backend: string;
}

export interface UploadResponse {
  document_id: string;
  document_name: string;
  n_pages: number;
  n_chunks: number;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  document_name: string;
  page_number: number;
  section_title: string | null;
  similarity: number;
  snippet: string;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_name: string;
  page_number: number;
  section_title: string | null;
  similarity: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  abstained: boolean;
  provider: string;
  citations: Citation[];
  retrieved: RetrievedChunk[];
}

export interface DocumentItem {
  id: string;
  user_id: string;
  name: string;
  n_chunks: number;
}

export interface DocumentsResponse {
  documents: DocumentItem[];
}

export interface PerQuestion {
  q: string;
  expected: string;
  in_topk: boolean;
  rank: number;
  abstained: boolean;
  faithfulness: number;
  answer: string;
}

export interface EvalResponse {
  n_questions: number;
  k: number;
  recall_at_k: number;
  mrr: number;
  citation_accuracy: number;
  faithfulness: number;
  abstention_correct: boolean;
  provider_note: string;
  per_question: PerQuestion[];
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { headers: headers() });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function getDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${API_URL}/documents`, { headers: headers() });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadResponse> {
  // Use XHR so we can report upload progress.
  return new Promise<UploadResponse>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/upload`);
    xhr.setRequestHeader("X-User-Id", getUserId());

    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch {
          reject(new ApiError("Invalid response from server", xhr.status));
        }
        return;
      }
      let detail = `Upload failed (${xhr.status})`;
      try {
        const body = JSON.parse(xhr.responseText) as { detail?: string };
        if (body?.detail) detail = body.detail;
      } catch {
        // keep default
      }
      reject(new ApiError(detail, xhr.status));
    };

    xhr.onerror = () =>
      reject(new ApiError("Network error during upload", 0));

    xhr.send(form);
  });
}

export async function query(
  question: string,
  k?: number
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(k != null ? { question, k } : { question }),
  });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function runEval(): Promise<EvalResponse> {
  const res = await fetch(`${API_URL}/eval/run`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
  });
  if (!res.ok) return parseError(res);
  return res.json();
}
