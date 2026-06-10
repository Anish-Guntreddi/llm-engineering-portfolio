"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getDocuments,
  uploadDocument,
  type DocumentItem,
  type UploadResponse,
} from "@/lib/api";
import { useToast } from "@/components/Toast";

const ACCEPT = ".pdf,.md,.txt";
const ACCEPT_EXT = ["pdf", "md", "txt"];

export default function DocumentSidebar() {
  const { toast } = useToast();
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [docsError, setDocsError] = useState<string | null>(null);

  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastUpload, setLastUpload] = useState<UploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoadingDocs(true);
    setDocsError(null);
    try {
      const res = await getDocuments();
      setDocuments(res.documents);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Failed to load documents";
      setDocsError(msg);
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
      if (!ACCEPT_EXT.includes(ext)) {
        const msg = `Unsupported file type ".${ext}". Allowed: ${ACCEPT_EXT.join(", ")}.`;
        setUploadError(msg);
        toast(msg, "error");
        return;
      }

      setUploading(true);
      setProgress(0);
      setUploadError(null);
      setLastUpload(null);
      try {
        const res = await uploadDocument(file, setProgress);
        setLastUpload(res);
        toast(
          `Uploaded "${res.document_name}" — ${res.n_pages} pages, ${res.n_chunks} chunks.`,
          "success"
        );
        await refresh();
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : "Upload failed";
        setUploadError(msg);
        toast(msg, "error");
      } finally {
        setUploading(false);
      }
    },
    [refresh, toast]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
    e.target.value = ""; // allow re-uploading same file
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  };

  return (
    <aside className="flex h-full flex-col gap-4">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-500">
          Upload
        </h2>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors ${
            dragging
              ? "border-accent-500 bg-accent-50"
              : "border-ink-300 bg-white hover:border-accent-400 hover:bg-ink-50"
          }`}
        >
          <svg
            className="h-6 w-6 text-ink-400"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5 7.5 12M12 7.5V21"
            />
          </svg>
          <p className="text-sm font-medium text-ink-700">
            Drop a file or click to browse
          </p>
          <p className="text-xs text-ink-400">PDF, Markdown, or text</p>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={onInputChange}
          />
        </div>

        {uploading && (
          <div className="mt-3">
            <div className="mb-1 flex justify-between text-xs text-ink-500">
              <span>Uploading…</span>
              <span>{progress}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-200">
              <div
                className="h-full bg-accent-600 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {lastUpload && !uploading && (
          <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
            <span className="font-medium">{lastUpload.document_name}</span>{" "}
            indexed — {lastUpload.n_pages} pages, {lastUpload.n_chunks} chunks.
          </div>
        )}

        {uploadError && !uploading && (
          <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {uploadError}
          </div>
        )}
      </section>

      <section className="flex min-h-0 flex-1 flex-col">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-500">
            Documents
          </h2>
          <button
            onClick={() => void refresh()}
            className="text-xs text-ink-400 hover:text-ink-700"
            title="Refresh"
          >
            refresh
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {loadingDocs ? (
            <p className="text-sm text-ink-400">Loading…</p>
          ) : docsError ? (
            <p className="text-sm text-red-600">{docsError}</p>
          ) : documents.length === 0 ? (
            <p className="text-sm text-ink-400">
              No documents yet. Upload one to get started.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {documents.map((doc) => (
                <li
                  key={doc.id}
                  className="flex items-center justify-between gap-2 rounded-lg border border-ink-200 bg-white px-3 py-2"
                >
                  <span
                    className="truncate text-sm font-medium text-ink-800"
                    title={doc.name}
                  >
                    {doc.name}
                  </span>
                  <span className="shrink-0 rounded-full bg-ink-100 px-2 py-0.5 text-[11px] text-ink-500">
                    {doc.n_chunks} chunks
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </aside>
  );
}
