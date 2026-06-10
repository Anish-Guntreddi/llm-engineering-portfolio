"use client";

import { useState } from "react";
import { ApiError, query, type QueryResponse } from "@/lib/api";
import { useToast } from "@/components/Toast";
import SourcesPanel from "@/components/SourcesPanel";

export default function Chat() {
  const { toast } = useToast();
  const [question, setQuestion] = useState("");
  const [askedQuestion, setAskedQuestion] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);
    setAskedQuestion(q);
    try {
      const res = await query(q);
      setResult(res);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Query failed";
      setError(msg);
      setResult(null);
      toast(msg, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Answer area */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {!askedQuestion && !loading && (
            <div className="rounded-xl border border-dashed border-ink-300 bg-white p-8 text-center">
              <h2 className="text-lg font-semibold text-ink-800">
                Ask your documents
              </h2>
              <p className="mt-1 text-sm text-ink-500">
                Upload files on the left, then ask a question. Answers cite the
                exact chunks they come from.
              </p>
            </div>
          )}

          {askedQuestion && (
            <div className="rounded-xl border border-ink-200 bg-ink-100 px-4 py-3">
              <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-400">
                Question
              </span>
              <p className="mt-0.5 text-sm font-medium text-ink-800">
                {askedQuestion}
              </p>
            </div>
          )}

          {loading && (
            <div className="rounded-xl border border-ink-200 bg-white p-6">
              <div className="flex items-center gap-2 text-sm text-ink-500">
                <span className="h-2 w-2 animate-pulse rounded-full bg-accent-500" />
                Thinking…
              </div>
            </div>
          )}

          {error && !loading && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && !loading && (
            <>
              {result.abstained ? (
                <div className="rounded-xl border-2 border-amber-300 bg-amber-50 p-5">
                  <div className="flex items-center gap-2">
                    <span className="flex h-7 w-7 items-center justify-center rounded-full bg-amber-400 text-white">
                      <svg
                        className="h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M12 9v3.75m0 3.75h.008M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    </span>
                    <h3 className="text-base font-semibold text-amber-900">
                      I don&apos;t know
                    </h3>
                  </div>
                  <p className="mt-2 text-sm text-amber-800">
                    {result.answer ||
                      "The retrieved documents don't contain enough information to answer this confidently."}
                  </p>
                </div>
              ) : (
                <div className="rounded-xl border border-ink-200 bg-white p-5">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-400">
                      Answer
                    </span>
                    <span className="rounded-full bg-ink-100 px-2 py-0.5 text-[11px] text-ink-500">
                      via {result.provider}
                    </span>
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-800">
                    {result.answer}
                  </p>
                </div>
              )}

              <SourcesPanel
                retrieved={result.retrieved}
                citations={result.citations}
              />
            </>
          )}
        </div>
      </div>

      {/* Input */}
      <form onSubmit={submit} className="mx-auto w-full max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-ink-300 bg-white p-2 shadow-sm focus-within:border-accent-500 focus-within:ring-1 focus-within:ring-accent-500">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submit(e);
              }
            }}
            rows={1}
            placeholder="Ask a question about your documents…"
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-ink-800 placeholder:text-ink-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-700 disabled:cursor-not-allowed disabled:bg-ink-300"
          >
            {loading ? "…" : "Ask"}
          </button>
        </div>
        <p className="mt-1 px-1 text-[11px] text-ink-400">
          Press Enter to send, Shift+Enter for a new line.
        </p>
      </form>
    </div>
  );
}
