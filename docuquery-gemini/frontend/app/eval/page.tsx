"use client";

import { useState } from "react";
import { ApiError, runEval, type EvalResponse } from "@/lib/api";
import { useToast } from "@/components/Toast";

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-ink-200 bg-white p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-ink-500">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-ink-900">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-ink-400">{hint}</div>}
    </div>
  );
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

export default function EvalPage() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<EvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runEval();
      setData(res);
      toast("Evaluation complete.", "success");
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Evaluation failed";
      setError(msg);
      toast(msg, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink-900">Evaluation</h1>
          <p className="mt-0.5 text-sm text-ink-500">
            Run the backend evaluation harness and inspect retrieval &amp;
            answer quality.
          </p>
        </div>
        <button
          onClick={() => void run()}
          disabled={loading}
          className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-700 disabled:cursor-not-allowed disabled:bg-ink-300"
        >
          {loading ? "Running…" : "Run evaluation"}
        </button>
      </div>

      {error && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {!data && !loading && !error && (
        <div className="mt-8 rounded-xl border border-dashed border-ink-300 bg-white p-10 text-center text-sm text-ink-500">
          Click <span className="font-medium">Run evaluation</span> to generate
          metrics.
        </div>
      )}

      {loading && !data && (
        <div className="mt-8 rounded-xl border border-ink-200 bg-white p-10 text-center text-sm text-ink-500">
          Running evaluation…
        </div>
      )}

      {data && (
        <div className="mt-6 flex flex-col gap-6">
          {/* Headline metrics */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Recall@k"
              value={pct(data.recall_at_k)}
              hint={`k = ${data.k}`}
            />
            <MetricCard label="MRR" value={data.mrr.toFixed(3)} />
            <MetricCard
              label="Citation accuracy"
              value={pct(data.citation_accuracy)}
            />
            <MetricCard label="Faithfulness" value={pct(data.faithfulness)} />
          </div>

          {/* Secondary info */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-xl border border-ink-200 bg-white px-4 py-2">
              <span className="text-xs font-medium uppercase tracking-wide text-ink-500">
                Abstention
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                  data.abstention_correct
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-red-100 text-red-800"
                }`}
              >
                {data.abstention_correct ? "PASS" : "FAIL"}
              </span>
            </div>
            <div className="rounded-xl border border-ink-200 bg-white px-4 py-2 text-xs text-ink-500">
              {data.n_questions} questions evaluated
            </div>
          </div>

          {data.provider_note && (
            <div className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-800">
              <span className="font-medium">Provider note: </span>
              {data.provider_note}
            </div>
          )}

          {/* Per-question table */}
          <div className="overflow-hidden rounded-xl border border-ink-200 bg-white">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="bg-ink-50 text-xs uppercase tracking-wide text-ink-500">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Question</th>
                    <th className="px-4 py-2.5 font-medium">Expected</th>
                    <th className="px-4 py-2.5 font-medium">In top-k</th>
                    <th className="px-4 py-2.5 font-medium">Rank</th>
                    <th className="px-4 py-2.5 font-medium">Abstained</th>
                    <th className="px-4 py-2.5 font-medium">Faithfulness</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {data.per_question.map((row, i) => (
                    <tr key={i} className="align-top hover:bg-ink-50">
                      <td className="px-4 py-3 text-ink-800">{row.q}</td>
                      <td className="px-4 py-3 text-ink-600">{row.expected}</td>
                      <td className="px-4 py-3">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                            row.in_topk
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-ink-100 text-ink-500"
                          }`}
                        >
                          {row.in_topk ? "yes" : "no"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-ink-600">
                        {row.rank > 0 ? row.rank : "—"}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                            row.abstained
                              ? "bg-amber-100 text-amber-800"
                              : "bg-ink-100 text-ink-500"
                          }`}
                        >
                          {row.abstained ? "yes" : "no"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-ink-600">
                        {pct(row.faithfulness)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
