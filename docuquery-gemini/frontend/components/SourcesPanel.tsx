"use client";

import type { Citation, RetrievedChunk } from "@/lib/api";

function formatSimilarity(sim: number): string {
  // Similarity is typically 0..1; show as percentage with 1 decimal.
  return `${(sim * 100).toFixed(1)}%`;
}

export default function SourcesPanel({
  retrieved,
  citations,
}: {
  retrieved: RetrievedChunk[];
  citations: Citation[];
}) {
  // Map chunk_id -> marker for highlighting cited chunks.
  const markerByChunk = new Map<string, number>();
  for (const c of citations) {
    if (!markerByChunk.has(c.chunk_id)) {
      markerByChunk.set(c.chunk_id, c.marker);
    }
  }

  if (retrieved.length === 0) {
    return (
      <div className="rounded-xl border border-ink-200 bg-white p-4 text-sm text-ink-400">
        Retrieved sources will appear here after you ask a question.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-500">
          Sources
        </h3>
        <span className="text-[11px] text-ink-400">
          {citations.length} cited / {retrieved.length} retrieved
        </span>
      </div>

      <p className="text-[11px] leading-relaxed text-ink-400">
        Every retrieved chunk is listed below. Chunks the answer actually cites
        are{" "}
        <span className="font-medium text-accent-700">
          highlighted and tagged with their [marker]
        </span>
        .
      </p>

      <ul className="space-y-2">
        {retrieved.map((chunk, idx) => {
          const marker = markerByChunk.get(chunk.chunk_id);
          const cited = marker !== undefined;
          return (
            <li
              key={`${chunk.chunk_id}-${idx}`}
              className={`rounded-xl border p-3 transition-colors ${
                cited
                  ? "border-accent-300 bg-accent-50 ring-1 ring-accent-200"
                  : "border-ink-200 bg-white"
              }`}
            >
              <div className="mb-1.5 flex items-start justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  {cited && (
                    <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-600 px-1.5 text-[11px] font-bold text-white">
                      {marker}
                    </span>
                  )}
                  <span
                    className="truncate text-sm font-medium text-ink-800"
                    title={chunk.document_name}
                  >
                    {chunk.document_name}
                  </span>
                </div>
                <span className="shrink-0 rounded-full bg-ink-100 px-2 py-0.5 text-[11px] font-medium text-ink-600">
                  {formatSimilarity(chunk.similarity)}
                </span>
              </div>

              <div className="mb-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-ink-500">
                <span>page {chunk.page_number}</span>
                {chunk.section_title && (
                  <span className="truncate" title={chunk.section_title}>
                    § {chunk.section_title}
                  </span>
                )}
                {cited && (
                  <span className="font-medium text-accent-700">cited</span>
                )}
              </div>

              <p className="text-xs leading-relaxed text-ink-600">
                {chunk.text}
              </p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
