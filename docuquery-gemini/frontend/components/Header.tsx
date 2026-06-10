"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getHealth, getUserId, type HealthResponse } from "@/lib/api";

function NavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const active = pathname === href;
  return (
    <Link
      href={href}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
        active
          ? "bg-ink-900 text-white"
          : "text-ink-600 hover:bg-ink-100 hover:text-ink-900"
      }`}
    >
      {label}
    </Link>
  );
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-ink-200 bg-white px-2 py-0.5 text-[11px] text-ink-600">
      <span className="text-ink-400">{label}</span>
      <span className="font-medium text-ink-800">{value}</span>
    </span>
  );
}

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [userId, setUserId] = useState<string>("");

  useEffect(() => {
    setUserId(getUserId());
    let active = true;
    getHealth()
      .then((h) => {
        if (active) {
          setHealth(h);
          setHealthError(false);
        }
      })
      .catch(() => {
        if (active) setHealthError(true);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-ink-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent-600 text-sm font-bold text-white">
              DQ
            </span>
            <span className="text-base font-semibold tracking-tight text-ink-900">
              DocuQuery<span className="text-accent-600">-Gemini</span>
            </span>
          </Link>
          <nav className="flex items-center gap-1">
            <NavLink href="/" label="Chat" />
            <NavLink href="/eval" label="Eval" />
          </nav>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {healthError ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-red-200 bg-red-50 px-2 py-0.5 text-[11px] text-red-700">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              backend offline
            </span>
          ) : health ? (
            <>
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[11px] text-emerald-700">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                {health.status}
              </span>
              <StatusPill label="embed" value={health.embedding_provider} />
              <StatusPill label="llm" value={health.llm_provider} />
              <StatusPill label="store" value={health.store_backend} />
              <span
                className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${
                  health.gemini_enabled
                    ? "border-accent-200 bg-accent-50 text-accent-700"
                    : "border-ink-200 bg-ink-100 text-ink-500"
                }`}
              >
                Gemini {health.gemini_enabled ? "on" : "off"}
              </span>
            </>
          ) : (
            <span className="text-[11px] text-ink-400">checking status…</span>
          )}
          {userId && (
            <span
              className="rounded-full border border-ink-200 bg-ink-50 px-2 py-0.5 font-mono text-[11px] text-ink-500"
              title="Your local user id (sent as X-User-Id)"
            >
              {userId}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
