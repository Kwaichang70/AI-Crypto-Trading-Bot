/**
 * Run List Page — Client Component (requires filter state).
 */
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchRuns } from "@/lib/api";
import type { Run, RunMode } from "@/lib/types";
import { DataTable, type Column } from "@/components/ui/data-table";
import { RunStatusBadge } from "@/components/ui/status-badge";
import { Header } from "@/components/layout/header";

const MODE_OPTIONS: { value: RunMode | "all"; label: string }[] = [
  { value: "all", label: "All modes" },
  { value: "backtest", label: "Backtest" },
  { value: "paper", label: "Paper" },
  { value: "live", label: "Live" },
];

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const;

const COLUMNS: Column<Run>[] = [
  {
    key: "id",
    header: "Run ID",
    render: (run) => (
      <Link
        href={`/runs/${run.id}`}
        className="font-mono text-xs text-indigo-400 hover:text-indigo-300 hover:underline"
      >
        {run.id.slice(0, 8)}…
      </Link>
    ),
  },
  {
    key: "mode",
    header: "Mode",
    render: (run) => (
      <span className="capitalize text-slate-400">{run.runMode}</span>
    ),
  },
  {
    key: "strategy",
    header: "Strategy",
    render: (run) => (
      <span className="text-slate-300">{run.config?.strategyName ?? "—"}</span>
    ),
  },
  {
    key: "status",
    header: "Status",
    render: (run) => <RunStatusBadge status={run.status} />,
  },
  {
    key: "symbols",
    header: "Symbols",
    render: (run) => (
      <span className="font-mono text-xs text-slate-400">
        {run.config?.symbols?.join(", ") ?? "—"}
      </span>
    ),
  },
  {
    key: "created",
    header: "Created",
    render: (run) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(run.createdAt).toLocaleString()}
      </span>
    ),
  },
];

export default function RunsPage() {
  const [runs, setRuns] = useState<readonly Run[]>([]);
  const [total, setTotal] = useState(0);
  const [modeFilter, setModeFilter] = useState<RunMode | "all">("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination state — page is 0-indexed.
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState<number>(25);

  useEffect(() => {
    async function load() {
      setIsLoading(true);
      setError(null);
      const result = await fetchRuns({ offset: page * pageSize, limit: pageSize });
      if (result.ok) {
        setRuns(result.data.items);
        setTotal(result.data.total);
      } else {
        setError(result.error.message);
      }
      setIsLoading(false);
    }
    void load();
  }, [page, pageSize]);

  // Client-side mode filter applied to the current fetched page only.
  const filtered =
    modeFilter === "all"
      ? runs
      : runs.filter((r) => r.runMode === modeFilter);

  // Pagination derived values.
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const rangeStart = total === 0 ? 0 : page * pageSize + 1;
  const rangeEnd = Math.min((page + 1) * pageSize, total);

  // When a mode filter is active, show the filtered row count for the range display.
  const displayCount = modeFilter === "all" ? total : filtered.length;

  function handleModeFilter(value: RunMode | "all") {
    setModeFilter(value);
    setPage(0);
  }

  return (
    <div className="space-y-6">
      <Header
        title="Runs"
        subtitle={`${total} total trading runs`}
        actions={
          <Link
            href="/runs/new"
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
          >
            New Run
          </Link>
        }
      />

      {/* Filter bar */}
      <div className="flex flex-wrap gap-2">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => handleModeFilter(opt.value)}
            className={[
              "rounded-full px-3 py-1 text-xs font-medium transition-colors",
              modeFilter === opt.value
                ? "bg-indigo-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200",
            ].join(" ")}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <DataTable
        columns={COLUMNS}
        data={filtered}
        keyExtractor={(r) => r.id}
        emptyMessage="No runs found. Create your first backtest."
        isLoading={isLoading}
      />

      {/* Pagination controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-slate-800 px-4 py-3 text-sm">
        {/* Left: rows per page */}
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Rows per page</span>
          <select
            value={pageSize}
            onChange={(e) => {
              const parsed = Number(e.target.value);
              if (!isNaN(parsed) && parsed > 0) {
                setPageSize(parsed);
                setPage(0);
              }
            }}
            className="rounded bg-slate-700 px-2 py-1 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            aria-label="Rows per page"
          >
            {PAGE_SIZE_OPTIONS.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </div>

        {/* Center: range display */}
        <span className="text-slate-400 tabular-nums">
          {displayCount === 0
            ? "No results"
            : modeFilter === "all"
              ? `${rangeStart}–${rangeEnd} of ${total}`
              : `${filtered.length} of ${total} (filtered)`}
        </span>

        {/* Right: prev / next buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-600 enabled:text-slate-300 enabled:hover:bg-slate-700 enabled:hover:text-slate-100"
            aria-label="Previous page"
          >
            Prev
          </button>
          <span className="text-slate-500 tabular-nums">
            {page + 1} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-600 enabled:text-slate-300 enabled:hover:bg-slate-700 enabled:hover:text-slate-100"
            aria-label="Next page"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
