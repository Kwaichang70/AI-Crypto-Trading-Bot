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

  useEffect(() => {
    async function load() {
      setIsLoading(true);
      setError(null);
      const result = await fetchRuns({ limit: 100 });
      if (result.ok) {
        setRuns(result.data.items);
        setTotal(result.data.total);
      } else {
        setError(result.error.message);
      }
      setIsLoading(false);
    }
    void load();
  }, []);

  const filtered =
    modeFilter === "all"
      ? runs
      : runs.filter((r) => r.runMode === modeFilter);

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
            onClick={() => setModeFilter(opt.value)}
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
    </div>
  );
}
