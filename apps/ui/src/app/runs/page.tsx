/**
 * Run List Page — Client Component (requires filter state).
 */
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { fetchRuns, formatPct } from "@/lib/api";
import type { Run, RunMode, RunStatus } from "@/lib/types";
import type { CsvColumn } from "@/lib/csv-export";
import { ExportCsvButton } from "@/components/ui/export-csv-button";
import { DataTable, type Column } from "@/components/ui/data-table";
import { RunStatusBadge } from "@/components/ui/status-badge";
import { Header } from "@/components/layout/header";

const MODE_OPTIONS: { value: RunMode | "all"; label: string }[] = [
  { value: "all", label: "All modes" },
  { value: "backtest", label: "Backtest" },
  { value: "paper", label: "Paper" },
  { value: "live", label: "Live" },
];

const STATUS_OPTIONS: { value: RunStatus | "all"; label: string }[] = [
  { value: "all", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "stopped", label: "Stopped" },
  { value: "error", label: "Error" },
];

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const;

const MAX_COMPARE = 5;

/**
 * Build the column list. The checkbox column is first and captures the
 * selectedIds set + toggle handler via closure — this means the columns array
 * must be computed inside the component render rather than at module scope.
 */
function buildColumns(
  selectedIds: ReadonlySet<string>,
  onToggle: (id: string) => void,
): Column<Run>[] {
  return [
    {
      key: "_select",
      header: "",
      className: "w-10",
      render: (run) => {
        const checked = selectedIds.has(run.id);
        const atLimit = selectedIds.size >= MAX_COMPARE && !checked;
        return (
          <input
            type="checkbox"
            checked={checked}
            disabled={atLimit}
            onChange={() => onToggle(run.id)}
            onClick={(e) => e.stopPropagation()}
            aria-label={`Select run ${run.id.slice(0, 8)}`}
            className="h-4 w-4 cursor-pointer rounded border-slate-300 bg-white accent-indigo-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:bg-slate-800"
          />
        );
      },
    },
    {
      key: "id",
      header: "Run ID",
      render: (run) => (
        <Link
          href={`/runs/${run.id}`}
          className="font-mono text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 dark:hover:text-indigo-300 hover:underline"
        >
          {run.id.slice(0, 8)}…
        </Link>
      ),
    },
    {
      key: "mode",
      header: "Mode",
      render: (run) => (
        <span className="capitalize text-slate-500 dark:text-slate-400">{run.runMode}</span>
      ),
    },
    {
      key: "strategy",
      header: "Strategy",
      render: (run) => (
        <span className="text-slate-700 dark:text-slate-300">{run.config?.strategy_name ?? "—"}</span>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (run) => <RunStatusBadge status={run.status} />,
    },
    {
      key: "returnPct",
      header: "Return %",
      sortable: true,
      sortValue: (run) => run.backtestMetrics?.totalReturnPct ?? 0,
      render: (run) => {
        const val = run.backtestMetrics?.totalReturnPct;
        if (val === undefined || val === null) {
          return <span className="text-slate-400 dark:text-slate-600 text-xs">—</span>;
        }
        const isPositive = val >= 0;
        return (
          <span
            className={`font-mono text-xs font-medium ${isPositive ? "text-profit" : "text-loss"}`}
          >
            {isPositive ? "+" : ""}{formatPct(val)}
          </span>
        );
      },
    },
    {
      key: "trades",
      header: "Trades",
      sortable: true,
      sortValue: (run) => run.backtestMetrics?.totalTrades ?? 0,
      render: (run) => {
        const val = run.backtestMetrics?.totalTrades;
        if (val === undefined || val === null) {
          return <span className="text-slate-400 dark:text-slate-600 text-xs">—</span>;
        }
        return <span className="font-mono text-xs text-slate-700 dark:text-slate-300">{val}</span>;
      },
    },
    {
      key: "sharpe",
      header: "Sharpe",
      sortable: true,
      sortValue: (run) => run.backtestMetrics?.sharpeRatio ?? 0,
      render: (run) => {
        const val = run.backtestMetrics?.sharpeRatio;
        if (val === undefined || val === null) {
          return <span className="text-slate-400 dark:text-slate-600 text-xs">—</span>;
        }
        const colorClass =
          val >= 1.0 ? "text-profit" : val < 0 ? "text-loss" : "text-slate-700 dark:text-slate-300";
        return (
          <span className={`font-mono text-xs font-medium ${colorClass}`}>
            {val.toFixed(2)}
          </span>
        );
      },
    },
    {
      key: "symbols",
      header: "Symbols",
      render: (run) => (
        <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
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
}

const RUNS_CSV_COLUMNS: CsvColumn<Run>[] = [
  { header: "Run ID", value: (r) => r.id },
  { header: "Mode", value: (r) => r.runMode },
  { header: "Strategy", value: (r) => r.config?.strategy_name ?? "" },
  { header: "Status", value: (r) => r.status },
  {
    header: "Return %",
    value: (r) =>
      r.backtestMetrics?.totalReturnPct != null
        ? (r.backtestMetrics.totalReturnPct * 100).toFixed(2)
        : "",
  },
  { header: "Trades", value: (r) => r.backtestMetrics?.totalTrades ?? "" },
  {
    header: "Sharpe",
    value: (r) =>
      r.backtestMetrics?.sharpeRatio != null
        ? r.backtestMetrics.sharpeRatio.toFixed(4)
        : "",
  },
  { header: "Symbols", value: (r) => r.config?.symbols?.join("; ") ?? "" },
  { header: "Created At", value: (r) => r.createdAt },
];

export default function RunsPage() {
  const router = useRouter();

  const [runs, setRuns] = useState<readonly Run[]>([]);
  const [total, setTotal] = useState(0);
  const [modeFilter, setModeFilter] = useState<RunMode | "all">("all");
  const [statusFilter, setStatusFilter] = useState<RunStatus | "all">("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination state — page is 0-indexed.
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState<number>(25);

  // Advanced filter state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [strategyFilter, setStrategyFilter] = useState("");
  const [symbolFilter, setSymbolFilter] = useState("");
  const [dateAfter, setDateAfter] = useState("");
  const [dateBefore, setDateBefore] = useState("");

  // Debounced versions — only used in fetchRuns call to avoid per-keystroke fetches
  const [debouncedStrategy, setDebouncedStrategy] = useState("");
  const [debouncedSymbol, setDebouncedSymbol] = useState("");

  // Run comparison selection — max MAX_COMPARE runs
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(new Set());

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedStrategy(strategyFilter), 500);
    return () => clearTimeout(timer);
  }, [strategyFilter]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSymbol(symbolFilter), 500);
    return () => clearTimeout(timer);
  }, [symbolFilter]);

  useEffect(() => {
    async function load() {
      setIsLoading(true);
      setError(null);
      const result = await fetchRuns({
        offset: page * pageSize,
        limit: pageSize,
        ...(modeFilter !== "all" ? { mode: modeFilter } : {}),
        ...(statusFilter !== "all" ? { status: statusFilter } : {}),
        ...(debouncedStrategy ? { strategy: debouncedStrategy } : {}),
        ...(debouncedSymbol ? { symbol: debouncedSymbol } : {}),
        ...(dateAfter ? { createdAfter: new Date(dateAfter).toISOString() } : {}),
        ...(dateBefore ? { createdBefore: new Date(dateBefore).toISOString() } : {}),
      });
      if (result.ok) {
        setRuns(result.data.items);
        setTotal(result.data.total);
      } else {
        setError(result.error.message);
      }
      setIsLoading(false);
    }
    void load();
  }, [page, pageSize, modeFilter, statusFilter, debouncedStrategy, debouncedSymbol, dateAfter, dateBefore]);

  // Pagination derived values.
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const rangeStart = total === 0 ? 0 : page * pageSize + 1;
  const rangeEnd = Math.min((page + 1) * pageSize, total);

  const displayCount = total;

  function handleModeFilter(value: RunMode | "all") {
    setModeFilter(value);
    setPage(0);
  }

  function handleStatusFilter(value: RunStatus | "all") {
    setStatusFilter(value);
    setPage(0);
  }

  function handleToggleSelect(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < MAX_COMPARE) {
        next.add(id);
      }
      return next;
    });
  }

  function handleCompare() {
    const ids = Array.from(selectedIds).join(",");
    router.push(`/runs/compare?ids=${ids}`);
  }

  const columns = buildColumns(selectedIds, handleToggleSelect);

  return (
    <div className="space-y-6">
      <Header
        title="Runs"
        subtitle={`${total} total trading runs`}
        actions={
          <div className="flex items-center gap-2">
            {selectedIds.size >= 2 && (
              <button
                onClick={handleCompare}
                className="rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-600"
              >
                Compare ({selectedIds.size})
              </button>
            )}
            <ExportCsvButton
              filename="runs.csv"
              columns={RUNS_CSV_COLUMNS}
              data={runs}
              disabled={isLoading}
              size="sm"
              label="Export Page"
            />
            <Link
              href="/runs/new"
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
            >
              New Run
            </Link>
          </div>
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
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200",
            ].join(" ")}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {STATUS_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => handleStatusFilter(opt.value)}
            className={[
              "rounded-full px-3 py-1 text-xs font-medium transition-colors",
              statusFilter === opt.value
                ? "bg-emerald-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200",
            ].join(" ")}
          >
            {opt.label}
          </button>
        ))}

        {/* Advanced filters toggle */}
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="ml-2 flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
            />
          </svg>
          {showAdvanced ? "Hide" : "More"} Filters
        </button>

        {/* Selection hint */}
        {selectedIds.size > 0 && (
          <span className="ml-auto text-xs text-slate-500">
            {selectedIds.size} / {MAX_COMPARE} selected
            {selectedIds.size >= 2 && (
              <button
                onClick={() => setSelectedIds(new Set())}
                className="ml-2 text-slate-600 hover:text-slate-400"
              >
                Clear
              </button>
            )}
          </span>
        )}
      </div>

      {showAdvanced && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 p-3">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">
              Strategy
            </label>
            <input
              type="text"
              value={strategyFilter}
              onChange={(e) => {
                setStrategyFilter(e.target.value);
                setPage(0);
              }}
              placeholder="e.g. rsi_mean_reversion"
              className="w-full rounded-lg border border-slate-300 bg-white text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none px-3 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-600"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">
              Symbol
            </label>
            <input
              type="text"
              value={symbolFilter}
              onChange={(e) => {
                setSymbolFilter(e.target.value);
                setPage(0);
              }}
              placeholder="e.g. BTC/USD"
              className="w-full rounded-lg border border-slate-300 bg-white text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none px-3 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-600"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">
              From
            </label>
            <input
              type="date"
              value={dateAfter}
              onChange={(e) => {
                setDateAfter(e.target.value);
                setPage(0);
              }}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">
              Until
            </label>
            <input
              type="date"
              value={dateBefore}
              onChange={(e) => {
                setDateBefore(e.target.value);
                setPage(0);
              }}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            />
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      <DataTable
        columns={columns}
        data={runs}
        keyExtractor={(r) => r.id}
        emptyMessage="No runs found. Create your first backtest."
        isLoading={isLoading}
      />

      {/* Pagination controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-slate-100 dark:bg-slate-800 px-4 py-3 text-sm">
        {/* Left: rows per page */}
        <div className="flex items-center gap-2">
          <span className="text-slate-500 dark:text-slate-400">Rows per page</span>
          <select
            value={pageSize}
            onChange={(e) => {
              const parsed = Number(e.target.value);
              if (!isNaN(parsed) && parsed > 0) {
                setPageSize(parsed);
                setPage(0);
              }
            }}
            className="rounded bg-slate-200 dark:bg-slate-700 px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
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
        <span className="text-slate-500 dark:text-slate-400 tabular-nums">
          {displayCount === 0
            ? "No results"
            : `${rangeStart}–${rangeEnd} of ${total}`}
        </span>

        {/* Right: prev / next buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-400 dark:disabled:text-slate-600 enabled:text-slate-700 dark:enabled:text-slate-300 enabled:hover:bg-slate-200 dark:enabled:hover:bg-slate-700 enabled:hover:text-slate-900 dark:enabled:hover:text-slate-100"
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
            className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-400 dark:disabled:text-slate-600 enabled:text-slate-700 dark:enabled:text-slate-300 enabled:hover:bg-slate-200 dark:enabled:hover:bg-slate-700 enabled:hover:text-slate-900 dark:enabled:hover:text-slate-100"
            aria-label="Next page"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
