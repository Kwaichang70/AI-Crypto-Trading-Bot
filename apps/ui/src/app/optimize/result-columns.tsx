/**
 * apps/ui/src/app/optimize/result-columns.tsx
 * --------------------------------------------
 * Shared column factory for the optimization results DataTable.
 *
 * Extracted from optimize/page.tsx so that both the main optimize page and the
 * saved-run detail page (optimize/[id]/page.tsx) can import and reuse the
 * identical column definitions without duplication.
 */

import type { Column } from "@/components/ui/data-table";
import type { OptimizeEntry } from "@/lib/types";

/**
 * Build the column definitions for the optimization results table.
 *
 * @param rankBy        The metric key currently used for ranking (e.g. "sharpe_ratio").
 *                      Columns matching this key receive a star suffix and bold indigo styling.
 * @param launchingRank The rank number of the entry currently being launched as a run
 *                      (shows a spinner/disabled state on that row's button), or null if idle.
 * @param onLaunch      Callback invoked when the user clicks "Launch Run" for an entry.
 */
export function buildResultColumns(
  rankBy: string,
  launchingRank: number | null,
  onLaunch: (entry: OptimizeEntry) => void,
): Column<OptimizeEntry>[] {
  return [
    {
      key: "rank",
      header: "#",
      render: (e) => (
        <span
          className={
            e.rank === 1
              ? "font-bold text-yellow-400"
              : "text-slate-400"
          }
        >
          {e.rank}
        </span>
      ),
    },
    {
      key: "params",
      header: "Parameters",
      render: (e) => (
        <div className="flex flex-wrap gap-1">
          {Object.entries(e.params).map(([k, v]) => (
            <span
              key={k}
              className="rounded bg-slate-700 px-1.5 py-0.5 font-mono text-xs text-slate-300"
            >
              {k}={String(v)}
            </span>
          ))}
        </div>
      ),
    },
    {
      key: "sharpe_ratio",
      header: rankBy === "sharpe_ratio" ? "Sharpe ★" : "Sharpe",
      sortable: true,
      sortValue: (e) => e.metrics["sharpe_ratio"] ?? -Infinity,
      render: (e) => (
        <span className={rankBy === "sharpe_ratio" ? "font-semibold text-indigo-300" : ""}>
          {(e.metrics["sharpe_ratio"] ?? 0).toFixed(3)}
        </span>
      ),
    },
    {
      key: "total_return_pct",
      header: rankBy === "total_return_pct" ? "Return % ★" : "Return %",
      sortable: true,
      sortValue: (e) => e.metrics["total_return_pct"] ?? -Infinity,
      render: (e) => {
        const v = e.metrics["total_return_pct"] ?? 0;
        return (
          <span
            className={
              rankBy === "total_return_pct"
                ? "font-semibold text-indigo-300"
                : v >= 0
                  ? "text-emerald-400"
                  : "text-red-400"
            }
          >
            {v >= 0 ? "+" : ""}
            {(v * 100).toFixed(2)}%
          </span>
        );
      },
    },
    {
      key: "max_drawdown_pct",
      header: rankBy === "max_drawdown_pct" ? "Max DD ★" : "Max DD",
      sortable: true,
      sortValue: (e) => e.metrics["max_drawdown_pct"] ?? Infinity,
      render: (e) => (
        <span className={rankBy === "max_drawdown_pct" ? "font-semibold text-indigo-300" : "text-red-400"}>
          {((e.metrics["max_drawdown_pct"] ?? 0) * 100).toFixed(2)}%
        </span>
      ),
    },
    {
      key: "win_rate",
      header: rankBy === "win_rate" ? "Win Rate ★" : "Win Rate",
      sortable: true,
      sortValue: (e) => e.metrics["win_rate"] ?? -Infinity,
      render: (e) => (
        <span className={rankBy === "win_rate" ? "font-semibold text-indigo-300" : ""}>
          {((e.metrics["win_rate"] ?? 0) * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      key: "total_trades",
      header: "Trades",
      sortable: true,
      sortValue: (e) => e.metrics["total_trades"] ?? 0,
      render: (e) => String(Math.round(e.metrics["total_trades"] ?? 0)),
    },
    {
      key: "actions",
      header: "",
      render: (e) => (
        <button
          onClick={() => onLaunch(e)}
          disabled={launchingRank === e.rank}
          className="rounded-lg bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {launchingRank === e.rank ? "Starting…" : "Launch Run"}
        </button>
      ),
    },
  ];
}
