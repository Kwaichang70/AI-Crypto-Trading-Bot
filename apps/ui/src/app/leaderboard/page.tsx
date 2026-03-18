/**
 * Strategy Leaderboard — aggregated performance metrics per strategy.
 *
 * Fetches all runs (up to 1000), groups by strategy_name, and computes
 * per-strategy aggregates: run count, avg return, best/worst return,
 * avg Sharpe, avg win rate, total trades.
 */
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchRuns, formatPct } from "@/lib/api";
import type { Run } from "@/lib/types";
import { Header } from "@/components/layout/header";
import { ExportCsvButton } from "@/components/ui/export-csv-button";
import type { CsvColumn } from "@/lib/csv-export";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StrategyStats {
  name: string;
  runCount: number;
  avgReturnPct: number;
  bestReturnPct: number;
  worstReturnPct: number;
  avgSharpe: number;
  avgWinRate: number;
  totalTrades: number;
  avgProfitFactor: number;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

function aggregateByStrategy(runs: readonly Run[]): StrategyStats[] {
  const groups = new Map<string, Run[]>();

  for (const run of runs) {
    const name = run.config?.strategy_name ?? "unknown";
    const list = groups.get(name);
    if (list) {
      list.push(run);
    } else {
      groups.set(name, [run]);
    }
  }

  const stats: StrategyStats[] = [];

  for (const [name, group] of groups) {
    const withMetrics = group.filter((r) => r.backtestMetrics != null);
    if (withMetrics.length === 0) {
      stats.push({
        name,
        runCount: group.length,
        avgReturnPct: 0,
        bestReturnPct: 0,
        worstReturnPct: 0,
        avgSharpe: 0,
        avgWinRate: 0,
        totalTrades: 0,
        avgProfitFactor: 0,
      });
      continue;
    }

    const returns = withMetrics.map((r) => r.backtestMetrics!.totalReturnPct);
    const sharpes = withMetrics.map((r) => r.backtestMetrics!.sharpeRatio);
    const winRates = withMetrics.map((r) => r.backtestMetrics!.winRate);
    const trades = withMetrics.map((r) => r.backtestMetrics!.totalTrades);
    const profitFactors = withMetrics.map((r) => r.backtestMetrics!.profitFactor);

    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

    stats.push({
      name,
      runCount: group.length,
      avgReturnPct: avg(returns),
      bestReturnPct: Math.max(...returns),
      worstReturnPct: Math.min(...returns),
      avgSharpe: avg(sharpes),
      avgWinRate: avg(winRates),
      totalTrades: trades.reduce((a, b) => a + b, 0),
      avgProfitFactor: avg(profitFactors),
    });
  }

  // Sort by avg Sharpe descending
  stats.sort((a, b) => b.avgSharpe - a.avgSharpe);
  return stats;
}

// ---------------------------------------------------------------------------
// CSV columns
// ---------------------------------------------------------------------------

const CSV_COLUMNS: CsvColumn<StrategyStats>[] = [
  { header: "Strategy", value: (s) => s.name },
  { header: "Runs", value: (s) => s.runCount },
  { header: "Avg Return %", value: (s) => (s.avgReturnPct * 100).toFixed(2) },
  { header: "Best Return %", value: (s) => (s.bestReturnPct * 100).toFixed(2) },
  { header: "Worst Return %", value: (s) => (s.worstReturnPct * 100).toFixed(2) },
  { header: "Avg Sharpe", value: (s) => s.avgSharpe.toFixed(4) },
  { header: "Avg Win Rate %", value: (s) => (s.avgWinRate * 100).toFixed(2) },
  { header: "Total Trades", value: (s) => s.totalTrades },
  { header: "Avg Profit Factor", value: (s) => s.avgProfitFactor.toFixed(2) },
];

// ---------------------------------------------------------------------------
// Sort state
// ---------------------------------------------------------------------------

type SortKey = keyof Omit<StrategyStats, "name">;

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "avgSharpe", label: "Avg Sharpe" },
  { value: "avgReturnPct", label: "Avg Return" },
  { value: "bestReturnPct", label: "Best Return" },
  { value: "avgWinRate", label: "Avg Win Rate" },
  { value: "avgProfitFactor", label: "Avg Profit Factor" },
  { value: "totalTrades", label: "Total Trades" },
  { value: "runCount", label: "Run Count" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pctCell(val: number): React.ReactNode {
  const cls = val > 0 ? "text-profit" : val < 0 ? "text-loss" : "text-slate-400";
  return (
    <span className={`font-mono text-xs font-medium ${cls}`}>
      {val >= 0 ? "+" : ""}{formatPct(val)}
    </span>
  );
}

function numCell(val: number, decimals = 2): React.ReactNode {
  const cls = val >= 1.0 ? "text-profit" : val < 0 ? "text-loss" : "text-slate-300";
  return (
    <span className={`font-mono text-xs font-medium ${cls}`}>
      {val.toFixed(decimals)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LeaderboardPage() {
  const [stats, setStats] = useState<StrategyStats[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortKey>("avgSharpe");

  useEffect(() => {
    async function load() {
      setIsLoading(true);
      setError(null);
      // Fetch all runs in pages of 500 (API max limit).
      const allRuns: Run[] = [];
      let offset = 0;
      const PAGE = 500;
      while (true) {
        const result = await fetchRuns({ limit: PAGE, offset });
        if (!result.ok) {
          setError(result.error.message);
          break;
        }
        allRuns.push(...result.data.items);
        if (allRuns.length >= result.data.total) break;
        offset += PAGE;
      }
      setStats(aggregateByStrategy(allRuns));
      setIsLoading(false);
    }
    void load();
  }, []);

  // Re-sort when sortBy changes
  const sorted = [...stats].sort((a, b) => {
    const av = a[sortBy] as number;
    const bv = b[sortBy] as number;
    return bv - av; // descending
  });

  return (
    <div className="space-y-6">
      <Header
        title="Strategy Leaderboard"
        subtitle={`${stats.length} strategies ranked by performance`}
        actions={
          <ExportCsvButton
            filename="strategy-leaderboard.csv"
            columns={CSV_COLUMNS}
            data={sorted}
            disabled={isLoading}
            size="sm"
          />
        }
      />

      {/* Sort control */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">Rank by</span>
        <div className="flex flex-wrap gap-1.5">
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setSortBy(opt.value)}
              className={[
                "rounded-full px-3 py-1 text-xs font-medium transition-colors",
                sortBy === opt.value
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200",
              ].join(" ")}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-16 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />
          ))}
        </div>
      ) : sorted.length === 0 ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50 py-12 text-center text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/40">
          No runs with metrics found. Run some backtests first.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/50">
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500 w-8">
                  #
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Strategy
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Runs
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Avg Return
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Best
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Worst
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Avg Sharpe
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Win Rate
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Trades
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Profit Factor
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((s, idx) => (
                <tr
                  key={s.name}
                  className="border-b border-slate-100 transition-colors hover:bg-slate-50 dark:border-slate-800/50 dark:hover:bg-slate-800/30"
                >
                  <td className="px-4 py-3">
                    <span className={`font-mono text-xs font-bold ${
                      idx === 0 ? "text-amber-400" : idx === 1 ? "text-slate-300" : idx === 2 ? "text-amber-700" : "text-slate-500"
                    }`}>
                      {idx + 1}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/runs?strategy=${encodeURIComponent(s.name)}`}
                      className="font-medium text-indigo-400 hover:text-indigo-300 hover:underline"
                    >
                      {s.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-slate-400">{s.runCount}</span>
                  </td>
                  <td className="px-4 py-3">{pctCell(s.avgReturnPct)}</td>
                  <td className="px-4 py-3">{pctCell(s.bestReturnPct)}</td>
                  <td className="px-4 py-3">{pctCell(s.worstReturnPct)}</td>
                  <td className="px-4 py-3">{numCell(s.avgSharpe, 3)}</td>
                  <td className="px-4 py-3">{pctCell(s.avgWinRate)}</td>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-slate-300">{s.totalTrades}</span>
                  </td>
                  <td className="px-4 py-3">{numCell(s.avgProfitFactor, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
