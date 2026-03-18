/**
 * Run Comparison Page — side-by-side metrics + overlaid equity curves.
 *
 * Route: /runs/compare?ids=<uuid1>,<uuid2>[,<uuid3>,<uuid4>,<uuid5>]
 *
 * Fetches run, portfolio, and equity-curve data in parallel for every
 * requested run ID, then renders:
 *  1. A metrics comparison table (runs as columns, metric rows)
 *  2. An overlaid equity-curve LineChart (one line per run)
 */
"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchRun,
  fetchPortfolio,
  fetchEquityCurve,
  formatCurrency,
  formatPct,
} from "@/lib/api";
import type { Run, Portfolio, EquityPoint } from "@/lib/types";
import { Header } from "@/components/layout/header";
import { RunStatusBadge } from "@/components/ui/status-badge";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Distinct Tailwind-compatible hex colors for up to 5 runs. */
const RUN_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"] as const;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RunData {
  run: Run;
  portfolio: Portfolio | null;
  equityPoints: readonly EquityPoint[];
  /** Fetch error message, if any non-fatal error occurred for secondary data. */
  error: string | null;
}

// ---------------------------------------------------------------------------
// Metric row definitions
// ---------------------------------------------------------------------------

/**
 * A MetricRow describes one row in the comparison table.
 * `extract` returns the raw number for best-value highlighting.
 * `display` returns the formatted string cell content.
 * `higherIsBetter` controls which end of the range gets highlighted green.
 */
interface MetricRow {
  label: string;
  extract: (d: RunData) => number | null;
  display: (d: RunData) => string;
  higherIsBetter: boolean;
}

function pick<T>(source: T | null | undefined, key: keyof T): T[keyof T] | null {
  return source != null ? source[key] : null;
}

const METRIC_ROWS: MetricRow[] = [
  {
    label: "Strategy",
    extract: () => null,
    display: (d) => d.run.config?.strategy_name ?? "—",
    higherIsBetter: true,
  },
  {
    label: "Mode",
    extract: () => null,
    display: (d) => d.run.runMode,
    higherIsBetter: true,
  },
  {
    label: "Status",
    extract: () => null,
    display: (d) => d.run.status,
    higherIsBetter: true,
  },
  {
    label: "Initial Capital",
    extract: (d) => {
      const v = d.portfolio?.initialCash ?? d.run.backtestMetrics?.initialCapital ?? null;
      return v !== null ? parseFloat(String(v)) : null;
    },
    display: (d) => {
      const v = d.portfolio?.initialCash ?? d.run.backtestMetrics?.initialCapital ?? null;
      return v !== null ? `$${formatCurrency(String(v))}` : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Final Equity",
    extract: (d) => {
      const v = d.portfolio?.currentEquity ?? d.run.backtestMetrics?.finalEquity ?? null;
      return v !== null ? parseFloat(String(v)) : null;
    },
    display: (d) => {
      const v = d.portfolio?.currentEquity ?? d.run.backtestMetrics?.finalEquity ?? null;
      return v !== null ? `$${formatCurrency(String(v))}` : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Total Return %",
    extract: (d) =>
      d.run.backtestMetrics?.totalReturnPct ?? d.portfolio?.totalReturnPct ?? null,
    display: (d) => {
      const v = d.run.backtestMetrics?.totalReturnPct ?? d.portfolio?.totalReturnPct ?? null;
      if (v === null) return "—";
      return `${v >= 0 ? "+" : ""}${formatPct(v)}`;
    },
    higherIsBetter: true,
  },
  {
    label: "Max Drawdown %",
    extract: (d) =>
      d.run.backtestMetrics?.maxDrawdownPct ?? d.portfolio?.maxDrawdownPct ?? null,
    display: (d) => {
      const v = d.run.backtestMetrics?.maxDrawdownPct ?? d.portfolio?.maxDrawdownPct ?? null;
      return v !== null ? formatPct(v) : "—";
    },
    // Lower drawdown is better — negate the extract for comparison
    higherIsBetter: false,
  },
  {
    label: "Win Rate",
    extract: (d) =>
      d.run.backtestMetrics?.winRate ?? d.portfolio?.winRate ?? null,
    display: (d) => {
      const v = d.run.backtestMetrics?.winRate ?? d.portfolio?.winRate ?? null;
      return v !== null ? formatPct(v) : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Total Trades",
    extract: (d) =>
      d.run.backtestMetrics?.totalTrades ?? d.portfolio?.totalTrades ?? null,
    display: (d) => {
      const v = d.run.backtestMetrics?.totalTrades ?? d.portfolio?.totalTrades ?? null;
      return v !== null ? String(v) : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Sharpe Ratio",
    extract: (d) => pick(d.run.backtestMetrics, "sharpeRatio") as number | null,
    display: (d) => {
      const v = d.run.backtestMetrics?.sharpeRatio;
      return v != null ? v.toFixed(3) : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Sortino Ratio",
    extract: (d) => pick(d.run.backtestMetrics, "sortinoRatio") as number | null,
    display: (d) => {
      const v = d.run.backtestMetrics?.sortinoRatio;
      return v != null ? v.toFixed(3) : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Profit Factor",
    extract: (d) => pick(d.run.backtestMetrics, "profitFactor") as number | null,
    display: (d) => {
      const v = d.run.backtestMetrics?.profitFactor;
      return v != null ? v.toFixed(2) : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "CAGR",
    extract: (d) => pick(d.run.backtestMetrics, "cagr") as number | null,
    display: (d) => {
      const v = d.run.backtestMetrics?.cagr;
      return v != null ? `${v >= 0 ? "+" : ""}${formatPct(v)}` : "—";
    },
    higherIsBetter: true,
  },
  {
    label: "Duration (days)",
    extract: (d) => pick(d.run.backtestMetrics, "durationDays") as number | null,
    display: (d) => {
      const v = d.run.backtestMetrics?.durationDays;
      return v != null ? `${v}d` : "—";
    },
    higherIsBetter: true,
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute which column index holds the best value for a given metric row.
 * Returns -1 if no run has numeric data for this row.
 */
function bestIndex(row: MetricRow, items: RunData[]): number {
  const values = items.map((d) => row.extract(d));
  const numeric = values.filter((v): v is number => v !== null);
  if (numeric.length === 0) return -1;

  const target = row.higherIsBetter ? Math.max(...numeric) : Math.min(...numeric);
  return values.findIndex((v) => v === target);
}

/**
 * Merge equity curves from multiple runs into a single array for Recharts.
 * Each point is keyed by barIndex. Missing values are left undefined so
 * Recharts renders a gap rather than a zero.
 */
function mergeEquityCurves(
  items: RunData[],
): Record<string, number | string>[] {
  // Use barIndex as the common x-axis key.
  const maxBars = Math.max(
    ...items.map((d) => (d.equityPoints.length > 0 ? d.equityPoints.length : 0)),
  );

  if (maxBars === 0) return [];

  return Array.from({ length: maxBars }, (_, i) => {
    const point: Record<string, number | string> = { barIndex: i };
    items.forEach((d, runIdx) => {
      const ep = d.equityPoints[i];
      if (ep !== undefined) {
        point[`equity_${runIdx}`] = parseFloat(ep.equity);
      }
    });
    return point;
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-48 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
      <div className="h-64 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />
      <div className="h-80 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />
    </div>
  );
}

interface ComparisonTooltipPayload {
  dataKey: string;
  value: number;
  color: string;
}

interface ComparisonTooltipProps {
  active?: boolean;
  payload?: ComparisonTooltipPayload[];
  label?: number;
  items: RunData[];
}

function ComparisonTooltip({ active, payload, label, items }: ComparisonTooltipProps) {
  if (!active || !payload || payload.length === 0 || label === undefined) return null;

  return (
    <div
      style={{
        backgroundColor: "#0f172a",
        border: "1px solid #334155",
        borderRadius: "8px",
        padding: "8px 12px",
        fontSize: "12px",
        lineHeight: "1.8",
        minWidth: "160px",
      }}
    >
      <p style={{ color: "#94a3b8", marginBottom: "4px" }}>Bar {label}</p>
      {payload.map((entry) => {
        const runIdx = parseInt(entry.dataKey.replace("equity_", ""), 10);
        const runData = items[runIdx];
        const runLabel = runData
          ? `Run ${runData.run.id.slice(0, 8)}`
          : `Run ${runIdx + 1}`;
        return (
          <p key={entry.dataKey} style={{ color: entry.color, margin: 0 }}>
            {runLabel}: <strong>${entry.value.toFixed(2)}</strong>
          </p>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunComparePage() {
  return (
    <Suspense fallback={<LoadingSkeleton />}>
      <RunCompareInner />
    </Suspense>
  );
}

function RunCompareInner() {
  const searchParams = useSearchParams();
  const idsParam = searchParams.get("ids") ?? "";

  const runIds = idsParam
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 5); // Hard cap at 5

  const [items, setItems] = useState<RunData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [fatalError, setFatalError] = useState<string | null>(null);

  useEffect(() => {
    if (runIds.length < 2) {
      setFatalError("At least 2 run IDs are required for comparison.");
      setIsLoading(false);
      return;
    }

    async function fetchAll() {
      setIsLoading(true);
      setFatalError(null);

      const results = await Promise.all(
        runIds.map(async (id): Promise<RunData | null> => {
          const runRes = await fetchRun(id);
          if (!runRes.ok) {
            // Fatal — we can't show a column without the run record.
            return null;
          }

          const [portRes, curveRes] = await Promise.all([
            fetchPortfolio(id),
            fetchEquityCurve(id, 1000),
          ]);

          return {
            run: runRes.data,
            portfolio: portRes.ok ? portRes.data : null,
            equityPoints: curveRes.ok ? curveRes.data.points : [],
            error:
              !portRes.ok || !curveRes.ok
                ? "Some secondary data failed to load."
                : null,
          };
        }),
      );

      const failed = results.filter((r) => r === null);
      if (failed.length === results.length) {
        setFatalError("All requested runs failed to load.");
        setIsLoading(false);
        return;
      }

      setItems(results.filter((r): r is RunData => r !== null));
      setIsLoading(false);
    }

    void fetchAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsParam]);

  if (isLoading) return <LoadingSkeleton />;

  if (fatalError) {
    return (
      <div className="space-y-4">
        <Link href="/runs" className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline">
          Back to Runs
        </Link>
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {fatalError}
        </div>
      </div>
    );
  }

  const chartData = mergeEquityCurves(items);

  return (
    <div className="space-y-6">
      {/* Back link */}
      <Link href="/runs" className="text-sm text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 hover:underline">
        Back to Runs
      </Link>

      <Header
        title="Run Comparison"
        subtitle={`Comparing ${items.length} runs`}
      />

      {/* ------------------------------------------------------------------ */}
      {/* Overlaid equity curves                                              */}
      {/* ------------------------------------------------------------------ */}
      <div className="card">
        <h3 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-300">Equity Curves</h3>
        {chartData.length === 0 ? (
          <div className="flex h-48 items-center justify-center rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 text-sm text-slate-500">
            No equity data available
          </div>
        ) : (
          <div className="w-full rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 p-3">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart
                data={chartData}
                margin={{ top: 8, right: 8, bottom: 0, left: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis
                  dataKey="barIndex"
                  stroke="#475569"
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={60}
                  tickFormatter={(v: number) => `Bar ${v}`}
                />
                <YAxis
                  stroke="#475569"
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  width={62}
                  tickFormatter={(v: number) => {
                    if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
                    if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}k`;
                    return `$${v.toFixed(0)}`;
                  }}
                />
                <Tooltip
                  content={<ComparisonTooltip items={items} />}
                />
                <Legend
                  wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }}
                  formatter={(_value, entry) => {
                    const dataKey = (entry as { dataKey?: string }).dataKey ?? "";
                    const runIdx = parseInt(dataKey.replace("equity_", ""), 10);
                    const runData = items[runIdx];
                    return runData
                      ? `Run ${runData.run.id.slice(0, 8)} (${runData.run.config?.strategy_name ?? runData.run.runMode})`
                      : `Run ${runIdx + 1}`;
                  }}
                />
                {items.map((_, idx) => (
                  <Line
                    key={`equity_${idx}`}
                    type="monotone"
                    dataKey={`equity_${idx}`}
                    stroke={RUN_COLORS[idx % RUN_COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, stroke: "#0f172a", strokeWidth: 2 }}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Metrics comparison table                                            */}
      {/* ------------------------------------------------------------------ */}
      <div className="card overflow-x-auto">
        <h3 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-300">Metrics</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
              {/* Row label column */}
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500 w-40">
                Metric
              </th>
              {/* One column per run */}
              {items.map((d, idx) => (
                <th
                  key={d.run.id}
                  className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500"
                >
                  <div className="flex flex-col gap-1">
                    <span style={{ color: RUN_COLORS[idx % RUN_COLORS.length] }}>
                      Run {d.run.id.slice(0, 8)}
                    </span>
                    <RunStatusBadge status={d.run.status} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map((row) => {
              const best = bestIndex(row, items);
              return (
                <tr
                  key={row.label}
                  className="border-b border-slate-100 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/20"
                >
                  <td className="px-4 py-2.5 text-xs font-medium text-slate-500">
                    {row.label}
                  </td>
                  {items.map((d, idx) => {
                    const isBest = best === idx;
                    const displayStr = row.display(d);
                    // Text-only rows (Strategy, Mode, Status) never get highlighted
                    const isTextOnlyRow = row.extract(d) === null && displayStr !== "—";
                    return (
                      <td
                        key={d.run.id}
                        className={[
                          "px-4 py-2.5 font-mono text-xs tabular-nums",
                          isBest && !isTextOnlyRow
                            ? "font-semibold text-emerald-600 dark:text-emerald-400"
                            : "text-slate-700 dark:text-slate-300",
                        ].join(" ")}
                      >
                        {row.label === "Status" ? (
                          <RunStatusBadge status={d.run.status} />
                        ) : (
                          <span>{displayStr}</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Per-run secondary data warnings */}
      {items.some((d) => d.error !== null) && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400">
          Some secondary data (portfolio / equity curve) could not be loaded for one or more runs.
          Metrics that depend on that data will show "—".
        </div>
      )}
    </div>
  );
}
