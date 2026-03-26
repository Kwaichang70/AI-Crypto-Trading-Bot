"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  fetchStrategies,
  fetchStrategySchema,
  runOptimization,
  createRun,
  fetchOptimizationRuns,
} from "@/lib/api";
import type {
  Strategy,
  JsonSchemaProperty,
  OptimizeResponse,
  OptimizeEntry,
  OptimizationRunSummary,
} from "@/lib/types";
import { Header } from "@/components/layout/header";
import { DataTable } from "@/components/ui/data-table";
import { ParamGridEditor } from "./param-grid-editor";
import type { ParamGridRow } from "./param-grid-editor";
import { buildResultColumns } from "./result-columns";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
const COMMON_SYMBOLS = ["BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR"];

const RANK_OPTIONS: { value: string; label: string }[] = [
  { value: "sharpe_ratio", label: "Sharpe Ratio" },
  { value: "sortino_ratio", label: "Sortino Ratio" },
  { value: "calmar_ratio", label: "Calmar Ratio" },
  { value: "total_return_pct", label: "Total Return %" },
  { value: "cagr", label: "CAGR" },
  { value: "profit_factor", label: "Profit Factor" },
  { value: "win_rate", label: "Win Rate" },
  { value: "max_drawdown_pct", label: "Max Drawdown % (lower = better)" },
];

// ---------------------------------------------------------------------------
// Page phase discriminated union
// ---------------------------------------------------------------------------

type PagePhase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "results"; data: OptimizeResponse }
  | { kind: "error"; message: string };

// ---------------------------------------------------------------------------
// Optimization history component
// ---------------------------------------------------------------------------

function OptimizationHistory() {
  const [runs, setRuns] = useState<readonly OptimizationRunSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsLoading(true);
    fetchOptimizationRuns({ limit: 10 }).then((r) => {
      if (r.ok) {
        // API returns plain array or { items } envelope — handle both
        const data = r.data as any;
        setRuns(Array.isArray(data) ? data : (data.items ?? []));
      } else {
        setError(r.error.message);
      }
      setIsLoading(false);
    });
  }, []);

  if (isLoading) {
    return (
      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-300">Recent Optimization Runs</h2>
        <div className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-10 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-300">Recent Optimization Runs</h2>
        <p className="text-sm text-slate-500">{error}</p>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="card p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-300">Recent Optimization Runs</h2>
        <p className="text-sm text-slate-500">No saved optimization runs yet. Run your first grid search above.</p>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Recent Optimization Runs</h2>
      </div>
      <div className="divide-y divide-slate-200 dark:divide-slate-800">
        {runs.map((run) => (
          <div
            key={run.optimizationRunId}
            className="flex items-center justify-between px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800/40"
          >
            <div className="min-w-0 flex-1 space-y-0.5">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                  {run.strategyName}
                </span>
                <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-600 dark:bg-slate-700 dark:text-slate-400">
                  {run.timeframe}
                </span>
                <span className="text-xs text-slate-500">
                  ranked by <span className="text-slate-400">{run.rankBy}</span>
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-500">
                <span>{run.symbols.join(", ")}</span>
                <span>·</span>
                <span>
                  {run.completedCombinations}/{run.totalCombinations} combinations
                </span>
                {run.failedCombinations > 0 && (
                  <>
                    <span>·</span>
                    <span className="text-amber-500">{run.failedCombinations} failed</span>
                  </>
                )}
                <span>·</span>
                <span>{run.elapsedSeconds}s</span>
                <span>·</span>
                <span>{new Date(run.createdAt).toLocaleDateString()}</span>
              </div>
            </div>
            <Link
              href={`/optimize/${run.optimizationRunId}`}
              className="ml-4 shrink-0 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors hover:border-indigo-500 hover:text-indigo-400"
            >
              View
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function OptimizePage() {
  const router = useRouter();

  // --- Strategies ---
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [paramTypes, setParamTypes] = useState<Record<string, JsonSchemaProperty>>({});

  // --- Form ---
  const [paramGridRows, setParamGridRows] = useState<ParamGridRow[]>([]);
  const [symbols, setSymbols] = useState<string[]>(["BTC/EUR"]);
  const [customSymbol, setCustomSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("1h");
  const [backtestStart, setBacktestStart] = useState("2024-01-01T00:00");
  const [backtestEnd, setBacktestEnd] = useState("2024-12-31T23:59");
  const [initialCapital, setInitialCapital] = useState("10000");
  const [rankBy, setRankBy] = useState("sharpe_ratio");
  const [topN, setTopN] = useState(10);
  const [maxCombinations, setMaxCombinations] = useState(500);

  // --- Results / phase ---
  const [phase, setPhase] = useState<PagePhase>({ kind: "idle" });
  const [launchingRank, setLaunchingRank] = useState<number | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [actualCombinations, setActualCombinations] = useState(0);

  // Load strategies on mount
  useEffect(() => {
    fetchStrategies().then((r) => {
      if (r.ok) setStrategies(r.data.strategies as Strategy[]);
    });
  }, []);

  // Load schema when strategy changes
  useEffect(() => {
    if (!selectedStrategy) return;
    fetchStrategySchema(selectedStrategy.name).then((r) => {
      if (r.ok && r.data.parameterSchema?.properties) {
        setParamTypes(r.data.parameterSchema.properties as Record<string, JsonSchemaProperty>);
      } else {
        setParamTypes({});
      }
      setParamGridRows([]);
    });
  }, [selectedStrategy]);

  function toggleSymbol(sym: string) {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
  }

  function addCustomSymbol() {
    const s = customSymbol.trim().toUpperCase();
    if (s && !symbols.includes(s)) {
      setSymbols((prev) => [...prev, s]);
    }
    setCustomSymbol("");
  }

  /** CR-001: normalize datetime-local string to UTC ISO without relying on browser Date parsing. */
  function toUtcIso(local: string): string | null {
    const normalized =
      local.includes("Z") || local.includes("+") ? local : local + ":00Z";
    const ms = Date.parse(normalized);
    return isNaN(ms) ? null : new Date(ms).toISOString();
  }

  /** CR-005: parse grid rows; returns null if any numeric value is NaN. */
  function parseGridRows(): Record<string, unknown[]> | string | null {
    const grid: Record<string, unknown[]> = {};
    for (const row of paramGridRows) {
      if (!row.paramName || !row.valuesRaw.trim()) continue;
      const schema = paramTypes[row.paramName];
      const type = schema?.type ?? "string";
      const values = row.valuesRaw
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean)
        .map((v) => {
          if (type === "integer") return parseInt(v, 10);
          if (type === "number") return parseFloat(v);
          return v;
        });
      const hasNaN = values.some((v) => typeof v === "number" && isNaN(v as number));
      if (hasNaN) return `Invalid value in parameter "${row.paramName}" — expected ${type === "integer" ? "integers" : "numbers"}.`;
      if (values.length > 0) grid[row.paramName] = values;
    }
    return Object.keys(grid).length > 0 ? grid : null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!selectedStrategy) return;
    if (symbols.length === 0) {
      setPhase({ kind: "error", message: "Select at least one symbol." });
      return;
    }

    // CR-002: guard NaN from cleared number inputs
    if (isNaN(topN) || topN < 1) {
      setPhase({ kind: "error", message: "Top N must be a valid number >= 1." });
      return;
    }
    if (isNaN(maxCombinations) || maxCombinations < 1) {
      setPhase({ kind: "error", message: "Max Combinations must be a valid number >= 1." });
      return;
    }

    // CR-001: normalize datetime-local strings to UTC ISO
    const startIso = toUtcIso(backtestStart);
    const endIso = toUtcIso(backtestEnd);
    if (!startIso || !endIso) {
      setPhase({ kind: "error", message: "Invalid date — please check Start Date and End Date." });
      return;
    }

    const gridResult = parseGridRows();
    if (gridResult === null) {
      setPhase({ kind: "error", message: "Add at least one parameter with values to the grid." });
      return;
    }
    if (typeof gridResult === "string") {
      setPhase({ kind: "error", message: gridResult });
      return;
    }

    // CR-007: track actual combination count for time estimate
    const actualCombinations = Object.values(gridResult).reduce(
      (acc, vals) => acc * vals.length,
      1,
    );

    setPhase({ kind: "loading" });
    setLaunchError(null);
    setActualCombinations(actualCombinations);

    const result = await runOptimization({
      strategyName: selectedStrategy.name,
      paramGrid: gridResult,
      symbols: [...symbols],
      timeframe,
      backtestStart: startIso,
      backtestEnd: endIso,
      initialCapital,
      rankBy,
      topN,
      maxCombinations,
    });

    if (result.ok) {
      setPhase({ kind: "results", data: result.data });
    } else {
      setPhase({ kind: "error", message: result.error.message });
    }
  }

  async function handleLaunchRun(entry: OptimizeEntry) {
    if (phase.kind !== "results") return;
    setLaunchingRank(entry.rank);
    setLaunchError(null);

    const startIso = toUtcIso(backtestStart);
    const endIso = toUtcIso(backtestEnd);

    const result = await createRun({
      strategyName: selectedStrategy?.name ?? phase.data.strategyName,
      strategyParams: entry.params,
      symbols: [...phase.data.symbols],
      timeframe: phase.data.timeframe,
      mode: "backtest",
      initialCapital,
      backtestStart: startIso,
      backtestEnd: endIso,
    });

    setLaunchingRank(null);
    if (result.ok) {
      router.push(`/runs/${result.data.id}`);
    } else {
      setLaunchError(`Failed to launch run: ${result.error.message}`);
    }
  }

  const paramNames = Object.keys(paramTypes);
  const isLoading = phase.kind === "loading";

  // CR-007: use actual grid size for time estimate, not the cap
  const estimatedSeconds =
    phase.kind === "loading"
      ? Math.round((actualCombinations * 0.3) / 5) * 5
      : 0;

  return (
    <div className="flex-1 space-y-6 p-6">
      <Header
        title="Optimize"
        subtitle="Grid search over strategy parameters"
        actions={
          phase.kind === "results" ? (
            <button
              onClick={() => setPhase({ kind: "idle" })}
              className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-400 hover:border-slate-600 hover:text-slate-200"
            >
              New Search
            </button>
          ) : undefined
        }
      />

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Strategy */}
        <div className="card space-y-3 p-4">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Strategy</h2>
          {strategies.length === 0 ? (
            <p className="text-sm text-slate-500">Loading strategies…</p>
          ) : (
            <select
              value={selectedStrategy?.name ?? ""}
              onChange={(e) => {
                const s = strategies.find((st) => st.name === e.target.value) ?? null;
                setSelectedStrategy(s);
              }}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">Select a strategy…</option>
              {strategies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Parameter grid */}
        {selectedStrategy && (
          <div className="card space-y-3 p-4">
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Parameter Grid</h2>
            <p className="text-xs text-slate-500">
              For each parameter, enter a comma-separated list of values to test.
            </p>
            <ParamGridEditor
              paramNames={paramNames}
              paramTypes={paramTypes}
              rows={paramGridRows}
              onChange={setParamGridRows}
            />
          </div>
        )}

        {/* Symbols */}
        <div className="card space-y-3 p-4">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Symbols</h2>
          <div className="flex flex-wrap gap-2">
            {COMMON_SYMBOLS.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => toggleSymbol(sym)}
                className={[
                  "rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors",
                  symbols.includes(sym)
                    ? "border-indigo-500 bg-indigo-600/20 text-indigo-600 dark:text-indigo-300"
                    : "border-slate-300 text-slate-600 hover:border-slate-400 hover:text-slate-900 dark:border-slate-700 dark:text-slate-400 dark:hover:border-slate-600 dark:hover:text-slate-200",
                ].join(" ")}
              >
                {sym}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={customSymbol}
              onChange={(e) => setCustomSymbol(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addCustomSymbol(); } }}
              placeholder="Custom symbol, e.g. LTC/USD"
              className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-500"
            />
            <button
              type="button"
              onClick={addCustomSymbol}
              className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:border-indigo-500 hover:text-indigo-400"
            >
              Add
            </button>
          </div>
          {symbols.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {symbols.map((s) => (
                <span
                  key={s}
                  className="flex items-center gap-1 rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700 dark:bg-slate-700 dark:text-slate-300"
                >
                  {s}
                  <button
                    type="button"
                    onClick={() => setSymbols((prev) => prev.filter((x) => x !== s))}
                    className="text-slate-500 hover:text-red-400"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Backtest config */}
        <div className="card space-y-3 p-4">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Backtest Configuration</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Timeframe</label>
              <div className="mt-1 flex flex-wrap gap-2">
                {TIMEFRAMES.map((tf) => (
                  <button
                    key={tf}
                    type="button"
                    onClick={() => setTimeframe(tf)}
                    className={[
                      "rounded-lg border px-3 py-1 text-sm font-medium",
                      timeframe === tf
                        ? "border-indigo-500 bg-indigo-600/20 text-indigo-600 dark:text-indigo-300"
                        : "border-slate-700 text-slate-400 hover:border-slate-600",
                    ].join(" ")}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Initial Capital</label>
              <input
                type="number"
                value={initialCapital}
                min="100"
                step="100"
                onChange={(e) => setInitialCapital(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Start Date</label>
              <input
                type="datetime-local"
                value={backtestStart}
                onChange={(e) => setBacktestStart(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">End Date</label>
              <input
                type="datetime-local"
                value={backtestEnd}
                onChange={(e) => setBacktestEnd(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              />
            </div>
          </div>
        </div>

        {/* Optimizer settings */}
        <div className="card space-y-3 p-4">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Optimizer Settings</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Rank By</label>
              <select
                value={rankBy}
                onChange={(e) => setRankBy(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              >
                {RANK_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Top N Results</label>
              <input
                type="number"
                value={topN}
                min={1}
                max={100}
                onChange={(e) => setTopN(parseInt(e.target.value, 10))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-600 dark:text-slate-400">Max Combinations</label>
              <input
                type="number"
                value={maxCombinations}
                min={1}
                max={1000}
                onChange={(e) => setMaxCombinations(parseInt(e.target.value, 10))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
              />
            </div>
          </div>
        </div>

        {/* Error banner */}
        {phase.kind === "error" && (
          <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-600 dark:border-red-700/50 dark:bg-red-900/20 dark:text-red-400">
            {phase.message}
          </div>
        )}

        {/* Submit */}
        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={isLoading || !selectedStrategy || symbols.length === 0}
            className="rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? "Running optimization…" : "Run Optimization"}
          </button>
          {isLoading && (
            <p className="text-xs text-slate-500">
              Running up to {maxCombinations} combinations — may take ~{estimatedSeconds}s
            </p>
          )}
        </div>
      </form>

      {/* Results */}
      {phase.kind === "results" && (
        <div className="space-y-4">
          {/* Summary bar */}
          <div className="card p-4">
            <div className="flex flex-wrap items-center gap-4 text-sm">
              <span className="text-slate-400">
                Strategy: <span className="text-slate-800 dark:text-slate-200">{phase.data.strategyName}</span>
              </span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-400">
                Ranked by: <span className="text-indigo-300">{phase.data.rankBy}</span>
              </span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-400">
                Combinations:{" "}
                <span className="text-slate-200">
                  {phase.data.completedCombinations}/{phase.data.totalCombinations}
                </span>
              </span>
              {phase.data.failedCombinations > 0 && (
                <>
                  <span className="text-slate-600">·</span>
                  <span className="text-amber-400">
                    {phase.data.failedCombinations} failed
                  </span>
                </>
              )}
              <span className="text-slate-600">·</span>
              <span className="text-slate-400">
                Elapsed: <span className="text-slate-800 dark:text-slate-200">{phase.data.elapsedSeconds}s</span>
              </span>
              {/* Link to saved detail page — only shown when the backend has persisted the run */}
              {phase.data.optimizationRunId && (
                <>
                  <span className="text-slate-600">·</span>
                  <span className="text-slate-500">Saved</span>
                  <Link
                    href={`/optimize/${phase.data.optimizationRunId}`}
                    className="text-indigo-400 hover:underline"
                  >
                    View
                  </Link>
                </>
              )}
            </div>
          </div>

          {phase.data.entries.length === 0 ? (
            <div className="card p-6 text-center text-sm text-slate-500">
              No results — all combinations failed. Check your parameter grid and date range.
            </div>
          ) : (
            <>
              {launchError && (
                <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-600 dark:border-red-700/50 dark:bg-red-900/20 dark:text-red-400">
                  {launchError}
                </div>
              )}
              <div className="card overflow-hidden">
                <DataTable
                  columns={buildResultColumns(phase.data.rankBy, launchingRank, handleLaunchRun)}
                  data={[...phase.data.entries]}
                  keyExtractor={(e) => String(e.rank)}
                  emptyMessage="No results"
                />
              </div>
            </>
          )}
        </div>
      )}

      {/* Optimization history — always visible, fetches saved runs on mount */}
      <OptimizationHistory />
    </div>
  );
}
