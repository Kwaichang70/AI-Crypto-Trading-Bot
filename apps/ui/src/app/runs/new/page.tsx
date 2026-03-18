"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  fetchStrategies,
  fetchStrategySchema,
  createRun,
} from "@/lib/api";
import type { Strategy, JsonSchemaProperty, RunMode } from "@/lib/types";
import { Header } from "@/components/layout/header";

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
const COMMON_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD"];

function ParamInput({
  name,
  schema,
  value,
  onChange,
}: {
  name: string;
  schema: JsonSchemaProperty;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  if (schema.type === "boolean") {
    return (
      <div className="flex items-center justify-between">
        <label className="text-sm text-slate-700 dark:text-slate-300">
          {label}
          {schema.description && (
            <span className="ml-1 text-xs text-slate-500"> — {schema.description}</span>
          )}
        </label>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-700 dark:bg-slate-800"
        />
      </div>
    );
  }

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
        {label}
        {schema.description && (
          <span className="ml-1 text-xs font-normal text-slate-500"> — {schema.description}</span>
        )}
      </label>
      <input
        type={schema.type === "integer" || schema.type === "number" ? "number" : "text"}
        value={String(value ?? schema.default ?? "")}
        min={schema.minimum}
        max={schema.maximum}
        step={schema.type === "integer" ? 1 : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          if (schema.type === "integer") onChange(parseInt(raw, 10));
          else if (schema.type === "number") onChange(parseFloat(raw));
          else onChange(raw);
        }}
        className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-500"
      />
    </div>
  );
}

export default function NewRunPage() {
  return (
    <Suspense fallback={<div className="h-64 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-800" />}>
      <NewRunInner />
    </Suspense>
  );
}

function NewRunInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Read pre-fill params supplied by the Duplicate Run button.
  const preStrategy = searchParams.get("strategy") ?? "";
  const preSymbols = searchParams.get("symbols") ?? "";
  const preTimeframe = searchParams.get("timeframe") ?? "";
  const preCapital = searchParams.get("initial_capital") ?? "";

  const [strategies, setStrategies] = useState<readonly Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>({});
  const [symbols, setSymbols] = useState<string[]>(
    preSymbols ? preSymbols.split(",").map((s) => s.trim()).filter(Boolean) : ["BTC/USD"],
  );
  const [customSymbol, setCustomSymbol] = useState("");
  const [timeframe, setTimeframe] = useState(preTimeframe || "1h");
  const [mode, setMode] = useState<RunMode>("backtest");
  const [initialCapital, setInitialCapital] = useState(preCapital || "10000");
  const [backtestStart, setBacktestStart] = useState("2024-01-01T00:00");
  const [backtestEnd, setBacktestEnd] = useState("2024-12-31T23:59");
  const [confirmToken, setConfirmToken] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(true);

  useEffect(() => {
    async function load() {
      const result = await fetchStrategies();
      if (result.ok && result.data.strategies.length > 0) {
        setStrategies(result.data.strategies);

        // If duplicating a run, try to match the pre-filled strategy name;
        // otherwise fall back to the first strategy in the list.
        const matchedByName = preStrategy
          ? result.data.strategies.find((s) => s.name === preStrategy) ?? null
          : null;
        const targetStrategy = matchedByName ?? result.data.strategies[0];

        if (matchedByName) {
          // Fetch full schema (includes parameterSchema) for the matched strategy.
          const schemaResult = await fetchStrategySchema(matchedByName.name);
          if (schemaResult.ok) {
            setSelectedStrategy(schemaResult.data);
            initDefaults(schemaResult.data);
          } else {
            setSelectedStrategy(targetStrategy);
            initDefaults(targetStrategy);
          }
        } else {
          setSelectedStrategy(targetStrategy);
          initDefaults(targetStrategy);
        }
      }
      setIsLoadingStrategies(false);
    }
    void load();
    // preStrategy intentionally read once on mount — stale-closure is acceptable here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function initDefaults(strategy: Strategy) {
    const defaults: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(strategy.parameterSchema.properties)) {
      defaults[key] = prop.default ?? (prop.type === "integer" || prop.type === "number" ? 0 : "");
    }
    setStrategyParams(defaults);
  }

  async function handleStrategyChange(name: string) {
    const result = await fetchStrategySchema(name);
    if (result.ok) {
      setSelectedStrategy(result.data);
      initDefaults(result.data);
    }
  }

  function toggleSymbol(sym: string) {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
  }

  function addCustomSymbol() {
    const s = customSymbol.trim().toUpperCase();
    if (s && s.includes("/") && !symbols.includes(s)) {
      setSymbols((prev) => [...prev, s]);
      setCustomSymbol("");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);

    if (symbols.length === 0) {
      setSubmitError("Select at least one symbol.");
      return;
    }
    if (!selectedStrategy) {
      setSubmitError("Select a strategy.");
      return;
    }

    setIsSubmitting(true);

    const body = {
      strategyName: selectedStrategy.name,
      strategyParams,
      symbols,
      timeframe,
      mode,
      initialCapital,
      backtestStart: mode === "backtest" ? new Date(backtestStart).toISOString() : null,
      backtestEnd: mode === "backtest" ? new Date(backtestEnd).toISOString() : null,
      confirmToken: mode === "live" ? confirmToken : undefined,
    };

    const result = await createRun(body);

    if (result.ok) {
      router.push(`/runs/${result.data.id}`);
    } else {
      setSubmitError(result.error.message);
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <Header
        title="New Run"
        subtitle="Configure and launch a new backtest, paper, or live trading run."
      />

      {preStrategy && (
        <div className="rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2 text-xs text-indigo-600 dark:border-indigo-800 dark:bg-indigo-900/20 dark:text-indigo-400">
          Pre-filled from a previous run. Adjust settings as needed and click Start Run.
        </div>
      )}

      <form onSubmit={(e) => void handleSubmit(e)} className="space-y-6 lg:max-w-2xl">
        {/* Mode selector */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Run Mode</h2>
          <div className="flex gap-3">
            {(["backtest", "paper", "live"] as const).map((m) => (
              <label
                key={m}
                className={[
                  "flex flex-1 cursor-pointer items-center justify-center rounded-lg border py-3 text-sm font-medium transition-colors",
                  mode === m
                    ? "border-indigo-500 bg-indigo-600/20 text-indigo-600 dark:text-indigo-400"
                    : "border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-600 hover:text-slate-700 dark:hover:text-slate-300",
                ].join(" ")}
              >
                <input
                  type="radio"
                  name="mode"
                  value={m}
                  checked={mode === m}
                  onChange={() => setMode(m)}
                  className="sr-only"
                />
                <span className="capitalize">{m}</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-slate-500">
            {mode === "backtest"
              ? "Backtest runs against historical data synchronously."
              : mode === "paper"
              ? "Paper trading simulates live execution without real funds."
              : "Live trading executes real orders on your exchange account."}
          </p>
          {mode === "live" && (
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                Confirm Token
              </label>
              <input
                type="password"
                value={confirmToken}
                onChange={(e) => setConfirmToken(e.target.value)}
                placeholder="LIVE_TRADING_CONFIRM_TOKEN from .env"
                className="mt-1 w-full rounded-lg border border-red-300 dark:border-red-800 bg-white dark:bg-slate-800 px-3 py-2 text-sm text-slate-900 dark:text-slate-200 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
              />
              <p className="mt-1 text-xs text-red-400">
                This will place real orders. Ensure ENABLE_LIVE_TRADING=true and your API key are set in .env.
              </p>
            </div>
          )}
        </div>

        {/* Strategy selection */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Strategy</h2>
          {isLoadingStrategies ? (
            <div className="h-10 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
          ) : (
            <select
              value={selectedStrategy?.name ?? ""}
              onChange={(e) => void handleStrategyChange(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            >
              {strategies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.displayName} v{s.version}
                </option>
              ))}
            </select>
          )}

          {selectedStrategy && (
            <p className="text-xs text-slate-500">{selectedStrategy.description}</p>
          )}

          {/* Dynamic strategy parameters */}
          {selectedStrategy &&
            Object.keys(selectedStrategy.parameterSchema.properties).length > 0 && (
              <div className="space-y-3 border-t border-slate-200 dark:border-slate-800 pt-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Parameters
                </p>
                {Object.entries(selectedStrategy.parameterSchema.properties).map(
                  ([key, prop]) => (
                    <ParamInput
                      key={key}
                      name={key}
                      schema={prop}
                      value={strategyParams[key]}
                      onChange={(v) =>
                        setStrategyParams((prev) => ({ ...prev, [key]: v }))
                      }
                    />
                  ),
                )}
              </div>
            )}
        </div>

        {/* Symbols */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Symbols</h2>
          <div className="flex flex-wrap gap-2">
            {COMMON_SYMBOLS.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => toggleSymbol(sym)}
                className={[
                  "rounded-full px-3 py-1 font-mono text-xs font-medium transition-colors",
                  symbols.includes(sym)
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200",
                ].join(" ")}
              >
                {sym}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Custom: ETH/BTC"
              value={customSymbol}
              onChange={(e) => setCustomSymbol(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addCustomSymbol();
                }
              }}
              className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-500"
            />
            <button
              type="button"
              onClick={addCustomSymbol}
              className="rounded-lg border border-slate-300 dark:border-slate-700 px-3 py-2 text-sm text-slate-500 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-600 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
            >
              Add
            </button>
          </div>
          {symbols.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {symbols.map((s) => (
                <span
                  key={s}
                  className="flex items-center gap-1 rounded bg-slate-100 dark:bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-700 dark:text-slate-300"
                >
                  {s}
                  <button
                    type="button"
                    onClick={() => setSymbols((prev) => prev.filter((x) => x !== s))}
                    className="text-slate-500 hover:text-slate-200"
                    aria-label={`Remove ${s}`}
                  >
                    x
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Timeframe */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Timeframe</h2>
          <div className="flex flex-wrap gap-2">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                type="button"
                onClick={() => setTimeframe(tf)}
                className={[
                  "rounded-full px-3 py-1 font-mono text-xs font-medium transition-colors",
                  timeframe === tf
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700 dark:hover:text-slate-200",
                ].join(" ")}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        {/* Capital + backtest dates */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Capital & Dates</h2>
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              Initial Capital (USD)
            </label>
            <input
              type="number"
              min="1"
              step="1"
              value={initialCapital}
              onChange={(e) => setInitialCapital(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            />
          </div>

          {mode === "backtest" && (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                  Backtest Start
                </label>
                <input
                  type="datetime-local"
                  value={backtestStart}
                  onChange={(e) => setBacktestStart(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                  Backtest End
                </label>
                <input
                  type="datetime-local"
                  value={backtestEnd}
                  onChange={(e) => setBacktestEnd(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                />
              </div>
            </div>
          )}
        </div>

        {submitError && (
          <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
            {submitError}
          </div>
        )}

        <button
          type="submit"
          disabled={isSubmitting || isLoadingStrategies}
          className="w-full rounded-lg bg-indigo-600 py-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          {isSubmitting ? "Starting run…" : "Start Run"}
        </button>
      </form>
    </div>
  );
}
