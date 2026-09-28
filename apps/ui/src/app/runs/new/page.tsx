"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  fetchStrategies,
  fetchStrategySchema,
  createRun,
} from "@/lib/api";
import type { Strategy, JsonSchemaProperty, RunMode, RunCreateRequest } from "@/lib/types";
import { Header } from "@/components/layout/header";
import { useToast } from "@/components/ui/toast";
import { LiveBanner } from "@/components/live-banner";
import { LiveConfirmDialog } from "@/components/live-confirm-dialog";
import { ExitConfigErrorPanel } from "@/components/exit-config-error-panel";
import {
  describeConfigWarning,
  isPyramidingByDesignStrategy,
  strategyDefaultAllowPyramiding,
} from "@/lib/exit-config";

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
const COMMON_SYMBOLS = ["BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR"];
const ALL_MODES: readonly RunMode[] = ["backtest", "paper", "live"];

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
  const effectiveType: string | undefined =
    schema.type ?? schema.anyOf?.find((s) => s.type !== "null")?.type;

  if (effectiveType === "boolean") {
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

  // Enumerated string fields (e.g. bracket_mode: fixed | atr) render as a select.
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return (
      <div>
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
          {label}
          {schema.description && (
            <span className="ml-1 text-xs font-normal text-slate-500"> — {schema.description}</span>
          )}
        </label>
        <select
          value={String(value ?? schema.default ?? schema.enum[0])}
          onChange={(e) => onChange(e.target.value)}
          className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
        >
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
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
        type={effectiveType === "integer" || effectiveType === "number" ? "number" : "text"}
        value={String(value ?? schema.default ?? "")}
        min={schema.minimum}
        max={schema.maximum}
        step={effectiveType === "integer" ? 1 : "any"}
        onChange={(e) => {
          const raw = e.target.value;
          if (effectiveType === "integer") onChange(parseInt(raw, 10));
          else if (effectiveType === "number") onChange(parseFloat(raw));
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
  const { toast } = useToast();

  // Read pre-fill params supplied by the Duplicate Run button.
  const preStrategy = searchParams.get("strategy") ?? "";
  const preSymbols = searchParams.get("symbols") ?? "";
  const preTimeframe = searchParams.get("timeframe") ?? "";
  const preCapital = searchParams.get("initial_capital") ?? "";

  const [strategies, setStrategies] = useState<readonly Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>({});
  const [symbols, setSymbols] = useState<string[]>(
    preSymbols ? preSymbols.split(",").map((s) => s.trim()).filter(Boolean) : ["BTC/EUR"],
  );
  const [customSymbol, setCustomSymbol] = useState("");
  const [timeframe, setTimeframe] = useState(preTimeframe || "1h");
  const [mode, setMode] = useState<RunMode>("backtest");
  const [initialCapital, setInitialCapital] = useState(preCapital || "10000");
  const [backtestStart, setBacktestStart] = useState("2024-01-01T00:00");
  const [backtestEnd, setBacktestEnd] = useState("2024-12-31T23:59");
  const [enableLearning, setEnableLearning] = useState(false);
  const [autoApplyLearning, setAutoApplyLearning] = useState(false);
  // WP1.3a (SY-13a-08, CF-13a-1 item 4): the paper/backtest checkbox value.
  // Re-initialised to the strategy's own default whenever the strategy
  // changes (see the effect below) -- this is what gets sent verbatim for
  // paper/backtest; LIVE always forces/sends `false` regardless of this
  // state (see handleSubmit).
  const [allowPyramiding, setAllowPyramiding] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitErrorDetail, setSubmitErrorDetail] = useState<unknown>(undefined);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(true);
  // WP1.7a/1.7b (SY-10, S13): the live confirmation token is NEVER held in
  // page-level state — only inside <LiveConfirmDialog>'s own transient
  // state, cleared on close. This page only remembers WHETHER the dialog
  // that will collect it is open, plus the already-validated body to submit
  // once the operator types it.
  const [pendingLiveBody, setPendingLiveBody] = useState<RunCreateRequest | null>(null);

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
      if (prop.default !== undefined) {
        defaults[key] = prop.default;
        continue;
      }
      const isNullable =
        prop.nullable === true || (prop.anyOf?.some((s) => s.type === "null") ?? false);
      if (isNullable) {
        // WP1.3a (SY-13a-02, CF-13a-1 item 1): this used to send a literal
        // `0` for every nullable numeric field with no explicit default
        // (bracket_stop_loss_pct, bracket_take_profit_pct,
        // bracket_atr_sl_multiplier/tp_multiplier, trailing_stop_pct). The
        // backend now treats bare `0`/`""`/`null` as "unset" for those
        // fields, but rsi_mean_reversion/dca_rsi_hybrid/grid_trading's
        // `trailing_stop_pct` schema minimum is 0.005 — sending `0` was
        // fragile (G-1: it happened to coerce to "unset" only because the
        // pre-1.3a backend zero-normalised it too) and, more importantly,
        // no longer expresses intent. `null` is unambiguous: the operator
        // never touched this field.
        defaults[key] = null;
      } else if (prop.type === "integer" || prop.type === "number") {
        defaults[key] = 0;
      } else {
        defaults[key] = "";
      }
    }
    setStrategyParams(defaults);
  }

  // Modes the currently-selected strategy may run in. Missing field (older
  // API) → all three modes allowed (graceful degrade).
  const allowedModes: readonly RunMode[] = selectedStrategy?.allowedModes ?? ALL_MODES;

  // Auto-correct: if the selected strategy changes (or loads) and the current
  // mode is no longer permitted, fall back to "backtest" (always allowed).
  // Covers both the initial-load path and handleStrategyChange because both
  // funnel through setSelectedStrategy. Prevents a stuck-on-paper state when
  // switching to a demoted strategy.
  useEffect(() => {
    if (selectedStrategy && !allowedModes.includes(mode)) {
      setMode("backtest");
    }
    // mode intentionally omitted: we only react to strategy/allowed changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStrategy, allowedModes]);

  // WP1.3a (SY-13a-08, CF-13a-1 item 4): re-seed the paper/backtest checkbox
  // to the newly-selected strategy's own default (dca_rsi_hybrid/
  // grid_trading -> true, everything else -> false) every time the
  // strategy changes. Mirrors the mode auto-correct effect above.
  useEffect(() => {
    setAllowPyramiding(strategyDefaultAllowPyramiding(selectedStrategy));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStrategy]);

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

  // WP1.7a/SY-10 (S13): submits an already-validated body, optionally with a
  // live confirmation token sent ONLY as the `X-Live-Confirm-Token` header
  // (never in the body — see `createRun` in `@/lib/api`).
  async function submitCreateRun(body: RunCreateRequest, liveConfirmToken?: string) {
    setIsSubmitting(true);
    const result = await createRun(body, liveConfirmToken);

    if (result.ok) {
      setPendingLiveBody(null);
      // WP1.3a (CF-13a-1 item 3): surface `configWarnings[]` from the 201
      // body as toasts BEFORE navigating away -- <ToastProvider> lives in
      // the root layout so these survive the client-side route change.
      const warnings = result.data.configWarnings ?? [];
      if (warnings.length === 0) {
        toast("Run started successfully", "success");
      } else {
        toast("Run started with warnings", "warning");
        for (const w of warnings) {
          toast(
            describeConfigWarning(w),
            w.code === "no_downside_exit" ? "error" : "warning",
          );
        }
      }
      router.push(`/runs/${result.data.id}`);
    } else {
      // WP17b-S-10 (round 2): a failed live create used to leave
      // `submitError` set behind the still-open <LiveConfirmDialog>
      // overlay -- fully hidden from the operator, who saw nothing happen.
      // Closing the dialog here surfaces the error in the underlying form
      // (the operator can simply reopen it by clicking Start Run again).
      setPendingLiveBody(null);
      setSubmitError(result.error.message);
      setSubmitErrorDetail(result.error.detail);
      setIsSubmitting(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitErrorDetail(undefined);

    if (symbols.length === 0) {
      setSubmitError("Select at least one symbol.");
      return;
    }
    if (!selectedStrategy) {
      setSubmitError("Select a strategy.");
      return;
    }

    // Defense-in-depth: block disallowed mode before the network round-trip.
    // The backend returns 422 as the authority, but a client-side guard gives
    // an immediate, specific message.
    const allowed = selectedStrategy.allowedModes ?? ALL_MODES;
    if (!allowed.includes(mode)) {
      setSubmitError(
        `"${selectedStrategy.displayName}" is not available for ${mode} mode. ` +
          `Allowed: ${allowed.join(", ")}.`,
      );
      return;
    }

    // WP1.3a (SY-13a-08/09, CF-13a-1 item 4): LIVE always sends an explicit
    // `false` -- never editable in the form, and never omitted (so a live
    // dca_rsi_hybrid/grid_trading request never resolves to the strategy's
    // own `true` default and 422s with `live_pyramiding_forbidden`). Paper
    // and backtest send the checkbox's current value explicitly.
    const resolvedAllowPyramiding = mode === "live" ? false : allowPyramiding;

    const body = {
      strategyName: selectedStrategy.name,
      strategyParams,
      symbols,
      timeframe,
      mode,
      initialCapital,
      backtestStart: mode === "backtest" ? new Date(backtestStart).toISOString() : null,
      backtestEnd: mode === "backtest" ? new Date(backtestEnd).toISOString() : null,
      enableAdaptiveLearning: mode !== "backtest" ? enableLearning : undefined,
      autoApplyLearning: mode === "paper" && enableLearning ? autoApplyLearning : undefined,
      allowPyramiding: resolvedAllowPyramiding,
    };

    if (mode === "live") {
      // Defer submission until the typed <LiveConfirmDialog> token is
      // collected — the body itself never carries a confirmToken field.
      setPendingLiveBody(body);
      return;
    }

    void submitCreateRun(body);
  }

  const isPyramidingByDesign = isPyramidingByDesignStrategy(selectedStrategy?.name);

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
            {ALL_MODES.map((m) => {
              const isAllowed = allowedModes.includes(m);
              return (
                <label
                  key={m}
                  aria-disabled={!isAllowed}
                  title={!isAllowed ? "This strategy is restricted to backtest only." : undefined}
                  className={[
                    "flex flex-1 items-center justify-center rounded-lg border py-3 text-sm font-medium transition-colors",
                    !isAllowed
                      ? "cursor-not-allowed border-slate-200 text-slate-400 opacity-50 dark:border-slate-800 dark:text-slate-600"
                      : mode === m
                      ? "cursor-pointer border-indigo-500 bg-indigo-600/20 text-indigo-600 dark:text-indigo-400"
                      : "cursor-pointer border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-600 hover:text-slate-700 dark:hover:text-slate-300",
                  ].join(" ")}
                >
                  <input
                    type="radio"
                    name="mode"
                    value={m}
                    checked={mode === m}
                    disabled={!isAllowed}
                    onChange={() => {
                      if (isAllowed) setMode(m);
                    }}
                    className="sr-only"
                  />
                  <span className="capitalize">{m}</span>
                </label>
              );
            })}
          </div>
          <p className="text-xs text-slate-500">
            {mode === "backtest"
              ? "Backtest runs against historical data synchronously."
              : mode === "paper"
              ? "Paper trading simulates live execution without real funds."
              : "Live trading executes real orders on your exchange account."}
          </p>
          {mode === "live" && (
            <div className="space-y-2">
              <LiveBanner compact />
              <p className="text-xs text-red-400">
                This will place real orders. Ensure ENABLE_LIVE_TRADING=true
                and your API key are set in .env. Clicking Start Run will ask
                for the live-trading confirmation token separately — it is
                never typed into this form.
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

          {selectedStrategy?.status === "demoted" && (
            <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400">
              {selectedStrategy.demotionReason
                ? `${selectedStrategy.demotionReason} `
                : "This strategy has been demoted. "}
              Available for backtest only.
            </div>
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

        {/* WP1.3a (CF-13a-1 item 4): allow-pyramiding control */}
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Pyramiding</h2>
          {mode === "live" ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between opacity-60">
                <label className="text-sm text-slate-700 dark:text-slate-300">
                  Allow Pyramiding
                </label>
                <input
                  type="checkbox"
                  checked={false}
                  disabled
                  aria-label="Allow Pyramiding (disabled in live mode)"
                  className="h-4 w-4 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-700 dark:bg-slate-800"
                />
              </div>
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Live pyramiding is disabled until WP1.3b/WP1.10.
              </p>
              {isPyramidingByDesign && selectedStrategy && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  Accumulation disabled: single-entry variant, not validated.
                  {" "}
                  {selectedStrategy.displayName} normally accumulates
                  (pyramids) into a held position — running it live single-entry
                  is unvalidated but is allowed because it has strictly less
                  exposure than its designed behaviour.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="allow-pyramiding-checkbox"
                  className="text-sm text-slate-700 dark:text-slate-300"
                >
                  Allow Pyramiding
                </label>
                <input
                  id="allow-pyramiding-checkbox"
                  type="checkbox"
                  checked={allowPyramiding}
                  onChange={(e) => setAllowPyramiding(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-700 dark:bg-slate-800"
                />
              </div>
              <p className="text-xs text-slate-500">
                When on, the strategy may submit a new BUY into an already-held
                position (repeat entries). Defaults to{" "}
                {isPyramidingByDesign ? "ON" : "OFF"} for this strategy.
              </p>
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

        {/* Adaptive Learning (paper/live only) */}
        {mode !== "backtest" && (
          <div className="card space-y-3">
            <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Adaptive Learning</h2>
            <p className="text-xs text-slate-500">
              When enabled, the bot analyzes its own trades and proposes parameter improvements automatically.
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm text-slate-700 dark:text-slate-300">
                    Enable Adaptive Learning
                  </label>
                  <p className="text-xs text-slate-500">
                    Runs PerformanceAnalyzer + AdaptiveOptimizer every 50 trades
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={enableLearning}
                  onChange={(e) => {
                    setEnableLearning(e.target.checked);
                    if (!e.target.checked) setAutoApplyLearning(false);
                  }}
                  className="h-4 w-4 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-700 dark:bg-slate-800"
                />
              </div>
              {enableLearning && mode === "paper" && (
                <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-900/40">
                  <div>
                    <label className="text-sm text-slate-700 dark:text-slate-300">
                      Auto-Apply Changes
                    </label>
                    <p className="text-xs text-slate-500">
                      Automatically apply parameter adjustments (max 20% change per cycle, rollback at -5% PnL)
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={autoApplyLearning}
                    onChange={(e) => setAutoApplyLearning(e.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-700 dark:bg-slate-800"
                  />
                </div>
              )}
              {enableLearning && mode === "live" && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  In live mode, learning runs in observation-only mode. Parameter changes are logged but never auto-applied.
                </p>
              )}
            </div>
          </div>
        )}

        {submitError && (
          <ExitConfigErrorPanel detail={submitErrorDetail} fallbackMessage={submitError} />
        )}

        <button
          type="submit"
          disabled={isSubmitting || isLoadingStrategies}
          className="w-full rounded-lg bg-indigo-600 py-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          {isSubmitting ? "Starting run…" : mode === "live" ? "Start Run…" : "Start Run"}
        </button>
      </form>

      <LiveConfirmDialog
        open={pendingLiveBody !== null}
        title="Confirm live run"
        description="This starts a LIVE run that places real orders. Type the live-trading confirmation token to proceed."
        confirmLabel="Start Live Run"
        loading={isSubmitting}
        onCancel={() => setPendingLiveBody(null)}
        onConfirm={(token) => {
          if (pendingLiveBody) void submitCreateRun(pendingLiveBody, token);
        }}
      />
    </div>
  );
}
