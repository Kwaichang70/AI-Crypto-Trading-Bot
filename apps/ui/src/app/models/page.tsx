/**
 * apps/ui/src/app/models/page.tsx
 * --------------------------------
 * ML Model Management page — "use client" because filter/retrain state is
 * interactive.
 *
 * Shows all trained ModelVersion records returned by GET /api/v1/ml/models.
 * Supports:
 *  - Symbol text filter and "Active only" checkbox (client-driven query params).
 *  - "Retrain" button that POSTs to /api/v1/ml/retrain/{symbol} and refreshes.
 *  - Per-row "Activate" button that PUTs to /api/v1/ml/models/{id}/activate.
 */
"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchModelVersions, retrainModel, trainModel, activateModelVersion } from "@/lib/api";
import type { ModelVersion } from "@/lib/types";
import { DataTable, type Column } from "@/components/ui/data-table";
import { Header } from "@/components/layout/header";

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------

function MethodBadge({ method }: { method: string }) {
  const isHorizon = method === "horizon";
  return (
    <span
      className={[
        "rounded-full px-2 py-0.5 text-xs font-medium",
        isHorizon
          ? "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300"
          : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300",
      ].join(" ")}
    >
      {method}
    </span>
  );
}

function TriggerBadge({ trigger }: { trigger: string }) {
  const isAuto = trigger === "auto";
  return (
    <span
      className={[
        "rounded-full px-2 py-0.5 text-xs font-medium",
        isAuto
          ? "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-400"
          : "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300",
      ].join(" ")}
    >
      {trigger}
    </span>
  );
}

function AccuracyCell({ accuracy }: { accuracy: number }) {
  const pct = (accuracy * 100).toFixed(1);
  const colorClass =
    accuracy >= 0.6
      ? "text-emerald-400"
      : accuracy >= 0.45
        ? "text-yellow-400"
        : "text-red-400";
  return (
    <span className={`font-mono text-xs font-medium ${colorClass}`}>
      {pct}%
    </span>
  );
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

// The Activate button needs access to a callback — we build COLUMNS inside the
// component as a factory so it can close over the handler without stale refs.
function buildColumns(
  onActivate: (id: string) => Promise<void>,
  activatingId: string | null,
): Column<ModelVersion>[] {
  return [
    {
      key: "symbol",
      header: "Symbol",
      render: (m) => (
        <span className="font-mono text-xs text-slate-700 dark:text-slate-200">{m.symbol}</span>
      ),
    },
    {
      key: "timeframe",
      header: "Timeframe",
      render: (m) => (
        <span className="text-xs text-slate-400">{m.timeframe}</span>
      ),
    },
    {
      key: "trainedAt",
      header: "Trained At",
      sortable: true,
      sortValue: (m) => new Date(m.trainedAt).getTime(),
      render: (m) => (
        <span className="font-mono text-xs text-slate-500">
          {new Date(m.trainedAt).toLocaleString()}
        </span>
      ),
    },
    {
      key: "labelMethod",
      header: "Method",
      render: (m) => <MethodBadge method={m.labelMethod} />,
    },
    {
      key: "trigger",
      header: "Trigger",
      render: (m) => <TriggerBadge trigger={m.trigger} />,
    },
    {
      key: "accuracy",
      header: "Accuracy",
      sortable: true,
      sortValue: (m) => m.accuracy,
      render: (m) => <AccuracyCell accuracy={m.accuracy} />,
    },
    {
      key: "nTradesUsed",
      header: "Trades Used",
      sortable: true,
      sortValue: (m) => m.nTradesUsed,
      render: (m) => (
        <span className="font-mono text-xs text-slate-400">{m.nTradesUsed}</span>
      ),
    },
    {
      key: "isActive",
      header: "Active",
      render: (m) =>
        m.isActive ? (
          <span
            className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400"
            title="Active model"
            aria-label="Active"
          />
        ) : (
          <span
            className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300 dark:bg-slate-700"
            title="Inactive"
            aria-label="Inactive"
          />
        ),
    },
    {
      key: "actions",
      header: "Actions",
      render: (m) => (
        <button
          onClick={() => void onActivate(m.id)}
          disabled={m.isActive || activatingId === m.id}
          className={[
            "rounded px-2 py-1 text-xs font-medium transition-colors",
            m.isActive || activatingId === m.id
              ? "cursor-not-allowed bg-slate-200 text-slate-400 dark:bg-slate-700 dark:text-slate-500"
              : "bg-indigo-700 text-white hover:bg-indigo-600 active:bg-indigo-800",
          ].join(" ")}
          aria-label={m.isActive ? "Already active" : `Activate model ${m.id.slice(0, 8)}`}
        >
          {activatingId === m.id ? "Activating\u2026" : "Activate"}
        </button>
      ),
    },
  ];
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function ModelsPage() {
  const [models, setModels] = useState<readonly ModelVersion[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [symbolFilter, setSymbolFilter] = useState("");
  const [activeOnly, setActiveOnly] = useState(false);

  // Action state
  const [retrainSymbol, setRetrainSymbol] = useState("");
  const [retrainTimeframe, setRetrainTimeframe] = useState("1h");
  const [isRetraining, setIsRetraining] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [retrainError, setRetrainError] = useState<string | null>(null);
  const [retrainSuccess, setRetrainSuccess] = useState<string | null>(null);
  const [activatingId, setActivatingId] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------------------

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    const result = await fetchModelVersions(
      symbolFilter.trim() || undefined,
      activeOnly || undefined,
    );
    if (result.ok) {
      setModels(result.data.items);
      setTotal(result.data.total);
    } else {
      setError(result.error.message);
    }
    setIsLoading(false);
  }, [symbolFilter, activeOnly]);

  useEffect(() => {
    void load();
  }, [load]);

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  async function handleTrain() {
    const sym = retrainSymbol.trim();
    if (!sym) return;
    setIsTraining(true);
    setRetrainError(null);
    setRetrainSuccess(null);

    const result = await trainModel(sym, retrainTimeframe);
    setIsTraining(false);

    if (result.ok) {
      const acc = result.data?.metrics as Record<string, unknown> | undefined;
      const accuracy = acc?.accuracy != null ? `${(Number(acc.accuracy) * 100).toFixed(1)}%` : "";
      setRetrainSuccess(`Model trained${accuracy ? ` — ${accuracy} accuracy` : ""}`);
      setTimeout(() => {
        setRetrainSuccess(null);
        void load();
      }, 2000);
    } else {
      setRetrainError(result.error.message);
    }
  }

  async function handleRetrain() {
    const sym = retrainSymbol.trim();
    if (!sym) return;
    setIsRetraining(true);
    setRetrainError(null);
    setRetrainSuccess(null);

    const result = await retrainModel(sym, retrainTimeframe);
    setIsRetraining(false);

    if (result.ok) {
      setRetrainSuccess("Retrain accepted — requires 50+ trades with this strategy");
      setTimeout(() => {
        setRetrainSuccess(null);
        void load();
      }, 3000);
    } else {
      setRetrainError(result.error.message);
    }
  }

  async function handleActivate(id: string) {
    setActivatingId(id);
    const result = await activateModelVersion(id);
    setActivatingId(null);
    if (result.ok) {
      // Optimistic refresh — replace the updated record in local state so the
      // dot flips immediately without a full round-trip.
      setModels((prev) =>
        prev.map((m) => {
          if (m.symbol !== result.data.symbol) return m;
          // Deactivate all siblings for this symbol, activate the target.
          return { ...m, isActive: m.id === result.data.id };
        }),
      );
    } else {
      setError(result.error.message);
    }
  }

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  const columns = buildColumns(handleActivate, activatingId);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      <Header
        title="ML Models"
        subtitle="Manage trained model versions and trigger retraining."
      />

      {/* Filter + Retrain bar */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/50">
        {/* Symbol filter */}
        <div className="flex flex-col gap-1">
          <label
            htmlFor="symbol-filter"
            className="text-xs font-medium text-slate-500"
          >
            Filter by symbol
          </label>
          <input
            id="symbol-filter"
            type="text"
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            placeholder="e.g. BTC/USD"
            className="w-40 rounded bg-white px-3 py-1.5 text-xs text-slate-900 border border-slate-300 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        {/* Active only toggle */}
        <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={activeOnly}
            onChange={(e) => setActiveOnly(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-slate-300 bg-white accent-indigo-500 dark:border-slate-600 dark:bg-slate-800"
          />
          Active only
        </label>

        {/* Divider */}
        <div className="hidden h-8 w-px bg-slate-300 dark:bg-slate-700 sm:block" aria-hidden="true" />

        {/* Retrain controls */}
        <div className="flex flex-col gap-1">
          <label
            htmlFor="retrain-symbol"
            className="text-xs font-medium text-slate-500"
          >
            Retrain symbol
          </label>
          <input
            id="retrain-symbol"
            type="text"
            value={retrainSymbol}
            onChange={(e) => setRetrainSymbol(e.target.value)}
            placeholder="e.g. BTC/USD"
            className="w-40 rounded bg-white px-3 py-1.5 text-xs text-slate-900 border border-slate-300 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label
            htmlFor="retrain-timeframe"
            className="text-xs font-medium text-slate-500"
          >
            Timeframe
          </label>
          <select
            id="retrain-timeframe"
            value={retrainTimeframe}
            onChange={(e) => setRetrainTimeframe(e.target.value)}
            className="rounded bg-white px-3 py-1.5 text-xs text-slate-900 border border-slate-300 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            {["1m", "5m", "15m", "1h", "4h", "1d"].map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={() => void handleTrain()}
          disabled={isTraining || isRetraining || !retrainSymbol.trim()}
          className={[
            "self-end rounded-lg px-4 py-2 text-xs font-medium transition-colors",
            isTraining || !retrainSymbol.trim()
              ? "cursor-not-allowed bg-slate-200 text-slate-400 dark:bg-slate-700 dark:text-slate-500"
              : "bg-indigo-600 text-white hover:bg-indigo-500 active:bg-indigo-700",
          ].join(" ")}
          title="Train a new model from historical OHLCV data (always works)"
        >
          {isTraining ? "Training\u2026" : "Train"}
        </button>

        <button
          onClick={() => void handleRetrain()}
          disabled={isRetraining || isTraining || !retrainSymbol.trim()}
          className={[
            "self-end rounded-lg px-4 py-2 text-xs font-medium transition-colors",
            isRetraining || !retrainSymbol.trim()
              ? "cursor-not-allowed bg-slate-200 text-slate-400 dark:bg-slate-700 dark:text-slate-500"
              : "bg-slate-600 text-white hover:bg-slate-500 active:bg-slate-700",
          ].join(" ")}
          title="Retrain from trade outcomes (requires 50+ trades with model_strategy)"
        >
          {isRetraining ? "Retraining\u2026" : "Retrain"}
        </button>

        {/* Feedback messages */}
        {retrainSuccess && (
          <span className="self-end text-xs font-medium text-emerald-400">
            {retrainSuccess}
          </span>
        )}
        {retrainError && (
          <span className="self-end text-xs text-red-400">{retrainError}</span>
        )}
      </div>

      {/* General error banner */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Model versions table */}
      <DataTable
        columns={columns}
        data={models}
        keyExtractor={(m) => m.id}
        emptyMessage="No trained model versions found. Run a training job first."
        isLoading={isLoading}
      />

      {/* Result count */}
      {!isLoading && total > 0 && (
        <p className="text-xs text-slate-500 dark:text-slate-600 tabular-nums">
          {total} model version{total !== 1 ? "s" : ""}
        </p>
      )}
    </div>
  );
}
