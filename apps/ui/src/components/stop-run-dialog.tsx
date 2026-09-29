/**
 * apps/ui/src/components/stop-run-dialog.tsx
 * ------------------------------------------------
 * WP1.7a/1.7b (CF-B3, S13) stop confirmation for a run.
 *
 * - Non-live runs: a single "Stop Run" confirm (flatten is OMITTED from the
 *   request entirely in that case — see WP17b-S-02 below — the backend
 *   still treats a paper run's flatten as optional/false there).
 * - A running LIVE run: the backend REQUIRES an explicit flatten decision
 *   (422 `flatten_decision_required` if omitted, synthesis-spec SY-05) — the
 *   Stop button stays disabled until the operator picks
 *   "Flatten (sell everything)" or "Keep positions".
 * - Handles the 409 `flatten_incomplete` response: the run is NOT stopped,
 *   stays 'running' with its own entries latch active. The operator can
 *   retry, stop again without flatten, or (admin) clear the per-run latch
 *   via `POST /runs/{id}/entries-latch/clear`.
 *
 * WP17b-S-01 (round 2, critical): FastAPI's default `HTTPException` handler
 * wraps `detail=` in a top-level `{"detail": {...}}` envelope on the wire —
 * `apiFetch` (lib/api.ts) stores that WHOLE parsed body in `error.detail`,
 * so a real 409/422 here is `{"detail": {"code": ..., ...}}`, one level
 * deeper than the `{code, ...}` shape this file's `doStop()` used to assume
 * directly. `unwrapDetail()` below normalises this at the point of
 * consumption (rather than in the shared `apiFetch`, which every other
 * caller in the app already treats as "the raw parsed body" and which has
 * no other caller reading `.detail` today — see the producer round-2
 * report's grep proof) so this is the ONLY place that needs to know about
 * FastAPI's envelope shape.
 *
 * WP17b-S-02 (round 2): a stale `run.status` (e.g. a live run the parent
 * still shows as 'resuming' because its own poll hasn't refreshed yet, or
 * simply hasn't refreshed since the backend moved it to 'running') must
 * NEVER cause this dialog to silently submit `flatten=false` for a run the
 * backend would otherwise require a decision for. So:
 *   - `doStop(flatten?)` OMITS the `flatten` query param entirely whenever
 *     no explicit choice was shown/made (`flatten === undefined`) — the
 *     backend's own 422 `flatten_decision_required` is then the safety net,
 *     never a client-guessed `false`.
 *   - On that 422, `forceChoice` is set so the radio-choice UI renders
 *     regardless of the `run.status` the parent originally passed in.
 *
 * AC8: `stopRun` already carries a >= 60s client timeout (see `lib/api.ts`).
 */

"use client";

import { useState } from "react";
import { stopRun, formatCurrency } from "@/lib/api";
import { adminFetch, ENTRIES_LATCH_CLEAR_TIMEOUT_MS } from "@/lib/admin-fetch";
import type {
  EntriesLatchClearResponse,
  FlattenDecisionRequiredDetail,
  FlattenIncompleteDetail,
  FlattenRequiresRunningEngineDetail,
  FlattenResult,
  Position,
  Run,
} from "@/lib/types";
import { AdminOnly } from "@/components/admin-only";
import { DataTable, type Column } from "@/components/ui/data-table";
import { FlattenResultView } from "@/components/flatten-result-view";
import { LiveConfirmDialog } from "@/components/live-confirm-dialog";

const POSITION_COLUMNS: Column<Position>[] = [
  { key: "symbol", header: "Symbol", render: (p) => <span className="font-mono text-xs">{p.symbol}</span> },
  { key: "quantity", header: "Quantity", render: (p) => <span className="font-mono text-xs">{p.quantity}</span> },
  {
    key: "unrealisedPnl",
    header: "Unrealised PnL",
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.unrealisedPnl)}</span>,
  },
];

type StopErrorDetail =
  | FlattenDecisionRequiredDetail
  | FlattenIncompleteDetail
  | FlattenRequiresRunningEngineDetail
  | { code?: string };

/**
 * WP17b-S-01: unwrap FastAPI's `{"detail": {...}}` HTTPException envelope.
 * `apiFetch` hands back the ENTIRE parsed response body as `error.detail`
 * (it has no way to know any particular endpoint's error shape), so a real
 * `HTTPException(409, detail={"code": "flatten_incomplete", ...})` arrives
 * here as `{ detail: { code: "flatten_incomplete", ... } }` — one level
 * deeper than the bare `{code, ...}` this function returns.
 *
 * WP17b-S-R2-02 (round 3, regression fix): FastAPI's `detail` is very often
 * a plain STRING, not an object -- a 404 ("Run ... not found"), a 409 from
 * a plain-string `HTTPException` ("Cannot stop run ...: current status is
 * 'stopped'"), a 500 (`{"detail": "Internal Server Error"}`), or a
 * text/plain body `apiFetch` stored as a raw string. Round 2's version
 * returned that string as-is, and the `"code" in detail` checks below then
 * threw `TypeError: Cannot use 'in' operator to search for 'code' in
 * "<string>"` -- `doStop` rejected before ever reaching `setLoading(false)`,
 * so the dialog froze on "Stopping…" with Stop AND Cancel both disabled.
 * This now returns `undefined` for anything that isn't a plain, non-array
 * object -- every string/number/array/null `detail` falls through to the
 * generic `result.error.message` branch in `doStop` below instead of
 * crashing.
 */
function unwrapDetail(raw: unknown): StopErrorDetail | undefined {
  const inner =
    raw && typeof raw === "object" && "detail" in raw
      ? (raw as { detail?: unknown }).detail
      : raw;
  return inner && typeof inner === "object" && !Array.isArray(inner)
    ? (inner as StopErrorDetail)
    : undefined;
}

interface StopRunDialogProps {
  run: Run;
  positions: readonly Position[];
  onClose: () => void;
  onStopped: (updated: Run) => void;
}

export function StopRunDialog({ run, positions, onClose, onStopped }: StopRunDialogProps) {
  const isLive = run.runMode === "live";
  const staleNeedsFlattenChoice = isLive && run.status === "running";

  const [flattenChoice, setFlattenChoice] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [heldSymbols, setHeldSymbols] = useState<readonly string[] | null>(null);
  const [incompleteFlatten, setIncompleteFlatten] = useState<FlattenResult | null>(null);
  const [successResult, setSuccessResult] = useState<{ flatten: FlattenResult | null } | null>(null);
  // WP17b-S-02: once the backend itself has told us (via 422) that a
  // decision is required, ALWAYS show the choice UI from then on, even if
  // `run.status` (as passed in by the parent) is stale and doesn't yet say
  // 'running' for this live run.
  const [forceChoice, setForceChoice] = useState(false);
  const needsFlattenChoice = staleNeedsFlattenChoice || forceChoice;

  // --- Admin-only entries-latch clear sub-form (shown after a 409) -------
  const [clearReason, setClearReason] = useState("");
  const [clearLoading, setClearLoading] = useState(false);
  const [clearError, setClearError] = useState<string | null>(null);
  const [clearTokenDialogOpen, setClearTokenDialogOpen] = useState(false);

  const canSubmit = needsFlattenChoice ? flattenChoice !== null : true;

  /**
   * @param flatten `undefined` OMITS the `?flatten=` query param entirely
   *   (WP17b-S-02) — never sends an implicit `false` for a run whose
   *   liveness/status this dialog cannot fully trust. Only ever called with
   *   an explicit `true`/`false` from a user action (the radio choice, or
   *   the 409 view's "Stop without flatten" button).
   */
  async function doStop(flatten?: boolean) {
    setLoading(true);
    setErrorMessage(null);
    setHeldSymbols(null);
    setIncompleteFlatten(null);

    // WP17b-S-R2-02 (round 3): try/finally so ANY exception between here
    // and the end of this function (a rejected `stopRun` promise, or a
    // future bug in the branches below) still clears `loading` -- the
    // round-2 regression was exactly this: an uncaught throw from
    // `"code" in detail` left `loading` stuck `true` forever, disabling
    // both Stop and Cancel with no way out but a page reload.
    try {
      const result = await stopRun(run.id, flatten === undefined ? {} : { flatten });

      if (result.ok) {
        setSuccessResult({ flatten: result.data.flatten });
        onStopped(result.data);
        if (!result.data.flatten) {
          onClose();
        }
        return;
      }

      const detail = unwrapDetail(result.error.detail);

      if (detail && detail.code === "flatten_decision_required") {
        setForceChoice(true);
        setHeldSymbols((detail as FlattenDecisionRequiredDetail).held_symbols);
      } else if (detail && detail.code === "flatten_incomplete") {
        setIncompleteFlatten((detail as FlattenIncompleteDetail).flatten);
      } else if (detail && detail.code === "flatten_requires_running_engine") {
        setErrorMessage(
          "This run has no live engine in this process — flatten cannot run. Stop without flatten instead.",
        );
      } else {
        setErrorMessage(result.error.message);
      }
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || loading) return;
    // WP17b-S-R2-02 (round 3): `doStop`'s own try/finally guarantees
    // `loading` is always cleared, but the promise can still legitimately
    // reject afterwards (a genuinely unexpected error, not one of the
    // handled `ApiResult`/detail branches) -- swallow it here rather than
    // leaving an unhandled promise rejection (the error is a real bug to
    // fix if it ever fires, but it must never crash/freeze the dialog).
    void doStop(needsFlattenChoice ? Boolean(flattenChoice) : undefined).catch(() => {});
  }

  async function submitClearLatch(liveConfirmToken?: string) {
    if (clearReason.trim().length < 3) {
      setClearError("Reason must be at least 3 characters.");
      return;
    }
    setClearLoading(true);
    setClearError(null);
    const result = await adminFetch<EntriesLatchClearResponse>(
      `/api/admin/runs/${run.id}/entries-latch/clear`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(liveConfirmToken ? { "X-Live-Confirm-Token": liveConfirmToken } : {}),
        },
        body: JSON.stringify({ reason: clearReason.trim() }),
      },
      ENTRIES_LATCH_CLEAR_TIMEOUT_MS,
    );
    setClearLoading(false);
    setClearTokenDialogOpen(false);
    if (result.ok) {
      setIncompleteFlatten(null);
      setClearReason("");
    } else {
      setClearError(result.error.message);
    }
  }

  function handleClearLatchClick() {
    if (isLive) {
      setClearTokenDialogOpen(true);
    } else {
      void submitClearLatch();
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget && !loading) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="stop-run-dialog-title"
        className="mx-4 w-full max-w-lg rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
      >
        <h2 id="stop-run-dialog-title" className="text-base font-semibold text-slate-100">
          Stop Run {run.id.slice(0, 8)}…
        </h2>

        {!successResult && !incompleteFlatten && (
          <form onSubmit={handleSubmit} className="mt-3 space-y-4">
            {needsFlattenChoice && (
              <>
                <p className="text-sm text-slate-400">
                  This is a running <span className="font-semibold text-red-400">LIVE</span> run.
                  Choose what happens to any open positions before it stops.
                </p>
                {positions.length > 0 && (
                  <DataTable
                    columns={POSITION_COLUMNS}
                    data={positions}
                    keyExtractor={(p) => p.symbol}
                    emptyMessage="No open positions."
                  />
                )}
                <fieldset className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input
                      type="radio"
                      name="flatten-choice"
                      checked={flattenChoice === true}
                      onChange={() => setFlattenChoice(true)}
                    />
                    Flatten (sell everything) before stopping
                  </label>
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input
                      type="radio"
                      name="flatten-choice"
                      checked={flattenChoice === false}
                      onChange={() => setFlattenChoice(false)}
                    />
                    Keep positions (stop without flattening)
                  </label>
                </fieldset>
                {heldSymbols && heldSymbols.length > 0 && (
                  <p className="text-xs text-amber-400">
                    Held symbols: {heldSymbols.join(", ")}
                  </p>
                )}
              </>
            )}

            {errorMessage && (
              <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
                {errorMessage}
              </div>
            )}

            <div className="flex gap-3">
              <button
                type="button"
                onClick={onClose}
                disabled={loading}
                className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!canSubmit || loading}
                className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Stopping…" : "Stop Run"}
              </button>
            </div>
          </form>
        )}

        {successResult?.flatten && (
          <div className="mt-3 space-y-4">
            <FlattenResultView result={successResult.flatten} />
            <button
              type="button"
              onClick={onClose}
              className="w-full rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600"
            >
              Done
            </button>
          </div>
        )}

        {incompleteFlatten && (
          <div className="mt-3 space-y-4">
            <FlattenResultView result={incompleteFlatten} />

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => {
                  setIncompleteFlatten(null);
                  setForceChoice(true);
                  setFlattenChoice(true);
                }}
                disabled={loading}
                className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
              >
                Retry flatten
              </button>
              <button
                type="button"
                onClick={() => void doStop(false).catch(() => {})}
                disabled={loading}
                className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                Stop without flatten
              </button>
            </div>

            <AdminOnly>
              <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3 space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Admin: clear entries latch
                </p>
                <textarea
                  value={clearReason}
                  onChange={(e) => setClearReason(e.target.value)}
                  placeholder="Reason (3-500 characters)"
                  maxLength={500}
                  rows={2}
                  className="w-full resize-none rounded-lg border border-slate-600 bg-slate-900 px-2 py-1.5 text-xs text-slate-200 placeholder-slate-500"
                />
                {clearError && <p className="text-xs text-red-400">{clearError}</p>}
                <button
                  type="button"
                  onClick={handleClearLatchClick}
                  disabled={clearLoading || clearReason.trim().length < 3}
                  className="w-full rounded-lg border border-amber-600 bg-amber-900/20 px-3 py-1.5 text-xs font-medium text-amber-400 hover:bg-amber-900/40 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {clearLoading ? "Clearing…" : "Clear entries latch"}
                </button>
              </div>
            </AdminOnly>

            <button
              type="button"
              onClick={onClose}
              className="w-full rounded-lg border border-slate-600 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800"
            >
              Close
            </button>
          </div>
        )}
      </div>

      {isLive && (
        <LiveConfirmDialog
          open={clearTokenDialogOpen}
          title="Confirm clearing the entries latch"
          description="This live run's entries latch will be cleared, re-enabling BUYs."
          confirmLabel="Clear latch"
          loading={clearLoading}
          onCancel={() => setClearTokenDialogOpen(false)}
          onConfirm={(token) => submitClearLatch(token)}
        />
      )}
    </div>
  );
}
