/**
 * apps/ui/src/components/flatten-result-view.tsx
 * -------------------------------------------------
 * Read-only view of a WP1.7a `FlattenResult` (reports/vp2-wp1.7/synthesis-
 * spec.md §5). Reused by:
 *   - `StopRunDialog` (the 200 and 409 `flatten_incomplete` paths)
 *   - the kill-switch panel (one instance per entry in `flattenResults`)
 *   - `emergencyStop`'s result, wherever a caller wants to show it
 *
 * Renders all four `outcome` values ("noop" | "flattened" | "partial" |
 * "failed") plus the `complete === false` (409 `flatten_incomplete`) case,
 * which persists the run's own `entries_latch_reason` and is NOT simply
 * "failed" — the operator can retry or clear the latch, so it gets its own
 * distinct banner instead of reusing the "failed" styling.
 */

import type { FlattenResult, FlattenSymbolResult } from "@/lib/types";

const OUTCOME_STYLES: Record<
  FlattenResult["outcome"],
  { label: string; className: string }
> = {
  noop: {
    label: "No positions held — nothing to flatten.",
    className: "border-slate-300 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-400",
  },
  flattened: {
    label: "Flattened — every held symbol was sold.",
    className: "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-400",
  },
  partial: {
    label: "Partially flattened — some symbols were not fully sold.",
    className: "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400",
  },
  failed: {
    label: "Flatten failed — no symbols were sold.",
    className: "border-red-300 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400",
  },
};

function SymbolRow({ row }: { row: FlattenSymbolResult }) {
  return (
    <tr className="border-b border-slate-100 last:border-0 dark:border-slate-800">
      <td className="py-1.5 pr-3 font-mono text-xs text-slate-700 dark:text-slate-300">{row.symbol}</td>
      <td className="py-1.5 pr-3 text-xs">{row.status}</td>
      <td className="py-1.5 pr-3 text-xs text-slate-500">{row.cause ?? "—"}</td>
      <td className="py-1.5 pr-3 font-mono text-xs text-slate-500">{row.heldBefore}</td>
      <td className="py-1.5 pr-3 font-mono text-xs text-slate-500">{row.soldQty}</td>
      <td className="py-1.5 pr-3 font-mono text-xs text-slate-500">{row.remainingQty}</td>
      <td className="py-1.5 text-xs text-red-500">{row.error ?? ""}</td>
    </tr>
  );
}

export function FlattenResultView({ result }: { result: FlattenResult }) {
  const style = OUTCOME_STYLES[result.outcome];

  return (
    <div className="space-y-2">
      <div className={`rounded-lg border px-3 py-2 text-xs font-medium ${style.className}`}>
        {style.label}
      </div>

      {!result.complete && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          Entries latch is now active on this run (
          <code className="font-mono">flatten_incomplete</code>) — the run
          stays running with new BUYs blocked. Retry the flatten, stop again
          without flatten, or have an admin clear the latch.
          {!result.latchPersisted && (
            <span className="block mt-1">
              Warning: the latch could not be persisted — it is enforced
              in-memory only until the next restart.
            </span>
          )}
        </div>
      )}

      {result.symbols.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-slate-500 dark:border-slate-800 dark:bg-slate-900/50">
                <th className="py-1.5 pr-3 pl-2">Symbol</th>
                <th className="py-1.5 pr-3">Status</th>
                <th className="py-1.5 pr-3">Cause</th>
                <th className="py-1.5 pr-3">Held</th>
                <th className="py-1.5 pr-3">Sold</th>
                <th className="py-1.5 pr-3">Remaining</th>
                <th className="py-1.5">Error</th>
              </tr>
            </thead>
            <tbody>
              {result.symbols.map((row) => (
                <SymbolRow key={row.symbol} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
