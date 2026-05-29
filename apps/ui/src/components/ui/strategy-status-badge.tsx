import type { StrategyStatus } from "@/lib/types";

/**
 * Strategy promotion-status badge (Sprint 51 Cycle 2).
 *
 * - "demoted"      → amber "Backtest only"
 * - "experimental" → blue "Experimental"
 * - "active"       → subtle neutral "Live-ready"
 *
 * Renders nothing when status is undefined (graceful degrade for older API
 * responses that predate the strategy-lockdown fields).
 */
export function StrategyStatusBadge({
  status,
}: {
  status?: StrategyStatus | undefined;
}) {
  if (!status) return null;

  if (status === "demoted") {
    return (
      <span className="inline-flex items-center rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/30 dark:text-amber-400">
        Backtest only
      </span>
    );
  }

  if (status === "experimental") {
    return (
      <span className="inline-flex items-center rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900/30 dark:text-blue-400">
        Experimental
      </span>
    );
  }

  // "active"
  return (
    <span className="inline-flex items-center rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
      Live-ready
    </span>
  );
}
