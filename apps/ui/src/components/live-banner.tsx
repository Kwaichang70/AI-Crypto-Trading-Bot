/**
 * apps/ui/src/components/live-banner.tsx
 * ------------------------------------------
 * Persistent "real funds at risk" banner (S13). Rendered whenever the
 * current context is live trading — a live run's detail page (any status)
 * or the New Run form once mode=live is selected.
 */

export function LiveBanner({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="status"
      className={
        compact
          ? "rounded-lg border border-red-700 bg-red-900/30 px-3 py-1.5 text-xs font-semibold text-red-300"
          : "rounded-lg border border-red-700 bg-red-900/30 px-4 py-2 text-sm font-semibold text-red-300"
      }
    >
      LIVE — real funds at risk
    </div>
  );
}
