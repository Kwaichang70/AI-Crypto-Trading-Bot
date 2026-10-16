/**
 * WP2.1b: "Cost model" card on the run detail page.
 * Live runs book the exchange's REAL fees; the profile is only an estimate.
 */

import type { Run } from "@/lib/types";
import { isLegacyCosts } from "@/lib/cost-profile";
import { CostProfileBadge } from "@/components/ui/cost-profile-badge";
import { formatCurrency } from "@/lib/api";
import { quoteCurrencyPrefix } from "@/lib/currency";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-xs">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono text-slate-700 dark:text-slate-300">{children}</dd>
    </div>
  );
}

export function CostModelCard({ run }: { run: Run }) {
  const p = run.costProfile ?? null;
  const legacy = isLegacyCosts(run);
  const isLive = run.runMode === "live";
  const fees = run.backtestMetrics?.totalFeesPaid;
  const ccy = quoteCurrencyPrefix(run.config?.symbols, run.backtestMetrics?.quoteCurrency);

  return (
    <div className="card" data-testid="cost-model-card">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Cost model</h3>
        <CostProfileBadge value={run} />
      </div>

      {isLive ? (
        <p className="mb-3 text-xs text-slate-600 dark:text-slate-400" data-testid="cost-model-live-note">
          fees: actual exchange fills; estimate profile{" "}
          <span className="font-mono">{p?.id ?? "unknown (legacy costs)"}</span>
        </p>
      ) : legacy ? (
        <p className="mb-3 text-xs text-amber-600 dark:text-amber-400">
          This run predates the WP2.1 cost model (or its cost profile could not be read).
          Its returns are not comparable with current runs.
        </p>
      ) : null}

      {p ? (
        <dl className="space-y-1.5">
          <Row label="Profile">{p.id}</Row>
          <Row label="Tier">{p.tier}</Row>
          <Row label="Taker fee">{p.takerFeeBps} bps</Row>
          <Row label="Maker fee">{p.makerFeeBps} bps</Row>
          <Row label="Slippage">{p.slippageBps} bps</Row>
          <Row label="Round trip">{p.roundTripBps} bps</Row>
          <Row label="Fee source">
            {p.feeSource === "exchange_fills" ? "exchange fills (estimate only)" : "model"}
          </Row>
          {fees != null && !isLive && (
            <Row label="Total fees paid">{`${ccy}${formatCurrency(fees)}`}</Row>
          )}
        </dl>
      ) : (
        <p className="text-xs text-slate-500">No cost profile reported for this run.</p>
      )}
    </div>
  );
}
