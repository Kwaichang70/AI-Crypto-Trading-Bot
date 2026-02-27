/**
 * Home Dashboard Page — Server Component.
 *
 * Fetches the API health status at request time (no client-side JS required
 * for the initial paint). The component is intentionally a thin shell that
 * will be filled out in subsequent implementation sprints with:
 *   - Equity curve chart
 *   - Drawdown chart
 *   - Open positions table
 *   - Recent trades feed
 *
 * Data-fetching uses Next.js 14 extended `fetch` with `{ cache: "no-store" }`
 * so every request gets a fresh status — appropriate for a live dashboard.
 */

import type { Metadata } from "next";
import { fetchHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/api";

export const metadata: Metadata = {
  title: "Dashboard",
};

// Force dynamic rendering so the health check fires on every request.
export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Sub-components (server-side, no 'use client' needed)
// ---------------------------------------------------------------------------

function StatusIndicator({
  status,
}: {
  readonly status: HealthResponse["status"] | "unreachable";
}) {
  const map = {
    ok: {
      dot: "bg-green-500",
      badge: "badge-success",
      label: "API Online",
    },
    degraded: {
      dot: "bg-amber-500",
      badge: "badge-warning",
      label: "API Degraded",
    },
    error: {
      dot: "bg-red-500",
      badge: "badge-danger",
      label: "API Error",
    },
    unreachable: {
      dot: "bg-slate-500",
      badge: "badge-neutral",
      label: "API Unreachable",
    },
  } as const;

  const { dot, badge, label } = map[status];

  return (
    <span className={badge}>
      <span
        className={`mr-1.5 inline-block h-2 w-2 rounded-full ${dot}`}
        aria-hidden="true"
      />
      {label}
    </span>
  );
}

function HealthCard({
  result,
}: {
  readonly result: Awaited<ReturnType<typeof fetchHealth>>;
}) {
  if (!result.ok) {
    return (
      <div className="card space-y-1">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
          Backend Health
        </p>
        <div className="flex items-center justify-between">
          <StatusIndicator status="unreachable" />
          <span className="text-xs text-slate-500">{result.error.message}</span>
        </div>
      </div>
    );
  }

  const { status, version, timestamp, components } = result.data;

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
          Backend Health
        </p>
        <StatusIndicator status={status} />
      </div>

      <dl className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-slate-500">Version</dt>
          <dd className="font-mono font-medium text-slate-200">{version}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Server time</dt>
          <dd className="font-mono font-medium text-slate-200">
            {new Date(timestamp).toLocaleTimeString()}
          </dd>
        </div>

        {components &&
          Object.entries(components).map(([name, compStatus]) => (
            <div key={name}>
              <dt className="capitalize text-slate-500">{name}</dt>
              <dd>
                <StatusIndicator status={compStatus} />
              </dd>
            </div>
          ))}
      </dl>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Placeholder panel — shown until real charts are implemented
// ---------------------------------------------------------------------------
function PlaceholderPanel({
  title,
  description,
}: {
  readonly title: string;
  readonly description: string;
}) {
  return (
    <div className="card flex min-h-[180px] flex-col items-center justify-center space-y-2 text-center">
      <p className="text-sm font-medium text-slate-300">{title}</p>
      <p className="text-xs text-slate-500">{description}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default async function DashboardPage() {
  const healthResult = await fetchHealth();

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-100">
          Trading Bot Dashboard
        </h1>
        <p className="text-sm text-slate-400">
          Monitor live runs, review backtest results, and manage strategy
          configuration.
        </p>
      </div>

      {/* Health status row */}
      <section aria-label="System status" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <HealthCard result={healthResult} />
      </section>

      {/* Primary chart area */}
      <section
        aria-label="Portfolio overview"
        className="grid gap-4 lg:grid-cols-2"
      >
        <PlaceholderPanel
          title="Equity Curve"
          description="Portfolio value over time — available after first run"
        />
        <PlaceholderPanel
          title="Drawdown Chart"
          description="Peak-to-trough drawdown — available after first run"
        />
      </section>

      {/* Secondary panels */}
      <section
        aria-label="Positions and trades"
        className="grid gap-4 lg:grid-cols-2"
      >
        <PlaceholderPanel
          title="Open Positions"
          description="Active positions across all running strategies"
        />
        <PlaceholderPanel
          title="Recent Trades"
          description="Latest executed fills from all runs"
        />
      </section>
    </div>
  );
}
