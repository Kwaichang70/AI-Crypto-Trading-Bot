/**
 * Dashboard Home — Server Component.
 * Fetches health + recent runs at request time.
 */

import type { Metadata } from "next";
import Link from "next/link";
import { fetchHealth, fetchRuns } from "@/lib/api";
import type { HealthResponse } from "@/lib/api";
import type { Run } from "@/lib/types";
import { StatCard } from "@/components/ui/stat-card";
import { RunStatusBadge } from "@/components/ui/status-badge";
import { Header } from "@/components/layout/header";

export const metadata: Metadata = { title: "Dashboard" };
export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusIndicator({
  status,
}: {
  readonly status: HealthResponse["status"] | "unreachable";
}) {
  const map = {
    ok: { dot: "bg-green-500", badge: "badge-success", label: "API Online" },
    degraded: { dot: "bg-amber-500", badge: "badge-warning", label: "API Degraded" },
    error: { dot: "bg-red-500", badge: "badge-danger", label: "API Error" },
    unreachable: { dot: "bg-slate-500", badge: "badge-neutral", label: "API Unreachable" },
  } as const;

  const { dot, badge, label } = map[status];
  return (
    <span className={badge}>
      <span className={`mr-1.5 inline-block h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      {label}
    </span>
  );
}

function RecentRunsTable({ runs }: { runs: readonly Run[] }) {
  if (runs.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-slate-500">
        No runs yet. Start by creating a new backtest.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800">
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">ID</th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Mode</th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Strategy</th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Status</th>
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id} className="border-b border-slate-800/50 hover:bg-slate-800/20">
              <td className="px-4 py-3">
                <Link
                  href={`/runs/${run.id}`}
                  className="font-mono text-xs text-indigo-400 hover:text-indigo-300 hover:underline"
                >
                  {run.id.slice(0, 8)}…
                </Link>
              </td>
              <td className="px-4 py-3 text-slate-400">{run.runMode}</td>
              <td className="px-4 py-3 text-slate-300">{run.config?.strategyName ?? "—"}</td>
              <td className="px-4 py-3">
                <RunStatusBadge status={run.status} />
              </td>
              <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                {new Date(run.startedAt).toLocaleDateString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default async function DashboardPage() {
  const [healthResult, runsResult] = await Promise.all([
    fetchHealth(),
    fetchRuns({ limit: 10 }),
  ]);

  const runs: readonly Run[] = runsResult.ok ? runsResult.data.items : [];
  const totalRuns = runsResult.ok ? runsResult.data.total : 0;
  const activeRuns = runs.filter((r) => r.status === "running").length;
  const errorRuns = runs.filter((r) => r.status === "error").length;

  return (
    <div className="space-y-6">
      <Header
        title="Trading Bot Dashboard"
        subtitle="Monitor live runs, review backtest results, and manage strategy configuration."
        actions={
          <Link
            href="/runs/new"
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            New Backtest
          </Link>
        }
      />

      {/* Summary cards */}
      <section aria-label="Summary metrics" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Runs"
          value={totalRuns}
          subValue="all time"
        />
        <StatCard
          label="Active Runs"
          value={activeRuns}
          trend={activeRuns > 0 ? "up" : "neutral"}
          subValue="currently running"
        />
        <StatCard
          label="Error Runs"
          value={errorRuns}
          trend={errorRuns > 0 ? "down" : "neutral"}
          subValue="needs attention"
        />
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              API Status
            </p>
            {healthResult.ok ? (
              <StatusIndicator status={healthResult.data.status} />
            ) : (
              <StatusIndicator status="unreachable" />
            )}
          </div>
          {healthResult.ok && (
            <p className="font-mono text-xs text-slate-500">
              v{healthResult.data.version}
            </p>
          )}
        </div>
      </section>

      {/* Recent runs */}
      <section aria-label="Recent runs">
        <div className="card">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-200">Recent Runs</h2>
            <Link
              href="/runs"
              className="text-xs text-indigo-400 hover:text-indigo-300 hover:underline"
            >
              View all
            </Link>
          </div>
          <RecentRunsTable runs={runs} />
        </div>
      </section>

      {/* API error notice */}
      {!runsResult.ok && (
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          Could not load run data: {runsResult.error.message}
        </div>
      )}
    </div>
  );
}
