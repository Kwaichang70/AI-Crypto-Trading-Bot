"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { fetchOptimizationRun, createRun } from "@/lib/api";
import type { OptimizeResponse, OptimizeEntry } from "@/lib/types";
import { Header } from "@/components/layout/header";
import { DataTable } from "@/components/ui/data-table";
import { buildResultColumns } from "../result-columns";

// ---------------------------------------------------------------------------
// Page phase discriminated union
// ---------------------------------------------------------------------------

type PagePhase =
  | { kind: "loading" }
  | { kind: "loaded"; data: OptimizeResponse }
  | { kind: "error"; message: string };

// ---------------------------------------------------------------------------
// Detail page
// ---------------------------------------------------------------------------

export default function OptimizationRunDetailPage() {
  const router = useRouter();
  // useParams() is the codebase-standard pattern for client components in Next.js
  // App Router — matches runs/[id]/page.tsx and is forward-compatible with Next.js 15+.
  const rawParams = useParams();
  const id = typeof rawParams.id === "string" ? rawParams.id : "";

  const [phase, setPhase] = useState<PagePhase>({ kind: "loading" });
  const [launchingRank, setLaunchingRank] = useState<number | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [launchCapital, setLaunchCapital] = useState("10000");

  useEffect(() => {
    if (!id) {
      setPhase({ kind: "error", message: "Invalid optimization run ID." });
      return;
    }
    setPhase({ kind: "loading" });
    fetchOptimizationRun(id).then((r) => {
      if (r.ok) {
        setPhase({ kind: "loaded", data: r.data });
      } else {
        setPhase({ kind: "error", message: r.error.message });
      }
    });
  }, [id]);

  async function handleLaunchRun(entry: OptimizeEntry) {
    if (phase.kind !== "loaded") return;
    setLaunchingRank(entry.rank);
    setLaunchError(null);

    const result = await createRun({
      strategyName: phase.data.strategyName,
      strategyParams: entry.params,
      symbols: [...phase.data.symbols],
      timeframe: phase.data.timeframe,
      mode: "backtest",
      initialCapital: launchCapital,
      backtestStart: null,
      backtestEnd: null,
    });

    setLaunchingRank(null);
    if (result.ok) {
      router.push(`/runs/${result.data.id}`);
    } else {
      setLaunchError(`Failed to launch run: ${result.error.message}`);
    }
  }

  // --- Loading state ---
  if (phase.kind === "loading") {
    return (
      <div className="flex-1 space-y-6 p-6">
        <div className="h-5 w-40 animate-pulse rounded bg-slate-800" />
        <div className="h-8 w-96 animate-pulse rounded bg-slate-800" />
        <div className="h-12 w-full animate-pulse rounded-xl bg-slate-800" />
        <div className="h-64 w-full animate-pulse rounded-xl bg-slate-800" />
      </div>
    );
  }

  // --- Error state ---
  if (phase.kind === "error") {
    return (
      <div className="flex-1 space-y-4 p-6">
        <Link
          href="/optimize"
          className="text-sm text-indigo-400 hover:underline"
        >
          Back to Optimize
        </Link>
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          {phase.message}
        </div>
      </div>
    );
  }

  // --- Loaded state ---
  const { data } = phase;

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Back link */}
      <Link
        href="/optimize"
        className="text-sm text-slate-500 hover:text-slate-300 hover:underline"
      >
        Back to Optimize
      </Link>

      {/* Page header */}
      <Header
        title={`Optimization: ${data.strategyName}`}
        subtitle={`${data.timeframe} · ${data.symbols.join(", ")} · ranked by ${data.rankBy}`}
      />

      {/* Summary bar */}
      <div className="card p-4">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <span className="text-slate-400">
            Strategy: <span className="text-slate-200">{data.strategyName}</span>
          </span>
          <span className="text-slate-600">·</span>
          <span className="text-slate-400">
            Ranked by: <span className="text-indigo-300">{data.rankBy}</span>
          </span>
          <span className="text-slate-600">·</span>
          <span className="text-slate-400">
            Combinations:{" "}
            <span className="text-slate-200">
              {data.completedCombinations}/{data.totalCombinations}
            </span>
          </span>
          {data.failedCombinations > 0 && (
            <>
              <span className="text-slate-600">·</span>
              <span className="text-amber-400">{data.failedCombinations} failed</span>
            </>
          )}
          <span className="text-slate-600">·</span>
          <span className="text-slate-400">
            Elapsed: <span className="text-slate-200">{data.elapsedSeconds}s</span>
          </span>
        </div>
      </div>

      {/* Launch configuration */}
      <div className="card p-4">
        <div className="flex flex-wrap items-center gap-4">
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Launch Configuration
          </h2>
          <div className="flex items-center gap-2">
            <label
              htmlFor="launch-capital"
              className="text-sm text-slate-500 dark:text-slate-400"
            >
              Initial Capital:
            </label>
            <input
              id="launch-capital"
              type="number"
              value={launchCapital}
              onChange={(e) => setLaunchCapital(e.target.value)}
              min="100"
              step="100"
              className="w-32 rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            />
          </div>
          <p className="text-xs text-slate-500">
            Applied to all runs launched from this page. Saved optimization results do not store the original capital.
          </p>
        </div>
      </div>

      {/* Launch error banner */}
      {launchError && (
        <div className="rounded-lg border border-red-700/50 bg-red-900/20 p-3 text-sm text-red-400">
          {launchError}
        </div>
      )}

      {/* Results table */}
      {data.entries.length === 0 ? (
        <div className="card p-6 text-center text-sm text-slate-500">
          No results were saved for this optimization run.
        </div>
      ) : (
        <div className="card overflow-hidden">
          <DataTable
            columns={buildResultColumns(data.rankBy, launchingRank, handleLaunchRun)}
            data={[...data.entries]}
            keyExtractor={(e) => String(e.rank)}
            emptyMessage="No results"
          />
        </div>
      )}
    </div>
  );
}
