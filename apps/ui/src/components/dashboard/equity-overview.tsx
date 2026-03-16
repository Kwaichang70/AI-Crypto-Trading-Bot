"use client";

import { useEffect, useState } from "react";
import { fetchRuns, fetchEquityCurve } from "@/lib/api";
import { EquitySparkline } from "@/components/charts/equity-sparkline";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SparklinePoint {
  value: number;
}

// ---------------------------------------------------------------------------
// Component
//
// Fetches the most recent completed (stopped) backtest run and renders a
// compact sparkline of its equity curve, with a percentage-return badge.
// This is a client component so the server home page stays cacheable.
// ---------------------------------------------------------------------------

export function EquityOverview() {
  const [sparkData, setSparkData] = useState<SparklinePoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const runsResult = await fetchRuns({ limit: 10, status: "stopped" });
        if (cancelled || !runsResult.ok) {
          setLoading(false);
          return;
        }

        // Prefer a backtest run that has metrics — fall back to any stopped run.
        const items = runsResult.data.items;
        const completedRun =
          items.find((r) => r.runMode === "backtest" && r.backtestMetrics) ??
          items[0];

        if (!completedRun) {
          setLoading(false);
          return;
        }

        const eqResult = await fetchEquityCurve(completedRun.id, 200);
        if (!cancelled && eqResult.ok && eqResult.data.points.length > 1) {
          setSparkData(
            eqResult.data.points.map((p) => ({ value: parseFloat(p.equity) })),
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <div className="mt-3 h-12 animate-pulse rounded bg-slate-800" />;
  }

  if (sparkData.length < 2) {
    return (
      <p className="mt-2 text-xs text-slate-600">
        Run a backtest to see equity trend
      </p>
    );
  }

  const first = sparkData[0].value;
  const last = sparkData[sparkData.length - 1].value;
  const trendColor = last >= first ? "#10b981" : "#ef4444";
  const returnPct = first !== 0 ? ((last - first) / first) * 100 : 0;
  const isPositive = returnPct >= 0;

  return (
    <div className="mt-3">
      <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
        <span>Latest Run Equity</span>
        <span className={isPositive ? "text-emerald-400" : "text-red-400"}>
          {isPositive ? "+" : ""}
          {returnPct.toFixed(2)}%
        </span>
      </div>
      <EquitySparkline data={sparkData} color={trendColor} height={48} />
    </div>
  );
}
