"use client";

import type { EquityPoint } from "@/lib/types";

interface EquityCurveChartProps {
  points: readonly EquityPoint[];
  height?: number;
}

export function EquityCurveChart({
  points,
  height = 240,
}: EquityCurveChartProps) {
  if (points.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-slate-800 bg-slate-900/40 text-slate-500 text-sm"
        style={{ height }}
      >
        No equity data yet
      </div>
    );
  }

  const W = 800;
  const H = height;
  const PAD = { top: 12, right: 12, bottom: 28, left: 64 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const equityValues = points.map((p) => parseFloat(p.equity));
  const minEq = Math.min(...equityValues);
  const maxEq = Math.max(...equityValues);
  const range = maxEq - minEq || 1;

  const xScale = (i: number) => PAD.left + (i / (points.length - 1)) * plotW;
  const yScale = (v: number) =>
    PAD.top + plotH - ((v - minEq) / range) * plotH;

  // Build equity polyline
  const equityPoints = points
    .map((p, i) => `${xScale(i).toFixed(1)},${yScale(parseFloat(p.equity)).toFixed(1)}`)
    .join(" ");

  // Build drawdown fill (inverted, red) — scale drawdown % onto the plot
  const maxDd = Math.max(...points.map((p) => p.drawdownPct), 0.001);
  const ddYScale = (pct: number) => PAD.top + (pct / maxDd) * (plotH * 0.3);
  const ddPoints = [
    `${xScale(0).toFixed(1)},${PAD.top}`,
    ...points.map(
      (p, i) =>
        `${xScale(i).toFixed(1)},${ddYScale(p.drawdownPct).toFixed(1)}`,
    ),
    `${xScale(points.length - 1).toFixed(1)},${PAD.top}`,
  ].join(" ");

  // Area fill polygon for equity
  const areaPoints = [
    `${xScale(0).toFixed(1)},${(PAD.top + plotH).toFixed(1)}`,
    equityPoints,
    `${xScale(points.length - 1).toFixed(1)},${(PAD.top + plotH).toFixed(1)}`,
  ].join(" ");

  // Axis labels
  const firstDate = new Date(points[0].timestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  const lastDate = new Date(
    points[points.length - 1].timestamp,
  ).toLocaleDateString("en-US", { month: "short", day: "numeric" });

  const formatK = (v: number) =>
    v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0);

  const midEq = (minEq + maxEq) / 2;

  return (
    <div className="w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-900/40 p-2">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height }}
        aria-label="Equity curve chart"
        role="img"
      >
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = PAD.top + plotH * frac;
          return (
            <line
              key={frac}
              x1={PAD.left}
              y1={y}
              x2={PAD.left + plotW}
              y2={y}
              stroke="#1e293b"
              strokeWidth={1}
            />
          );
        })}

        {/* Y-axis labels */}
        <text x={PAD.left - 6} y={PAD.top + 4} textAnchor="end" fontSize={10} fill="#64748b">
          {formatK(maxEq)}
        </text>
        <text x={PAD.left - 6} y={PAD.top + plotH / 2 + 4} textAnchor="end" fontSize={10} fill="#64748b">
          {formatK(midEq)}
        </text>
        <text x={PAD.left - 6} y={PAD.top + plotH + 4} textAnchor="end" fontSize={10} fill="#64748b">
          {formatK(minEq)}
        </text>

        {/* Drawdown shaded area */}
        {points.length > 1 && (
          <polygon
            points={ddPoints}
            fill="rgba(239,68,68,0.15)"
          />
        )}

        {/* Equity area fill */}
        {points.length > 1 && (
          <polygon points={areaPoints} fill="rgba(99,102,241,0.08)" />
        )}

        {/* Equity line */}
        {points.length > 1 && (
          <polyline
            points={equityPoints}
            fill="none"
            stroke="#6366f1"
            strokeWidth={1.5}
            strokeLinejoin="round"
          />
        )}

        {/* X-axis date labels */}
        <text
          x={PAD.left}
          y={H - 6}
          fontSize={10}
          fill="#64748b"
        >
          {firstDate}
        </text>
        <text
          x={PAD.left + plotW}
          y={H - 6}
          textAnchor="end"
          fontSize={10}
          fill="#64748b"
        >
          {lastDate}
        </text>
      </svg>

      {/* Legend */}
      <div className="mt-1 flex items-center gap-4 px-2 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-6 rounded bg-indigo-500/60" />
          Equity
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-6 rounded bg-red-500/40" />
          Drawdown
        </span>
      </div>
    </div>
  );
}
