"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPoint } from "@/lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EquityCurveChartProps {
  points: readonly EquityPoint[];
  height?: number;
  showDrawdown?: boolean;
}

interface ChartDatum {
  timestamp: number;
  equity: number;
  drawdown: number;
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatDate(ts: number): string {
  return new Date(ts).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function formatCurrency(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

interface TooltipPayloadEntry {
  name: string;
  value: number;
  dataKey: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: number;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0 || label === undefined) {
    return null;
  }

  const equityEntry = payload.find((p) => p.dataKey === "equity");
  const drawdownEntry = payload.find((p) => p.dataKey === "drawdown");

  return (
    <div
      style={{
        backgroundColor: "#0f172a",
        border: "1px solid #334155",
        borderRadius: "8px",
        padding: "8px 12px",
        fontSize: "12px",
        lineHeight: "1.6",
      }}
    >
      <p style={{ color: "#94a3b8", marginBottom: "4px" }}>
        {new Date(label).toLocaleString()}
      </p>
      {equityEntry && (
        <p style={{ color: "#6366f1", margin: 0 }}>
          Equity: <strong>${equityEntry.value.toFixed(2)}</strong>
        </p>
      )}
      {drawdownEntry && (
        <p style={{ color: "#ef4444", margin: 0 }}>
          Drawdown: <strong>{drawdownEntry.value.toFixed(2)}%</strong>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function EquityCurveChart({
  points,
  height = 280,
  showDrawdown = true,
}: EquityCurveChartProps) {
  if (points.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-400 text-sm dark:border-slate-800 dark:bg-slate-900/40 dark:text-slate-500"
        style={{ height }}
      >
        No equity data yet
      </div>
    );
  }

  const data: ChartDatum[] = points.map((p) => ({
    timestamp: new Date(p.timestamp).getTime(),
    equity: parseFloat(p.equity),
    // Convert fraction (0.12) to percentage (12) for right-axis display
    drawdown: p.drawdownPct * 100,
  }));

  return (
    <div className="w-full rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900/40">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={data}
          margin={{ top: 8, right: showDrawdown ? 48 : 8, bottom: 0, left: 8 }}
        >
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
            </linearGradient>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e293b"
            vertical={false}
          />

          <XAxis
            dataKey="timestamp"
            tickFormatter={formatDate}
            stroke="#475569"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            minTickGap={60}
          />

          {/* Left axis — equity in dollars */}
          <YAxis
            yAxisId="left"
            tickFormatter={formatCurrency}
            stroke="#475569"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            width={58}
          />

          {/* Right axis — drawdown percentage (only rendered when showDrawdown is true) */}
          {showDrawdown && (
            <YAxis
              yAxisId="right"
              orientation="right"
              tickFormatter={(v: number) => `${v.toFixed(1)}%`}
              stroke="#475569"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              width={42}
            />
          )}

          <Tooltip content={<CustomTooltip />} />

          {/* Equity area */}
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="equity"
            stroke="#6366f1"
            strokeWidth={2}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#6366f1", stroke: "#0f172a", strokeWidth: 2 }}
          />

          {/* Drawdown area — overlaid on right axis */}
          {showDrawdown && (
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="drawdown"
              stroke="#ef4444"
              strokeWidth={1}
              fill="url(#drawdownGradient)"
              dot={false}
              activeDot={{ r: 3, fill: "#ef4444", stroke: "#0f172a", strokeWidth: 2 }}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-1 flex items-center gap-4 px-1 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-6 rounded bg-indigo-500/60" />
          Equity
        </span>
        {showDrawdown && (
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-6 rounded bg-red-500/40" />
            Drawdown
          </span>
        )}
      </div>
    </div>
  );
}
