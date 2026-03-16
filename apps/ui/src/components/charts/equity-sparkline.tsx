"use client";

import { Area, AreaChart, ResponsiveContainer } from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SparklinePoint {
  value: number;
}

interface EquitySparklineProps {
  data: SparklinePoint[];
  color?: string;
  height?: number;
}

// ---------------------------------------------------------------------------
// Component
//
// Note: The gradient ID "sparklineGradient" is intentionally distinct from
// the "equityGradient" / "drawdownGradient" IDs used in equity-curve.tsx so
// that both charts can coexist on the same page without SVG defs collision.
// ---------------------------------------------------------------------------

export function EquitySparkline({
  data,
  color = "#6366f1",
  height = 48,
}: EquitySparklineProps) {
  if (data.length < 2) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
        <defs>
          <linearGradient id="sparklineGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0.0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.5}
          fill="url(#sparklineGradient)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
