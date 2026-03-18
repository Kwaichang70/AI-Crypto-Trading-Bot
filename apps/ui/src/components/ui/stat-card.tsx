interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string | undefined;
  trend?: "up" | "down" | "neutral";
  isLoading?: boolean;
}

export function StatCard({
  label,
  value,
  subValue,
  trend,
  isLoading = false,
}: StatCardProps) {
  const trendColor =
    trend === "up"
      ? "text-profit"
      : trend === "down"
        ? "text-loss"
        : "text-slate-400";

  if (isLoading) {
    return (
      <div className="card space-y-2">
        <div className="h-3 w-24 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-7 w-32 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
      </div>
    );
  }

  return (
    <div className="card space-y-1">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className={`text-2xl font-semibold tabular-nums ${trendColor}`}>
        {value}
      </p>
      {subValue && (
        <p className="text-xs text-slate-500">{subValue}</p>
      )}
    </div>
  );
}
