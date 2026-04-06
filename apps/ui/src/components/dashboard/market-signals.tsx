"use client";

/**
 * MarketSignals — client component that displays the current cached values
 * from all market signal sources (Fear & Greed, CoinGecko, FRED, Whale Alert).
 *
 * Polls the unauthenticated GET /api/v1/signals/current endpoint every 60 s
 * so the dashboard stays reasonably fresh without hammering backend caches.
 *
 * Design notes
 * ------------
 * - All fields can be null when a source is not configured or not yet cached.
 * - Loading state renders animated skeleton placeholders (no layout shift).
 * - Color semantics follow financial convention: green = risk-on / bullish,
 *   red = risk-off / bearish, amber = caution, blue = informational.
 * - Whale net flow: negative = outflow from exchanges (accumulation) = green;
 *   positive = inflow to exchanges (sell pressure) = red.
 * - Fear & Greed: extreme greed is a contrarian warning shown in red.
 * - Yield curve: inversion (negative spread) = recession signal = red.
 */

import { useEffect, useState } from "react";
import { fetchMarketSignals } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SignalData {
  fearGreedIndex: number | null;
  fearGreedClassification: string | null;
  btcDominance: number | null;
  marketCapChange24h: number | null;
  totalVolumeChange24h: number | null;
  fedFundsRate: number | null;
  yieldCurveSpread: number | null;
  whaleNetFlow: number | null;
  whaleTxCount: number | null;
}

type SignalColor = "green" | "red" | "amber" | "blue" | "neutral";

// ---------------------------------------------------------------------------
// SignalCard
// ---------------------------------------------------------------------------

function SignalCard({
  label,
  value,
  unit,
  color = "neutral",
}: {
  readonly label: string;
  readonly value: string | number | null;
  readonly unit?: string;
  readonly color?: SignalColor;
}) {
  const colorMap: Record<SignalColor, string> = {
    green: "text-emerald-600 dark:text-emerald-400",
    red: "text-red-600 dark:text-red-400",
    amber: "text-amber-600 dark:text-amber-400",
    blue: "text-indigo-600 dark:text-indigo-400",
    neutral: "text-slate-700 dark:text-slate-300",
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900">
      <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-1 text-lg font-semibold tabular-nums ${colorMap[color]}`}>
        {value !== null ? `${value}${unit ?? ""}` : "—"}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton row shown during initial load
// ---------------------------------------------------------------------------

function SignalsSkeleton() {
  return (
    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5" aria-busy="true" aria-label="Loading market signals">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="h-[72px] animate-pulse rounded-lg bg-slate-200 dark:bg-slate-800"
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers — determine card colors from numeric values
// ---------------------------------------------------------------------------

function fgiColor(value: number | null): SignalColor {
  if (value === null) return "neutral";
  if (value <= 25) return "red";        // Extreme Fear
  if (value <= 45) return "amber";      // Fear
  if (value <= 55) return "neutral";    // Neutral
  if (value <= 75) return "green";      // Greed
  return "red";                         // Extreme Greed — contrarian warning
}

function changeColor(value: number | null): SignalColor {
  if (value === null) return "neutral";
  return value >= 0 ? "green" : "red";
}

function whaleFlowColor(netFlow: number | null): SignalColor {
  if (netFlow === null) return "neutral";
  // Negative net_flow = outflow from exchanges = accumulation = bullish
  return netFlow < 0 ? "green" : "red";
}

function yieldCurveColor(spread: number | null): SignalColor {
  if (spread === null) return "neutral";
  return spread < 0 ? "red" : "green";  // Inverted = red
}

/** Format a signed number with an explicit + prefix when positive. */
function signedPct(value: number | null): string | null {
  if (value === null) return null;
  return `${value >= 0 ? "+" : ""}${value}`;
}

/** Format whale net flow as millions with sign prefix. */
function whaleFlowLabel(netFlow: number | null): string | null {
  if (netFlow === null) return null;
  const millions = (netFlow / 1_000_000).toFixed(1);
  return `${netFlow >= 0 ? "+" : ""}${millions}M`;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function MarketSignals() {
  const [signals, setSignals] = useState<SignalData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const result = await fetchMarketSignals();
      if (result.ok) {
        setSignals(result.data as unknown as SignalData);
      }
      setIsLoading(false);
    }

    void load();

    // Poll every 60 seconds — signal caches update at 6 h / 30 min / 24 h / 1 h
    // cadence so 60 s is more than frequent enough to pick up new data promptly.
    const interval = setInterval(() => void load(), 60_000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return <SignalsSkeleton />;
  }

  if (!signals) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        Market signal data unavailable.
      </p>
    );
  }

  const fgiLabel = signals.fearGreedClassification
    ? `Fear & Greed (${signals.fearGreedClassification})`
    : "Fear & Greed";

  const whaleLabel = signals.whaleTxCount !== null
    ? `Whale Flow (${signals.whaleTxCount} txs)`
    : "Whale Flow";

  return (
    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5" aria-label="Market signals">
      <SignalCard
        label={fgiLabel}
        value={signals.fearGreedIndex}
        unit="/100"
        color={fgiColor(signals.fearGreedIndex)}
      />
      <SignalCard
        label="BTC Dominance"
        value={signals.btcDominance}
        unit="%"
        color="blue"
      />
      <SignalCard
        label="Market Cap 24h"
        value={signedPct(signals.marketCapChange24h)}
        unit="%"
        color={changeColor(signals.marketCapChange24h)}
      />
      <SignalCard
        label="Yield Curve (10Y-2Y)"
        value={signals.yieldCurveSpread}
        unit="%"
        color={yieldCurveColor(signals.yieldCurveSpread)}
      />
      <SignalCard
        label={whaleLabel}
        value={whaleFlowLabel(signals.whaleNetFlow)}
        color={whaleFlowColor(signals.whaleNetFlow)}
      />
    </div>
  );
}
