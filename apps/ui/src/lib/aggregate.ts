/**
 * Client-side aggregation helpers.
 *
 * Lives in lib/ (not in a page module) because Next.js App Router pages may
 * not export arbitrary named symbols — and lib code is unit-testable.
 */
import type { Trade } from "@/lib/types";

export interface SymbolAgg {
  symbol: string;
  trades: number;
  wins: number;
  netPnl: number;
  fees: number;
}

/**
 * Aggregate a window of trades into per-symbol totals, sorted by net PnL
 * (descending). Non-numeric PnL/fee strings are skipped defensively.
 */
export function aggregateTradesBySymbol(trades: readonly Trade[]): SymbolAgg[] {
  const bySymbol = new Map<string, SymbolAgg>();
  for (const t of trades) {
    const pnl = parseFloat(t.realisedPnl);
    const fees = parseFloat(t.totalFees);
    const agg = bySymbol.get(t.symbol) ?? {
      symbol: t.symbol,
      trades: 0,
      wins: 0,
      netPnl: 0,
      fees: 0,
    };
    agg.trades += 1;
    if (Number.isFinite(pnl)) {
      agg.netPnl += pnl;
      if (pnl > 0) agg.wins += 1;
    }
    if (Number.isFinite(fees)) agg.fees += fees;
    bySymbol.set(t.symbol, agg);
  }
  return [...bySymbol.values()].sort((a, b) => b.netPnl - a.netPnl);
}
