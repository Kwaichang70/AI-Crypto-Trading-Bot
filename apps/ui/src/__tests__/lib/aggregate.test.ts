/**
 * Tests for aggregateTradesBySymbol (lib/aggregate.ts) — the per-symbol PnL
 * breakdown shown on the run-detail Overview tab.
 */
import { aggregateTradesBySymbol } from "@/lib/aggregate";
import type { Trade } from "@/lib/types";

function trade(overrides: Partial<Trade>): Trade {
  return {
    id: "t1",
    runId: "r1",
    symbol: "BTC/EUR",
    side: "buy",
    entryPrice: "100",
    exitPrice: "110",
    quantity: "1",
    realisedPnl: "10",
    totalFees: "0.5",
    entryAt: "2026-01-01T00:00:00Z",
    exitAt: "2026-01-02T00:00:00Z",
    strategyId: "s1",
    ...overrides,
  } as Trade;
}

describe("aggregateTradesBySymbol", () => {
  it("returns empty array for no trades", () => {
    expect(aggregateTradesBySymbol([])).toEqual([]);
  });

  it("aggregates trades, wins, pnl and fees per symbol", () => {
    const rows = aggregateTradesBySymbol([
      trade({ symbol: "BTC/EUR", realisedPnl: "10", totalFees: "1" }),
      trade({ symbol: "BTC/EUR", realisedPnl: "-4", totalFees: "1" }),
      trade({ symbol: "ETH/EUR", realisedPnl: "2", totalFees: "0.25" }),
    ]);
    const btc = rows.find((r) => r.symbol === "BTC/EUR");
    expect(btc).toEqual({
      symbol: "BTC/EUR",
      trades: 2,
      wins: 1,
      netPnl: 6,
      fees: 2,
    });
    const eth = rows.find((r) => r.symbol === "ETH/EUR");
    expect(eth?.wins).toBe(1);
    expect(eth?.netPnl).toBeCloseTo(2);
  });

  it("sorts by net PnL descending", () => {
    const rows = aggregateTradesBySymbol([
      trade({ symbol: "A/EUR", realisedPnl: "-5" }),
      trade({ symbol: "B/EUR", realisedPnl: "20" }),
      trade({ symbol: "C/EUR", realisedPnl: "1" }),
    ]);
    expect(rows.map((r) => r.symbol)).toEqual(["B/EUR", "C/EUR", "A/EUR"]);
  });

  it("skips non-numeric pnl/fees defensively without dropping the trade count", () => {
    const rows = aggregateTradesBySymbol([
      trade({ symbol: "BTC/EUR", realisedPnl: "not-a-number", totalFees: "" }),
      trade({ symbol: "BTC/EUR", realisedPnl: "3", totalFees: "0.1" }),
    ]);
    expect(rows[0]).toEqual({
      symbol: "BTC/EUR",
      trades: 2,
      wins: 1,
      netPnl: 3,
      fees: 0.1,
    });
  });

  it("zero-pnl trades count as trades but not wins", () => {
    const rows = aggregateTradesBySymbol([
      trade({ realisedPnl: "0" }),
    ]);
    expect(rows[0].trades).toBe(1);
    expect(rows[0].wins).toBe(0);
  });
});
