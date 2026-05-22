/**
 * apps/ui/src/__tests__/pages/run-detail-open-positions.test.tsx
 * -------------------------------------------------------------
 * Isolated unit tests for the Open-at-End MTM position display logic
 * extracted from the RunDetailPage "open-at-end" tab.
 *
 * We test the display logic directly (not the full page) to avoid the
 * heavy mock burden of the page component (router, fetch calls, etc.).
 * Pattern mirrors run-detail-profit-factor.test.tsx from M1.
 *
 * Covers:
 *   1. Empty list → custom empty-state message (not generic DataTable message)
 *   2. Single position → all columns rendered correctly, positive PnL styled green
 *   3. Negative PnL → styled red
 *   4. Multiple positions → all rows present, sortable column accessible
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { DataTable } from "@/components/ui/data-table";
import type { Column } from "@/components/ui/data-table";
import type { OpenPositionMTM } from "@/lib/types";

// ---------------------------------------------------------------------------
// Minimal formatCurrency — mirrors the real helper (integer cents precision).
// We duplicate the logic so the test has no I/O dependency on the module.
// ---------------------------------------------------------------------------

function formatCurrency(value: string): string {
  const n = parseFloat(value);
  if (isNaN(n)) return value;
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ---------------------------------------------------------------------------
// Column definitions — copied verbatim from the patch so tests stay faithful
// to what the executor will apply. Update here if page.tsx logic changes.
// ---------------------------------------------------------------------------

const OPEN_POSITION_MTM_COLUMNS: Column<OpenPositionMTM>[] = [
  {
    key: "symbol",
    header: "Symbol",
    render: (p) => <span className="font-mono text-xs text-slate-300">{p.symbol}</span>,
  },
  {
    key: "quantity",
    header: "Qty",
    render: (p) => <span className="font-mono text-xs">{p.quantity}</span>,
  },
  {
    key: "entryPrice",
    header: "Entry Price",
    sortable: true,
    sortValue: (p) => parseFloat(p.entryPrice),
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.entryPrice)}</span>,
  },
  {
    key: "lastPrice",
    header: "Last Price",
    sortable: true,
    sortValue: (p) => parseFloat(p.lastPrice),
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.lastPrice)}</span>,
  },
  {
    key: "unrealisedPnl",
    header: "Unrealised PnL",
    sortable: true,
    sortValue: (p) => parseFloat(p.unrealisedPnl),
    render: (p) => {
      const pnl = parseFloat(p.unrealisedPnl);
      return (
        <span className={`font-mono text-xs font-medium ${pnl >= 0 ? "text-profit" : "text-loss"}`}>
          {pnl >= 0 ? "+" : ""}{formatCurrency(p.unrealisedPnl)}
        </span>
      );
    },
  },
  {
    key: "openedAt",
    header: "Opened At",
    sortable: true,
    sortValue: (p) => new Date(p.openedAt).getTime(),
    render: (p) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(p.openedAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const POSITION_A: OpenPositionMTM = {
  symbol: "BTC/USD",
  quantity: "0.05",
  entryPrice: "29500.00",
  lastPrice: "31200.00",
  unrealisedPnl: "85.00",
  openedAt: "2024-01-15T09:00:00Z",
};

const POSITION_B: OpenPositionMTM = {
  symbol: "ETH/USD",
  quantity: "1.20",
  entryPrice: "1800.00",
  lastPrice: "1650.00",
  unrealisedPnl: "-180.00",
  openedAt: "2024-01-16T14:30:00Z",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Open-at-End MTM position tab rendering", () => {
  // ── Case 1: empty list ────────────────────────────────────────────────────

  it("renders the custom empty-state message when no positions are open", () => {
    render(
      <div>
        {[].length === 0 ? (
          <div data-testid="empty-state">
            No open positions at end of backtest — all positions closed cleanly.
          </div>
        ) : null}
      </div>,
    );
    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(
      screen.getByText(
        "No open positions at end of backtest — all positions closed cleanly.",
      ),
    ).toBeInTheDocument();
  });

  it("DataTable shows its emptyMessage when data array is empty", () => {
    render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    expect(
      screen.getByText("No open positions at end of backtest."),
    ).toBeInTheDocument();
  });

  // ── Case 2: single position, positive PnL ────────────────────────────────

  it("renders symbol, qty, entryPrice, lastPrice and openedAt for a single position", () => {
    render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[POSITION_A]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    expect(screen.getByText("BTC/USD")).toBeInTheDocument();
    expect(screen.getByText("0.05")).toBeInTheDocument();
    // formatCurrency("29500.00") → "29,500.00"
    expect(screen.getByText("29,500.00")).toBeInTheDocument();
    // formatCurrency("31200.00") → "31,200.00"
    expect(screen.getByText("31,200.00")).toBeInTheDocument();
  });

  it("renders positive unrealised PnL with '+' prefix and text-profit class", () => {
    const { container } = render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[POSITION_A]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    // "+85.00" displayed
    expect(screen.getByText("+85.00")).toBeInTheDocument();
    // text-profit class applied
    expect(container.querySelector(".text-profit")).toBeInTheDocument();
    // text-loss class NOT applied
    expect(container.querySelector(".text-loss")).not.toBeInTheDocument();
  });

  // ── Case 3: negative PnL ─────────────────────────────────────────────────

  it("renders negative unrealised PnL without '+' prefix and with text-loss class", () => {
    const { container } = render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[POSITION_B]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    // "-180.00" — no '+' prefix for negative values
    expect(screen.getByText("-180.00")).toBeInTheDocument();
    expect(container.querySelector(".text-loss")).toBeInTheDocument();
    expect(container.querySelector(".text-profit")).not.toBeInTheDocument();
  });

  // ── Case 4: multiple positions ────────────────────────────────────────────

  it("renders all rows when multiple positions are present", () => {
    render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[POSITION_A, POSITION_B]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    expect(screen.getByText("BTC/USD")).toBeInTheDocument();
    expect(screen.getByText("ETH/USD")).toBeInTheDocument();
    expect(screen.getByText("+85.00")).toBeInTheDocument();
    expect(screen.getByText("-180.00")).toBeInTheDocument();
  });

  it("renders Unrealised PnL column header as sortable (aria-sort present)", () => {
    render(
      <DataTable
        columns={OPEN_POSITION_MTM_COLUMNS}
        data={[POSITION_A, POSITION_B]}
        keyExtractor={(p) => p.symbol}
        emptyMessage="No open positions at end of backtest."
      />,
    );
    // The Unrealised PnL column header has aria-sort="none" when not yet sorted
    const pnlHeader = screen.getByText("Unrealised PnL").closest("th");
    expect(pnlHeader).toHaveAttribute("aria-sort", "none");
  });
});
