/**
 * apps/ui/src/__tests__/components/strategy-status-badge.test.tsx
 * ----------------------------------------------------------------
 * Sprint 51 Cycle 2 — strategy-availability lockdown UI.
 *
 * Covers:
 *   1. StrategyStatusBadge — renders "Backtest only" for demoted, "Experimental"
 *      for experimental, "Live-ready" for active, and NOTHING when status is
 *      undefined (graceful degrade for pre-lockdown API responses).
 *   2. StrategyCard (exported from components/strategies/strategy-card.tsx) — renders the demoted
 *      banner + demotionReason + promotion requirements for a demoted strategy;
 *      graceful-degrades (treated as active, no demoted UI) when the lockdown
 *      fields are absent.
 *
 * TEST-S51C2-400 .. 409
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StrategyStatusBadge } from "@/components/ui/strategy-status-badge";
import { StrategyCard } from "@/components/strategies/strategy-card";
import type { Strategy } from "@/lib/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMPTY_SCHEMA = {
  type: "object" as const,
  properties: {},
};

function makeStrategy(overrides: Partial<Strategy> = {}): Strategy {
  return {
    name: "ma_crossover",
    displayName: "MA Crossover",
    version: "1.0.0",
    description: "Moving-average crossover.",
    tags: ["trend"],
    parameterSchema: EMPTY_SCHEMA,
    ...overrides,
  };
}

// ===========================================================================
// 1. StrategyStatusBadge
// ===========================================================================

describe("StrategyStatusBadge", () => {
  it("TEST-S51C2-400: renders 'Backtest only' for demoted status", () => {
    render(<StrategyStatusBadge status="demoted" />);
    expect(screen.getByText("Backtest only")).toBeInTheDocument();
  });

  it("TEST-S51C2-401: renders 'Experimental' for experimental status", () => {
    render(<StrategyStatusBadge status="experimental" />);
    expect(screen.getByText("Experimental")).toBeInTheDocument();
  });

  it("TEST-S51C2-402: renders 'Live-ready' for active status", () => {
    render(<StrategyStatusBadge status="active" />);
    expect(screen.getByText("Live-ready")).toBeInTheDocument();
  });

  it("TEST-S51C2-403: renders nothing when status is undefined (graceful degrade)", () => {
    const { container } = render(<StrategyStatusBadge status={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });
});

// ===========================================================================
// 2. StrategyCard — demoted banner + graceful degrade
// ===========================================================================

describe("StrategyCard (lockdown UI)", () => {
  it("TEST-S51C2-404: renders the 'Backtest only' badge for a demoted strategy", () => {
    render(
      <StrategyCard
        strategy={makeStrategy({
          status: "demoted",
          demotionReason: "Underperformance in non-trending regime.",
        })}
      />,
    );
    expect(screen.getByText("Backtest only")).toBeInTheDocument();
  });

  it("TEST-S51C2-405: renders the demotionReason text for a demoted strategy", () => {
    const reason = "Underperformance in non-trending regime.";
    render(
      <StrategyCard
        strategy={makeStrategy({ status: "demoted", demotionReason: reason })}
      />,
    );
    expect(screen.getByText(reason)).toBeInTheDocument();
  });

  it("TEST-S51C2-406: renders promotion requirements for a demoted strategy", () => {
    render(
      <StrategyCard
        strategy={makeStrategy({
          status: "demoted",
          demotionReason: "x",
          promotionRequirements: ["oos_walk_forward_pass", "regime_filter_added"],
        })}
      />,
    );
    expect(screen.getByText("To re-promote:")).toBeInTheDocument();
    expect(screen.getByText("oos_walk_forward_pass")).toBeInTheDocument();
    expect(screen.getByText("regime_filter_added")).toBeInTheDocument();
  });

  it("TEST-S51C2-407: active strategy shows 'Live-ready' and no demoted banner", () => {
    render(
      <StrategyCard
        strategy={makeStrategy({
          name: "grid_trading",
          displayName: "Grid Trading",
          status: "active",
        })}
      />,
    );
    expect(screen.getByText("Live-ready")).toBeInTheDocument();
    expect(screen.queryByText("To re-promote:")).not.toBeInTheDocument();
  });

  it("TEST-S51C2-408: graceful degrade — no status field renders no badge and no demoted UI", () => {
    render(<StrategyCard strategy={makeStrategy({ status: undefined })} />);
    // No badge variants present.
    expect(screen.queryByText("Backtest only")).not.toBeInTheDocument();
    expect(screen.queryByText("Live-ready")).not.toBeInTheDocument();
    expect(screen.queryByText("Experimental")).not.toBeInTheDocument();
    // No demoted-only UI.
    expect(screen.queryByText("To re-promote:")).not.toBeInTheDocument();
  });

  it("TEST-S51C2-409: demoted strategy without reason still does not render an empty reason node", () => {
    render(
      <StrategyCard
        strategy={makeStrategy({ status: "demoted", demotionReason: null })}
      />,
    );
    // Badge still present, but no reason paragraph (demotionReason is null).
    expect(screen.getByText("Backtest only")).toBeInTheDocument();
  });
});
