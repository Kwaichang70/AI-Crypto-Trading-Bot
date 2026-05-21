/**
 * apps/ui/src/__tests__/pages/run-detail-profit-factor.test.tsx
 * --------------------------------------------------------------
 * Isolated unit tests for the profit-factor display logic extracted
 * from the RunDetailPage Backtest Performance section.
 *
 * We test the derivation logic directly (not the full page) to avoid
 * the heavy mock burden of the page component (router, fetch calls, etc.).
 *
 * Covers all three semantic states introduced by the M1 backend change:
 *   1. null + !infinite  → em dash  (no trades)
 *   2. null +  infinite  → infinity (all winners, zero losses)
 *   3. number            → formatted decimal (normal + all-losers cases)
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/ui/stat-card";

// ---------------------------------------------------------------------------
// Helper: mirrors the inline IIFE logic from page.tsx so tests stay DRY.
// If the page logic changes, update this helper to match.
// ---------------------------------------------------------------------------

function ProfitFactorCard({
  profitFactor,
  profitFactorIsInfinite,
}: {
  profitFactor: number | null;
  profitFactorIsInfinite: boolean;
}) {
  const pf = profitFactor;
  const pfInf = profitFactorIsInfinite;

  const pfDisplay =
    pf === null
      ? pfInf
        ? "∞"
        : "—"
      : pf.toFixed(2);

  const pfTrend: "up" | "down" | "neutral" =
    pf === null
      ? pfInf
        ? "up"
        : "neutral"
      : pf >= 1.5
        ? "up"
        : pf < 1.0
          ? "down"
          : "neutral";

  return <StatCard label="Profit Factor" value={pfDisplay} trend={pfTrend} />;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Profit Factor StatCard render states", () => {
  it("renders em dash when profitFactor is null and not infinite (no trades)", () => {
    render(
      <ProfitFactorCard profitFactor={null} profitFactorIsInfinite={false} />,
    );
    // U+2014 em dash
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders infinity symbol when profitFactor is null and infinite (all winners)", () => {
    render(
      <ProfitFactorCard profitFactor={null} profitFactorIsInfinite={true} />,
    );
    // U+221E infinity
    expect(screen.getByText("∞")).toBeInTheDocument();
  });

  it("renders formatted decimal for a normal profit factor value", () => {
    render(
      <ProfitFactorCard profitFactor={2.35} profitFactorIsInfinite={false} />,
    );
    expect(screen.getByText("2.35")).toBeInTheDocument();
  });

  it("renders 0.00 for all-losers case (profitFactor=0.0)", () => {
    render(
      <ProfitFactorCard profitFactor={0.0} profitFactorIsInfinite={false} />,
    );
    expect(screen.getByText("0.00")).toBeInTheDocument();
  });

  it("applies neutral trend for no-trades case", () => {
    const { container } = render(
      <ProfitFactorCard profitFactor={null} profitFactorIsInfinite={false} />,
    );
    // neutral → text-slate-400
    expect(container.querySelector(".text-slate-400")).toBeInTheDocument();
  });

  it("applies up trend for all-winners case", () => {
    const { container } = render(
      <ProfitFactorCard profitFactor={null} profitFactorIsInfinite={true} />,
    );
    expect(container.querySelector(".text-profit")).toBeInTheDocument();
  });

  it("applies down trend for all-losers case (profitFactor=0.0)", () => {
    const { container } = render(
      <ProfitFactorCard profitFactor={0.0} profitFactorIsInfinite={false} />,
    );
    expect(container.querySelector(".text-loss")).toBeInTheDocument();
  });
});
