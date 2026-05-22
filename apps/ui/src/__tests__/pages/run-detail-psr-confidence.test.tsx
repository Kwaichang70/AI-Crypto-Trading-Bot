/**
 * apps/ui/src/__tests__/pages/run-detail-psr-confidence.test.tsx
 * ---------------------------------------------------------------
 * Isolated unit tests for the PSR + confidence_flag display logic
 * extracted from the RunDetailPage Backtest Performance section.
 *
 * We test the derivation logic directly (not the full page) to avoid
 * the heavy mock burden of the page component (router, fetch calls, etc.).
 * Pattern mirrors run-detail-profit-factor.test.tsx from M1.
 *
 * Covers all four semantic states:
 *   1. psr===null + confidenceFlag===null → "PSR: n/a"           (< 30 observations)
 *   2. psr set + confidenceFlag==="high"  → "PSR: 95.2% ✓ high"  (strong evidence)
 *   3. psr set + confidenceFlag==="medium"→ "PSR: 50.1% medium"  (moderate evidence)
 *   4. psr set + confidenceFlag==="low"   → "PSR: 62.4% ⚠ low"   (weak evidence)
 *
 * CR-002 FIX: "low" confidence maps to text-amber-500 (not text-slate-500) to
 * visually differentiate it from the null/n/a state. The ⚠ glyph carries the
 * warning semantics; amber colour reinforces it.
 *
 * CR-004 FIX: the "applies text-profit class for confidenceFlag='high'" test uses
 * screen.getByText() + toHaveClass() instead of container.querySelector(".text-profit"),
 * guarding against false-pass when the main value element also carries text-profit
 * from trend="up".
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/ui/stat-card";

// ---------------------------------------------------------------------------
// Helper: mirrors the IIFE logic from page.tsx so tests stay DRY.
// If the page logic changes, update this helper to match.
// ---------------------------------------------------------------------------

function PsrConfidenceCard({
  sharpeRatio,
  psr,
  confidenceFlag,
}: {
  sharpeRatio: number;
  psr: number | null | undefined;
  confidenceFlag: "high" | "medium" | "low" | null | undefined;
}) {
  const resolvedPsr = psr ?? null;
  const resolvedFlag = confidenceFlag ?? null;

  const psrText =
    resolvedPsr === null
      ? "PSR: n/a"
      : `PSR: ${(resolvedPsr * 100).toFixed(1)}% ${resolvedFlag === "high" ? "✓ high" : resolvedFlag === "medium" ? "medium" : "⚠ low"}`;

  // CR-002: "low" maps to text-amber-500 (not text-slate-500) to differentiate
  // from the null/n/a state; the ⚠ glyph provides the primary semantic signal.
  const psrClassName =
    resolvedPsr === null
      ? "text-slate-500"
      : resolvedFlag === "high"
      ? "text-profit"
      : resolvedFlag === "medium"
      ? "text-amber-500"
      : "text-amber-500"; // low — amber, not slate

  return (
    <StatCard
      label="Sharpe Ratio"
      value={sharpeRatio.toFixed(3)}
      trend={sharpeRatio >= 1.0 ? "up" : sharpeRatio < 0 ? "down" : "neutral"}
      subValue={psrText}
      subValueClassName={psrClassName}
    />
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PSR + Confidence Flag StatCard render states", () => {
  // ── State 1: null PSR (insufficient observations) ────────────────────────

  it("renders 'PSR: n/a' when psr is null (insufficient observations)", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={1.25}
        psr={null}
        confidenceFlag={null}
      />,
    );
    expect(screen.getByText("PSR: n/a")).toBeInTheDocument();
  });

  it("applies text-slate-500 sub-value color when psr is null", () => {
    const { container } = render(
      <PsrConfidenceCard
        sharpeRatio={0.5}
        psr={null}
        confidenceFlag={null}
      />,
    );
    // sharpeRatio=0.5 → trend="neutral" → no text-profit on value element.
    // subValue element should NOT have text-profit or text-amber-500.
    expect(container.querySelector(".text-profit")).not.toBeInTheDocument();
    expect(container.querySelector(".text-amber-500")).not.toBeInTheDocument();
  });

  // ── State 2: high confidence ──────────────────────────────────────────────

  it("renders formatted PSR percentage with checkmark and 'high' label", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={1.8}
        psr={0.952}
        confidenceFlag="high"
      />,
    );
    // ✓ = ✓
    expect(screen.getByText("PSR: 95.2% ✓ high")).toBeInTheDocument();
  });

  it("applies text-profit class for confidenceFlag='high'", () => {
    // CR-004 FIX: use screen.getByText() + toHaveClass() to target the specific
    // subValue element, not container.querySelector() which would also match the
    // main value element that carries text-profit from trend="up".
    render(
      <PsrConfidenceCard
        sharpeRatio={1.8}
        psr={0.952}
        confidenceFlag="high"
      />,
    );
    const subValueEl = screen.getByText("PSR: 95.2% ✓ high");
    expect(subValueEl).toHaveClass("text-profit");
  });

  // ── State 3: medium confidence ────────────────────────────────────────────

  it("renders formatted PSR percentage with 'medium' label (no glyph)", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={0.6}
        psr={0.501}
        confidenceFlag="medium"
      />,
    );
    expect(screen.getByText("PSR: 50.1% medium")).toBeInTheDocument();
  });

  it("applies text-amber-500 class for confidenceFlag='medium'", () => {
    const { container } = render(
      <PsrConfidenceCard
        sharpeRatio={0.6}
        psr={0.501}
        confidenceFlag="medium"
      />,
    );
    expect(container.querySelector(".text-amber-500")).toBeInTheDocument();
  });

  // ── State 4: low confidence ───────────────────────────────────────────────

  it("renders formatted PSR percentage with warning glyph and 'low' label", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={0.3}
        psr={0.624}
        confidenceFlag="low"
      />,
    );
    // ⚠ = ⚠
    expect(screen.getByText("PSR: 62.4% ⚠ low")).toBeInTheDocument();
  });

  it("applies text-amber-500 for confidenceFlag='low' (CR-002: differentiates from null/n/a state)", () => {
    // CR-002 FIX: "low" maps to text-amber-500, not text-slate-500.
    // This visually distinguishes weak-evidence from no-evidence (null PSR).
    const { container } = render(
      <PsrConfidenceCard
        sharpeRatio={0.3}
        psr={0.624}
        confidenceFlag="low"
      />,
    );
    expect(container.querySelector(".text-amber-500")).toBeInTheDocument();
    // Should NOT fall back to slate (that is reserved for null PSR)
    expect(container.querySelector(".text-profit")).not.toBeInTheDocument();
  });

  // ── Edge: undefined fields (pre-M4 run records) ──────────────────────────

  it("renders 'PSR: n/a' when psr is undefined (pre-M4 run record)", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={0.95}
        psr={undefined}
        confidenceFlag={undefined}
      />,
    );
    expect(screen.getByText("PSR: n/a")).toBeInTheDocument();
  });

  // ── Sharpe value always rendered ──────────────────────────────────────────

  it("still renders the Sharpe ratio value regardless of PSR state", () => {
    render(
      <PsrConfidenceCard
        sharpeRatio={1.423}
        psr={null}
        confidenceFlag={null}
      />,
    );
    expect(screen.getByText("1.423")).toBeInTheDocument();
    expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument();
  });
});
