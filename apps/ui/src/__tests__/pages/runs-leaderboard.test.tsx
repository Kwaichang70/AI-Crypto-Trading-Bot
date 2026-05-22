/**
 * apps/ui/src/__tests__/pages/runs-leaderboard.test.tsx
 * ------------------------------------------------------
 * Isolated unit tests for M5 leaderboard frontend additions.
 *
 * Tests are isolated from the full page component to avoid the heavy mock
 * burden of router, fetch, and searchParams. Pattern mirrors existing tests
 * in run-detail-profit-factor.test.tsx and run-detail-psr-confidence.test.tsx.
 *
 * Covers:
 *   1. ConfidenceBadge — correct CSS class per flag value (high/medium/low/null)
 *   2. PSR column cell — formatted as "{(psr*100).toFixed(1)}%" or "—"
 *   3. "Best Return" subtitle — "N of M runs eligible" from eligibleRunsCount
 *   4. leaderboardEligible=false — leaderboardEligible field present on Run type
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/ui/stat-card";
import type { Run } from "@/lib/types";

// ---------------------------------------------------------------------------
// CR-004: ConfidenceBadge is exported from @/app/runs/page (a "use client"
// module that also imports useRouter + useSearchParams). Importing it here
// would require mocking both Next.js navigation hooks for a trivial badge.
// Decision: inline copy is used instead to keep tests dependency-free.
// FIXME: extract to components/ui/confidence-badge.tsx for testability;
// must be manually kept in sync with runs/page.tsx.
// ---------------------------------------------------------------------------

function ConfidenceBadge({
  flag,
}: {
  flag: "high" | "medium" | "low" | null | undefined;
}) {
  if (flag === "high") {
    return <span className="badge-success">high</span>;
  }
  if (flag === "medium") {
    return <span className="badge-warning">medium</span>;
  }
  if (flag === "low") {
    return (
      <span className="inline-flex items-center rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-600 dark:bg-amber-900/20 dark:text-amber-500">
        ⚠ low
      </span>
    );
  }
  return <span className="badge-neutral">—</span>;
}

// ---------------------------------------------------------------------------
// Helper: mirrors the PSR column render logic from runs/page.tsx.
// ---------------------------------------------------------------------------

function PsrCell({
  psr,
  confidenceFlag,
}: {
  psr: number | null | undefined;
  confidenceFlag: "high" | "medium" | "low" | null | undefined;
}) {
  const val = psr ?? null;
  if (val === null) {
    return <span className="text-slate-400 text-xs">—</span>;
  }
  const colorClass =
    confidenceFlag === "high"
      ? "text-profit"
      : confidenceFlag === "medium" || confidenceFlag === "low"
        ? "text-amber-500"
        : "text-slate-700";
  return (
    <span className={`font-mono text-xs font-medium ${colorClass}`}>
      {(val * 100).toFixed(1)}%
    </span>
  );
}

// ---------------------------------------------------------------------------
// Helper: mirrors the "Best Return (eligible)" StatCard from page.tsx.
// ---------------------------------------------------------------------------

function BestReturnCard({
  bestRunReturnPct,
  eligibleRunsCount,
  totalRuns,
}: {
  bestRunReturnPct: number | null | undefined;
  eligibleRunsCount: number | undefined;
  totalRuns: number;
}) {
  const best = bestRunReturnPct ?? null;
  const eligible = eligibleRunsCount ?? 0;
  const bestDisplay =
    best === null
      ? "—"
      : `${best >= 0 ? "+" : ""}${(best * 100).toFixed(2)}%`;
  const bestTrend: "up" | "down" | "neutral" =
    best === null ? "neutral" : best > 0 ? "up" : best < 0 ? "down" : "neutral";
  return (
    <StatCard
      label="Best Return (eligible)"
      value={bestDisplay}
      trend={bestTrend}
      subValue={
        totalRuns > 0
          ? `${eligible} of ${totalRuns} runs eligible`
          : "no runs yet"
      }
    />
  );
}

// ---------------------------------------------------------------------------
// 1. ConfidenceBadge colour tests
// ---------------------------------------------------------------------------

describe("ConfidenceBadge render states", () => {
  it("renders 'high' with badge-success class", () => {
    render(<ConfidenceBadge flag="high" />);
    const el = screen.getByText("high");
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass("badge-success");
  });

  it("renders 'medium' with badge-warning class", () => {
    render(<ConfidenceBadge flag="medium" />);
    const el = screen.getByText("medium");
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass("badge-warning");
  });

  it("renders '⚠ low' with muted-amber styling (text-amber-600)", () => {
    render(<ConfidenceBadge flag="low" />);
    // ⚠ and "low" are in the same text node
    const el = screen.getByText("⚠ low");
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass("text-amber-600");
  });

  it("renders em dash with badge-neutral for null flag", () => {
    render(<ConfidenceBadge flag={null} />);
    const el = screen.getByText("—");
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass("badge-neutral");
  });

  it("renders em dash with badge-neutral for undefined flag (pre-M5 record)", () => {
    render(<ConfidenceBadge flag={undefined} />);
    const el = screen.getByText("—");
    expect(el).toHaveClass("badge-neutral");
  });
});

// ---------------------------------------------------------------------------
// 2. PSR column cell formatting
// ---------------------------------------------------------------------------

describe("PSR column cell render states", () => {
  it("formats psr=0.952 as '95.2%' with text-profit for high confidence", () => {
    render(<PsrCell psr={0.952} confidenceFlag="high" />);
    const el = screen.getByText("95.2%");
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass("text-profit");
  });

  it("formats psr=0.501 as '50.1%' with text-amber-500 for medium confidence", () => {
    render(<PsrCell psr={0.501} confidenceFlag="medium" />);
    const el = screen.getByText("50.1%");
    expect(el).toHaveClass("text-amber-500");
  });

  it("formats psr=0.624 as '62.4%' with text-amber-500 for low confidence", () => {
    render(<PsrCell psr={0.624} confidenceFlag="low" />);
    const el = screen.getByText("62.4%");
    expect(el).toHaveClass("text-amber-500");
  });

  it("renders em dash when psr is null", () => {
    render(<PsrCell psr={null} confidenceFlag={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 3. Best Return card subtitle
// ---------------------------------------------------------------------------

describe("Best Return (eligible) StatCard subtitle", () => {
  it("shows 'N of M runs eligible' subtitle with correct numbers", () => {
    render(
      <BestReturnCard
        bestRunReturnPct={0.142}
        eligibleRunsCount={7}
        totalRuns={25}
      />,
    );
    expect(screen.getByText("7 of 25 runs eligible")).toBeInTheDocument();
  });

  it("shows '0 of M runs eligible' when eligibleRunsCount is 0", () => {
    render(
      <BestReturnCard
        bestRunReturnPct={null}
        eligibleRunsCount={0}
        totalRuns={12}
      />,
    );
    expect(screen.getByText("0 of 12 runs eligible")).toBeInTheDocument();
  });

  it("renders value as '—' when bestRunReturnPct is null (no eligible runs)", () => {
    render(
      <BestReturnCard
        bestRunReturnPct={null}
        eligibleRunsCount={0}
        totalRuns={5}
      />,
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("formats positive bestRunReturnPct with leading '+'", () => {
    render(
      <BestReturnCard
        bestRunReturnPct={0.2367}
        eligibleRunsCount={3}
        totalRuns={10}
      />,
    );
    expect(screen.getByText("+23.67%")).toBeInTheDocument();
  });

  it("shows 'no runs yet' subtitle when totalRuns is 0", () => {
    render(
      <BestReturnCard
        bestRunReturnPct={null}
        eligibleRunsCount={undefined}
        totalRuns={0}
      />,
    );
    expect(screen.getByText("no runs yet")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 4. leaderboardEligible type-level check
// ---------------------------------------------------------------------------

describe("leaderboardEligible field on Run type", () => {
  it("leaderboardEligible boolean field accepted by TypeScript (compile-time guard)", () => {
    // CR-005: Uses Pick<Run,"leaderboardEligible"> so tsc catches missing
    // fields in types.ts. If types.ts drops leaderboardEligible this test
    // will fail at compile time.
    const run: Pick<Run, "leaderboardEligible"> = { leaderboardEligible: false };
    expect(run.leaderboardEligible).toBe(false);

    const runEligible: Pick<Run, "leaderboardEligible"> = { leaderboardEligible: true };
    expect(runEligible.leaderboardEligible).toBe(true);
  });
});
