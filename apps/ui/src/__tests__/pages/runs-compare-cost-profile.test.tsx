/**
 * WP21-T-35 (compare): "Cost profile" and "Round-trip cost (bps)" rows, and
 * the mixed-profile banner (different fingerprints, or any legacy run).
 */

import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import type { Run } from "@/lib/types";
import { makeRun, ADV1, D6, LEGACY_BT } from "../helpers/cost-profile-fixtures";

let mockIds = "a,b";
jest.mock("next/navigation", () => ({
  useSearchParams: () => ({ get: (k: string) => (k === "ids" ? mockIds : null) }),
}));

const mockFetchRun = jest.fn();
jest.mock("@/lib/api", () => ({
  fetchRun: (...args: unknown[]) => mockFetchRun(...args),
  fetchPortfolio: () => Promise.resolve({ ok: false, error: { status: 404, message: "n/a" } }),
  fetchEquityCurve: () =>
    Promise.resolve({ ok: true, data: { runId: "x", totalPoints: 0, points: [] } }),
  formatCurrency: (v: string) => v,
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

import RunComparePage from "@/app/runs/compare/page";

const BANNER = "These runs use different or legacy cost models; returns and Sharpe are not directly comparable.";

function load(a: Run, b: Run) {
  mockFetchRun.mockImplementation((id: string) =>
    Promise.resolve({ ok: true, data: id === "a" ? a : b }),
  );
}

function rowCells(label: string): string[] {
  const labelCell = screen.getByText(label, { selector: "td" });
  const tr = labelCell.closest("tr") as HTMLElement;
  return within(tr)
    .getAllByRole("cell")
    .slice(1)
    .map((c) => c.textContent ?? "");
}

beforeEach(() => {
  jest.clearAllMocks();
  mockIds = "a,b";
});

describe("Compare page cost rows and mixed-profile banner", () => {
  it("shows the cost profile and round-trip rows", async () => {
    load(
      makeRun({ id: "a", costProfile: ADV1, legacyCosts: false }),
      makeRun({ id: "b", costProfile: D6, legacyCosts: false }),
    );
    render(<RunComparePage />);
    await screen.findByText("Cost profile", { selector: "td" });

    expect(rowCells("Cost profile")).toEqual(["coinbase_advanced_1", "coinbase_advanced_d6"]);
    expect(rowCells("Round-trip cost (bps)")).toEqual(["260", "140"]);
  });

  it("shows the banner when the profile fingerprints differ", async () => {
    load(
      makeRun({ id: "a", costProfile: ADV1, legacyCosts: false }),
      makeRun({ id: "b", costProfile: D6, legacyCosts: false }),
    );
    render(<RunComparePage />);
    expect(await screen.findByTestId("mixed-cost-banner")).toHaveTextContent(BANNER);
  });

  it("shows the banner when any run is legacy (including a run with no cost fields)", async () => {
    load(
      makeRun({ id: "a", costProfile: ADV1, legacyCosts: false }),
      makeRun({ id: "b", costProfile: LEGACY_BT, legacyCosts: true }),
    );
    const { unmount } = render(<RunComparePage />);
    expect(await screen.findByTestId("mixed-cost-banner")).toBeInTheDocument();
    expect(rowCells("Cost profile")).toEqual(["coinbase_advanced_1", "legacy costs"]);
    unmount();

    load(makeRun({ id: "a", costProfile: ADV1, legacyCosts: false }), makeRun({ id: "b" }));
    render(<RunComparePage />);
    expect(await screen.findByTestId("mixed-cost-banner")).toBeInTheDocument();
  });

  it("shows no banner when every run shares one non-legacy profile", async () => {
    load(
      makeRun({ id: "a", costProfile: ADV1, legacyCosts: false }),
      makeRun({ id: "b", costProfile: ADV1, legacyCosts: false }),
    );
    render(<RunComparePage />);
    await screen.findByText("Cost profile", { selector: "td" });
    await waitFor(() => expect(mockFetchRun).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId("mixed-cost-banner")).not.toBeInTheDocument();
  });

  it("labels a live run's profile as an estimate", async () => {
    load(
      makeRun({ id: "a", mode: "live", costProfile: { ...ADV1, feeSource: "exchange_fills" }, legacyCosts: false }),
      makeRun({ id: "b", costProfile: ADV1, legacyCosts: false }),
    );
    render(<RunComparePage />);
    await screen.findByText("Cost profile", { selector: "td" });
    expect(rowCells("Cost profile")).toEqual(["coinbase_advanced_1 (estimate)", "coinbase_advanced_1"]);
  });
});
