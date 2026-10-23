/**
 * WP21-T-35 (leaderboard): legacy, stress and diagnostic runs are hidden by
 * default ("Live-equivalent costs"); the "All cost models" toggle shows them
 * with labels; rows group by strategy x profile; the CSV has a profile column.
 */

import React from "react";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { makeRun, ADV1, D6, STRESS80, DIAG, LEGACY_BT, LEGACY_RT } from "../helpers/cost-profile-fixtures";

const mockFetchRuns = jest.fn();
jest.mock("@/lib/api", () => ({
  fetchRuns: (...args: unknown[]) => mockFetchRuns(...args),
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

const mockExportToCsv = jest.fn();
jest.mock("@/lib/csv-export", () => ({
  exportToCsv: (...args: unknown[]) => mockExportToCsv(...args),
}));

// Imported AFTER the mocks are registered.
import LeaderboardPage from "@/app/leaderboard/page";

const RUNS = [
  makeRun({ strategy: "rsi_mean_reversion", costProfile: ADV1, legacyCosts: false, returnPct: 0.1 }),
  makeRun({ strategy: "rsi_mean_reversion", costProfile: ADV1, legacyCosts: false, returnPct: 0.3 }),
  makeRun({ strategy: "rsi_mean_reversion", costProfile: D6, legacyCosts: false }),
  makeRun({ strategy: "grid_trading", costProfile: STRESS80, legacyCosts: false }),
  makeRun({ strategy: "grid_trading", costProfile: DIAG, legacyCosts: false }),
  makeRun({ strategy: "grid_trading", costProfile: LEGACY_BT, legacyCosts: true }),
  makeRun({ strategy: "grid_trading", costProfile: LEGACY_RT, legacyCosts: true }),
  // older API: no cost fields at all
  makeRun({ strategy: "grid_trading" }),
  makeRun({ strategy: "grid_trading", costProfile: ADV1, legacyCosts: false }),
];

function rows(): HTMLElement[] {
  return within(screen.getAllByRole("table")[0] as HTMLElement)
    .getAllByRole("row")
    .slice(1);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockFetchRuns.mockResolvedValue({
    ok: true,
    data: { items: RUNS, total: RUNS.length, offset: 0, limit: 500 },
  });
});

describe("Leaderboard cost-model filter (SY-21-10)", () => {
  it("defaults to live-equivalent costs: legacy, stress, diagnostic and cost-less runs are hidden", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows().length).toBeGreaterThan(0));

    // rsi x adv1, rsi x d6, grid x adv1
    expect(rows()).toHaveLength(3);
    expect(screen.queryByText("legacy costs")).not.toBeInTheDocument();
    expect(screen.queryByText("stress")).not.toBeInTheDocument();
    expect(screen.queryByText("diagnostic")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Live-equivalent costs" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    // 5 runs are hidden: stress, diagnostic, 2 legacy ids, 1 cost-less
    expect(screen.getByTestId("leaderboard-hidden-count")).toHaveTextContent("5 runs");
  });

  it("'All cost models' shows the hidden runs with their labels", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(3));

    fireEvent.click(screen.getByRole("button", { name: "All cost models" }));

    const badges = screen.getAllByTestId("cost-profile-badge");
    const byKind = (k: string) => badges.filter((b) => b.getAttribute("data-kind") === k);
    expect(byKind("legacy")).toHaveLength(1); // both legacy ids and the cost-less run share ONE group
    expect(byKind("legacy")[0]).toHaveTextContent("legacy costs");
    expect(byKind("legacy")[0]).toHaveClass("badge-warning"); // amber
    expect(byKind("stress")[0]).toHaveTextContent("stress");
    expect(byKind("stress")[0]).toHaveClass("badge-neutral"); // slate
    expect(byKind("diagnostic")[0]).toHaveTextContent("diagnostic");
    expect(byKind("diagnostic")[0]).toHaveClass("badge-neutral");
    expect(byKind("profile").map((b) => b.textContent).sort()).toEqual([
      "coinbase_advanced_1",
      "coinbase_advanced_1",
      "coinbase_advanced_d6",
    ]);
    expect(screen.queryByTestId("leaderboard-hidden-count")).not.toBeInTheDocument();
  });

  it("groups by strategy x profile, and all legacy ids / cost-less runs form one 'legacy' group", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: "All cost models" }));

    // rsi: adv1 (2 runs), d6 (1)   grid: adv1, stress80, diag, legacy (3 runs)
    expect(rows()).toHaveLength(6);

    const rsiAdv1 = rows().find(
      (r) => within(r).queryByText("rsi_mean_reversion") && within(r).queryByText("coinbase_advanced_1"),
    ) as HTMLElement;
    expect(rsiAdv1).toBeDefined();
    // Runs column: 2 runs in the rsi x adv1 group
    expect(within(rsiAdv1).getByText("2")).toBeInTheDocument();

    const gridLegacy = rows().find(
      (r) => within(r).queryByText("grid_trading") && within(r).queryByText("legacy costs"),
    ) as HTMLElement;
    expect(within(gridLegacy).getByText("3")).toBeInTheDocument();
  });

  it("shows the profile id next to the stress / diagnostic labels", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: "All cost models" }));
    expect(screen.getByText("coinbase_stress_t80")).toBeInTheDocument();
    expect(screen.getByText("frictionless_diagnostic")).toBeInTheDocument();
  });

  it("has a 'Cost profile' column header", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows().length).toBeGreaterThan(0));
    expect(screen.getByRole("columnheader", { name: "Cost profile" })).toBeInTheDocument();
  });

  it("CSV export has a Cost Profile column carrying the group's profile id", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: "All cost models" }));

    fireEvent.click(screen.getByRole("button", { name: /Export strategy-leaderboard\.csv as CSV/ }));
    expect(mockExportToCsv).toHaveBeenCalledTimes(1);
    const [filename, columns, data] = mockExportToCsv.mock.calls[0] as [
      string,
      { header: string; value: (row: unknown) => unknown }[],
      unknown[],
    ];
    expect(filename).toBe("strategy-leaderboard.csv");
    const costCol = columns.find((c) => c.header === "Cost Profile");
    expect(costCol).toBeDefined();
    const values = data.map((row) => costCol?.value(row)).sort();
    expect(values).toEqual(
      [
        "coinbase_advanced_1",
        "coinbase_advanced_1",
        "coinbase_advanced_d6",
        "coinbase_stress_t80",
        "frictionless_diagnostic",
        "legacy",
      ].sort(),
    );
  });

  it("default-filter CSV only contains live-equivalent groups", async () => {
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: /Export strategy-leaderboard\.csv as CSV/ }));
    const [, columns, data] = mockExportToCsv.mock.calls[0] as [
      string,
      { header: string; value: (row: unknown) => unknown }[],
      unknown[],
    ];
    const costCol = columns.find((c) => c.header === "Cost Profile");
    expect(data.map((row) => costCol?.value(row)).sort()).toEqual([
      "coinbase_advanced_1",
      "coinbase_advanced_1",
      "coinbase_advanced_d6",
    ]);
  });

  it("uses the server liveEquivalentCosts flag when the API sends it", async () => {
    mockFetchRuns.mockResolvedValue({
      ok: true,
      data: {
        items: [
          makeRun({ strategy: "a_strategy", costProfile: ADV1, legacyCosts: false, liveEquivalentCosts: true }),
          // same-looking profile, but the server says no: must stay hidden
          makeRun({ strategy: "b_strategy", costProfile: ADV1, legacyCosts: false, liveEquivalentCosts: false }),
        ],
        total: 2,
        offset: 0,
        limit: 500,
      },
    });
    render(<LeaderboardPage />);
    await waitFor(() => expect(rows()).toHaveLength(1));
    expect(within(rows()[0] as HTMLElement).getByText("a_strategy")).toBeInTheDocument();
    expect(screen.queryByText("b_strategy")).not.toBeInTheDocument();
  });

  it("explains an empty default view when only non-live-equivalent runs exist", async () => {
    mockFetchRuns.mockResolvedValue({
      ok: true,
      data: {
        items: [makeRun({ costProfile: LEGACY_BT, legacyCosts: true }), makeRun({})],
        total: 2,
        offset: 0,
        limit: 500,
      },
    });
    render(<LeaderboardPage />);
    await screen.findByText(/No runs with live-equivalent costs/);
    fireEvent.click(screen.getByRole("button", { name: "All cost models" }));
    await waitFor(() => expect(rows()).toHaveLength(1));
    expect(screen.getByText("legacy costs")).toBeInTheDocument();
  });
});
