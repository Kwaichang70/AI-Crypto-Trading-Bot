/**
 * WP21-T-35 (runs list): cost badges -- `legacy costs` amber, `stress` /
 * `diagnostic` slate, otherwise the profile id -- and the CSV column.
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { makeRun, ADV1, STRESS80, DIAG, LEGACY_RT } from "../helpers/cost-profile-fixtures";

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn(), replace: jest.fn() }),
  useSearchParams: () => ({ get: () => null }),
}));

const mockFetchRuns = jest.fn();
jest.mock("@/lib/api", () => ({
  fetchRuns: (...args: unknown[]) => mockFetchRuns(...args),
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

const mockExportToCsv = jest.fn();
jest.mock("@/lib/csv-export", () => ({
  exportToCsv: (...args: unknown[]) => mockExportToCsv(...args),
}));

import RunsPage from "@/app/runs/page";

const RUNS = [
  makeRun({ costProfile: ADV1, legacyCosts: false }),
  makeRun({ costProfile: STRESS80, legacyCosts: false }),
  makeRun({ costProfile: DIAG, legacyCosts: false }),
  makeRun({ mode: "paper", costProfile: LEGACY_RT, legacyCosts: true }),
  makeRun({}), // older API: no cost fields -> legacy
];

beforeEach(() => {
  jest.clearAllMocks();
  mockFetchRuns.mockResolvedValue({
    ok: true,
    data: { items: RUNS, total: RUNS.length, offset: 0, limit: 25 },
  });
});

describe("Runs list cost badges", () => {
  it("renders one badge per run with the SY-21-10 label and colour", async () => {
    render(<RunsPage />);
    await waitFor(() => expect(screen.getAllByTestId("cost-profile-badge")).toHaveLength(5));

    const badges = screen.getAllByTestId("cost-profile-badge");
    expect(badges.map((b) => b.textContent)).toEqual([
      "coinbase_advanced_1",
      "stress",
      "diagnostic",
      "legacy costs",
      "legacy costs",
    ]);
    expect(badges[1]).toHaveClass("badge-neutral");
    expect(badges[2]).toHaveClass("badge-neutral");
    expect(badges[3]).toHaveClass("badge-warning");
    expect(badges[4]).toHaveClass("badge-warning");
    // tooltip carries the bps summary
    expect(badges[0]).toHaveAttribute("title", "taker 120 · maker 60 · slip 10 · RT 260 bps");
  });

  it("exports a Cost Profile CSV column", async () => {
    render(<RunsPage />);
    await waitFor(() => expect(screen.getAllByTestId("cost-profile-badge")).toHaveLength(5));
    fireEvent.click(screen.getByRole("button", { name: /Export .* as CSV/ }));
    const [, columns, data] = mockExportToCsv.mock.calls[0] as [
      string,
      { header: string; value: (r: unknown) => unknown }[],
      unknown[],
    ];
    const col = columns.find((c) => c.header === "Cost Profile");
    expect(col).toBeDefined();
    expect(data.map((r) => col?.value(r))).toEqual([
      "coinbase_advanced_1",
      "stress",
      "diagnostic",
      "legacy costs",
      "legacy costs",
    ]);
  });
});
