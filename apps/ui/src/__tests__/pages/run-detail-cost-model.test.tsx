/**
 * WP21-T-35 (run detail): the live run's cost card says the fees are actual
 * exchange fills and the profile is only an estimate; backtest / legacy runs
 * show their model.
 */

import React from "react";
import { render, screen, act } from "@testing-library/react";
import { useSession } from "next-auth/react";
import type { Run } from "@/lib/types";
import { makeRun, ADV1, LEGACY_RT } from "../helpers/cost-profile-fixtures";

jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "11111111-1111-4111-8111-111111111111" }),
  useRouter: () => ({ push: jest.fn() }),
}));
jest.mock("next-auth/react", () => ({ useSession: jest.fn() }));

const mockFetchRun = jest.fn();
const ok = (data: unknown) => Promise.resolve({ ok: true, data });

jest.mock("@/lib/api", () => ({
  fetchRun: (...a: unknown[]) => mockFetchRun(...a),
  fetchPortfolio: () => Promise.resolve({ ok: false, error: { status: 404, message: "n/a" } }),
  fetchEquityCurve: () => Promise.resolve({ ok: true, data: { runId: "x", totalPoints: 0, points: [] } }),
  fetchTrades: () => Promise.resolve({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } }),
  fetchOrders: () => Promise.resolve({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } }),
  fetchFills: () => Promise.resolve({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } }),
  fetchPositions: () => Promise.resolve({ ok: true, data: { runId: "x", positions: [], count: 0 } }),
  fetchLearningState: () => Promise.resolve({ ok: false, error: { status: 404, message: "n/a" } }),
  fetchDiagnostics: () => Promise.resolve({ ok: false, error: { status: 404, message: "n/a" } }),
  archiveRun: jest.fn(),
  stopRun: jest.fn(),
  formatCurrency: (v: string) => v,
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

import RunDetailPage from "@/app/runs/[id]/page";

async function renderWith(run: Run) {
  mockFetchRun.mockImplementation(() => ok(run));
  render(<RunDetailPage />);
  await act(async () => {
    await jest.advanceTimersByTimeAsync(0);
  });
}

beforeEach(() => {
  jest.useFakeTimers();
  (useSession as jest.Mock).mockReturnValue({ data: { user: { role: "admin" } }, status: "authenticated" });
});
afterEach(() => jest.useRealTimers());

describe("Run detail cost model card", () => {
  it("live: 'fees: actual exchange fills; estimate profile X'", async () => {
    await renderWith(
      makeRun({
        id: "11111111-1111-4111-8111-111111111111",
        mode: "live",
        costProfile: { ...ADV1, feeSource: "exchange_fills" },
        legacyCosts: false,
        metrics: false,
      }),
    );
    const note = screen.getByTestId("cost-model-live-note");
    expect(note).toHaveTextContent("fees: actual exchange fills; estimate profile coinbase_advanced_1");
    expect(screen.getByText("exchange fills (estimate only)")).toBeInTheDocument();
    expect(screen.getAllByTestId("cost-profile-badge")[0]).toHaveTextContent("coinbase_advanced_1");
  });

  it("backtest: shows the profile bps and fee source 'model'", async () => {
    await renderWith(
      makeRun({
        id: "11111111-1111-4111-8111-111111111111",
        costProfile: ADV1,
        legacyCosts: false,
        metrics: false,
      }),
    );
    const card = screen.getByTestId("cost-model-card");
    expect(card).toHaveTextContent("120 bps");
    expect(card).toHaveTextContent("260 bps");
    expect(screen.queryByTestId("cost-model-live-note")).not.toBeInTheDocument();
    expect(card).toHaveTextContent("model");
  });

  it("legacy / cost-less run: 'legacy costs' badge and the not-comparable note", async () => {
    await renderWith(
      makeRun({
        id: "11111111-1111-4111-8111-111111111111",
        mode: "paper",
        costProfile: LEGACY_RT,
        legacyCosts: true,
        metrics: false,
      }),
    );
    expect(screen.getAllByText("legacy costs").length).toBeGreaterThan(0);
    expect(screen.getByTestId("cost-model-card")).toHaveTextContent("not comparable");
  });

  it("an older API response with no cost fields degrades to legacy without throwing", async () => {
    await renderWith(makeRun({ id: "11111111-1111-4111-8111-111111111111", metrics: false }));
    expect(screen.getAllByText("legacy costs").length).toBeGreaterThan(0);
    expect(screen.getByText("No cost profile reported for this run.")).toBeInTheDocument();
  });
});
