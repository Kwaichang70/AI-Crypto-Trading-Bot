/**
 * WP21BD-UI-02 (CR-05): the Duplicate Run SENDER. `handleDuplicate` puts
 * `cost_profile=<id>` on the /runs/new URL for a registry backtest/paper
 * source (pinned, including the default id and an explicit legacy_* id) and
 * omits it for a live source and for legacy-inferred / invalid / cost-less
 * sources. The receiver is covered in new-run-cost-profile.test.tsx.
 */

import React from "react";
import { render, screen, act, fireEvent } from "@testing-library/react";
import { useSession } from "next-auth/react";
import type { Run } from "@/lib/types";
import { makeRun, ADV1, LEGACY_BT, LEGACY_RT, STRESS80 } from "../helpers/cost-profile-fixtures";

const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "11111111-1111-4111-8111-111111111111" }),
  useRouter: () => ({ push: mockPush }),
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

const RUN_ID = "11111111-1111-4111-8111-111111111111";

async function duplicate(run: Run): Promise<URLSearchParams> {
  mockFetchRun.mockImplementation(() => ok(run));
  render(<RunDetailPage />);
  await act(async () => {
    await jest.advanceTimersByTimeAsync(0);
  });
  fireEvent.click(screen.getByRole("button", { name: "Duplicate Run" }));
  expect(mockPush).toHaveBeenCalledTimes(1);
  const url = mockPush.mock.calls[0]![0] as string;
  expect(url.startsWith("/runs/new?")).toBe(true);
  return new URLSearchParams(url.slice("/runs/new?".length));
}

beforeEach(() => {
  jest.useFakeTimers();
  mockPush.mockClear();
  (useSession as jest.Mock).mockReturnValue({ data: { user: { role: "admin" } }, status: "authenticated" });
});
afterEach(() => jest.useRealTimers());

describe("Run detail: Duplicate Run carries the cost profile", () => {
  it.each([
    ["a stress registry backtest", STRESS80, "coinbase_stress_t80"],
    ["a default-profile registry backtest (pinned, not omitted)", ADV1, "coinbase_advanced_1"],
    ["an explicit legacy_backtest_v0 registry backtest", LEGACY_BT, "legacy_backtest_v0"],
  ])("%s -> cost_profile=<id>", async (_name, profile, expected) => {
    const params = await duplicate(
      makeRun({ id: RUN_ID, mode: "backtest", costProfile: profile, legacyCosts: profile.legacy, metrics: false }),
    );
    expect(params.get("cost_profile")).toBe(expected);
    // The existing hand-over fields are still there.
    expect(params.get("strategy")).toBe("rsi_mean_reversion");
    expect(params.get("symbols")).toBe("BTC/EUR");
  });

  it("a live run never carries cost_profile, even with a registry profile", async () => {
    const params = await duplicate(
      makeRun({ id: RUN_ID, mode: "live", costProfile: STRESS80, legacyCosts: false, metrics: false }),
    );
    expect(params.has("cost_profile")).toBe(false);
    expect(params.get("strategy")).toBe("rsi_mean_reversion");
  });

  it.each([
    ["a legacy-inferred profile", { costProfile: LEGACY_RT, legacyCosts: true }],
    ["an invalid persisted profile", { costProfile: STRESS80, legacyCosts: true, costProfileInvalid: true }],
    ["an override (custom) profile", { costProfile: { ...ADV1, id: "custom", source: "override" as const } }],
    ["an older API with no cost fields", {}],
  ])("%s -> no cost_profile", async (_name, cost) => {
    const params = await duplicate(makeRun({ id: RUN_ID, mode: "backtest", metrics: false, ...cost }));
    expect(params.has("cost_profile")).toBe(false);
    expect(params.get("strategy")).toBe("rsi_mean_reversion");
  });
});
