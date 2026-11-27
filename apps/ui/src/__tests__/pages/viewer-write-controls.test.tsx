/**
 * CF-97-43: every control that issues a non-GET through apiFetch is hidden
 * (header actions, same as Resume / Promote) or shown disabled with the text
 * "admin only" (form submits and the per-row Launch Run) for a viewer session,
 * and present and enabled for an admin session. Presentation only: the BFF
 * role check stays the control.
 */

import React from "react";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
import { useSession } from "next-auth/react";
import type { Strategy, StrategyListResponse, OptimizeEntry } from "@/lib/types";
import { makeRun } from "../helpers/cost-profile-fixtures";

jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "11111111-1111-4111-8111-111111111111" }),
  useRouter: () => ({ push: jest.fn() }),
  useSearchParams: () => ({ get: () => null }),
}));
jest.mock("next-auth/react", () => ({ useSession: jest.fn() }));

const mockFetchRun = jest.fn();
const mockFetchStrategies = jest.fn();
const mockFetchStrategySchema = jest.fn();
const mockFetchOptimizationRuns = jest.fn();
const mockFetchModelVersions = jest.fn();
const ok = (data: unknown) => Promise.resolve({ ok: true, data });
const notFound = () => Promise.resolve({ ok: false, error: { status: 404, message: "n/a" } });
const emptyPage = () => ok({ total: 0, offset: 0, limit: 100, items: [] });

jest.mock("@/lib/api", () => ({
  fetchRun: (...a: unknown[]) => mockFetchRun(...a),
  fetchPortfolio: () => notFound(),
  fetchEquityCurve: () => ok({ runId: "x", totalPoints: 0, points: [] }),
  fetchTrades: () => emptyPage(),
  fetchOrders: () => emptyPage(),
  fetchFills: () => emptyPage(),
  fetchPositions: () => ok({ runId: "x", positions: [], count: 0 }),
  fetchLearningState: () => notFound(),
  fetchDiagnostics: () => notFound(),
  fetchStrategies: (...a: unknown[]) => mockFetchStrategies(...a),
  fetchStrategySchema: (...a: unknown[]) => mockFetchStrategySchema(...a),
  fetchOptimizationRuns: (...a: unknown[]) => mockFetchOptimizationRuns(...a),
  fetchModelVersions: (...a: unknown[]) => mockFetchModelVersions(...a),
  trainModel: jest.fn(),
  retrainModel: jest.fn(),
  activateModelVersion: jest.fn(),
  archiveRun: jest.fn(),
  stopRun: jest.fn(),
  createRun: jest.fn(),
  runOptimization: jest.fn(),
  formatCurrency: (v: string) => v,
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

import RunDetailPage from "@/app/runs/[id]/page";
import NewRunPage from "@/app/runs/new/page";
import OptimizePage from "@/app/optimize/page";
import ModelsPage from "@/app/models/page";
import { buildResultColumns } from "@/app/optimize/result-columns";

const RUN_ID = "11111111-1111-4111-8111-111111111111";
const ROLES = [
  ["viewer", { data: { user: { role: "viewer" } }, status: "authenticated" }],
  ["admin", { data: { user: { role: "admin" } }, status: "authenticated" }],
] as const;
const setRole = (role: "viewer" | "admin") =>
  (useSession as jest.Mock).mockReturnValue(ROLES.find(([r]) => r === role)![1]);

const STRATEGY: Strategy = {
  name: "momentum_breakout",
  displayName: "Momentum Breakout",
  version: "1.0.0",
  description: "",
  tags: [],
  parameterSchema: { type: "object", properties: { lookback: { type: "integer", default: 20 } } },
};

const adminOnlyNotice = () => screen.getByRole("button", { name: "admin only" });

beforeEach(() => {
  jest.clearAllMocks();
  mockFetchStrategies.mockResolvedValue({
    ok: true,
    data: { strategies: [STRATEGY], total: 1 } as StrategyListResponse,
  });
  mockFetchStrategySchema.mockResolvedValue({ ok: true, data: STRATEGY });
  mockFetchOptimizationRuns.mockResolvedValue({ ok: true, data: { items: [], total: 0, offset: 0, limit: 10 } });
  mockFetchModelVersions.mockResolvedValue({
    ok: true,
    data: {
      total: 1,
      items: [
        {
          id: "22222222-2222-4222-8222-222222222222", symbol: "BTC/USD", timeframe: "1h", trainedAt: "2026-01-01T00:00:00Z",
          accuracy: 0.6, nTradesUsed: 60, nBarsUsed: 500, labelMethod: "horizon", trigger: "manual", modelPath: "m.bin",
          isActive: false, extra: null,
        },
      ],
    },
  });
});

async function renderRunPage(status: "running" | "stopped") {
  mockFetchRun.mockImplementation(() => ok({ ...makeRun({ id: RUN_ID, mode: "paper", metrics: false }), status }));
  jest.useFakeTimers();
  render(<RunDetailPage />);
  await act(async () => {
    await jest.advanceTimersByTimeAsync(0);
  });
}

describe.each(["viewer", "admin"] as const)("run detail header, %s session", (role) => {
  afterEach(() => jest.useRealTimers());

  it("Stop Run (DELETE /runs/{id}): hidden for a viewer, enabled for an admin", async () => {
    setRole(role);
    await renderRunPage("running");
    const stop = screen.queryByRole("button", { name: "Stop Run" });
    if (role === "viewer") expect(stop).not.toBeInTheDocument();
    else expect(stop).toBeEnabled();
  });

  it("Archive (PATCH /runs/{id}/archive): hidden for a viewer, enabled for an admin", async () => {
    setRole(role);
    await renderRunPage("stopped");
    const archive = screen.queryByRole("button", { name: "Archive" });
    if (role === "viewer") expect(archive).not.toBeInTheDocument();
    else expect(archive).toBeEnabled();
  });

  it("Duplicate Run only navigates (GET): unchanged for both roles", async () => {
    setRole(role);
    await renderRunPage("stopped");
    expect(screen.getByRole("button", { name: "Duplicate Run" })).toBeEnabled();
  });
});

describe.each(["viewer", "admin"] as const)("run and optimization forms, %s session", (role) => {
  it("Start Run (POST /runs): disabled 'admin only' for a viewer, enabled for an admin", async () => {
    setRole(role);
    render(<NewRunPage />);
    if (role === "viewer") {
      expect(await screen.findByRole("button", { name: "admin only" })).toBeDisabled();
      expect(screen.queryByRole("button", { name: /^Start Run/ })).not.toBeInTheDocument();
    } else {
      const start = await screen.findByRole("button", { name: /^Start Run/ });
      await waitFor(() => expect(start).toBeEnabled());
      expect(screen.queryByRole("button", { name: "admin only" })).not.toBeInTheDocument();
    }
  });

  it("Run Optimization (POST /optimize): disabled 'admin only' for a viewer, enabled for an admin", async () => {
    setRole(role);
    render(<OptimizePage />);
    await screen.findByRole("option", { name: "momentum_breakout" });
    if (role === "viewer") {
      expect(adminOnlyNotice()).toBeDisabled();
      expect(screen.queryByRole("button", { name: "Run Optimization" })).not.toBeInTheDocument();
    } else {
      fireEvent.change(screen.getAllByRole("combobox")[0]!, { target: { value: "momentum_breakout" } });
      await waitFor(() => expect(mockFetchStrategySchema).toHaveBeenCalledWith("momentum_breakout"));
      await waitFor(() => expect(screen.getByRole("button", { name: "Run Optimization" })).toBeEnabled());
    }
  });

  it("Launch Run (POST /runs, per result row): disabled 'admin only' for a viewer, enabled for an admin", () => {
    setRole(role);
    const entry = { rank: 1, params: { lookback: 5 }, metrics: {} } as OptimizeEntry;
    const actions = buildResultColumns("sharpe_ratio", null, jest.fn()).find((c) => c.key === "actions")!;
    render(<>{actions.render(entry)}</>);
    if (role === "viewer") {
      expect(adminOnlyNotice()).toBeDisabled();
      expect(screen.queryByRole("button", { name: "Launch Run" })).not.toBeInTheDocument();
    } else {
      expect(screen.getByRole("button", { name: "Launch Run" })).toBeEnabled();
    }
  });
});

describe.each(["viewer", "admin"] as const)("models page, %s session", (role) => {
  async function renderModels() {
    setRole(role);
    render(<ModelsPage />);
    await screen.findByText("BTC/USD");
    fireEvent.change(screen.getByLabelText("Retrain symbol"), { target: { value: "BTC/USD" } });
  }

  it("Train (POST /ml/train) and Retrain (POST /ml/retrain/{symbol}): one disabled 'admin only' for a viewer, enabled for an admin", async () => {
    await renderModels();
    if (role === "viewer") {
      expect(screen.queryByRole("button", { name: "Train" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Retrain" })).not.toBeInTheDocument();
    } else {
      expect(screen.getByRole("button", { name: "Train" })).toBeEnabled();
      expect(screen.getByRole("button", { name: "Retrain" })).toBeEnabled();
    }
  });

  it("Activate (PUT /ml/models/{id}/activate, per row): disabled 'admin only' for a viewer, enabled for an admin", async () => {
    await renderModels();
    const activate = screen.queryByRole("button", { name: /^Activate model/ });
    if (role === "viewer") {
      expect(activate).not.toBeInTheDocument();
      // one notice for the toolbar pair, one for the row
      const notices = screen.getAllByRole("button", { name: "admin only" });
      expect(notices).toHaveLength(2);
      notices.forEach((n) => expect(n).toBeDisabled());
    } else {
      expect(activate).toBeEnabled();
      expect(screen.queryByRole("button", { name: "admin only" })).not.toBeInTheDocument();
    }
  });
});
