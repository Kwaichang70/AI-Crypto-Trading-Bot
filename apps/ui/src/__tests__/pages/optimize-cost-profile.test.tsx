/**
 * WP21-T-35 (optimize): the request form offers any registry id and omits the
 * key for the default; the result header and history rows show the cost
 * profile (NULL / missing reads as "legacy costs"); a launched backtest is
 * priced like the optimization.
 */

import React from "react";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import type { Strategy, StrategyListResponse, OptimizeResponse } from "@/lib/types";
import { ADV1, STRESS80, LEGACY_BT } from "../helpers/cost-profile-fixtures";

const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  useParams: () => ({ id: "22222222-2222-4222-8222-222222222222" }),
}));

const mockFetchStrategies = jest.fn();
const mockFetchStrategySchema = jest.fn();
const mockRunOptimization = jest.fn();
const mockCreateRun = jest.fn();
const mockFetchOptimizationRuns = jest.fn();
const mockFetchOptimizationRun = jest.fn();

jest.mock("@/lib/api", () => ({
  fetchStrategies: (...a: unknown[]) => mockFetchStrategies(...a),
  fetchStrategySchema: (...a: unknown[]) => mockFetchStrategySchema(...a),
  runOptimization: (...a: unknown[]) => mockRunOptimization(...a),
  createRun: (...a: unknown[]) => mockCreateRun(...a),
  fetchOptimizationRuns: (...a: unknown[]) => mockFetchOptimizationRuns(...a),
  fetchOptimizationRun: (...a: unknown[]) => mockFetchOptimizationRun(...a),
}));

import OptimizePage from "@/app/optimize/page";
import OptimizationRunDetailPage from "@/app/optimize/[id]/page";

const MOMENTUM: Strategy = {
  name: "momentum_breakout",
  displayName: "Momentum Breakout",
  version: "1.0.0",
  description: "",
  tags: [],
  parameterSchema: { type: "object", properties: { lookback: { type: "integer", default: 20 } } },
};

const BASE_RESULT: OptimizeResponse = {
  strategyName: "momentum_breakout",
  rankBy: "sharpe_ratio",
  symbols: ["BTC/EUR"],
  timeframe: "1h",
  totalCombinations: 1,
  completedCombinations: 1,
  failedCombinations: 0,
  elapsedSeconds: 1,
  entries: [{ rank: 1, params: { lookback: 5 }, metrics: {} }],
};

const SUMMARY = {
  optimizationRunId: "22222222-2222-4222-8222-222222222222",
  strategyName: "momentum_breakout",
  symbols: ["BTC/EUR"],
  timeframe: "1h",
  rankBy: "sharpe_ratio",
  totalCombinations: 1,
  completedCombinations: 1,
  failedCombinations: 0,
  elapsedSeconds: 1,
  createdAt: "2026-01-01T00:00:00Z",
};

async function fillAndSubmit() {
  await screen.findByRole("option", { name: "momentum_breakout" });
  const [strategySelect] = screen.getAllByRole("combobox");
  fireEvent.change(strategySelect!, { target: { value: "momentum_breakout" } });
  await waitFor(() => expect(mockFetchStrategySchema).toHaveBeenCalledWith("momentum_breakout"));
  fireEvent.click(await screen.findByRole("button", { name: /Add Parameter/ }));
  fireEvent.change(screen.getByPlaceholderText(/Comma-separated/), { target: { value: "5" } });
  fireEvent.click(screen.getByRole("button", { name: "Run Optimization" }));
}

beforeEach(() => {
  jest.clearAllMocks();
  mockFetchStrategies.mockResolvedValue({
    ok: true as const,
    data: { strategies: [MOMENTUM], total: 1 } as StrategyListResponse,
  });
  mockFetchStrategySchema.mockResolvedValue({ ok: true, data: MOMENTUM });
  mockFetchOptimizationRuns.mockResolvedValue({ ok: true, data: { items: [], total: 0, offset: 0, limit: 10 } });
  mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1" } });
});

describe("Optimize form: cost-profile selector", () => {
  it("offers the default plus any registry id (SY-21-03)", async () => {
    render(<OptimizePage />);
    const sel = screen.getByLabelText("Cost profile") as HTMLSelectElement;
    expect(
      within(sel)
        .getAllByRole("option")
        .map((o) => (o as HTMLOptionElement).value),
    ).toEqual([
      "",
      "coinbase_advanced_1",
      "coinbase_advanced_d6",
      "coinbase_stress_t80",
      "coinbase_stress_t100",
      "frictionless_diagnostic",
      "legacy_backtest_v0",
      "legacy_runtime_v0",
    ]);
  });

  it("omits costProfile for the default and sends the chosen id otherwise", async () => {
    mockRunOptimization.mockResolvedValue({ ok: true, data: BASE_RESULT });
    const { unmount } = render(<OptimizePage />);
    await fillAndSubmit();
    await waitFor(() => expect(mockRunOptimization).toHaveBeenCalledTimes(1));
    expect("costProfile" in (mockRunOptimization.mock.calls[0] as [object])[0]).toBe(false);
    unmount();

    render(<OptimizePage />);
    fireEvent.change(screen.getByLabelText("Cost profile"), { target: { value: "frictionless_diagnostic" } });
    await fillAndSubmit();
    await waitFor(() => expect(mockRunOptimization).toHaveBeenCalledTimes(2));
    expect((mockRunOptimization.mock.calls[1] as [object])[0]).toMatchObject({
      costProfile: "frictionless_diagnostic",
    });
  });
});

describe("Optimize results and launch", () => {
  it("shows the response's cost profile in the result header, and a launched run reuses it", async () => {
    mockRunOptimization.mockResolvedValue({ ok: true, data: { ...BASE_RESULT, costProfile: STRESS80 } });
    render(<OptimizePage />);
    await fillAndSubmit();
    await screen.findByText("Strategy:");
    const badges = screen.getAllByTestId("cost-profile-badge");
    expect(badges.some((b) => b.getAttribute("data-kind") === "stress")).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Launch Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect((mockCreateRun.mock.calls[0] as [object])[0]).toMatchObject({
      mode: "backtest",
      costProfile: "coinbase_stress_t80",
    });
  });

  it("a response without a profile reads as legacy and the launched run carries none", async () => {
    mockRunOptimization.mockResolvedValue({ ok: true, data: BASE_RESULT });
    render(<OptimizePage />);
    await fillAndSubmit();
    await screen.findByText("Strategy:");
    expect(screen.getByText("legacy costs")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Launch Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect("costProfile" in (mockCreateRun.mock.calls[0] as [object])[0]).toBe(false);
  });

  it("history rows show the profile; NULL / missing reads as legacy costs", async () => {
    mockFetchOptimizationRuns.mockResolvedValue({
      ok: true,
      data: {
        items: [
          { ...SUMMARY, optimizationRunId: "a", costProfile: ADV1 },
          { ...SUMMARY, optimizationRunId: "b", costProfile: null },
          { ...SUMMARY, optimizationRunId: "c" },
        ],
        total: 3,
        offset: 0,
        limit: 10,
      },
    });
    render(<OptimizePage />);
    await waitFor(() => expect(screen.getAllByTestId("cost-profile-badge")).toHaveLength(3));
    expect(screen.getAllByTestId("cost-profile-badge").map((b) => b.textContent)).toEqual([
      "coinbase_advanced_1",
      "legacy costs",
      "legacy costs",
    ]);
  });
});

describe("Optimize detail page", () => {
  it("shows the stored profile in the header line", async () => {
    mockFetchOptimizationRun.mockResolvedValue({
      ok: true,
      data: { ...BASE_RESULT, costProfile: STRESS80 },
    });
    render(<OptimizationRunDetailPage />);
    const line = await screen.findByTestId("optimize-cost-line");
    expect(line).toHaveTextContent("stress");
    expect(line).toHaveTextContent("coinbase_stress_t80");
  });

  it("NULL costProfile shows 'legacy costs' (amber)", async () => {
    mockFetchOptimizationRun.mockResolvedValue({ ok: true, data: { ...BASE_RESULT, costProfile: null } });
    render(<OptimizationRunDetailPage />);
    const line = await screen.findByTestId("optimize-cost-line");
    expect(within(line).getByText("legacy costs")).toHaveClass("badge-warning");
  });

  it("a legacy_* profile is also labelled legacy and is not carried into a launched run", async () => {
    mockFetchOptimizationRun.mockResolvedValue({
      ok: true,
      data: { ...BASE_RESULT, costProfile: LEGACY_BT },
    });
    render(<OptimizationRunDetailPage />);
    await screen.findByTestId("optimize-cost-line");
    fireEvent.click(screen.getByRole("button", { name: "Launch Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect("costProfile" in (mockCreateRun.mock.calls[0] as [object])[0]).toBe(false);
  });

  it("a registry profile is carried into a launched run", async () => {
    mockFetchOptimizationRun.mockResolvedValue({ ok: true, data: { ...BASE_RESULT, costProfile: ADV1 } });
    render(<OptimizationRunDetailPage />);
    await screen.findByTestId("optimize-cost-line");
    fireEvent.click(screen.getByRole("button", { name: "Launch Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect((mockCreateRun.mock.calls[0] as [object])[0]).toMatchObject({ costProfile: "coinbase_advanced_1" });
  });
});
