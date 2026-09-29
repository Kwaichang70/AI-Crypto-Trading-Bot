/**
 * apps/ui/src/__tests__/pages/optimize-idempotency.test.tsx
 * -----------------------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md C-2, SY-70-19/20, UT-27): the
 * optimize list page's `handleLaunchRun` sends an `Idempotency-Key`, a
 * double-click launches only once, and launching two DIFFERENT entries
 * mints two DIFFERENT keys (distinct `strategyParams` -> distinct
 * JSON-stringified snapshot).
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { Strategy, StrategyListResponse, OptimizeResponse } from "@/lib/types";

const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

const mockFetchStrategies = jest.fn();
const mockFetchStrategySchema = jest.fn();
const mockRunOptimization = jest.fn();
const mockCreateRun = jest.fn();
const mockFetchOptimizationRuns = jest.fn();

jest.mock("@/lib/api", () => ({
  fetchStrategies: (...args: unknown[]) => mockFetchStrategies(...args),
  fetchStrategySchema: (...args: unknown[]) => mockFetchStrategySchema(...args),
  runOptimization: (...args: unknown[]) => mockRunOptimization(...args),
  createRun: (...args: unknown[]) => mockCreateRun(...args),
  fetchOptimizationRuns: (...args: unknown[]) => mockFetchOptimizationRuns(...args),
}));

// Imported AFTER the mocks are registered.
import OptimizePage from "@/app/optimize/page";

const MOMENTUM: Strategy = {
  name: "momentum_breakout",
  displayName: "Momentum Breakout",
  version: "1.0.0",
  description: "",
  tags: [],
  parameterSchema: { type: "object", properties: { lookback: { type: "integer", default: 20 } } },
};

const RESULTS: OptimizeResponse = {
  strategyName: "momentum_breakout",
  rankBy: "sharpe_ratio",
  symbols: ["BTC/EUR"],
  timeframe: "1h",
  totalCombinations: 2,
  completedCombinations: 2,
  failedCombinations: 0,
  elapsedSeconds: 1,
  entries: [
    { rank: 1, params: { lookback: 5 }, metrics: {} },
    { rank: 2, params: { lookback: 10 }, metrics: {} },
  ],
};

function strategiesResult(strategies: Strategy[]) {
  return { ok: true as const, data: { strategies, total: strategies.length } as StrategyListResponse };
}

function keyOf(callIndex: number): string {
  const [, opts] = mockCreateRun.mock.calls[callIndex] as [unknown, { idempotencyKey: string }];
  return opts.idempotencyKey;
}

async function renderWithResults() {
  mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
  mockFetchStrategySchema.mockResolvedValue({ ok: true, data: MOMENTUM });
  mockFetchOptimizationRuns.mockResolvedValue({ ok: true, data: { items: [], total: 0, offset: 0, limit: 10 } });
  mockRunOptimization.mockResolvedValue({ ok: true, data: RESULTS });

  render(<OptimizePage />);

  await screen.findByRole("option", { name: "momentum_breakout" });
  const [strategySelect] = screen.getAllByRole("combobox");
  fireEvent.change(strategySelect, { target: { value: "momentum_breakout" } });
  await waitFor(() => expect(mockFetchStrategySchema).toHaveBeenCalledWith("momentum_breakout"));

  const addParamButton = await screen.findByRole("button", { name: /Add Parameter/ });
  fireEvent.click(addParamButton);
  fireEvent.change(screen.getByPlaceholderText(/Comma-separated/), { target: { value: "5, 10" } });
  fireEvent.click(screen.getByRole("button", { name: "Run Optimization" }));

  await waitFor(() => expect(screen.getByText("Strategy:")).toBeInTheDocument());
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe("OptimizePage — Idempotency-Key on launch (C-2, UT-27)", () => {
  it("sends createRun(body, {idempotencyKey}) — never a bare string 2nd argument", async () => {
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1" } });
    await renderWithResults();

    const launchButtons = screen.getAllByRole("button", { name: "Launch Run" });
    fireEvent.click(launchButtons[0]!);

    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const [, opts] = mockCreateRun.mock.calls[0] as [unknown, unknown];
    expect(typeof opts).toBe("object");
    expect((opts as { idempotencyKey: string }).idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("launching two different entries mints two different keys", async () => {
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1" } });
    await renderWithResults();

    const launchButtons = screen.getAllByRole("button", { name: "Launch Run" });
    fireEvent.click(launchButtons[0]!);
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));

    fireEvent.click(launchButtons[1]!);
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));

    expect(keyOf(0)).not.toBe(keyOf(1));
  });

  it("double-clicking the same launch button results in exactly one createRun call", async () => {
    let resolveCreate: (value: unknown) => void = () => {};
    mockCreateRun.mockImplementationOnce(
      () => new Promise((resolve) => { resolveCreate = resolve; }),
    );
    await renderWithResults();

    const launchButtons = screen.getAllByRole("button", { name: "Launch Run" });
    fireEvent.click(launchButtons[0]!);
    fireEvent.click(launchButtons[0]!);
    fireEvent.click(launchButtons[0]!);

    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    resolveCreate({ ok: true, data: { id: "run-1" } });
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/runs/run-1"));
  });

  it("409 idempotency_in_progress shows a neutral message on the launch error banner", async () => {
    mockCreateRun.mockResolvedValueOnce({
      ok: false,
      error: { status: 409, message: "conflict", detail: { code: "idempotency_in_progress" } },
    });
    await renderWithResults();

    const launchButtons = screen.getAllByRole("button", { name: "Launch Run" });
    fireEvent.click(launchButtons[0]!);

    await waitFor(() =>
      expect(screen.getByText("Your previous request is still being processed…")).toBeInTheDocument(),
    );
  });
});
