/**
 * apps/ui/src/__tests__/pages/optimize-exit-config.test.tsx
 * -----------------------------------------------------------------
 * WP1.3a (CF-13a-1 item 2/6): the optimize page renders the shared
 * structured 422 panel (combo_index/params-scoped `errors[]`,
 * `total_invalid`) for `POST /api/v1/optimize`'s `invalid_exit_config` /
 * `exit_manager_required` (SY-13a-17), with the generic message as
 * fallback, using the real `{"detail": {...}}` envelope.
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { Strategy, StrategyListResponse } from "@/lib/types";

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
  parameterSchema: {
    type: "object",
    properties: {
      lookback: { type: "integer", default: 20 },
    },
  },
};

function strategiesResult(strategies: Strategy[]) {
  return { ok: true as const, data: { strategies, total: strategies.length } as StrategyListResponse };
}

async function selectMomentumStrategy() {
  // The <select> always renders; its strategy <option>s only appear once
  // the async fetchStrategies() resolves -- wait for the real option
  // before firing change, otherwise the handler's `strategies.find(...)`
  // runs against a still-empty array and silently no-ops.
  await screen.findByRole("option", { name: "momentum_breakout" });
  // Scoped to the FIRST combobox: the page also has a "Rank By" <select>
  // (role=combobox too) unconditionally rendered further down the form.
  const [strategySelect] = screen.getAllByRole("combobox");
  fireEvent.change(strategySelect, { target: { value: "momentum_breakout" } });
  // Let the schema-fetch effect's promise resolution land inside act().
  await waitFor(() => expect(mockFetchStrategySchema).toHaveBeenCalledWith("momentum_breakout"));
}

async function fillMinimalGridAndSubmit() {
  await selectMomentumStrategy();

  const addParamButton = await screen.findByRole("button", { name: /Add Parameter/ });
  await waitFor(() => expect(addParamButton).not.toBeDisabled());
  fireEvent.click(addParamButton);

  const valuesInput = screen.getByPlaceholderText(/Comma-separated/);
  fireEvent.change(valuesInput, { target: { value: "5, 10" } });

  fireEvent.click(screen.getByRole("button", { name: "Run Optimization" }));
}

beforeEach(() => {
  jest.clearAllMocks();
  mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
  mockFetchStrategySchema.mockResolvedValue({ ok: true, data: MOMENTUM });
  mockFetchOptimizationRuns.mockResolvedValue({ ok: true, data: { items: [], total: 0, offset: 0, limit: 10 } });
});

describe("OptimizePage — structured 422 rendering (CF-13a-1 item 2)", () => {
  it("renders invalid_exit_config with combo_index-scoped errors and total_invalid", async () => {
    mockRunOptimization.mockResolvedValue({
      ok: false,
      error: {
        status: 422,
        message: "Validation error — check the request payload.",
        detail: {
          detail: {
            code: "invalid_exit_config",
            errors: [
              {
                combo_index: 0,
                params: { lookback: 5 },
                field: "bracket_stop_loss_pct",
                reason: "out_of_range",
                value: "0",
                min: 0.0065,
                max: 0.5,
                message: "unset via zero, but momentum_breakout requires an exit.",
              },
            ],
            total_invalid: 2,
          },
        },
      },
    });

    render(<OptimizePage />);
    await fillMinimalGridAndSubmit();

    await waitFor(() => expect(screen.getByText("Invalid exit configuration")).toBeInTheDocument());
    expect(screen.getByText(/2 combinations failed validation/)).toBeInTheDocument();
    expect(screen.getByText(/Combination #0:/)).toBeInTheDocument();
  });

  it("renders exit_manager_required for a momentum grid with no bracket keys", async () => {
    mockRunOptimization.mockResolvedValue({
      ok: false,
      error: {
        status: 422,
        message: "Validation error — check the request payload.",
        detail: {
          detail: {
            code: "exit_manager_required",
            errors: [
              {
                combo_index: 0,
                params: { lookback: 5 },
                field: null,
                reason: "exit_manager_required",
                value: null,
                min: null,
                max: null,
                message: "momentum_breakout requires a downside exit.",
              },
            ],
            total_invalid: 1,
          },
        },
      },
    });

    render(<OptimizePage />);
    await fillMinimalGridAndSubmit();

    await waitFor(() => expect(screen.getByText("This strategy requires a downside exit")).toBeInTheDocument());
  });

  it("falls back to the generic message for a plain client-side validation error", async () => {
    render(<OptimizePage />);

    // Submit with a strategy selected but no grid rows -> the page's own
    // client-side guard fires before any network call, with `detail`
    // undefined -- the generic fallback path must still render cleanly.
    await selectMomentumStrategy();
    const addParamButton = await screen.findByRole("button", { name: /Add Parameter/ });
    await waitFor(() => expect(addParamButton).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Run Optimization" }));

    await waitFor(() =>
      expect(screen.getByText("Add at least one parameter with values to the grid.")).toBeInTheDocument(),
    );
    expect(mockRunOptimization).not.toHaveBeenCalled();
  });
});
