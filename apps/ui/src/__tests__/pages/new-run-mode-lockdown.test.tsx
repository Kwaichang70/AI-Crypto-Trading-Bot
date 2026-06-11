/**
 * apps/ui/src/__tests__/pages/new-run-mode-lockdown.test.tsx
 * -----------------------------------------------------------
 * Sprint 51 Cycle 2 — strategy-availability lockdown on the New Run form.
 *
 * Exercises the real NewRunPage client component with mocked next/navigation
 * and @/lib/api so the mode-card disabling + auto-correct behaviour is tested
 * end-to-end (the decision logic lives inline in NewRunInner).
 *
 * Covers:
 *   1. A DEMOTED strategy disables the paper + live mode radios (backtest only).
 *   2. The demoted banner renders the demotionReason.
 *   3. Auto-correct: when paper/live is selected and the strategy switches to a
 *      demoted one, the mode falls back to "backtest".
 *   4. Graceful degrade: a strategy with NO allowedModes/status field is treated
 *      as allowing all three modes (paper/live not disabled).
 *
 * TEST-S51C2-410 .. 414
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { Strategy, StrategyListResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Mocks: next/navigation + @/lib/api
// ---------------------------------------------------------------------------

const mockPush = jest.fn();
const mockSearchGet = jest.fn(() => null);

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  useSearchParams: () => ({ get: mockSearchGet }),
}));

const mockFetchStrategies = jest.fn();
const mockFetchStrategySchema = jest.fn();
const mockCreateRun = jest.fn();

jest.mock("@/lib/api", () => ({
  fetchStrategies: (...args: unknown[]) => mockFetchStrategies(...args),
  fetchStrategySchema: (...args: unknown[]) => mockFetchStrategySchema(...args),
  createRun: (...args: unknown[]) => mockCreateRun(...args),
}));

// Imported AFTER the mocks are registered.
import NewRunPage from "@/app/runs/new/page";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMPTY_SCHEMA = { type: "object" as const, properties: {} };

// Overrides may set fields to `undefined` explicitly: several tests simulate
// a legacy API response with absent lockdown fields, which under
// exactOptionalPropertyTypes requires `| undefined` in the override type.
type StrategyOverrides = { [K in keyof Strategy]?: Strategy[K] | undefined };

function makeStrategy(overrides: StrategyOverrides = {}): Strategy {
  const base: Record<string, unknown> = {
    name: "grid_trading",
    displayName: "Grid Trading",
    version: "1.0.0",
    description: "Active strategy.",
    tags: [],
    parameterSchema: EMPTY_SCHEMA,
    allowedModes: ["backtest", "paper", "live"],
    status: "active",
  };
  // An explicit `undefined` override DELETES the key — faithfully modelling a
  // legacy API response in which the field is absent (not present-undefined).
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) delete base[key];
    else base[key] = value;
  }
  return base as unknown as Strategy;
}

const DEMOTED = makeStrategy({
  name: "ma_crossover",
  displayName: "MA Crossover",
  description: "Demoted strategy.",
  allowedModes: ["backtest"],
  status: "demoted",
  demotionReason: "Underperformance in non-trending regime.",
});

function strategiesResult(strategies: Strategy[]) {
  return {
    ok: true as const,
    data: { strategies, total: strategies.length } as StrategyListResponse,
  };
}

function mode(name: "backtest" | "paper" | "live"): HTMLInputElement {
  return screen.getByRole("radio", { name: new RegExp(name, "i") }) as HTMLInputElement;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockSearchGet.mockReturnValue(null);
});

// ===========================================================================
// 1 + 2. Demoted strategy disables paper/live + renders banner
// ===========================================================================

describe("NewRunPage — demoted strategy lockdown", () => {
  it("TEST-S51C2-410: paper + live mode radios are disabled for a demoted strategy", async () => {
    // First (auto-selected) strategy is the demoted one.
    mockFetchStrategies.mockResolvedValue(strategiesResult([DEMOTED]));

    render(<NewRunPage />);

    await waitFor(() => expect(mode("backtest")).toBeInTheDocument());

    expect(mode("backtest")).not.toBeDisabled();
    expect(mode("paper")).toBeDisabled();
    expect(mode("live")).toBeDisabled();
  });

  it("TEST-S51C2-411: demoted banner renders the demotionReason", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([DEMOTED]));

    render(<NewRunPage />);

    await waitFor(() =>
      expect(
        screen.getByText(/Underperformance in non-trending regime\./),
      ).toBeInTheDocument(),
    );
    // The banner also states it's backtest-only.
    expect(screen.getByText(/backtest only/i)).toBeInTheDocument();
  });
});

// ===========================================================================
// 3. Auto-correct to backtest when switching to a demoted strategy
// ===========================================================================

describe("NewRunPage — auto-correct mode", () => {
  it("TEST-S51C2-412: switching to a demoted strategy while in paper auto-corrects to backtest", async () => {
    const active = makeStrategy();
    // List has both; active is auto-selected first (index 0).
    mockFetchStrategies.mockResolvedValue(strategiesResult([active, DEMOTED]));
    // Switching the <select> triggers fetchStrategySchema for the chosen name.
    mockFetchStrategySchema.mockResolvedValue({ ok: true, data: DEMOTED });

    render(<NewRunPage />);

    await waitFor(() => expect(mode("paper")).toBeInTheDocument());

    // Select paper (allowed for the active strategy).
    fireEvent.click(mode("paper"));
    expect(mode("paper")).toBeChecked();

    // Switch strategy to the demoted one via the dropdown.
    const select = screen.getByRole("combobox") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "ma_crossover" } });

    // The auto-correct effect must flip the mode back to backtest.
    await waitFor(() => expect(mode("backtest")).toBeChecked());
    expect(mode("paper")).toBeDisabled();
    expect(mode("live")).toBeDisabled();
  });
});

// ===========================================================================
// 4. Graceful degrade — missing lockdown fields => all modes allowed
// ===========================================================================

describe("NewRunPage — graceful degrade", () => {
  it("TEST-S51C2-413: a strategy without allowedModes/status allows paper + live", async () => {
    // Simulate an older API response: no allowedModes, no status.
    const legacy = makeStrategy({
      name: "legacy_strategy",
      displayName: "Legacy Strategy",
      allowedModes: undefined,
      status: undefined,
    });
    mockFetchStrategies.mockResolvedValue(strategiesResult([legacy]));

    render(<NewRunPage />);

    await waitFor(() => expect(mode("backtest")).toBeInTheDocument());

    expect(mode("paper")).not.toBeDisabled();
    expect(mode("live")).not.toBeDisabled();
    // No demoted banner.
    expect(screen.queryByText(/backtest only/i)).not.toBeInTheDocument();
  });

  it("TEST-S51C2-414: an active strategy allows paper + live and shows no demoted banner", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([makeStrategy()]));

    render(<NewRunPage />);

    await waitFor(() => expect(mode("backtest")).toBeInTheDocument());

    expect(mode("paper")).not.toBeDisabled();
    expect(mode("live")).not.toBeDisabled();
    expect(screen.queryByText(/backtest only/i)).not.toBeInTheDocument();
  });
});
