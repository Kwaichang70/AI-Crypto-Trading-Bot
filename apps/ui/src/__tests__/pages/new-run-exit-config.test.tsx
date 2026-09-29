/**
 * apps/ui/src/__tests__/pages/new-run-exit-config.test.tsx
 * -----------------------------------------------------------------
 * WP1.3a (CF-13a-1 items 1, 2, 3, 4, 6): the new-run form (`app/runs/new/
 * page.tsx`) —
 *   1. `initDefaults` sends `null` (not `0`) for unset nullable numeric
 *      fields, and keeps `0` for non-nullable numeric fields with no
 *      explicit default.
 *   2. Structured 422 rendering on create (`<ExitConfigErrorPanel>`).
 *   3. `configWarnings[]` from a successful 201 create surfaced as toasts.
 *   4. `allowPyramiding` control: live forced false + W7 copy for dca/grid;
 *      paper/backtest checkbox defaults to the strategy default and is
 *      explicitly sent either way.
 *
 * Mirrors the mocking style of `new-run-mode-lockdown.test.tsx`.
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import type { Strategy, StrategyListResponse } from "@/lib/types";

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

const mockToast = jest.fn();
jest.mock("@/components/ui/toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

// Imported AFTER the mocks are registered.
import NewRunPage from "@/app/runs/new/page";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const MOMENTUM: Strategy = {
  name: "momentum_breakout",
  displayName: "Momentum Breakout",
  version: "1.0.0",
  description: "Momentum breakout strategy.",
  tags: [],
  parameterSchema: {
    type: "object",
    properties: {
      lookback: { type: "integer", default: 20 },
      // Nullable numeric, no explicit default -> must init to null (item 1).
      bracket_stop_loss_pct: { type: "number", nullable: true },
      trailing_stop_pct: { type: "number", nullable: true },
      // Non-nullable numeric, no explicit default -> keeps the pre-1.3a 0.
      position_size: { type: "number" },
    },
  },
  allowedModes: ["backtest", "paper", "live"],
  status: "active",
};

const GRID: Strategy = {
  name: "grid_trading",
  displayName: "Grid Trading",
  version: "1.0.0",
  description: "Grid trading strategy.",
  tags: [],
  parameterSchema: { type: "object", properties: {} },
  allowedModes: ["backtest", "paper", "live"],
  status: "active",
};

function strategiesResult(strategies: Strategy[]) {
  return { ok: true as const, data: { strategies, total: strategies.length } as StrategyListResponse };
}

function paramInput(labelText: string): HTMLInputElement {
  return screen.getByText(labelText).closest("div")!.querySelector("input") as HTMLInputElement;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockSearchGet.mockReturnValue(null);
});

// ===========================================================================
// Item 1: initDefaults sends null for unset nullable numerics
// ===========================================================================

describe("NewRunPage — initDefaults (SY-13a-02, CF-13a-1 item 1)", () => {
  it("renders a blank input for a nullable numeric field with no default (null, not 0)", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Bracket Stop Loss Pct")).toBeInTheDocument());
    expect(paramInput("Bracket Stop Loss Pct").value).toBe("");
    expect(paramInput("Trailing Stop Pct").value).toBe("");
  });

  it("keeps 0 for a non-nullable numeric field with no default", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Position Size")).toBeInTheDocument());
    expect(paramInput("Position Size").value).toBe("0");
  });

  it("submits null (not 0) for the untouched nullable field", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({
      ok: true,
      data: { id: "run-1", configWarnings: [] },
    });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Bracket Stop Loss Pct")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));

    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const [body] = mockCreateRun.mock.calls[0];
    expect(body.strategyParams.bracket_stop_loss_pct).toBeNull();
    expect(body.strategyParams.trailing_stop_pct).toBeNull();
    expect(body.strategyParams.position_size).toBe(0);
  });
});

// ===========================================================================
// Item 4: allowPyramiding control
// ===========================================================================

describe("NewRunPage — allowPyramiding control (SY-13a-08/09, CF-13a-1 item 4)", () => {
  it("defaults OFF for a non-dca/grid strategy in paper mode, and sends it explicitly", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1", configWarnings: [] } });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByRole("radio", { name: /paper/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /paper/i }));

    const checkbox = screen.getByLabelText("Allow Pyramiding") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect(mockCreateRun.mock.calls[0][0].allowPyramiding).toBe(false);
  });

  it("defaults ON for grid_trading in paper mode, and toggling off sends false", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([GRID]));
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1", configWarnings: [] } });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByRole("radio", { name: /paper/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /paper/i }));

    const checkbox = screen.getByLabelText("Allow Pyramiding") as HTMLInputElement;
    await waitFor(() => expect(checkbox.checked).toBe(true));

    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect(mockCreateRun.mock.calls[0][0].allowPyramiding).toBe(false);
  });

  it("forces the live checkbox to false, disabled, and shows the WP1.3b/WP1.10 explanation", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByRole("radio", { name: /^live/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /^live/i }));

    const checkbox = screen.getByLabelText(/Allow Pyramiding \(disabled in live mode\)/i) as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    expect(checkbox).toBeDisabled();
    expect(screen.getByText(/Live pyramiding is disabled until WP1\.3b\/WP1\.10\./)).toBeInTheDocument();
  });

  it("shows the W7 'accumulation disabled' copy for live grid_trading, and sends allowPyramiding=false", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([GRID]));
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByRole("radio", { name: /^live/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /^live/i }));

    expect(screen.getByText(/Accumulation disabled: single-entry variant, not validated\./)).toBeInTheDocument();
  });
});

// ===========================================================================
// Item 2: structured 422 rendering on create
// ===========================================================================

describe("NewRunPage — structured 422 rendering (CF-13a-1 item 2)", () => {
  it("renders the exit_manager_required panel from a real {detail:{...}} envelope", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({
      ok: false,
      error: {
        status: 422,
        message: "Validation error — check the request payload.",
        detail: {
          detail: {
            code: "exit_manager_required",
            strategy: "momentum_breakout",
            requires_one_of: ["bracket_stop_loss_pct (bracket_mode=fixed)", "trailing_stop_pct"],
            errors: [],
            warnings: [],
          },
        },
      },
    });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Position Size")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));

    await waitFor(() =>
      expect(screen.getByText("This strategy requires a downside exit")).toBeInTheDocument(),
    );
    expect(screen.getByText(/trailing_stop_pct/)).toBeInTheDocument();
  });
});

// ===========================================================================
// Item 3: configWarnings toasted after a successful create
// ===========================================================================

describe("NewRunPage — configWarnings toasted on success (CF-13a-1 item 3)", () => {
  it("toasts each warning and still navigates to the new run", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({
      ok: true,
      data: {
        id: "run-42",
        configWarnings: [
          { code: "sl_below_round_trip_cost", field: "bracket_stop_loss_pct", message: "SL < 1.3%." },
        ],
      },
    });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Position Size")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/runs/run-42"));
    expect(mockToast).toHaveBeenCalledWith("Run started with warnings", "warning");
    expect(mockToast).toHaveBeenCalledWith("bracket_stop_loss_pct: SL < 1.3%.", "warning");
  });

  it("toasts a no_downside_exit (W8) warning as a critical error, not a plain warning", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({
      ok: true,
      data: {
        id: "run-43",
        configWarnings: [{ code: "no_downside_exit", field: null, message: "No downside exit configured." }],
      },
    });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Position Size")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/runs/run-43"));
    expect(mockToast).toHaveBeenCalledWith("No downside exit configured.", "error");
  });

  it("shows the plain success toast when there are no warnings", async () => {
    mockFetchStrategies.mockResolvedValue(strategiesResult([MOMENTUM]));
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-44", configWarnings: [] } });
    render(<NewRunPage />);

    await waitFor(() => expect(screen.getByText("Position Size")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/runs/run-44"));
    expect(mockToast).toHaveBeenCalledWith("Run started successfully", "success");
  });
});
