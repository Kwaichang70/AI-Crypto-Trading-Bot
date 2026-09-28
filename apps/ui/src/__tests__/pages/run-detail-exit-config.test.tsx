/**
 * apps/ui/src/__tests__/pages/run-detail-exit-config.test.tsx
 * -----------------------------------------------------------------
 * WP1.3a (CF-13a-1 items 2/3/6): the run detail page
 * (`app/runs/[id]/page.tsx`) renders:
 *   - `<ConfigWarningsBanner>` from `run.configWarnings`
 *   - `<ProtectiveResumeBanner>` from `run.exitConfigWaived`/
 *     `run.exitManagerMissing`
 *   - a "Promote to Live" action, gated to stopped paper runs, that opens
 *     `<PromoteRunDialog>`
 *
 * `<PromoteRunDialog>` itself is unit-tested in
 * `components/promote-run-dialog.test.tsx` — mocked here so this file only
 * asserts the page's own wiring (gating + prop plumbing), mirroring the
 * `new-run-mode-lockdown.test.tsx` mocking style.
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { useSession } from "next-auth/react";
import type { Run } from "@/lib/types";

const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "66666666-6666-6666-6666-666666666666" }),
  useRouter: () => ({ push: mockPush }),
}));

// A protective-resumed ORPHANED live run's <AdminOnly>-gated Resume button
// (unrelated to CF-13a-1) needs a session -- admin, so it's a no-op here.
jest.mock("next-auth/react", () => ({ useSession: jest.fn() }));

const mockFetchRun = jest.fn();
const mockFetchPortfolio = jest.fn();
const mockFetchEquityCurve = jest.fn();
const mockFetchTrades = jest.fn();
const mockFetchOrders = jest.fn();
const mockFetchFills = jest.fn();
const mockFetchPositions = jest.fn();
const mockFetchLearningState = jest.fn();
const mockFetchDiagnostics = jest.fn();
const mockArchiveRun = jest.fn();

jest.mock("@/lib/api", () => ({
  fetchRun: (...args: unknown[]) => mockFetchRun(...args),
  fetchPortfolio: (...args: unknown[]) => mockFetchPortfolio(...args),
  fetchEquityCurve: (...args: unknown[]) => mockFetchEquityCurve(...args),
  fetchTrades: (...args: unknown[]) => mockFetchTrades(...args),
  fetchOrders: (...args: unknown[]) => mockFetchOrders(...args),
  fetchFills: (...args: unknown[]) => mockFetchFills(...args),
  fetchPositions: (...args: unknown[]) => mockFetchPositions(...args),
  fetchLearningState: (...args: unknown[]) => mockFetchLearningState(...args),
  fetchDiagnostics: (...args: unknown[]) => mockFetchDiagnostics(...args),
  archiveRun: (...args: unknown[]) => mockArchiveRun(...args),
  formatCurrency: (v: string) => v,
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

jest.mock("@/components/promote-run-dialog", () => ({
  PromoteRunDialog: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="mock-promote-run-dialog">
      <button onClick={onClose}>close-mock-promote-dialog</button>
    </div>
  ),
}));

// Imported AFTER the mocks are registered.
import RunDetailPage from "@/app/runs/[id]/page";

const BASE_RUN: Run = {
  id: "66666666-6666-6666-6666-666666666666",
  runMode: "paper",
  status: "stopped",
  config: {
    strategy_name: "rsi_mean_reversion",
    strategy_params: {},
    symbols: ["BTC/EUR"],
    timeframe: "1h",
    mode: "paper",
    initial_capital: "10000",
  },
  startedAt: "2026-01-01T00:00:00Z",
  stoppedAt: "2026-01-02T00:00:00Z",
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-02T00:00:00Z",
};

function mockAllOk(run: Run) {
  mockFetchRun.mockResolvedValue({ ok: true, data: run });
  mockFetchPortfolio.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
  mockFetchEquityCurve.mockResolvedValue({ ok: true, data: { runId: run.id, totalPoints: 0, points: [] } });
  mockFetchTrades.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchOrders.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchFills.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchPositions.mockResolvedValue({ ok: true, data: { runId: run.id, positions: [], count: 0 } });
  mockFetchLearningState.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
  mockFetchDiagnostics.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
}

beforeEach(() => {
  jest.clearAllMocks();
  (useSession as jest.Mock).mockReturnValue({
    data: { user: { role: "admin" } },
    status: "authenticated",
  });
});

describe("RunDetailPage — Promote to Live gating (CF-13a-1 item 2)", () => {
  it("shows 'Promote to Live' for a stopped paper run and opens the dialog", async () => {
    mockAllOk(BASE_RUN);
    render(<RunDetailPage />);

    const button = await screen.findByRole("button", { name: "Promote to Live" });
    expect(button).toBeInTheDocument();

    fireEvent.click(button);
    await waitFor(() =>
      expect(screen.getByTestId("mock-promote-run-dialog")).toBeInTheDocument(),
    );
  });

  it("hides 'Promote to Live' for a running paper run", async () => {
    mockAllOk({ ...BASE_RUN, status: "running", stoppedAt: null });
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/paper ·/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Promote to Live" })).not.toBeInTheDocument();
  });

  it("hides 'Promote to Live' for a stopped BACKTEST run", async () => {
    mockAllOk({ ...BASE_RUN, runMode: "backtest" });
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/backtest ·/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Promote to Live" })).not.toBeInTheDocument();
  });

  it("hides 'Promote to Live' for a stopped LIVE run", async () => {
    mockAllOk({ ...BASE_RUN, runMode: "live", config: { ...BASE_RUN.config, mode: "live" } });
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/live ·/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Promote to Live" })).not.toBeInTheDocument();
  });
});

describe("RunDetailPage — Promote to Live is AdminOnly (WP13a-S-02, security round 2)", () => {
  it("hides 'Promote to Live' for a viewer session", async () => {
    (useSession as jest.Mock).mockReturnValue({
      data: { user: { role: "viewer" } },
      status: "authenticated",
    });
    mockAllOk(BASE_RUN);
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/paper ·/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Promote to Live" })).not.toBeInTheDocument();
  });

  it("hides 'Promote to Live' while the session is still loading (fail-closed, not a flash of the control)", async () => {
    (useSession as jest.Mock).mockReturnValue({ data: null, status: "loading" });
    mockAllOk(BASE_RUN);
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/paper ·/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Promote to Live" })).not.toBeInTheDocument();
  });

  it("shows 'Promote to Live' for an admin session", async () => {
    (useSession as jest.Mock).mockReturnValue({
      data: { user: { role: "admin" } },
      status: "authenticated",
    });
    mockAllOk(BASE_RUN);
    render(<RunDetailPage />);

    expect(await screen.findByRole("button", { name: "Promote to Live" })).toBeInTheDocument();
  });
});

describe("RunDetailPage — configWarnings / protective-resume banners (CF-13a-1 item 3)", () => {
  it("renders ConfigWarningsBanner from run.configWarnings", async () => {
    mockAllOk({
      ...BASE_RUN,
      configWarnings: [
        { code: "sl_below_round_trip_cost", field: "bracket_stop_loss_pct", message: "SL < 1.3% round-trip cost." },
      ],
    });
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByTestId("config-warnings-banner")).toBeInTheDocument());
    expect(screen.getByText(/SL < 1\.3% round-trip cost\./)).toBeInTheDocument();
  });

  it("renders the critical ProtectiveResumeBanner when exitManagerMissing is true", async () => {
    mockAllOk({
      ...BASE_RUN,
      runMode: "live",
      status: "orphaned",
      config: { ...BASE_RUN.config, mode: "live" },
      exitConfigWaived: {
        code: "invalid_exit_config",
        errors: [
          {
            field: "trailing_stop_pct",
            reason: "out_of_range",
            value: "0.6",
            min: 0.005,
            max: 0.5,
            message: "Trailing stop out of range; dropped.",
          },
        ],
      },
      exitManagerMissing: true,
    });
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByTestId("protective-resume-banner")).toBeInTheDocument());
    expect(screen.getByText("No downside exit — flatten recommended")).toBeInTheDocument();
    expect(screen.getByText(/Trailing stop out of range; dropped\./)).toBeInTheDocument();
  });

  it("renders neither banner when the fields are absent", async () => {
    mockAllOk(BASE_RUN);
    render(<RunDetailPage />);

    await waitFor(() => expect(screen.getByText(/paper ·/)).toBeInTheDocument());
    expect(screen.queryByTestId("config-warnings-banner")).not.toBeInTheDocument();
    expect(screen.queryByTestId("protective-resume-banner")).not.toBeInTheDocument();
  });
});
