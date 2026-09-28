/**
 * apps/ui/src/__tests__/pages/run-detail-polling.test.tsx
 * -----------------------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-24, G-6, UT-11..14):
 *   - UT-12: an initial-load failure shows the full-page error; a poll
 *     failure (including a 404) shows ONLY the stale banner.
 *   - UT-11: the banner is `role="status"`/`aria-live="polite"` with
 *     "Stale · since", and a poll failure never unmounts the tabs, the
 *     Stop button, or an open dialog.
 *   - UT-13: after `handleStopped(updated)` calls `mainPoll.refetch()`, a
 *     pre-stop in-flight poll that resolves afterwards as "running" is
 *     discarded, not applied over the just-stopped state.
 *   - UT-14: a diagnostics-poll failure gets its own card-scoped badge,
 *     never the page-level banner.
 *
 * Under `jest.useFakeTimers()`, React's own scheduler can fall back to a
 * zero-delay `setTimeout` to flush a commit under jsdom, so every await
 * point below drives BOTH real microtasks AND that macrotask via
 * `jest.advanceTimersByTimeAsync(0)` inside `act()` -- a bare `waitFor()`
 * (which polls on its own faked timer) is not used for state this test
 * itself drives.
 */

import React from "react";
import { render, screen, fireEvent, act, within } from "@testing-library/react";
import { useSession } from "next-auth/react";
import type { Run } from "@/lib/types";

jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "66666666-6666-6666-6666-666666666666" }),
  useRouter: () => ({ push: jest.fn() }),
}));

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
const mockStopRun = jest.fn();

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
  stopRun: (...args: unknown[]) => mockStopRun(...args),
  formatCurrency: (v: string) => v,
  formatPct: (v: number) => `${(v * 100).toFixed(2)}%`,
}));

// Imported AFTER the mocks are registered.
import RunDetailPage from "@/app/runs/[id]/page";

const RUN_ID = "66666666-6666-6666-6666-666666666666";

const RUNNING_RUN: Run = {
  id: RUN_ID,
  runMode: "paper",
  status: "running",
  config: {
    strategy_name: "rsi_mean_reversion",
    strategy_params: {},
    symbols: ["BTC/EUR"],
    timeframe: "1h",
    mode: "paper",
    initial_capital: "10000",
  },
  startedAt: "2026-01-01T00:00:00Z",
  stoppedAt: null,
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
};

function okEmptySubFetches() {
  mockFetchPortfolio.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
  mockFetchEquityCurve.mockResolvedValue({ ok: true, data: { runId: RUN_ID, totalPoints: 0, points: [] } });
  mockFetchTrades.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchOrders.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchFills.mockResolvedValue({ ok: true, data: { total: 0, offset: 0, limit: 100, items: [] } });
  mockFetchPositions.mockResolvedValue({ ok: true, data: { runId: RUN_ID, positions: [], count: 0 } });
  mockFetchLearningState.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
  mockFetchDiagnostics.mockResolvedValue({ ok: false, error: { status: 404, message: "n/a" } });
}

/** Flushes both real microtasks and any zero-delay scheduler macrotask. */
async function flush() {
  await act(async () => {
    await jest.advanceTimersByTimeAsync(0);
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
  (useSession as jest.Mock).mockReturnValue({ data: { user: { role: "admin" } }, status: "authenticated" });
  okEmptySubFetches();
});

afterEach(() => {
  jest.useRealTimers();
});

describe("RunDetailPage — UT-12: initial load vs. poll failures", () => {
  it("an initial-load failure shows the full-page error banner", async () => {
    mockFetchRun.mockResolvedValue({ ok: false, error: { status: 500, message: "Internal server error." } });

    render(<RunDetailPage />);
    await flush();

    expect(screen.getByText("Internal server error.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop Run" })).not.toBeInTheDocument();
  });

  it("a poll failure (404) shows ONLY the stale banner -- tabs and Stop stay mounted (UT-11)", async () => {
    mockFetchRun
      .mockResolvedValueOnce({ ok: true, data: RUNNING_RUN }) // initial load
      .mockResolvedValueOnce({ ok: false, error: { status: 404, message: "Run not found." } }); // main poll tick

    render(<RunDetailPage />);
    await flush();
    expect(screen.getByRole("button", { name: "Stop Run" })).toBeInTheDocument();

    await act(async () => {
      await jest.advanceTimersByTimeAsync(5000);
    });

    const staleBanner = screen.getByText(/Stale · since/);
    expect(staleBanner.closest('[role="status"]')).toHaveAttribute("aria-live", "polite");

    // The page did NOT go blank: tabs, Stop button, run header all remain.
    expect(screen.getByRole("button", { name: "Stop Run" })).toBeInTheDocument();
    expect(screen.getByText(/paper ·/)).toBeInTheDocument();
  });

  it("'Retry now' on the stale banner triggers a refetch", async () => {
    mockFetchRun
      .mockResolvedValueOnce({ ok: true, data: RUNNING_RUN })
      .mockResolvedValueOnce({ ok: false, error: { status: 404, message: "Run not found." } })
      .mockResolvedValueOnce({ ok: true, data: RUNNING_RUN });

    render(<RunDetailPage />);
    await flush();
    expect(screen.getByRole("button", { name: "Stop Run" })).toBeInTheDocument();

    await act(async () => {
      await jest.advanceTimersByTimeAsync(5000);
    });
    expect(screen.getByText(/Stale · since/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry now" }));
    await flush();

    expect(screen.queryByText(/Stale · since/)).not.toBeInTheDocument();
    expect(mockFetchRun).toHaveBeenCalledTimes(3);
  });
});

describe("RunDetailPage — UT-13: a superseded pre-stop poll is discarded", () => {
  it("a stale in-flight 'running' poll result never overwrites the just-stopped state", async () => {
    const STOPPED_RUN: Run = { ...RUNNING_RUN, status: "stopped", stoppedAt: "2026-01-02T00:00:00Z" };

    let resolveSecondFetchRun!: (value: unknown) => void;
    mockFetchRun
      .mockResolvedValueOnce({ ok: true, data: RUNNING_RUN }) // initial load
      .mockImplementationOnce(
        () => new Promise((resolve) => { resolveSecondFetchRun = resolve; }),
      ) // main-poll tick #1 (deliberately left pending -- simulates "in flight")
      .mockResolvedValueOnce({ ok: true, data: STOPPED_RUN }); // refetch() from handleStopped

    mockStopRun.mockResolvedValue({ ok: true, data: { ...STOPPED_RUN, flatten: null } });

    render(<RunDetailPage />);
    await flush();
    expect(screen.getByRole("button", { name: "Stop Run" })).toBeInTheDocument();

    // Trigger the main poll tick -- it is now in flight (fetchRun #2 pending).
    await act(async () => {
      await jest.advanceTimersByTimeAsync(5000);
    });
    expect(mockFetchRun).toHaveBeenCalledTimes(2);

    // Stop the run WHILE that poll is still in flight.
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));
    await flush();
    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Stop Run" }));

    // handleStopped(updated) -> setRun(stopped) + mainPoll.refetch(), which
    // issues fetchRun #3 and resolves it immediately (mocked above).
    await flush();
    expect(mockFetchRun).toHaveBeenCalledTimes(3);
    expect(screen.getByText(/paper ·/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop Run" })).not.toBeInTheDocument();

    // NOW the stale poll #2 resolves, reporting "running" -- it must be
    // discarded by usePolling's sequence guard (SY-70-23 (b)).
    await act(async () => {
      resolveSecondFetchRun({ ok: true, data: RUNNING_RUN });
      await jest.advanceTimersByTimeAsync(0);
    });

    // Still stopped -- the stale "running" result never applied.
    expect(screen.queryByRole("button", { name: "Stop Run" })).not.toBeInTheDocument();
  });
});

describe("RunDetailPage — UT-14: diagnostics-poll failure gets its own badge, not the page banner", () => {
  it("shows 'Diagnostics unavailable' on the card without the page-level stale banner", async () => {
    mockFetchRun.mockResolvedValue({ ok: true, data: RUNNING_RUN });
    mockFetchDiagnostics.mockResolvedValue({ ok: false, error: { status: 500, message: "boom" } });

    render(<RunDetailPage />);
    await flush();

    expect(screen.getByText("Diagnostics unavailable")).toBeInTheDocument();
    expect(screen.queryByText(/Stale · since/)).not.toBeInTheDocument();
  });
});
