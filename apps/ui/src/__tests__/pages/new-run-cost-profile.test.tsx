/**
 * WP21-T-35 (new run): the cost-profile selector follows SY-21-03 --
 *   backtest: any registry id   paper: live-equivalent ids only   live: none
 * and the request body omits `costProfile` for the platform default.
 * WP21BD-UI-01 adds the live-confirm path: a stale backtest profile must not
 * reach the live `createRun` body.
 */

import React from "react";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import type { Strategy, StrategyListResponse } from "@/lib/types";

const mockPush = jest.fn();
const mockSearchGet = jest.fn((_key: string): string | null => null);

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  useSearchParams: () => ({ get: (k: string) => mockSearchGet(k) }),
}));

const mockFetchStrategies = jest.fn();
const mockFetchStrategySchema = jest.fn();
const mockCreateRun = jest.fn();

jest.mock("@/lib/api", () => ({
  fetchStrategies: (...a: unknown[]) => mockFetchStrategies(...a),
  fetchStrategySchema: (...a: unknown[]) => mockFetchStrategySchema(...a),
  createRun: (...a: unknown[]) => mockCreateRun(...a),
}));

jest.mock("@/components/ui/toast", () => ({ useToast: () => ({ toast: jest.fn() }) }));

import NewRunPage from "@/app/runs/new/page";

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

async function renderForm() {
  mockFetchStrategies.mockResolvedValue({
    ok: true as const,
    data: { strategies: [GRID], total: 1 } as StrategyListResponse,
  });
  render(<NewRunPage />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Start Run" })).toBeInTheDocument());
}

function costSelect(): HTMLSelectElement {
  return screen.getByLabelText("Cost profile") as HTMLSelectElement;
}

function optionValues(sel: HTMLSelectElement): string[] {
  return within(sel)
    .getAllByRole("option")
    .map((o) => (o as HTMLOptionElement).value);
}

function chooseMode(m: "backtest" | "paper" | "live") {
  fireEvent.click(screen.getByRole("radio", { name: new RegExp(m, "i") }));
}

function submittedBody(): Record<string, unknown> {
  return (mockCreateRun.mock.calls[0] as [Record<string, unknown>])[0];
}

beforeEach(() => {
  jest.clearAllMocks();
  mockSearchGet.mockReturnValue(null);
  mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1", configWarnings: [] } });
});

describe("New run: cost-profile selector per mode (SY-21-03)", () => {
  it("backtest offers the default plus every registry id", async () => {
    await renderForm();
    expect(optionValues(costSelect())).toEqual([
      "",
      "coinbase_advanced_1",
      "coinbase_advanced_d6",
      "coinbase_stress_t80",
      "coinbase_stress_t100",
      "frictionless_diagnostic",
      "legacy_backtest_v0",
      "legacy_runtime_v0",
    ]);
    expect(within(costSelect()).getByRole("option", { name: /Platform default \(coinbase_advanced_1\)/ })).toBeInTheDocument();
  });

  it("paper offers the default plus live-equivalent ids only", async () => {
    await renderForm();
    chooseMode("paper");
    expect(optionValues(costSelect())).toEqual(["", "coinbase_advanced_1", "coinbase_advanced_d6"]);
  });

  it("live has no selector, only the actual-fees note", async () => {
    await renderForm();
    chooseMode("live");
    expect(screen.queryByLabelText("Cost profile")).not.toBeInTheDocument();
    expect(screen.getByTestId("cost-profile-live-note")).toHaveTextContent("actual fees");
  });

  it("warns that stress / legacy / diagnostic profiles are hidden from the leaderboard", async () => {
    await renderForm();
    fireEvent.change(costSelect(), { target: { value: "coinbase_stress_t80" } });
    expect(screen.getByTestId("cost-profile-hint")).toHaveTextContent(/hidden from the leaderboard/);
    fireEvent.change(costSelect(), { target: { value: "legacy_backtest_v0" } });
    expect(screen.getByTestId("cost-profile-hint")).toHaveTextContent(/Legacy cost model/);
    fireEvent.change(costSelect(), { target: { value: "coinbase_advanced_d6" } });
    expect(screen.queryByTestId("cost-profile-hint")).not.toBeInTheDocument();
  });
});

describe("New run: costProfile in the request body", () => {
  it("omits the key for the platform default", async () => {
    await renderForm();
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect("costProfile" in submittedBody()).toBe(false);
  });

  it("sends a chosen backtest profile", async () => {
    await renderForm();
    fireEvent.change(costSelect(), { target: { value: "coinbase_stress_t100" } });
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect(submittedBody()).toMatchObject({ mode: "backtest", costProfile: "coinbase_stress_t100" });
  });

  it("drops a backtest-only profile after switching to paper (never sends an ineligible id)", async () => {
    await renderForm();
    fireEvent.change(costSelect(), { target: { value: "coinbase_stress_t100" } });
    chooseMode("paper");
    expect(costSelect().value).toBe("");
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect(submittedBody()).toMatchObject({ mode: "paper" });
    expect("costProfile" in submittedBody()).toBe(false);
  });

  it("sends a live-equivalent profile in paper mode", async () => {
    await renderForm();
    chooseMode("paper");
    fireEvent.change(costSelect(), { target: { value: "coinbase_advanced_d6" } });
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    expect(submittedBody()).toMatchObject({ mode: "paper", costProfile: "coinbase_advanced_d6" });
  });

  it("pre-selects a valid cost_profile query param (Duplicate Run)", async () => {
    mockSearchGet.mockImplementation((k: string) => (k === "cost_profile" ? "coinbase_advanced_d6" : null));
    await renderForm();
    expect(costSelect().value).toBe("coinbase_advanced_d6");
  });

  it("ignores an unknown cost_profile query param", async () => {
    mockSearchGet.mockImplementation((k: string) => (k === "cost_profile" ? "nope" : null));
    await renderForm();
    expect(costSelect().value).toBe("");
  });
});

describe("New run: live submit never carries a stale costProfile (CR-04)", () => {
  it("picks coinbase_stress_t80, switches to live, confirms -> createRun body has no costProfile key", async () => {
    await renderForm();
    fireEvent.change(costSelect(), { target: { value: "coinbase_stress_t80" } });
    expect(costSelect().value).toBe("coinbase_stress_t80");

    // The page keeps the picked id in state while the selector is hidden in
    // live mode; only the body assembly stands between it and the wire.
    chooseMode("live");
    expect(screen.queryByLabelText("Cost profile")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Start Run/ }));
    // Live defers submission to LiveConfirmDialog; nothing is sent yet.
    expect(mockCreateRun).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText(/Live trading confirmation token/i), {
      target: { value: "typed-token" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Start Live Run" }));

    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const [body, options] = mockCreateRun.mock.calls[0] as [
      Record<string, unknown>,
      Record<string, unknown>,
    ];
    expect(body).toMatchObject({ mode: "live", allowPyramiding: false });
    expect("costProfile" in body).toBe(false);
    // The confirmation token travels in the options (header), never the body.
    expect(options).toMatchObject({ liveConfirmToken: "typed-token" });
    expect(JSON.stringify(body)).not.toContain("typed-token");
  });
});
