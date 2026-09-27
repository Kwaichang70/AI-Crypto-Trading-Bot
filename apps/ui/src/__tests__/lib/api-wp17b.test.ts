/**
 * apps/ui/src/__tests__/lib/api-wp17b.test.ts
 * -----------------------------------------------
 * WP1.7a/1.7b (spec §6, AC8):
 *   - `createRun` sends the live-trading confirmation token as the
 *     `X-Live-Confirm-Token` HEADER, never in the JSON body.
 *   - `stopRun` / `emergencyStop` use timeouts >= 60s (AC8) — a flatten
 *     pass can run for up to ~30s plus a 5s outer margin server-side.
 */

import { createRun, stopRun, emergencyStop, STOP_TIMEOUT_MS, EMERGENCY_STOP_TIMEOUT_MS } from "@/lib/api";
import { ADMIN_MIN_TIMEOUT_MS } from "@/lib/admin-fetch";
import type { RunCreateRequest } from "@/lib/types";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

const BODY: RunCreateRequest = {
  strategyName: "grid_trading",
  strategyParams: {},
  symbols: ["BTC/EUR"],
  timeframe: "1h",
  mode: "live",
  initialCapital: "1000",
};

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(fakeJsonResponse(body, status));
}

describe("createRun — WP1.7a/SY-10 header-only live confirmation", () => {
  it("sends the token as X-Live-Confirm-Token and never in the body", async () => {
    const fetchMock: jest.Mock = jest.fn(() => jsonResponse({ id: "run-1" }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await createRun(BODY, "typed-secret-token");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBe("typed-secret-token");

    const sentBody = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(sentBody).not.toHaveProperty("confirmToken");
    expect(JSON.stringify(sentBody)).not.toContain("typed-secret-token");
  });

  it("omits the header entirely when no token is supplied (non-live runs)", async () => {
    const fetchMock: jest.Mock = jest.fn(() => jsonResponse({ id: "run-1" }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await createRun({ ...BODY, mode: "backtest" });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBeUndefined();
  });
});

describe("AC8 — stop / emergency-stop / kill-switch timeouts are >= 60s", () => {
  it("STOP_TIMEOUT_MS meets the 60s floor", () => {
    expect(STOP_TIMEOUT_MS).toBeGreaterThanOrEqual(60_000);
  });

  it("EMERGENCY_STOP_TIMEOUT_MS meets the 60s floor", () => {
    expect(EMERGENCY_STOP_TIMEOUT_MS).toBeGreaterThanOrEqual(60_000);
  });

  it("ADMIN_MIN_TIMEOUT_MS (kill-switch/resume/entries-latch proxies) meets the 60s floor", () => {
    expect(ADMIN_MIN_TIMEOUT_MS).toBeGreaterThanOrEqual(60_000);
  });

  it("stopRun actually configures its AbortController with the >= 60s timeout", async () => {
    jest.useFakeTimers();
    const setTimeoutSpy = jest.spyOn(global, "setTimeout");
    const fetchMock = jest.fn(
      () => new Promise<Response>(() => {}), // never resolves — we only inspect the timer
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    void stopRun("run-1", { flatten: true });
    // Flush the microtask queue so apiFetch's setTimeout(...) call happens.
    await Promise.resolve();

    const timeoutCalls = setTimeoutSpy.mock.calls.filter(
      (call) => typeof call[1] === "number" && call[1] >= 60_000,
    );
    expect(timeoutCalls.length).toBeGreaterThan(0);

    setTimeoutSpy.mockRestore();
    jest.useRealTimers();
  });
});

describe("emergencyStop", () => {
  it("sends the optional reason as X-Emergency-Reason and flatten as a query param", async () => {
    const fetchMock: jest.Mock = jest.fn(() => jsonResponse({ id: "run-1", status: "stopped" }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await emergencyStop("run-1", { flatten: true, reason: "incident-123" });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/emergency-stop?flatten=true");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Emergency-Reason"]).toBe("incident-123");
  });
});
