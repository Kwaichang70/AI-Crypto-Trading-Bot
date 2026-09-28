/**
 * apps/ui/src/__tests__/lib/api-wp17b.test.ts
 * -----------------------------------------------
 * WP1.7a/1.7b (spec §6, AC8):
 *   - `createRun` sends the live-trading confirmation token as the
 *     `X-Live-Confirm-Token` HEADER, never in the JSON body.
 *   - `stopRun` / `emergencyStop` use timeouts >= 60s (AC8) — a flatten
 *     pass can run for up to ~30s plus a 5s outer margin server-side.
 *
 * WP7.0 (SY-70-18, G-4, UT-16): `createRun`/`promoteRun` now take a
 * `RunSubmitOptions` object (`{idempotencyKey, liveConfirmToken?}`) as their
 * 2nd argument, never a bare string — this file's own `createRun(BODY,
 * "typed-secret-token")` call was G-4's worked example of the positional
 * signature's misbinding hazard, so it is the first thing WP7.0 must fix.
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

const IDEMPOTENCY_KEY = "11111111-2222-4333-8444-555555555555";

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(fakeJsonResponse(body, status));
}

describe("createRun — WP1.7a/SY-10 header-only live confirmation, WP7.0 options object", () => {
  it("sends the token as X-Live-Confirm-Token and never in the body, and the key as Idempotency-Key", async () => {
    const fetchMock: jest.Mock = jest.fn(() => jsonResponse({ id: "run-1" }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await createRun(BODY, { idempotencyKey: IDEMPOTENCY_KEY, liveConfirmToken: "typed-secret-token" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBe("typed-secret-token");
    expect(headers["Idempotency-Key"]).toBe(IDEMPOTENCY_KEY);
    // G-4: the two headers must never collide.
    expect(headers["Idempotency-Key"]).not.toBe(headers["X-Live-Confirm-Token"]);

    const sentBody = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(sentBody).not.toHaveProperty("confirmToken");
    expect(JSON.stringify(sentBody)).not.toContain("typed-secret-token");
    expect(JSON.stringify(sentBody)).not.toContain(IDEMPOTENCY_KEY);
  });

  it("omits X-Live-Confirm-Token entirely when no token is supplied (non-live runs), but always sends Idempotency-Key", async () => {
    const fetchMock: jest.Mock = jest.fn(() => jsonResponse({ id: "run-1" }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await createRun({ ...BODY, mode: "backtest" }, { idempotencyKey: IDEMPOTENCY_KEY });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBeUndefined();
    expect(headers["Idempotency-Key"]).toBe(IDEMPOTENCY_KEY);
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
