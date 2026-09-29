/**
 * apps/ui/src/__tests__/components/kill-switch-panel-polling.test.tsx
 * ------------------------------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-25, G-6, UT-15): the
 * kill-switch panel now derives `status`/`unavailable` from `usePolling`,
 * and `refetch()` (used after a clear) is not dropped by the in-flight
 * guard even while the periodic tick is itself in flight.
 *
 * ROUND 2 (WP70-C-03, small carry-forward wired up): a successful PRESS
 * (via `<KillSwitchButton onPressed={...}>`) also triggers `poll.refetch()`
 * -- previously only `handleCleared` did.
 */

import React from "react";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { KillSwitchPanel } from "@/components/kill-switch-panel";
import type { KillSwitchStatus } from "@/lib/types";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

jest.mock("next-auth/react", () => ({
  useSession: () => ({ data: { user: { role: "admin" } }, status: "authenticated" }),
}));

const NORMAL: KillSwitchStatus = { latched: false, since: null, reason: null, source: "db" };

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("KillSwitchPanel — usePolling adoption (SY-70-25)", () => {
  it("shows 'Unavailable' on a poll failure, distinct from the loading state", async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve(fakeJsonResponse({ detail: "boom" }, 500)),
    ) as unknown as typeof fetch;

    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Unavailable")).toBeInTheDocument());
  });

  it("refetch() after Clear fires even while the periodic tick is still in flight (G-6)", async () => {
    const statusDeferred = deferred<Response>();
    let statusCallCount = 0;
    const clearResponse = {
      wasLatched: true,
      runsUnlatched: ["run-1"],
      runsKeptLatched: [],
    };

    global.fetch = jest.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/v1/emergency/kill-switch")) {
        statusCallCount += 1;
        if (statusCallCount === 1) {
          return Promise.resolve(fakeJsonResponse(NORMAL));
        }
        // The SECOND status call (triggered by refetch() after Clear) hangs
        // -- proves refetch() actually issued a NEW request rather than
        // being swallowed by an in-flight guard from a periodic tick.
        return statusDeferred.promise;
      }
      if (url.includes("/api/admin/kill-switch/clear")) {
        return Promise.resolve(fakeJsonResponse(clearResponse));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    }) as unknown as typeof fetch;

    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());
    expect(statusCallCount).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: "Clear Kill Switch" }));
    const reasonInput = await screen.findByPlaceholderText(/Reason \(3-500 characters\)/);
    fireEvent.change(reasonInput, { target: { value: "incident resolved" } });
    fireEvent.click(screen.getByRole("button", { name: "Clear Latch" }));

    await waitFor(() => expect(statusCallCount).toBe(2));

    await act(async () => {
      statusDeferred.resolve(fakeJsonResponse(NORMAL));
      await Promise.resolve();
    });
  });

  it("WP70-C-03: a successful press (KillSwitchButton) also fires poll.refetch()", async () => {
    const statusDeferred = deferred<Response>();
    let statusCallCount = 0;
    const pressResponse = {
      latched: true,
      latchPersisted: true,
      since: "2026-09-28T00:00:00Z",
      runsLatched: ["run-1"],
      orphanedLiveRunIds: [],
      resumingRunIds: [],
      flattenResults: {},
      errors: [],
    };

    global.fetch = jest.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/v1/emergency/kill-switch")) {
        statusCallCount += 1;
        if (statusCallCount === 1) {
          return Promise.resolve(fakeJsonResponse(NORMAL));
        }
        // The SECOND status call must be the one triggered by
        // `onPressed` -> `poll.refetch()`, not the next scheduled tick
        // (POLL_INTERVAL_MS is 15s, far longer than this test runs).
        return statusDeferred.promise;
      }
      if (url.includes("/api/admin/kill-switch")) {
        return Promise.resolve(fakeJsonResponse(pressResponse));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    }) as unknown as typeof fetch;

    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());
    expect(statusCallCount).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: "Global Kill Switch" }));
    fireEvent.change(screen.getByPlaceholderText("EMERGENCY STOP"), {
      target: { value: "EMERGENCY STOP" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Activate Kill Switch" }));

    await waitFor(() => expect(statusCallCount).toBe(2));

    await act(async () => {
      statusDeferred.resolve(fakeJsonResponse(NORMAL));
      await Promise.resolve();
    });
  });
});
