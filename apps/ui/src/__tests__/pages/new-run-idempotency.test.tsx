/**
 * apps/ui/src/__tests__/pages/new-run-idempotency.test.tsx
 * -----------------------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-19..22, UT-19..25):
 * the new-run form's `Idempotency-Key` lifecycle end to end, mocking
 * `@/lib/api`'s `createRun` (never the network) the same way
 * `new-run-exit-config.test.tsx` does.
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

async function renderAndReachStartButton() {
  mockFetchStrategies.mockResolvedValue(strategiesResult([GRID]));
  render(<NewRunPage />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Start Run" })).toBeInTheDocument());
}

function keyOf(callIndex: number): string {
  const [, opts] = mockCreateRun.mock.calls[callIndex] as [unknown, { idempotencyKey: string }];
  return opts.idempotencyKey;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockSearchGet.mockReturnValue(null);
});

describe("NewRunPage — Idempotency-Key options object (G-4, AC8)", () => {
  it("sends createRun(body, {idempotencyKey}) — never a bare string 2nd argument", async () => {
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1", configWarnings: [] } });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));

    const [, opts] = mockCreateRun.mock.calls[0] as [unknown, unknown];
    expect(typeof opts).toBe("object");
    expect(opts).not.toBeNull();
    expect(typeof opts).not.toBe("string");
    expect((opts as { idempotencyKey: string }).idempotencyKey).toMatch(/^[0-9a-f-]{36}$/);
  });
});

describe("NewRunPage — key lifecycle (SY-70-19, UT-19)", () => {
  it("success, then a form change, then submit again -> a NEW key (UT-19)", async () => {
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-1", configWarnings: [] } });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const firstKey = keyOf(0);
    expect(mockPush).toHaveBeenCalledWith("/runs/run-1");

    // A form change (initial capital) changes the JSON-stringified body
    // snapshot, so the next submit must mint a fresh key. The label and
    // input are siblings (no htmlFor/id), so query via the DOM like
    // `new-run-exit-config.test.tsx`'s own `paramInput` helper does.
    const capitalInput = screen
      .getByText("Initial Capital (USD)")
      .closest("div")!
      .querySelector("input") as HTMLInputElement;
    fireEvent.change(capitalInput, { target: { value: "20000" } });
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));
    const secondKey = keyOf(1);
    expect(secondKey).not.toBe(firstKey);
  });

  it("keeps the same key across a network-error (status 0) retry (UT-18/25)", async () => {
    mockCreateRun.mockResolvedValueOnce({
      ok: false,
      error: { status: 0, message: "Request timed out. The operation may still be running on the server." },
    });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const firstKey = keyOf(0);

    mockCreateRun.mockResolvedValueOnce({ ok: true, data: { id: "run-2", configWarnings: [] } });
    fireEvent.click(screen.getByRole("button", { name: "Retry with the same key" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));
    expect(keyOf(1)).toBe(firstKey);
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/runs/run-2"));
  });

  it("keeps the key after 403 / WP1.3a 422 / kill_switch_active 409 / 502 (UT-25)", async () => {
    await renderAndReachStartButton();

    const codes: Array<{ status: number; detail: unknown }> = [
      { status: 403, detail: { code: "live_trading_disabled" } },
      // `invalid_exit_config` is a KNOWN_CODES member for
      // <ExitConfigErrorPanel> -- it must carry `errors`/`warnings` or the
      // panel throws trying to read `.errors.length` on `undefined`.
      { status: 422, detail: { code: "invalid_exit_config", errors: [], warnings: [] } },
      { status: 409, detail: { code: "kill_switch_active" } },
      { status: 502, detail: { code: "upstream_error" } },
    ];

    let previousKey: string | null = null;
    for (let i = 0; i < codes.length; i++) {
      const { status, detail } = codes[i]!;
      mockCreateRun.mockResolvedValueOnce({
        ok: false,
        error: { status, message: "failed", detail },
      });
      fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
      await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(i + 1));
      const thisKey = keyOf(i);
      if (previousKey !== null) expect(thisKey).toBe(previousKey);
      previousKey = thisKey;
    }
  });
});

describe("NewRunPage — 409 idempotency_in_progress (SY-70-21/C-13, UT-21)", () => {
  it("shows a neutral status notice, keeps the key, and 'Check again' resubmits with it", async () => {
    mockCreateRun.mockResolvedValueOnce({
      ok: false,
      error: { status: 409, message: "conflict", detail: { code: "idempotency_in_progress" } },
    });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const firstKey = keyOf(0);

    const notice = await screen.findByRole("status");
    expect(notice).toHaveTextContent("Your previous request is still being processed…");
    expect(notice).toHaveAttribute("aria-live", "polite");

    // Start Run is re-enabled (isSubmitting released).
    expect(screen.getByRole("button", { name: "Start Run" })).not.toBeDisabled();

    mockCreateRun.mockResolvedValueOnce({ ok: true, data: { id: "run-9", configWarnings: [] } });
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));
    expect(keyOf(1)).toBe(firstKey);
  });
});

describe("NewRunPage — 422 idempotency_key_reused (SY-70-19, UT-22)", () => {
  it("shows an error and the next submit gets a new key", async () => {
    mockCreateRun.mockResolvedValueOnce({
      ok: false,
      error: { status: 422, message: "conflict", detail: { code: "idempotency_key_reused" } },
    });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const firstKey = keyOf(0);
    await waitFor(() =>
      expect(
        screen.getByText("This request was already used for a different submission. Please try again."),
      ).toBeInTheDocument(),
    );

    mockCreateRun.mockResolvedValueOnce({ ok: true, data: { id: "run-10", configWarnings: [] } });
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));
    expect(keyOf(1)).not.toBe(firstKey);
  });
});

describe("NewRunPage — 428/400 idempotency client-error codes (SY-70-21, UT-23)", () => {
  it.each([
    ["idempotency_key_required", 428],
    ["idempotency_key_invalid_format", 400],
  ])("code=%s status=%i -> generic message, console.error(code), reset, no auto-retry", async (code, status) => {
    const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation(() => {});
    mockCreateRun.mockResolvedValueOnce({
      ok: false,
      error: { status, message: "client bug", detail: { code } },
    });
    await renderAndReachStartButton();

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    const firstKey = keyOf(0);

    await waitFor(() =>
      expect(
        screen.getByText(
          "Client error: the request was sent without a valid retry token. Reload the page and try again.",
        ),
      ).toBeInTheDocument(),
    );
    expect(consoleErrorSpy).toHaveBeenCalledWith("idempotency client error", code);
    // No automatic retry call was made beyond the original submit.
    expect(mockCreateRun).toHaveBeenCalledTimes(1);

    mockCreateRun.mockResolvedValueOnce({ ok: true, data: { id: "run-11", configWarnings: [] } });
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }));
    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(2));
    expect(keyOf(1)).not.toBe(firstKey);

    consoleErrorSpy.mockRestore();
  });
});

describe("NewRunPage — synchronous double-click submit lock (SY-70-20, UT-20)", () => {
  it("two synchronous clicks result in exactly one createRun call (DOM disabled-attribute path)", async () => {
    let resolveCreate: (value: unknown) => void = () => {};
    mockCreateRun.mockImplementationOnce(
      () => new Promise((resolve) => { resolveCreate = resolve; }),
    );
    await renderAndReachStartButton();

    const button = screen.getByRole("button", { name: "Start Run" });
    fireEvent.click(button);
    fireEvent.click(button);
    fireEvent.click(button);

    await waitFor(() => expect(mockCreateRun).toHaveBeenCalledTimes(1));
    resolveCreate({ ok: true, data: { id: "run-x", configWarnings: [] } });
    await waitFor(() => expect(mockPush).toHaveBeenCalled());
  });

  // WP70-C-02 (round 2): the test above is NOT load-bearing for
  // `useSubmitLock` on its own -- `fireEvent.click()` triggers React's
  // synchronous discrete-event flush, so `isSubmitting` (and the button's
  // DOM `disabled` attribute) is already committed before the 2nd/3rd
  // `fireEvent.click()` line even runs; the native "a disabled button does
  // not dispatch click" behaviour alone would make this test pass even
  // with `tryAcquire()` hard-coded to always return `true` (verified: see
  // the revert-and-fail check in the round-2 producer report).
  //
  // This test isolates the ref-lock itself: it dispatches TWO `submit`
  // events directly on the <form> (bypassing the submit button, and thus
  // its `disabled` attribute, entirely -- exactly how a real double-firing
  // channel would look, e.g. an Enter keypress racing a click, or a
  // synthetic re-dispatch) with NO `await`/act flush between them, so both
  // reach `handleSubmit` -> `submitCreateRun` before either's internal
  // `await createRun(...)` has a chance to yield. If `useSubmitLock` were
  // disabled, BOTH calls would reach the mocked `createRun` synchronously,
  // before this test's own assertion runs.
  it("two synchronous form submits (bypassing the button's disabled attribute) still result in exactly one createRun call", async () => {
    let resolveCreate: (value: unknown) => void = () => {};
    // A safe fallback for any call beyond the first (only reached if the
    // lock is broken) -- avoids an unhandled-rejection crash from
    // `result.ok` on an unconfigured `undefined` return, which would
    // otherwise mask this test's own assertion under the mutation.
    mockCreateRun.mockResolvedValue({ ok: true, data: { id: "run-fallback", configWarnings: [] } });
    mockCreateRun.mockImplementationOnce(
      () => new Promise((resolve) => { resolveCreate = resolve; }),
    );
    await renderAndReachStartButton();

    const form = screen.getByRole("button", { name: "Start Run" }).closest("form") as HTMLFormElement;
    expect(form).not.toBeNull();

    // No `await`/`waitFor` between these two -- both fire synchronously in
    // the same task, well before `submitCreateRun`'s own `await
    // createRun(...)` yields control back to the microtask queue.
    fireEvent.submit(form);
    fireEvent.submit(form);

    // Synchronous assertion, deliberately NOT wrapped in `waitFor()`: under
    // correct code the SECOND `submitCreateRun` invocation returns
    // immediately from `if (!submitLock.tryAcquire()) return;`, so
    // `createRun` has only ever been called once by this point, with no
    // need to wait for anything to settle.
    expect(mockCreateRun).toHaveBeenCalledTimes(1);

    resolveCreate({ ok: true, data: { id: "run-x", configWarnings: [] } });
    await waitFor(() => expect(mockPush).toHaveBeenCalled());
  });
});
