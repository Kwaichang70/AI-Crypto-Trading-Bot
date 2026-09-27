/**
 * apps/ui/src/__tests__/components/stop-run-dialog.test.tsx
 * -----------------------------------------------------------------
 * WP1.7a SY-05 / WP1.7b CF-B3 (spec §6): a running LIVE run's Stop button
 * must stay disabled until the operator picks an explicit flatten/keep
 * choice; non-live runs keep the original one-click behaviour.
 *
 * WP1.7b round 2:
 *   - WP17b-C-01/S-01 (critical): every 409/422 mock below wraps its body
 *     in FastAPI's REAL `{"detail": {...}}` HTTPException envelope — the
 *     round-1 mocks supplied the unwrapped `{code, ...}` shape directly,
 *     which never happens against the real backend and masked the bug
 *     (`apiFetch` stores the WHOLE parsed body as `error.detail`, so a real
 *     409 arrives as `{detail: {code, flatten}}`, one level deeper).
 *   - WP17b-S-02: a stale/omitted flatten choice must never silently
 *     become `flatten=false` for a run this dialog cannot fully vouch for
 *     as non-live/non-running.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { StopRunDialog } from "@/components/stop-run-dialog";
import * as apiModule from "@/lib/api";
import type { Run } from "@/lib/types";
import { fakeJsonResponse, fakeTextResponse } from "@/__tests__/test-utils/fake-response";

jest.mock("next-auth/react", () => ({
  useSession: () => ({ data: { user: { role: "admin" } }, status: "authenticated" }),
}));

// WP17b-S-R2-02 (round 3): `stopRun` is wrapped in a jest.fn() that DEFAULTS
// to the real implementation (so every existing fetch-mock-based test below
// keeps working unchanged) -- only the dedicated "safety net" test at the
// bottom overrides it (once) to REJECT, to prove `doStop`'s try/finally
// clears `loading` even when the call itself throws, not just when it
// resolves with an error `ApiResult`.
jest.mock("@/lib/api", () => {
  const actual = jest.requireActual("@/lib/api");
  return { ...actual, stopRun: jest.fn(actual.stopRun) };
});

function makeRun(overrides: Partial<Run> = {}): Run {
  return {
    id: "11111111-1111-1111-1111-111111111111",
    runMode: "live",
    status: "running",
    config: {
      strategy_name: "grid_trading",
      strategy_params: {},
      symbols: ["BTC/EUR"],
      timeframe: "1h",
      mode: "live",
      initial_capital: "1000",
    },
    startedAt: "2026-01-01T00:00:00Z",
    stoppedAt: null,
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

/** FastAPI's real `HTTPException(status, detail={...})` wire shape. */
function fakeHttpExceptionResponse(detail: unknown, status: number) {
  return fakeJsonResponse({ detail }, status);
}

describe("StopRunDialog — live running run requires an explicit flatten choice", () => {
  it("disables Stop Run until a radio is chosen, and sends the explicitly chosen value", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeJsonResponse({
          ...makeRun({ status: "stopped" }),
          flatten: null,
          unprotectedPositions: [],
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const onStopped = jest.fn();
    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={onStopped} />,
    );

    const stopButton = screen.getByRole("button", { name: "Stop Run" });
    expect(stopButton).toBeDisabled();

    fireEvent.click(screen.getByLabelText(/Keep positions/i));
    expect(stopButton).not.toBeDisabled();

    fireEvent.click(stopButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    // An EXPLICIT user choice ("Keep positions") is sent as an explicit
    // `flatten=false` — this is not the S-02 "stale status" case.
    expect(url).toContain("?flatten=false");
    await waitFor(() => expect(onStopped).toHaveBeenCalledTimes(1));
  });

  it("enables Stop Run immediately for a non-live run and omits flatten entirely (no choice was shown)", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeJsonResponse({
          ...makeRun({ runMode: "paper", status: "stopped" }),
          flatten: null,
          unprotectedPositions: [],
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog
        run={makeRun({ runMode: "paper" })}
        positions={[]}
        onClose={jest.fn()}
        onStopped={jest.fn()}
      />,
    );
    const stopButton = screen.getByRole("button", { name: "Stop Run" });
    expect(stopButton).not.toBeDisabled();
    expect(screen.queryByLabelText(/Flatten \(sell everything\)/i)).not.toBeInTheDocument();

    fireEvent.click(stopButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    // WP17b-S-02: no choice was ever shown/made -- `flatten` must be
    // OMITTED entirely, never defaulted to `false` client-side.
    expect(url).not.toContain("flatten=");
  });

  it("WP17b-S-02: a live run shown as 'resuming' (stale status) must NOT send flatten=false", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeJsonResponse({
          ...makeRun({ status: "orphaned" }),
          flatten: null,
          unprotectedPositions: [],
        }),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog
        run={makeRun({ status: "resuming" })}
        positions={[]}
        onClose={jest.fn()}
        onStopped={jest.fn()}
      />,
    );

    // No radio is shown -- the dialog only knows to force the choice UI
    // once the backend itself says so via a 422 (see the next test).
    expect(screen.queryByLabelText(/Flatten \(sell everything\)/i)).not.toBeInTheDocument();
    const stopButton = screen.getByRole("button", { name: "Stop Run" });
    expect(stopButton).not.toBeDisabled();

    fireEvent.click(stopButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).not.toContain("flatten=false");
    expect(url).not.toContain("flatten=");
  });

  it("WP17b-S-01/S-02: a real 422 flatten_decision_required envelope forces the choice UI to render, even for a stale non-'running' status", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeHttpExceptionResponse(
          { code: "flatten_decision_required", held_symbols: ["BTC/EUR"] },
          422,
        ),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog
        run={makeRun({ status: "resuming" })}
        positions={[]}
        onClose={jest.fn()}
        onStopped={jest.fn()}
      />,
    );

    expect(screen.queryByLabelText(/Flatten \(sell everything\)/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    // The 422 must force the radio-choice UI to appear, and show the held
    // symbols the backend reported.
    await screen.findByLabelText(/Flatten \(sell everything\)/i);
    expect(screen.getByText(/BTC\/EUR/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop Run" })).toBeDisabled();
  });

  it("renders the 409 flatten_incomplete result (real FastAPI envelope) and offers retry / stop-without-flatten", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeHttpExceptionResponse(
          {
            code: "flatten_incomplete",
            flatten: {
              runId: "11111111-1111-1111-1111-111111111111",
              outcome: "partial",
              complete: false,
              latchPersisted: true,
              symbols: [
                {
                  symbol: "BTC/EUR",
                  status: "partial",
                  cause: "ledger_doubt",
                  heldBefore: "0.01",
                  soldQty: "0.005",
                  remainingQty: "0.005",
                  orderIds: [],
                  error: null,
                },
              ],
            },
          },
          409,
        ),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={jest.fn()} />,
    );

    fireEvent.click(screen.getByLabelText(/Flatten \(sell everything\)/i));
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    await screen.findByText(/Partially flattened/i);
    expect(screen.getByRole("button", { name: /Retry flatten/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Stop without flatten/i })).toBeInTheDocument();
  });

  it("renders a flatten_requires_running_engine error (real FastAPI envelope) as a plain message", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeHttpExceptionResponse({ code: "flatten_requires_running_engine" }, 409),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={jest.fn()} />,
    );

    fireEvent.click(screen.getByLabelText(/Flatten \(sell everything\)/i));
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    await screen.findByText(/flatten cannot run/i);
    // Must not be confused with the generic HTTP-status fallback message.
    expect(screen.queryByText(/Conflict — the request could not be completed/i)).not.toBeInTheDocument();
  });
});

describe("StopRunDialog — WP17b-S-R2-02 (round 3 regression): string/non-object `detail` must never freeze the dialog", () => {
  it("a 409 with a plain STRING detail shows an error message and leaves Stop and Cancel usable", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeJsonResponse(
          { detail: "Cannot stop run 11111111-...: current status is 'stopped'." },
          409,
        ),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={jest.fn()} />,
    );

    fireEvent.click(screen.getByLabelText(/Flatten \(sell everything\)/i));
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    // Before the fix, `"code" in "<string>"` threw and the dialog froze on
    // "Stopping…" with both buttons permanently disabled.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Stop Run" })).not.toBeDisabled(),
    );
    expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled();
    expect(screen.queryByText(/Stopping…/)).not.toBeInTheDocument();
  });

  it("a non-JSON text/plain 500 shows an error message and leaves Stop and Cancel usable", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(fakeTextResponse("Internal Server Error", 500)),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={jest.fn()} />,
    );

    fireEvent.click(screen.getByLabelText(/Flatten \(sell everything\)/i));
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Stop Run" })).not.toBeDisabled(),
    );
    expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled();
    expect(screen.queryByText(/Stopping…/)).not.toBeInTheDocument();
  });

  it("clears loading via try/finally even when stopRun itself rejects (not just an error ApiResult)", async () => {
    (apiModule.stopRun as jest.Mock).mockRejectedValueOnce(new Error("boom"));

    render(
      <StopRunDialog run={makeRun()} positions={[]} onClose={jest.fn()} onStopped={jest.fn()} />,
    );

    fireEvent.click(screen.getByLabelText(/Keep positions/i));
    fireEvent.click(screen.getByRole("button", { name: "Stop Run" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Stop Run" })).not.toBeDisabled(),
    );
    expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled();
  });
});
