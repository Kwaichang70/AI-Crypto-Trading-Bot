/**
 * apps/ui/src/__tests__/components/promote-run-dialog-idempotency.test.tsx
 * ------------------------------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-19..22, UT-24/UT-26) —
 * ROUND 2 (WP70-S-03): the promote dialog mints/reuses an
 * `Idempotency-Key` snapshotted on `sourceRunId`, and closes the token
 * dialog (`<LiveConfirmDialog>` fully unmounts — it `return`s `null` while
 * `open=false`) on EVERY settled non-OK branch (status 0, reused,
 * client_error, and any other code), not just success/in-progress:
 *   - the typed token must not stay in the DOM once an attempt has settled
 *     (I9) -- asserted below by re-querying the token input and getting
 *     `null`/a FRESH, empty field, never the previous value;
 *   - the error panel must render on the now-visible underlying dialog,
 *     not behind the (now-unmounted) overlay -- asserted below by
 *     confirming the `<LiveConfirmDialog>` `role="dialog"` node is gone
 *     from the document while the error text is present;
 *   - "Retry with the same key" (clicking "Promote…" again) reopens
 *     `<LiveConfirmDialog>` with an EMPTY field, never the stale token.
 *
 * Uses the real `fetch` mock (like `promote-run-dialog.test.tsx`) rather
 * than mocking `@/lib/api`, so the header assertions are end-to-end.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { PromoteRunDialog } from "@/components/promote-run-dialog";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

const SOURCE_RUN_ID = "33333333-3333-3333-3333-333333333333";

function headersOfCall(fetchMock: jest.Mock, index: number): Record<string, string> {
  const [, init] = fetchMock.mock.calls[index] as [string, RequestInit];
  return init.headers as Record<string, string>;
}

async function openDialogAndTypeToken(token: string) {
  fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
  const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
  fireEvent.change(tokenInput, { target: { value: token } });
  fireEvent.click(screen.getByRole("button", { name: "Promote" }));
}

describe("PromoteRunDialog — Idempotency-Key lifecycle (SY-70-19/22, UT-26)", () => {
  it("double-clicking Promote (two quick confirms) results in exactly one network call", async () => {
    let resolveFetch: (value: Response) => void = () => {};
    const fetchMock: jest.Mock = jest.fn(
      () => new Promise<Response>((resolve) => { resolveFetch = resolve; }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    const confirmButton = screen.getByRole("button", { name: "Promote" });
    fireEvent.click(confirmButton);
    fireEvent.click(confirmButton);
    fireEvent.click(confirmButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    resolveFetch(
      fakeJsonResponse({
        id: "44444444-4444-4444-4444-444444444444",
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
      }),
    );
  });

  it("WP70-S-03/UT-24: a live-timeout (status 0) retry CLOSES the token dialog, shows the error, then re-opens EMPTY with the SAME key", async () => {
    const fetchMock: jest.Mock = jest.fn(() => Promise.reject(new Error("network down")));
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);
    await openDialogAndTypeToken("first-typed-token");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const firstKey = headersOfCall(fetchMock, 0)["Idempotency-Key"];
    const firstToken = headersOfCall(fetchMock, 0)["X-Live-Confirm-Token"];
    expect(firstToken).toBe("first-typed-token");

    // WP70-S-03: the token dialog must be GONE (fully unmounted, not just
    // covered) once the attempt has settled -- the typed token cannot
    // linger in the DOM, and the error panel underneath must be visible.
    await waitFor(() =>
      expect(screen.queryByLabelText(/Live trading confirmation token/i)).not.toBeInTheDocument(),
    );
    expect(screen.queryByRole("dialog", { name: "Confirm promotion to live" })).not.toBeInTheDocument();
    expect(screen.getByText(/network error occurred|Request timed out|network down/i)).toBeInTheDocument();

    // "Retry with the same key": click "Promote…" again -- this must
    // re-open <LiveConfirmDialog> with a FRESH, EMPTY field, never the
    // previous "first-typed-token" value.
    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const reopenedInput = await screen.findByLabelText(/Live trading confirmation token/i);
    expect((reopenedInput as HTMLInputElement).value).toBe("");

    fireEvent.change(reopenedInput, { target: { value: "second-typed-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const secondKey = headersOfCall(fetchMock, 1)["Idempotency-Key"];
    const secondToken = headersOfCall(fetchMock, 1)["X-Live-Confirm-Token"];
    expect(secondKey).toBe(firstKey); // SY-70-19: status 0 keeps the key
    expect(secondToken).toBe("second-typed-token");
    expect(secondToken).not.toBe(firstToken);
  });

  it("WP70-S-03: 422 idempotency_key_reused also closes the dialog and re-opens EMPTY, with a NEW key", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(fakeJsonResponse({ detail: { code: "idempotency_key_reused" } }, 422)),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);
    await openDialogAndTypeToken("typed-token");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const firstKey = headersOfCall(fetchMock, 0)["Idempotency-Key"];

    await waitFor(() =>
      expect(screen.queryByLabelText(/Live trading confirmation token/i)).not.toBeInTheDocument(),
    );
    expect(
      screen.getByText("This request was already used for a different submission. Please try again."),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const reopenedInput = await screen.findByLabelText(/Live trading confirmation token/i);
    expect((reopenedInput as HTMLInputElement).value).toBe("");

    fireEvent.change(reopenedInput, { target: { value: "typed-token-2" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    // SY-70-19: 422 idempotency_key_reused resets the key -> the retry mints a NEW one.
    expect(headersOfCall(fetchMock, 1)["Idempotency-Key"]).not.toBe(firstKey);
  });

  it("UT-26: 409 idempotency_in_progress shows a neutral notice; 'Check again' re-opens the token dialog with the same key", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(fakeJsonResponse({ detail: { code: "idempotency_in_progress" } }, 409)),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);
    await openDialogAndTypeToken("typed-token");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const notice = await screen.findByRole("status");
    expect(notice).toHaveTextContent("Your previous request is still being processed…");
    const firstKey = headersOfCall(fetchMock, 0)["Idempotency-Key"];

    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    expect((tokenInput as HTMLInputElement).value).toBe("");
    fireEvent.change(tokenInput, { target: { value: "typed-token-2" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(headersOfCall(fetchMock, 1)["Idempotency-Key"]).toBe(firstKey);
  });
});
