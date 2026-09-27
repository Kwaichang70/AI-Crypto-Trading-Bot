/**
 * apps/ui/src/__tests__/components/resume-run-dialog.test.tsx
 * -----------------------------------------------------------------
 * WP1.7b round 2 (UI-02 coverage gap, closed): `ResumeRunDialog` requires
 * the live-trading confirmation token for BOTH `mode=normal` and
 * `mode=protective` (`LiveTradingGate.check_gate` is unconditional in
 * `resume_run`, runs.py:2463-2474) — sent only as the
 * `X-Live-Confirm-Token` request header (never in the body), and cleared
 * when the token dialog closes.
 */

import React from "react";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { ResumeRunDialog } from "@/components/resume-run-dialog";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

const RUN_ID = "22222222-2222-2222-2222-222222222222";

function mockFetchOk() {
  const fetchMock: jest.Mock = jest.fn(() =>
    Promise.resolve(
      fakeJsonResponse({
        id: RUN_ID,
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
    ),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("ResumeRunDialog", () => {
  it("requires the token for the default 'protective' mode, and sends it only as a header", async () => {
    const fetchMock = mockFetchOk();
    const onResumed = jest.fn();
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={onResumed} />);

    fireEvent.click(screen.getByRole("button", { name: /Resume \(protective\)/ }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("mode=protective");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBe("typed-secret");
    // Never in the body.
    expect(JSON.stringify(init.body)).not.toContain("typed-secret");
    await waitFor(() => expect(onResumed).toHaveBeenCalledTimes(1));
  });

  it("also requires the token for 'normal' mode (LiveTradingGate is unconditional)", async () => {
    const fetchMock = mockFetchOk();
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={jest.fn()} />);

    fireEvent.click(screen.getByLabelText(/^Normal/));
    fireEvent.click(screen.getByRole("button", { name: /Resume \(normal\)/ }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret-normal" } });
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("mode=normal");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBe("typed-secret-normal");
  });

  it("clears the token field when the token dialog is cancelled", async () => {
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={jest.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /Resume \(protective\)/ }));
    const tokenInput = (await screen.findByLabelText(
      /Live trading confirmation token/i,
    )) as HTMLInputElement;
    fireEvent.change(tokenInput, { target: { value: "leftover-token" } });
    expect(tokenInput.value).toBe("leftover-token");

    // Two "Cancel" buttons coexist while the token dialog is open (the
    // outer ResumeRunDialog's own Cancel, and LiveConfirmDialog's) --
    // scope to the live-confirm dialog specifically.
    const liveDialog = screen.getByRole("dialog", { name: /Confirm resume/i });
    fireEvent.click(within(liveDialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(screen.queryByLabelText(/Live trading confirmation token/i)).not.toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("button", { name: /Resume \(protective\)/ }));
    const reopenedInput = (await screen.findByLabelText(
      /Live trading confirmation token/i,
    )) as HTMLInputElement;
    expect(reopenedInput.value).toBe("");
  });
});
