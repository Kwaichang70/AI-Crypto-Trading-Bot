/**
 * apps/ui/src/__tests__/components/promote-run-dialog.test.tsx
 * ---------------------------------------------------------------
 * WP1.3a (CF-13a-1 item 2/6): `<PromoteRunDialog>` calls
 * `POST /api/v1/runs/{id}/promote-to-live` directly (no admin proxy — the
 * endpoint requires only X-Live-Confirm-Token), sends the token ONLY as a
 * header, and renders the shared structured 422 panel using the real
 * `{"detail": {...}}` FastAPI envelope.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { PromoteRunDialog } from "@/components/promote-run-dialog";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

const SOURCE_RUN_ID = "33333333-3333-3333-3333-333333333333";
const NEW_LIVE_RUN_ID = "44444444-4444-4444-4444-444444444444";

function mockFetchOk() {
  const fetchMock: jest.Mock = jest.fn(() =>
    Promise.resolve(
      fakeJsonResponse({
        id: NEW_LIVE_RUN_ID,
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

function mockFetch422(detail: unknown) {
  const fetchMock: jest.Mock = jest.fn(() =>
    Promise.resolve(fakeJsonResponse({ detail }, 422)),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("PromoteRunDialog", () => {
  it("calls promote-to-live directly (no /api/admin proxy) and sends the token only as a header", async () => {
    const fetchMock = mockFetchOk();
    const onPromoted = jest.fn();
    render(
      <PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={onPromoted} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`/api/v1/runs/${SOURCE_RUN_ID}/promote-to-live`);
    expect(url).not.toContain("/api/admin");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Live-Confirm-Token"]).toBe("typed-secret");
    expect(JSON.stringify(init.body)).not.toContain("typed-secret");
    await waitFor(() => expect(onPromoted).toHaveBeenCalledTimes(1));
    expect(onPromoted.mock.calls[0][0].id).toBe(NEW_LIVE_RUN_ID);
  });

  it("renders the structured exit_manager_required 422 detail", async () => {
    mockFetch422({
      code: "exit_manager_required",
      strategy: "momentum_breakout",
      requires_one_of: ["bracket_stop_loss_pct (bracket_mode=fixed)", "trailing_stop_pct"],
      errors: [],
      warnings: [],
    });
    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() =>
      expect(screen.getByText("This strategy requires a downside exit")).toBeInTheDocument(),
    );
    expect(screen.getByText("momentum_breakout")).toBeInTheDocument();
  });

  it("renders live_pyramiding_forbidden with its hint (paper dca/grid source with pyramiding on)", async () => {
    mockFetch422({
      code: "live_pyramiding_forbidden",
      strategy: "dca_rsi_hybrid",
      hint: "you promote only what you validated in paper; re-run paper with allowPyramiding=false first",
      errors: [],
    });
    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() => expect(screen.getByText("Live pyramiding is forbidden")).toBeInTheDocument());
    expect(screen.getByText(/re-run paper with allowPyramiding=false/)).toBeInTheDocument();
  });

  it("falls back to the generic message for a plain-string 400 (promotion gate not met)", async () => {
    const fetchMock: jest.Mock = jest.fn(() =>
      Promise.resolve(
        fakeJsonResponse(
          { detail: "Paper run ... is not eligible for promotion. trade_count=2 (min=10)." },
          400,
        ),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<PromoteRunDialog sourceRunId={SOURCE_RUN_ID} onClose={jest.fn()} onPromoted={jest.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Promote…" }));
    const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
    fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Promote" }));

    await waitFor(() =>
      expect(screen.getByText("Bad request — the server rejected the input.")).toBeInTheDocument(),
    );
  });
});
