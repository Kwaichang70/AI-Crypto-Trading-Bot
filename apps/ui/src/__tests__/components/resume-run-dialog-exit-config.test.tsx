/**
 * apps/ui/src/__tests__/components/resume-run-dialog-exit-config.test.tsx
 * ----------------------------------------------------------------------------
 * WP1.3a (CF-13a-1 item 2/6): `<ResumeRunDialog>` renders the shared
 * structured exit-config/pyramiding 422 panel for a NORMAL-mode resume
 * (`invalid_exit_config` / `exit_manager_required` / `live_pyramiding_forbidden`,
 * synthesis-spec.md §5/§9/SY-13a-18), with the generic message as fallback,
 * using the real `{"detail": {...}}` envelope `adminFetch` surfaces via
 * `ApiError.detail`.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ResumeRunDialog } from "@/components/resume-run-dialog";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

const RUN_ID = "55555555-5555-5555-5555-555555555555";

function mockFetch422(detail: unknown) {
  const fetchMock: jest.Mock = jest.fn(() =>
    Promise.resolve(fakeJsonResponse({ detail }, 422)),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

async function openNormalModeAndTypeToken() {
  fireEvent.click(screen.getByLabelText(/^Normal/));
  fireEvent.click(screen.getByRole("button", { name: /Resume \(normal\)/ }));
  const tokenInput = await screen.findByLabelText(/Live trading confirmation token/i);
  fireEvent.change(tokenInput, { target: { value: "typed-secret" } });
  fireEvent.click(screen.getByRole("button", { name: "Resume" }));
}

describe("ResumeRunDialog — structured exit-config 422 (normal mode)", () => {
  it("renders invalid_exit_config errors", async () => {
    mockFetch422({
      code: "invalid_exit_config",
      errors: [
        {
          field: "trailing_stop_pct",
          reason: "out_of_range",
          value: "0.6",
          min: 0.005,
          max: 0.5,
          message: "Trailing stop must be in [0.5%, 50%].",
        },
      ],
      warnings: [],
    });
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={jest.fn()} />);

    await openNormalModeAndTypeToken();

    await waitFor(() => expect(screen.getByText("Invalid exit configuration")).toBeInTheDocument());
    expect(screen.getByText(/Trailing stop must be in/)).toBeInTheDocument();
  });

  it("renders live_pyramiding_forbidden for a legacy live dca_rsi_hybrid normal resume", async () => {
    mockFetch422({
      code: "live_pyramiding_forbidden",
      strategy: "dca_rsi_hybrid",
      hint: "strategy accumulates by design; pass allowPyramiding=false to run single-entry (unvalidated), or use paper",
      errors: [],
    });
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={jest.fn()} />);

    await openNormalModeAndTypeToken();

    await waitFor(() => expect(screen.getByText("Live pyramiding is forbidden")).toBeInTheDocument());
    expect(screen.getByText("dca_rsi_hybrid")).toBeInTheDocument();
  });

  it("falls back to the generic message for a 409 kill-switch-active detail", async () => {
    mockFetch422({ code: "kill_switch_active" });
    // adminFetch treats any non-ok response the same way regardless of the
    // actual status code sent by fakeJsonResponse -- what matters is that
    // `code` isn't one of the three exit-config codes.
    render(<ResumeRunDialog runId={RUN_ID} onClose={jest.fn()} onResumed={jest.fn()} />);

    await openNormalModeAndTypeToken();

    await waitFor(() =>
      expect(screen.getByText("Request failed (HTTP 422)")).toBeInTheDocument(),
    );
  });
});
