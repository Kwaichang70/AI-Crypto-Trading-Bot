/**
 * apps/ui/src/__tests__/components/kill-switch-panel.test.tsx
 * -----------------------------------------------------------------
 * WP1.7b (CF-B1, spec §6): the kill-switch UI must render the WP1.7a
 * camelCase shapes end-to-end — `runsLatched`, `latchPersisted`,
 * `orphanedLiveRunIds`, `resumingRunIds`, `flattenResults`, `errors` on
 * press; `wasLatched`/`runsUnlatched`/`runsKeptLatched` on clear — and must
 * NEVER reference the old Sprint 50 cycle 3 shape (`runs_stopped`,
 * `tasks_cancelled`).
 */

import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { KillSwitchPanel } from "@/components/kill-switch-panel";
import type { KillSwitchPressResponse, KillSwitchStatus } from "@/lib/types";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";

jest.mock("next-auth/react", () => ({
  useSession: () => ({ data: { user: { role: "admin" } }, status: "authenticated" }),
}));

const STATUS_RESPONSE: KillSwitchStatus = {
  latched: false,
  since: null,
  reason: null,
  source: "db",
};

const PRESS_RESPONSE: KillSwitchPressResponse = {
  latched: true,
  latchPersisted: true,
  since: "2026-09-27T00:00:00Z",
  runsLatched: ["run-1", "run-2"],
  orphanedLiveRunIds: ["run-3"],
  resumingRunIds: [],
  flattenResults: {
    "run-1": {
      runId: "run-1",
      outcome: "flattened",
      complete: true,
      latchPersisted: true,
      symbols: [
        {
          symbol: "BTC/EUR",
          status: "flat",
          cause: null,
          heldBefore: "0.01",
          soldQty: "0.01",
          remainingQty: "0",
          orderIds: ["ord-1"],
          error: null,
        },
      ],
    },
  },
  errors: [],
};

beforeEach(() => {
  global.fetch = jest.fn((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/v1/emergency/kill-switch")) {
      return Promise.resolve(fakeJsonResponse(STATUS_RESPONSE));
    }
    if (url.includes("/api/admin/kill-switch")) {
      return Promise.resolve(fakeJsonResponse(PRESS_RESPONSE));
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
  }) as unknown as typeof fetch;
});

describe("KillSwitchPanel — WP1.7a camelCase shape", () => {
  it("renders the latch status badge from GET /emergency/kill-switch", async () => {
    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());
  });

  it("renders runsLatched / flattenResults / orphanedLiveRunIds after a press — never the old runs_stopped shape", async () => {
    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Global Kill Switch" }));
    fireEvent.change(screen.getByPlaceholderText("EMERGENCY STOP"), {
      target: { value: "EMERGENCY STOP" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Activate Kill Switch" }));

    const dialog = await screen.findByRole("dialog");
    await waitFor(() => expect(dialog).toHaveTextContent("Latched 2 runs"));
    expect(dialog).toHaveTextContent("1 orphaned live run");
    // The per-symbol flatten result table renders from the camelCase fields.
    expect(dialog).toHaveTextContent("BTC/EUR");
    expect(dialog).toHaveTextContent("Flattened");
  });
});
