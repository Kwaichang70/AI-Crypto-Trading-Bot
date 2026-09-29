/**
 * apps/ui/src/__tests__/components/admin-only-gating.test.tsx
 * -----------------------------------------------------------------
 * WP1.7b (spec §6): "AdminOnly hides admin controls from viewers" — every
 * S13 admin action (kill-switch press/clear) must be invisible (not just
 * disabled) to a signed-in non-admin ("viewer") session.
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { useSession } from "next-auth/react";
import { KillSwitchPanel } from "@/components/kill-switch-panel";
import { fakeJsonResponse } from "@/__tests__/test-utils/fake-response";
import type { KillSwitchStatus } from "@/lib/types";

jest.mock("next-auth/react", () => ({ useSession: jest.fn() }));
const mockUseSession = useSession as jest.Mock;

const STATUS_RESPONSE: KillSwitchStatus = {
  latched: false,
  since: null,
  reason: null,
  source: "db",
};

beforeEach(() => {
  global.fetch = jest.fn(() => Promise.resolve(fakeJsonResponse(STATUS_RESPONSE))) as unknown as typeof fetch;
});

describe("AdminOnly gating — viewer role never sees admin kill-switch controls", () => {
  it("hides the press and clear buttons for a viewer session", async () => {
    mockUseSession.mockReturnValue({ data: { user: { role: "viewer" } }, status: "authenticated" });
    render(<KillSwitchPanel />);
    // Let the panel's own status poll settle (its useEffect fires
    // regardless of role) so no state update escapes past the test body.
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());

    expect(screen.queryByRole("button", { name: "Global Kill Switch" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear Kill Switch" })).not.toBeInTheDocument();
  });

  it("shows both admin actions for an admin session", async () => {
    mockUseSession.mockReturnValue({ data: { user: { role: "admin" } }, status: "authenticated" });
    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: "Global Kill Switch" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear Kill Switch" })).toBeInTheDocument();
  });

  it("hides admin actions while the session is still loading (fail-closed, not a flash of controls)", async () => {
    mockUseSession.mockReturnValue({ data: null, status: "loading" });
    render(<KillSwitchPanel />);
    await waitFor(() => expect(screen.getByText("Normal")).toBeInTheDocument());

    expect(screen.queryByRole("button", { name: "Global Kill Switch" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear Kill Switch" })).not.toBeInTheDocument();
  });
});
