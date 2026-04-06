/**
 * apps/ui/src/__tests__/components/status-badge.test.tsx
 * ---------------------------------------------------------
 * Unit tests for the StatusBadge component.
 *
 * StatusBadge({ status, variant? }) infers a BadgeVariant from RunStatus or
 * OrderStatus maps; an explicit `variant` prop overrides inference. Variant
 * controls CSS class applied to the <span>. A green dot is rendered for
 * "success" badges, a red dot for "danger" badges.
 *
 * RunStatus map:
 *   running  → success  (badge-success, green dot)
 *   stopped  → neutral  (badge-neutral)
 *   error    → danger   (badge-danger, red dot)
 *   archived → neutral  (badge-neutral)
 *
 * Unknown status → neutral fallback.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StatusBadge } from "@/components/ui/status-badge";

describe("StatusBadge", () => {
  // ---- RunStatus: running --------------------------------------------------

  it("renders 'running' status text", () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("applies badge-success class for running status", () => {
    const { container } = render(<StatusBadge status="running" />);
    expect(container.firstChild).toHaveClass("badge-success");
  });

  it("renders a green indicator dot for running (success variant)", () => {
    const { container } = render(<StatusBadge status="running" />);
    // success variant includes a small dot span with bg-green-500
    const dot = container.querySelector(".bg-green-500");
    expect(dot).toBeInTheDocument();
  });

  // ---- RunStatus: stopped --------------------------------------------------

  it("renders 'stopped' status text", () => {
    render(<StatusBadge status="stopped" />);
    expect(screen.getByText("stopped")).toBeInTheDocument();
  });

  it("applies badge-neutral class for stopped status", () => {
    const { container } = render(<StatusBadge status="stopped" />);
    expect(container.firstChild).toHaveClass("badge-neutral");
  });

  // ---- RunStatus: error ----------------------------------------------------

  it("renders 'error' status text", () => {
    render(<StatusBadge status="error" />);
    expect(screen.getByText("error")).toBeInTheDocument();
  });

  it("applies badge-danger class for error status", () => {
    const { container } = render(<StatusBadge status="error" />);
    expect(container.firstChild).toHaveClass("badge-danger");
  });

  it("renders a red indicator dot for error (danger variant)", () => {
    const { container } = render(<StatusBadge status="error" />);
    const dot = container.querySelector(".bg-red-500");
    expect(dot).toBeInTheDocument();
  });

  // ---- RunStatus: archived -------------------------------------------------

  it("applies badge-neutral class for archived status", () => {
    const { container } = render(<StatusBadge status="archived" />);
    expect(container.firstChild).toHaveClass("badge-neutral");
  });

  // ---- Unknown status fallback ---------------------------------------------

  it("renders unknown status text without throwing", () => {
    render(<StatusBadge status="unknown_xyz" />);
    expect(screen.getByText("unknown_xyz")).toBeInTheDocument();
  });

  it("applies badge-neutral class for unknown status", () => {
    const { container } = render(<StatusBadge status="unknown_xyz" />);
    expect(container.firstChild).toHaveClass("badge-neutral");
  });

  // ---- Explicit variant prop overrides inference --------------------------

  it("applies badge-warning class when variant=warning is passed explicitly", () => {
    const { container } = render(
      <StatusBadge status="partial" variant="warning" />,
    );
    expect(container.firstChild).toHaveClass("badge-warning");
  });

  // ---- OrderStatus variants -----------------------------------------------

  it("infers success variant for filled order status", () => {
    const { container } = render(<StatusBadge status="filled" />);
    expect(container.firstChild).toHaveClass("badge-success");
  });

  it("infers danger variant for rejected order status", () => {
    const { container } = render(<StatusBadge status="rejected" />);
    expect(container.firstChild).toHaveClass("badge-danger");
  });
});
