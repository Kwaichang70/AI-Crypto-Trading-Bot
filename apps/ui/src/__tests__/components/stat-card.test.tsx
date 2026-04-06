/**
 * apps/ui/src/__tests__/components/stat-card.test.tsx
 * -------------------------------------------------------
 * Unit tests for the StatCard component.
 *
 * StatCardProps:
 *   label: string
 *   value: string | number
 *   subValue?: string | undefined
 *   trend?: "up" | "down" | "neutral"
 *   isLoading?: boolean
 *
 * The component renders a card with a label, large tabular-nums value, and an
 * optional subValue line. When isLoading=true it renders animate-pulse skeletons
 * instead of content. The trend prop controls text colour but does not render
 * directional arrows — colour is applied via CSS classes (text-profit, text-loss,
 * text-slate-400).
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/ui/stat-card";

describe("StatCard", () => {
  // ---- Basic content -------------------------------------------------------

  it("renders the label text", () => {
    render(<StatCard label="Total PnL" value="$1,234.56" />);
    expect(screen.getByText("Total PnL")).toBeInTheDocument();
  });

  it("renders the value text", () => {
    render(<StatCard label="Total PnL" value="$1,234.56" />);
    expect(screen.getByText("$1,234.56")).toBeInTheDocument();
  });

  it("renders a numeric value correctly", () => {
    render(<StatCard label="Win Rate" value={0.72} />);
    expect(screen.getByText("0.72")).toBeInTheDocument();
  });

  // ---- subValue (optional) ------------------------------------------------

  it("renders subValue when provided", () => {
    render(<StatCard label="Sharpe" value="1.42" subValue="annualised" />);
    expect(screen.getByText("annualised")).toBeInTheDocument();
  });

  it("does not render a subValue element when the prop is omitted", () => {
    render(<StatCard label="Sharpe" value="1.42" />);
    // The component uses `{subValue && ...}` so nothing extra should appear
    expect(screen.queryByText("annualised")).toBeNull();
  });

  // ---- Loading skeleton ----------------------------------------------------

  it("renders loading skeleton when isLoading=true", () => {
    const { container } = render(
      <StatCard label="Total PnL" value="$1,234.56" isLoading />,
    );
    const skeletons = container.querySelectorAll(".animate-pulse");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("does not render label or value text when isLoading=true", () => {
    render(<StatCard label="Total PnL" value="$1,234.56" isLoading />);
    expect(screen.queryByText("Total PnL")).toBeNull();
    expect(screen.queryByText("$1,234.56")).toBeNull();
  });

  // ---- tabular-nums on value element --------------------------------------

  it("applies tabular-nums class to the value element", () => {
    const { container } = render(<StatCard label="Equity" value="50000" />);
    const valueEl = container.querySelector(".tabular-nums");
    expect(valueEl).toBeInTheDocument();
    expect(valueEl).toHaveTextContent("50000");
  });

  // ---- trend class applied ------------------------------------------------

  it("applies text-profit class for trend=up", () => {
    const { container } = render(
      <StatCard label="PnL" value="+5%" trend="up" />,
    );
    const valueEl = container.querySelector(".text-profit");
    expect(valueEl).toBeInTheDocument();
  });

  it("applies text-loss class for trend=down", () => {
    const { container } = render(
      <StatCard label="PnL" value="-3%" trend="down" />,
    );
    const valueEl = container.querySelector(".text-loss");
    expect(valueEl).toBeInTheDocument();
  });

  it("applies text-slate-400 class for trend=neutral", () => {
    const { container } = render(
      <StatCard label="PnL" value="0%" trend="neutral" />,
    );
    // neutral maps to text-slate-400 in the component
    const valueEl = container.querySelector(".text-slate-400");
    expect(valueEl).toBeInTheDocument();
  });
});
