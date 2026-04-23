/**
 * apps/ui/src/__tests__/components/side-badge.test.tsx
 * -----------------------------------------------------
 * Unit tests for the SideBadge component (Sprint 40 Stap 10 / CR-010).
 *
 * getSideClassName(side) returns the "text-profit" / "text-loss" class based
 * on exact string match against "buy".  Any other value (including "sell"
 * and malformed data) resolves to the loss class so API regressions are
 * visually obvious rather than silently styled like winners.
 *
 * <SideBadge side={"buy" | "sell" | string} /> renders the side text in
 * uppercase, wrapped in a <span> with the resolved class.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { SideBadge, getSideClassName } from "@/components/ui/side-badge";

describe("getSideClassName", () => {
  it("returns profit token for buy", () => {
    expect(getSideClassName("buy")).toBe("text-profit text-xs font-medium");
  });

  it("returns loss token for sell", () => {
    expect(getSideClassName("sell")).toBe("text-loss text-xs font-medium");
  });

  it("treats unknown sides as loss so regressions are visually obvious", () => {
    expect(getSideClassName("foo")).toBe("text-loss text-xs font-medium");
  });
});

describe("SideBadge", () => {
  it("renders uppercase BUY with the profit class", () => {
    render(<SideBadge side="buy" />);
    const node = screen.getByText("BUY");
    expect(node).toBeInTheDocument();
    expect(node).toHaveClass("text-profit");
  });

  it("renders uppercase SELL with the loss class", () => {
    render(<SideBadge side="sell" />);
    const node = screen.getByText("SELL");
    expect(node).toBeInTheDocument();
    expect(node).toHaveClass("text-loss");
  });
});
