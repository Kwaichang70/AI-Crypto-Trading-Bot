/**
 * apps/ui/src/__tests__/components/exit-config-error-panel.test.tsx
 * ----------------------------------------------------------------------
 * WP1.3a (CF-13a-1 item 6): `<ExitConfigErrorPanel>` renders every one of
 * the three structured 422 codes (SY-13a-18), the optimize variant's
 * combo_index-scoped errors, and falls back to a generic message for
 * anything else (real `{detail: {...}}` FastAPI envelope shape throughout).
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { ExitConfigErrorPanel } from "@/components/exit-config-error-panel";

describe("ExitConfigErrorPanel — default variant", () => {
  it("renders invalid_exit_config errors and warnings", () => {
    render(
      <ExitConfigErrorPanel
        detail={{
          detail: {
            code: "invalid_exit_config",
            errors: [
              {
                field: "bracket_stop_loss_pct",
                reason: "out_of_range",
                value: "0.51",
                min: 0.0065,
                max: 0.5,
                message: "Stop loss must be in (0.65%, 50%].",
              },
            ],
            warnings: [{ code: "reward_risk_le_1", field: null, message: "TP <= SL." }],
          },
        }}
        fallbackMessage="fallback"
      />,
    );
    expect(screen.getByText("Invalid exit configuration")).toBeInTheDocument();
    expect(screen.getByText(/Stop loss must be in/)).toBeInTheDocument();
    expect(screen.getByText(/TP <= SL\./)).toBeInTheDocument();
    expect(screen.queryByText("fallback")).not.toBeInTheDocument();
  });

  it("renders exit_manager_required with strategy + requires_one_of", () => {
    render(
      <ExitConfigErrorPanel
        detail={{
          detail: {
            code: "exit_manager_required",
            strategy: "momentum_breakout",
            requires_one_of: ["bracket_stop_loss_pct (bracket_mode=fixed)", "trailing_stop_pct"],
            errors: [],
            warnings: [],
          },
        }}
        fallbackMessage="fallback"
      />,
    );
    expect(screen.getByText("This strategy requires a downside exit")).toBeInTheDocument();
    expect(screen.getByText("momentum_breakout")).toBeInTheDocument();
    expect(screen.getByText(/trailing_stop_pct/)).toBeInTheDocument();
  });

  it("renders live_pyramiding_forbidden with strategy + hint", () => {
    render(
      <ExitConfigErrorPanel
        detail={{
          detail: {
            code: "live_pyramiding_forbidden",
            strategy: "grid_trading",
            hint: "pass allowPyramiding=false to run single-entry (unvalidated), or use paper",
            errors: [],
          },
        }}
        fallbackMessage="fallback"
      />,
    );
    expect(screen.getByText("Live pyramiding is forbidden")).toBeInTheDocument();
    expect(screen.getByText("grid_trading")).toBeInTheDocument();
    expect(screen.getByText(/pass allowPyramiding=false/)).toBeInTheDocument();
  });

  it("falls back to the generic message for a plain-string 404 detail", () => {
    render(
      <ExitConfigErrorPanel detail={{ detail: "Run abc not found." }} fallbackMessage="Run not found." />,
    );
    expect(screen.getByText("Run not found.")).toBeInTheDocument();
    expect(screen.queryByTestId("exit-config-error-panel")).not.toBeInTheDocument();
  });

  it("falls back to the generic message for an unrelated structured detail", () => {
    render(
      <ExitConfigErrorPanel
        detail={{ detail: { code: "flatten_decision_required", held_symbols: ["BTC/EUR"] } }}
        fallbackMessage="Validation error — check the request payload."
      />,
    );
    expect(screen.getByText("Validation error — check the request payload.")).toBeInTheDocument();
  });
});

describe("ExitConfigErrorPanel — optimize variant", () => {
  it("renders combo_index-scoped errors and total_invalid", () => {
    render(
      <ExitConfigErrorPanel
        variant="optimize"
        detail={{
          detail: {
            code: "invalid_exit_config",
            errors: [
              {
                combo_index: 2,
                params: { lookback: 5 },
                field: "bracket_stop_loss_pct",
                reason: "out_of_range",
                value: "0.6",
                min: 0.0065,
                max: 0.5,
                message: "Stop loss out of range.",
              },
            ],
            total_invalid: 30,
          },
        }}
        fallbackMessage="fallback"
      />,
    );
    expect(screen.getByText(/30 combinations failed/)).toBeInTheDocument();
    expect(screen.getByText(/Combination #2:/)).toBeInTheDocument();
  });

  it("falls back to generic message when the optimize detail doesn't unwrap", () => {
    render(
      <ExitConfigErrorPanel variant="optimize" detail={{ detail: "boom" }} fallbackMessage="fallback msg" />,
    );
    expect(screen.getByText("fallback msg")).toBeInTheDocument();
  });
});
