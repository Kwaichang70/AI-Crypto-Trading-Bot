/**
 * WP21-T-35 (components): CostProfileBadge, CostProfileSelect, CostModelCard.
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import type { Run } from "@/lib/types";
import { CostProfileBadge } from "@/components/ui/cost-profile-badge";
import { CostProfileSelect } from "@/components/cost-profile-select";
import { CostModelCard } from "@/components/cost-model-card";
import { makeRun, ADV1, STRESS80, DIAG, LEGACY_BT } from "../helpers/cost-profile-fixtures";

describe("CostProfileBadge", () => {
  it("legacy -> amber 'legacy costs'", () => {
    render(<CostProfileBadge value={{ costProfile: LEGACY_BT, legacyCosts: true }} />);
    const b = screen.getByTestId("cost-profile-badge");
    expect(b).toHaveTextContent("legacy costs");
    expect(b).toHaveClass("badge-warning");
  });

  it("stress / diagnostic -> slate; showId appends the hidden id", () => {
    const { rerender } = render(<CostProfileBadge value={{ costProfile: STRESS80, legacyCosts: false }} />);
    expect(screen.getByTestId("cost-profile-badge")).toHaveClass("badge-neutral");
    expect(screen.getByTestId("cost-profile-badge").textContent).toBe("stress");

    rerender(<CostProfileBadge value={{ costProfile: DIAG, legacyCosts: false }} showId />);
    expect(screen.getByTestId("cost-profile-badge").textContent).toBe("diagnosticfrictionless_diagnostic");
  });

  it("otherwise the profile id, with a bps tooltip", () => {
    render(<CostProfileBadge value={{ costProfile: ADV1, legacyCosts: false }} showId />);
    const b = screen.getByTestId("cost-profile-badge");
    expect(b.textContent).toBe("coinbase_advanced_1");
    expect(b).toHaveAttribute("title", expect.stringContaining("RT 260 bps"));
  });
});

describe("CostProfileSelect", () => {
  it("live renders no select and never calls onChange", () => {
    const onChange = jest.fn();
    render(<CostProfileSelect mode="live" value="coinbase_advanced_d6" onChange={onChange} />);
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByTestId("cost-profile-live-note")).toBeInTheDocument();
  });

  it("paper shows the default for a stale ineligible value", () => {
    render(<CostProfileSelect mode="paper" value="coinbase_stress_t80" onChange={jest.fn()} />);
    expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("");
  });

  it("reports the chosen id", () => {
    const onChange = jest.fn();
    render(<CostProfileSelect mode="optimize" value="" onChange={onChange} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "coinbase_stress_t100" } });
    expect(onChange).toHaveBeenCalledWith("coinbase_stress_t100");
  });
});

describe("CostModelCard", () => {
  it("backtest shows total fees paid", () => {
    const run = makeRun({ costProfile: ADV1, legacyCosts: false }) as Run;
    const withFees = {
      ...run,
      backtestMetrics: { ...run.backtestMetrics, totalFeesPaid: "12.50" },
    } as unknown as Run;
    render(<CostModelCard run={withFees} />);
    expect(screen.getByText("Total fees paid")).toBeInTheDocument();
    expect(screen.getByTestId("cost-model-card")).toHaveTextContent("12.50");
  });

  it("live never shows a model fee total and states the estimate", () => {
    render(
      <CostModelCard
        run={makeRun({ mode: "live", costProfile: { ...ADV1, feeSource: "exchange_fills" }, legacyCosts: false })}
      />,
    );
    expect(screen.queryByText("Total fees paid")).not.toBeInTheDocument();
    expect(screen.getByTestId("cost-model-live-note")).toHaveTextContent(
      "fees: actual exchange fills; estimate profile coinbase_advanced_1",
    );
  });

  it("live run without a profile says the estimate is unknown", () => {
    render(<CostModelCard run={makeRun({ mode: "live" })} />);
    expect(screen.getByTestId("cost-model-live-note")).toHaveTextContent("estimate profile unknown (legacy costs)");
  });
});
