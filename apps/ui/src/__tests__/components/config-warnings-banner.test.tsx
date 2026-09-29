/**
 * apps/ui/src/__tests__/components/config-warnings-banner.test.tsx
 * ---------------------------------------------------------------------
 * WP1.3a (CF-13a-1 item 3/6): `<ConfigWarningsBanner>` (post-create
 * configWarnings[]) and `<ProtectiveResumeBanner>` (protective-resume
 * exitConfigWaived/exitManagerMissing).
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { ConfigWarningsBanner, ProtectiveResumeBanner } from "@/components/config-warnings-banner";

describe("ConfigWarningsBanner", () => {
  it("renders nothing when there are no warnings", () => {
    const { container } = render(<ConfigWarningsBanner warnings={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for null/undefined", () => {
    const { container: c1 } = render(<ConfigWarningsBanner warnings={null} />);
    expect(c1).toBeEmptyDOMElement();
    const { container: c2 } = render(<ConfigWarningsBanner warnings={undefined} />);
    expect(c2).toBeEmptyDOMElement();
  });

  it("splits no_downside_exit (W8) into a critical 'flatten recommended' section", () => {
    render(
      <ConfigWarningsBanner
        warnings={[
          { code: "no_downside_exit", field: null, message: "No downside exit configured." },
          { code: "sl_below_round_trip_cost", field: "bracket_stop_loss_pct", message: "SL < 1.3%." },
        ]}
      />,
    );
    expect(screen.getByText("No downside exit — flatten recommended")).toBeInTheDocument();
    expect(screen.getByText(/No downside exit configured\./)).toBeInTheDocument();
    expect(screen.getByText("Warnings")).toBeInTheDocument();
    expect(screen.getByText(/SL < 1\.3%\./)).toBeInTheDocument();
  });

  it("renders only the non-critical section when W8 is absent", () => {
    render(
      <ConfigWarningsBanner
        warnings={[{ code: "accumulation_disabled", field: null, message: "Accumulation disabled: single-entry variant, not validated" }]}
      />,
    );
    expect(screen.queryByText("No downside exit — flatten recommended")).not.toBeInTheDocument();
    expect(screen.getByText(/Accumulation disabled/)).toBeInTheDocument();
  });
});

describe("ProtectiveResumeBanner", () => {
  it("renders nothing when neither field is set", () => {
    const { container } = render(
      <ProtectiveResumeBanner exitConfigWaived={null} exitManagerMissing={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the critical 'flatten recommended' framing when exitManagerMissing is true", () => {
    render(
      <ProtectiveResumeBanner
        exitConfigWaived={{
          code: "invalid_exit_config",
          errors: [
            {
              field: "bracket_stop_loss_pct",
              reason: "out_of_range",
              value: "0.6",
              min: 0.0065,
              max: 0.5,
              message: "Stop loss out of range; dropped.",
            },
          ],
        }}
        exitManagerMissing={true}
      />,
    );
    expect(screen.getByText("No downside exit — flatten recommended")).toBeInTheDocument();
    expect(screen.getByText(/Stop loss out of range; dropped\./)).toBeInTheDocument();
  });

  it("renders a lighter 'partially waived' framing when exitManagerMissing is false", () => {
    render(
      <ProtectiveResumeBanner
        exitConfigWaived={{
          code: "invalid_exit_config",
          errors: [
            {
              field: "bracket_atr_sl_multiplier",
              reason: "out_of_range",
              value: "25",
              min: 0.1,
              max: 20.0,
              message: "ATR multiplier out of range; dropped, trailing stop still active.",
            },
          ],
        }}
        exitManagerMissing={false}
      />,
    );
    expect(screen.queryByText("No downside exit — flatten recommended")).not.toBeInTheDocument();
    expect(screen.getByText("Exit config partially waived (protective resume)")).toBeInTheDocument();
  });

  it("renders the critical banner even with no waiver record, purely from exitManagerMissing", () => {
    render(<ProtectiveResumeBanner exitConfigWaived={null} exitManagerMissing={true} />);
    expect(screen.getByText("No downside exit — flatten recommended")).toBeInTheDocument();
  });
});
