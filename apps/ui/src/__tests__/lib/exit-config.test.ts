/**
 * apps/ui/src/__tests__/lib/exit-config.test.ts
 * -------------------------------------------------
 * WP1.3a (CF-13a-1 item 6) unit tests for the shared exit-config/pyramiding
 * helpers, using the real `{"detail": {...}}` FastAPI envelope shape
 * (synthesis-spec.md §5/§18).
 */

import {
  describeConfigWarning,
  describeExitConfigIssue,
  describeOptimizeExitConfigIssue,
  isPyramidingByDesignStrategy,
  PYRAMIDING_DEFAULT_TRUE_STRATEGIES,
  strategyDefaultAllowPyramiding,
  unwrapExitConfigDetail,
  unwrapOptimizeExitConfigDetail,
} from "@/lib/exit-config";
import type { Strategy } from "@/lib/types";

describe("unwrapExitConfigDetail", () => {
  it("unwraps a real {detail: {...}} FastAPI envelope for invalid_exit_config", () => {
    const raw = {
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
        warnings: [],
      },
    };
    const parsed = unwrapExitConfigDetail(raw);
    expect(parsed).toBeDefined();
    expect(parsed?.code).toBe("invalid_exit_config");
    if (parsed?.code === "invalid_exit_config") {
      expect(parsed.errors).toHaveLength(1);
      expect(parsed.errors[0]?.reason).toBe("out_of_range");
    }
  });

  it("unwraps exit_manager_required with strategy + requires_one_of", () => {
    const raw = {
      detail: {
        code: "exit_manager_required",
        strategy: "momentum_breakout",
        requires_one_of: [
          "bracket_stop_loss_pct (bracket_mode=fixed)",
          "bracket_atr_sl_multiplier (bracket_mode=atr)",
          "trailing_stop_pct",
        ],
        errors: [],
        warnings: [],
      },
    };
    const parsed = unwrapExitConfigDetail(raw);
    expect(parsed?.code).toBe("exit_manager_required");
    if (parsed?.code === "exit_manager_required") {
      expect(parsed.strategy).toBe("momentum_breakout");
      expect(parsed.requires_one_of).toHaveLength(3);
    }
  });

  it("unwraps live_pyramiding_forbidden with a hint", () => {
    const raw = {
      detail: {
        code: "live_pyramiding_forbidden",
        strategy: "grid_trading",
        hint: "strategy accumulates by design; pass allowPyramiding=false to run single-entry (unvalidated), or use paper",
        errors: [],
      },
    };
    const parsed = unwrapExitConfigDetail(raw);
    expect(parsed?.code).toBe("live_pyramiding_forbidden");
    if (parsed?.code === "live_pyramiding_forbidden") {
      expect(parsed.hint).toContain("allowPyramiding=false");
    }
  });

  it("returns undefined for a plain-string detail (e.g. a 404)", () => {
    expect(unwrapExitConfigDetail({ detail: "Run abc not found." })).toBeUndefined();
  });

  it("returns undefined for an unrelated structured detail (e.g. flatten_decision_required)", () => {
    expect(
      unwrapExitConfigDetail({ detail: { code: "flatten_decision_required", held_symbols: ["BTC/EUR"] } }),
    ).toBeUndefined();
  });

  it("returns undefined for null/array details", () => {
    expect(unwrapExitConfigDetail(null)).toBeUndefined();
    expect(unwrapExitConfigDetail([1, 2, 3])).toBeUndefined();
    expect(unwrapExitConfigDetail(undefined)).toBeUndefined();
  });

  it("also accepts an already-unwrapped body (no outer detail key)", () => {
    const parsed = unwrapExitConfigDetail({ code: "invalid_exit_config", errors: [], warnings: [] });
    expect(parsed?.code).toBe("invalid_exit_config");
  });
});

describe("unwrapOptimizeExitConfigDetail", () => {
  it("unwraps combo_index/params-scoped errors, capped with total_invalid", () => {
    const raw = {
      detail: {
        code: "invalid_exit_config",
        errors: [
          {
            combo_index: 3,
            params: { lookback: 5, bracket_stop_loss_pct: 0 },
            field: "bracket_stop_loss_pct",
            reason: "out_of_range",
            value: "0",
            min: 0.0065,
            max: 0.5,
            message: "unset via zero, but strategy requires an exit",
          },
        ],
        total_invalid: 42,
      },
    };
    const parsed = unwrapOptimizeExitConfigDetail(raw);
    expect(parsed).toBeDefined();
    expect(parsed?.total_invalid).toBe(42);
    expect(parsed?.errors[0]?.combo_index).toBe(3);
    expect(parsed?.errors[0]?.params).toEqual({ lookback: 5, bracket_stop_loss_pct: 0 });
  });

  it("rejects live_pyramiding_forbidden (not an optimize code)", () => {
    expect(
      unwrapOptimizeExitConfigDetail({ detail: { code: "live_pyramiding_forbidden", errors: [] } }),
    ).toBeUndefined();
  });

  it("rejects a detail with no errors array", () => {
    expect(unwrapOptimizeExitConfigDetail({ detail: { code: "invalid_exit_config" } })).toBeUndefined();
  });
});

describe("describeExitConfigIssue", () => {
  it("renders field, message and bounds", () => {
    const line = describeExitConfigIssue({
      field: "bracket_stop_loss_pct",
      reason: "out_of_range",
      value: "0.51",
      min: 0.0065,
      max: 0.5,
      message: "Stop loss must be in (0.65%, 50%].",
    });
    expect(line).toContain("bracket_stop_loss_pct");
    expect(line).toContain("Stop loss must be in (0.65%, 50%].");
    expect(line).toContain("min 0.0065");
    expect(line).toContain("max 0.5");
    expect(line).toContain("got 0.51");
  });

  it("omits a null field/bounds gracefully", () => {
    const line = describeExitConfigIssue({
      field: null,
      reason: "inactive_mode_value",
      value: null,
      min: null,
      max: null,
      message: "bracket_atr_sl_multiplier is set but bracket_mode is fixed.",
    });
    expect(line).toBe("bracket_atr_sl_multiplier is set but bracket_mode is fixed.");
  });
});

describe("describeOptimizeExitConfigIssue", () => {
  it("prefixes the combination index", () => {
    const line = describeOptimizeExitConfigIssue({
      combo_index: 7,
      params: { x: 1 },
      field: "trailing_stop_pct",
      reason: "out_of_range",
      value: "0.6",
      min: 0.005,
      max: 0.5,
      message: "Trailing stop must be in [0.5%, 50%].",
    });
    expect(line).toMatch(/^Combination #7: /);
    expect(line).toContain("trailing_stop_pct");
  });
});

describe("describeConfigWarning", () => {
  it("includes the field when present", () => {
    expect(
      describeConfigWarning({ code: "sl_below_round_trip_cost", field: "bracket_stop_loss_pct", message: "SL < 1.3%." }),
    ).toBe("bracket_stop_loss_pct: SL < 1.3%.");
  });

  it("omits the field prefix when null", () => {
    expect(
      describeConfigWarning({ code: "no_downside_exit", field: null, message: "No downside exit configured." }),
    ).toBe("No downside exit configured.");
  });
});

describe("PYRAMIDING_DEFAULT_TRUE_STRATEGIES / strategyDefaultAllowPyramiding", () => {
  function makeStrategy(name: string, defaultAllowPyramiding?: boolean): Strategy {
    return {
      name,
      displayName: name,
      version: "1.0.0",
      description: "",
      tags: [],
      parameterSchema: { type: "object", properties: {} },
      ...(defaultAllowPyramiding !== undefined ? { defaultAllowPyramiding } : {}),
    };
  }

  it("contains exactly dca_rsi_hybrid and grid_trading (SY-13a-08)", () => {
    expect(PYRAMIDING_DEFAULT_TRUE_STRATEGIES.has("dca_rsi_hybrid")).toBe(true);
    expect(PYRAMIDING_DEFAULT_TRUE_STRATEGIES.has("grid_trading")).toBe(true);
    expect(PYRAMIDING_DEFAULT_TRUE_STRATEGIES.size).toBe(2);
  });

  it("returns false for null strategy", () => {
    expect(strategyDefaultAllowPyramiding(null)).toBe(false);
  });

  it("hard-codes true for dca_rsi_hybrid/grid_trading when the schema doesn't expose it", () => {
    expect(strategyDefaultAllowPyramiding(makeStrategy("dca_rsi_hybrid"))).toBe(true);
    expect(strategyDefaultAllowPyramiding(makeStrategy("grid_trading"))).toBe(true);
  });

  it("defaults to false for every other strategy", () => {
    expect(strategyDefaultAllowPyramiding(makeStrategy("momentum_breakout"))).toBe(false);
    expect(strategyDefaultAllowPyramiding(makeStrategy("rsi_mean_reversion"))).toBe(false);
  });

  it("prefers an explicit Strategy.defaultAllowPyramiding field if the backend ever adds it", () => {
    expect(strategyDefaultAllowPyramiding(makeStrategy("momentum_breakout", true))).toBe(true);
    expect(strategyDefaultAllowPyramiding(makeStrategy("dca_rsi_hybrid", false))).toBe(false);
  });
});

describe("isPyramidingByDesignStrategy", () => {
  it("is true only for dca_rsi_hybrid/grid_trading", () => {
    expect(isPyramidingByDesignStrategy("dca_rsi_hybrid")).toBe(true);
    expect(isPyramidingByDesignStrategy("grid_trading")).toBe(true);
    expect(isPyramidingByDesignStrategy("momentum_breakout")).toBe(false);
  });

  it("is false for undefined/null", () => {
    expect(isPyramidingByDesignStrategy(undefined)).toBe(false);
    expect(isPyramidingByDesignStrategy(null)).toBe(false);
  });
});
