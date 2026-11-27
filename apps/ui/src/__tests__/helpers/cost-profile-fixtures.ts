/**
 * WP2.1b test fixtures: cost profiles as the API returns them
 * (`CostProfileSummary`) and a `Run` factory carrying the SY-21-10 fields.
 * Numbers mirror packages/trading/costs.py COST_PROFILES.
 */

import type { CostProfileSummary, Run, RunMode } from "@/lib/types";

function summary(
  id: string,
  tier: string,
  maker: number,
  taker: number,
  slip: number,
  flags: Partial<Pick<CostProfileSummary, "liveEquivalent" | "diagnostic" | "legacy" | "stress">>,
  extra: Partial<CostProfileSummary> = {},
): CostProfileSummary {
  return {
    id,
    exchange: "coinbase",
    tier,
    makerFeeBps: String(maker),
    takerFeeBps: String(taker),
    slippageBps: String(slip),
    roundTripBps: String(2 * (taker + slip)),
    source: "registry",
    fingerprint: `fp-${id}`,
    feeSource: "model",
    liveEquivalent: false,
    diagnostic: false,
    legacy: false,
    stress: false,
    ...flags,
    ...extra,
  };
}

export const ADV1 = summary("coinbase_advanced_1", "advanced_1_usd_0_1k_30d", 60, 120, 10, { liveEquivalent: true });
export const D6 = summary("coinbase_advanced_d6", "advanced_d6_assumed_0_10k", 40, 60, 10, { liveEquivalent: true });
export const STRESS80 = summary("coinbase_stress_t80", "stress_taker_80", 40, 80, 10, { stress: true });
export const DIAG = summary("frictionless_diagnostic", "diagnostic_zero_cost", 0, 0, 0, { diagnostic: true });
export const LEGACY_BT = summary("legacy_backtest_v0", "pre_wp21_backtest", 10, 15, 5, { legacy: true });
export const LEGACY_RT = summary(
  "legacy_runtime_v0",
  "pre_wp21_runtime",
  40,
  60,
  5,
  { legacy: true },
  { source: "legacy_inferred" },
);

let counter = 0;

export interface RunOptions {
  id?: string;
  mode?: RunMode;
  strategy?: string;
  /** undefined = field absent (older API); null = explicitly null. */
  costProfile?: CostProfileSummary | null;
  legacyCosts?: boolean;
  costProfileInvalid?: boolean;
  liveEquivalentCosts?: boolean;
  returnPct?: number;
  /** false -> no backtestMetrics (pages that render the full metrics grid need complete metrics or none). */
  metrics?: boolean;
}

export function makeRun(opts: RunOptions = {}): Run {
  counter += 1;
  const id = opts.id ?? `00000000-0000-4000-8000-${String(counter).padStart(12, "0")}`;
  const mode = opts.mode ?? "backtest";
  const run: Record<string, unknown> = {
    id,
    runMode: mode,
    status: "stopped",
    config: {
      strategy_name: opts.strategy ?? "rsi_mean_reversion",
      strategy_params: {},
      symbols: ["BTC/EUR"],
      timeframe: "1h",
      mode,
      initial_capital: "10000",
    },
    startedAt: "2026-01-01T00:00:00Z",
    stoppedAt: "2026-01-02T00:00:00Z",
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-02T00:00:00Z",
    backtestMetrics: {
      totalReturnPct: opts.returnPct ?? 0.1,
      sharpeRatio: 1.5,
      winRate: 0.5,
      totalTrades: 20,
      profitFactor: 1.4,
      profitFactorIsInfinite: false,
    },
  };
  if (opts.metrics === false) delete run.backtestMetrics;
  if (opts.costProfile !== undefined) run.costProfile = opts.costProfile;
  if (opts.legacyCosts !== undefined) run.legacyCosts = opts.legacyCosts;
  if (opts.costProfileInvalid !== undefined) run.costProfileInvalid = opts.costProfileInvalid;
  if (opts.liveEquivalentCosts !== undefined) run.liveEquivalentCosts = opts.liveEquivalentCosts;
  return run as unknown as Run;
}
