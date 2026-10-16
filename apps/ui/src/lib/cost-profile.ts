/**
 * apps/ui/src/lib/cost-profile.ts
 * --------------------------------
 * WP2.1b: pure helpers for the cost-model UI (labels, leaderboard filter,
 * compare banner, per-mode profile selector rules).
 *
 * Rules (reports/vp2-wp2.1/synthesis-spec.md SY-21-03, SY-21-10):
 * - The SERVER decides what is legacy (`Run.legacyCosts`) and what is
 *   live-equivalent (`CostProfileSummary.liveEquivalent`). Nothing here reads
 *   `run.config` or recomputes eligibility; a missing value degrades to
 *   "legacy" (older API, malformed row).
 * - Labels: `legacy costs` (amber), `stress` / `diagnostic` (slate), else the
 *   profile id.
 */

import type { CostProfileSummary, RunMode } from "./types";

// ---------------------------------------------------------------------------
// Server-flag helpers
// ---------------------------------------------------------------------------

/** The cost fields shared by `Run`, `OptimizeResponse` and `OptimizationRunSummary`. */
export interface CostCarrier {
  costProfile?: CostProfileSummary | null;
  legacyCosts?: boolean;
  costProfileInvalid?: boolean;
  /** Server-computed cost half of leaderboard eligibility (WP2.1b); optional. */
  liveEquivalentCosts?: boolean;
}

/**
 * True when the run's costs are legacy. Mirrors SY-21-10 using only the
 * server-provided fields: the explicit flag, an invalid persisted profile, a
 * `legacy` profile, or no profile at all (older API / unpersisted optimizer
 * run) all mean legacy.
 */
export function isLegacyCosts(c: CostCarrier): boolean {
  if (c.legacyCosts === true) return true;
  if (c.costProfileInvalid === true) return true;
  const p = c.costProfile;
  if (p == null) return true;
  return p.legacy;
}

/**
 * The leaderboard default filter "Live-equivalent costs". The server flag
 * `liveEquivalentCosts` decides when the API sends it; an API without it
 * falls back to the profile flags the server did send.
 */
export function isLiveEquivalentCost(c: CostCarrier): boolean {
  if (typeof c.liveEquivalentCosts === "boolean") return c.liveEquivalentCosts;
  if (isLegacyCosts(c)) return false;
  const p = c.costProfile;
  return p != null && p.liveEquivalent && p.source === "registry";
}

export type CostLabelKind = "legacy" | "stress" | "diagnostic" | "profile";

export interface CostLabel {
  kind: CostLabelKind;
  /** Visible badge text. */
  text: string;
  /** Hover text: "taker 120 · maker 60 · slip 10 · RT 260 bps" (+ id / note). */
  tooltip: string;
}

function bpsSummary(p: CostProfileSummary): string {
  return (
    `taker ${p.takerFeeBps} · maker ${p.makerFeeBps} · ` +
    `slip ${p.slippageBps} · RT ${p.roundTripBps} bps`
  );
}

export function costProfileLabel(c: CostCarrier): CostLabel {
  const p = c.costProfile ?? null;
  if (isLegacyCosts(c)) {
    const base = "Pre-WP2.1 cost model; not comparable with current results.";
    return {
      kind: "legacy",
      text: "legacy costs",
      tooltip: p ? `${base} ${p.id}: ${bpsSummary(p)}` : base,
    };
  }
  // isLegacyCosts() is false, so a profile exists here.
  const profile = p as CostProfileSummary;
  if (profile.stress) {
    return {
      kind: "stress",
      text: "stress",
      tooltip: `${profile.id}: ${bpsSummary(profile)}`,
    };
  }
  if (profile.diagnostic) {
    return {
      kind: "diagnostic",
      text: "diagnostic",
      tooltip: `${profile.id}: ${bpsSummary(profile)}`,
    };
  }
  return { kind: "profile", text: profile.id, tooltip: bpsSummary(profile) };
}

/**
 * Leaderboard grouping key (strategy x profile): the profile id, with every
 * legacy run (any legacy id, inferred, invalid or missing) grouped as
 * `legacy`.
 */
export function costGroupKey(c: CostCarrier): string {
  if (isLegacyCosts(c)) return "legacy";
  return (c.costProfile as CostProfileSummary).id;
}

/** Round-trip cost in bps as shown in the compare table; null when unknown. */
export function roundTripBps(c: CostCarrier): string | null {
  return c.costProfile?.roundTripBps ?? null;
}

export const COST_MODELS_DIFFER_MESSAGE =
  "These runs use different cost models; returns and Sharpe are not directly comparable.";

/**
 * Compare-page banner condition (quant-design 7): any run is legacy, or the
 * runs do not all share one profile fingerprint.
 */
export function hasMixedCostProfiles(items: readonly CostCarrier[]): boolean {
  if (items.length < 2) return false;
  if (items.some(isLegacyCosts)) return true;
  const fingerprints = new Set(items.map((i) => i.costProfile?.fingerprint ?? ""));
  return fingerprints.size > 1;
}

// ---------------------------------------------------------------------------
// Registry mirror (selector only)
// ---------------------------------------------------------------------------

export type CostOptionKind = "live" | "stress" | "diagnostic" | "legacy";

export interface CostProfileOption {
  id: string;
  kind: CostOptionKind;
  liveEquivalent: boolean;
  makerFeeBps: number;
  takerFeeBps: number;
  slippageBps: number;
}

/**
 * The backend exposes NO cost-profile list endpoint (WP2.1a), so the selector
 * mirrors the registry ids and flags.
 *
 * SOURCE OF TRUTH: `packages/trading/costs.py` (`COST_PROFILES`,
 * `DEFAULT_COST_PROFILE_ID`). The registry is append-only; a drift test
 * (`src/__tests__/lib/cost-profile.test.ts`) parses that file when it is
 * reachable. The wire/persisted values always come from the server; this list
 * only decides which ids the form offers.
 */
export const DEFAULT_COST_PROFILE_ID = "coinbase_advanced_1";

export const COST_PROFILE_OPTIONS: readonly CostProfileOption[] = [
  { id: "coinbase_advanced_1", kind: "live", liveEquivalent: true, makerFeeBps: 60, takerFeeBps: 120, slippageBps: 10 },
  { id: "coinbase_advanced_d6", kind: "live", liveEquivalent: true, makerFeeBps: 40, takerFeeBps: 60, slippageBps: 10 },
  { id: "coinbase_stress_t80", kind: "stress", liveEquivalent: false, makerFeeBps: 40, takerFeeBps: 80, slippageBps: 10 },
  { id: "coinbase_stress_t100", kind: "stress", liveEquivalent: false, makerFeeBps: 40, takerFeeBps: 100, slippageBps: 10 },
  { id: "frictionless_diagnostic", kind: "diagnostic", liveEquivalent: false, makerFeeBps: 0, takerFeeBps: 0, slippageBps: 0 },
  { id: "legacy_backtest_v0", kind: "legacy", liveEquivalent: false, makerFeeBps: 10, takerFeeBps: 15, slippageBps: 5 },
  { id: "legacy_runtime_v0", kind: "legacy", liveEquivalent: false, makerFeeBps: 40, takerFeeBps: 60, slippageBps: 5 },
];

/** Where a profile is being chosen. `optimize` behaves like `backtest` (SY-21-03). */
export type CostSelectorMode = RunMode | "optimize";

/**
 * SY-21-03: backtest / optimize accept any registry id, paper only
 * live-equivalent ids, live accepts none.
 */
export function costProfileOptionsForMode(mode: CostSelectorMode): readonly CostProfileOption[] {
  if (mode === "live") return [];
  if (mode === "paper") return COST_PROFILE_OPTIONS.filter((o) => o.liveEquivalent);
  return COST_PROFILE_OPTIONS;
}

/**
 * The `costProfile` to put on a create/optimize body: `undefined` (omit the
 * key) for the platform default, for live, for an empty value and for an id
 * that is not offered in this mode. Omitting keeps the request fingerprint
 * identical to a pre-WP2.1 body (SY-21-17).
 */
export function resolveCostProfileForMode(
  mode: CostSelectorMode,
  selected: string,
): string | undefined {
  if (!selected) return undefined;
  return costProfileOptionsForMode(mode).some((o) => o.id === selected)
    ? selected
    : undefined;
}

/** Spread into a request body: adds `costProfile` only when there is one (exactOptionalPropertyTypes-safe). */
export function costProfileField(id: string | undefined): { costProfile?: string } {
  return id ? { costProfile: id } : {};
}

export function describeCostProfileOption(o: CostProfileOption): string {
  const rt = 2 * (o.takerFeeBps + o.slippageBps);
  return (
    `${o.id} · taker ${o.takerFeeBps} · maker ${o.makerFeeBps} · ` +
    `slip ${o.slippageBps} · RT ${rt} bps`
  );
}

/**
 * Registry id to carry over when re-launching a backtest from an existing
 * run / optimization (so the new run is priced like the original). Returns
 * `undefined` for legacy / inferred / custom profiles, which fall back to the
 * platform default.
 */
export function relaunchCostProfile(c: CostCarrier): string | undefined {
  const p = c.costProfile;
  if (p == null || isLegacyCosts(c) || p.source !== "registry") return undefined;
  return resolveCostProfileForMode("backtest", p.id);
}
