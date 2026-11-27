/**
 * WP21-T-35 (lib): cost-profile labels, leaderboard filter, compare banner
 * condition and the per-mode selector rules (SY-21-03, SY-21-10).
 */

import fs from "fs";
import path from "path";
import {
  COST_PROFILE_OPTIONS,
  DEFAULT_COST_PROFILE_ID,
  costGroupKey,
  costProfileField,
  costProfileLabel,
  costProfileOptionsForMode,
  hasMixedCostProfiles,
  isLegacyCosts,
  isLiveEquivalentCost,
  relaunchCostProfile,
  resolveCostProfileForMode,
  roundTripBps,
} from "@/lib/cost-profile";
import { ADV1, D6, DIAG, LEGACY_BT, LEGACY_RT, STRESS80 } from "../helpers/cost-profile-fixtures";

describe("isLegacyCosts / isLiveEquivalentCost (server flags only)", () => {
  it("a missing costProfile (older API) reads as legacy", () => {
    expect(isLegacyCosts({})).toBe(true);
    expect(isLegacyCosts({ costProfile: null })).toBe(true);
    expect(isLiveEquivalentCost({})).toBe(false);
  });

  it("the server legacyCosts flag wins even with a live-equivalent looking profile", () => {
    expect(isLegacyCosts({ costProfile: ADV1, legacyCosts: true })).toBe(true);
    expect(isLiveEquivalentCost({ costProfile: ADV1, legacyCosts: true })).toBe(false);
  });

  it("an invalid persisted profile or a legacy profile is legacy", () => {
    expect(isLegacyCosts({ costProfile: LEGACY_RT, costProfileInvalid: true })).toBe(true);
    expect(isLegacyCosts({ costProfile: LEGACY_BT })).toBe(true);
  });

  it("only live-equivalent registry profiles pass the default filter", () => {
    expect(isLiveEquivalentCost({ costProfile: ADV1, legacyCosts: false })).toBe(true);
    expect(isLiveEquivalentCost({ costProfile: D6, legacyCosts: false })).toBe(true);
    expect(isLiveEquivalentCost({ costProfile: STRESS80, legacyCosts: false })).toBe(false);
    expect(isLiveEquivalentCost({ costProfile: DIAG, legacyCosts: false })).toBe(false);
    expect(isLiveEquivalentCost({ costProfile: LEGACY_BT, legacyCosts: true })).toBe(false);
    expect(
      isLiveEquivalentCost({ costProfile: { ...ADV1, source: "override", liveEquivalent: false } }),
    ).toBe(false);
  });
});

describe("server liveEquivalentCosts flag (WP2.1b)", () => {
  it("decides the default filter when present, in both directions", () => {
    expect(isLiveEquivalentCost({ costProfile: STRESS80, liveEquivalentCosts: true })).toBe(true);
    expect(isLiveEquivalentCost({ costProfile: ADV1, legacyCosts: false, liveEquivalentCosts: false })).toBe(false);
  });

  it("falls back to the profile flags when the API does not send it", () => {
    expect(isLiveEquivalentCost({ costProfile: ADV1, legacyCosts: false })).toBe(true);
    expect(isLiveEquivalentCost({ costProfile: STRESS80, legacyCosts: false })).toBe(false);
  });
});

describe("costProfileLabel (SY-21-10)", () => {
  it("legacy -> 'legacy costs'", () => {
    expect(costProfileLabel({ costProfile: LEGACY_RT, legacyCosts: true })).toMatchObject({
      kind: "legacy",
      text: "legacy costs",
    });
    expect(costProfileLabel({}).text).toBe("legacy costs");
  });

  it("stress and diagnostic keep their kind label", () => {
    expect(costProfileLabel({ costProfile: STRESS80, legacyCosts: false })).toMatchObject({
      kind: "stress",
      text: "stress",
    });
    expect(costProfileLabel({ costProfile: DIAG, legacyCosts: false })).toMatchObject({
      kind: "diagnostic",
      text: "diagnostic",
    });
  });

  it("otherwise the profile id, with the bps tooltip", () => {
    const l = costProfileLabel({ costProfile: ADV1, legacyCosts: false });
    expect(l).toMatchObject({ kind: "profile", text: "coinbase_advanced_1" });
    expect(l.tooltip).toBe("taker 120 · maker 60 · slip 10 · RT 260 bps");
  });
});

describe("grouping, round trip, compare banner condition", () => {
  it("every legacy run groups as 'legacy'; others by profile id", () => {
    expect(costGroupKey({})).toBe("legacy");
    expect(costGroupKey({ costProfile: LEGACY_BT, legacyCosts: true })).toBe("legacy");
    expect(costGroupKey({ costProfile: LEGACY_RT, legacyCosts: true })).toBe("legacy");
    expect(costGroupKey({ costProfile: D6, legacyCosts: false })).toBe("coinbase_advanced_d6");
    expect(costGroupKey({ costProfile: STRESS80, legacyCosts: false })).toBe("coinbase_stress_t80");
  });

  it("roundTripBps reads the server string; null when unknown", () => {
    expect(roundTripBps({ costProfile: ADV1 })).toBe("260");
    expect(roundTripBps({})).toBeNull();
  });

  it("banner: different fingerprints, or any legacy run; not for identical profiles or <2 runs", () => {
    const a = { costProfile: ADV1, legacyCosts: false };
    const b = { costProfile: D6, legacyCosts: false };
    expect(hasMixedCostProfiles([a, b])).toBe(true);
    expect(hasMixedCostProfiles([a, { costProfile: ADV1, legacyCosts: false }])).toBe(false);
    expect(hasMixedCostProfiles([a, { costProfile: LEGACY_BT, legacyCosts: true }])).toBe(true);
    expect(hasMixedCostProfiles([a, {}])).toBe(true);
    expect(hasMixedCostProfiles([a])).toBe(false);
  });
});

describe("per-mode profile rules (SY-21-03)", () => {
  const ids = (m: Parameters<typeof costProfileOptionsForMode>[0]) =>
    costProfileOptionsForMode(m).map((o) => o.id);

  it("backtest and optimize offer every registry id", () => {
    expect(ids("backtest")).toEqual(COST_PROFILE_OPTIONS.map((o) => o.id));
    expect(ids("optimize")).toEqual(ids("backtest"));
    expect(ids("backtest")).toEqual(
      expect.arrayContaining([
        "legacy_backtest_v0",
        "legacy_runtime_v0",
        "frictionless_diagnostic",
        "coinbase_stress_t80",
        "coinbase_stress_t100",
      ]),
    );
  });

  it("paper offers live-equivalent ids only", () => {
    expect(ids("paper")).toEqual(["coinbase_advanced_1", "coinbase_advanced_d6"]);
  });

  it("live offers none and never sends a profile", () => {
    expect(ids("live")).toEqual([]);
    expect(resolveCostProfileForMode("live", "coinbase_advanced_d6")).toBeUndefined();
  });

  it("resolveCostProfileForMode omits default / empty / ineligible values", () => {
    expect(resolveCostProfileForMode("backtest", "")).toBeUndefined();
    expect(resolveCostProfileForMode("backtest", "coinbase_stress_t80")).toBe("coinbase_stress_t80");
    expect(resolveCostProfileForMode("paper", "coinbase_stress_t80")).toBeUndefined();
    expect(resolveCostProfileForMode("paper", "coinbase_advanced_d6")).toBe("coinbase_advanced_d6");
    expect(resolveCostProfileForMode("optimize", "no_such_id")).toBeUndefined();
  });

  it("costProfileField adds the key only when there is a value", () => {
    expect(costProfileField(undefined)).toEqual({});
    expect(costProfileField("coinbase_advanced_d6")).toEqual({ costProfile: "coinbase_advanced_d6" });
    expect("costProfile" in costProfileField(undefined)).toBe(false);
  });

  it("relaunchCostProfile carries every registry profile, including the default and explicit legacy_* ids", () => {
    expect(relaunchCostProfile({ costProfile: STRESS80, legacyCosts: false })).toBe("coinbase_stress_t80");
    // SEC-07: the default id is pinned, not omitted.
    expect(relaunchCostProfile({ costProfile: ADV1, legacyCosts: false })).toBe(DEFAULT_COST_PROFILE_ID);
    // CR-06: an explicit legacy registry id is carried (a backtest accepts it, SY-21-03).
    expect(relaunchCostProfile({ costProfile: LEGACY_BT, legacyCosts: true })).toBe("legacy_backtest_v0");
  });

  it("relaunchCostProfile falls back to the default for missing / invalid / non-registry profiles", () => {
    expect(relaunchCostProfile({})).toBeUndefined();
    expect(relaunchCostProfile({ costProfile: null })).toBeUndefined();
    // legacy_inferred (source !== "registry"), even with a registry-looking id.
    expect(relaunchCostProfile({ costProfile: LEGACY_RT, legacyCosts: true })).toBeUndefined();
    expect(
      relaunchCostProfile({ costProfile: { ...ADV1, id: "custom", source: "override" } }),
    ).toBeUndefined();
    // An invalid persisted profile is never carried, even if it is a registry id.
    expect(
      relaunchCostProfile({ costProfile: STRESS80, legacyCosts: true, costProfileInvalid: true }),
    ).toBeUndefined();
    // An id that is not in the registry is not offered for a backtest.
    expect(
      relaunchCostProfile({ costProfile: { ...ADV1, id: "no_such_id" }, legacyCosts: false }),
    ).toBeUndefined();
  });
});

// Registry drift guard. The backend has no profile-list endpoint, so the UI
// mirrors packages/trading/costs.py; this reads the Python source when it is
// reachable (monorepo checkout) and is skipped otherwise (e.g. a UI-only image).
const COSTS_PY = path.resolve(__dirname, "../../../../../packages/trading/costs.py");
const maybeDescribe = fs.existsSync(COSTS_PY) ? describe : describe.skip;

maybeDescribe("registry mirror matches packages/trading/costs.py", () => {
  // describe.skip still runs this body, so only read the file when it exists.
  const src = fs.existsSync(COSTS_PY) ? fs.readFileSync(COSTS_PY, "utf8") : "";
  const re =
    /_p\(\s*"([a-z0-9_]+)",\s*"[a-z]+",\s*"[a-z0-9_]+",\s*(\d+),\s*(\d+),\s*(\d+),([^)]*)\)/g;
  const parsed = [...src.matchAll(re)].map((m) => ({
    id: m[1] as string,
    maker: Number(m[2]),
    taker: Number(m[3]),
    slip: Number(m[4]),
    flags: m[5] as string,
  }));

  it("parses every registry row", () => {
    expect(parsed.length).toBe(COST_PROFILE_OPTIONS.length);
  });

  it("ids, bps and flags agree (order-insensitive)", () => {
    for (const row of parsed) {
      const opt = COST_PROFILE_OPTIONS.find((o) => o.id === row.id);
      expect(opt).toBeDefined();
      expect(opt).toMatchObject({
        makerFeeBps: row.maker,
        takerFeeBps: row.taker,
        slippageBps: row.slip,
        liveEquivalent: /live_equivalent=True/.test(row.flags),
      });
      const kind = /stress=True/.test(row.flags)
        ? "stress"
        : /diagnostic=True/.test(row.flags)
          ? "diagnostic"
          : /legacy=True/.test(row.flags)
            ? "legacy"
            : "live";
      expect(opt?.kind).toBe(kind);
    }
  });

  it("the default id agrees", () => {
    const m = /DEFAULT_COST_PROFILE_ID:\s*Final\[str\]\s*=\s*"([a-z0-9_]+)"/.exec(src);
    expect(m?.[1]).toBe(DEFAULT_COST_PROFILE_ID);
  });
});
