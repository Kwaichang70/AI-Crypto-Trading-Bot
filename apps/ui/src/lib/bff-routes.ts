/**
 * apps/ui/src/lib/bff-routes.ts
 * -------------------------------
 * CF-97-37 (37-B, DR-9737-16/26): the BFF's default-deny route table and the
 * segment validator. Pure and environment-free, so it can be unit-tested
 * against tests/fixtures/api_routes_v1.json.
 *
 * The input is Next's `params.path`: Next has ALREADY decoded each segment
 * once, and that single decode is the one decode. Nothing here decodes again;
 * a segment that still holds `%` is double encoding and is rejected.
 *
 * A request is forwarded only if an entry matches (method, segments). Admin
 * routes (kill-switch POST/clear, resume, entries-latch/clear), the circuit
 * breaker, promotion-eligibility, single-order, health/background and metrics
 * have no entry, so they can never be reached through the BFF.
 */

const GRAMMAR: Record<string, RegExp> = {
  id: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
  name: /^[A-Za-z0-9_-]{1,64}$/,
  // The API's `{symbol:path}`: the only parameter that may hold a "/".
  symbol: /^[A-Z0-9]{2,12}\/[A-Z0-9]{2,12}$/,
};

export interface BffRoute {
  readonly method: string;
  readonly parts: readonly string[];
  /** BFF upstream budget in ms (browser budget + 5 s). */
  readonly budgetMs: number;
}

const route = (method: string, pattern: string, budgetS = 20): BffRoute => ({
  method,
  parts: pattern.split("/"),
  budgetMs: budgetS * 1000,
});

export const BFF_TABLE: readonly BffRoute[] = [
  route("GET", "runs"),
  route("POST", "runs", 125),
  route("GET", "runs/{id}"),
  route("DELETE", "runs/{id}", 70),
  route("POST", "runs/{id}/emergency-stop", 70),
  route("POST", "runs/{id}/promote-to-live", 125),
  route("PATCH", "runs/{id}/archive"),
  ...["portfolio", "equity-curve", "trades", "orders", "fills", "positions", "learning", "diagnostics"].map(
    (leaf) => route("GET", `runs/{id}/${leaf}`),
  ),
  route("GET", "portfolio/summary"),
  route("GET", "strategies"),
  route("GET", "strategies/{name}/schema"),
  route("GET", "ml/models"),
  route("PUT", "ml/models/{id}/activate"),
  route("POST", "ml/retrain/{symbol}"),
  route("POST", "ml/train"),
  route("GET", "optimize"),
  route("POST", "optimize", 305),
  route("GET", "optimize/{id}", 35),
  route("GET", "emergency/kill-switch"),
  route("GET", "signals/current"),
];

export type BffMatch =
  | { readonly ok: true; readonly route: BffRoute }
  | { readonly ok: false; readonly status: 400 | 404 };

// Positive per-segment grammar: ASCII letters, digits, "_", "-", "." and at
// most one "/" (only the symbol parameter can accept it). No "%", "\", control
// or non-ASCII character can pass, so there is no second decode to bypass.
const SEGMENT = /^[A-Za-z0-9_.-]+(\/[A-Za-z0-9_.-]+)?$/;

export function matchBff(method: string, segments: readonly string[]): BffMatch {
  const dotted = (s: string) => s.split("/").some((p) => p === "." || p === "..");
  if (segments.length === 0 || segments.some((s) => !SEGMENT.test(s) || dotted(s))) {
    return { ok: false, status: 400 };
  }
  const hit = BFF_TABLE.find(
    (r) =>
      r.method === method &&
      r.parts.length === segments.length &&
      r.parts.every((p, i) =>
        p.startsWith("{") ? (GRAMMAR[p.slice(1, -1)]?.test(segments[i]!) ?? false) : p === segments[i],
      ),
  );
  if (hit) return { ok: true, route: hit };
  // A "/" inside a segment that no typed parameter accepted is a bad path.
  return { ok: false, status: segments.some((s) => s.includes("/")) ? 400 : 404 };
}
