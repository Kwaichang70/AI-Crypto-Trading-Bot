/**
 * apps/ui/src/lib/exit-config.ts
 * ---------------------------------
 * WP1.3a (CF-13a-1) shared helpers for the exit-config / pyramiding
 * validator's 422 error envelope and 2xx warnings (reports/vp2-wp1.3a/
 * synthesis-spec.md §5, §8, §9, SY-13a-02/08/09/16/18).
 *
 * `unwrapExitConfigDetail` follows the WP1.7b `unwrapDetail` lesson
 * (stop-run-dialog.tsx): FastAPI wraps `HTTPException(detail=...)` in a
 * top-level `{"detail": {...}}` envelope, and `detail` is very often a
 * plain STRING (404s, some 400/403s) rather than the structured object
 * this module knows how to render — only a non-array OBJECT with one of
 * the three known top-level codes counts as a structured exit-config
 * detail; everything else returns `undefined` so callers fall back to the
 * generic `result.error.message` string.
 */

import type {
  ConfigWarning,
  ExitConfigErrorCode,
  ExitConfigErrorDetail,
  ExitConfigIssue,
  OptimizeExitConfigErrorDetail,
  OptimizeExitConfigIssue,
  Strategy,
} from "./types";

const KNOWN_CODES: ReadonlySet<string> = new Set<ExitConfigErrorCode>([
  "invalid_exit_config",
  "exit_manager_required",
  "live_pyramiding_forbidden",
]);

/** Narrows `unknown` to a plain, non-array, non-null object. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Unwraps `ApiError.detail` (the raw parsed response body) down to a
 * structured exit-config/pyramiding error body, for create/promote/resume.
 * Returns `undefined` for anything else (a plain string, a different 422
 * shape such as `flatten_decision_required`, a 404 string, etc.) so callers
 * can fall back to the generic error message.
 */
export function unwrapExitConfigDetail(raw: unknown): ExitConfigErrorDetail | undefined {
  const inner = isPlainObject(raw) && "detail" in raw ? raw.detail : raw;
  if (!isPlainObject(inner)) return undefined;
  const code = inner.code;
  if (typeof code !== "string" || !KNOWN_CODES.has(code)) return undefined;
  return inner as unknown as ExitConfigErrorDetail;
}

/**
 * Same unwrap, for `POST /api/v1/optimize`'s 422 — `errors[]` items are
 * combination-scoped (`combo_index`, `params`) rather than field-scoped.
 * `live_pyramiding_forbidden` is not one of the optimize codes (SY-13a-17
 * pre-validates the grid before any `allow_pyramiding` resolution), so this
 * only recognises the two config-validation codes.
 */
export function unwrapOptimizeExitConfigDetail(
  raw: unknown,
): OptimizeExitConfigErrorDetail | undefined {
  const inner = isPlainObject(raw) && "detail" in raw ? raw.detail : raw;
  if (!isPlainObject(inner)) return undefined;
  const code = inner.code;
  if (code !== "invalid_exit_config" && code !== "exit_manager_required") return undefined;
  if (!Array.isArray(inner.errors)) return undefined;
  return inner as unknown as OptimizeExitConfigErrorDetail;
}

/** Human-readable one-liner for a single `ExitConfigIssue`. */
export function describeExitConfigIssue(issue: ExitConfigIssue): string {
  const parts: string[] = [];
  if (issue.field) parts.push(issue.field);
  parts.push(issue.message || issue.reason);
  const bounds: string[] = [];
  if (issue.min !== null && issue.min !== undefined) bounds.push(`min ${issue.min}`);
  if (issue.max !== null && issue.max !== undefined) bounds.push(`max ${issue.max}`);
  if (issue.value !== null && issue.value !== undefined) bounds.push(`got ${issue.value}`);
  const suffix = bounds.length > 0 ? ` (${bounds.join(", ")})` : "";
  return `${parts.join(": ")}${suffix}`;
}

/** Human-readable one-liner for an optimize-grid-scoped issue, with its combo index. */
export function describeOptimizeExitConfigIssue(issue: OptimizeExitConfigIssue): string {
  return `Combination #${issue.combo_index}: ${describeExitConfigIssue(issue)}`;
}

/** Human-readable one-liner for a `ConfigWarning`. */
export function describeConfigWarning(warning: ConfigWarning): string {
  return warning.field ? `${warning.field}: ${warning.message}` : warning.message;
}

/**
 * SY-13a-08/09: strategy names whose `default_allow_pyramiding` ClassVar is
 * `True` on the backend (`packages/trading/strategy.py` base default is
 * `False`; only these two override it). Hard-coded here per CF-13a-1 item 4
 * because `GET /api/v1/strategies/{name}/schema` does not expose
 * `default_allow_pyramiding` on the wire as of WP1.3a (§4 only adds it as an
 * internal ClassVar, not a schema field) — `Strategy.defaultAllowPyramiding`
 * is checked first so this hard-coded fallback is superseded for free if a
 * later backend change does add it.
 */
export const PYRAMIDING_DEFAULT_TRUE_STRATEGIES: ReadonlySet<string> = new Set([
  "dca_rsi_hybrid",
  "grid_trading",
]);

/** Resolves the strategy-level `allow_pyramiding` default the paper/backtest checkbox should start at. */
export function strategyDefaultAllowPyramiding(strategy: Strategy | null): boolean {
  if (!strategy) return false;
  if (typeof strategy.defaultAllowPyramiding === "boolean") {
    return strategy.defaultAllowPyramiding;
  }
  return PYRAMIDING_DEFAULT_TRUE_STRATEGIES.has(strategy.name);
}

/** True for the two strategies SY-13a-09 special-cases in LIVE (W7 `accumulation_disabled`). */
export function isPyramidingByDesignStrategy(strategyName: string | undefined | null): boolean {
  return !!strategyName && PYRAMIDING_DEFAULT_TRUE_STRATEGIES.has(strategyName);
}
