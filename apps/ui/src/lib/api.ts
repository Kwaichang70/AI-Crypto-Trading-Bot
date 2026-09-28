/**
 * Type-safe API client for the FastAPI trading-bot backend.
 *
 * All public functions are async and return a discriminated-union result type
 * so callers never have to catch raw errors — they always receive either
 * { ok: true; data: T } or { ok: false; error: ApiError }.
 *
 * Base URL is read from NEXT_PUBLIC_API_URL at build time (Next.js inlines it).
 * The fallback "/api" works with the proxy rewrite in next.config.ts so the
 * browser never makes cross-origin requests during development.
 */

import type {
  AggregatePortfolio,
  EquityCurveResponse,
  FillListResponse,
  KillSwitchStatus,
  LearningState,
  ModelVersion,
  ModelVersionListResponse,
  OptimizationRunListResponse,
  OptimizeRequest,
  OptimizeResponse,
  OrderListResponse,
  Portfolio,
  PositionListResponse,
  Run,
  RunCreateRequest,
  RunEmergencyStopResponse,
  RunListResponse,
  RunStopResponse,
  Strategy,
  StrategyListResponse,
  TradeListResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Sprint 50 cycle 2 fix: paths in this module all start with /api/v1/...
// (already include the /api prefix). BASE_URL must therefore be EITHER:
//   - a full URL like "http://api:8000" (server-side SSR, prepended to /api/v1/...)
//   - empty string "" (browser, makes URLs same-origin relative; Caddy proxies /api/v1/*)
// The previous fallback `|| "/api"` caused double-prefix /api/api/v1/... in browser
// when NEXT_PUBLIC_API_URL was set to "/api" or empty.
const BASE_URL =
  typeof window === "undefined"
    ? (
        process.env.INTERNAL_API_URL ??
        process.env.NEXT_PUBLIC_API_URL ??
        "http://api:8000"
      ).replace(/\/$/, "")
    : (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Shared response types
// ---------------------------------------------------------------------------

/** Discriminated-union result — callers inspect `ok` before accessing data. */
export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: ApiError };

/** Structured error produced by every failed request. */
export interface ApiError {
  /** HTTP status code, or 0 when the request never reached the server. */
  status: number;
  /** Human-readable message safe to display in the UI. */
  message: string;
  /** Raw error detail from the response body when available. */
  detail?: unknown;
}

// ---------------------------------------------------------------------------
// Domain types — re-exported for backwards compatibility
// ---------------------------------------------------------------------------

/**
 * Response shape of GET /health.
 */
export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  /** ISO-8601 timestamp of the server's internal clock. */
  timestamp: string;
  /** Semantic version of the running API. */
  version: string;
  uptime_seconds?: number;
  /** Component-level status map, e.g. { db: "ok", redis: "ok" } */
  components?: Record<string, "ok" | "degraded" | "error">;
}

// ---------------------------------------------------------------------------
// WP1.7a/1.7b (AC8): timeouts for stop / emergency-stop / kill-switch-shaped
// requests must allow >= 60s -- flatten can take up to ~30s server-side plus
// a 5s outer margin. Admin-proxy-route timeouts of the same shape live in
// ./admin-fetch (kill-switch press/clear, resume, entries-latch clear); these
// two are for the direct (non-admin, X-API-Key-only) backend calls below.
// ---------------------------------------------------------------------------

/** DELETE /api/v1/runs/{id} — a flatten pass can take up to ~35s server-side. */
export const STOP_TIMEOUT_MS = 65_000;
/** POST /api/v1/runs/{id}/emergency-stop — same flatten budget as stop. */
export const EMERGENCY_STOP_TIMEOUT_MS = 65_000;
/**
 * POST /api/v1/runs/{id}/promote-to-live — WP1.3a (CF-13a-1): a plain 20s
 * fallback is too tight for a cold-DB promotion-gate query; use the same
 * budget as createRun's backtest-eligible timeout (120s) since promotion
 * also runs the full exit-config/pyramiding validator before any DB write.
 */
export const PROMOTE_TIMEOUT_MS = 120_000;

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Generic fetch wrapper with:
 * - Automatic JSON serialisation / deserialisation
 * - Configurable timeout via AbortController (default 10 s; callers may override)
 * - Structured error normalisation into ApiResult
 *
 * @template T - The expected shape of a successful JSON response body.
 */
async function apiFetch<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = 10_000,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const url = `${BASE_URL}${path}`;

  try {
    const response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...init?.headers,
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text().catch(() => undefined);
      }

      return {
        ok: false,
        error: {
          status: response.status,
          message: httpStatusMessage(response.status),
          detail,
        },
      };
    }

    // Skip JSON parsing when the response carries no JSON body.
    // Covers HTTP 204 (No Content) and any 2xx with a non-JSON content-type
    // (e.g. HTTP 202 Accepted with an empty body from the retrain endpoint).
    const ct = response.headers.get("content-type") ?? "";
    if (response.status === 204 || !ct.includes("application/json")) {
      return { ok: true, data: undefined as unknown as T };
    }

    const data: T = (await response.json()) as T;
    return { ok: true, data };
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof DOMException && err.name === "AbortError") {
      return {
        ok: false,
        error: {
          status: 0,
          message: "Request timed out. The operation may still be running on the server.",
        },
      };
    }

    return {
      ok: false,
      error: {
        status: 0,
        message:
          err instanceof Error
            ? err.message
            : "An unexpected network error occurred.",
      },
    };
  }
}

// ---------------------------------------------------------------------------
// Convenience helpers for HTTP verbs
// ---------------------------------------------------------------------------

export function apiGet<T>(path: string, init?: Omit<RequestInit, "method">) {
  return apiFetch<T>(path, { ...init, method: "GET" });
}

export function apiPost<T>(
  path: string,
  body?: unknown,
  init?: Omit<RequestInit, "method" | "body">,
) {
  return apiFetch<T>(path, {
    ...init,
    method: "POST",
    body: body !== undefined ? JSON.stringify(body) : null,
  });
}

export function apiDelete<T>(
  path: string,
  init?: Omit<RequestInit, "method">,
) {
  return apiFetch<T>(path, { ...init, method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/** GET /health — lightweight liveness probe used by the dashboard. */
export async function fetchHealth(): Promise<ApiResult<HealthResponse>> {
  return apiGet<HealthResponse>("/health", { cache: "no-store" });
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/** GET /api/v1/runs — paginated list of all runs. */
export async function fetchRuns(params?: {
  offset?: number;
  limit?: number;
  mode?: string;
  status?: string;
  strategy?: string;
  symbol?: string;
  createdAfter?: string;
  createdBefore?: string;
  minClosedTrades?: number;
  sortBy?: "created_at" | "n_closed_trades";
  sortOrder?: "asc" | "desc";
}): Promise<ApiResult<RunListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.mode) qs.set("mode", params.mode);
  if (params?.status) qs.set("status", params.status);
  if (params?.strategy) qs.set("strategy", params.strategy);
  if (params?.symbol) qs.set("symbol", params.symbol);
  if (params?.createdAfter) qs.set("created_after", params.createdAfter);
  if (params?.createdBefore) qs.set("created_before", params.createdBefore);
  if (params?.minClosedTrades !== undefined)
    qs.set("min_closed_trades", String(params.minClosedTrades));
  if (params?.sortBy) qs.set("sort_by", params.sortBy);
  if (params?.sortOrder) qs.set("sort_order", params.sortOrder);
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<RunListResponse>(`/api/v1/runs${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id} — single run detail. */
export async function fetchRun(id: string): Promise<ApiResult<Run>> {
  return apiGet<Run>(`/api/v1/runs/${id}`, { cache: "no-store" });
}

/**
 * POST /api/v1/runs — create / start a new run (120 s timeout for backtests).
 *
 * WP1.7a/SY-10 (S13): a live-mode confirmation is sent EXCLUSIVELY via the
 * `X-Live-Confirm-Token` request header, never in the JSON body — pass the
 * typed token as `liveConfirmToken`, not on `body.confirmToken` (which the
 * UI must never populate; see the field's own deprecation note in
 * `./types`). The backend already prefers the header over the deprecated
 * body field (`runs.py:515`).
 *
 * WP1.3a (CF-13a-1): `body.allowPyramiding` is always sent explicitly by the
 * new-run form now (never omitted) — see `strategyDefaultAllowPyramiding` in
 * `./exit-config`. A 201 here may carry `configWarnings[]` (SY-13a-18/§5);
 * a 422 may carry the structured `invalid_exit_config` /
 * `exit_manager_required` / `live_pyramiding_forbidden` envelope, unwrap
 * with `unwrapExitConfigDetail` from `./exit-config`.
 */
export async function createRun(
  body: RunCreateRequest,
  liveConfirmToken?: string,
): Promise<ApiResult<Run>> {
  return apiFetch<Run>(
    "/api/v1/runs",
    {
      method: "POST",
      body: JSON.stringify(body),
      // Conditional spread (not `headers: token ? {...} : undefined`) so no
      // property is ever explicitly assigned `undefined` under this repo's
      // `exactOptionalPropertyTypes` tsconfig.
      ...(liveConfirmToken
        ? { headers: { "X-Live-Confirm-Token": liveConfirmToken } }
        : {}),
    },
    120_000,
  );
}

/**
 * POST /api/v1/runs/{id}/promote-to-live — promote a stopped, eligible paper
 * run to a new live run (apps/api/routers/runs.py `promote_to_live`).
 *
 * Unlike `resume_run`, this endpoint requires only the live-trading
 * confirmation header (`X-Live-Confirm-Token`) — no `X-Admin-Key` — so it is
 * called directly here rather than through an `/api/admin/*` proxy.
 *
 * WP1.3a (CF-13a-1, SY-13a-18): a 422 here may carry the same structured
 * `invalid_exit_config` / `exit_manager_required` / `live_pyramiding_forbidden`
 * envelope as `createRun` — render with `<ExitConfigErrorPanel>`.
 */
export async function promoteRun(
  sourceRunId: string,
  liveConfirmToken?: string,
): Promise<ApiResult<Run>> {
  return apiFetch<Run>(
    `/api/v1/runs/${sourceRunId}/promote-to-live`,
    {
      method: "POST",
      body: JSON.stringify({}),
      ...(liveConfirmToken
        ? { headers: { "X-Live-Confirm-Token": liveConfirmToken } }
        : {}),
    },
    PROMOTE_TIMEOUT_MS,
  );
}

/**
 * DELETE /api/v1/runs/{id} — stop a running run (WP1.7a).
 *
 * `flatten` is REQUIRED by the backend for a running LIVE run (omitting it
 * returns 422 `flatten_decision_required`); optional for paper/orphaned/
 * resuming (default false there). AC8: >= 60s timeout — an accepted
 * `flatten=true` can run the engine's flatten pass for up to ~30s plus a 5s
 * outer margin server-side.
 */
export async function stopRun(
  id: string,
  opts?: { flatten?: boolean },
): Promise<ApiResult<RunStopResponse>> {
  const qs = opts?.flatten === undefined ? "" : `?flatten=${opts.flatten}`;
  return apiFetch<RunStopResponse>(
    `/api/v1/runs/${id}${qs}`,
    { method: "DELETE" },
    STOP_TIMEOUT_MS,
  );
}

/**
 * POST /api/v1/runs/{id}/emergency-stop — hard-stop, never refuses (SY-07).
 * `flatten` is optional (default false). AC8: >= 60s timeout, same rationale
 * as `stopRun`.
 */
export async function emergencyStop(
  id: string,
  opts?: { flatten?: boolean; reason?: string },
): Promise<ApiResult<RunEmergencyStopResponse>> {
  const qs = opts?.flatten === undefined ? "" : `?flatten=${opts.flatten}`;
  return apiFetch<RunEmergencyStopResponse>(
    `/api/v1/runs/${id}/emergency-stop${qs}`,
    {
      method: "POST",
      ...(opts?.reason ? { headers: { "X-Emergency-Reason": opts.reason } } : {}),
    },
    EMERGENCY_STOP_TIMEOUT_MS,
  );
}

/**
 * GET /api/v1/emergency/kill-switch — read-only global latch status for the
 * S13 sidebar badge. Requires only X-API-Key (not admin) — called directly,
 * never through an /api/admin/* proxy (SY-11: only admin-key-gated
 * endpoints get a proxy; this one doesn't need one).
 */
export async function fetchKillSwitchStatus(): Promise<ApiResult<KillSwitchStatus>> {
  return apiGet<KillSwitchStatus>("/api/v1/emergency/kill-switch", {
    cache: "no-store",
  });
}

/** PATCH /api/v1/runs/{id}/archive — archive a stopped/error run. */
export async function archiveRun(id: string): Promise<ApiResult<Run>> {
  return apiFetch<Run>(`/api/v1/runs/${id}/archive`, { method: "PATCH" });
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

/** GET /api/v1/runs/{id}/portfolio — portfolio summary. */
export async function fetchPortfolio(
  runId: string,
): Promise<ApiResult<Portfolio>> {
  return apiGet<Portfolio>(`/api/v1/runs/${runId}/portfolio`, {
    cache: "no-store",
  });
}

/** GET /api/v1/portfolio/summary — aggregate cross-run portfolio. */
export async function fetchAggregatePortfolio(): Promise<ApiResult<AggregatePortfolio>> {
  return apiGet<AggregatePortfolio>("/api/v1/portfolio/summary", {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id}/equity-curve — equity curve time series. */
export async function fetchEquityCurve(
  runId: string,
  limit = 1000,
): Promise<ApiResult<EquityCurveResponse>> {
  return apiGet<EquityCurveResponse>(
    `/api/v1/runs/${runId}/equity-curve?limit=${limit}`,
    { cache: "no-store" },
  );
}

/** GET /api/v1/runs/{id}/trades — completed trades. */
export async function fetchTrades(
  runId: string,
  params?: { offset?: number; limit?: number },
): Promise<ApiResult<TradeListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<TradeListResponse>(`/api/v1/runs/${runId}/trades${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id}/orders — orders for a run. */
export async function fetchOrders(
  runId: string,
  params?: { offset?: number; limit?: number; status?: string },
): Promise<ApiResult<OrderListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.status) qs.set("status", params.status);
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<OrderListResponse>(`/api/v1/runs/${runId}/orders${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id}/fills — execution fills for a run. */
export async function fetchFills(
  runId: string,
  params?: { offset?: number; limit?: number; symbol?: string },
): Promise<ApiResult<FillListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.symbol) qs.set("symbol", params.symbol);
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<FillListResponse>(`/api/v1/runs/${runId}/fills${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id}/positions — open positions. */
export async function fetchPositions(
  runId: string,
): Promise<ApiResult<PositionListResponse>> {
  return apiGet<PositionListResponse>(`/api/v1/runs/${runId}/positions`, {
    cache: "no-store",
  });
}

// ---------------------------------------------------------------------------
// Adaptive Learning
// ---------------------------------------------------------------------------

/** GET /api/v1/runs/{runId}/learning — adaptive learning state for a running engine. */
export async function fetchLearningState(
  runId: string,
): Promise<ApiResult<LearningState>> {
  return apiGet<LearningState>(`/api/v1/runs/${runId}/learning`, {
    cache: "no-store",
  });
}

// ---------------------------------------------------------------------------
// Live Diagnostics
// ---------------------------------------------------------------------------

/** GET /api/v1/runs/{id}/diagnostics - real-time engine state for a running run. */
export async function fetchDiagnostics(
  runId: string,
): Promise<ApiResult<Record<string, unknown>>> {
  return apiGet<Record<string, unknown>>(
    `/api/v1/runs/${runId}/diagnostics`,
    { cache: "no-store" },
  );
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/** GET /api/v1/strategies — list all available strategies. */
export async function fetchStrategies(): Promise<ApiResult<StrategyListResponse>> {
  return apiGet<StrategyListResponse>("/api/v1/strategies", { cache: "no-store" });
}

/** GET /api/v1/strategies/{name}/schema — strategy parameter schema. */
export async function fetchStrategySchema(
  name: string,
): Promise<ApiResult<Strategy>> {
  return apiGet<Strategy>(`/api/v1/strategies/${name}/schema`);
}

// ---------------------------------------------------------------------------
// ML Model Versions
// ---------------------------------------------------------------------------

/**
 * GET /api/v1/ml/models — list trained model versions.
 *
 * @param symbol  Optional filter by trading pair (e.g. "BTC/USD").
 * @param activeOnly  When true, return only the active version per symbol.
 */
export async function fetchModelVersions(
  symbol?: string,
  activeOnly?: boolean,
): Promise<ApiResult<ModelVersionListResponse>> {
  const qs = new URLSearchParams();
  if (symbol) qs.set("symbol", symbol);
  if (activeOnly !== undefined) qs.set("active_only", String(activeOnly));
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<ModelVersionListResponse>(`/api/v1/ml/models${query}`, {
    cache: "no-store",
  });
}

/**
 * POST /api/v1/ml/retrain/{symbol} — trigger a model retrain for a symbol.
 * The backend returns 202 Accepted with a JSON body; the function returns ApiResult<void>.
 *
 * @param symbol    Trading pair to retrain, e.g. "BTC/USD".
 * @param timeframe OHLCV timeframe, e.g. "1h".
 */
export async function retrainModel(
  symbol: string,
  timeframe: string,
): Promise<ApiResult<void>> {
  const encodedSymbol = encodeURIComponent(symbol);
  return apiPost<void>(
    `/api/v1/ml/retrain/${encodedSymbol}?timeframe=${encodeURIComponent(timeframe)}`,
  );
}

/**
 * POST /api/v1/ml/train — train a new model from OHLCV data.
 * Unlike retrain (which needs 50+ trades), this always works.
 */
export async function trainModel(
  symbol: string,
  timeframe: string,
  bars: number = 2000,
  exchange: string = "coinbase",
): Promise<ApiResult<Record<string, unknown>>> {
  const params = new URLSearchParams({
    symbol,
    timeframe,
    bars: String(bars),
    exchange,
  });
  return apiPost<Record<string, unknown>>(`/api/v1/ml/train?${params.toString()}`);
}

/**
 * PUT /api/v1/ml/models/{id}/activate — mark a model version as active.
 * Returns the updated ModelVersion record.
 *
 * @param id  UUID of the model version to activate.
 */
export async function activateModelVersion(
  id: string,
): Promise<ApiResult<ModelVersion>> {
  return apiFetch<ModelVersion>(`/api/v1/ml/models/${id}/activate`, {
    method: "PUT",
  });
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

function httpStatusMessage(status: number): string {
  const messages: Partial<Record<number, string>> = {
    400: "Bad request — the server rejected the input.",
    401: "Unauthorised — authentication is required.",
    403: "Forbidden — you do not have permission.",
    404: "Not found — the resource does not exist.",
    409: "Conflict — the request could not be completed due to a conflict.",
    422: "Validation error — check the request payload.",
    429: "Rate limited — too many requests, please slow down.",
    500: "Internal server error — the API encountered an unexpected error.",
    502: "Bad gateway — the API is unreachable.",
    503: "Service unavailable — the API is temporarily down.",
    504: "Gateway timeout — the API took too long to respond.",
  };
  return messages[status] ?? `HTTP ${status} — an error occurred.`;
}

// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------

/**
 * POST /api/v1/optimize — run a parameter grid search.
 *
 * Uses a 300 s timeout because a large grid (e.g. 500 combinations) can take
 * several minutes of sequential backtest execution on the server.
 */
export async function runOptimization(
  body: OptimizeRequest,
): Promise<ApiResult<OptimizeResponse>> {
  return apiFetch<OptimizeResponse>(
    "/api/v1/optimize",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
    300_000,
  );
}

/**
 * GET /api/v1/optimize — paginated list of saved optimization run summaries.
 * Does NOT include the entries array — use fetchOptimizationRun(id) for full detail.
 */
export async function fetchOptimizationRuns(params?: {
  offset?: number;
  limit?: number;
}): Promise<ApiResult<OptimizationRunListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<OptimizationRunListResponse>(`/api/v1/optimize${query}`, {
    cache: "no-store",
  });
}

/**
 * GET /api/v1/optimize/{id} — full detail of a saved optimization run, including all entries.
 *
 * Uses a 30 s timeout — the entries JSONB column can be large on cold DB connections.
 */
export async function fetchOptimizationRun(
  id: string,
): Promise<ApiResult<OptimizeResponse>> {
  // 30 s: entries JSONB column can be large on cold DB connections.
  return apiFetch<OptimizeResponse>(`/api/v1/optimize/${id}`, {
    cache: "no-store",
  }, 30_000);
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/** Format a monetary string for display (2 decimal places, with commas). */
export function formatCurrency(value: string, decimals = 2): string {
  const n = parseFloat(value);
  if (isNaN(n)) return value;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(n);
}

/** Format a percent fraction (0.0-1.0) as a percentage string. */
export function formatPct(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

// ---------------------------------------------------------------------------
// Market Signals
// ---------------------------------------------------------------------------

/** GET /api/v1/signals/current — current cached values for all market signal sources. */
export async function fetchMarketSignals(): Promise<ApiResult<Record<string, unknown>>> {
  return apiGet<Record<string, unknown>>("/api/v1/signals/current", {
    cache: "no-store",
  });
}
