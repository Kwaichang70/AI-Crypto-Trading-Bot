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
  OrderListResponse,
  Portfolio,
  PositionListResponse,
  Run,
  RunCreateRequest,
  RunListResponse,
  Strategy,
  StrategyListResponse,
  TradeListResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL =
  typeof window === "undefined"
    ? (
        process.env.INTERNAL_API_URL ??
        process.env.NEXT_PUBLIC_API_URL ??
        "http://api:8000"
      ).replace(/\/$/, "")
    : (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "") || "/api";

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
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Generic fetch wrapper with:
 * - Automatic JSON serialisation / deserialisation
 * - 10-second timeout via AbortController
 * - Structured error normalisation into ApiResult
 *
 * @template T - The expected shape of a successful JSON response body.
 */
async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10_000);

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

    if (response.status === 204) {
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
          message: "Request timed out. Check the API server is running.",
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
}): Promise<ApiResult<RunListResponse>> {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.mode) qs.set("mode", params.mode);
  if (params?.status) qs.set("status", params.status);
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<RunListResponse>(`/api/v1/runs${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id} — single run detail. */
export async function fetchRun(id: string): Promise<ApiResult<Run>> {
  return apiGet<Run>(`/api/v1/runs/${id}`, { cache: "no-store" });
}

/** POST /api/v1/runs — create / start a new run. */
export async function createRun(
  body: RunCreateRequest,
): Promise<ApiResult<Run>> {
  return apiPost<Run>("/api/v1/runs", body);
}

/** DELETE /api/v1/runs/{id} — stop a running run. */
export async function stopRun(id: string): Promise<ApiResult<Run>> {
  return apiDelete<Run>(`/api/v1/runs/${id}`);
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
