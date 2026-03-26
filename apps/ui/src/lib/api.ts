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
  const query = qs.toString() ? `?${qs.toString()}` : "";
  return apiGet<RunListResponse>(`/api/v1/runs${query}`, {
    cache: "no-store",
  });
}

/** GET /api/v1/runs/{id} — single run detail. */
export async function fetchRun(id: string): Promise<ApiResult<Run>> {
  return apiGet<Run>(`/api/v1/runs/${id}`, { cache: "no-store" });
}

/** POST /api/v1/runs — create / start a new run (120 s timeout for backtests). */
export async function createRun(
  body: RunCreateRequest,
): Promise<ApiResult<Run>> {
  return apiFetch<Run>("/api/v1/runs", {
    method: "POST",
    body: JSON.stringify(body),
  }, 120_000);
}

/** DELETE /api/v1/runs/{id} — stop a running run. */
export async function stopRun(id: string): Promise<ApiResult<Run>> {
  return apiDelete<Run>(`/api/v1/runs/${id}`);
}


/** PATCH /api/v1/runs/{id}/archive — soft-archive a stopped or error run. */
export async function archiveRun(id: string): Promise<ApiResult<Run>> {
  return apiFetch<Run>(`/api/v1/runs/${id}/archive`, { method: "PATCH" });
}

