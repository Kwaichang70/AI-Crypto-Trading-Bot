import "server-only";
/**
 * apps/ui/src/lib/api-server.ts
 * -------------------------------
 * CF-97-37 (37-B, DR-9737-24/27): the ONLY module (with the BFF handler path
 * it feeds) that reads the server-side API key. Importing it from a client
 * module fails `next build` (`server-only`).
 *
 *   - `serverFetch` re-checks the login session with `getServerSession`
 *     (which runs the jwt allow-list re-check) BEFORE any upstream fetch, and
 *     redirects to sign-in on null. The Edge middleware only verifies the JWT.
 *   - every fetch passes `cache: "no-store"` (Next 14 data cache).
 *   - the key is read inside the functions, at request time, from the runtime
 *     environment; it is never a build argument or a `next.config` env entry.
 */

import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import { authOptions } from "@/lib/auth-options";
import { requestJson } from "./api";
import type { ApiResult, HealthResponse } from "./api";
import type { AggregatePortfolio, RunListResponse, StrategyListResponse } from "./types";

/** `INTERNAL_API_URL` from the server environment only (never from a request). */
export function apiBaseUrl(): string {
  return (process.env.INTERNAL_API_URL ?? "http://api:8000").replace(/\/$/, "");
}

/** `{ "X-API-Key": key }`, or null when `API_KEY` is not configured. */
export function apiKeyHeader(): Record<string, string> | null {
  const key = process.env.API_KEY;
  return key ? { "X-API-Key": key } : null;
}

/** `Retry-After` of an upstream response, so a 429 reaches the browser. */
export function retryAfterHeaders(upstream: Response): Record<string, string> {
  const value = upstream.headers.get("retry-after");
  return value ? { "Retry-After": value } : {};
}

async function serverFetch<T>(path: string): Promise<ApiResult<T>> {
  if (!(await getServerSession(authOptions))) redirect("/api/auth/signin");
  return requestJson<T>(`${apiBaseUrl()}${path}`, {
    cache: "no-store",
    headers: { Accept: "application/json", ...apiKeyHeader() },
  });
}

/** GET /health — liveness probe for the dashboard (not under /api/v1). */
export const fetchHealth = () => serverFetch<HealthResponse>("/health");

/** GET /api/v1/runs?limit=N — the dashboard's recent runs. */
export const fetchRuns = (params: { limit: number }) =>
  serverFetch<RunListResponse>(`/api/v1/runs?limit=${params.limit}`);

/** GET /api/v1/portfolio/summary — aggregate cross-run portfolio. */
export const fetchAggregatePortfolio = () =>
  serverFetch<AggregatePortfolio>("/api/v1/portfolio/summary");

/** GET /api/v1/strategies — the strategies page. */
export const fetchStrategies = () => serverFetch<StrategyListResponse>("/api/v1/strategies");
