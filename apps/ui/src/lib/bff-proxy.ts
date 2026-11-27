import "server-only";
/**
 * apps/ui/src/lib/bff-proxy.ts
 * ------------------------------
 * CF-97-37 (37-B): the upstream half of the BFF handler
 * (apps/ui/src/app/bff/api/v1/[...path]/route.ts). The handler has already
 * checked the session (401) and, for writes, same-origin (403). Here, in this
 * order and with no upstream call before the last step:
 *   path validation 400 -> default-deny table 404 -> body cap 413 ->
 *   API_KEY / INTERNAL_API_URL read 503 -> structural URL build -> upstream.
 *
 * Upstream calls use undici's own `fetch` with a module-scoped `Agent` per
 * route class (DR-9737-23; never Node's built-in fetch with an npm Agent, no
 * global dispatcher), `redirect: "manual"` (a 3xx would replay the key),
 * and an AbortController budget as the primary limit. Error bodies are
 * `{ "error": "<code>" }` and carry no request data; logs carry a fixed code
 * and a route label only, never a path, header or key.
 */

import { Agent, fetch } from "undici";
import { apiBaseUrl, apiKeyHeader } from "./api-server";
import { matchBff } from "./bff-routes";

const MAX_BODY_BYTES = 1024 * 1024;
const MARGIN_MS = 5_000;
// A response is buffered in memory (ui limit 512m): cap it (FI-9737-I).
const MAX_RESPONSE_BYTES = 64 * 1024 * 1024;

// Request headers that are forwarded; everything else is dropped. The XFF is
// forwarded verbatim (DR-9737-05); the API key is added, never taken.
const REQUEST_ALLOW = [
  "accept",
  "content-type",
  "if-none-match",
  "idempotency-key",
  "x-live-confirm-token",
  "x-emergency-reason",
  "x-forwarded-for",
];
const RESPONSE_ALLOW = ["content-type", "retry-after", "etag", "x-request-id", "idempotency-key", "idempotent-replay"];

const agents = new Map<number, Agent>();
function agentFor(budgetMs: number): Agent {
  let agent = agents.get(budgetMs);
  if (!agent) {
    agent = new Agent({ headersTimeout: budgetMs + MARGIN_MS, bodyTimeout: budgetMs + MARGIN_MS, maxResponseSize: MAX_RESPONSE_BYTES });
    agents.set(budgetMs, agent);
  }
  return agent;
}

const fail = (error: string, status: number) => Response.json({ error }, { status });

export async function proxyBff(request: Request, segments: readonly string[]): Promise<Response> {
  const match = matchBff(request.method, segments);
  if (!match.ok) {
    if (match.status === 400) console.warn("bff.bad_path", { segments: segments.length });
    return fail(match.status === 400 ? "bad_path" : "not_found", match.status);
  }
  const { budgetMs } = match.route;
  const label = `${request.method} ${match.route.parts.join("/")}`;

  let body: ArrayBuffer | null = null;
  if (request.method !== "GET") {
    if (Number(request.headers.get("content-length") ?? 0) > MAX_BODY_BYTES) return fail("body_too_large", 413);
    const buffer = await request.arrayBuffer();
    if (buffer.byteLength > MAX_BODY_BYTES) return fail("body_too_large", 413);
    body = buffer.byteLength ? buffer : null;
  }

  const keyHeader = apiKeyHeader();
  if (!keyHeader) return fail("api_key_not_configured", 503);
  const base = new URL(apiBaseUrl());
  const path = `/api/v1/${segments.map(encodeURIComponent).join("/")}`;
  const target = new URL(base);
  target.pathname = path;
  const search = new URL(request.url).searchParams;
  search.delete("api_key");
  target.search = search.toString();
  if (target.origin !== base.origin || target.pathname !== path) return fail("bad_path", 400);

  const headers: Record<string, string> = { ...keyHeader };
  for (const name of REQUEST_ALLOW) {
    const value = request.headers.get(name);
    if (value !== null) headers[name] = value;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), budgetMs);
  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      redirect: "manual",
      signal: controller.signal,
      dispatcher: agentFor(budgetMs),
    });
    if (upstream.status >= 300 && upstream.status < 400) return fail("upstream_redirect", 502);
    if (upstream.status === 401) {
      console.error("bff.upstream_auth_failed", { route: label });
      return fail("upstream_auth_failed", 502);
    }
    if (upstream.status === 429) console.warn("bff.upstream_rate_limited", { route: label });
    const payload = await upstream.arrayBuffer();
    const out = new Headers({ "cache-control": "no-store" });
    for (const name of RESPONSE_ALLOW) {
      const value = upstream.headers.get(name);
      if (value !== null) out.set(name, value);
    }
    return new Response(payload.byteLength ? payload : null, { status: upstream.status, headers: out });
  } catch (error) {
    if (controller.signal.aborted) return fail("upstream_timeout", 504);
    if ((error as { cause?: { code?: string } }).cause?.code === "UND_ERR_RES_EXCEEDED_MAX_SIZE") {
      console.warn("bff.upstream_too_large", { route: label });
      return fail("upstream_too_large", 502);
    }
    return fail("upstream_unreachable", 502);
  } finally {
    clearTimeout(timer);
  }
}
