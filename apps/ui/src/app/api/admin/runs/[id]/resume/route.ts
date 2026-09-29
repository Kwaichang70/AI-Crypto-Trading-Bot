/**
 * apps/ui/src/app/api/admin/runs/[id]/resume/route.ts
 * ---------------------------------------------------------
 * Next.js App Router Route Handler — POST /api/admin/runs/{id}/resume
 *
 * Server-side proxy for `POST /api/v1/runs/{id}/resume`
 * (WP1.7a synthesis-spec.md §4/§5, WP1.8a). This endpoint requires
 * X-Admin-Key AND the full 3-layer live-trading safety gate (env flag +
 * API keys + X-Live-Confirm-Token header) for BOTH mode=normal and
 * mode=protective (runs.py:2463-2474) — so, unlike the reason header on
 * the kill-switch routes, the client's typed `X-Live-Confirm-Token` is
 * forwarded UNCHANGED (not sanitised/rewritten — it is a secret the
 * backend compares byte-for-byte, not free text rendered anywhere).
 *
 *   1. Rejects cross-origin / non-JSON requests (WP17b-S-05).
 *   2. Validates `[id]` is a plain UUID BEFORE the admin key is ever read
 *      (WP17b-S-03) — `encodeURIComponent` alone leaves `..`/`.` untouched,
 *      which the WHATWG URL parser then normalises away, letting an
 *      admin-keyed request reach an unintended upstream path.
 *   3. Validates the NextAuth session and admin role (UI-level gate).
 *   4. Reads INTERNAL_ADMIN_API_KEY from server-side env — NEVER sent to
 *      the browser, NEVER echoed back in any response body.
 *   5. Forwards `?mode=` and `X-Live-Confirm-Token` to FastAPI with
 *      X-Admin-Key added.
 *   6. Returns the upstream JSON + status untouched on 2xx (the
 *      `RunDetailResponse` shape is already camelCase); non-2xx responses
 *      are replaced with a safe synthetic error (FE-SEC-003).
 */

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth-options";
import { isAdmin } from "@/lib/auth";
import {
  checkSameOriginJsonRequest,
  invalidRunIdResponse,
  isValidRunId,
} from "@/lib/admin-route-guards";

// The WP1.8b exchange order scan/cancel/import pass can run for a while —
// share the same >= 60s floor as the other kill-switch-adjacent routes.
const UPSTREAM_TIMEOUT_MS = 65_000;

export async function POST(
  request: Request,
  { params }: { params: { id: string } },
): Promise<Response> {
  const rejected = checkSameOriginJsonRequest(request);
  if (rejected) return rejected;

  // WP17b-S-03: validated BEFORE any session/admin-key work, so a
  // malformed [id] can never trigger a downstream admin-keyed fetch.
  if (!isValidRunId(params.id)) {
    return invalidRunIdResponse();
  }

  const session = await getServerSession(authOptions);
  if (!session) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!isAdmin(session)) {
    return Response.json({ error: "Admin role required" }, { status: 403 });
  }

  const adminKey = process.env.INTERNAL_ADMIN_API_KEY;
  if (!adminKey) {
    return Response.json(
      { error: "Admin key not configured on server" },
      { status: 503 },
    );
  }

  const { searchParams } = new URL(request.url);
  const mode = searchParams.get("mode") === "normal" ? "normal" : "protective";

  // Forwarded UNCHANGED — this is a secret token compared byte-for-byte by
  // the backend's LiveTradingGate, never rendered or logged (I11).
  const liveConfirmToken = request.headers.get("X-Live-Confirm-Token");

  const apiBase = (process.env.INTERNAL_API_URL ?? "http://api:8000").replace(/\/$/, "");
  const apiKey = process.env.API_KEY;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let upstream: Response;
  try {
    upstream = await fetch(
      `${apiBase}/api/v1/runs/${encodeURIComponent(params.id)}/resume?mode=${mode}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key": adminKey,
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
          ...(liveConfirmToken ? { "X-Live-Confirm-Token": liveConfirmToken } : {}),
        },
        body: JSON.stringify({}),
        signal: controller.signal,
      },
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown network error";
    return Response.json({ error: `Failed to reach backend: ${message}` }, { status: 502 });
  } finally {
    clearTimeout(timeoutId);
  }

  if (!upstream.ok) {
    return Response.json(
      { error: `Upstream returned HTTP ${upstream.status}`, status: upstream.status },
      { status: upstream.status },
    );
  }

  let data: unknown;
  try {
    data = await upstream.json();
  } catch {
    return Response.json(
      { error: `Backend returned non-JSON response (HTTP ${upstream.status})` },
      { status: 502 },
    );
  }

  return Response.json(data, { status: upstream.status });
}
