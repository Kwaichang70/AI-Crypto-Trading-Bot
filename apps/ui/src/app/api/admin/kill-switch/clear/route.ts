/**
 * apps/ui/src/app/api/admin/kill-switch/clear/route.ts
 * --------------------------------------------------------
 * Next.js App Router Route Handler — POST /api/admin/kill-switch/clear
 *
 * Server-side proxy for `POST /api/v1/emergency/kill-switch/clear`
 * (WP1.7a synthesis-spec.md §5 / §9 CF-B4). Same pattern as
 * `../route.ts` (kill-switch press):
 *   1. Rejects cross-origin / non-JSON requests (WP17b-S-05).
 *   2. Validates the NextAuth session and admin role (UI-level gate).
 *   3. Reads INTERNAL_ADMIN_API_KEY from server-side env — NEVER sent to
 *      the browser, NEVER echoed back in any response body.
 *   4. Forwards `{ reason }` to FastAPI with X-Admin-Key.
 *   5. Returns the upstream JSON + status untouched on 2xx (the WP1.7a
 *      `KillSwitchClearResponse` shape is already camelCase); non-2xx
 *      responses are replaced with a safe synthetic error (FE-SEC-003).
 */

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth-options";
import { isAdmin } from "@/lib/auth";
import { checkSameOriginJsonRequest } from "@/lib/admin-route-guards";

interface RequestBody {
  reason?: string;
}

// AC8: the backend's own clear path is a couple of short DB transactions
// (no flatten pass), but this still shares the >= 60s floor for consistency
// with every other kill-switch-adjacent admin request.
const UPSTREAM_TIMEOUT_MS = 65_000;

export async function POST(request: Request): Promise<Response> {
  const rejected = checkSameOriginJsonRequest(request);
  if (rejected) return rejected;

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

  let body: RequestBody = {};
  try {
    const text = await request.text();
    if (text.trim()) body = JSON.parse(text) as RequestBody;
  } catch {
    // Tolerant — validated (and rejected) below instead.
  }

  // WP17b-S-04: a non-string `reason` (e.g. `{"reason": 12345}`) would
  // otherwise crash `.replace(...)` below with an unhandled 500.
  const rawReason = typeof body.reason === "string" ? body.reason : "";
  const sanitisedReason = rawReason.replace(/[\r\n\x00-\x1f\x7f]/g, " ").slice(0, 500);
  if (sanitisedReason.trim().length < 3) {
    return Response.json(
      { error: "reason must be at least 3 characters" },
      { status: 400 },
    );
  }

  const apiBase = (process.env.INTERNAL_API_URL ?? "http://api:8000").replace(/\/$/, "");
  const apiKey = process.env.API_KEY;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let upstream: Response;
  try {
    upstream = await fetch(`${apiBase}/api/v1/emergency/kill-switch/clear`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": adminKey,
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
      },
      body: JSON.stringify({ reason: sanitisedReason }),
      signal: controller.signal,
    });
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
