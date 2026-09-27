/**
 * apps/ui/src/app/api/admin/runs/[id]/entries-latch/clear/route.ts
 * -----------------------------------------------------------------------
 * Next.js App Router Route Handler — POST /api/admin/runs/{id}/entries-latch/clear
 *
 * Server-side proxy for `POST /api/v1/runs/{id}/entries-latch/clear`
 * (WP1.7a synthesis-spec.md §4/§5, SY-09). Removes the run's own
 * `flatten_incomplete` per-run latch — requires X-Admin-Key plus a
 * 3-500 character reason; a LIVE run also requires X-Live-Confirm-Token
 * (forwarded unchanged, same rationale as the resume proxy).
 *
 *   1. Rejects cross-origin / non-JSON requests (WP17b-S-05).
 *   2. Validates `[id]` is a plain UUID BEFORE the admin key is ever read
 *      (WP17b-S-03).
 *   3. Validates the NextAuth session and admin role (UI-level gate).
 *   4. Reads INTERNAL_ADMIN_API_KEY from server-side env.
 *   5. Forwards `{ reason }` (+ `X-Live-Confirm-Token` when present) to
 *      FastAPI with X-Admin-Key added.
 *
 * WP17b-C-02 (round 2): this WP ships `clear_entries_latch`'s CF-B2 backend
 * change (a proper `API_MODEL_CONFIG` response model, camelCase on the
 * wire) in THE SAME work package as this proxy — api, infra and ui patches
 * are applied together, not as independent parallel agents — so there is
 * no longer a real merge-order race to hedge against. The previous
 * either-shape (`run_id`/`still_latched_by` snake_case fallback) normaliser
 * has been REMOVED; this proxy now only ever forwards the already-camelCase
 * `EntriesLatchClearResponse` body through untouched.
 */

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth-options";
import { isAdmin } from "@/lib/auth";
import {
  checkSameOriginJsonRequest,
  invalidRunIdResponse,
  isValidRunId,
} from "@/lib/admin-route-guards";

interface RequestBody {
  reason?: string;
}

// No flatten pass on this path — just a couple of short DB transactions —
// but this still shares the >= 60s floor for consistency (AC8).
const UPSTREAM_TIMEOUT_MS = 65_000;

export async function POST(
  request: Request,
  { params }: { params: { id: string } },
): Promise<Response> {
  const rejected = checkSameOriginJsonRequest(request);
  if (rejected) return rejected;

  // WP17b-S-03: validated BEFORE any session/admin-key work.
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

  // Forwarded UNCHANGED — a secret compared byte-for-byte by the backend,
  // never rendered or logged (I11).
  const liveConfirmToken = request.headers.get("X-Live-Confirm-Token");

  const apiBase = (process.env.INTERNAL_API_URL ?? "http://api:8000").replace(/\/$/, "");
  const apiKey = process.env.API_KEY;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let upstream: Response;
  try {
    upstream = await fetch(
      `${apiBase}/api/v1/runs/${encodeURIComponent(params.id)}/entries-latch/clear`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key": adminKey,
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
          ...(liveConfirmToken ? { "X-Live-Confirm-Token": liveConfirmToken } : {}),
        },
        body: JSON.stringify({ reason: sanitisedReason }),
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

  // CF-B2: already camelCase on the wire (see file docstring) — forwarded
  // through untouched.
  return Response.json(data, { status: upstream.status });
}
