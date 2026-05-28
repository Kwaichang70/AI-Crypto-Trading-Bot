/**
 * apps/ui/src/app/api/admin/kill-switch/route.ts
 * ------------------------------------------------
 * Next.js App Router Route Handler — POST /api/admin/kill-switch
 *
 * This is a server-side proxy. It:
 *   1. Validates the NextAuth session and admin role (UI-level gate).
 *   2. Reads INTERNAL_ADMIN_API_KEY from server-side env — NEVER sent to browser.
 *   3. Forwards the call to the FastAPI /emergency/kill-switch endpoint.
 *   4. Returns the upstream JSON + status to the browser (non-2xx replaced with
 *      a safe synthetic error; 2xx KillSwitchResponse shape is passed through).
 *
 * SEC-C3-008: INTERNAL_ADMIN_API_KEY must equal the backend ADMIN_API_KEY value.
 * It must NEVER carry the NEXT_PUBLIC_ prefix — it is server-side only.
 *
 * No edge runtime opt-in: defaults to Node.js runtime, required for
 * process.env.INTERNAL_ADMIN_API_KEY access and next-auth getServerSession.
 */

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth-options";
import { isAdmin } from "@/lib/auth";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface KillSwitchRunError {
  run_id: string;
  error: string;
}

interface KillSwitchUpstreamResponse {
  runs_stopped: string[];
  tasks_cancelled: number;
  engines_removed: number;
  errors: KillSwitchRunError[];
  note: string | null;
}

interface RequestBody {
  reason?: string;
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function POST(request: Request): Promise<Response> {
  // --- Step 1: Session check ---
  const session = await getServerSession(authOptions);
  if (!session) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  // --- Step 2: Admin role check ---
  if (!isAdmin(session)) {
    return Response.json({ error: "Admin role required" }, { status: 403 });
  }

  // --- Step 3: Server-side env guard ---
  const adminKey = process.env.INTERNAL_ADMIN_API_KEY;
  if (!adminKey) {
    return Response.json(
      { error: "Admin key not configured on server" },
      { status: 503 }
    );
  }

  // --- Step 4: Parse optional body (tolerant — body may be absent) ---
  let body: RequestBody = {};
  try {
    const text = await request.text();
    if (text.trim()) {
      body = JSON.parse(text) as RequestBody;
    }
  } catch {
    // Non-JSON or empty body — treat as no reason provided. Intentionally tolerant.
  }

  // --- Step 5: Forward to FastAPI ---
  const apiBase = (
    process.env.INTERNAL_API_URL ?? "http://api:8000"
  ).replace(/\/$/, "");

  // FE-002: undefined when unset (not empty string) — keeps the conditional spread
  // semantics clean: an empty string is never forwarded as X-API-Key.
  const apiKey = process.env.API_KEY;

  const rawReason = body.reason ?? "";

  // FE-SEC-002: sanitise reason before forwarding to an HTTP header to prevent
  // header injection via control characters (CRLF injection, null byte, etc.).
  const sanitisedReason = rawReason.replace(/[\r\n\x00-\x1f\x7f]/g, " ").slice(0, 500);

  let upstream: Response;
  try {
    upstream = await fetch(
      `${apiBase}/api/v1/emergency/kill-switch`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Key": adminKey,
          // Defence-in-depth: include the regular API key even though the admin
          // endpoint uses X-Admin-Key. The backend ignores extra headers safely.
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
          // FE-SEC-002: use sanitised reason, not raw body.reason.
          ...(sanitisedReason ? { "X-Emergency-Reason": sanitisedReason } : {}),
        },
        body: JSON.stringify({}),
      }
    );
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Unknown network error";
    return Response.json(
      { error: `Failed to reach backend: ${message}` },
      { status: 502 }
    );
  }

  // --- Step 6: Return upstream response ---
  // FE-SEC-003: For non-2xx responses, do NOT echo the raw backend body — it may
  // contain internal details. Return a safe synthetic error instead.
  if (!upstream.ok) {
    return Response.json(
      { error: `Upstream returned HTTP ${upstream.status}`, status: upstream.status },
      { status: upstream.status }
    );
  }

  // FE-SEC-004: Wrap upstream.json() in try/catch — a non-JSON 2xx body would
  // otherwise throw an unhandled error and produce a 500 with a raw stack trace.
  let data: KillSwitchUpstreamResponse;
  try {
    data = (await upstream.json()) as KillSwitchUpstreamResponse;
  } catch {
    return Response.json(
      { error: `Backend returned non-JSON response (HTTP ${upstream.status})` },
      { status: 502 }
    );
  }

  return Response.json(data, { status: upstream.status });
}
