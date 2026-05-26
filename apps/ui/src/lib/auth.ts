/**
 * apps/ui/src/lib/auth.ts
 * ------------------------
 * Server-side auth helpers.
 *
 * WARNING: This is UI-only role gating. The FastAPI backend currently
 * authorizes ALL endpoints via X-API-Key only (no role check). A user
 * with `role: "viewer"` who knows the shared API key can call any
 * mutation endpoint (kill-switch, run-create, model-activate) directly
 * via curl/fetch.
 *
 * Backend role enforcement is tracked for Sprint 51 (cycle 3+ backlog).
 * Until then, the Tailscale-only ingress (cycle 1) is the primary
 * authorization boundary.
 *
 * Usage in Server Components / Route Handlers:
 *
 *   import { getSession, isAdmin } from "@/lib/auth";
 *
 *   const session = await getSession();
 *   if (!session) redirect("/api/auth/signin");
 *   if (!isAdmin(session)) return new Response("Forbidden", { status: 403 });
 */

import { getServerSession } from "next-auth";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth-options";

/**
 * Returns the current server-side session or null if the user is not
 * authenticated.  Wraps next-auth's getServerSession with the authOptions
 * so callers don't have to import authOptions themselves.
 */
export async function getSession(): Promise<Session | null> {
  return getServerSession(authOptions);
}

/**
 * Returns true if the session belongs to an admin user.
 * Safe to call with null (returns false).
 */
export function isAdmin(session: Session | null): boolean {
  return session?.user?.role === "admin";
}
