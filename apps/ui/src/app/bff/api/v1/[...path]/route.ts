/**
 * apps/ui/src/app/bff/api/v1/[...path]/route.ts
 * -----------------------------------------------
 * CF-97-37 (37-B): the browser's only door to the API. The browser calls
 * `/bff/api/v1/<rest>`; this handler checks the login session, then
 * `proxyBff` validates the path, applies the default-deny table and calls
 * `INTERNAL_API_URL/api/v1/<rest>` with the server-side key (apps/ui/src/lib/
 * bff-proxy.ts). Exactly one route file may exist under app/bff (pinned).
 *
 * Order: middleware 401 (middleware.ts) -> session 401 (below, the first
 * statement; `getServerSession` runs the jwt allow-list re-check, DR-9737-27,
 * so a revoked address is refused here even with a valid-looking cookie) ->
 * admin role for every write 403 (viewers are read-only, stop and emergency-
 * stop included) -> same-origin check for writes 403 -> path 400 -> table 404
 * -> upstream.
 */

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth-options";
import { isAdmin } from "@/lib/auth";
import { checkSameOriginJsonRequest } from "@/lib/admin-route-guards";
import { proxyBff } from "@/lib/bff-proxy";

// Hard pins (DR-9737-21/25): never prerendered, never on the Edge runtime.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function handle(request: Request, { params }: { params: { path: string[] } }): Promise<Response> {
  const session = await getServerSession(authOptions);
  if (!session) return Response.json({ error: "Unauthorized" }, { status: 401 });
  if (request.method !== "GET") {
    // Viewers are read-only (SE-37B-01): no table lookup, key read or upstream call.
    if (!isAdmin(session)) return Response.json({ error: "forbidden" }, { status: 403 });
    const rejected = checkSameOriginJsonRequest(request, { bodyless: true });
    if (rejected) return rejected;
  }
  return proxyBff(request, params.path);
}

export { handle as GET, handle as POST, handle as PUT, handle as PATCH, handle as DELETE };
