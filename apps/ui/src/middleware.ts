/**
 * apps/ui/src/middleware.ts
 * --------------------------
 * Next.js Edge middleware that enforces authentication for all dashboard
 * routes.  Unauthenticated requests are redirected to the NextAuth
 * sign-in page.
 *
 * NEXTAUTH_SECRET must be available in the Edge runtime bundle. This is
 * satisfied by baking it in as a build ARG in Dockerfile.ui (CR-001):
 *   ARG NEXTAUTH_SECRET
 *   ENV NEXTAUTH_SECRET=$NEXTAUTH_SECRET
 * The same value is also injected at runtime via docker-compose environment,
 * so route handlers and server components read an identical value.
 *
 * Matcher deliberately excludes:
 *   /api/auth/*    — NextAuth's own OAuth callback routes (must be public)
 *   /_next/*       — Next.js static internals
 *   /favicon.ico   — favicon (no auth needed)
 *   /*.png etc.    — public static assets
 */

import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

// CF-97-37 (DR-9737-18): `/bff/*` stays matched, but an unauthenticated browser
// call gets a JSON 401 (no sign-in redirect that fetch would follow into HTML);
// every other path keeps the redirect. The handler checks the session again with
// `getServerSession`. This file reads no allow-list env (it is build-time baked).
export default withAuth(
  function middleware(req) {
    if (req.nextUrl.pathname.startsWith("/bff/") && !req.nextauth.token) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.next();
  },
  {
    callbacks: {
      authorized: ({ req, token }) => req.nextUrl.pathname.startsWith("/bff/") || !!token,
    },
  },
);

export const config = {
  matcher: [
    /*
     * Match every path EXCEPT (each alternative is anchored to whole segments or to the end):
     *   - /api/auth and below (NextAuth routes)
     *   - /_next/static and /_next/image, and below
     *   - /favicon.ico
     *   - a ROOT-level image file (/logo.png): one segment, ending in the extension.
     *     Never a path with a further segment, so /runs/x.png, /bff/.../x.png and
     *     /api/admin/runs/x.png/resume stay behind the session check (CF-97-45).
     */
    // MUST continue to match /api/admin/* — see Sprint 50 cycle 3 SEC-001 (frontend security audit).
    // Removing this coverage would silently bypass NextAuth on /api/admin/kill-switch.
    "/((?!api/auth(?:/|$)|_next/static(?:/|$)|_next/image(?:/|$)|favicon\\.ico$|[^/]+\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
