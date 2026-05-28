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

export { default } from "next-auth/middleware";

export const config = {
  matcher: [
    /*
     * Match every path EXCEPT:
     *   - /api/auth/* (NextAuth routes)
     *   - /_next/static/* and /_next/image/* (Next.js internals)
     *   - /favicon.ico, /*.svg, /*.png, /*.jpg, /*.webp (static assets)
     */
    // MUST continue to match /api/admin/* — see Sprint 50 cycle 3 SEC-001 (frontend security audit).
    // Removing this coverage would silently bypass NextAuth on /api/admin/kill-switch.
    "/((?!api/auth|_next/static|_next/image|favicon\\.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)).*)",
  ],
};
