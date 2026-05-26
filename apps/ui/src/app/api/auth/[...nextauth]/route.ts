/**
 * apps/ui/src/app/api/auth/[...nextauth]/route.ts
 * -------------------------------------------------
 * NextAuth v4 route handler for the App Router.
 *
 * authOptions is defined in lib/auth-options.ts (CR-002: extracted so that
 * lib/auth.ts and test files can import authOptions without routing through
 * this file's NextAuth side effect).
 *
 * validateAuthConfig() is called here — at handler creation time (server
 * startup) — so the SR-003 / CR-003 startup guards run before any HTTP
 * request is ever served, without blocking test-time module imports of
 * authOptions.callbacks.
 *
 * NEXTAUTH_URL and NEXTAUTH_SECRET are read from process.env at runtime
 * (injected via docker-compose environment block).
 * NEXTAUTH_SECRET is ALSO baked into the Edge bundle at build time via
 * ARG/ENV in Dockerfile.ui (CR-001).
 */

import NextAuth from "next-auth";
import { authOptions, validateAuthConfig } from "@/lib/auth-options";

// SR-003 / CR-003: Validate required env vars at startup before serving any request.
validateAuthConfig();

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
