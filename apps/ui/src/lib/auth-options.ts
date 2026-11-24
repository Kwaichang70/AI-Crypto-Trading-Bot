/**
 * apps/ui/src/lib/auth-options.ts
 * ---------------------------------
 * Shared NextAuth authOptions configuration.
 *
 * Extracted from the route handler (CR-002) so that:
 *   - route.ts can import and call NextAuth(authOptions) without re-exporting
 *   - lib/auth.ts can import authOptions without a cross-directory route import
 *   - Test files can import authOptions without triggering NextAuth side effects
 *
 * Strategy: Google OAuth + JWT sessions (no DB tables needed).
 * Access: the signIn callback admits only ADMIN_EMAILS ∪ VIEWER_EMAILS
 * (both comma-separated, case-insensitive; both empty → nobody signs in).
 * Sessions slide: they end after 24 h without use, and every session read
 * re-checks both lists, so a removed address loses access at its next request
 * and a demoted admin becomes a viewer.
 * Role determination: ADMIN_EMAILS env var.
 *   - If session.user.email is in ADMIN_EMAILS → role "admin"
 *   - Otherwise (an admitted VIEWER_EMAILS address) → role "viewer"
 *
 * SR-003: Startup guards are in validateAuthConfig(), called from route.ts
 * at handler creation time (not at module import time) so that unit tests
 * can import authOptions.callbacks without needing real env vars.
 * The guards still run before any HTTP request is ever served.
 */

import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** The one email normalisation used by both callbacks and the list parser. */
function normalizeEmail(raw: unknown): string {
  return typeof raw === "string" ? raw.trim().toLowerCase() : "";
}

function getEmailSet(envName: "ADMIN_EMAILS" | "VIEWER_EMAILS"): Set<string> {
  return new Set((process.env[envName] ?? "").split(",").map(normalizeEmail).filter(Boolean));
}

function getAdminEmails(): Set<string> {
  return getEmailSet("ADMIN_EMAILS");
}

function isAllowedEmail(email: string): boolean {
  return email !== "" && (getAdminEmails().has(email) || getEmailSet("VIEWER_EMAILS").has(email));
}

// ---------------------------------------------------------------------------
// SR-003 / CR-003 / SR-008: Startup guards
//
// Called from route.ts immediately before NextAuth(authOptions) so the guards
// run at handler-creation time (server startup) rather than module-import time.
// This preserves full security coverage while allowing test files to import
// authOptions.callbacks without requiring real environment variables.
// ---------------------------------------------------------------------------

export function validateAuthConfig(): void {
  const nextAuthSecret = process.env.NEXTAUTH_SECRET;
  if (!nextAuthSecret || nextAuthSecret.startsWith("REPLACE_ME")) {
    throw new Error(
      "NEXTAUTH_SECRET is missing or is still the .env.example placeholder. " +
      "Generate with `openssl rand -hex 32` and paste into .env."
    );
  }
  if (nextAuthSecret.length < 32) {
    throw new Error("NEXTAUTH_SECRET must be at least 32 characters.");
  }

  // CR-003 / SR-008: Fail-fast on missing Google OAuth credentials
  const clientId = process.env.GOOGLE_OAUTH_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_OAUTH_CLIENT_SECRET;
  if (!clientId || !clientSecret) {
    throw new Error(
      "GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET must both be set. " +
      "See .env.example for Google Cloud Console setup steps."
    );
  }
}

// ---------------------------------------------------------------------------
// authOptions — the single shared configuration object
//
// clientId / clientSecret are read lazily from process.env so that:
//   a) The module can be imported in tests without real OAuth credentials.
//   b) Route.ts calls validateAuthConfig() before NextAuth(authOptions),
//      guaranteeing the credentials are present before any request is served.
// ---------------------------------------------------------------------------

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_OAUTH_CLIENT_ID ?? "",
      clientSecret: process.env.GOOGLE_OAUTH_CLIENT_SECRET ?? "",
    }),
  ],

  session: {
    strategy: "jwt",
    // 24-hour sliding sessions; revocation takes effect at the next session read
    // (jwt re-check, 37-S).
    maxAge: 60 * 60 * 24,
  },

  // SR-002: Stricter cookie settings — sameSite: "strict" (NextAuth default is "lax")
  cookies: {
    sessionToken: {
      name: `__Secure-next-auth.session-token`,
      options: {
        httpOnly: true,
        sameSite: "strict" as const,
        path: "/",
        secure: true,
      },
    },
  },

  callbacks: {
    /**
     * signIn callback (CF-97-37, UQ-97-12) — allow-list. Only a Google account
     * with a verified email that is in ADMIN_EMAILS or VIEWER_EMAILS may sign
     * in; everything else is denied (NextAuth answers AccessDenied). Fails
     * closed: with both lists empty nobody is admitted.
     */
    async signIn({ user, account, profile }) {
      if (account?.provider !== "google") return false;
      if ((profile as { email_verified?: boolean } | undefined)?.email_verified !== true) {
        return false;
      }
      return isAllowedEmail(normalizeEmail(user?.email));
    },

    /**
     * jwt callback — fires when a JWT is created or updated.
     * We store the role in the token so it survives across session reads
     * without a DB round-trip.
     * Type is inferred from NextAuthOptions — no explicit annotation needed.
     */
    async jwt({ token, user }) {
      const email = normalizeEmail(user?.email ?? token.email);
      // Sessions slide (next-auth re-issues the cookie on every session read),
      // so every refresh (no `user`) re-checks both lists: an address that was
      // removed ends the session (the throw clears the cookie; the message is
      // fixed and carries no email), and a demoted admin becomes a viewer.
      if (!user && !isAllowedEmail(email)) throw new Error("access_revoked");
      if (email) token.role = getAdminEmails().has(email) ? "admin" : "viewer";
      return token;
    },

    /**
     * session callback — shapes what getServerSession / useSession return.
     * Reads role from the JWT token and copies it onto session.user.
     * Type is inferred from NextAuthOptions — no explicit annotation needed.
     */
    async session({ session, token }) {
      if (session.user) {
        session.user.role =
          (token.role as "admin" | "viewer" | undefined) ?? "viewer";
      }
      return session;
    },
  },

  // CR-006: pages.signIn block removed — NextAuth's built-in sign-in page
  // (/api/auth/signin with the Google button) is the default. No custom
  // override needed; specifying signIn: "/api/auth/signin" explicitly is
  // redundant and was flagged as noise by the code critic.
};
