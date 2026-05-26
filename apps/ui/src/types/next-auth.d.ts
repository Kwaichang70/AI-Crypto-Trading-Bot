/**
 * apps/ui/src/types/next-auth.d.ts
 * ---------------------------------
 * Module augmentation for next-auth v4.
 *
 * Adds `role: "admin" | "viewer"` to:
 *   - Session.user  (returned by useSession / getServerSession)
 *   - JWT           (stored in the encrypted cookie)
 *
 * Without this file TypeScript would complain that `session.user.role`
 * does not exist on the built-in Session type.
 */

import type { DefaultSession, DefaultJWT } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      role: "admin" | "viewer";
    } & DefaultSession["user"];
  }
}

declare module "next-auth/jwt" {
  interface JWT extends DefaultJWT {
    role?: "admin" | "viewer";
  }
}
