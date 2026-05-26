/**
 * apps/ui/src/__tests__/auth/session-callback.test.tsx
 * -------------------------------------------------------
 * Unit tests for the NextAuth session + JWT callbacks.
 *
 * Tests the role-assignment logic in isolation by importing the authOptions
 * callbacks directly from lib/auth-options and calling them with mock data.
 * This avoids the NextAuth route-handler side effect entirely (CR-002).
 *
 * Baseline: 118 jest tests.  This file adds 8 tests.
 */

import { authOptions } from "@/lib/auth-options";
import type { Session } from "next-auth";
import type { JWT } from "next-auth/jwt";
import type { AdapterUser } from "next-auth/adapters";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type JwtCallbackParams = Parameters<NonNullable<NonNullable<typeof authOptions.callbacks>["jwt"]>>[0];
type SessionCallbackParams = Parameters<NonNullable<NonNullable<typeof authOptions.callbacks>["session"]>>[0];

const jwtCallback = authOptions.callbacks!.jwt!;
const sessionCallback = authOptions.callbacks!.session!;

function makeToken(overrides: Partial<JWT> = {}): JWT {
  return { sub: "google-uid-123", iat: 0, exp: 0, jti: "test", ...overrides };
}

function makeSession(role?: "admin" | "viewer"): Session {
  return {
    expires: new Date(Date.now() + 86400_000).toISOString(),
    user: {
      name: "Test User",
      email: "test@example.com",
      image: null,
      role: role ?? "viewer",
    },
  };
}

function makeAdapterUser(email: string): AdapterUser {
  return {
    id: "uid-1",
    email,
    emailVerified: null,
  };
}

// ---------------------------------------------------------------------------
// JWT callback tests
// ---------------------------------------------------------------------------

describe("NextAuth jwt callback — role assignment", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = {
      ...originalEnv,
      ADMIN_EMAILS: "de.lacombe@gmail.com,another-admin@example.com",
    };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("assigns role admin when email matches ADMIN_EMAILS", async () => {
    const token = makeToken();
    const user = makeAdapterUser("de.lacombe@gmail.com");
    const params = { token, user, trigger: "signIn", account: null } as unknown as JwtCallbackParams;
    const result = await jwtCallback(params);
    expect(result.role).toBe("admin");
  });

  it("assigns role admin case-insensitively", async () => {
    const token = makeToken();
    const user = makeAdapterUser("DE.LACOMBE@GMAIL.COM");
    const params = { token, user, trigger: "signIn", account: null } as unknown as JwtCallbackParams;
    const result = await jwtCallback(params);
    expect(result.role).toBe("admin");
  });

  it("assigns role admin for second admin in ADMIN_EMAILS list", async () => {
    const token = makeToken();
    const user = makeAdapterUser("another-admin@example.com");
    const params = { token, user, trigger: "signIn", account: null } as unknown as JwtCallbackParams;
    const result = await jwtCallback(params);
    expect(result.role).toBe("admin");
  });

  it("assigns role viewer for an email not in ADMIN_EMAILS", async () => {
    const token = makeToken();
    const user = makeAdapterUser("stranger@example.com");
    const params = { token, user, trigger: "signIn", account: null } as unknown as JwtCallbackParams;
    const result = await jwtCallback(params);
    expect(result.role).toBe("viewer");
  });

  it("returns token unchanged when no user is present (token refresh path)", async () => {
    const token = makeToken({ role: "admin" });
    // CR-005: user: undefined — matches NextAuth's actual token refresh call shape.
    // Cast via unknown because NextAuth's TS union requires user but runtime omits it.
    const params = { token, user: undefined, trigger: "update", account: null } as unknown as JwtCallbackParams;
    const result = await jwtCallback(params);
    expect(result.role).toBe("admin");
  });
});

// ---------------------------------------------------------------------------
// Session callback tests
// ---------------------------------------------------------------------------

describe("NextAuth session callback — role propagation", () => {
  it("copies admin role from token onto session.user", async () => {
    const session = makeSession();
    const token = makeToken({ role: "admin" });
    // user field is required by NextAuth types (database strategy) but unused in JWT strategy.
    // Cast via unknown to keep the test focused on JWT-strategy behaviour.
    const params = { session, token, trigger: "update" } as unknown as SessionCallbackParams;
    const result = await sessionCallback(params);
    expect((result.user as { role?: string })?.role).toBe("admin");
  });

  it("copies viewer role from token onto session.user", async () => {
    const session = makeSession();
    const token = makeToken({ role: "viewer" });
    const params = { session, token, trigger: "update" } as unknown as SessionCallbackParams;
    const result = await sessionCallback(params);
    expect((result.user as { role?: string })?.role).toBe("viewer");
  });

  it("defaults to viewer when token has no role (defensive)", async () => {
    const session = makeSession();
    const token = makeToken();
    const params = { session, token, trigger: "update" } as unknown as SessionCallbackParams;
    const result = await sessionCallback(params);
    expect((result.user as { role?: string })?.role).toBe("viewer");
  });
});
