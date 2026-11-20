/**
 * apps/ui/src/__tests__/auth/signin-callback.test.tsx
 * -------------------------------------------------------
 * CF-97-37 / 37-S: the NextAuth `signIn` callback is an allow-list over
 * ADMIN_EMAILS and VIEWER_EMAILS (both configuration-driven). It fails
 * closed when both lists are empty. Also pins that no UI source links to the
 * API host's :8000/docs page.
 */

import fs from "fs";
import path from "path";
import { authOptions } from "@/lib/auth-options";

type SignInParams = Parameters<NonNullable<NonNullable<typeof authOptions.callbacks>["signIn"]>>[0];

const signIn = authOptions.callbacks!.signIn!;

const GOOGLE = { provider: "google", type: "oauth" };

function attempt(
  email: string | null | undefined,
  opts: { profile?: Record<string, unknown>; account?: Record<string, unknown> | null } = {},
) {
  return signIn({
    user: { id: "u1", email: email ?? null },
    account: opts.account === undefined ? GOOGLE : opts.account,
    profile: opts.profile ?? { email_verified: true },
  } as unknown as SignInParams);
}

describe("NextAuth signIn callback — allow-list (37-S)", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env.ADMIN_EMAILS;
    delete process.env.VIEWER_EMAILS;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it("admits an ADMIN_EMAILS address", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    await expect(attempt("admin@example.com")).resolves.toBe(true);
  });

  it("admits a VIEWER_EMAILS address", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    process.env.VIEWER_EMAILS = "viewer@example.com";
    await expect(attempt("viewer@example.com")).resolves.toBe(true);
  });

  it("admits a viewer when ADMIN_EMAILS holds only the none@example.invalid placeholder", async () => {
    process.env.ADMIN_EMAILS = "none@example.invalid";
    process.env.VIEWER_EMAILS = "viewer@example.com";
    await expect(attempt("viewer@example.com")).resolves.toBe(true);
  });

  it("denies an address in neither list", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    process.env.VIEWER_EMAILS = "viewer@example.com";
    await expect(attempt("stranger@example.com")).resolves.toBe(false);
  });

  it("fails closed: both lists unset denies everyone", async () => {
    await expect(attempt("admin@example.com")).resolves.toBe(false);
  });

  it("fails closed: both lists empty or blank-only denies everyone", async () => {
    process.env.ADMIN_EMAILS = "";
    process.env.VIEWER_EMAILS = " , ,";
    await expect(attempt("admin@example.com")).resolves.toBe(false);
    await expect(attempt("")).resolves.toBe(false);
  });

  it("compares case-insensitively and ignores whitespace around entries", async () => {
    process.env.ADMIN_EMAILS = " Admin@Example.com ";
    process.env.VIEWER_EMAILS = "a@example.com,  Viewer@Example.COM ";
    await expect(attempt("ADMIN@example.COM")).resolves.toBe(true);
    await expect(attempt("viewer@example.com")).resolves.toBe(true);
    await expect(attempt("  viewer@example.com  ")).resolves.toBe(true);
  });

  it("matches the whole address only (no substring, suffix or domain match)", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    process.env.VIEWER_EMAILS = "viewer@example.com";
    await expect(attempt("admin@example.com.evil.test")).resolves.toBe(false);
    await expect(attempt("xadmin@example.com")).resolves.toBe(false);
    await expect(attempt("other@example.com")).resolves.toBe(false);
  });

  it("denies a missing or null email", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    await expect(attempt(null)).resolves.toBe(false);
    await expect(attempt(undefined)).resolves.toBe(false);
  });

  it("requires email_verified === true: false, missing and non-boolean are all denied", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    await expect(attempt("admin@example.com", { profile: { email_verified: true } })).resolves.toBe(true);
    await expect(attempt("admin@example.com", { profile: { email_verified: false } })).resolves.toBe(false);
    await expect(attempt("admin@example.com", { profile: {} })).resolves.toBe(false);
    await expect(attempt("admin@example.com", { profile: { email_verified: "true" } })).resolves.toBe(false);
  });

  it("requires the google provider: another provider or a missing account is denied", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    await expect(attempt("admin@example.com", { account: { provider: "github" } })).resolves.toBe(false);
    await expect(attempt("admin@example.com", { account: null })).resolves.toBe(false);
  });

  it("reads the lists at call time (a changed env needs no code change)", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    await expect(attempt("late@example.com")).resolves.toBe(false);
    process.env.VIEWER_EMAILS = "late@example.com";
    await expect(attempt("late@example.com")).resolves.toBe(true);
  });

  it("keeps roles: an admitted VIEWER_EMAILS address gets role viewer, ADMIN_EMAILS admin", async () => {
    process.env.ADMIN_EMAILS = "admin@example.com";
    process.env.VIEWER_EMAILS = "viewer@example.com";
    const jwt = authOptions.callbacks!.jwt!;
    const asAdmin = await jwt({ token: {}, user: { id: "1", email: "admin@example.com" } } as never);
    const asViewer = await jwt({ token: {}, user: { id: "2", email: "viewer@example.com" } } as never);
    expect(asAdmin.role).toBe("admin");
    expect(asViewer.role).toBe("viewer");
  });
});

describe("NextAuth jwt callback — allow-list re-check on refresh (37-S r2)", () => {
  const originalEnv = process.env;
  const jwt = authOptions.callbacks!.jwt!;
  type JwtParams = Parameters<typeof jwt>[0];

  const refresh = async (token: Record<string, unknown>) =>
    jwt({ token, user: undefined, trigger: "update", account: null } as unknown as JwtParams);

  beforeEach(() => {
    process.env = { ...originalEnv };
    process.env.ADMIN_EMAILS = "admin@example.com";
    process.env.VIEWER_EMAILS = "viewer@example.com";
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it("an unlisted token makes the callback throw (next-auth then clears the cookie)", async () => {
    await expect(refresh({ email: "outsider@example.com", role: "viewer" })).rejects.toThrow();
  });

  it("a token with no email also throws (fail closed)", async () => {
    await expect(refresh({ role: "admin" })).rejects.toThrow();
  });

  it("the thrown message is fixed and contains no email", async () => {
    const email = "outsider@example.com";
    const err = await refresh({ email, role: "viewer" }).catch((e: Error) => e);
    expect((err as Error).message).toBe("access_revoked");
    expect((err as Error).message).not.toContain("example.com");
    expect(String((err as Error).stack ?? "").split("\n")[0]).not.toContain(email);
  });

  it("a demoted admin becomes viewer when still listed as a viewer", async () => {
    process.env.ADMIN_EMAILS = "someone-else@example.com";
    process.env.VIEWER_EMAILS = "admin@example.com";
    const out = await refresh({ email: "admin@example.com", role: "admin" });
    expect(out.role).toBe("viewer");
  });

  it("a demoted admin on neither list is cleared", async () => {
    process.env.ADMIN_EMAILS = "someone-else@example.com";
    process.env.VIEWER_EMAILS = "";
    await expect(refresh({ email: "admin@example.com", role: "admin" })).rejects.toThrow();
  });

  it("a promoted viewer becomes admin; a listed viewer and a listed admin survive unchanged", async () => {
    expect((await refresh({ email: "viewer@example.com", role: "viewer" })).role).toBe("viewer");
    expect((await refresh({ email: "admin@example.com", role: "admin" })).role).toBe("admin");
    process.env.ADMIN_EMAILS = "admin@example.com,viewer@example.com";
    expect((await refresh({ email: "viewer@example.com", role: "viewer" })).role).toBe("admin");
  });

  it("uses the same normalisation as signIn (case and surrounding whitespace)", async () => {
    expect((await refresh({ email: "  ADMIN@Example.com ", role: "viewer" })).role).toBe("admin");
  });

  it("both lists empty: every refresh is cleared", async () => {
    process.env.ADMIN_EMAILS = "";
    process.env.VIEWER_EMAILS = "";
    await expect(refresh({ email: "admin@example.com", role: "admin" })).rejects.toThrow();
  });

  it("the initial sign-in (user present) does not re-check; signIn already decided", async () => {
    const out = await jwt({ token: {}, user: { id: "1", email: " Viewer@Example.com " } } as unknown as JwtParams);
    expect(out.role).toBe("viewer");
  });
});

describe("UI sources do not link to the API host's /docs page (37-S)", () => {
  function walk(dir: string, out: string[] = []): string[] {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== "__tests__") walk(full, out);
      } else if (/\.(ts|tsx)$/.test(entry.name)) {
        out.push(full);
      }
    }
    return out;
  }

  it("no non-test source contains ':8000/docs'", () => {
    const srcRoot = path.resolve(__dirname, "../..");
    const offenders = walk(srcRoot).filter((f) => fs.readFileSync(f, "utf8").includes(":8000/docs"));
    expect(offenders).toEqual([]);
  });
});
