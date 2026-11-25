/** @jest-environment node */
/**
 * CF-97-37 / 37-B (DR-9737-18, V-9737-04): `/bff/*` stays matched by the Edge
 * middleware, but an unauthenticated call gets a JSON 401 (no redirect); every
 * other path keeps the sign-in redirect. The middleware reads no allow-list env.
 */
import { NextRequest } from "next/server";
import { encode } from "next-auth/jwt";
import fs from "fs";
import path from "path";
import { dummySecret } from "@/__tests__/helpers/dummy-secrets";

const SECRET = dummySecret("dummy-session-secret");

beforeAll(() => {
  process.env.NEXTAUTH_SECRET = SECRET;
  process.env.NEXTAUTH_URL = "https://localhost:3100";
});

async function run(pathname: string, cookie?: string) {
  const { default: middleware } = await import("@/middleware");
  const request = new NextRequest(`https://localhost:3100${pathname}`, cookie ? { headers: { cookie } } : {});
  return (middleware as unknown as (r: NextRequest, e: unknown) => Promise<Response>)(request, {});
}

describe("middleware", () => {
  it("unauthenticated /bff/*: 401 JSON, no Location", async () => {
    const res = await run("/bff/api/v1/runs");
    expect(res.status).toBe(401);
    expect(res.headers.get("location")).toBeNull();
    expect(res.headers.get("content-type")).toContain("application/json");
    expect(await res.json()).toEqual({ error: "Unauthorized" });
  });

  it("a forged cookie on /bff/* is the same 401", async () => {
    const res = await run("/bff/api/v1/runs", "__Secure-next-auth.session-token=garbage");
    expect(res.status).toBe(401);
  });

  it("a valid token passes /bff/* through", async () => {
    const token = await encode({ token: { email: "a@example.com", sub: "1" }, secret: SECRET });
    const res = await run("/bff/api/v1/runs", `__Secure-next-auth.session-token=${token}`);
    expect(res.status).toBe(200);
    expect(res.headers.get("x-middleware-next")).toBe("1");
  });

  it.each(["/runs", "/", "/api/admin/kill-switch"])("unauthenticated %s still redirects to sign-in", async (p) => {
    const res = await run(p);
    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toContain("/api/auth/signin");
  });

  it("the matcher still covers /bff and /api/admin (it only skips auth, static and images)", async () => {
    const { config } = await import("@/middleware");
    const re = new RegExp(`^${config.matcher[0]}$`);
    expect(re.test("/bff/api/v1/runs")).toBe(true);
    expect(re.test("/api/admin/kill-switch")).toBe(true);
    expect(re.test("/api/auth/session")).toBe(false);
  });

  it("reads neither ADMIN_EMAILS nor VIEWER_EMAILS (its env is baked at build time)", () => {
    const source = fs.readFileSync(path.resolve(__dirname, "../../middleware.ts"), "utf8");
    expect(source).not.toMatch(/ADMIN_EMAILS|VIEWER_EMAILS/);
  });
});
