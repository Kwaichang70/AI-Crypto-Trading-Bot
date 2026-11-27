/** @jest-environment node */
/**
 * CF-97-40-B (GHSA-xmf8-cvqr-rfgj, SE-37C2-04): with next-auth 4.24.14,
 * `getToken()` threw on a malformed `Authorization: Bearer` value
 * (`decodeURIComponent`) and the Edge middleware has no try/catch, so any
 * caller could make a request fail with 500. From 4.24.15 `getToken()` returns
 * null: the caller is unauthenticated (redirect for pages, JSON 401 for /bff/*)
 * and the next request is served normally.
 */
import { randomBytes } from "crypto";
import fs from "fs";
import path from "path";
import { NextRequest } from "next/server";
import { encode } from "next-auth/jwt";

const SECRET = randomBytes(32).toString("hex");
const MALFORMED = "Bearer %E0%A4%A";

beforeAll(() => {
  process.env.NEXTAUTH_SECRET = SECRET;
  process.env.NEXTAUTH_URL = "https://localhost:3100";
});

async function run(pathname: string, headers: Record<string, string> = {}) {
  const { default: middleware } = await import("@/middleware");
  const request = new NextRequest(`https://localhost:3100${pathname}`, { headers });
  return (middleware as unknown as (r: NextRequest, e: unknown) => Promise<Response>)(request, {});
}

describe("a malformed Bearer value is an unauthenticated request, never a 500", () => {
  it("a page redirects to sign-in", async () => {
    const res = await run("/runs", { authorization: MALFORMED });
    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toContain("/api/auth/signin");
  });

  it("/bff/* gets the JSON 401", async () => {
    const res = await run("/bff/api/v1/runs", { authorization: MALFORMED });
    expect(res.status).toBe(401);
    expect(res.headers.get("location")).toBeNull();
    expect(await res.json()).toEqual({ error: "Unauthorized" });
  });

  it("/api/admin/* redirects to sign-in", async () => {
    const res = await run("/api/admin/kill-switch", { authorization: MALFORMED });
    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toContain("/api/auth/signin");
  });

  it("the next request with a valid session is still served", async () => {
    await run("/bff/api/v1/runs", { authorization: MALFORMED });
    const token = await encode({ token: { email: "a@example.com", sub: "1" }, secret: SECRET });
    const res = await run("/bff/api/v1/runs", { cookie: `__Secure-next-auth.session-token=${token}` });
    expect(res.status).toBe(200);
    expect(res.headers.get("x-middleware-next")).toBe("1");
  });
});

// SE-9740B-01 / D-016: the 37-S re-check relies on next-auth internals (the jwt
// callback runs on every session read; a throw clears the cookie) that no API
// contract covers, so the version is pinned exactly, like undici.
describe("next-auth is pinned exactly (CF-97-40-B)", () => {
  const UI = path.resolve(__dirname, "../../..");
  const json = (...p: string[]) => JSON.parse(fs.readFileSync(path.join(UI, ...p), "utf8"));
  const manifest = json("package.json").dependencies["next-auth"] as string;
  const lock = json("package-lock.json");

  it("package.json has an exact version, no range character", () => {
    expect(manifest).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it("the lockfile root range and the locked entry equal the manifest", () => {
    expect(lock.packages[""].dependencies["next-auth"]).toBe(manifest);
    expect(lock.packages["node_modules/next-auth"].version).toBe(manifest);
  });

  it("the locked tarball URL is the registry URL of the manifest version (CF-97-50)", () => {
    expect(lock.packages["node_modules/next-auth"].resolved).toBe(`https://registry.npmjs.org/next-auth/-/next-auth-${manifest}.tgz`);
  });

  it("the installed next-auth is the manifest version, and it is at least the fixed 4.24.15", () => {
    const installed = json("node_modules", "next-auth", "package.json").version as string;
    expect(installed).toBe(manifest);
    const [a, b, c] = installed.split(".").map(Number) as [number, number, number];
    expect(a === 4 && (b > 24 || (b === 24 && c >= 15))).toBe(true);
  });
});
