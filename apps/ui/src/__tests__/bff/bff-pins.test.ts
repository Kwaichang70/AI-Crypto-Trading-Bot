/** @jest-environment node */
/**
 * CF-97-37 / 37-B: source-level pins (DR-9737-18/21/24/25/27, SE-DR3-07/08/10).
 * Each rule is a small pure checker, run on the real tree AND on a planted
 * offender, so a checker that stops matching anything fails here.
 */
import fs from "fs";
import path from "path";

const UI = path.resolve(__dirname, "../../..");
const SRC = path.join(UI, "src");
const REPO = path.resolve(UI, "../..");
const read = (...p: string[]) => fs.readFileSync(path.join(...p), "utf8");

function walk(dir: string, out: string[] = []): string[] {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name !== "__tests__") walk(full, out);
    } else if (/\.(ts|tsx|mjs)$/.test(e.name)) out.push(full);
  }
  return out;
}
const SOURCES = walk(SRC).map((f) => ({ file: path.relative(SRC, f), text: fs.readFileSync(f, "utf8") }));
const offenders = (check: (text: string) => boolean, allowed: string[] = []) =>
  SOURCES.filter((s) => check(s.text) && !allowed.includes(s.file)).map((s) => s.file);

// --- the key is read in one module only -------------------------------------
const readsKey = (t: string) =>
  /process\.env\.API_KEY\b|process\.env\[\s*["'`]API_KEY["'`]\s*\]|\{[^}]*\bAPI_KEY\b[^}]*\}\s*=\s*process\.env|["'`]x-api-key["'`]/i.test(t);

describe("the server-side API key", () => {
  it("only lib/api-server.ts reads API_KEY or names the key header in code", () => {
    expect(offenders(readsKey, ["lib/api-server.ts"])).toEqual([]);
    expect(readsKey(read(SRC, "lib/api-server.ts"))).toBe(true);
  });
  it("the checker finds planted offenders", () => {
    expect(readsKey("const k = process.env.API_KEY;")).toBe(true);
    expect(readsKey('headers: { "X-API-Key": k }')).toBe(true);
    expect(readsKey("process.env['API_KEY']")).toBe(true);
    expect(readsKey('process.env["API_KEY"]')).toBe(true);
    expect(readsKey("process.env[`API_KEY`]")).toBe(true);
    expect(readsKey("const { API_KEY } = process.env;")).toBe(true);
    expect(readsKey("const { A, API_KEY: k, B } = process.env;")).toBe(true);
    expect(readsKey("const header = `Bearer ${process.env.API_KEY}`;")).toBe(true);
    expect(readsKey("// the X-API-Key header is added server-side")).toBe(false);
  });
  it("no NEXT_PUBLIC_ variable carries a key or secret", () => {
    for (const s of SOURCES) expect(s.text).not.toMatch(/NEXT_PUBLIC_[A-Z_]*(KEY|SECRET|TOKEN)/);
  });
});

// --- api-server.ts: server-only, session before every fetch -----------------
const firstCodeLine = (t: string) => t.split("\n").find((l) => l.trim() && !l.trim().startsWith("//"))?.trim();
const sessionBeforeFetch = (t: string) => {
  const body = t.slice(t.indexOf("async function serverFetch"));
  const session = body.indexOf("getServerSession(");
  const fetchAt = body.indexOf("requestJson<");
  const fetches = t.match(/requestJson<|\bfetch\(/g) ?? [];
  return session > -1 && fetchAt > session && fetches.length === 1;
};

describe("api-server.ts (DR-9737-24, DR-9737-27)", () => {
  const text = read(SRC, "lib/api-server.ts");
  it("begins with import server-only", () => expect(firstCodeLine(text)).toBe('import "server-only";'));
  it("its single fetch helper reads the session first and is the only fetch", () => {
    expect(sessionBeforeFetch(text)).toBe(true);
  });
  it("the checker rejects a planted offender (a second fetch, or no session)", () => {
    expect(sessionBeforeFetch(text + '\nexport const leak = () => fetch("http://x");')).toBe(false);
    expect(sessionBeforeFetch(text.replace("getServerSession(authOptions)", "null"))).toBe(false);
    expect(sessionBeforeFetch(text.replace(/\n.*getServerSession\(authOptions\).*\n/, "\n"))).toBe(false);
  });
  it("both server pages use it and force-dynamic", () => {
    for (const page of ["app/page.tsx", "app/strategies/page.tsx"]) {
      const t = read(SRC, page);
      expect(t).toMatch(/from "@\/lib\/api-server"/);
      expect(t).toMatch(/export const dynamic = "force-dynamic"/);
      expect(t).not.toMatch(/\b(fetchHealth|fetchRuns|fetchAggregatePortfolio|fetchStrategies)\b[^;]*from "@\/lib\/api"/);
    }
  });
});

// --- no client module imports the server modules ----------------------------
const importsServerModule = (t: string) =>
  /from\s+["'](@\/lib\/|\.\.?\/(?:.*\/)?)(api-server|bff-proxy|bff-routes)["']|from\s+["']undici["']/.test(t);
const isClient = (t: string) => /^\s*["']use client["']/m.test(t.split("\n").slice(0, 8).join("\n"));

describe("client modules and the server modules", () => {
  it("no 'use client' file imports api-server, bff-proxy, bff-routes or undici", () => {
    expect(SOURCES.filter((s) => isClient(s.text) && importsServerModule(s.text)).map((s) => s.file)).toEqual([]);
  });
  it("the checker finds a planted offender", () => {
    expect(isClient('"use client";\nimport { x } from "@/lib/api-server";') && importsServerModule('import { x } from "@/lib/api-server";')).toBe(true);
    expect(importsServerModule('import { fetch } from "undici";')).toBe(true);
  });
  it("only bff-proxy.ts imports undici", () => {
    expect(offenders((t) => /from\s+["']undici["']/.test(t), ["lib/bff-proxy.ts"])).toEqual([]);
  });
  it("no literal /api/v1 or /bff fetch outside the audited modules", () => {
    const literal = (t: string) => /\bfetch\(\s*[`"']\/(api\/v1|bff)/.test(t);
    expect(offenders(literal)).toEqual([]);
    expect(literal('fetch("/api/v1/runs")')).toBe(true);
  });
});

// --- the BFF route and proxy ------------------------------------------------
describe("the BFF route (DR-9737-18, -21, -25, -26)", () => {
  const routes = SOURCES.filter((s) => s.file.startsWith("app/bff/") && /route\.ts$/.test(s.file));
  it("exactly one route file exists under app/bff", () => {
    expect(routes.map((r) => r.file)).toEqual(["app/bff/api/v1/[...path]/route.ts"]);
  });
  const pinned = (t: string) => /export const runtime = "nodejs";/.test(t) && /export const dynamic = "force-dynamic";/.test(t);
  it("exports runtime nodejs and dynamic force-dynamic (a planted route without them fails)", () => {
    expect(routes.every((r) => pinned(r.text))).toBe(true);
    expect(pinned('export async function GET() { return Response.json({ k: process.env.X }); }')).toBe(false);
    expect(pinned(routes[0]!.text.replace('export const dynamic = "force-dynamic";', ""))).toBe(false);
  });
  const sessionFirst = (t: string) => {
    const body = t.slice(t.indexOf("async function handle"));
    return /^[^\n]*\{\n\s*const session = await getServerSession\(authOptions\);/.test(body);
  };
  it("the handler's first statement is getServerSession(authOptions), not getToken", () => {
    expect(sessionFirst(routes[0]!.text)).toBe(true);
    expect(sessionFirst(routes[0]!.text.replace("const session = await getServerSession(authOptions);", "const s = 1;"))).toBe(false);
    expect(routes[0]!.text).not.toMatch(/getToken/);
  });
  it("the role check sits between the session check and the proxy, and only for writes", () => {
    const t = routes[0]!.text;
    const at = (needle: string) => t.indexOf(needle);
    expect(at("getServerSession(authOptions)")).toBeGreaterThan(-1);
    expect(at("getServerSession(authOptions)")).toBeLessThan(at("!isAdmin(session)"));
    expect(at("!isAdmin(session)")).toBeLessThan(at("checkSameOriginJsonRequest(request"));
    expect(at("!isAdmin(session)")).toBeLessThan(at("proxyBff(request"));
    expect(t).toMatch(/request\.method !== "GET"\) \{\n\s*\/\/[^\n]*\n\s*if \(!isAdmin\(session\)\) return Response\.json\(\{ error: "forbidden" \}, \{ status: 403 \}\)/);
  });
  it("the proxy never decodes again and never reads the raw pathname (T2, T4)", () => {
    for (const f of ["lib/bff-proxy.ts", "lib/bff-routes.ts", "app/bff/api/v1/[...path]/route.ts"]) {
      const t = read(SRC, f);
      expect(t).not.toMatch(/decodeURI(Component)?\(/);
      expect(t).not.toMatch(/nextUrl\.pathname|\.pathname\.split/);
    }
  });
  it("upstream calls use undici's own fetch, redirect manual, no global dispatcher", () => {
    const t = read(SRC, "lib/bff-proxy.ts");
    expect(t).toMatch(/import \{ Agent, fetch \} from "undici";/);
    expect(t).toMatch(/redirect: "manual"/);
    expect(t).toMatch(/dispatcher: agentFor\(/);
    expect(offenders((x) => /setGlobalDispatcher/.test(x))).toEqual([]);
    expect(/setGlobalDispatcher/.test("setGlobalDispatcher(agent)")).toBe(true);
  });
  it("the admin routes take the key from api-server (mandatory 503) and pass Retry-After", () => {
    for (const f of ["kill-switch", "kill-switch/clear", "runs/[id]/resume", "runs/[id]/entries-latch/clear"]) {
      const t = read(SRC, "app/api/admin", f, "route.ts");
      expect(t).toMatch(/apiKeyHeader\(\)/);
      expect(t).toMatch(/API key not configured on server/);
      expect(t).toMatch(/retryAfterHeaders\(upstream\)/);
    }
  });
});

// --- configuration: no build-time path for server secrets -------------------
describe("no build-time env path for server secrets (SE-DR3-10)", () => {
  const SECRETS = /\b(API_KEY|INTERNAL_API_URL|INTERNAL_ADMIN_API_KEY)\b/;
  it("next.config.mjs has no env entry, no rewrites and no browser source maps", () => {
    const code = read(UI, "next.config.mjs").split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
    expect(code).not.toMatch(/\benv\s*:/);
    expect(code).not.toMatch(/rewrites/);
    expect(code).not.toMatch(/productionBrowserSourceMaps\s*:\s*true/);
    expect(code).not.toMatch(SECRETS);
  });
  it("Dockerfile.ui has no ARG or ENV for those names and keeps npm ci", () => {
    const docker = read(REPO, "infra/Dockerfile.ui");
    const argsAndEnvs = docker.split("\n").filter((l) => /^\s*(ARG|ENV)\b/.test(l)).join("\n");
    expect(argsAndEnvs).not.toMatch(SECRETS);
    expect(docker).toMatch(/npm ci/);
  });
  it("the middleware (Edge, build-time env) never names the key variables", () => {
    expect(read(SRC, "middleware.ts")).not.toMatch(SECRETS);
  });
});

// --- supply chain: undici ---------------------------------------------------
describe("undici dependency (DR-9737-23)", () => {
  const pkg = JSON.parse(read(UI, "package.json"));
  const lock = JSON.parse(read(UI, "package-lock.json"));
  const entry = lock.packages["node_modules/undici"];
  it("is an exact version in package.json and in the lockfile root", () => {
    expect(pkg.dependencies.undici).toMatch(/^\d+\.\d+\.\d+$/);
    expect(lock.packages[""].dependencies.undici).toBe(pkg.dependencies.undici);
  });
  it("the lockfile entry is registry-resolved, integrity-pinned, MIT, script-free and Node 20 compatible", () => {
    expect(entry.version).toBe(pkg.dependencies.undici);
    expect(entry.resolved).toMatch(/^https:\/\/registry\.npmjs\.org\/undici\/-\/undici-[\d.]+\.tgz$/);
    expect(entry.integrity).toMatch(/^sha512-/);
    expect(entry.license).toBe("MIT");
    expect(entry.hasInstallScript).toBeUndefined();
    expect(entry.dependencies).toBeUndefined();
    expect(entry.engines.node).toMatch(/^>=20\.\d+\.\d+$/);
  });
});

// --- the measured Next version (R6-2) -----------------------------------------
describe("the traversal check is tied to the locked Next version", () => {
  it("MEASURED_NEXT_VERSION in check-bff-traversal.mjs equals the locked next version", () => {
    const script = read(UI, "scripts/check-bff-traversal.mjs");
    const measured = /MEASURED_NEXT_VERSION = "([^"]+)"/.exec(script)?.[1];
    const lock = JSON.parse(read(UI, "package-lock.json"));
    expect(measured).toBeDefined();
    expect(lock.packages["node_modules/next"].version).toBe(measured);
  });
});
