/** @jest-environment node */
/**
 * CF-97-37 / 37-B: the real BFF route handler against a real local upstream
 * (real undici, real sockets). Only the session source is mocked. Each case
 * asserts what the UPSTREAM saw, because the point of the BFF is what it sends.
 */
import http from "http";
import type { AddressInfo } from "net";

const getServerSession = jest.fn();
jest.mock("next-auth", () => ({ getServerSession: (...a: unknown[]) => getServerSession(...a) }));
jest.mock("@/lib/auth-options", () => ({ authOptions: {} }));

import * as route from "@/app/bff/api/v1/[...path]/route";
import * as api from "@/lib/api";
import { dummySecret } from "@/__tests__/helpers/dummy-secrets";

interface Seen { method: string; url: string; headers: http.IncomingHttpHeaders; body: string }
const KEY = dummySecret("dummy-api-key");
const UUID = "123e4567-e89b-42d3-a456-426614174000";
const SESSION = { user: { email: "admin@example.com", role: "admin" } };

let upstream: http.Server;
let other: http.Server;
let seen: Seen[] = [];
let otherHits = 0;
let reply: (req: http.IncomingMessage, res: http.ServerResponse) => void;
const jsonOk = (_req: http.IncomingMessage, res: http.ServerResponse) => {
  res.writeHead(200, { "content-type": "application/json", "x-request-id": "r1", "set-cookie": "a=b", "x-secret": "no" });
  res.end('{"ok":true}');
};

const listen = (server: http.Server) =>
  new Promise<number>((resolve) => server.listen(0, "127.0.0.1", () => resolve((server.address() as AddressInfo).port)));

beforeAll(async () => {
  upstream = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      seen.push({ method: req.method ?? "", url: req.url ?? "", headers: req.headers, body: Buffer.concat(chunks).toString() });
      reply(req, res);
    });
  });
  other = http.createServer((_req, res) => {
    otherHits += 1;
    res.end("other");
  });
  process.env.INTERNAL_API_URL = `http://127.0.0.1:${await listen(upstream)}`;
  (global as { __otherPort?: number }).__otherPort = await listen(other);
});

afterAll(async () => {
  await Promise.all([upstream, other].map((s) => new Promise((r) => s.close(r))));
});

beforeEach(() => {
  seen = [];
  otherHits = 0;
  reply = jsonOk;
  process.env.API_KEY = KEY;
  getServerSession.mockReset();
  getServerSession.mockResolvedValue(SESSION);
  jest.spyOn(console, "warn").mockImplementation(() => undefined);
  jest.spyOn(console, "error").mockImplementation(() => undefined);
});

type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
function call(method: Method, segments: string[], init: { headers?: Record<string, string>; body?: string; search?: string } = {}) {
  const request = new Request(`http://ui.test/bff/api/v1/${segments.join("/")}${init.search ?? ""}`, {
    method,
    headers: init.headers ?? {},
    ...(init.body !== undefined ? { body: init.body } : {}),
  });
  return route[method](request, { params: { path: segments } });
}
const JSON_HDR = { "content-type": "application/json" };

describe("order: session, path, table, env, upstream", () => {
  it("no session: 401 JSON and no upstream call", async () => {
    getServerSession.mockResolvedValue(null);
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({ error: "Unauthorized" });
    expect(seen).toHaveLength(0);
  });

  it("session is checked before the path: an unauthenticated bad path gets 401, not 400 (no path oracle)", async () => {
    getServerSession.mockResolvedValue(null);
    expect((await call("GET", ["runs", "..", "emergency"])).status).toBe(401);
    expect((await call("GET", ["nope"])).status).toBe(401);
    expect(seen).toHaveLength(0);
  });

  it("a revoked address (the jwt callback throws) makes getServerSession null: 401, no upstream call (DR-9737-27)", async () => {
    getServerSession.mockImplementation(async () => {
      try {
        const { authOptions: real } = jest.requireActual("@/lib/auth-options") as typeof import("@/lib/auth-options");
        process.env.ADMIN_EMAILS = "someone-else@example.com";
        process.env.VIEWER_EMAILS = "";
        await real.callbacks!.jwt!({ token: { email: "admin@example.com", role: "admin" } } as never);
        return SESSION;
      } catch {
        return null;
      }
    });
    expect((await call("GET", ["runs"])).status).toBe(401);
    expect(seen).toHaveLength(0);
  });

  it("the path is validated before the key is read: a bad path with no key is 400, not 503", async () => {
    delete process.env.API_KEY;
    expect((await call("GET", ["runs", ".."])).status).toBe(400);
    expect((await call("GET", ["runs"])).status).toBe(503);
    expect(seen).toHaveLength(0);
  });
});

describe("path validation through the real handler: 0 upstream calls", () => {
  // The segments are what Next hands the handler after its single decode.
  it.each([
    ["..", ["runs", "..", "emergency", "kill-switch"]],
    ["%2e%2e / %2E%2E / .%2e after Next's decode", ["runs", "..", "emergency"]],
    ["%252e%252e (double encoding)", ["runs", "%2e%2e", "x"]],
    ["%2F outside {symbol}", ["runs", "a/b"]],
    ["a double-encoded symbol (%252F)", ["ml", "retrain", "BTC%2FUSDT"]],
    ["backslash / %5C", ["runs", "a\\b"]],
    ["runs;x", ["runs;x"]],
    ["%20", ["runs", " "]],
    ["+", ["runs", "a+b"]],
    ["%00", ["runs", "a\u0000"]],
    ["non-ASCII dot", ["runs", "．．"]],
    ["empty segment (//)", ["runs", "", UUID]],
    ["a host-changing path", ["runs", "@evil.test"]],
    ["a decoded BTC/USDT/x", ["ml", "retrain", "BTC/USDT/x"]],
    ["runs/%2e%2e/emergency/kill-switch", ["runs", "%2e%2e", "emergency", "kill-switch"]],
  ])("%s -> 400 bad_path", async (_label, segments) => {
    const res = await call("GET", segments);
    expect([400, 404]).toContain(res.status);
    if (res.status === 400) expect(await res.json()).toEqual({ error: "bad_path" });
    expect(seen).toHaveLength(0);
  });

  it.each([
    ["kill-switch.", ["emergency", "kill-switch."]],
    ["a non-UUID {id}", ["runs", "not-a-uuid"]],
    ["an unknown path", ["admin", "x"]],
    ["metrics", ["metrics"]],
  ])("%s -> 404 not_found", async (_label, segments) => {
    const res = await call("GET", segments);
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: "not_found" });
    expect(seen).toHaveLength(0);
  });

  it.each([
    ["POST", ["emergency", "kill-switch"]],
    ["POST", ["emergency", "kill-switch", "clear"]],
    ["POST", ["runs", UUID, "resume"]],
    ["POST", ["runs", UUID, "entries-latch", "clear"]],
  ] as [Method, string[]][])("admin route %s %j is not reachable through the BFF", async (method, segments) => {
    const res = await call(method, segments, { headers: JSON_HDR, body: "{}" });
    expect(res.status).toBe(404);
    expect(seen).toHaveLength(0);
  });

  it.each([
    ["a lower-case symbol", "btc/usdt"],
    ["a one-character side", "B/USDT"],
    ["a dot inside", "BT.C/USDT"],
    ["a third part", "BTC/USDT/EUR"],
    ["a symbol with no slash", "BTCUSDT"],
  ])("POST ml/retrain with %s is refused with no upstream call", async (_label, symbol) => {
    // One segment: Next hands a decoded %2F to the handler inside a single segment.
    const res = await call("POST", ["ml", "retrain", symbol], { headers: JSON_HDR, body: "{}" });
    expect([400, 404]).toContain(res.status);
    expect(seen).toHaveLength(0);
  });

  it("an admitted ml/retrain/BTC%2FUSDT reaches the upstream as /api/v1/ml/retrain/BTC%2FUSDT", async () => {
    const res = await call("POST", ["ml", "retrain", "BTC/USDT"], { headers: JSON_HDR, body: "{}", search: "?timeframe=1h" });
    expect(res.status).toBe(200);
    expect(seen[0]!.url).toBe("/api/v1/ml/retrain/BTC%2FUSDT?timeframe=1h");
  });
});

describe("what the upstream receives", () => {
  it("forwards method, path, body and the key; the response passes through with only allowed headers", async () => {
    const res = await call("POST", ["runs"], { headers: JSON_HDR, body: '{"a":1}' });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
    expect(res.headers.get("x-request-id")).toBe("r1");
    expect(res.headers.get("cache-control")).toBe("no-store");
    expect(res.headers.get("set-cookie")).toBeNull();
    expect(res.headers.get("x-secret")).toBeNull();
    expect(seen[0]).toMatchObject({ method: "POST", url: "/api/v1/runs", body: '{"a":1}' });
    expect(seen[0]!.headers["x-api-key"]).toBe(KEY);
  });

  it("strips every api_key query parameter and nothing else", async () => {
    await call("GET", ["runs"], { search: "?api_key=evil&limit=3&api_key=again&status=stopped" });
    expect(seen[0]!.url).toBe("/api/v1/runs?limit=3&status=stopped");
  });

  it("drops smuggling headers and any client key; forwards the allow-list and XFF verbatim", async () => {
    const dropped = {
      "x-original-url": "/admin", "x-rewrite-url": "/admin", "x-forwarded-prefix": "/x", "x-forwarded-host": "evil.test",
      "x-http-method-override": "DELETE", "x-api-key": "client-key", "x-admin-key": "client-admin", authorization: "Bearer x",
      cookie: "__Secure-next-auth.session-token=abc", "x-middleware-subrequest": "middleware",
    };
    const kept = {
      accept: "application/json", "content-type": "application/json", "idempotency-key": "k-1",
      "x-live-confirm-token": "t-1", "x-emergency-reason": "reason", "x-forwarded-for": "203.0.113.9, 10.0.0.2",
    };
    await call("POST", ["runs"], { headers: { ...dropped, ...kept }, body: "{}" });
    const got = seen[0]!;
    for (const name of Object.keys(dropped)) {
      if (name !== "x-api-key") expect(got.headers[name]).toBeUndefined();
    }
    expect(got.headers["x-api-key"]).toBe(KEY);
    for (const [name, value] of Object.entries(kept)) expect(got.headers[name]).toBe(value);
    expect(got.method).toBe("POST");
  });

  it("the upstream method is exactly the request method (override header ignored)", async () => {
    await call("DELETE", ["runs", UUID], { headers: { "x-http-method-override": "POST" } });
    expect(seen[0]!.method).toBe("DELETE");
  });

  it("stop and emergency-stop pass X-Emergency-Reason (DR-9737-19)", async () => {
    await call("POST", ["runs", UUID, "emergency-stop"], { headers: { "x-emergency-reason": "halt" }, search: "?flatten=true" });
    expect(seen[0]).toMatchObject({ method: "POST", url: `/api/v1/runs/${UUID}/emergency-stop?flatten=true` });
    expect(seen[0]!.headers["x-emergency-reason"]).toBe("halt");
  });
});

describe("upstream outcomes", () => {
  it("a 307 to another origin is 502 upstream_redirect: one upstream call, the other origin never sees the key", async () => {
    reply = (_req, res) => {
      res.writeHead(307, { location: `http://127.0.0.1:${(global as { __otherPort?: number }).__otherPort}/steal` });
      res.end();
    };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "upstream_redirect" });
    expect(seen).toHaveLength(1);
    expect(otherHits).toBe(0);
  });

  it("an upstream 401 (bad server key) is 502 upstream_auth_failed, not a logout 401", async () => {
    reply = (_req, res) => { res.writeHead(401, JSON_HDR); res.end('{"detail":"nope"}'); };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "upstream_auth_failed" });
    const logged = JSON.stringify((console.error as jest.Mock).mock.calls);
    expect(logged).toContain("bff.upstream_auth_failed");
    expect(logged).not.toContain(KEY);
  });

  it("a 429 reaches the browser unchanged with Retry-After", async () => {
    reply = (_req, res) => { res.writeHead(429, { ...JSON_HDR, "retry-after": "17" }); res.end('{"detail":"slow down"}'); };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("17");
    expect(await res.json()).toEqual({ detail: "slow down" });
    expect(JSON.stringify((console.warn as jest.Mock).mock.calls)).toContain("bff.upstream_rate_limited");
  });

  it.each([403, 404, 409, 422, 500, 503])("an upstream %i and its body pass through", async (status) => {
    reply = (_req, res) => { res.writeHead(status, JSON_HDR); res.end(`{"detail":"${status}"}`); };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(status);
    expect(await res.json()).toEqual({ detail: String(status) });
  });

  it("an unreachable upstream is 502 upstream_unreachable", async () => {
    const saved = process.env.INTERNAL_API_URL;
    process.env.INTERNAL_API_URL = "http://127.0.0.1:1";
    const res = await call("GET", ["runs"]);
    process.env.INTERNAL_API_URL = saved;
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "upstream_unreachable" });
  });

  it("a missing API_KEY is 503 api_key_not_configured and nothing is sent", async () => {
    delete process.env.API_KEY;
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ error: "api_key_not_configured" });
    expect(seen).toHaveLength(0);
  });

  it("a body above 1 MiB is 413 and is not forwarded (declared and actual)", async () => {
    const big = "x".repeat(1024 * 1024 + 1);
    expect((await call("POST", ["optimize"], { headers: JSON_HDR, body: big })).status).toBe(413);
    const lying = new Request("http://ui.test/bff/api/v1/optimize", { method: "POST", headers: { ...JSON_HDR, "content-length": "99999999" }, body: "{}" });
    expect((await route.POST(lying, { params: { path: ["optimize"] } })).status).toBe(413);
    expect(seen).toHaveLength(0);
  });
});

describe("CSRF for state-changing calls", () => {
  it("a cross-site write is 403 with no upstream call; a cross-site GET is not checked", async () => {
    const res = await call("POST", ["runs"], { headers: { ...JSON_HDR, "sec-fetch-site": "cross-site" }, body: "{}" });
    expect(res.status).toBe(403);
    expect(seen).toHaveLength(0);
    expect((await call("GET", ["runs"], { headers: { "sec-fetch-site": "cross-site" } })).status).toBe(200);
  });

  it("a same-origin bodiless DELETE with no Content-Type passes; a text/plain body is 415", async () => {
    expect((await call("DELETE", ["runs", UUID], { headers: { "sec-fetch-site": "same-origin" } })).status).toBe(200);
    const res = await call("POST", ["runs"], { headers: { "content-type": "text/plain" }, body: "x" });
    expect(res.status).toBe(415);
    expect(seen).toHaveLength(1);
  });

  it("a write whose Origin is another host is 403", async () => {
    process.env.NEXTAUTH_URL = "https://dash.example.test";
    const res = await call("PATCH", ["runs", UUID, "archive"], { headers: { origin: "https://evil.test" } });
    expect(res.status).toBe(403);
    expect(seen).toHaveLength(0);
  });
});

describe("viewers are read-only (SE-37B-01): role check right after the session", () => {
  const VIEWER = { user: { email: "viewer@example.com", role: "viewer" } };
  beforeEach(() => getServerSession.mockResolvedValue(VIEWER));

  it("a viewer's GET passes", async () => {
    expect((await call("GET", ["runs"])).status).toBe(200);
    expect(seen).toHaveLength(1);
  });

  it.each([
    ["POST", ["runs"]],
    ["POST", ["runs", UUID, "promote-to-live"]],
    ["POST", ["runs", UUID, "emergency-stop"]],
    ["DELETE", ["runs", UUID]],
    ["PUT", ["ml", "models", UUID, "activate"]],
    ["PATCH", ["runs", UUID, "archive"]],
    ["POST", ["optimize"]],
    ["POST", ["ml", "train"]],
  ] as [Method, string[]][])("a viewer's %s %j is 403 forbidden with no upstream call", async (method, segments) => {
    const res = await call(method, segments, { headers: JSON_HDR, ...(method === "DELETE" ? {} : { body: "{}" }) });
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ error: "forbidden" });
    expect(seen).toHaveLength(0);
  });

  it("the role check comes before the path, the table and the key: a viewer's bad-path or keyless write is 403", async () => {
    expect((await call("POST", ["runs", "..", "x"], { headers: JSON_HDR, body: "{}" })).status).toBe(403);
    expect((await call("POST", ["admin", "x"], { headers: JSON_HDR, body: "{}" })).status).toBe(403);
    delete process.env.API_KEY;
    expect((await call("POST", ["runs"], { headers: JSON_HDR, body: "{}" })).status).toBe(403);
    expect(seen).toHaveLength(0);
  });

  it("the session check still comes before the role check: no session is 401, not 403", async () => {
    getServerSession.mockResolvedValue(null);
    expect((await call("POST", ["runs"], { headers: JSON_HDR, body: "{}" })).status).toBe(401);
  });

  it("an admin's writes are unaffected", async () => {
    getServerSession.mockResolvedValue(SESSION);
    expect((await call("POST", ["runs"], { headers: JSON_HDR, body: "{}" })).status).toBe(200);
    expect((await call("DELETE", ["runs", UUID])).status).toBe(200);
    expect(seen.map((r) => r.method)).toEqual(["POST", "DELETE"]);
  });
});

describe("stop and emergency-stop exactly as the browser client builds them (RK-37B-01)", () => {
  // Capture what requestJson really sends, then replay it into the handler.
  async function browserRequest(run: () => Promise<unknown>): Promise<{ request: Request; segments: string[] }> {
    const real = global.fetch;
    let captured: [string, RequestInit] | undefined;
    global.fetch = (async (url: string, init: RequestInit) => {
      captured = [url, init];
      return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => ({}) };
    }) as unknown as typeof fetch;
    try {
      await run();
    } finally {
      global.fetch = real;
    }
    const [url, init] = captured!;
    const { signal: _signal, ...rest } = init;
    const request = new Request(`http://ui.test${url}`, { ...rest, headers: { ...(init.headers as object), "sec-fetch-site": "same-origin" } });
    const segments = new URL(request.url).pathname.replace("/bff/api/v1/", "").split("/");
    return { request, segments };
  }

  it("stopRun: DELETE runs/{id}?flatten=true reaches the upstream with no body", async () => {
    const { request, segments } = await browserRequest(() => api.stopRun(UUID, { flatten: true }));
    expect(request.headers.get("content-type")).toBe("application/json");
    const res = await route.DELETE(request, { params: { path: segments } });
    expect(res.status).toBe(200);
    expect(seen[0]).toMatchObject({ method: "DELETE", url: `/api/v1/runs/${UUID}?flatten=true`, body: "" });
  });

  it("emergencyStop: POST runs/{id}/emergency-stop?flatten=true carries X-Emergency-Reason and no body", async () => {
    const { request, segments } = await browserRequest(() => api.emergencyStop(UUID, { flatten: true, reason: "halt now" }));
    const res = await route.POST(request, { params: { path: segments } });
    expect(res.status).toBe(200);
    expect(seen[0]).toMatchObject({ method: "POST", url: `/api/v1/runs/${UUID}/emergency-stop?flatten=true`, body: "" });
    expect(seen[0]!.headers["x-emergency-reason"]).toBe("halt now");
  });
});

describe("response size cap (FI-9737-I)", () => {
  it("an upstream body above 64 MiB is 502 upstream_too_large, logged without body content", async () => {
    reply = (_req, res) => {
      res.writeHead(200, { "content-type": "application/octet-stream" });
      const chunk = Buffer.alloc(1024 * 1024, 0x41);
      for (let i = 0; i < 64; i += 1) res.write(chunk);
      res.end("B");
    };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "upstream_too_large" });
    const logged = JSON.stringify((console.warn as jest.Mock).mock.calls);
    expect(logged).toContain("bff.upstream_too_large");
    expect(logged).not.toContain("AAAA");
  }, 30_000);

  it("the upstream_too_large log object carries the route label only: no URL, path, id or query (CF-97-43)", async () => {
    (console.warn as jest.Mock).mockClear();
    reply = (_req, res) => {
      res.writeHead(200, { "content-type": "application/octet-stream" });
      const chunk = Buffer.alloc(1024 * 1024, 0x41);
      for (let i = 0; i < 64; i += 1) res.write(chunk);
      res.end("B");
    };
    const res = await call("GET", ["runs", UUID], { search: "?marker=queryvalue-7731" });
    expect(res.status).toBe(502);
    const calls = (console.warn as jest.Mock).mock.calls.filter((c) => c[0] === "bff.upstream_too_large");
    expect(calls).toHaveLength(1);
    const fields = calls[0].slice(1);
    expect(fields).toHaveLength(1);
    expect(Object.keys(fields[0])).toEqual(["route"]);
    expect(fields[0].route).toMatch(/^GET [A-Za-z{}/_-]+$/);
    const logged = JSON.stringify(calls);
    for (const leak of [UUID, "queryvalue-7731", "marker", "127.0.0.1", String(process.env.INTERNAL_API_URL), "/api/v1", "http"]) {
      expect(logged).not.toContain(leak);
    }
  }, 30_000);

  it("a body just under the cap still passes", async () => {
    reply = (_req, res) => {
      res.writeHead(200, { "content-type": "application/octet-stream" });
      res.end(Buffer.alloc(1024 * 1024, 0x41));
    };
    const res = await call("GET", ["runs"]);
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBe(1024 * 1024);
  });
});
