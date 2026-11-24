/** @jest-environment node */
/**
 * CF-97-37 / 37-B: defence-in-depth layers of the proxy that the validator
 * normally makes unreachable (pathname/origin assertion), plus the per-class
 * Agent and the timeout. `matchBff` is mocked so the assertion layer is tested
 * on its own; the real handler tests are in bff-handler.test.ts.
 */
import http from "http";
import type { AddressInfo } from "net";

const mockMatch = jest.fn();
jest.mock("@/lib/bff-routes", () => ({ matchBff: (...a: unknown[]) => mockMatch(...a) }));

const agentCtor = jest.fn();
jest.mock("undici", () => {
  const actual = jest.requireActual("undici");
  return {
    ...actual,
    Agent: class extends actual.Agent {
      constructor(opts: unknown) {
        super(opts);
        agentCtor(opts);
      }
    },
  };
});

import { proxyBff } from "@/lib/bff-proxy";

let server: http.Server;
let hits: string[] = [];
let delayMs = 0;

beforeAll(async () => {
  server = http.createServer((req, res) => {
    hits.push(req.url ?? "");
    setTimeout(() => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    }, delayMs);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  process.env.INTERNAL_API_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(() => new Promise((r) => server.close(r)));
beforeEach(() => {
  hits = [];
  delayMs = 0;
  process.env.API_KEY = "dummy-api-key-0123456789";
  jest.spyOn(console, "warn").mockImplementation(() => undefined);
});

const route = (budgetMs: number) => ({ ok: true, route: { method: "GET", parts: ["runs"], budgetMs } });
const req = () => new Request("http://ui.test/bff/api/v1/runs");

describe("pathname and origin assertions (T3, T6)", () => {
  it.each([
    ["dot-dot that the URL parser would normalise", ["x", "..", ".."]],
    ["a single dot segment", ["x", ".", "y"]],
  ])("%s: 400 and no upstream call", async (_label, segments) => {
    mockMatch.mockReturnValue(route(20_000));
    const res = await proxyBff(req(), segments);
    expect(res.status).toBe(400);
    expect(hits).toHaveLength(0);
  });

  it("an INTERNAL_API_URL with a path prefix cannot move the request outside /api/v1", async () => {
    process.env.INTERNAL_API_URL += "/prefix";
    mockMatch.mockReturnValue(route(20_000));
    expect((await proxyBff(req(), ["runs"])).status).toBe(200);
    expect(hits).toEqual(["/api/v1/runs"]);
  });
});

describe("Agents and timeouts (DR-9737-23)", () => {
  it("one module-scoped Agent per route class, with headers/body timeouts of budget + 5 s", async () => {
    mockMatch.mockReturnValue(route(35_000));
    await proxyBff(req(), ["runs"]);
    await proxyBff(req(), ["runs"]);
    mockMatch.mockReturnValue(route(305_000));
    await proxyBff(req(), ["runs"]);
    await proxyBff(req(), ["runs"]);
    // Four requests over two classes: two constructions (earlier tests built the 20 s class).
    const built = agentCtor.mock.calls.map(([o]) => o);
    expect(built.filter((o) => o.headersTimeout === 40_000)).toEqual([{ headersTimeout: 40_000, bodyTimeout: 40_000, maxResponseSize: 64 * 1024 * 1024 }]);
    expect(built.filter((o) => o.headersTimeout === 310_000)).toEqual([{ headersTimeout: 310_000, bodyTimeout: 310_000, maxResponseSize: 64 * 1024 * 1024 }]);
  });

  it("an upstream slower than the class budget is 504 upstream_timeout", async () => {
    delayMs = 400;
    mockMatch.mockReturnValue(route(60));
    const res = await proxyBff(req(), ["runs"]);
    expect(res.status).toBe(504);
    expect(await res.json()).toEqual({ error: "upstream_timeout" });
  });
});

describe("logs carry a fixed code and a route label, never the path, the key or a header", () => {
  it("bad_path logs only the segment count", async () => {
    mockMatch.mockReturnValue({ ok: false, status: 400 });
    await proxyBff(req(), ["runs", "SECRET-SEGMENT"]);
    const logged = JSON.stringify((console.warn as jest.Mock).mock.calls);
    expect(logged).toContain("bff.bad_path");
    expect(logged).not.toContain("SECRET-SEGMENT");
    expect(logged).not.toContain("dummy-api-key");
  });
});
