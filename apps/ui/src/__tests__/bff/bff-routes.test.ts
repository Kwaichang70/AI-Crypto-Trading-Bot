/**
 * CF-97-37 / 37-B: the BFF default-deny table and the segment validator
 * (DR-9737-16, DR-9737-26), checked against 37-A's route fixture.
 */
import fs from "fs";
import path from "path";
import { BFF_TABLE, matchBff } from "@/lib/bff-routes";

const FIXTURE = path.resolve(__dirname, "../../../../../tests/fixtures/api_routes_v1.json");
const API_ROUTES: { method: string; path: string; dependency: string }[] = JSON.parse(
  fs.readFileSync(FIXTURE, "utf8"),
);

const UUID = "123e4567-e89b-42d3-a456-426614174000";

// API routes the browser must never reach through the BFF. Every fixture route
// is either in BFF_TABLE or here; a new API route fails this test until it is
// classified (DR-9737-16).
const NOT_FOR_BROWSER = new Set([
  "GET /api/v1/health/background",
  "GET /api/v1/metrics",
  "POST /api/v1/emergency/kill-switch",
  "POST /api/v1/emergency/kill-switch/clear",
  "GET /api/v1/runs/{id}/circuit-breaker",
  "POST /api/v1/runs/{id}/circuit-breaker/reset",
  "POST /api/v1/runs/{id}/entries-latch/clear",
  "POST /api/v1/runs/{id}/resume",
  "GET /api/v1/runs/{id}/orders/{id}",
  "GET /api/v1/runs/{id}/promotion-eligibility",
]);

// Fixture path -> table pattern: {symbol:path} -> {symbol}, {name} stays, every
// other parameter in the API is a UUID.
const normalise = (apiPath: string) =>
  apiPath
    .replace("/api/v1/", "")
    .replace("{symbol:path}", "{symbol}")
    .replace(/\{(?!symbol\}|name\})[a-z_]+\}/g, "{id}");

const tableKeys = BFF_TABLE.map((r) => `${r.method} ${r.parts.join("/")}`);

describe("BFF table against tests/fixtures/api_routes_v1.json", () => {
  it("every API route is either in the table or explicitly not for the browser", () => {
    const unclassified = API_ROUTES.map((r) => `${r.method} ${normalise(r.path)}`).filter(
      (key) => !tableKeys.includes(key) && !NOT_FOR_BROWSER.has(`${key.split(" ")[0]} /api/v1/${key.split(" ")[1]}`),
    );
    expect(unclassified).toEqual([]);
  });

  it("every table entry exists in the fixture (no stale entries)", () => {
    const apiKeys = API_ROUTES.map((r) => `${r.method} ${normalise(r.path)}`);
    expect(tableKeys.filter((key) => !apiKeys.includes(key))).toEqual([]);
  });

  it("no table entry is an admin route, the metrics route or a not-for-browser route", () => {
    const admin = API_ROUTES.filter((r) => r.dependency.includes("admin") || r.path.includes("metrics"));
    for (const r of admin) expect(tableKeys).not.toContain(`${r.method} ${normalise(r.path)}`);
    for (const key of NOT_FOR_BROWSER) {
      const [method, p] = key.split(" ");
      expect(tableKeys).not.toContain(`${method} ${p!.replace("/api/v1/", "")}`);
    }
  });

  it("every {x:path} parameter in the fixture has a declared grammar (symbol)", () => {
    const pathParams = API_ROUTES.flatMap((r) => r.path.match(/\{[a-z_]+:path\}/g) ?? []);
    expect([...new Set(pathParams)]).toEqual(["{symbol:path}"]);
    expect(matchBff("POST", ["ml", "retrain", "BTC/USDT"]).ok).toBe(true);
  });

  it("stop and emergency-stop are named with a budget of at least 70 s (DR-9737-19)", () => {
    const budget = (m: string, p: string) => BFF_TABLE.find((r) => r.method === m && r.parts.join("/") === p)?.budgetMs;
    expect(budget("DELETE", "runs/{id}")).toBeGreaterThanOrEqual(70_000);
    expect(budget("POST", "runs/{id}/emergency-stop")).toBeGreaterThanOrEqual(70_000);
    expect(budget("POST", "optimize")).toBeGreaterThanOrEqual(305_000);
  });

  it("only GET is allowed on emergency/*", () => {
    expect(tableKeys.filter((k) => k.includes("emergency/"))).toEqual(["GET emergency/kill-switch"]);
  });
});

describe("matchBff: positive grammar, then the table", () => {
  it.each([
    ["GET", ["runs"]],
    ["GET", ["runs", UUID]],
    ["GET", ["runs", UUID.toUpperCase(), "trades"]],
    ["DELETE", ["runs", UUID]],
    ["POST", ["runs", UUID, "emergency-stop"]],
    ["GET", ["strategies", "ma_cross-1", "schema"]],
    ["GET", ["emergency", "kill-switch"]],
    ["POST", ["ml", "retrain", "BTC/USDT"]],
  ])("admits %s %j", (method, segments) => {
    expect(matchBff(method, segments).ok).toBe(true);
  });

  it.each([
    ["..", ["runs", "..", "emergency", "kill-switch"]],
    [".", ["runs", ".", UUID]],
    ["empty segment", ["runs", "", UUID]],
    ["no segments", []],
    ["percent left after the decode (double encoding)", ["runs", "%2e%2e"]],
    ["double-encoded dots", ["runs", "%252e%252e"]],
    ["a double-encoded symbol", ["ml", "retrain", "BTC%2FUSDT"]],
    ["backslash", ["runs", "a\\b"]],
    ["control character", ["runs", "a\u0000b"]],
    ["non-ASCII dot", ["runs", "．．"]],
    ["semicolon", ["runs;x"]],
    ["space", ["runs", "a b"]],
    ["plus", ["runs", "a+b"]],
    ["a slash outside {symbol}", ["runs", "a/b"]],
    ["too many symbol parts", ["ml", "retrain", "BTC/USDT/x"]],
    ["dot-dot inside the symbol", ["ml", "retrain", "../USDT"]],
    ["a lower-case symbol (fails the symbol grammar)", ["ml", "retrain", "btc/usdt"]],
  ])("rejects %s with 400", (_label, segments) => {
    expect(matchBff("GET", segments)).toEqual({ ok: false, status: 400 });
  });

  it.each([
    ["a trailing dot", "GET", ["emergency", "kill-switch."]],
    ["a non-UUID {id}", "GET", ["runs", "not-a-uuid"]],
    ["an unknown path", "GET", ["admin", "x"]],
    ["a known path with the wrong method", "POST", ["runs", UUID, "trades"]],
    ["admin resume", "POST", ["runs", UUID, "resume"]],
    ["admin latch clear", "POST", ["runs", UUID, "entries-latch", "clear"]],
    ["kill-switch press", "POST", ["emergency", "kill-switch"]],
    ["metrics", "GET", ["metrics"]],
  ])("answers %s with 404", (_label, method, segments) => {
    expect(matchBff(method, segments)).toEqual({ ok: false, status: 404 });
  });
});

describe("the {symbol} grammar on its real method (POST ml/retrain/{symbol})", () => {
  it.each([
    ["BTC/USDT"],
    ["ETH/USD"],
    ["AB/CD"],
    ["ABCDEFGHIJKL/MNOPQRSTUVWX"],
  ])("admits %s", (symbol) => {
    expect(matchBff("POST", ["ml", "retrain", symbol]).ok).toBe(true);
  });

  it.each([
    ["lower case", "btc/usdt"],
    ["mixed case", "Btc/USDT"],
    ["a one-character first side", "B/USDT"],
    ["a one-character second side", "BTC/U"],
    ["a thirteen-character side", "ABCDEFGHIJKLM/USDT"],
    ["a dot inside", "BT.C/USDT"],
    ["a hyphen inside", "BT-C/USDT"],
    ["no slash", "BTCUSDT"],
    ["an empty side", "BTC/"],
    ["a third part", "BTC/USDT/EUR"],
    ["an encoded slash left in", "BTC%2FUSDT"],
    ["a dot-dot side", "../USDT"],
  ])("refuses %s with no table match", (_label, symbol) => {
    const result = matchBff("POST", ["ml", "retrain", symbol]);
    expect(result.ok).toBe(false);
  });

  it("refuses a symbol where a name is expected, and a name where a symbol is expected", () => {
    expect(matchBff("GET", ["strategies", "BTC/USDT", "schema"]).ok).toBe(false);
    expect(matchBff("POST", ["ml", "retrain", "ma_cross"]).ok).toBe(false);
  });
});
