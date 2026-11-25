/**
 * CF-97-37 / 37-B (DR-9737-20, mutants P1/P2): the browser client builds
 * `/bff/api/v1/...` for EVERY exported call, and each of those calls is
 * admitted by the BFF's own default-deny table.
 */
import { randomUUID } from "crypto";
import * as api from "@/lib/api";
import { matchBff } from "@/lib/bff-routes";

const ID = "123e4567-e89b-42d3-a456-426614174000";
const OPTS = { idempotencyKey: randomUUID() } as const;

const CALLS: [string, () => Promise<unknown>][] = [
  ["fetchRuns", () => api.fetchRuns({ limit: 5, status: "stopped" })],
  ["fetchRun", () => api.fetchRun(ID)],
  ["createRun", () => api.createRun({} as never, OPTS)],
  ["promoteRun", () => api.promoteRun(ID, OPTS)],
  ["stopRun", () => api.stopRun(ID, { flatten: true })],
  ["emergencyStop", () => api.emergencyStop(ID, { reason: "halt" })],
  ["fetchKillSwitchStatus", () => api.fetchKillSwitchStatus()],
  ["archiveRun", () => api.archiveRun(ID)],
  ["fetchPortfolio", () => api.fetchPortfolio(ID)],
  ["fetchAggregatePortfolio", () => api.fetchAggregatePortfolio()],
  ["fetchEquityCurve", () => api.fetchEquityCurve(ID)],
  ["fetchTrades", () => api.fetchTrades(ID)],
  ["fetchOrders", () => api.fetchOrders(ID)],
  ["fetchFills", () => api.fetchFills(ID)],
  ["fetchPositions", () => api.fetchPositions(ID)],
  ["fetchLearningState", () => api.fetchLearningState(ID)],
  ["fetchDiagnostics", () => api.fetchDiagnostics(ID)],
  ["fetchStrategies", () => api.fetchStrategies()],
  ["fetchStrategySchema", () => api.fetchStrategySchema("ma_cross")],
  ["fetchModelVersions", () => api.fetchModelVersions("BTC/USDT", true)],
  ["retrainModel", () => api.retrainModel("BTC/USDT", "1h")],
  ["trainModel", () => api.trainModel("BTC/USDT", "1h")],
  ["activateModelVersion", () => api.activateModelVersion(ID)],
  ["runOptimization", () => api.runOptimization({} as never)],
  ["fetchOptimizationRuns", () => api.fetchOptimizationRuns()],
  ["fetchOptimizationRun", () => api.fetchOptimizationRun(ID)],
  ["fetchMarketSignals", () => api.fetchMarketSignals()],
];

describe("browser apiFetch prefix (DR-9737-20)", () => {
  const fetchMock = jest.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: async () => ({}),
    });
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  it("covers every exported API call of api.ts", () => {
    const exported = Object.keys(api).filter((n) => /^(fetch|create|promote|stop|emergency|archive|retrain|train|activate|run)[A-Z]/.test(n));
    const covered = new Set(CALLS.map(([n]) => n));
    expect(exported.filter((n) => !covered.has(n) && n !== "formatCurrency")).toEqual([]);
  });

  it.each(CALLS)("%s requests /bff/api/v1/... and the BFF table admits it", async (_name, run) => {
    await run();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url.startsWith("/bff/api/v1/")).toBe(true);
    expect(url).not.toMatch(/^https?:/);
    const pathname = url.slice("/bff/api/v1/".length).split("?")[0]!;
    // Next hands the handler each segment decoded once.
    const segments = pathname.split("/").map(decodeURIComponent);
    expect(matchBff(init.method ?? "GET", segments)).toMatchObject({ ok: true });
  });

  it("never sends a key header from the browser", async () => {
    await api.fetchRuns();
    const headers = (fetchMock.mock.calls[0]![1] as RequestInit).headers as Record<string, string>;
    expect(Object.keys(headers).map((h) => h.toLowerCase())).not.toContain("x-api-key");
  });
});

describe("the dashboard's message for a rejected server key (RK-37B-03)", () => {
  const respond = (status: number, body: unknown) =>
    ({ ok: false, status, headers: { get: () => "application/json" }, json: async () => body }) as unknown as Response;

  it("a 502 upstream_auth_failed points to the operator fallback, not 'unreachable'", async () => {
    global.fetch = jest.fn().mockResolvedValue(respond(502, { error: "upstream_auth_failed" })) as unknown as typeof fetch;
    const result = await api.fetchRuns();
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.status).toBe(502);
      expect(result.error.message).toMatch(/API key was rejected/);
      expect(result.error.message).toMatch(/operator fallback/);
      expect(result.error.message).not.toMatch(/unreachable/);
    }
  });

  it.each([
    [502, { error: "upstream_unreachable" }],
    [502, { error: "upstream_too_large" }],
    [502, "plain text"],
    [401, { error: "upstream_auth_failed" }],
  ])("%i %j keeps the generic text", async (status, body) => {
    global.fetch = jest.fn().mockResolvedValue(respond(status, body)) as unknown as typeof fetch;
    const result = await api.fetchRuns();
    if (!result.ok) expect(result.error.message).not.toMatch(/API key was rejected/);
    expect(result.ok).toBe(false);
  });
});

describe("the dashboard's message for a viewer's refused write (CF-97-43)", () => {
  const respond = (status: number, body: unknown) =>
    ({ ok: false, status, headers: { get: () => "application/json" }, json: async () => body }) as unknown as Response;
  const READ_ONLY = "Your account is read-only (viewer). Ask an admin, or use the operator fallback (runbook: dashboard cannot stop a run).";

  it("a 403 {error: forbidden} gets the read-only message, for a write call", async () => {
    global.fetch = jest.fn().mockResolvedValue(respond(403, { error: "forbidden" })) as unknown as typeof fetch;
    const result = await api.stopRun(ID, { flatten: true });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.status).toBe(403);
      expect(result.error.message).toBe(READ_ONLY);
    }
  });

  it.each([
    [403, { error: "csrf" }],
    [403, { detail: "Forbidden" }],
    [403, "plain text"],
    [403, null],
    [401, { error: "forbidden" }],
    [502, { error: "forbidden" }],
  ])("%i %j keeps the generic text", async (status, body) => {
    global.fetch = jest.fn().mockResolvedValue(respond(status, body)) as unknown as typeof fetch;
    const result = await api.fetchRuns();
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.status).toBe(status);
      expect(result.error.message).not.toBe(READ_ONLY);
      expect(result.error.message).not.toMatch(/read-only/);
    }
  });
});
