/** @jest-environment node */
/**
 * CF-97-37 / 37-B (DR-9737-20, -24, -27): the server-component client builds
 * INTERNAL_API_URL + /api/v1/... with the key and `no-store`, and re-checks the
 * session (which runs the jwt allow-list re-check) BEFORE any fetch.
 */
const getServerSession = jest.fn();
jest.mock("next-auth", () => ({ getServerSession: (...a: unknown[]) => getServerSession(...a) }));
jest.mock("@/lib/auth-options", () => ({ authOptions: {} }));
const redirect = jest.fn((to: string) => {
  throw new Error(`NEXT_REDIRECT:${to}`);
});
jest.mock("next/navigation", () => ({ redirect: (to: string) => redirect(to) }));

import * as server from "@/lib/api-server";

const KEY = "dummy-api-key-0123456789";
const fetchMock = jest.fn();

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(new Response("{}", { status: 200, headers: { "content-type": "application/json" } }));
  global.fetch = fetchMock as unknown as typeof fetch;
  getServerSession.mockReset();
  getServerSession.mockResolvedValue({ user: { email: "admin@example.com" } });
  redirect.mockClear();
  process.env.API_KEY = KEY;
  process.env.INTERNAL_API_URL = "http://api.internal:8000/";
});

const CALLS: [string, () => Promise<unknown>, string][] = [
  ["fetchHealth", () => server.fetchHealth(), "http://api.internal:8000/health"],
  ["fetchRuns", () => server.fetchRuns({ limit: 25 }), "http://api.internal:8000/api/v1/runs?limit=25"],
  ["fetchAggregatePortfolio", () => server.fetchAggregatePortfolio(), "http://api.internal:8000/api/v1/portfolio/summary"],
  ["fetchStrategies", () => server.fetchStrategies(), "http://api.internal:8000/api/v1/strategies"],
];

describe("api-server.ts", () => {
  it.each(CALLS)("%s: INTERNAL_API_URL + path, X-API-Key, cache no-store", async (_n, run, expected) => {
    await run();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(expected);
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe(KEY);
    expect(init.cache).toBe("no-store");
  });

  it.each(CALLS)("%s: no session means no fetch and a redirect to sign-in", async (_n, run) => {
    getServerSession.mockResolvedValue(null);
    await expect(run()).rejects.toThrow("NEXT_REDIRECT:/api/auth/signin");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("the session is read before the fetch, every time", async () => {
    await server.fetchStrategies();
    expect(getServerSession.mock.invocationCallOrder[0]).toBeLessThan(fetchMock.mock.invocationCallOrder[0]!);
  });

  it("an unset API_KEY sends no key header (auth off) instead of an empty one", async () => {
    delete process.env.API_KEY;
    await server.fetchStrategies();
    expect((fetchMock.mock.calls[0]![1] as RequestInit).headers).not.toHaveProperty("X-API-Key");
    expect(server.apiKeyHeader()).toBeNull();
  });

  it("retryAfterHeaders passes only Retry-After", () => {
    expect(server.retryAfterHeaders(new Response("", { headers: { "retry-after": "9", "x-other": "1" } }))).toEqual({ "Retry-After": "9" });
    expect(server.retryAfterHeaders(new Response(""))).toEqual({});
  });
});
