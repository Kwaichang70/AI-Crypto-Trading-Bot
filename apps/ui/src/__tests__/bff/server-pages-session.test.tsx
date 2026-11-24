/**
 * CF-97-37 / 37-B (DR-9737-27, SE-37S-07): the two server pages (`/` and
 * `/strategies`) render API data only after `getServerSession`, which runs the
 * jwt allow-list re-check. A token for an address on neither list renders no
 * API data and makes no upstream call.
 */
const mockGetServerSession = jest.fn();
jest.mock("next-auth", () => ({ getServerSession: (...a: unknown[]) => mockGetServerSession(...a) }));
const mockRedirect = jest.fn((to: string) => {
  throw new Error(`NEXT_REDIRECT:${to}`);
});
jest.mock("next/navigation", () => ({
  redirect: (to: string) => mockRedirect(to),
  usePathname: () => "/",
}));
jest.mock("next/link", () => ({ __esModule: true, default: ({ children }: { children: unknown }) => children }));

import { authOptions } from "@/lib/auth-options";
import DashboardPage from "@/app/page";
import StrategiesPage from "@/app/strategies/page";

const fetchMock = jest.fn();
const okJson = (body: unknown) => ({
  ok: true,
  status: 200,
  headers: { get: () => "application/json" },
  json: async () => body,
});

/** getServerSession as next-auth behaves: the jwt callback runs; a throw gives null. */
function sessionFor(email: string, role: string) {
  mockGetServerSession.mockImplementation(async () => {
    try {
      await authOptions.callbacks!.jwt!({ token: { email, role } } as never);
      return { user: { email, role } };
    } catch {
      return null;
    }
  });
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (url: string) =>
    okJson(String(url).includes("strategies") ? { strategies: [], total: 0 } : { items: [], totalRuns: 0, status: "ok" }),
  );
  global.fetch = fetchMock as unknown as typeof fetch;
  mockRedirect.mockClear();
  process.env.ADMIN_EMAILS = "admin@example.com";
  process.env.VIEWER_EMAILS = "viewer@example.com";
  process.env.API_KEY = "dummy-api-key-0123456789";
  process.env.INTERNAL_API_URL = "http://api.internal:8000";
});

describe("a revoked token on the server pages", () => {
  it("/ : an address on neither list renders no API data, makes no upstream call, and goes to sign-in", async () => {
    sessionFor("removed@example.com", "viewer");
    await expect(DashboardPage()).rejects.toThrow("NEXT_REDIRECT:/api/auth/signin");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("/strategies: the same", async () => {
    sessionFor("removed@example.com", "viewer");
    await expect(StrategiesPage()).rejects.toThrow("NEXT_REDIRECT:/api/auth/signin");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("/ : an admin who was removed from both lists after signing in is refused the same way", async () => {
    process.env.ADMIN_EMAILS = "someone-else@example.com";
    sessionFor("admin@example.com", "admin");
    await expect(DashboardPage()).rejects.toThrow("NEXT_REDIRECT");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("a listed token on the server pages", () => {
  it("/ : fetches health, runs and the aggregate portfolio from INTERNAL_API_URL with the key", async () => {
    sessionFor("viewer@example.com", "viewer");
    await DashboardPage();
    const urls = fetchMock.mock.calls.map(([u]) => u as string).sort();
    expect(urls).toEqual([
      "http://api.internal:8000/api/v1/portfolio/summary",
      "http://api.internal:8000/api/v1/runs?limit=25",
      "http://api.internal:8000/health",
    ]);
    for (const [, init] of fetchMock.mock.calls) {
      expect((init as RequestInit).headers).toMatchObject({ "X-API-Key": "dummy-api-key-0123456789" });
      expect((init as RequestInit).cache).toBe("no-store");
    }
  });

  it("/strategies: fetches the strategy list", async () => {
    sessionFor("admin@example.com", "admin");
    await StrategiesPage();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
