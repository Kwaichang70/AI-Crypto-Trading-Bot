/**
 * @jest-environment node
 *
 * apps/ui/src/__tests__/api/admin-proxy-routes.test.ts
 * ---------------------------------------------------------
 * WP1.7b (spec §6, round 2 S-03/S-04/S-05): tests every /api/admin/* proxy
 * route handler for:
 *   - 401 (no session) / 403 (non-admin session) / 503 (admin key not
 *     configured on the server)
 *   - injecting X-Admin-Key into the upstream request
 *   - the admin key NEVER appearing anywhere in the response returned to
 *     the browser
 *   - forwarding X-Live-Confirm-Token UNCHANGED (resume, entries-latch clear)
 *   - WP17b-S-03: `[id]` must be a plain UUID (400 otherwise, checked
 *     before the admin key is ever read)
 *   - WP17b-S-04: a non-string `reason` must not crash the route (no
 *     unhandled 500)
 *   - WP17b-S-05: same-origin + `Content-Type: application/json` are
 *     enforced on every state-changing route (415/403 otherwise)
 *   - WP17b-C-R2-01 (round 3): the Origin fallback compares against the
 *     PUBLIC host (`NEXTAUTH_URL`, then `X-Forwarded-Host`, then `Host`),
 *     never against `request.url` (always the server's own internal bind
 *     address behind Caddy/standalone-server topology)
 *   - WP17b-S-R2-03 (round 3): the Content-Type check compares only the
 *     media type (`text/plain; x=application/json` must still 415)
 *
 * `@jest-environment node` (not the file-level default jsdom) so the
 * platform's native `Request`/`Response`/`fetch` globals (Node 18+) are
 * used untouched by jsdom's own (incomplete) Fetch API shims.
 */

import { getServerSession } from "next-auth";
import { isAdmin } from "@/lib/auth";

jest.mock("next-auth", () => ({ getServerSession: jest.fn() }));
jest.mock("@/lib/auth-options", () => ({ authOptions: {} }));
jest.mock("@/lib/auth", () => ({ isAdmin: jest.fn() }));

const mockGetServerSession = getServerSession as jest.Mock;
const mockIsAdmin = isAdmin as jest.Mock;

const ADMIN_KEY = "test-internal-admin-key-value";
const VALID_ID = "11111111-1111-1111-1111-111111111111";

const ORIGINAL_ENV = process.env;

beforeEach(() => {
  // Deliberately NOT calling jest.resetModules() here -- these route
  // modules are re-imported (dynamic `import()`) inside every test via
  // `importRoute()`; resetting the module registry would force each of
  // those imports to re-run the `jest.mock(...)` factories above, handing
  // the freshly re-imported route module a DIFFERENT mock instance than
  // the `mockGetServerSession`/`mockIsAdmin` references captured at this
  // file's top level -- silently breaking every `.mockResolvedValue(...)`
  // configured below (each route would then see a bare, unconfigured
  // jest.fn() and 401 unconditionally).
  jest.clearAllMocks();
  process.env = { ...ORIGINAL_ENV, INTERNAL_ADMIN_API_KEY: ADMIN_KEY, API_KEY: "test-api-key" };
  mockGetServerSession.mockResolvedValue({ user: { role: "admin", email: "a@b.com" } });
  mockIsAdmin.mockReturnValue(true);
});

afterAll(() => {
  process.env = ORIGINAL_ENV;
});

function jsonResponseBodyContainsKey(body: string): boolean {
  return body.includes(ADMIN_KEY);
}

function okUpstream(body: unknown = { ok: true }): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Shared auth-gate behaviour, parameterised over every route module.
// ---------------------------------------------------------------------------

interface RouteFixture {
  name: string;
  hasId: boolean;
  hasReason: boolean;
  importRoute: () => Promise<{
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    POST: (...args: any[]) => Promise<Response>;
  }>;
  url: (id?: string) => string;
  /** Body/extra-header defaults for a VALID request; overridable per-test. */
  makeRequest: (opts?: {
    id?: string;
    headers?: Record<string, string | null>;
    reason?: unknown;
  }) => Request;
  ctx: (id?: string) => unknown;
}

const ROUTES: RouteFixture[] = [
  {
    name: "kill-switch press",
    hasId: false,
    hasReason: true,
    importRoute: () => import("@/app/api/admin/kill-switch/route"),
    url: () => "http://localhost/api/admin/kill-switch",
    makeRequest: ({ headers = {}, reason = "test" } = {}) =>
      buildRequest("http://localhost/api/admin/kill-switch", { reason }, headers),
    ctx: () => undefined,
  },
  {
    name: "kill-switch clear",
    hasId: false,
    hasReason: true,
    importRoute: () => import("@/app/api/admin/kill-switch/clear/route"),
    url: () => "http://localhost/api/admin/kill-switch/clear",
    makeRequest: ({ headers = {}, reason = "incident resolved" } = {}) =>
      buildRequest("http://localhost/api/admin/kill-switch/clear", { reason }, headers),
    ctx: () => undefined,
  },
  {
    name: "run resume",
    hasId: true,
    hasReason: false,
    importRoute: () => import("@/app/api/admin/runs/[id]/resume/route"),
    url: (id = VALID_ID) => `http://localhost/api/admin/runs/${id}/resume?mode=protective`,
    makeRequest: ({ id = VALID_ID, headers = {} } = {}) =>
      buildRequest(
        `http://localhost/api/admin/runs/${id}/resume?mode=protective`,
        undefined,
        { "X-Live-Confirm-Token": "typed-token-abc", ...headers },
      ),
    ctx: (id = VALID_ID) => ({ params: { id } }),
  },
  {
    name: "entries-latch clear",
    hasId: true,
    hasReason: true,
    importRoute: () => import("@/app/api/admin/runs/[id]/entries-latch/clear/route"),
    url: (id = VALID_ID) => `http://localhost/api/admin/runs/${id}/entries-latch/clear`,
    makeRequest: ({ id = VALID_ID, headers = {}, reason = "manual recovery" } = {}) =>
      buildRequest(
        `http://localhost/api/admin/runs/${id}/entries-latch/clear`,
        { reason },
        { "X-Live-Confirm-Token": "typed-token-abc", ...headers },
      ),
    ctx: (id = VALID_ID) => ({ params: { id } }),
  },
];

/**
 * Builds a same-origin, `Content-Type: application/json` POST `Request` by
 * default (the shape every legitimate browser call sends) — individual
 * tests override/omit headers via `headers` to probe WP17b-S-05, or pass
 * `null` for a header value to OMIT it entirely (Node's `Headers` drops
 * `undefined`-valued entries at construction, so `null` is filtered below).
 */
function buildRequest(
  url: string,
  body: unknown,
  headers: Record<string, string | null> = {},
): Request {
  const merged: Record<string, string> = { "Content-Type": "application/json" };
  for (const [k, v] of Object.entries(headers)) {
    if (v === null) delete merged[k];
    else merged[k] = v;
  }
  return new Request(url, {
    method: "POST",
    headers: merged,
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
}

/**
 * Rebuilds `req` with a different URL (same method/headers/body) --
 * `Request.url` is read-only, so this is the only way to test "what if
 * `request.url`'s host differs from the deployment's real public host"
 * (WP17b-C-R2-01) without changing anything else about the request.
 */
async function withUrl(req: Request, newUrl: string): Promise<Request> {
  const bodyText = await req.clone().text();
  return new Request(newUrl, {
    method: req.method,
    headers: req.headers,
    ...(bodyText ? { body: bodyText } : {}),
  });
}

/** Swaps only the scheme+host of a fixture URL, e.g. for the standalone
 * server's internal bind address (`0.0.0.0:3000`) vs. the real public host. */
function atHost(url: string, hostAndScheme: string): string {
  return url.replace(/^https?:\/\/[^/]+/, hostAndScheme);
}

describe.each(ROUTES)("admin proxy route: $name", ({ importRoute, makeRequest, ctx }) => {
  it("returns 401 when there is no session", async () => {
    mockGetServerSession.mockResolvedValue(null);
    const { POST } = await importRoute();
    const res = await POST(makeRequest(), ctx() as never);
    expect(res.status).toBe(401);
  });

  it("returns 403 when the session is not admin", async () => {
    mockIsAdmin.mockReturnValue(false);
    const { POST } = await importRoute();
    const res = await POST(makeRequest(), ctx() as never);
    expect(res.status).toBe(403);
  });

  it("returns 503 when INTERNAL_ADMIN_API_KEY is not configured", async () => {
    delete process.env.INTERNAL_ADMIN_API_KEY;
    const { POST } = await importRoute();
    const res = await POST(makeRequest(), ctx() as never);
    expect(res.status).toBe(503);
  });

  it("injects X-Admin-Key upstream and never leaks it back to the browser", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(makeRequest(), ctx() as never);

    // The upstream call carried the admin key...
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Admin-Key"]).toBe(ADMIN_KEY);

    // ...but the key is never present anywhere in what the browser gets back.
    const bodyText = await res.clone().text();
    expect(jsonResponseBodyContainsKey(bodyText)).toBe(false);
    expect(res.headers.get("X-Admin-Key")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// X-Live-Confirm-Token forwarding — resume + entries-latch clear only.
// ---------------------------------------------------------------------------

describe.each(ROUTES.filter((r) => r.name === "run resume" || r.name === "entries-latch clear"))(
  "$name forwards X-Live-Confirm-Token unchanged",
  ({ importRoute, makeRequest, ctx }) => {
    it("forwards the exact header value upstream", async () => {
      const fetchMock = jest.fn().mockResolvedValue(okUpstream());
      global.fetch = fetchMock as unknown as typeof fetch;

      const { POST } = await importRoute();
      await POST(makeRequest(), ctx() as never);

      const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      const headers = init.headers as Record<string, string>;
      expect(headers["X-Live-Confirm-Token"]).toBe("typed-token-abc");
    });
  },
);

// ---------------------------------------------------------------------------
// WP17b-S-03: [id] must be a plain UUID, checked BEFORE the admin key read.
// ---------------------------------------------------------------------------

describe.each(ROUTES.filter((r) => r.hasId))("$name — WP17b-S-03 [id] validation", ({ importRoute, makeRequest, ctx }) => {
  it.each([".", "..", "not-a-uuid", "11111111-1111-1111-1111", ""])(
    "rejects id=%p with 400 before ever calling upstream or reading the admin key",
    async (badId) => {
      // Even with NO admin key configured and NO session, an invalid id
      // must still short-circuit to 400 -- proving the check runs first.
      delete process.env.INTERNAL_ADMIN_API_KEY;
      mockGetServerSession.mockResolvedValue(null);
      const fetchMock = jest.fn();
      global.fetch = fetchMock as unknown as typeof fetch;

      const { POST } = await importRoute();
      const res = await POST(makeRequest({ id: badId }), ctx(badId) as never);

      expect(res.status).toBe(400);
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("accepts a well-formed UUID", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(makeRequest({ id: VALID_ID }), ctx(VALID_ID) as never);

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// WP17b-S-04: a non-string `reason` must not crash the route (no raw 500).
// ---------------------------------------------------------------------------

describe.each(ROUTES.filter((r) => r.hasReason))("$name — WP17b-S-04 non-string reason", ({ importRoute, makeRequest, ctx }) => {
  it("handles a numeric reason without throwing an unhandled 500", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(makeRequest({ reason: 12345 }), ctx() as never);

    // A non-string reason is simply treated as absent/too-short -- 400 (too
    // short) is the expected, handled outcome; the only wrong answer is an
    // unhandled exception surfacing as Next's generic 500.
    expect(res.status).not.toBe(500);
  });
});

// ---------------------------------------------------------------------------
// WP17b-S-05: same-origin + Content-Type: application/json enforcement.
// ---------------------------------------------------------------------------

describe.each(ROUTES)("$name — WP17b-S-05 origin/content-type guard", ({ importRoute, makeRequest, ctx }) => {
  it("rejects a missing Content-Type with 415", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(makeRequest({ headers: { "Content-Type": null } }), ctx() as never);

    expect(res.status).toBe(415);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a text/plain Content-Type with 415 (a CSRF 'simple request')", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { "Content-Type": "text/plain" } }),
      ctx() as never,
    );

    expect(res.status).toBe(415);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a cross-site Sec-Fetch-Site with 403", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { "Sec-Fetch-Site": "cross-site" } }),
      ctx() as never,
    );

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a cross-origin Origin header (no Sec-Fetch-Site) with 403", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { Origin: "https://evil.example.com" } }),
      ctx() as never,
    );

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allows a same-origin Sec-Fetch-Site request through", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { "Sec-Fetch-Site": "same-origin" } }),
      ctx() as never,
    );

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("allows a request with neither Origin nor Sec-Fetch-Site through (no cross-origin signal to reject on)", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(makeRequest(), ctx() as never);

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("WP17b-S-R2-03: a crafted 'text/plain; x=application/json' Content-Type still 415s (media type only, params ignored)", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { "Content-Type": "text/plain; x=application/json" } }),
      ctx() as never,
    );

    expect(res.status).toBe(415);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("still accepts 'application/json; charset=utf-8' (a real, legitimate parameterised media type)", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await importRoute();
    const res = await POST(
      makeRequest({ headers: { "Content-Type": "application/json; charset=utf-8" } }),
      ctx() as never,
    );

    expect(res.status).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// WP17b-C-R2-01 (round 3): the Origin fallback must compare against the
// deployment's PUBLIC host (NEXTAUTH_URL, then X-Forwarded-Host, then
// Host) -- NEVER against `request.url`, which is always the standalone
// Next.js server's own internal bind address (e.g. `0.0.0.0:3000`) behind
// Caddy, and can never equal a real browser Origin in production.
// ---------------------------------------------------------------------------

describe.each(ROUTES)("$name — WP17b-C-R2-01 Origin fallback uses the public host, not request.url", ({ importRoute, makeRequest, ctx, url }) => {
  const BOUND_URL = atHost(url(VALID_ID), "http://0.0.0.0:3000");
  const PUBLIC_ORIGIN = "https://bot.example.ts.net";

  it("passes when Origin matches NEXTAUTH_URL, even though request.url's host is the internal 0.0.0.0:3000 bind address", async () => {
    process.env.NEXTAUTH_URL = PUBLIC_ORIGIN;
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const req = await withUrl(
      makeRequest({ id: VALID_ID, headers: { Origin: PUBLIC_ORIGIN } }),
      BOUND_URL,
    );
    const { POST } = await importRoute();
    const res = await POST(req, ctx(VALID_ID) as never);

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects with 403 when Origin does NOT match NEXTAUTH_URL (same bound-0.0.0.0:3000 request.url either way)", async () => {
    process.env.NEXTAUTH_URL = PUBLIC_ORIGIN;
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const req = await withUrl(
      makeRequest({ id: VALID_ID, headers: { Origin: "https://not-the-bot.example.com" } }),
      BOUND_URL,
    );
    const { POST } = await importRoute();
    const res = await POST(req, ctx(VALID_ID) as never);

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("falls back to the Host header when NEXTAUTH_URL is unset, and passes when Origin matches it", async () => {
    delete process.env.NEXTAUTH_URL;
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const req = await withUrl(
      makeRequest({
        id: VALID_ID,
        headers: { Origin: PUBLIC_ORIGIN, Host: "bot.example.ts.net" },
      }),
      BOUND_URL,
    );
    const { POST } = await importRoute();
    const res = await POST(req, ctx(VALID_ID) as never);

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("falls back to X-Forwarded-Host (preferred over Host) when NEXTAUTH_URL is unset", async () => {
    delete process.env.NEXTAUTH_URL;
    const fetchMock = jest.fn().mockResolvedValue(okUpstream());
    global.fetch = fetchMock as unknown as typeof fetch;

    const req = await withUrl(
      makeRequest({
        id: VALID_ID,
        headers: {
          Origin: PUBLIC_ORIGIN,
          "X-Forwarded-Host": "bot.example.ts.net",
          Host: "ui:3000",
        },
      }),
      BOUND_URL,
    );
    const { POST } = await importRoute();
    const res = await POST(req, ctx(VALID_ID) as never);

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects with 403 when NEXTAUTH_URL, X-Forwarded-Host and Host are all unset (fail closed, no expected host to compare against)", async () => {
    delete process.env.NEXTAUTH_URL;
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const req = await withUrl(
      makeRequest({ id: VALID_ID, headers: { Origin: PUBLIC_ORIGIN, Host: null } }),
      BOUND_URL,
    );
    const { POST } = await importRoute();
    const res = await POST(req, ctx(VALID_ID) as never);

    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// entries-latch clear: CF-B2 ships in this same WP (WP17b-C-02) — only
// camelCase pass-through remains; the old either-shape normalisation (and
// its "pre-CF-B2 snake_case" test) has been REMOVED as dead code now that
// api+infra+ui merge together, not as independent parallel agents.
// ---------------------------------------------------------------------------

describe("entries-latch clear route: response pass-through", () => {
  it("passes the camelCase backend response through unchanged", async () => {
    const fetchMock = jest.fn().mockResolvedValue(
      okUpstream({ runId: VALID_ID, cleared: "flatten_incomplete", stillLatchedBy: ["global_kill_switch"] }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("@/app/api/admin/runs/[id]/entries-latch/clear/route");
    const res = await POST(
      buildRequest(`http://localhost/api/admin/runs/${VALID_ID}/entries-latch/clear`, { reason: "manual recovery" }, {
        "X-Live-Confirm-Token": "t",
      }),
      { params: { id: VALID_ID } } as never,
    );
    const data = (await res.json()) as { runId: string; cleared: string; stillLatchedBy: string[] };
    expect(data).toEqual({
      runId: VALID_ID,
      cleared: "flatten_incomplete",
      stillLatchedBy: ["global_kill_switch"],
    });
  });

  it("rejects a reason shorter than 3 characters before ever calling upstream", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("@/app/api/admin/runs/[id]/entries-latch/clear/route");
    const res = await POST(
      buildRequest(`http://localhost/api/admin/runs/${VALID_ID}/entries-latch/clear`, { reason: "ab" }, {
        "X-Live-Confirm-Token": "t",
      }),
      { params: { id: VALID_ID } } as never,
    );
    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// kill-switch clear: reason length validation
// ---------------------------------------------------------------------------

describe("kill-switch clear route: reason validation", () => {
  it("rejects a reason shorter than 3 characters before ever calling upstream", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("@/app/api/admin/kill-switch/clear/route");
    const res = await POST(buildRequest("http://localhost/api/admin/kill-switch/clear", { reason: "x" }));
    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
