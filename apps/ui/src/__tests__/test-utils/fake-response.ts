/**
 * apps/ui/src/__tests__/test-utils/fake-response.ts
 * -----------------------------------------------------
 * Minimal `fetch` Response fake for jsdom-environment component tests.
 *
 * jsdom (the default `testEnvironment` for this project's jest config) does
 * NOT provide the Fetch API globals (`Response`/`Request`/`fetch`) that a
 * plain Node process has natively (Node 18+) — constructing `new
 * Response(...)` inside a jsdom test throws `ReferenceError: Response is
 * not defined`. Route-handler tests avoid this with an explicit
 * `@jest-environment node` docblock (native Response works there), but
 * ordinary component tests need something that works under the project's
 * default jsdom environment — hence this tiny fake covering only the
 * subset of the Response interface `apiFetch`/`adminFetch`
 * (apps/ui/src/lib/api.ts, apps/ui/src/lib/admin-fetch.ts) actually use.
 */

export function fakeJsonResponse(body: unknown, status = 200): Response {
  const text = JSON.stringify(body);
  const fake = {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-type" ? "application/json" : null,
    },
    json: async () => JSON.parse(text) as unknown,
    text: async () => text,
    clone(): Response {
      return fake as unknown as Response;
    },
  };
  return fake as unknown as Response;
}

/**
 * WP17b-S-R2-02 (round 3): a raw, non-JSON error body (e.g. a bare-text 500
 * "Internal Server Error", or any `text/plain` response) — `.json()`
 * rejects exactly like a real `Response` would on invalid JSON, so
 * `apiFetch`'s own `try { await response.json() } catch { await
 * response.text() }` fallback is exercised the same way it would be
 * against a real non-JSON body.
 */
export function fakeTextResponse(body: string, status = 500): Response {
  const fake = {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-type" ? "text/plain" : null,
    },
    json: async () => {
      throw new SyntaxError("Unexpected token I in JSON at position 0");
    },
    text: async () => body,
    clone(): Response {
      return fake as unknown as Response;
    },
  };
  return fake as unknown as Response;
}
