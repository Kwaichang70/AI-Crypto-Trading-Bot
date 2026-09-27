/**
 * apps/ui/src/lib/admin-route-guards.ts
 * -----------------------------------------
 * Shared request-validation helpers for the four state-changing
 * `/api/admin/*` proxy routes (WP1.7b round 2, findings S-03/S-05):
 *   - `POST /api/admin/kill-switch`
 *   - `POST /api/admin/kill-switch/clear`
 *   - `POST /api/admin/runs/[id]/resume`
 *   - `POST /api/admin/runs/[id]/entries-latch/clear`
 *
 * Both helpers return early (a `Response` to send back immediately) rather
 * than throwing, so every route stays a flat sequence of
 * `const rejected = guard(...); if (rejected) return rejected;` checks.
 */

const RUN_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * WP17b-S-03: the two `/runs/[id]/*` proxies interpolate `params.id`
 * straight into the upstream URL. `encodeURIComponent` stops `/`
 * injection, but leaves `..`/`.` untouched — the WHATWG URL parser then
 * normalises those away, so e.g. `id=".."` silently sends the ADMIN-KEYED
 * request to `.../api/v1/resume` instead of `.../api/v1/runs/../resume`.
 * Reject anything that isn't a plain UUID BEFORE the admin key is ever
 * read (see call sites), so a malformed `[id]` segment can never reach the
 * upstream fetch with the admin key attached.
 */
export function isValidRunId(id: string): boolean {
  return RUN_ID_RE.test(id);
}

export function invalidRunIdResponse(): Response {
  return Response.json({ error: "Invalid run id" }, { status: 400 });
}

/**
 * WP17b-C-R2-01 (round 3): the deployed topology (`infra/Dockerfile.ui` runs
 * the Next.js standalone server with `HOSTNAME=0.0.0.0`, behind Caddy) means
 * `request.url`'s host is ALWAYS the server's own internal bind address
 * (e.g. `0.0.0.0:3000`/`localhost:3000`) — it never reflects the client-
 * visible host, no matter what the browser or Caddy sent. Comparing
 * `Origin` against `new URL(request.url).host` (round 2's approach) was
 * therefore comparing against a constant that can never equal a real
 * browser `Origin` in production, false-403'ing every admin action for any
 * caller that sends `Origin` without `Sec-Fetch-Site`.
 *
 * The correct comparison is against the PUBLIC host the operator actually
 * browses to:
 *   1. `NEXTAUTH_URL` (preferred — next-auth already requires this env var
 *      to be set to the app's real public URL, so it is the one source of
 *      truth this deployment already trusts).
 *   2. `X-Forwarded-Host` (set by Caddy's default reverse-proxy behaviour).
 *   3. `Host` (Caddy preserves the client's original Host header by
 *      default, confirmed against `infra/Caddyfile`).
 * If none of these is available, there is no trustworthy expected host to
 * compare against, so the request is rejected (fail closed).
 */
function expectedAdminHost(request: Request): string | null {
  const nextAuthUrl = process.env.NEXTAUTH_URL;
  if (nextAuthUrl) {
    try {
      return new URL(nextAuthUrl).host;
    } catch {
      // Malformed NEXTAUTH_URL -- fall through to the header-based
      // fallbacks below rather than treating this as fatal.
    }
  }
  const forwardedHost = request.headers.get("x-forwarded-host");
  if (forwardedHost) return forwardedHost;
  return request.headers.get("host");
}

/**
 * WP17b-S-05 (defence-in-depth): the session cookie is already
 * `SameSite=strict` + `Secure` + `HttpOnly` (`auth-options.ts`), which
 * mitigates classic CSRF, but these routes previously accepted ANY request
 * `Content-Type` (a lenient `request.text()` + `JSON.parse`), so a
 * `text/plain` "simple request" (no CORS preflight) was still parseable.
 * This guard requires:
 *   - An explicit `application/json` Content-Type (415 otherwise).
 *   - `Sec-Fetch-Site` (sent by every modern browser fetch/XHR) to be
 *     `same-origin` or `none` when present.
 *   - When `Sec-Fetch-Site` is ABSENT but an `Origin` header IS present
 *     (older browsers, privacy-focused proxies/extensions that strip
 *     Fetch-Metadata headers, or a direct curl/script call that happens to
 *     set `Origin`), the `Origin` header's host is compared against
 *     `expectedAdminHost()` above (WP17b-C-R2-01) — NOT against
 *     `request.url`, which is always this server's own internal bind
 *     address behind Caddy and can never match a real browser `Origin`.
 *     403 on any mismatch, on an unparseable `Origin`, or when no expected
 *     host could be determined at all (fail closed).
 *   - Requests with NEITHER `Sec-Fetch-Site` NOR `Origin` (e.g. a same-
 *     origin server-to-server call, or the documented operator curl
 *     fallback, with no browser fetch metadata at all) are allowed through
 *     — there is no cross-origin signal to reject on, and rejecting here
 *     would only break legitimate non-browser callers.
 *
 * Returns a `Response` to return immediately on failure, or `null` to
 * continue processing the request.
 */
export function checkSameOriginJsonRequest(request: Request): Response | null {
  // WP17b-S-R2-03 (round 3): compare only the media type -- a crafted
  // `Content-Type: text/plain; x=application/json` would otherwise pass
  // `.includes("application/json")` while still being a CORS-safelisted
  // "simple request" content type in the browser (no preflight).
  const contentType = (request.headers.get("content-type") ?? "")
    .split(";")[0]
    .trim()
    .toLowerCase();
  if (contentType !== "application/json") {
    return Response.json(
      { error: "Content-Type must be application/json" },
      { status: 415 },
    );
  }

  const secFetchSite = request.headers.get("sec-fetch-site");
  if (secFetchSite !== null) {
    if (secFetchSite !== "same-origin" && secFetchSite !== "none") {
      return Response.json({ error: "Cross-origin request rejected" }, { status: 403 });
    }
    return null;
  }

  const origin = request.headers.get("origin");
  if (origin !== null) {
    let originHost: string;
    try {
      originHost = new URL(origin).host;
    } catch {
      return Response.json({ error: "Invalid Origin header" }, { status: 403 });
    }
    const expectedHost = expectedAdminHost(request);
    if (!expectedHost || originHost !== expectedHost) {
      return Response.json({ error: "Cross-origin request rejected" }, { status: 403 });
    }
  }

  return null;
}
