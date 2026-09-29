/**
 * apps/ui/src/lib/admin-fetch.ts
 * --------------------------------
 * Shared client-side helper for calling THIS app's own /api/admin/* proxy
 * routes:
 *   - POST /api/admin/kill-switch
 *   - POST /api/admin/kill-switch/clear
 *   - POST /api/admin/runs/[id]/resume
 *   - POST /api/admin/runs/[id]/entries-latch/clear
 *
 * These routes run entirely on THIS Next.js server (they inject the
 * server-only INTERNAL_ADMIN_API_KEY, SEC-C3-008) -- always call them with a
 * root-relative path, never through the FastAPI BASE_URL in `./api`, so the
 * browser never needs (or sees) the admin key.
 *
 * AC8 (reports/vp2-wp1.7/synthesis-spec.md §7): the kill switch and stop
 * paths may run an optional flatten pass that takes up to ~30s server-side
 * plus a 5s outer margin -- every timeoutMs passed to `adminFetch` below
 * must be >= ADMIN_MIN_TIMEOUT_MS (60s) so the browser never aborts a
 * request the backend is still legitimately processing.
 */

import type { ApiResult, ApiError } from "./api";

/** AC8: the floor every admin-proxy (and stop/emergency-stop) timeout must meet. */
export const ADMIN_MIN_TIMEOUT_MS = 60_000;

/** Kill-switch press/clear: optional flatten pass, up to ~35s server-side. */
export const KILL_SWITCH_TIMEOUT_MS = 65_000;

/** Per-run entries-latch clear: a couple of short DB transactions, no flatten. */
export const ENTRIES_LATCH_CLEAR_TIMEOUT_MS = 65_000;

/** Resume: an exchange order scan/cancel/import pass can take a while. */
export const RESUME_TIMEOUT_MS = 65_000;

/**
 * Fetch wrapper for same-origin /api/admin/* proxy routes only. Mirrors the
 * `ApiResult<T>` discriminated union from `./api` so callers use one error-
 * handling shape across the whole app.
 */
export async function adminFetch<T>(
  path: string,
  init: RequestInit,
  timeoutMs: number = ADMIN_MIN_TIMEOUT_MS,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(path, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    let data: unknown;
    try {
      data = await res.json();
    } catch {
      data = undefined;
    }

    if (!res.ok) {
      const body = (data ?? {}) as { error?: string };
      const apiError: ApiError = {
        status: res.status,
        message: body.error ?? `Request failed (HTTP ${res.status})`,
        detail: data,
      };
      return { ok: false, error: apiError };
    }

    return { ok: true, data: data as T };
  } catch (err) {
    clearTimeout(timeoutId);
    const isAbort = err instanceof DOMException && err.name === "AbortError";
    return {
      ok: false,
      error: {
        status: 0,
        message: isAbort
          ? "Request timed out. The operation may still be running on the server."
          : err instanceof Error
            ? err.message
            : "An unexpected network error occurred.",
      },
    };
  }
}
