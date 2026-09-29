/**
 * apps/ui/src/lib/idempotency.ts
 * ---------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-18/19/20/21) shared
 * helpers for the `Idempotency-Key` request lifecycle:
 *   - `newIdempotencyKey()` mints a v4 UUID (G-9: never `Math.random`).
 *   - `useIdempotencyKey()` is the ref-based per-submit-intent key lifecycle
 *     (SY-70-19): the same key is returned for the same "snapshot" (e.g. the
 *     JSON-stringified request body) so a retry reuses it, and a changed
 *     snapshot mints a fresh key automatically.
 *   - `useSubmitLock()` is a synchronous check-and-set guard (SY-70-20) that
 *     closes the double-submit race a React state flag alone cannot (two
 *     synchronous clicks can both observe the pre-update `isSubmitting`
 *     value before either state update is committed).
 *   - `getStructuredErrorCode()`/`classifyIdempotencyError()` read the wire
 *     contract's `{"detail": {"code": ...}}` envelope (§5) to drive the
 *     reset/keep matrix in SY-70-19 and the error-handling table in
 *     SY-70-21.
 *
 * None of this reads or writes `sessionStorage`/`localStorage` — the key
 * lives only in a component-instance ref, so a page reload always starts a
 * new logical submission with a fresh key (CF-70-8, deferred).
 */

"use client";

import { useCallback, useRef } from "react";

// ---------------------------------------------------------------------------
// Key minting (SY-70-18, G-9)
// ---------------------------------------------------------------------------

interface MinimalCrypto {
  randomUUID?: () => string;
  getRandomValues?: <T extends ArrayBufferView>(array: T) => T;
}

function getCrypto(): MinimalCrypto | undefined {
  return typeof crypto !== "undefined" ? (crypto as unknown as MinimalCrypto) : undefined;
}

/**
 * Mints a canonical (lowercase, hyphenated) v4 UUID for one `Idempotency-Key`.
 *
 * Preference order (G-9):
 *   1. `crypto.randomUUID()` — available in every secure context.
 *   2. A v4 UUID built by hand from `crypto.getRandomValues()` (the same
 *      fallback pattern already used by `app/optimize/param-grid-editor.tsx`
 *      for non-secure-context browsers).
 *   3. Throws — this function NEVER falls back to `Math.random()`, which is
 *      not cryptographically strong and must never seed a value the backend
 *      uses to guarantee at-most-one-run-per-key.
 */
export function newIdempotencyKey(): string {
  const c = getCrypto();

  if (c?.randomUUID) {
    return c.randomUUID();
  }

  if (c?.getRandomValues) {
    const bytes = new Uint8Array(16);
    c.getRandomValues(bytes);
    // RFC 4122 §4.4: set the version (4) and variant (10) bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  throw new Error(
    "No secure random source (crypto.randomUUID/getRandomValues) is available to mint an Idempotency-Key.",
  );
}

// ---------------------------------------------------------------------------
// Structured error codes (SY-70-04/21, §5)
// ---------------------------------------------------------------------------

export const IDEMPOTENCY_ERROR_CODES = {
  KEY_REQUIRED: "idempotency_key_required",
  INVALID_FORMAT: "idempotency_key_invalid_format",
  REUSED: "idempotency_key_reused",
  IN_PROGRESS: "idempotency_in_progress",
} as const;

export type IdempotencyErrorCode =
  (typeof IDEMPOTENCY_ERROR_CODES)[keyof typeof IDEMPOTENCY_ERROR_CODES];

/** Every code this module recognises as idempotency-specific (vs. an unrelated 4xx/5xx). */
export const IDEMPOTENCY_CODES: ReadonlySet<string> = new Set(
  Object.values(IDEMPOTENCY_ERROR_CODES),
);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Unwraps `ApiError.detail` (the raw parsed response body `apiFetch` hands
 * back) down to the FastAPI `HTTPException(detail={"code": ...})` envelope's
 * `code` string — `{"detail": {"code": ...}}` per §5 wire contract. Falls
 * back to a flat `{"code": ...}` shape for defence in depth. Returns
 * `undefined` for a pydantic validation-error list, a plain string, or
 * anything else without a string `code` (mirrors `exit-config.ts`'s
 * `unwrapExitConfigDetail`/`isPlainObject` pattern, UT-28).
 */
export function getStructuredErrorCode(raw: unknown): string | undefined {
  const inner = isPlainObject(raw) && "detail" in raw ? raw.detail : raw;
  if (!isPlainObject(inner)) return undefined;
  const code = inner.code;
  return typeof code === "string" ? code : undefined;
}

export type IdempotencyOutcome =
  | { kind: "in_progress" }
  | { kind: "reused" }
  /** 428 `idempotency_key_required` or 400 `idempotency_key_invalid_format` — a client bug, not a retriable condition. */
  | { kind: "client_error" }
  /** Not one of the four idempotency codes — render with the existing generic/exit-config error path. */
  | { kind: "other" };

/** Classifies a failed response's `error.detail` per the SY-70-21 handling table. */
export function classifyIdempotencyError(detail: unknown): IdempotencyOutcome {
  const code = getStructuredErrorCode(detail);
  if (code === IDEMPOTENCY_ERROR_CODES.IN_PROGRESS) return { kind: "in_progress" };
  if (code === IDEMPOTENCY_ERROR_CODES.REUSED) return { kind: "reused" };
  if (code === IDEMPOTENCY_ERROR_CODES.KEY_REQUIRED || code === IDEMPOTENCY_ERROR_CODES.INVALID_FORMAT) {
    return { kind: "client_error" };
  }
  return { kind: "other" };
}

// User-facing copy shared verbatim across the new-run form, the promote
// dialog and the optimize launch pages, so the exact strings only exist once.
export const IDEMPOTENCY_IN_PROGRESS_MESSAGE =
  "Your previous request is still being processed…";
export const IDEMPOTENCY_REUSED_MESSAGE =
  "This request was already used for a different submission. Please try again.";
export const IDEMPOTENCY_CLIENT_ERROR_MESSAGE =
  "Client error: the request was sent without a valid retry token. Reload the page and try again.";

// ---------------------------------------------------------------------------
// useIdempotencyKey (SY-70-19)
// ---------------------------------------------------------------------------

export interface UseIdempotencyKeyResult {
  /**
   * Returns the key for this submit intent, keyed by `snapshot` (e.g. the
   * JSON-stringified request body). The SAME key is returned across repeated
   * calls with an unchanged snapshot (a retry) — a changed snapshot mints
   * a fresh key automatically (SY-70-19's "implicitly on a snapshot change").
   */
  keyFor: (snapshot: string) => string;
  /**
   * Explicit reset — call on success, on 422 `idempotency_key_reused`, and
   * on 428/400 idempotency codes (SY-70-19's reset column). Never call this
   * for status 0, 409 (either code), 403, or any other 4xx/5xx — those keep
   * the key so a same-key retry is safe.
   */
  reset: () => void;
}

/** Ref-based per-submit-intent `Idempotency-Key` lifecycle (SY-70-19). Never `sessionStorage` (CF-70-8). */
export function useIdempotencyKey(): UseIdempotencyKeyResult {
  const keyRef = useRef<string | null>(null);
  const snapshotRef = useRef<string | null>(null);

  const keyFor = useCallback((snapshot: string): string => {
    if (keyRef.current === null || snapshotRef.current !== snapshot) {
      keyRef.current = newIdempotencyKey();
      snapshotRef.current = snapshot;
    }
    return keyRef.current;
  }, []);

  const reset = useCallback(() => {
    keyRef.current = null;
    snapshotRef.current = null;
  }, []);

  return { keyFor, reset };
}

// ---------------------------------------------------------------------------
// useSubmitLock (SY-70-20)
// ---------------------------------------------------------------------------

export interface UseSubmitLockResult {
  /** Synchronous check-and-set. Returns `false` (and does nothing) when already locked. */
  tryAcquire: () => boolean;
  /** Always call from a `finally` block so a thrown/rejected submit still releases the lock. */
  release: () => void;
}

/**
 * A synchronous ref-based lock guarding the FIRST statement of a submit
 * handler (SY-70-20) — closes the race where two synchronous clicks both
 * read the same pre-update `isSubmitting` boolean before React commits
 * either state update. `isSubmitting`/`loading` state stays as the visual
 * (disabled-button) guard; this is the correctness guard underneath it.
 */
export function useSubmitLock(): UseSubmitLockResult {
  const lockedRef = useRef(false);

  const tryAcquire = useCallback((): boolean => {
    if (lockedRef.current) return false;
    lockedRef.current = true;
    return true;
  }, []);

  const release = useCallback(() => {
    lockedRef.current = false;
  }, []);

  return { tryAcquire, release };
}
