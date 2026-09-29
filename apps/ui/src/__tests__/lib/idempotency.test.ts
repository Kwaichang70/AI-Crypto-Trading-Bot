/**
 * apps/ui/src/__tests__/lib/idempotency.test.ts
 * ---------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-18/19/20/21, G-9,
 * UT-17..19, UT-25, UT-28):
 *   - `newIdempotencyKey`: `randomUUID` when present, a hand-built v4 UUID
 *     via `getRandomValues` otherwise, and a throw when neither exists
 *     (never `Math.random`).
 *   - `getStructuredErrorCode`/`classifyIdempotencyError`: the §5 wire
 *     contract's `{"detail": {"code": ...}}` envelope, a flat fallback, a
 *     pydantic list, and a plain string.
 *   - `useIdempotencyKey`: same key for an unchanged snapshot (a retry),
 *     a fresh key for a changed snapshot, and `reset()`.
 *   - `useSubmitLock`: a synchronous, re-entrant-safe check-and-set.
 */

import { act, renderHook } from "@testing-library/react";
import {
  classifyIdempotencyError,
  getStructuredErrorCode,
  IDEMPOTENCY_CODES,
  IDEMPOTENCY_ERROR_CODES,
  newIdempotencyKey,
  useIdempotencyKey,
  useSubmitLock,
} from "@/lib/idempotency";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("newIdempotencyKey (G-9)", () => {
  it("uses crypto.randomUUID when present", () => {
    // jsdom's own Crypto implementation does not define `randomUUID` at
    // all (verified below), so `jest.spyOn` (which requires the property to
    // already exist as a function) cannot be used here -- define it
    // directly, then remove the own property again.
    Object.defineProperty(crypto, "randomUUID", {
      value: () => "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
      configurable: true,
    });
    try {
      expect(newIdempotencyKey()).toBe("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
    } finally {
      delete (crypto as { randomUUID?: unknown }).randomUUID;
    }
  });

  it("falls back to a hand-built v4 UUID via crypto.getRandomValues when randomUUID is absent (jsdom's real gap)", () => {
    // jsdom does not implement `crypto.randomUUID` at all (verified: this
    // repo's jest environment exposes `getRandomValues` but not
    // `randomUUID`), so this exercises the REAL fallback path, not a mock.
    expect(typeof (crypto as { randomUUID?: unknown }).randomUUID).toBe("undefined");
    const key = newIdempotencyKey();
    expect(key).toMatch(UUID_RE);
  });

  it("mints a different key on each call", () => {
    const a = newIdempotencyKey();
    const b = newIdempotencyKey();
    expect(a).not.toBe(b);
  });

  it("throws when no crypto primitive exists (source review: the implementation never calls Math.random at all -- grep-verified, not just untested here)", () => {
    const original = globalThis.crypto;
    delete (globalThis as { crypto?: unknown }).crypto;
    try {
      expect(() => newIdempotencyKey()).toThrow(
        /No secure random source/,
      );
    } finally {
      Object.defineProperty(globalThis, "crypto", { value: original, configurable: true });
    }
  });
});

describe("getStructuredErrorCode / classifyIdempotencyError (UT-28)", () => {
  it("reads the FastAPI {detail:{code}} envelope", () => {
    expect(getStructuredErrorCode({ detail: { code: "idempotency_in_progress" } })).toBe(
      "idempotency_in_progress",
    );
  });

  it("reads a flat {code} shape", () => {
    expect(getStructuredErrorCode({ code: "idempotency_key_reused" })).toBe("idempotency_key_reused");
  });

  it("returns undefined for a pydantic validation-error list", () => {
    expect(getStructuredErrorCode({ detail: [{ loc: ["body", "mode"], msg: "bad" }] })).toBeUndefined();
  });

  it("returns undefined for a plain string detail", () => {
    expect(getStructuredErrorCode({ detail: "Run not found." })).toBeUndefined();
    expect(getStructuredErrorCode("Run not found.")).toBeUndefined();
  });

  it("classifies each of the four idempotency codes and falls back to 'other'", () => {
    expect(classifyIdempotencyError({ detail: { code: IDEMPOTENCY_ERROR_CODES.IN_PROGRESS } })).toEqual({
      kind: "in_progress",
    });
    expect(classifyIdempotencyError({ detail: { code: IDEMPOTENCY_ERROR_CODES.REUSED } })).toEqual({
      kind: "reused",
    });
    expect(classifyIdempotencyError({ detail: { code: IDEMPOTENCY_ERROR_CODES.KEY_REQUIRED } })).toEqual({
      kind: "client_error",
    });
    expect(classifyIdempotencyError({ detail: { code: IDEMPOTENCY_ERROR_CODES.INVALID_FORMAT } })).toEqual({
      kind: "client_error",
    });
    expect(classifyIdempotencyError({ detail: { code: "kill_switch_active" } })).toEqual({ kind: "other" });
    expect(classifyIdempotencyError({ detail: "some string" })).toEqual({ kind: "other" });
  });

  it("IDEMPOTENCY_CODES contains exactly the four wire codes", () => {
    expect(IDEMPOTENCY_CODES.size).toBe(4);
    for (const code of Object.values(IDEMPOTENCY_ERROR_CODES)) {
      expect(IDEMPOTENCY_CODES.has(code)).toBe(true);
    }
  });
});

describe("useIdempotencyKey (SY-70-19)", () => {
  it("returns the same key for an unchanged snapshot (a retry)", () => {
    const { result } = renderHook(() => useIdempotencyKey());
    const k1 = result.current.keyFor("snapshot-a");
    const k2 = result.current.keyFor("snapshot-a");
    expect(k1).toBe(k2);
  });

  it("mints a fresh key when the snapshot changes (implicit reset)", () => {
    const { result } = renderHook(() => useIdempotencyKey());
    const k1 = result.current.keyFor("snapshot-a");
    const k2 = result.current.keyFor("snapshot-b");
    expect(k1).not.toBe(k2);
  });

  it("reset() forces a fresh key even for the same snapshot", () => {
    const { result } = renderHook(() => useIdempotencyKey());
    const k1 = result.current.keyFor("snapshot-a");
    act(() => result.current.reset());
    const k2 = result.current.keyFor("snapshot-a");
    expect(k1).not.toBe(k2);
  });
});

describe("useSubmitLock (SY-70-20, UT-20)", () => {
  it("the second synchronous tryAcquire() fails while locked", () => {
    const { result } = renderHook(() => useSubmitLock());
    expect(result.current.tryAcquire()).toBe(true);
    expect(result.current.tryAcquire()).toBe(false);
  });

  it("release() allows a subsequent tryAcquire() to succeed", () => {
    const { result } = renderHook(() => useSubmitLock());
    expect(result.current.tryAcquire()).toBe(true);
    act(() => result.current.release());
    expect(result.current.tryAcquire()).toBe(true);
  });
});
