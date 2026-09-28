/**
 * apps/ui/src/hooks/use-polling.ts
 * -----------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-23) — a self-rescheduling
 * poll with a superseding `refetch()`, adopted by the run detail page (main
 * poll + diagnostics poll) and the kill-switch panel (SY-70-24/25).
 *
 * Contract (binding, SY-70-23):
 *   (a) A self-rescheduling `setTimeout` — the next tick is scheduled only
 *       after the current one settles (never a bare `setInterval`, which
 *       would let ticks overlap under load).
 *   (b) `tick()` from the timer, `visibilitychange`, or `enabled` becoming
 *       true is a NO-OP while a request is in flight. `refetch()`
 *       SUPERSEDES: it synchronously bumps a sequence ref before any
 *       `await`, so an older in-flight result is discarded when it lands.
 *       The in-flight guard does not apply to `refetch()`, and it works
 *       even while `enabled=false` (one-shot use).
 *   (c) `immediate` (default `true`): when `enabled` becomes `true`, tick
 *       now, or after `intervalMs` if `false`.
 *   (d) Backoff: after the n-th consecutive failure, the delay is
 *       `min(maxBackoffMs, intervalMs * backoffFactor ** min(n, 6))`
 *       (defaults 60_000 / 2). A success resets the delay to `intervalMs`.
 *   (e) `fetcher` and `onSuccess` are held in refs — always the latest
 *       closure fires, even from a timer scheduled several renders ago.
 *       `onSuccess` fires only for an APPLIED (non-superseded) result.
 *   (f) A thrown fetcher becomes `{status: 0, message}` — never an unhandled
 *       rejection.
 *   (g) Hidden tab: clear the timer, schedule nothing. Becoming visible:
 *       tick now unless a request is already in flight.
 *   (h) Unmount: a `mountedRef` guard, timer cleared, listener removed.
 *
 * Deferred: threading an `AbortSignal` through (`apiFetch` at `api.ts:143`
 * overwrites any caller-supplied `signal`) — CF-70-2.
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiError, ApiResult } from "@/lib/api";

export interface UsePollingOptions<T> {
  fetcher: () => Promise<ApiResult<T>>;
  intervalMs: number;
  enabled: boolean;
  /** Default `true`. */
  immediate?: boolean;
  /** Fires only for an applied (non-superseded) successful result. */
  onSuccess?: (data: T) => void;
  /** Default `60_000`. */
  maxBackoffMs?: number;
  /** Default `2`. */
  backoffFactor?: number;
}

export interface UsePollingResult<T> {
  /** The last successfully-applied payload. Never cleared by a failure — "keep the last good data" (I10). */
  data: T | undefined;
  /** The most recent failure, or `null` right after a success. */
  error: ApiError | null;
  /** `Date.now()` of the last successful tick, or `null` before the first one. */
  lastSuccessAt: number | null;
  /** `true` while a request (timer-driven or `refetch()`-driven) is in flight. */
  isFetching: boolean;
  /** Bypasses the in-flight guard and supersedes anything currently in flight. Works even when `enabled=false`. */
  refetch: () => Promise<void>;
}

const DEFAULT_MAX_BACKOFF_MS = 60_000;
const DEFAULT_BACKOFF_FACTOR = 2;

function isBrowserHidden(): boolean {
  return typeof document !== "undefined" && document.visibilityState === "hidden";
}

export function usePolling<T>(options: UsePollingOptions<T>): UsePollingResult<T> {
  const {
    fetcher,
    intervalMs,
    enabled,
    immediate = true,
    onSuccess,
    maxBackoffMs = DEFAULT_MAX_BACKOFF_MS,
    backoffFactor = DEFAULT_BACKOFF_FACTOR,
  } = options;

  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<ApiError | null>(null);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const [isFetching, setIsFetching] = useState(false);

  // (e) Always-fresh closures for the caller-supplied callbacks.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const onSuccessRef = useRef(onSuccess);
  onSuccessRef.current = onSuccess;

  // Config read via refs too, so a stale closure captured by an old timer
  // callback still sees the latest values once it actually fires.
  const intervalMsRef = useRef(intervalMs);
  intervalMsRef.current = intervalMs;
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const maxBackoffMsRef = useRef(maxBackoffMs);
  maxBackoffMsRef.current = maxBackoffMs;
  const backoffFactorRef = useRef(backoffFactor);
  backoffFactorRef.current = backoffFactor;

  const seqRef = useRef(0);
  const inFlightRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const failuresRef = useRef(0);
  const mountedRef = useRef(true);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Forward-declared: assigned below, referenced by scheduleNext's timer
  // callback and by the visibility/enable effects.
  const runTickRef = useRef<(force: boolean) => Promise<void>>(async () => {});

  const scheduleNext = useCallback(
    (delayMs: number) => {
      clearTimer();
      if (!mountedRef.current) return;
      if (!enabledRef.current) return; // one-shot refetch never self-reschedules while disabled
      if (isBrowserHidden()) return; // (g) schedule nothing while hidden
      timerRef.current = setTimeout(() => {
        void runTickRef.current(false);
      }, delayMs);
    },
    [clearTimer],
  );

  const runTick = useCallback(
    async (force: boolean) => {
      if (!mountedRef.current) return;
      // (b) A non-forced tick is a no-op while a request is already in flight.
      if (!force && inFlightRef.current) return;

      clearTimer();
      inFlightRef.current = true;
      // (b) `refetch()` bumps the sequence BEFORE any await, so it supersedes
      // whatever is already in flight (which kept its own, now-stale, mySeq).
      const mySeq = force ? ++seqRef.current : seqRef.current;

      setIsFetching(true);

      let result: ApiResult<T>;
      try {
        result = await fetcherRef.current();
      } catch (err) {
        // (f) A thrown fetcher becomes a synthetic status-0 error.
        result = {
          ok: false,
          error: {
            status: 0,
            message: err instanceof Error ? err.message : "An unexpected error occurred.",
          },
        };
      }

      if (!mountedRef.current) return;
      if (mySeq !== seqRef.current) {
        // Superseded by a later refetch() while this one was in flight —
        // discard silently. Do NOT touch inFlightRef/isFetching: the
        // winning attempt owns clearing those.
        return;
      }

      inFlightRef.current = false;
      setIsFetching(false);

      if (result.ok) {
        failuresRef.current = 0;
        setData(result.data);
        setError(null);
        setLastSuccessAt(Date.now());
        onSuccessRef.current?.(result.data);
        scheduleNext(intervalMsRef.current);
      } else {
        failuresRef.current += 1;
        setError(result.error);
        const n = Math.min(failuresRef.current, 6);
        const delay = Math.min(
          maxBackoffMsRef.current,
          intervalMsRef.current * Math.pow(backoffFactorRef.current, n),
        );
        scheduleNext(delay);
      }
    },
    [clearTimer, scheduleNext],
  );

  runTickRef.current = runTick;

  // Mount/unmount lifecycle (h).
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearTimer();
    };
  }, [clearTimer]);

  // (c) enabled -> tick now (immediate) or after intervalMs; disabled -> stop.
  useEffect(() => {
    if (!enabled) {
      clearTimer();
      return;
    }
    if (immediate) {
      void runTickRef.current(false);
    } else {
      scheduleNext(intervalMsRef.current);
    }
    return () => {
      clearTimer();
    };
    // Only re-run when `enabled` flips — `immediate`/`intervalMs` are read
    // fresh via refs/closure at the moment this effect actually fires.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, clearTimer, scheduleNext]);

  // (g) Visibility: pause while hidden, tick immediately on becoming visible.
  useEffect(() => {
    function handleVisibilityChange() {
      if (isBrowserHidden()) {
        clearTimer();
      } else if (enabledRef.current) {
        void runTickRef.current(false);
      }
    }
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, [clearTimer]);

  const refetch = useCallback(async () => {
    await runTickRef.current(true);
  }, []);

  return { data, error, lastSuccessAt, isFetching, refetch };
}
