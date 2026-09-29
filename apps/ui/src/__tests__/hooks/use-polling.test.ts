/**
 * apps/ui/src/__tests__/hooks/use-polling.test.ts
 * ----------------------------------------------------
 * WP7.0 (reports/vp2-wp7.0/synthesis-spec.md SY-70-23, UT-01..10):
 * `usePolling`'s self-rescheduling tick, superseding `refetch()`,
 * exponential backoff, hidden-tab pause, and unmount cleanup.
 */

import { act, renderHook } from "@testing-library/react";
import { usePolling } from "@/hooks/use-polling";
import type { ApiResult } from "@/lib/api";

function ok<T>(data: T): ApiResult<T> {
  return { ok: true, data };
}

function err(message: string, status = 500): ApiResult<never> {
  return { ok: false, error: { status, message } };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", {
    value: state,
    configurable: true,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => {
  jest.useFakeTimers();
  setVisibility("visible");
});

afterEach(() => {
  jest.useRealTimers();
});

describe("usePolling — immediate tick + applying a successful result", () => {
  it("fetches immediately when enabled (default immediate=true) and stores data/lastSuccessAt", async () => {
    const fetcher = jest.fn().mockResolvedValue(ok({ n: 1 }));
    const { result } = renderHook(() =>
      usePolling({ fetcher, intervalMs: 1000, enabled: true }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual({ n: 1 });
    expect(result.current.error).toBeNull();
    expect(result.current.lastSuccessAt).not.toBeNull();
  });

  it("immediate:false does not fetch until intervalMs elapses (UT-10)", async () => {
    const fetcher = jest.fn().mockResolvedValue(ok({ n: 1 }));
    renderHook(() => usePolling({ fetcher, intervalMs: 1000, enabled: true, immediate: false }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).not.toHaveBeenCalled();

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});

describe("usePolling — I10: keeps the last good data on a later failure", () => {
  it("does not clear `data` when a subsequent tick fails, but sets `error`", async () => {
    const fetcher = jest
      .fn()
      .mockResolvedValueOnce(ok({ n: 1 }))
      .mockResolvedValueOnce(err("boom"));
    const { result } = renderHook(() =>
      usePolling({ fetcher, intervalMs: 1000, enabled: true }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ n: 1 });

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1000);
    });

    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(result.current.data).toEqual({ n: 1 }); // kept
    expect(result.current.error).not.toBeNull();
    expect(result.current.error?.message).toBe("boom");
  });
});

describe("usePolling — exponential backoff (d)", () => {
  it("doubles the delay per consecutive failure, capped at maxBackoffMs, and resets to intervalMs after a success", async () => {
    const fetcher = jest
      .fn()
      .mockResolvedValueOnce(err("f1"))
      .mockResolvedValueOnce(err("f2"))
      .mockResolvedValueOnce(err("f3"))
      .mockResolvedValueOnce(ok({ n: 42 }))
      .mockResolvedValue(err("f-after-reset"));

    renderHook(() =>
      usePolling({ fetcher, intervalMs: 1000, enabled: true, maxBackoffMs: 5000 }),
    );

    // Immediate tick #1 -> failure #1 -> next delay = 1000*2^1 = 2000
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1999);
    });
    expect(fetcher).toHaveBeenCalledTimes(1); // not yet

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1);
    });
    // tick #2 -> failure #2 -> next delay = 1000*2^2 = 4000
    expect(fetcher).toHaveBeenCalledTimes(2);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(3999);
    });
    expect(fetcher).toHaveBeenCalledTimes(2);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1);
    });
    // tick #3 -> failure #3 -> next delay = min(5000, 1000*2^3=8000) = 5000 (capped)
    expect(fetcher).toHaveBeenCalledTimes(3);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(4999);
    });
    expect(fetcher).toHaveBeenCalledTimes(3);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1);
    });
    // tick #4 -> SUCCESS -> resets backoff; next delay = intervalMs = 1000
    expect(fetcher).toHaveBeenCalledTimes(4);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(999);
    });
    expect(fetcher).toHaveBeenCalledTimes(4);

    await act(async () => {
      await jest.advanceTimersByTimeAsync(1);
    });
    // tick #5 -> failure again, proving the counter was reset to 0 by the
    // success (not still at 3, which would have scheduled 5000ms again).
    expect(fetcher).toHaveBeenCalledTimes(5);
  });
});

describe("usePolling — refetch() supersedes an in-flight tick (G-6, UT-09)", () => {
  it("discards the older in-flight result, applies the newer one, and fires onSuccess only for the applied result", async () => {
    const d1 = deferred<ApiResult<{ tag: string }>>();
    const d2 = deferred<ApiResult<{ tag: string }>>();
    const fetcher = jest.fn().mockReturnValueOnce(d1.promise).mockReturnValueOnce(d2.promise);
    const onSuccess = jest.fn();

    const { result } = renderHook(() =>
      usePolling({ fetcher, intervalMs: 1000, enabled: true, onSuccess }),
    );

    // Kick off the immediate tick (in flight, unresolved).
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    // refetch() supersedes it — bypasses the in-flight guard entirely.
    let refetchPromise!: Promise<void>;
    await act(async () => {
      refetchPromise = result.current.refetch();
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(2);

    // Resolve the NEWER call first, then the OLDER one.
    await act(async () => {
      d2.resolve(ok({ tag: "newer" }));
      await refetchPromise;
    });
    expect(result.current.data).toEqual({ tag: "newer" });
    expect(onSuccess).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledWith({ tag: "newer" });

    // The stale, older result lands afterwards and must be discarded.
    await act(async () => {
      d1.resolve(ok({ tag: "older" }));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ tag: "newer" });
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("refetch() works even while enabled=false (one-shot)", async () => {
    const fetcher = jest.fn().mockResolvedValue(ok({ n: 7 }));
    const { result } = renderHook(() =>
      usePolling({ fetcher, intervalMs: 1000, enabled: false }),
    );

    expect(fetcher).not.toHaveBeenCalled();

    await act(async () => {
      await result.current.refetch();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual({ n: 7 });

    // Being disabled, a successful refetch must NOT silently start
    // auto-polling — no further calls even once intervalMs elapses.
    await act(async () => {
      await jest.advanceTimersByTimeAsync(5000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});

describe("usePolling — hidden tab pause / resume (g)", () => {
  it("clears the timer while hidden and schedules nothing, then ticks immediately on becoming visible", async () => {
    const fetcher = jest.fn().mockResolvedValue(ok({ n: 1 }));
    renderHook(() => usePolling({ fetcher, intervalMs: 1000, enabled: true }));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => {
      setVisibility("hidden");
    });

    // Even well past intervalMs, no further fetch while hidden.
    await act(async () => {
      await jest.advanceTimersByTimeAsync(10_000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    // Becoming visible ticks immediately (no in-flight request).
    await act(async () => {
      setVisibility("visible");
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("does not double-fetch on visibilitychange while a request is already in flight", async () => {
    const d = deferred<ApiResult<{ n: number }>>();
    const fetcher = jest.fn().mockReturnValue(d.promise);
    renderHook(() => usePolling({ fetcher, intervalMs: 1000, enabled: true }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1); // still in flight (unresolved)

    // A visibilitychange (hidden -> visible without an intervening fetch
    // settling) must not add a second concurrent call.
    act(() => {
      setVisibility("hidden");
    });
    act(() => {
      setVisibility("visible");
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      d.resolve(ok({ n: 1 }));
      await Promise.resolve();
      await Promise.resolve();
    });
  });
});

describe("usePolling — thrown fetcher becomes a status-0 error (f)", () => {
  it("never produces an unhandled rejection; error.status is 0", async () => {
    const fetcher = jest.fn().mockRejectedValue(new Error("network died"));
    const { result } = renderHook(() => usePolling({ fetcher, intervalMs: 1000, enabled: true }));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.error).toEqual({ status: 0, message: "network died" });
  });
});

describe("usePolling — unmount cleanup (h)", () => {
  it("stops firing the fetcher after unmount even once timers elapse", async () => {
    const fetcher = jest.fn().mockResolvedValue(ok({ n: 1 }));
    const { unmount } = renderHook(() => usePolling({ fetcher, intervalMs: 1000, enabled: true }));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    unmount();

    await act(async () => {
      await jest.advanceTimersByTimeAsync(10_000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
