"""
apps/api/services/fx_cache_warmer.py
--------------------------------------
M6 MVP (Sprint 49) -- FX rate cache warmer background task stub.

Mirrors ``HistoryCacheWarmer`` (S47-1) in structure and lifecycle semantics.

In M6 MVP, ``_tick()`` is a **no-op log stub** — no network calls are made.
The class exists to:
  1. Provide the ``BackgroundTaskRegistry.fx_cache_warmer`` slot with the
     correct interface (``running``, ``last_run_at``, ``last_rates_cached``).
  2. Surface diagnostics on ``/api/v1/health/background``.
  3. Give M6b a clear location to wire in real CCXT FX fetching without any
     structural refactor (just replace ``_tick()`` body).

M6b will:
  - Accept an ``FxService`` reference and a list of currency pairs to warm.
  - In ``_tick()``, call ``FxService.seed_rate()`` for each pair after
    fetching from the live CCXT exchange client.

Shutdown
--------
``stop()`` cancels the internal asyncio.Task and awaits clean exit.
``BackgroundTaskRegistry.cancel_all()`` drives ``stop()`` — mirrors the
``HistoryCacheWarmer`` pattern established in Sprint 47.
"""

from __future__ import annotations

import asyncio
import time

import structlog

__all__ = ["FxCacheWarmer"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (AR-M4-004 / M5 CR-002 pattern)
# ---------------------------------------------------------------------------

_DEFAULT_INTERVAL_SECONDS: float = 1800.0    # 30-minute cadence
_DEFAULT_STARTUP_DELAY_SECONDS: float = 10.0  # stagger after HistoryCacheWarmer (5 s delay)


class FxCacheWarmer:
    """Periodic FX rate cache warmer background task.

    M6 MVP: ``_tick()`` is a no-op stub.  Real fetching lands in M6b.

    Parameters
    ----------
    interval_seconds:
        Seconds between ticks.  Defaults to 1800 (30 min).
    startup_delay_seconds:
        Seconds to sleep before the first tick.  Defaults to 10 s to
        stagger startup behind ``HistoryCacheWarmer`` (5 s delay).
    """

    def __init__(
        self,
        *,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        startup_delay_seconds: float = _DEFAULT_STARTUP_DELAY_SECONDS,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._startup_delay_seconds = startup_delay_seconds
        self._task: asyncio.Task[None] | None = None
        self._log = logger.bind(component="fx_cache_warmer")
        # Diagnostics — read by /api/v1/health/background
        self.last_run_at: float | None = None
        self.last_rates_cached: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background warm loop.  Idempotent if already running."""
        if self._task is not None and not self._task.done():
            self._log.warning("fx_cache_warmer.already_running")
            return
        self._task = asyncio.create_task(
            self._warm_loop(), name="fx_cache_warmer"
        )
        self._log.info(
            "fx_cache_warmer.started",
            interval_seconds=self._interval_seconds,
            note="M6 MVP stub — no network calls until M6b",
        )

    async def stop(self) -> None:
        """Cancel the warm loop and await clean exit."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                self._log.exception("fx_cache_warmer.stop_error")
        self._task = None
        self._log.info("fx_cache_warmer.stopped")

    @property
    def running(self) -> bool:
        """True while the internal asyncio.Task is alive."""
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Loop body
    # ------------------------------------------------------------------

    async def _warm_loop(self) -> None:
        """Best-effort periodic tick loop."""
        try:
            await asyncio.sleep(self._startup_delay_seconds)
        except asyncio.CancelledError:
            return

        while True:
            # M6b: wrap _tick() with CancelledError re-raise when real I/O is added
            await self._tick()
            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        """Single warm cycle.  Public-ish for testability.

        M6 MVP: logs a heartbeat and records zero cached rates.
        M6b will call ``FxService.seed_rate()`` for configured pairs here.
        """
        # M6 MVP stub — no network I/O.
        # M6b: replace with real CCXT FX rate fetch + cache populate
        rates_cached = 0

        self.last_run_at = time.time()
        self.last_rates_cached = rates_cached
        self._log.info(
            "fx_cache_warmer.tick_mvp_stub",
            rates_cached=rates_cached,
            note="M6 MVP stub — real fetch in M6b",
        )
