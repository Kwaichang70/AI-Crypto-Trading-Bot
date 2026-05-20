"""
apps/api/services/history_cache_warmer.py
------------------------------------------
S47-1 (Sprint 47) -- periodic warmer for the FGI + BTC dominance history
caches that the v2 feature pipeline reads synchronously on every bar.

Without this task, ``FearGreedClient._history_cache`` and
``CoinGeckoClient._btc_dom_history`` are never populated in production,
which makes ``value_at_offset_from_cache`` / ``btc_dominance_at_offset_from_cache``
perpetually return None.  The v2 feature builder then substitutes neutral
defaults (0.0) for ``fgi_delta_7d`` and ``btc_dom_delta_7d``, silently
nullifying half of the QT-007 v2 schema improvements.

Cache TTLs
----------
* FGI (alternative.me): 6-hour TTL.  A 30-minute warmer therefore makes
  one real network call every 6 hours; intermediate ticks hit the cache.
* CoinGecko (BTC dominance history): 30-minute TTL aligned with the
  warmer cadence -- one real network call per tick when the cache expires.

Shutdown
--------
The lifespan calls ``stop()``, which cancels the task and awaits
``CancelledError``.  Mirrors the RetrainingService pattern.  The
``BackgroundTaskRegistry.cancel_all()`` method also invokes ``stop()``
on the registered instance for belt-and-suspenders cleanup.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

__all__ = ["HistoryCacheWarmer"]

logger = structlog.get_logger(__name__)

# Sprint 47 default cadence: aligned with the CoinGecko client's 30-min
# TTL so every tick that finds the cache expired produces exactly one
# network call.  The FGI client's 6-hour TTL means most ticks are cache
# hits -- harmless overhead, beneficial for cache freshness on restart.
_DEFAULT_INTERVAL_SECONDS: float = 1800.0
_DEFAULT_STARTUP_DELAY_SECONDS: float = 5.0

# Number of history points / days the warmer requests.  The v2 feature
# builder only needs 7 days for the *_7d_ago lookups; 30 days gives
# operational headroom and matches the upstream API's natural page size.
_FGI_HISTORY_LIMIT: int = 30
_BTC_DOM_HISTORY_DAYS: int = 30


class HistoryCacheWarmer:
    """Periodic warmer for FGI + CoinGecko BTC-dominance history caches."""

    def __init__(
        self,
        fgi_client: Any | None,
        coingecko_client: Any | None,
        *,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        startup_delay_seconds: float = _DEFAULT_STARTUP_DELAY_SECONDS,
        fgi_history_limit: int = _FGI_HISTORY_LIMIT,
        btc_dom_history_days: int = _BTC_DOM_HISTORY_DAYS,
    ) -> None:
        self._fgi_client = fgi_client
        self._cg_client = coingecko_client
        self._interval_seconds = interval_seconds
        self._startup_delay_seconds = startup_delay_seconds
        self._fgi_history_limit = fgi_history_limit
        self._btc_dom_history_days = btc_dom_history_days
        self._task: asyncio.Task[None] | None = None
        self._log = logger.bind(component="history_cache_warmer")
        # Diagnostics surface for S47-6 health endpoint.
        self.last_run_at: float | None = None
        self.last_fgi_points: int = 0
        self.last_btc_dom_points: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            self._log.warning("history_cache_warmer.already_running")
            return
        if self._fgi_client is None and self._cg_client is None:
            self._log.info(
                "history_cache_warmer.skipped",
                reason="no_clients_provided",
            )
            return
        self._task = asyncio.create_task(
            self._warm_loop(), name="history_cache_warmer"
        )
        self._log.info(
            "history_cache_warmer.started",
            interval_seconds=self._interval_seconds,
            fgi_enabled=self._fgi_client is not None,
            cg_enabled=self._cg_client is not None,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                self._log.exception("history_cache_warmer.stop_error")
        self._task = None
        self._log.info("history_cache_warmer.stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Loop body
    # ------------------------------------------------------------------

    async def _warm_loop(self) -> None:
        """Best-effort periodic warm of both history caches."""
        try:
            await asyncio.sleep(self._startup_delay_seconds)
        except asyncio.CancelledError:
            return

        while True:
            await self._tick_once()
            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                break

    async def _tick_once(self) -> None:
        """Run one warm cycle.  Public-ish for testability."""
        fgi_points = 0
        cg_points = 0

        if self._fgi_client is not None:
            try:
                snapshots = await self._fgi_client.get_history(
                    limit=self._fgi_history_limit,
                )
                fgi_points = len(snapshots) if snapshots else 0
            except Exception:
                self._log.warning(
                    "history_cache_warmer.fgi_fetch_failed",
                    exc_info=True,
                )

        if self._cg_client is not None:
            try:
                series = await self._cg_client.fetch_btc_dominance_history(
                    days=self._btc_dom_history_days,
                )
                # ``series`` is always a pd.Series (never None); pd.Series
                # has ambiguous truthiness so ``is not None`` is the only
                # safe guard here.  empty-series len == 0 naturally.
                cg_points = len(series) if series is not None else 0
            except Exception:
                self._log.warning(
                    "history_cache_warmer.btc_dom_fetch_failed",
                    exc_info=True,
                )

        self.last_run_at = time.monotonic()
        self.last_fgi_points = fgi_points
        self.last_btc_dom_points = cg_points
        self._log.info(
            "history_cache_warmer.tick",
            fgi_points=fgi_points,
            btc_dom_points=cg_points,
        )
