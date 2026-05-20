"""
tests/unit/test_history_cache_warmer.py
----------------------------------------
S47-1 (Sprint 47) -- HistoryCacheWarmer.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from api.services.history_cache_warmer import HistoryCacheWarmer


class TestHistoryCacheWarmerStartStop:

    @pytest.mark.asyncio
    async def test_skipped_when_no_clients(self) -> None:
        w = HistoryCacheWarmer(fgi_client=None, coingecko_client=None)
        await w.start()
        assert w.running is False

    @pytest.mark.asyncio
    async def test_starts_and_stops_cleanly(self) -> None:
        fgi = MagicMock()
        fgi.get_history = AsyncMock(return_value=[])
        w = HistoryCacheWarmer(
            fgi_client=fgi,
            coingecko_client=None,
            startup_delay_seconds=0.01,
            interval_seconds=10.0,
        )
        await w.start()
        assert w.running is True
        await w.stop()
        assert w.running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self) -> None:
        fgi = MagicMock()
        fgi.get_history = AsyncMock(return_value=[])
        w = HistoryCacheWarmer(
            fgi_client=fgi, coingecko_client=None,
            startup_delay_seconds=0.01, interval_seconds=10.0,
        )
        await w.start()
        task_first = w._task
        await w.start()   # second call ignored
        assert w._task is task_first
        await w.stop()


class TestHistoryCacheWarmerTick:

    @pytest.mark.asyncio
    async def test_tick_calls_both_clients(self) -> None:
        fgi = MagicMock()
        cg = MagicMock()
        fgi.get_history = AsyncMock(return_value=[1, 2, 3])
        # Use a real pd.Series so the warmer's len() call exercises the
        # real return-type contract from CoinGeckoClient (CR-001 fix).
        cg.fetch_btc_dominance_history = AsyncMock(
            return_value=pd.Series([10.0, 20.0])
        )

        w = HistoryCacheWarmer(fgi_client=fgi, coingecko_client=cg)
        await w._tick_once()

        fgi.get_history.assert_awaited_once_with(limit=30)
        cg.fetch_btc_dominance_history.assert_awaited_once_with(days=30)
        assert w.last_fgi_points == 3
        assert w.last_btc_dom_points == 2
        assert w.last_run_at is not None

    @pytest.mark.asyncio
    async def test_tick_swallows_fgi_failure(self) -> None:
        fgi = MagicMock()
        cg = MagicMock()
        fgi.get_history = AsyncMock(side_effect=RuntimeError("network down"))
        cg.fetch_btc_dominance_history = AsyncMock(
            return_value=pd.Series([1.0])
        )
        w = HistoryCacheWarmer(fgi_client=fgi, coingecko_client=cg)
        await w._tick_once()
        assert w.last_fgi_points == 0
        assert w.last_btc_dom_points == 1

    @pytest.mark.asyncio
    async def test_tick_swallows_cg_failure(self) -> None:
        fgi = MagicMock()
        cg = MagicMock()
        fgi.get_history = AsyncMock(return_value=[1, 2])
        cg.fetch_btc_dominance_history = AsyncMock(side_effect=RuntimeError("404"))
        w = HistoryCacheWarmer(fgi_client=fgi, coingecko_client=cg)
        await w._tick_once()
        assert w.last_fgi_points == 2
        assert w.last_btc_dom_points == 0

    @pytest.mark.asyncio
    async def test_tick_handles_one_client_only(self) -> None:
        fgi = MagicMock()
        fgi.get_history = AsyncMock(return_value=[1, 2])
        w = HistoryCacheWarmer(fgi_client=fgi, coingecko_client=None)
        await w._tick_once()
        assert w.last_fgi_points == 2
        assert w.last_btc_dom_points == 0

    @pytest.mark.asyncio
    async def test_tick_handles_empty_cg_series(self) -> None:
        """Free-tier CoinGecko returns an empty Series -- len() == 0, no crash."""
        fgi = MagicMock()
        cg = MagicMock()
        fgi.get_history = AsyncMock(return_value=[])
        cg.fetch_btc_dominance_history = AsyncMock(
            return_value=pd.Series(dtype="float64")
        )
        w = HistoryCacheWarmer(fgi_client=fgi, coingecko_client=cg)
        await w._tick_once()
        assert w.last_btc_dom_points == 0


class TestHistoryCacheWarmerCancellation:

    @pytest.mark.asyncio
    async def test_cancel_during_startup_delay(self) -> None:
        """Cancellation during the initial sleep must exit cleanly.

        NOTE: startup_delay_seconds=10.0 is intentional -- it guarantees
        the task is blocked in asyncio.sleep() when cancel() arrives so
        _tick_once is never called.  Do NOT change to 0.0; that would
        race the tick against cancel() and make the assertion flaky.
        """
        fgi = MagicMock()
        fgi.get_history = AsyncMock(return_value=[])
        w = HistoryCacheWarmer(
            fgi_client=fgi, coingecko_client=None,
            startup_delay_seconds=10.0,
            interval_seconds=1.0,
        )
        await w.start()
        await asyncio.sleep(0.05)
        await w.stop()
        fgi.get_history.assert_not_called()
