"""
tests/unit/test_market_signals.py
-----------------------------------
Unit tests for the CoinGecko market signals client.

Covers:
- CoinGeckoSnapshot Pydantic validation (bounds, frozen)
- CoinGeckoClient cache hit / miss / stale fallback / total failure
- Module-level singleton set/get helpers
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.market_signals import (
    CoinGeckoClient,
    CoinGeckoSnapshot,
    get_global_client,
    set_global_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    btc_dominance: float = 52.5,
    market_cap_change_24h: float = 1.5,
    total_volume_change_24h: float = 0.0,
) -> CoinGeckoSnapshot:
    return CoinGeckoSnapshot(
        btc_dominance=btc_dominance,
        market_cap_change_24h=market_cap_change_24h,
        total_volume_change_24h=total_volume_change_24h,
        timestamp=datetime.now(UTC),
    )


def _make_api_response(
    btc_dominance: float = 52.5,
    market_cap_change_24h: float = 1.5,
) -> dict:
    return {
        "data": {
            "market_cap_percentage": {"btc": btc_dominance, "eth": 18.0},
            "market_cap_change_percentage_24h_usd": market_cap_change_24h,
        }
    }


# ---------------------------------------------------------------------------
# TestCoinGeckoSnapshot
# ---------------------------------------------------------------------------

class TestCoinGeckoSnapshot:
    def test_valid_snapshot(self) -> None:
        snap = _make_snapshot(btc_dominance=45.0, market_cap_change_24h=-2.3)
        assert snap.btc_dominance == 45.0
        assert snap.market_cap_change_24h == -2.3

    def test_btc_dominance_lower_bound(self) -> None:
        snap = _make_snapshot(btc_dominance=0.0)
        assert snap.btc_dominance == 0.0

    def test_btc_dominance_upper_bound(self) -> None:
        snap = _make_snapshot(btc_dominance=100.0)
        assert snap.btc_dominance == 100.0

    def test_btc_dominance_below_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            _make_snapshot(btc_dominance=-0.1)

    def test_btc_dominance_above_100_rejected(self) -> None:
        with pytest.raises(Exception):
            _make_snapshot(btc_dominance=100.1)

    def test_frozen(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(Exception):
            snap.btc_dominance = 99.0  # type: ignore[misc]

    def test_negative_market_cap_change_accepted(self) -> None:
        snap = _make_snapshot(market_cap_change_24h=-50.0)
        assert snap.market_cap_change_24h == -50.0


# ---------------------------------------------------------------------------
# TestCoinGeckoClientCache
# ---------------------------------------------------------------------------

class TestCoinGeckoClientCache:
    @pytest.mark.asyncio
    async def test_cache_miss_fetches_api(self) -> None:
        client = CoinGeckoClient(cache_ttl_seconds=60)
        response_data = _make_api_response()

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)

        client._session = mock_session

        snap = await client.get_latest()
        assert snap is not None
        assert snap.btc_dominance == 52.5
        assert snap.market_cap_change_24h == 1.5

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self) -> None:
        client = CoinGeckoClient(cache_ttl_seconds=3600)
        cached_snap = _make_snapshot(btc_dominance=48.0)
        client._latest_cache = (cached_snap, time.monotonic())

        # No session — would crash if network was called
        snap = await client.get_latest()
        assert snap is cached_snap
        assert snap.btc_dominance == 48.0

    @pytest.mark.asyncio
    async def test_stale_cache_fallback_on_failure(self) -> None:
        client = CoinGeckoClient(cache_ttl_seconds=0.001)  # expires instantly
        stale_snap = _make_snapshot(btc_dominance=61.0)
        client._latest_cache = (stale_snap, time.monotonic() - 10)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
        client._session = mock_session

        snap = await client.get_latest()
        assert snap is stale_snap

    @pytest.mark.asyncio
    async def test_total_failure_returns_none(self) -> None:
        client = CoinGeckoClient(cache_ttl_seconds=60)
        # No cache, network fails
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
        client._session = mock_session

        snap = await client.get_latest()
        assert snap is None

    def test_cached_value_property_none_initially(self) -> None:
        client = CoinGeckoClient()
        assert client.cached_value is None

    def test_cached_value_property_returns_snapshot(self) -> None:
        client = CoinGeckoClient()
        snap = _make_snapshot(btc_dominance=55.0)
        client._latest_cache = (snap, time.monotonic())
        assert client.cached_value is snap


# ---------------------------------------------------------------------------
# TestCoinGeckoClientSingleton
# ---------------------------------------------------------------------------

class TestCoinGeckoClientSingleton:
    def test_get_returns_none_before_set(self) -> None:
        import data.market_signals as ms
        original = ms._global_client
        ms._global_client = None
        try:
            assert get_global_client() is None
        finally:
            ms._global_client = original

    def test_set_and_get_round_trip(self) -> None:
        import data.market_signals as ms
        original = ms._global_client
        client = CoinGeckoClient()
        try:
            set_global_client(client)
            assert get_global_client() is client
        finally:
            ms._global_client = original

    def test_parse_snapshot_extracts_fields(self) -> None:
        client = CoinGeckoClient()
        response = _make_api_response(btc_dominance=43.2, market_cap_change_24h=-1.8)
        snap = client._parse_snapshot(response)
        assert abs(snap.btc_dominance - 43.2) < 0.001
        assert abs(snap.market_cap_change_24h - (-1.8)) < 0.001
