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


# ===================================================================
# QT-007e (Sprint 46) -- Historical BTC dominance fetcher
# ===================================================================


class TestQT007eBTCDominanceHistory:
    """``fetch_btc_dominance_history`` + ``set_btc_dominance_history`` --
    PRO-tier fetch with free-tier graceful degradation + manual-override
    hook (sticky semantics)."""

    @staticmethod
    def _make_manual_series(values: list[float], days: int = 14):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
        return pd.Series(values[:days], index=idx, dtype="float64",
                         name="btc_dominance")

    @staticmethod
    def _mock_session_returning_status(status: int):
        """Build a 3-layer aiohttp session mock that returns the given status."""
        from unittest.mock import AsyncMock, MagicMock
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.raise_for_status = MagicMock()

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        return mock_session

    # ----- set_btc_dominance_history validation -----------------------

    @pytest.mark.asyncio
    async def test_manual_override_skips_network(self) -> None:
        import pandas as pd
        from unittest.mock import AsyncMock
        client = CoinGeckoClient()
        series = self._make_manual_series([50.0] * 14)
        client.set_btc_dominance_history(series)
        # Patch session-getter to fail loudly if called.
        client._get_session = AsyncMock(side_effect=RuntimeError("net hit"))   # type: ignore[method-assign]
        result = await client.fetch_btc_dominance_history(days=14)
        pd.testing.assert_series_equal(result, series)

    @pytest.mark.asyncio
    async def test_manual_override_is_sticky_after_ttl_window(self) -> None:
        """CR-001/CR-002 remediation: a manually-injected series survives
        TTL expiry and is NOT replaced by the network on subsequent calls."""
        import pandas as pd
        from unittest.mock import AsyncMock
        client = CoinGeckoClient(cache_ttl_seconds=0.001)
        series = self._make_manual_series([55.0] * 14)
        client.set_btc_dominance_history(series)
        # Force TTL to look expired.
        client._btc_dom_history_fetched_at = time.monotonic() - 10
        # If sticky path is honored, the session getter is never called.
        client._get_session = AsyncMock(side_effect=RuntimeError("net hit"))   # type: ignore[method-assign]
        result = await client.fetch_btc_dominance_history(days=14)
        pd.testing.assert_series_equal(result, series)

    @pytest.mark.asyncio
    async def test_set_rejects_non_datetime_index(self) -> None:
        import pandas as pd
        client = CoinGeckoClient()
        bad = pd.Series([50.0, 51.0], index=[0, 1], dtype="float64")
        with pytest.raises(ValueError, match="DatetimeIndex"):
            client.set_btc_dominance_history(bad)

    @pytest.mark.asyncio
    async def test_set_rejects_tz_naive_index(self) -> None:
        import pandas as pd
        client = CoinGeckoClient()
        idx = pd.date_range("2024-01-01", periods=3, freq="D")   # no tz
        bad = pd.Series([50.0, 51.0, 52.0], index=idx, dtype="float64")
        with pytest.raises(ValueError, match="tz-aware"):
            client.set_btc_dominance_history(bad)

    @pytest.mark.asyncio
    async def test_set_rejects_non_monotonic_index(self) -> None:
        import pandas as pd
        client = CoinGeckoClient()
        idx = pd.DatetimeIndex(
            ["2024-01-03", "2024-01-01", "2024-01-02"], tz="UTC",
        )
        bad = pd.Series([50.0, 51.0, 52.0], index=idx, dtype="float64")
        with pytest.raises(ValueError, match="monotonic"):
            client.set_btc_dominance_history(bad)

    # ----- free-tier degradation --------------------------------------

    @pytest.mark.asyncio
    async def test_pro_endpoint_401_returns_empty_series(self) -> None:
        from unittest.mock import AsyncMock
        client = CoinGeckoClient()
        client._get_session = AsyncMock(   # type: ignore[method-assign]
            return_value=self._mock_session_returning_status(401),
        )
        result = await client.fetch_btc_dominance_history(days=30)
        assert result.empty
        assert result.dtype == "float64"
        assert result.index.tz is not None

    @pytest.mark.asyncio
    async def test_pro_endpoint_404_logs_warning(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        client = CoinGeckoClient()
        warn_spy = MagicMock()
        client._log.warning = warn_spy   # type: ignore[method-assign]
        client._get_session = AsyncMock(   # type: ignore[method-assign]
            return_value=self._mock_session_returning_status(404),
        )
        await client.fetch_btc_dominance_history(days=30)
        events = [c.args[0] for c in warn_spy.call_args_list if c.args]
        assert "coingecko_client.btc_dom_history_pro_only" in events

    # ----- _parse_dom_history success paths ---------------------------

    @pytest.mark.asyncio
    async def test_parse_dom_history_extracts_btc_dominance_series(self) -> None:
        payload = {
            "market_cap_chart": {
                "btc_dominance": [
                    [1_704_067_200_000, 52.0],
                    [1_704_153_600_000, 53.0],
                    [1_704_240_000_000, 54.0],
                ],
            },
        }
        series = CoinGeckoClient._parse_dom_history(payload)
        assert series.dtype == "float64"
        assert len(series) == 3
        assert series.index.is_monotonic_increasing
        assert series.index.tz is not None
        assert float(series.iloc[0]) == 52.0
        assert float(series.iloc[-1]) == 54.0

    @pytest.mark.asyncio
    async def test_parse_dom_history_derives_from_btc_total_when_absent(self) -> None:
        payload = {
            "market_cap_chart": {
                "market_cap": [
                    [1_704_067_200_000, 2_000_000_000_000.0],
                    [1_704_153_600_000, 2_100_000_000_000.0],
                ],
                "btc_market_cap": [
                    [1_704_067_200_000, 1_000_000_000_000.0],
                    [1_704_153_600_000, 1_100_000_000_000.0],
                ],
            },
        }
        series = CoinGeckoClient._parse_dom_history(payload)
        assert len(series) == 2
        assert float(series.iloc[0]) == pytest.approx(50.0, abs=1e-6)
        assert float(series.iloc[1]) == pytest.approx(
            (1.1 / 2.1) * 100.0, abs=1e-6,
        )

    @pytest.mark.asyncio
    async def test_empty_dom_series_shape(self) -> None:
        s = CoinGeckoClient._empty_dom_series()
        assert s.empty
        assert s.dtype == "float64"
        assert s.index.tz is not None
        assert s.name == "btc_dominance"
