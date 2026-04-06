"""
tests/unit/test_macro_data.py
------------------------------
Unit tests for the FRED macro-economic data client.

Covers:
- MacroSnapshot Pydantic validation (optional fields, frozen)
- FREDClient cache hit / miss / stale fallback / series failure degradation
- FREDClient _fetch_series: normal value, missing sentinel, network error
- Module-level singleton set/get helpers
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.macro_data import (
    FREDClient,
    MacroSnapshot,
    get_global_client,
    set_global_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    fed_funds_rate: float | None = 5.33,
    yield_curve_spread: float | None = -0.25,
) -> MacroSnapshot:
    return MacroSnapshot(
        fed_funds_rate=fed_funds_rate,
        yield_curve_spread=yield_curve_spread,
        timestamp=datetime.now(UTC),
    )


def _make_fred_response(series_id: str, value: str = "5.33") -> dict:
    return {
        "observations": [
            {
                "date": "2025-01-01",
                "value": value,
                "realtime_start": "2025-01-01",
                "realtime_end": "2025-01-01",
            }
        ]
    }


# ---------------------------------------------------------------------------
# TestMacroSnapshot
# ---------------------------------------------------------------------------

class TestMacroSnapshot:
    def test_all_fields_populated(self) -> None:
        snap = _make_snapshot(fed_funds_rate=5.33, yield_curve_spread=-0.25)
        assert snap.fed_funds_rate == 5.33
        assert snap.yield_curve_spread == -0.25

    def test_both_fields_optional(self) -> None:
        snap = MacroSnapshot(timestamp=datetime.now(UTC))
        assert snap.fed_funds_rate is None
        assert snap.yield_curve_spread is None

    def test_fed_funds_rate_only(self) -> None:
        snap = MacroSnapshot(fed_funds_rate=4.5, timestamp=datetime.now(UTC))
        assert snap.fed_funds_rate == 4.5
        assert snap.yield_curve_spread is None

    def test_yield_spread_only(self) -> None:
        snap = MacroSnapshot(yield_curve_spread=1.2, timestamp=datetime.now(UTC))
        assert snap.yield_curve_spread == 1.2
        assert snap.fed_funds_rate is None

    def test_frozen(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(Exception):
            snap.fed_funds_rate = 99.0  # type: ignore[misc]

    def test_negative_yield_spread_accepted(self) -> None:
        snap = _make_snapshot(yield_curve_spread=-1.5)
        assert snap.yield_curve_spread == -1.5


# ---------------------------------------------------------------------------
# TestFREDClientCache
# ---------------------------------------------------------------------------

class TestFREDClientCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self) -> None:
        client = FREDClient(api_key="test_key", cache_ttl_seconds=3600)
        cached_snap = _make_snapshot()
        client._latest_cache = (cached_snap, time.monotonic())

        snap = await client.get_latest()
        assert snap is cached_snap

    @pytest.mark.asyncio
    async def test_stale_cache_fallback_when_outer_raises(self) -> None:
        """
        Verify stale cache is returned when the outer try block raises.

        _fetch_series handles per-series failures internally and returns None.
        To trigger the outer except block in get_latest(), we need to patch
        _fetch_series itself to raise unconditionally — simulating an
        unexpected error at the orchestration level (e.g. OS-level crash).
        """
        client = FREDClient(api_key="test_key", cache_ttl_seconds=0.001)
        stale_snap = _make_snapshot(fed_funds_rate=4.5)
        client._latest_cache = (stale_snap, time.monotonic() - 10)

        # Patch _fetch_series at the instance level to raise — this simulates
        # a catastrophic failure that bypasses the per-series error handling.
        async def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("catastrophic failure")

        client._fetch_series = _raise  # type: ignore[method-assign]

        snap = await client.get_latest()
        assert snap is stale_snap

    @pytest.mark.asyncio
    async def test_all_series_fail_returns_partial_snapshot(self) -> None:
        """
        When all individual series fetches fail, get_latest() still returns a
        MacroSnapshot with all-None fields (graceful degradation).

        FREDClient's design: _fetch_series catches per-series errors and
        returns None. The outer get_latest() constructs a snapshot from
        whatever data is available. This is intentional — a partial snapshot
        is more useful than no snapshot at all.
        """
        client = FREDClient(api_key="test_key", cache_ttl_seconds=60)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
        client._session = mock_session

        snap = await client.get_latest()
        # Returns partial snapshot (not None) — graceful degradation
        assert snap is not None
        assert snap.fed_funds_rate is None
        assert snap.yield_curve_spread is None

    def test_cached_value_property_none_initially(self) -> None:
        client = FREDClient(api_key="test_key")
        assert client.cached_value is None

    def test_cached_value_property_returns_snapshot(self) -> None:
        client = FREDClient(api_key="test_key")
        snap = _make_snapshot()
        client._latest_cache = (snap, time.monotonic())
        assert client.cached_value is snap


# ---------------------------------------------------------------------------
# TestFREDClientFetch
# ---------------------------------------------------------------------------

class TestFREDClientFetch:
    @pytest.mark.asyncio
    async def test_fetch_series_normal_value(self) -> None:
        client = FREDClient(api_key="test_key")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=_make_fred_response("FEDFUNDS", "5.33"))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        client._session = mock_session

        value = await client._fetch_series("FEDFUNDS")
        assert value == pytest.approx(5.33)

    @pytest.mark.asyncio
    async def test_fetch_series_missing_sentinel_returns_none(self) -> None:
        client = FREDClient(api_key="test_key")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=_make_fred_response("T10Y2Y", "."))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        client._session = mock_session

        value = await client._fetch_series("T10Y2Y")
        assert value is None

    @pytest.mark.asyncio
    async def test_fetch_series_empty_observations_returns_none(self) -> None:
        client = FREDClient(api_key="test_key")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"observations": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        client._session = mock_session

        value = await client._fetch_series("FEDFUNDS")
        assert value is None

    @pytest.mark.asyncio
    async def test_fetch_series_network_error_returns_none(self) -> None:
        client = FREDClient(api_key="test_key")

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("timeout"))
        client._session = mock_session

        value = await client._fetch_series("FEDFUNDS")
        assert value is None


# ---------------------------------------------------------------------------
# TestFREDClientSingleton
# ---------------------------------------------------------------------------

class TestFREDClientSingleton:
    def test_get_returns_none_before_set(self) -> None:
        import data.macro_data as md
        original = md._global_client
        md._global_client = None
        try:
            assert get_global_client() is None
        finally:
            md._global_client = original

    def test_set_and_get_round_trip(self) -> None:
        import data.macro_data as md
        original = md._global_client
        client = FREDClient(api_key="test_key")
        try:
            set_global_client(client)
            assert get_global_client() is client
        finally:
            md._global_client = original
