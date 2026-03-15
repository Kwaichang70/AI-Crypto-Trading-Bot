"""
tests/unit/test_sentiment.py
------------------------------
Unit tests for Fear & Greed Index client (Sprint 32).

Module under test
-----------------
packages/data/sentiment.py

  FearGreedSnapshot   -- Pydantic model with timestamp coercion and regime_boost
  FearGreedClient     -- Async HTTP client with 6-hour in-process cache
  set_global_client   -- Module-level singleton setter
  get_global_client   -- Module-level singleton getter

Coverage groups (14 tests)
---------------------------
TestFearGreedSnapshot       (5) -- model validation, coercion, bounds, regime_boost bands
TestFearGreedClientCache    (5) -- cache hit/miss/stale/fail-stale/fail-empty-None
TestFearGreedClientFetch    (4) -- success parse, missing key, empty data, global singleton

Design notes
------------
- asyncio_mode = "auto" in pyproject.toml: async tests are auto-detected.
- _fetch logic is tested by patching FearGreedClient._get_session to return a
  mock aiohttp session whose context-manager .get() returns controlled responses.
- Cache TTL is set to a very small value (0.001 s) or manually invalidated by
  replacing _latest_cache with a stale timestamp to simulate expiry.
- Module-level singleton tests must reset _global_client after each test to
  avoid cross-test contamination.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data.sentiment as sentiment_module
from data.sentiment import (
    FearGreedClient,
    FearGreedSnapshot,
    get_global_client,
    set_global_client,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _snapshot(value: int = 25) -> FearGreedSnapshot:
    """Factory: create a FearGreedSnapshot with minimal required fields."""
    return FearGreedSnapshot(
        value=value,
        classification="Fear",
        timestamp=datetime(2024, 3, 10, 12, 0, 0, tzinfo=UTC),
    )


def _make_mock_response(json_data: dict) -> MagicMock:
    """Build a fake aiohttp response that returns json_data on .json()."""
    response = AsyncMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=json_data)
    # context manager support: __aenter__ returns self
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


def _make_mock_session(response: MagicMock) -> MagicMock:
    """Build a fake aiohttp session whose .get() yields the given response."""
    session = MagicMock()
    session.closed = False
    session.get = MagicMock(return_value=response)
    return session


# ===========================================================================
# TestFearGreedSnapshot
# ===========================================================================


class TestFearGreedSnapshot:
    """Tests for the FearGreedSnapshot Pydantic model."""

    def test_valid_snapshot(self) -> None:
        """A snapshot with integer value, string classification, and UTC datetime is valid."""
        snap = FearGreedSnapshot(
            value=25,
            classification="Extreme Fear",
            timestamp=datetime(2024, 3, 10, 12, 0, 0, tzinfo=UTC),
        )
        assert snap.value == 25
        assert snap.classification == "Extreme Fear"
        assert snap.timestamp.tzinfo is not None

    def test_unix_timestamp_coercion_string(self) -> None:
        """A string Unix timestamp is coerced to a timezone-aware datetime."""
        snap = FearGreedSnapshot(
            value=50,
            classification="Neutral",
            timestamp="1710000000",
        )
        assert isinstance(snap.timestamp, datetime)
        assert snap.timestamp.tzinfo is not None
        assert snap.timestamp == datetime.fromtimestamp(1710000000, tz=UTC)

    def test_unix_timestamp_coercion_int(self) -> None:
        """An integer Unix timestamp is coerced to a timezone-aware datetime."""
        snap = FearGreedSnapshot(
            value=50,
            classification="Neutral",
            timestamp=1710000000,
        )
        assert isinstance(snap.timestamp, datetime)
        assert snap.timestamp == datetime.fromtimestamp(1710000000, tz=UTC)

    def test_value_bounds_below_zero_raises(self) -> None:
        """value=-1 must raise a ValidationError (ge=0 constraint)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FearGreedSnapshot(
                value=-1,
                classification="Invalid",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_value_bounds_above_100_raises(self) -> None:
        """value=101 must raise a ValidationError (le=100 constraint)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FearGreedSnapshot(
                value=101,
                classification="Invalid",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )

    @pytest.mark.parametrize(
        "value,expected_boost",
        [
            (10, 0.15),    # Extreme Fear (0-24)
            (24, 0.15),    # Extreme Fear boundary
            (25, 0.05),    # Fear (25-44)
            (44, 0.05),    # Fear boundary
            (45, 0.0),     # Neutral (45-55)
            (55, 0.0),     # Neutral boundary
            (56, 0.05),    # Greed (56-75)
            (75, 0.05),    # Greed boundary
            (76, -0.10),   # Extreme Greed (76-100)
            (85, -0.10),   # Extreme Greed
            (100, -0.10),  # Max value
        ],
    )
    def test_regime_boost_all_bands(self, value: int, expected_boost: float) -> None:
        """regime_boost maps each value band to the correct contrarian modifier."""
        snap = FearGreedSnapshot(
            value=value,
            classification="Test",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert snap.regime_boost == expected_boost


# ===========================================================================
# TestFearGreedClientCache
# ===========================================================================


class TestFearGreedClientCache:
    """Tests for the FearGreedClient caching behaviour."""

    async def test_get_latest_returns_parsed_snapshot(self) -> None:
        """get_latest() fetches and returns a FearGreedSnapshot."""
        client = FearGreedClient(cache_ttl_seconds=3600)
        snap = _snapshot(40)

        with patch.object(client, "_get_session") as mock_get_session:
            json_payload = {
                "data": [
                    {"value": "40", "value_classification": "Fear", "timestamp": "1710000000"}
                ]
            }
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            result = await client.get_latest()

        assert result is not None
        assert result.value == 40
        assert result.classification == "Fear"

    async def test_cache_hit_no_refetch(self) -> None:
        """A second get_latest() within TTL must use cache, not re-fetch."""
        client = FearGreedClient(cache_ttl_seconds=3600)

        with patch.object(client, "_get_session") as mock_get_session:
            json_payload = {
                "data": [
                    {"value": "30", "value_classification": "Fear", "timestamp": "1710000000"}
                ]
            }
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            first = await client.get_latest()
            second = await client.get_latest()

        # _get_session should only be called once (second call hits cache)
        assert mock_get_session.call_count == 1
        assert first is second

    async def test_stale_cache_refetch(self) -> None:
        """With zero TTL, each get_latest() call must re-fetch."""
        client = FearGreedClient(cache_ttl_seconds=0)

        with patch.object(client, "_get_session") as mock_get_session:
            json_payload = {
                "data": [
                    {"value": "20", "value_classification": "Extreme Fear", "timestamp": "1710000000"}
                ]
            }
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            await client.get_latest()
            await client.get_latest()

        # Both calls should trigger a fetch
        assert mock_get_session.call_count == 2

    async def test_failed_fetch_returns_stale_cache(self) -> None:
        """When fetch fails and a stale cache entry exists, return the stale snapshot."""
        client = FearGreedClient(cache_ttl_seconds=0)
        stale_snap = _snapshot(55)
        # Pre-fill cache with a very old timestamp so TTL is expired
        client._latest_cache = (stale_snap, 0.0)

        with patch.object(client, "_get_session", side_effect=RuntimeError("network down")):
            result = await client.get_latest()

        assert result is stale_snap

    async def test_failed_fetch_empty_cache_returns_none(self) -> None:
        """When fetch fails and the cache is empty, get_latest() returns None."""
        client = FearGreedClient(cache_ttl_seconds=3600)
        # No cache pre-filled
        assert client._latest_cache is None

        with patch.object(client, "_get_session", side_effect=RuntimeError("network down")):
            result = await client.get_latest()

        assert result is None


# ===========================================================================
# TestFearGreedClientFetch
# ===========================================================================


class TestFearGreedClientFetch:
    """Tests for _parse_snapshot and raw HTTP response handling."""

    async def test_success_parse(self) -> None:
        """A well-formed API response is parsed into a FearGreedSnapshot."""
        client = FearGreedClient(cache_ttl_seconds=3600)

        with patch.object(client, "_get_session") as mock_get_session:
            json_payload = {
                "data": [
                    {
                        "value": "72",
                        "value_classification": "Greed",
                        "timestamp": "1710000000",
                    }
                ]
            }
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            result = await client.get_latest()

        assert result is not None
        assert result.value == 72
        assert result.classification == "Greed"

    async def test_missing_data_key_returns_none(self) -> None:
        """A response missing the 'data' key triggers the exception handler, returns None."""
        client = FearGreedClient(cache_ttl_seconds=3600)

        with patch.object(client, "_get_session") as mock_get_session:
            # No 'data' key — will raise KeyError inside get_latest
            json_payload: dict = {}
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            result = await client.get_latest()

        # Exception caught internally; no cache; returns None
        assert result is None

    async def test_empty_data_array_returns_none(self) -> None:
        """An empty 'data' array causes IndexError, which is caught; returns None."""
        client = FearGreedClient(cache_ttl_seconds=3600)

        with patch.object(client, "_get_session") as mock_get_session:
            json_payload = {"data": []}
            response = _make_mock_response(json_payload)
            mock_session = _make_mock_session(response)
            mock_get_session.return_value = mock_session

            result = await client.get_latest()

        assert result is None

    def test_global_singleton(self) -> None:
        """set_global_client / get_global_client round-trip preserves identity."""
        # Reset before test
        sentiment_module._global_client = None

        client = FearGreedClient()
        set_global_client(client)

        retrieved = get_global_client()
        assert retrieved is client

        # Clean up
        sentiment_module._global_client = None
