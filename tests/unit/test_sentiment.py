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
from datetime import UTC, datetime, timedelta
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


# ===================================================================
# QT-007d (Sprint 46) -- Historical FGI fetcher adapters
# ===================================================================


class TestQT007dFearGreedHistory:
    """``get_history_as_series`` + ``get_value_at_offset`` are pure data
    adapters over the existing ``get_history`` cache.  Tests use a
    MagicMock-driven FearGreedClient so we never hit the real API."""

    async def _make_client_with_history(
        self,
        snapshots: list[FearGreedSnapshot],
    ) -> FearGreedClient:
        from unittest.mock import AsyncMock
        client = FearGreedClient()
        client.get_history = AsyncMock(return_value=snapshots)   # type: ignore[method-assign]
        return client

    def _snapshot(self, value: int, days_ago: int) -> FearGreedSnapshot:
        ts = datetime.now(tz=UTC) - timedelta(days=days_ago)
        return FearGreedSnapshot(
            value=value,
            classification="neutral",
            timestamp=ts,
        )

    # ----- get_history_as_series ----------------------------------------

    async def test_empty_history_returns_empty_series(self) -> None:
        client = await self._make_client_with_history([])
        series = await client.get_history_as_series()
        assert series.empty
        assert series.dtype == "int64"
        assert series.index.tz is not None   # UTC index even when empty

    async def test_series_is_sorted_ascending_by_timestamp(self) -> None:
        # Snapshots delivered newest-first as alternative.me does.
        snapshots = [
            self._snapshot(value=80, days_ago=1),
            self._snapshot(value=70, days_ago=2),
            self._snapshot(value=60, days_ago=3),
        ]
        client = await self._make_client_with_history(snapshots)
        series = await client.get_history_as_series()
        assert series.index.is_monotonic_increasing
        assert int(series.iloc[0]) == 60       # oldest first
        assert int(series.iloc[-1]) == 80      # newest last

    async def test_series_index_is_utc_aware(self) -> None:
        client = await self._make_client_with_history(
            [self._snapshot(value=50, days_ago=1)],
        )
        series = await client.get_history_as_series()
        assert series.index.tz is not None
        assert str(series.index.tz) == "UTC"

    async def test_series_values_are_int64(self) -> None:
        client = await self._make_client_with_history(
            [self._snapshot(value=42, days_ago=1)],
        )
        series = await client.get_history_as_series()
        assert series.dtype == "int64"

    # ----- get_value_at_offset ------------------------------------------

    async def test_value_at_offset_returns_matched_value(self) -> None:
        snapshots = [
            self._snapshot(value=30, days_ago=7),
            self._snapshot(value=50, days_ago=3),
            self._snapshot(value=80, days_ago=1),
        ]
        client = await self._make_client_with_history(snapshots)
        assert await client.get_value_at_offset(days_ago=7) == 30

    async def test_value_at_offset_picks_closest_match(self) -> None:
        # Snapshots at 6d and 9d ago -- closest to 7d-ago is 6d (1d off).
        snapshots = [
            self._snapshot(value=40, days_ago=6),
            self._snapshot(value=20, days_ago=9),
        ]
        client = await self._make_client_with_history(snapshots)
        assert await client.get_value_at_offset(days_ago=7) == 40

    async def test_value_at_offset_returns_none_when_no_close_match(self) -> None:
        # Only snapshot is 7 days off the 7d-ago target -> exceeds 2d
        # tolerance -> None.
        snapshots = [self._snapshot(value=99, days_ago=14)]
        client = await self._make_client_with_history(snapshots)
        assert await client.get_value_at_offset(days_ago=7) is None

    async def test_value_at_offset_returns_none_on_empty_history(self) -> None:
        client = await self._make_client_with_history([])
        assert await client.get_value_at_offset(days_ago=7) is None

    async def test_value_at_offset_rejects_zero(self) -> None:
        client = await self._make_client_with_history([])
        with pytest.raises(ValueError, match="days_ago must be >= 1"):
            await client.get_value_at_offset(days_ago=0)

    async def test_value_at_offset_rejects_negative(self) -> None:
        client = await self._make_client_with_history([])
        with pytest.raises(ValueError, match="days_ago must be >= 1"):
            await client.get_value_at_offset(days_ago=-3)

    async def test_value_at_offset_warns_when_days_exceeds_api_ceiling(self) -> None:
        """CR-005 remediation: a WARNING log must surface the API limit
        so callers can distinguish capability from real gaps."""
        from unittest.mock import MagicMock
        client = FearGreedClient()
        # Stub history with a usable nearby value so the warn path executes
        # even though the result itself is None (no match within tolerance).
        from unittest.mock import AsyncMock
        client.get_history = AsyncMock(return_value=[])   # type: ignore[method-assign]
        warn_spy = MagicMock()
        client._log.warning = warn_spy                    # type: ignore[method-assign]
        await client.get_value_at_offset(days_ago=60)
        # Find the API-ceiling warning among the calls (must be present).
        events = [c.args[0] for c in warn_spy.call_args_list if c.args]
        assert "fear_greed_client.value_at_offset_beyond_api_limit" in events

    async def test_value_at_offset_logs_cache_shortfall(self) -> None:
        """CR-003 remediation: when cache has fewer points than days_ago,
        a DEBUG log records the mismatch so operators can correlate
        silent-None results with cache-limit-misuse."""
        from unittest.mock import MagicMock
        client = await self._make_client_with_history(
            [self._snapshot(value=42, days_ago=1)],   # only 1 point in cache
        )
        debug_spy = MagicMock()
        client._log.debug = debug_spy                  # type: ignore[method-assign]
        result = await client.get_value_at_offset(days_ago=10)
        assert result is None     # no match found
        events = [c.args[0] for c in debug_spy.call_args_list if c.args]
        assert "fear_greed_client.value_at_offset_insufficient_cache" in events


# ===================================================================
# QT-007f-1 (Sprint 46) -- value_at_offset_from_cache sync accessor
# ===================================================================


class TestQT007f1ValueAtOffsetFromCache:
    """Synchronous cache-only accessor for the 7d-ago MTF context field."""

    def _snapshot(self, value: int, days_ago: int) -> FearGreedSnapshot:
        ts = datetime.now(tz=UTC) - timedelta(days=days_ago)
        return FearGreedSnapshot(
            value=value, classification="neutral", timestamp=ts,
        )

    def test_returns_none_when_history_cache_is_unset(self) -> None:
        client = FearGreedClient()
        assert client.value_at_offset_from_cache(days_ago=7) is None

    def test_returns_none_when_history_cache_is_empty(self) -> None:
        client = FearGreedClient()
        client._history_cache = ([], time.monotonic())
        assert client.value_at_offset_from_cache(days_ago=7) is None

    def test_returns_value_when_cache_has_matching_snapshot(self) -> None:
        client = FearGreedClient()
        client._history_cache = (
            [self._snapshot(value=30, days_ago=7)],
            time.monotonic(),
        )
        assert client.value_at_offset_from_cache(days_ago=7) == 30

    def test_returns_closest_match_within_tolerance(self) -> None:
        client = FearGreedClient()
        client._history_cache = (
            [
                self._snapshot(value=40, days_ago=6),
                self._snapshot(value=20, days_ago=9),
            ],
            time.monotonic(),
        )
        assert client.value_at_offset_from_cache(days_ago=7) == 40

    def test_returns_none_beyond_2_day_tolerance(self) -> None:
        client = FearGreedClient()
        client._history_cache = (
            [self._snapshot(value=99, days_ago=14)],
            time.monotonic(),
        )
        assert client.value_at_offset_from_cache(days_ago=7) is None

    def test_rejects_zero_days_ago(self) -> None:
        client = FearGreedClient()
        with pytest.raises(ValueError, match="days_ago must be >= 1"):
            client.value_at_offset_from_cache(days_ago=0)
