"""
tests/unit/test_ccxt_market_data.py
-------------------------------------
Unit tests for CCXTMarketDataService.

Module under test
-----------------
    packages/data/services/ccxt_market_data.py

Coverage groups (64 tests)
--------------------------
1.  TestConstructorAndConnect       -- constructor, sandbox, connect lifecycle (11 tests)
2.  TestFetchOHLCVBasic             -- basic fetch, field mapping, validation (8 tests)
3.  TestFetchOHLCVCache             -- L1 cache hit/miss/TTL/invalidate (7 tests)
4.  TestFetchOHLCVRetry             -- retry policy, backoff, non-retryable errors (10 tests)
5.  TestFetchOHLCVRangePagination   -- pagination, dedup, boundary, short-page (10 tests)
6.  TestGetLatestBar                -- closed-candle selection edge cases (4 tests)
7.  TestSupportedSymbolsTimeframes  -- get_supported_* after connect (5 tests)
8.  TestThrottleAndBackoff          -- inter-request spacing, backoff formula (4 tests)
9.  TestCacheInvalidation           -- invalidate_cache API (4 tests)
10. TestStaticHelpers               -- static/class utility methods (1 test)

Design notes
------------
- asyncio_mode = "auto" is set in pyproject.toml; async tests auto-detect.
- The module under test imports ``ccxt.async_support as ccxt_async`` at the top level.
  We patch the module-level name ``data.services.ccxt_market_data.ccxt_async`` so that
  ``getattr(ccxt_async, exchange_id)`` — the lookup the constructor performs —
  resolves to a mock class that returns our controlled exchange instance.
- _validate_symbol and _validate_timeframe SKIP validation when _markets /
  _exchange_timeframes are empty (before connect()). Tests that need validation to fire
  must call connect() or manually populate those attributes.
- CCXT exception classes live on ``ccxt.async_support`` (aliased as ccxt_async). We import
  the real exception types for use as side_effect values and to verify isinstance checks.
- IMPORTANT: ccxt.ExchangeNotAvailable is a subclass of ccxt.NetworkError. The implementation
  catches ExchangeNotAvailable BEFORE NetworkError in its except chain, so it raises
  MarketDataError immediately on the first attempt without any retry. Tests reflect this.
- Cache TTL expiry tests use direct cache injection (expiry=0.0) rather than patching
  time.monotonic, which is called at many different points during fetch (throttle, cache_set,
  cache_get). Direct injection is deterministic and avoids call-count fragility.
- For throttle tests, time is patched at the module level so both _throttle() and
  _last_request_time assignment see the mocked values.
- All OHLCVBar raw data must satisfy: low <= open/close <= high. Default _raw_bar values
  use o=100.0, h=105.0, l=95.0, c=102.0 which satisfies all constraints.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt.async_support as ccxt_async
import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from data.market_data import (
    DataNotAvailableError,
    MarketDataError,
    RateLimitError,
)
from data.services.ccxt_market_data import (
    CCXTMarketDataService,
    _BACKOFF_BASE,
    _BACKOFF_CAP,
    _MAX_RETRIES,
)

# ---------------------------------------------------------------------------
# Helpers & factories
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_ALT_SYMBOL = "ETH/USDT"

# Epoch ms reference for deterministic bars
_BASE_TS_MS: int = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
_ONE_HOUR_MS: int = 60 * 60 * 1_000


def _raw_bar(
    ts_ms: int,
    o: float = 100.0,
    h: float = 105.0,
    l: float = 95.0,  # noqa: E741
    c: float = 102.0,
    v: float = 1_000.0,
) -> list[int | float]:
    """
    Create a raw CCXT OHLCV bar: [timestamp_ms, open, high, low, close, volume].

    Default values satisfy OHLCVBar constraints: low(95) <= open(100)/close(102) <= high(105).
    When supplying custom c, ensure l <= c <= h to pass Pydantic validation.
    """
    return [ts_ms, o, h, l, c, v]


def _raw_bars(
    count: int,
    start_ts_ms: int = _BASE_TS_MS,
    step_ms: int = _ONE_HOUR_MS,
) -> list[list[int | float]]:
    """Create ``count`` sequential raw bars spaced ``step_ms`` apart."""
    return [_raw_bar(start_ts_ms + i * step_ms) for i in range(count)]


def _make_mock_exchange(
    markets: dict[str, Any] | None = None,
    timeframes: dict[str, str] | None = None,
    raw_ohlcv: list[list[int | float]] | None = None,
) -> MagicMock:
    """Build a fully-configured mock CCXT exchange instance."""
    if markets is None:
        markets = {_SYMBOL: {}, _ALT_SYMBOL: {}}
    if timeframes is None:
        timeframes = {"1m": "1 minute", "1h": "1 hour", "1d": "1 day"}
    if raw_ohlcv is None:
        raw_ohlcv = []

    mock_ex = MagicMock()
    mock_ex.load_markets = AsyncMock(return_value=markets)
    mock_ex.fetch_ohlcv = AsyncMock(return_value=raw_ohlcv)
    mock_ex.close = AsyncMock()
    mock_ex.set_sandbox_mode = MagicMock()
    mock_ex.timeframes = timeframes
    return mock_ex


def _make_service(
    exchange_id: str = "binance",
    cache_ttl: int = 60,
    page_limit: int = 500,
    sandbox: bool = False,
    mock_exchange: MagicMock | None = None,
) -> tuple[CCXTMarketDataService, MagicMock]:
    """
    Create a CCXTMarketDataService with its internal CCXT exchange replaced
    by a MagicMock.  Returns ``(service, mock_exchange)``.

    The patch targets ``data.services.ccxt_market_data.ccxt_async`` so that
    ``getattr(ccxt_async, exchange_id)`` — the lookup the constructor performs —
    resolves to a mock class that returns our controlled exchange instance.

    After construction the patch context exits, but the real exception classes
    are correctly bound because we set them on the patched module during
    construction.  The service's except blocks use the exception classes that
    were bound at import time (via ``import ccxt.async_support as ccxt_async``
    at module level inside ccxt_market_data.py), so they always work.
    """
    if mock_exchange is None:
        mock_exchange = _make_mock_exchange()

    mock_cls = MagicMock(return_value=mock_exchange)

    with patch("data.services.ccxt_market_data.ccxt_async") as patched_ccxt:
        # Make getattr(ccxt_async, exchange_id) return our mock class.
        setattr(patched_ccxt, exchange_id, mock_cls)
        # Exception classes must resolve correctly for except blocks in the module.
        patched_ccxt.AuthenticationError = ccxt_async.AuthenticationError
        patched_ccxt.NetworkError = ccxt_async.NetworkError
        patched_ccxt.ExchangeError = ccxt_async.ExchangeError
        patched_ccxt.RateLimitExceeded = ccxt_async.RateLimitExceeded
        patched_ccxt.BadSymbol = ccxt_async.BadSymbol
        patched_ccxt.BadRequest = ccxt_async.BadRequest
        patched_ccxt.ExchangeNotAvailable = ccxt_async.ExchangeNotAvailable

        service = CCXTMarketDataService(
            exchange_id=exchange_id,
            cache_ttl_seconds=cache_ttl,
            page_limit=page_limit,
            sandbox=sandbox,
        )

    # Inject mock exchange directly so tests can configure it further.
    service._exchange = mock_exchange
    return service, mock_exchange


# ---------------------------------------------------------------------------
# 1. TestConstructorAndConnect
# ---------------------------------------------------------------------------


class TestConstructorAndConnect:
    """Constructor validation and connect() lifecycle — 11 tests."""

    def test_constructor_default_exchange_id(self) -> None:
        """Service stores the exchange_id correctly after construction."""
        service, _ = _make_service(exchange_id="binance")
        assert service.exchange_id == "binance"

    def test_constructor_repr_contains_exchange_id(self) -> None:
        """__repr__ contains the exchange ID, cache TTL, and page limit."""
        service, _ = _make_service(exchange_id="binance", cache_ttl=30, page_limit=200)
        r = repr(service)
        assert "binance" in r
        assert "30" in r
        assert "200" in r

    def test_constructor_sandbox_calls_set_sandbox_mode(self) -> None:
        """When sandbox=True the constructor calls set_sandbox_mode(True) on the exchange."""
        mock_ex = _make_mock_exchange()
        _make_service(sandbox=True, mock_exchange=mock_ex)
        mock_ex.set_sandbox_mode.assert_called_once_with(True)

    def test_constructor_no_sandbox_does_not_call_set_sandbox_mode(self) -> None:
        """When sandbox=False the constructor must NOT call set_sandbox_mode."""
        mock_ex = _make_mock_exchange()
        _make_service(sandbox=False, mock_exchange=mock_ex)
        mock_ex.set_sandbox_mode.assert_not_called()

    async def test_connect_calls_load_markets(self) -> None:
        """connect() must call exchange.load_markets() exactly once."""
        service, mock_ex = _make_service()
        await service.connect()
        mock_ex.load_markets.assert_called_once()

    async def test_connect_populates_markets_and_timeframes(self) -> None:
        """connect() populates _markets from load_markets and _exchange_timeframes from exchange.timeframes."""
        service, _ = _make_service()
        assert not service._markets
        await service.connect()
        assert _SYMBOL in service._markets
        assert "1h" in service._exchange_timeframes

    async def test_connect_authentication_error_raises_market_data_error(self) -> None:
        """connect() wraps ccxt.AuthenticationError as MarketDataError."""
        service, mock_ex = _make_service()
        mock_ex.load_markets = AsyncMock(
            side_effect=ccxt_async.AuthenticationError("bad key")
        )
        with pytest.raises(MarketDataError, match="Authentication failed"):
            await service.connect()

    async def test_connect_network_error_raises_market_data_error(self) -> None:
        """connect() wraps ccxt.NetworkError as MarketDataError."""
        service, mock_ex = _make_service()
        mock_ex.load_markets = AsyncMock(
            side_effect=ccxt_async.NetworkError("timeout")
        )
        with pytest.raises(MarketDataError, match="Network error"):
            await service.connect()

    async def test_connect_exchange_error_raises_market_data_error(self) -> None:
        """connect() wraps generic ccxt.ExchangeError as MarketDataError."""
        service, mock_ex = _make_service()
        mock_ex.load_markets = AsyncMock(
            side_effect=ccxt_async.ExchangeError("maintenance")
        )
        with pytest.raises(MarketDataError):
            await service.connect()

    async def test_close_calls_exchange_close(self) -> None:
        """close() must delegate to exchange.close()."""
        service, mock_ex = _make_service()
        await service.close()
        mock_ex.close.assert_called_once()

    async def test_connect_is_idempotent(self) -> None:
        """connect() is safe to call multiple times; each call re-loads markets."""
        service, mock_ex = _make_service()
        await service.connect()
        await service.connect()
        assert mock_ex.load_markets.call_count == 2
        assert _SYMBOL in service._markets

    async def test_close_suppresses_exception_on_cleanup(self) -> None:
        """close() does not propagate exceptions from exchange.close()."""
        service, mock_ex = _make_service()
        mock_ex.close = AsyncMock(side_effect=Exception("connection reset"))
        await service.close()  # must not raise


# ---------------------------------------------------------------------------
# 2. TestFetchOHLCVBasic
# ---------------------------------------------------------------------------


class TestFetchOHLCVBasic:
    """Basic fetch_ohlcv behaviour — 8 tests."""

    async def test_fetch_returns_list_of_ohlcv_bars(self) -> None:
        """fetch_ohlcv returns a non-empty list of OHLCVBar when exchange returns data."""
        raw = _raw_bars(3)
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)
        # No connect() — validation is skipped when _markets is empty.
        result = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert len(result) == 3
        assert all(isinstance(b, OHLCVBar) for b in result)

    async def test_fetch_ohlcv_bar_fields_match_raw_data(self) -> None:
        """OHLCVBar fields are populated correctly from the raw CCXT row."""
        ts_ms = _BASE_TS_MS
        raw = [_raw_bar(ts_ms, o=200.0, h=210.0, l=190.0, c=205.0, v=50.0)]
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)
        bars = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        bar = bars[0]
        assert bar.symbol == _SYMBOL
        assert bar.timeframe == TimeFrame.ONE_HOUR
        assert bar.open == Decimal("200.0")
        assert bar.high == Decimal("210.0")
        assert bar.low == Decimal("190.0")
        assert bar.close == Decimal("205.0")
        assert bar.volume == Decimal("50.0")
        expected_ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
        assert bar.timestamp == expected_ts

    async def test_fetch_validates_symbol_after_connect(self) -> None:
        """fetch_ohlcv raises DataNotAvailableError for unknown symbol once markets are loaded."""
        service, mock_ex = _make_service()
        await service.connect()
        with pytest.raises(DataNotAvailableError, match="UNKNOWN/USDT"):
            await service.fetch_ohlcv("UNKNOWN/USDT", TimeFrame.ONE_HOUR)

    async def test_fetch_validates_timeframe_after_connect(self) -> None:
        """fetch_ohlcv raises DataNotAvailableError for unsupported timeframe once markets are loaded."""
        service, mock_ex = _make_service()
        # Default mock exchange only supports 1m, 1h, 1d — not 4h.
        await service.connect()
        with pytest.raises(DataNotAvailableError, match="4h"):
            await service.fetch_ohlcv(_SYMBOL, TimeFrame.FOUR_HOURS)

    async def test_fetch_skips_symbol_validation_before_connect(self) -> None:
        """Before connect() _markets is empty so symbol validation is skipped — no error raised."""
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])
        # Should NOT raise even though _markets is empty.
        result = await service.fetch_ohlcv("UNKNOWN/USDT", TimeFrame.ONE_HOUR)
        assert result == []

    async def test_fetch_empty_exchange_response_returns_empty_list(self) -> None:
        """fetch_ohlcv returns an empty list when the exchange returns no bars."""
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])
        result = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert result == []

    async def test_fetch_respects_limit_parameter(self) -> None:
        """The limit parameter is forwarded verbatim to exchange.fetch_ohlcv."""
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR, limit=100)
        # Verify limit=100 is forwarded as a keyword argument.
        call_args = mock_ex.fetch_ohlcv.call_args
        assert call_args.kwargs.get("limit") == 100

    async def test_fetch_result_is_sorted_ascending_by_timestamp(self) -> None:
        """fetch_ohlcv sorts bars ascending by timestamp even when exchange returns them reversed."""
        raw = list(reversed(_raw_bars(5)))  # deliberately reversed
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)
        bars = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 3. TestFetchOHLCVCache
# ---------------------------------------------------------------------------


class TestFetchOHLCVCache:
    """L1 in-process TTL cache — 7 tests."""

    async def test_cache_first_call_is_cache_miss(self) -> None:
        """First fetch_ohlcv call triggers one exchange request (cache miss)."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(2))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 1

    async def test_cache_second_call_is_cache_hit(self) -> None:
        """Second identical call does NOT reach the exchange (cache hit)."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(2))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 1

    async def test_cache_different_args_produce_cache_miss(self) -> None:
        """Calls with different symbols each hit the exchange (distinct cache keys)."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 2

    async def test_cache_expires_after_ttl(self) -> None:
        """When a cache entry has expired (expiry in the past), fetch_ohlcv re-fetches from exchange."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))

        # Populate the cache legitimately on the first call.
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 1

        # Manually expire the cache entry by setting its expiry to 0 (far in the past).
        # The cache key matches what fetch_ohlcv constructs internally.
        cache_key = (_SYMBOL, TimeFrame.ONE_HOUR.value, None, 500)
        assert cache_key in service._cache, "Cache entry missing after first fetch"
        bars_payload, _ = service._cache[cache_key]
        service._cache[cache_key] = (bars_payload, 0.0)  # expiry = epoch 0 → always expired

        # Second call should detect expiry and re-fetch.
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 2

    async def test_cache_ttl_zero_disables_caching(self) -> None:
        """cache_ttl_seconds=0 means every call goes to the exchange — no caching."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 2

    async def test_cache_invalidate_symbol_clears_that_symbols_entries(self) -> None:
        """invalidate_cache(symbol=...) clears entries for that symbol only."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 2

        service.invalidate_cache(symbol=_SYMBOL)

        # BTC re-fetch must go to exchange again.
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 3
        # ETH must still be cached.
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 3

    async def test_cache_invalidate_all_clears_everything(self) -> None:
        """invalidate_cache() with no args clears all cache entries."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 2

        service.invalidate_cache()

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 4


# ---------------------------------------------------------------------------
# 4. TestFetchOHLCVRetry
# ---------------------------------------------------------------------------


class TestFetchOHLCVRetry:
    """Retry policy and exception mapping — 8 tests."""

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_rate_limit_exceeded_then_success(
        self, mock_sleep: AsyncMock
    ) -> None:
        """RateLimitExceeded on attempt 0, success on attempt 1 — returns bars."""
        service, mock_ex = _make_service(cache_ttl=0)
        raw = _raw_bars(1)
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.RateLimitExceeded("429"), raw]
        )
        result = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert len(result) == 1
        assert mock_sleep.called

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_rate_limit_exhausts_max_retries_raises_rate_limit_error(
        self, mock_sleep: AsyncMock
    ) -> None:
        """RateLimitExceeded on every attempt raises RateLimitError after _MAX_RETRIES."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=ccxt_async.RateLimitExceeded("429")
        )
        with pytest.raises(RateLimitError):
            await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        # Loop runs while attempt <= _MAX_RETRIES → _MAX_RETRIES+1 total attempts.
        assert mock_ex.fetch_ohlcv.call_count == _MAX_RETRIES + 1

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_network_error_succeeds_on_second_attempt(
        self, mock_sleep: AsyncMock
    ) -> None:
        """NetworkError on attempt 0, success on attempt 1."""
        service, mock_ex = _make_service(cache_ttl=0)
        raw = _raw_bars(2)
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.NetworkError("connection reset"), raw]
        )
        result = await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert len(result) == 2
        assert mock_sleep.called

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_bad_symbol_not_retried_raises_data_not_available(
        self, mock_sleep: AsyncMock
    ) -> None:
        """BadSymbol is not retried — immediately raises DataNotAvailableError."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=ccxt_async.BadSymbol("unknown symbol")
        )
        with pytest.raises(DataNotAvailableError):
            await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 1
        mock_sleep.assert_not_called()

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_bad_request_not_retried_raises_data_not_available(
        self, mock_sleep: AsyncMock
    ) -> None:
        """BadRequest is not retried — immediately raises DataNotAvailableError."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=ccxt_async.BadRequest("invalid params")
        )
        with pytest.raises(DataNotAvailableError):
            await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        assert mock_ex.fetch_ohlcv.call_count == 1
        mock_sleep.assert_not_called()

    async def test_exchange_not_available_raises_immediately_without_retry(
        self,
    ) -> None:
        """ExchangeNotAvailable raises MarketDataError immediately — no retry."""
        service, mock_ex = _make_service()
        mock_ex.fetch_ohlcv = AsyncMock(
            side_effect=ccxt_async.ExchangeNotAvailable("maintenance"),
        )

        with pytest.raises(MarketDataError, match="currently unavailable"):
            await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR, limit=10)

        assert mock_ex.fetch_ohlcv.call_count == 1

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_header_parsed_from_exception(
        self, mock_sleep: AsyncMock
    ) -> None:
        """Retry-After value embedded in the exception message is used as the sleep duration."""
        service, mock_ex = _make_service(cache_ttl=0)
        # The _extract_retry_after regex: retry.after[\"'\s:=]+(\d+(?:\.\d+)?)
        raw = _raw_bars(1)
        exc = ccxt_async.RateLimitExceeded("retry-after: 30")
        mock_ex.fetch_ohlcv = AsyncMock(side_effect=[exc, raw])

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)

        assert mock_sleep.called
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == pytest.approx(30.0)

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_regex_mismatch_falls_back_to_exponential_backoff(
        self, mock_sleep: AsyncMock
    ) -> None:
        """When the exception message has no parseable Retry-After, exponential backoff is used."""
        service, mock_ex = _make_service(cache_ttl=0)
        raw = _raw_bars(1)
        exc = ccxt_async.RateLimitExceeded("too many requests")
        mock_ex.fetch_ohlcv = AsyncMock(side_effect=[exc, raw])

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)

        assert mock_sleep.called
        sleep_arg = mock_sleep.call_args[0][0]
        # Backoff for attempt=0: base * 2^0 + jitter = 1.0 + U(0, 0.5) → range [1.0, 1.5].
        assert 1.0 <= sleep_arg <= 1.51  # small tolerance for floating point

    def test_extract_retry_after_equals_delimiter(self) -> None:
        """Regex matches equals-sign delimiter form."""
        exc = ccxt_async.RateLimitExceeded("retry-after=45")
        result = CCXTMarketDataService._extract_retry_after(exc)
        assert result == pytest.approx(45.0)

    def test_extract_retry_after_returns_none_when_no_match(self) -> None:
        """Returns None when exception has no retry-after info."""
        exc = ccxt_async.RateLimitExceeded("too many requests")
        result = CCXTMarketDataService._extract_retry_after(exc)
        assert result is None


# ---------------------------------------------------------------------------
# 5. TestFetchOHLCVRangePagination
# ---------------------------------------------------------------------------


class TestFetchOHLCVRangePagination:
    """Pagination, dedup, and boundary behaviour for fetch_ohlcv_range — 10 tests."""

    async def _setup_range_service(
        self, page_limit: int = 5
    ) -> tuple[CCXTMarketDataService, MagicMock]:
        """Helper: connected service with small page_limit for pagination tests."""
        service, mock_ex = _make_service(cache_ttl=0, page_limit=page_limit)
        await service.connect()
        return service, mock_ex

    async def test_range_single_page_no_pagination(self) -> None:
        """When all bars fit in one page (< page_limit rows returned), no second request is made."""
        service, mock_ex = await self._setup_range_service(page_limit=10)
        raw = _raw_bars(3)  # 3 < 10 (page_limit) → short page, stops immediately
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        assert mock_ex.fetch_ohlcv.call_count == 1
        assert len(bars) == 3

    async def test_range_two_pages_advances_cursor(self) -> None:
        """A full first page triggers a second request with the cursor advanced past the last bar."""
        service, mock_ex = await self._setup_range_service(page_limit=3)
        # Page 1: 3 bars (= page_limit → triggers pagination).
        page1 = _raw_bars(3, start_ts_ms=_BASE_TS_MS)
        # Page 2: 2 bars (< page_limit → stops pagination).
        page2 = _raw_bars(2, start_ts_ms=_BASE_TS_MS + 3 * _ONE_HOUR_MS)
        mock_ex.fetch_ohlcv = AsyncMock(side_effect=[page1, page2])

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        assert mock_ex.fetch_ohlcv.call_count == 2
        assert len(bars) == 5

    async def test_range_stops_at_end_boundary(self) -> None:
        """Bars with timestamp >= end are excluded and pagination stops immediately."""
        service, mock_ex = await self._setup_range_service(page_limit=10)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 1, tzinfo=UTC)  # 1-hour window only

        end_ms = int(end.timestamp() * 1000)
        raw = [
            _raw_bar(_BASE_TS_MS),            # inside range
            _raw_bar(end_ms),                 # ts_ms >= end_ms → boundary break
            _raw_bar(end_ms + _ONE_HOUR_MS),  # outside range
        ]
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)
        assert len(bars) == 1
        assert bars[0].timestamp == datetime.fromtimestamp(_BASE_TS_MS / 1000, tz=UTC)

    async def test_range_short_page_stops_pagination(self) -> None:
        """A page with fewer rows than page_limit signals end of data — stops."""
        service, mock_ex = await self._setup_range_service(page_limit=5)
        raw = _raw_bars(3)  # 3 < 5 → short page
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        assert mock_ex.fetch_ohlcv.call_count == 1
        assert len(bars) == 3

    async def test_range_empty_page_stops_pagination(self) -> None:
        """An empty page response causes the loop to break immediately."""
        service, mock_ex = await self._setup_range_service(page_limit=5)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        assert mock_ex.fetch_ohlcv.call_count == 1
        assert bars == []

    async def test_range_no_progress_stops_pagination(self) -> None:
        """All bars in a page are duplicates — no forward progress triggers loop break."""
        service, mock_ex = await self._setup_range_service(page_limit=5)
        # First page: 5 unique bars.
        page1 = _raw_bars(5)
        # Second page: same 5 bars again (all duplicates of page1).
        page2 = _raw_bars(5)
        mock_ex.fetch_ohlcv = AsyncMock(side_effect=[page1, page2])

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        assert mock_ex.fetch_ohlcv.call_count == 2
        assert len(bars) == 5  # deduped — only 5 unique bars

    async def test_range_result_sorted_by_timestamp(self) -> None:
        """Result of fetch_ohlcv_range is always sorted ascending by timestamp."""
        service, mock_ex = await self._setup_range_service(page_limit=10)
        raw = list(reversed(_raw_bars(4)))  # deliberately reversed
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    async def test_range_deduplicates_by_timestamp(self) -> None:
        """Bars with identical timestamps appearing in two pages are deduplicated."""
        service, mock_ex = await self._setup_range_service(page_limit=3)
        # Page 1: bars at hours 0, 1, 2.
        page1 = _raw_bars(3, start_ts_ms=_BASE_TS_MS)
        # Page 2: bar at hour 2 (duplicate) + bar at hour 3 (new) — 2 entries < page_limit.
        overlap_ts = _BASE_TS_MS + 2 * _ONE_HOUR_MS
        new_ts = _BASE_TS_MS + 3 * _ONE_HOUR_MS
        page2 = [_raw_bar(overlap_ts), _raw_bar(new_ts)]
        mock_ex.fetch_ohlcv = AsyncMock(side_effect=[page1, page2])

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        bars = await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

        timestamps = [b.timestamp for b in bars]
        # Should be 4 unique bars (hours 0, 1, 2, 3) despite hour-2 appearing twice.
        assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"
        assert len(bars) == 4

    async def test_range_end_before_start_raises_value_error(self) -> None:
        """fetch_ohlcv_range raises ValueError when end is before start."""
        service, _ = await self._setup_range_service()
        start = datetime(2024, 1, 2, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)  # before start
        with pytest.raises(ValueError, match="must be strictly after"):
            await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, start, end)

    async def test_range_equal_start_end_raises_value_error(self) -> None:
        """fetch_ohlcv_range raises ValueError when end equals start."""
        service, _ = await self._setup_range_service()
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be strictly after"):
            await service.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, ts, ts)


# ---------------------------------------------------------------------------
# 6. TestGetLatestBar
# ---------------------------------------------------------------------------


class TestGetLatestBar:
    """get_latest_bar closed-candle selection — 4 tests."""

    async def test_latest_bar_returns_second_to_last_when_two_bars(self) -> None:
        """With 2 bars returned, get_latest_bar returns bars[-2] (last closed candle)."""
        ts0 = _BASE_TS_MS
        ts1 = _BASE_TS_MS + _ONE_HOUR_MS
        # Bar 0: close=100.0 within [low=95, high=105] ✓
        # Bar 1: close=104.0 within [low=100, high=108] ✓
        raw = [
            _raw_bar(ts0, o=100.0, h=105.0, l=95.0, c=100.0),
            _raw_bar(ts1, o=102.0, h=108.0, l=100.0, c=104.0),
        ]
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        bar = await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_HOUR)
        # get_latest_bar returns bars[-2] = bar at ts0 with close=100.0.
        assert bar.close == Decimal("100.0")

    async def test_latest_bar_returns_single_bar_as_best_effort(self) -> None:
        """With only 1 bar returned, get_latest_bar returns bars[-1] as best-effort."""
        # close=99.0 within [low=95, high=105] ✓
        raw = [_raw_bar(_BASE_TS_MS, o=100.0, h=105.0, l=95.0, c=99.0)]
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        bar = await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_HOUR)
        assert bar.close == Decimal("99.0")

    async def test_latest_bar_empty_response_raises_data_not_available(self) -> None:
        """Empty exchange response causes get_latest_bar to raise DataNotAvailableError."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])

        with pytest.raises(DataNotAvailableError):
            await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_HOUR)

    async def test_latest_bar_result_is_cached(self) -> None:
        """get_latest_bar results pass through the L1 cache (second call is a cache hit)."""
        raw = [
            _raw_bar(_BASE_TS_MS, o=100.0, h=105.0, l=95.0, c=102.0),
            _raw_bar(_BASE_TS_MS + _ONE_HOUR_MS, o=102.0, h=108.0, l=100.0, c=104.0),
        ]
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=raw)

        await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.get_latest_bar(_SYMBOL, TimeFrame.ONE_HOUR)
        # Underlying exchange called once — second call is a cache hit.
        assert mock_ex.fetch_ohlcv.call_count == 1


# ---------------------------------------------------------------------------
# 7. TestSupportedSymbolsTimeframes
# ---------------------------------------------------------------------------


class TestSupportedSymbolsTimeframes:
    """get_supported_symbols / get_supported_timeframes — 5 tests."""

    async def test_symbols_returns_sorted_market_keys(self) -> None:
        """get_supported_symbols returns the exchange market keys sorted alphabetically."""
        markets = {"ZEC/USDT": {}, "BTC/USDT": {}, "ETH/USDT": {}}
        mock_ex = _make_mock_exchange(markets=markets)
        service, _ = _make_service(mock_exchange=mock_ex)
        await service.connect()
        symbols = await service.get_supported_symbols()
        assert symbols == sorted(markets.keys())

    async def test_timeframes_returns_intersection_with_enum(self) -> None:
        """get_supported_timeframes returns only timeframes in both the exchange list and TimeFrame enum."""
        # Exchange supports 1m, 1h, 1d (all in enum) plus "99x" (not in enum).
        mock_ex = _make_mock_exchange(
            timeframes={"1m": "...", "1h": "...", "1d": "...", "99x": "..."}
        )
        service, _ = _make_service(mock_exchange=mock_ex)
        await service.connect()
        tfs = await service.get_supported_timeframes()
        tf_values = {tf.value for tf in tfs}
        assert "1m" in tf_values
        assert "1h" in tf_values
        assert "1d" in tf_values
        assert "99x" not in tf_values

    async def test_symbols_raises_before_connect(self) -> None:
        """get_supported_symbols raises MarketDataError when connect() has not been called."""
        service, _ = _make_service()
        with pytest.raises(MarketDataError, match="Markets not loaded"):
            await service.get_supported_symbols()

    async def test_timeframes_raises_before_connect(self) -> None:
        """get_supported_timeframes raises MarketDataError when connect() has not been called."""
        service, _ = _make_service()
        with pytest.raises(MarketDataError, match="Markets not loaded"):
            await service.get_supported_timeframes()

    async def test_timeframes_sorted_by_duration_ascending(self) -> None:
        """get_supported_timeframes returns timeframes sorted shortest to longest duration."""
        mock_ex = _make_mock_exchange(timeframes={"1d": "...", "1m": "...", "1h": "..."})
        service, _ = _make_service(mock_exchange=mock_ex)
        await service.connect()
        tfs = await service.get_supported_timeframes()
        durations = [service._timeframe_duration_ms(tf) for tf in tfs]
        assert durations == sorted(durations)


# ---------------------------------------------------------------------------
# 8. TestThrottleAndBackoff
# ---------------------------------------------------------------------------


class TestThrottleAndBackoff:
    """Inter-request spacing and backoff formula — 4 tests."""

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    @patch("data.services.ccxt_market_data.time")
    async def test_throttle_sleeps_when_interval_not_elapsed(
        self, mock_time: MagicMock, mock_sleep: AsyncMock
    ) -> None:
        """_throttle() calls asyncio.sleep when less than _MIN_REQUEST_INTERVAL has elapsed."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])

        # _last_request_time=100.0, current time=100.05 → elapsed=0.05 < 0.1.
        # time.monotonic() call sequence inside _throttle: [100.05].
        # Then _last_request_time = time.monotonic() after fetch: [100.1].
        service._last_request_time = 100.0
        mock_time.monotonic.side_effect = [100.05, 100.1]

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)

        # asyncio.sleep should have been called with ~0.05s.
        assert mock_sleep.called
        assert mock_sleep.call_count == 1
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == pytest.approx(0.05, abs=0.01)

    @patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock)
    @patch("data.services.ccxt_market_data.time")
    async def test_throttle_does_not_sleep_when_interval_elapsed(
        self, mock_time: MagicMock, mock_sleep: AsyncMock
    ) -> None:
        """_throttle() does NOT call asyncio.sleep when more than _MIN_REQUEST_INTERVAL has elapsed."""
        service, mock_ex = _make_service(cache_ttl=0)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=[])

        # elapsed = 101.0 - 100.0 = 1.0 >> 0.1 → no sleep.
        service._last_request_time = 100.0
        mock_time.monotonic.return_value = 101.0

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)

        mock_sleep.assert_not_called()

    def test_backoff_formula_respects_cap(self) -> None:
        """_backoff() never returns a value exceeding _BACKOFF_CAP, even for high attempt numbers."""
        for attempt in range(15):
            result = CCXTMarketDataService._backoff(attempt)
            assert result <= _BACKOFF_CAP, f"Attempt {attempt} exceeded cap: {result}"

    def test_backoff_formula_minimum_bound(self) -> None:
        """_backoff(0) always returns at least _BACKOFF_BASE (1.0) before adding jitter."""
        for _ in range(50):
            result = CCXTMarketDataService._backoff(0)
            assert result >= _BACKOFF_BASE, f"Backoff {result} < base {_BACKOFF_BASE}"


# ---------------------------------------------------------------------------
# 9. TestCacheInvalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """invalidate_cache API — 4 tests."""

    async def test_invalidate_symbol_clears_only_that_symbols_entries(self) -> None:
        """invalidate_cache(symbol=BTC/USDT) removes BTC entries and returns the count."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))

        # Populate cache for two symbols.
        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert len(service._cache) == 2

        removed = service.invalidate_cache(symbol=_SYMBOL)
        assert removed == 1
        assert len(service._cache) == 1
        # Remaining entry must be for ETH.
        remaining_key = next(iter(service._cache))
        assert remaining_key[0] == _ALT_SYMBOL

    async def test_invalidate_symbol_and_timeframe_clears_specific_key(self) -> None:
        """invalidate_cache(symbol, timeframe) removes exactly one matching entry."""
        service, _ = _make_service(cache_ttl=60)

        # Manually insert two entries for the same symbol with different timeframes.
        key_1h = (_SYMBOL, TimeFrame.ONE_HOUR.value, None, 500)
        key_1d = (_SYMBOL, TimeFrame.ONE_DAY.value, None, 500)
        far_future = 9_999_999_999.0  # never expires during test
        service._cache[key_1h] = ([], far_future)
        service._cache[key_1d] = ([], far_future)

        removed = service.invalidate_cache(symbol=_SYMBOL, timeframe=TimeFrame.ONE_HOUR)
        assert removed == 1
        assert key_1h not in service._cache
        assert key_1d in service._cache

    async def test_invalidate_all_clears_entire_cache(self) -> None:
        """invalidate_cache() with no args empties the entire cache and returns entry count."""
        service, mock_ex = _make_service(cache_ttl=60)
        mock_ex.fetch_ohlcv = AsyncMock(return_value=_raw_bars(1))

        await service.fetch_ohlcv(_SYMBOL, TimeFrame.ONE_HOUR)
        await service.fetch_ohlcv(_ALT_SYMBOL, TimeFrame.ONE_HOUR)
        assert len(service._cache) == 2

        removed = service.invalidate_cache()
        assert removed == 2
        assert len(service._cache) == 0

    async def test_invalidate_returns_zero_when_cache_empty(self) -> None:
        """invalidate_cache() on an already-empty cache returns 0."""
        service, _ = _make_service(cache_ttl=60)
        assert service._cache == {}
        removed = service.invalidate_cache()
        assert removed == 0


# ---------------------------------------------------------------------------
# 10. TestStaticHelpers
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    """Tests for static/class utility methods."""

    def test_datetime_to_ms_naive_datetime_assumes_utc(self) -> None:
        """A timezone-naive datetime is treated as UTC."""
        naive = datetime(2024, 1, 1, 0, 0, 0)
        aware = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert CCXTMarketDataService._datetime_to_ms(naive) == \
               CCXTMarketDataService._datetime_to_ms(aware)
