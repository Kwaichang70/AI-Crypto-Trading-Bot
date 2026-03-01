"""
packages/data/services/ccxt_market_data.py
--------------------------------------------
Concrete CCXT implementation of BaseMarketDataService.

Architecture overview
---------------------
- Exchange I/O uses ``ccxt.async_support`` (HTTP, not WebSocket).
- An asyncio Semaphore throttles concurrent outbound requests; a simple
  token-bucket style enforces minimum inter-request spacing per exchange.
- On HTTP 429 the request is retried with truncated binary exponential
  backoff plus uniform jitter (max 3 retries by default).
- An in-process TTL dict cache (L1) short-circuits repeated identical
  queries within the configured ``cache_ttl_seconds`` window.
  Cache key is ``(symbol, timeframe_value, since_ms_or_None, limit)``.
- ``fetch_ohlcv_range`` paginates transparently: it advances ``since``
  by the exchange-reported (or inferred) rows per page until either the
  exchange returns fewer rows than requested or ``end`` is exceeded.
  Duplicate timestamps arising at page boundaries are deduplicated.
- All domain exceptions are mapped from CCXT exception hierarchy.

What this module does NOT do
-----------------------------
- No L2 (PostgreSQL) caching — reserved for a subsequent sprint.
- No WebSocket streaming — post-MVP.
- No leverage / derivatives — spot only for MVP.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
from datetime import UTC, datetime
from typing import Any

import ccxt.async_support as ccxt_async
import structlog

from common.types import TimeFrame
from common.models import OHLCVBar
from data.market_data import (
    BaseMarketDataService,
    DataNotAvailableError,
    MarketDataError,
    RateLimitError,
)

__all__ = ["CCXTMarketDataService"]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum number of candles per exchange request.
#: Most exchanges cap at 500; a conservative default prevents silent truncation.
_DEFAULT_PAGE_LIMIT: int = 500

#: Maximum number of retry attempts on transient errors (429, NetworkError).
_MAX_RETRIES: int = 3

#: Base wait in seconds for exponential backoff (doubles each retry).
_BACKOFF_BASE: float = 1.0

#: Adds up to this many seconds of uniform random jitter per retry attempt.
_BACKOFF_JITTER: float = 0.5

#: Maximum seconds to wait between retries regardless of backoff formula.
_BACKOFF_CAP: float = 60.0

#: Maximum concurrent in-flight requests to the same exchange.
_DEFAULT_MAX_CONCURRENCY: int = 5

#: Minimum seconds between consecutive outbound requests (token bucket floor).
_MIN_REQUEST_INTERVAL: float = 0.1  # 100 ms — well inside most exchange limits

#: Candle duration in milliseconds for each supported timeframe.
_TIMEFRAME_DURATION_MS: dict[str, int] = {
    "1m":  60 * 1_000,
    "3m":  3 * 60 * 1_000,
    "5m":  5 * 60 * 1_000,
    "15m": 15 * 60 * 1_000,
    "30m": 30 * 60 * 1_000,
    "1h":  60 * 60 * 1_000,
    "4h":  4 * 60 * 60 * 1_000,
    "1d":  24 * 60 * 60 * 1_000,
    "1w":  7 * 24 * 60 * 60 * 1_000,
}

# ---------------------------------------------------------------------------
# Internal cache entry type alias
# ---------------------------------------------------------------------------

# (result_payload, expiry_epoch_float)
_CacheEntry = tuple[list[OHLCVBar], float]


class CCXTMarketDataService(BaseMarketDataService):
    """
    Concrete market data service backed by CCXT's async HTTP client.

    Parameters
    ----------
    exchange_id:
        CCXT exchange identifier (e.g. ``"binance"``, ``"kraken"``).
    api_key:
        Exchange API key. Required only for authenticated endpoints.
        Public OHLCV data does not require authentication on most exchanges.
    api_secret:
        Exchange API secret. Required only for authenticated endpoints.
    cache_ttl_seconds:
        Lifetime of L1 in-process cache entries in seconds. Pass ``0`` to
        disable caching entirely.
    page_limit:
        Maximum candles per exchange request. Defaults to ``500``.
        Reduce for exchanges with smaller per-page caps.
    max_concurrency:
        Maximum simultaneous in-flight HTTP requests. Defaults to ``5``.
    sandbox:
        When ``True``, enables the exchange's sandbox/testnet endpoint if
        supported. Default ``False``.
    extra_exchange_config:
        Additional key/value pairs forwarded verbatim to the CCXT exchange
        constructor (e.g. ``{"options": {"defaultType": "spot"}}``).

    Examples
    --------
    Typical FastAPI lifespan usage::

        service = CCXTMarketDataService(exchange_id="binance", cache_ttl_seconds=30)
        await service.connect()
        bars = await service.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=200)
        await service.close()
    """

    def __init__(
        self,
        exchange_id: str,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        cache_ttl_seconds: int = 60,
        page_limit: int = _DEFAULT_PAGE_LIMIT,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        sandbox: bool = False,
        extra_exchange_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(exchange_id=exchange_id, cache_ttl_seconds=cache_ttl_seconds)

        self._api_key = api_key
        self._api_secret = api_secret
        self._page_limit = page_limit
        self._sandbox = sandbox

        # ---- L1 in-process cache ----
        # Key: (symbol, timeframe_str, since_ms_or_None, limit)
        # Value: (list[OHLCVBar], expiry_epoch)
        self._cache: dict[tuple[Any, ...], _CacheEntry] = {}

        # ---- Concurrency / rate-limit controls ----
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._last_request_time: float = 0.0

        # ---- Loaded markets (populated in connect()) ----
        self._markets: dict[str, Any] = {}
        self._exchange_timeframes: set[str] = set()

        # ---- CCXT exchange instance ----
        exchange_cls = getattr(ccxt_async, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(
                f"Exchange '{exchange_id}' is not supported by CCXT. "
                "Call ccxt.exchanges to see all available exchange IDs."
            )

        ccxt_config: dict[str, Any] = {
            "enableRateLimit": False,  # We handle rate-limiting ourselves.
            "newUpdates": False,
        }
        if api_key is not None:
            ccxt_config["apiKey"] = api_key
        if api_secret is not None:
            ccxt_config["secret"] = api_secret
        if extra_exchange_config:
            ccxt_config.update(extra_exchange_config)

        self._exchange: ccxt_async.Exchange = exchange_cls(ccxt_config)

        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._log.debug(
            "exchange_instance_created",
            page_limit=page_limit,
            max_concurrency=max_concurrency,
            sandbox=sandbox,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Load exchange markets and cache supported symbols and timeframes.

        Must be awaited before calling any data-fetching method.
        Idempotent: safe to call multiple times (re-loads markets each time).
        """
        self._log.info("connecting_to_exchange")
        try:
            self._markets = await self._exchange.load_markets()
        except ccxt_async.AuthenticationError as exc:
            raise MarketDataError(
                f"Authentication failed for exchange '{self._exchange_id}': {exc}"
            ) from exc
        except ccxt_async.NetworkError as exc:
            raise MarketDataError(
                f"Network error loading markets for '{self._exchange_id}': {exc}"
            ) from exc
        except ccxt_async.ExchangeError as exc:
            raise MarketDataError(
                f"Exchange error loading markets for '{self._exchange_id}': {exc}"
            ) from exc

        raw_timeframes: dict[str, str] = getattr(self._exchange, "timeframes", {}) or {}
        self._exchange_timeframes = set(raw_timeframes.keys())

        self._log.info(
            "exchange_connected",
            symbol_count=len(self._markets),
            timeframe_count=len(self._exchange_timeframes),
        )

    async def close(self) -> None:
        """
        Close the underlying HTTP session.

        Always await this on graceful shutdown to avoid resource leaks.
        """
        self._log.info("closing_exchange_connection")
        try:
            await self._exchange.close()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            self._log.warning("exchange_close_error", error=str(exc))

    # ------------------------------------------------------------------
    # Public data-fetching API
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """
        Fetch up to ``limit`` candles starting at ``since``.

        Results are served from the L1 cache when a non-expired entry exists
        for the exact ``(symbol, timeframe, since, limit)`` combination.
        Live candles (``since=None``) have a shorter effective TTL because the
        most-recent bar may not yet be closed; the standard TTL still applies
        since we cache the response, not the "is closed" status.
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        since_ms: int | None = self._datetime_to_ms(since) if since is not None else None
        cache_key = (symbol, timeframe.value, since_ms, limit)

        cached = self._cache_get(cache_key)
        if cached is not None:
            self._log.debug(
                "cache_hit",
                symbol=symbol,
                timeframe=timeframe.value,
                since_ms=since_ms,
                limit=limit,
            )
            return cached

        bars = await self._fetch_raw_with_retry(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            limit=limit,
        )

        self._cache_set(cache_key, bars)
        return bars

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """
        Fetch all candles within the half-open interval ``[start, end)``.

        Paginates automatically: issues repeated requests advancing the
        ``since`` cursor until all candles up to ``end`` are retrieved or
        the exchange returns an empty page.

        Duplicates at page boundaries (same timestamp appearing in two
        consecutive pages) are removed.  The result is always sorted
        ascending by timestamp.
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        if end <= start:
            raise ValueError(
                f"end ({end.isoformat()}) must be strictly after start ({start.isoformat()})"
            )

        start_ms = self._datetime_to_ms(start)
        end_ms = self._datetime_to_ms(end)

        self._log.info(
            "fetch_ohlcv_range_start",
            symbol=symbol,
            timeframe=timeframe.value,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        all_bars: list[OHLCVBar] = []
        seen_timestamps: set[int] = set()
        cursor_ms = start_ms

        while True:
            page = await self._fetch_raw_with_retry(
                symbol=symbol,
                timeframe=timeframe,
                since_ms=cursor_ms,
                limit=self._page_limit,
            )

            if not page:
                # Exchange returned nothing — we have exhausted available data.
                self._log.debug(
                    "fetch_ohlcv_range_empty_page",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    cursor_ms=cursor_ms,
                )
                break

            new_bars_in_page = 0
            last_ts_ms = cursor_ms

            for bar in page:
                ts_ms = int(bar.timestamp.timestamp() * 1000)

                if ts_ms >= end_ms:
                    # Past the requested end boundary — stop pagination.
                    # Bars before this boundary were already collected by the
                    # normal path on earlier iterations; do NOT extend again.
                    self._log.debug(
                        "fetch_ohlcv_range_reached_end",
                        symbol=symbol,
                        last_ts=bar.timestamp.isoformat(),
                    )
                    cursor_ms = end_ms
                    break

                if ts_ms not in seen_timestamps:
                    all_bars.append(bar)
                    seen_timestamps.add(ts_ms)
                    new_bars_in_page += 1
                    last_ts_ms = ts_ms
            else:
                # for-loop completed without break: all bars were before end_ms
                if new_bars_in_page == 0:
                    # All bars in this page were duplicates — no forward progress.
                    self._log.warning(
                        "fetch_ohlcv_range_no_progress",
                        symbol=symbol,
                        timeframe=timeframe.value,
                        cursor_ms=cursor_ms,
                    )
                    break

                if len(page) < self._page_limit:
                    # Fewer rows than requested — no more data available.
                    self._log.debug(
                        "fetch_ohlcv_range_short_page",
                        symbol=symbol,
                        rows_returned=len(page),
                        page_limit=self._page_limit,
                    )
                    break

                # Advance cursor past the last bar we received.
                # Add one timeframe-duration to avoid refetching the same bar.
                tf_ms = self._timeframe_duration_ms(timeframe)
                cursor_ms = last_ts_ms + tf_ms
                continue

            # for-loop broke via the ts_ms >= end_ms guard — stop outer loop.
            break

        # Final sort + dedup pass (defensive; should already be clean)
        all_bars.sort(key=lambda b: b.timestamp)

        self._log.info(
            "fetch_ohlcv_range_complete",
            symbol=symbol,
            timeframe=timeframe.value,
            bars_fetched=len(all_bars),
        )
        return all_bars

    async def get_latest_bar(self, symbol: str, timeframe: TimeFrame) -> OHLCVBar:
        """
        Return the most recently closed candle.

        Fetches the last 2 bars and returns index ``-2`` (the last complete
        candle) to avoid returning a still-forming bar.  On exchanges that
        return only 1 bar, index ``-1`` is returned as a best-effort result.

        The result is cached for ``cache_ttl_seconds`` (inherited from the
        base class).
        """
        bars = await self.fetch_ohlcv(symbol, timeframe, since=None, limit=2)

        if not bars:
            raise DataNotAvailableError(
                f"No bars available for {symbol!r} / {timeframe.value}"
            )

        # If the exchange returned 2 bars, the second-to-last is confirmed closed.
        if len(bars) >= 2:
            return bars[-2]
        return bars[-1]

    async def get_supported_symbols(self) -> list[str]:
        """
        Return all spot trading pairs loaded during ``connect()``.

        Returns
        -------
        list[str]:
            Symbols in CCXT format (e.g. ``["BTC/USDT", "ETH/USDT"]``).

        Raises
        ------
        MarketDataError:
            If ``connect()`` has not been called.
        """
        if not self._markets:
            raise MarketDataError(
                "Markets not loaded. Call await service.connect() first."
            )
        return sorted(self._markets.keys())

    async def get_supported_timeframes(self) -> list[TimeFrame]:
        """
        Return the intersection of exchange-supported timeframes and our
        ``TimeFrame`` enum values.

        Returns
        -------
        list[TimeFrame]:
            Supported timeframe enum members sorted by duration ascending.
        """
        if not self._markets:
            raise MarketDataError(
                "Markets not loaded. Call await service.connect() first."
            )

        supported: list[TimeFrame] = []
        for tf in TimeFrame:
            if tf.value in self._exchange_timeframes:
                supported.append(tf)

        # Sort by duration so callers can iterate from shortest to longest.
        supported.sort(key=lambda tf: self._timeframe_duration_ms(tf))
        return supported

    # ------------------------------------------------------------------
    # Internal fetch with retry + rate-limiting
    # ------------------------------------------------------------------

    async def _fetch_raw_with_retry(
        self,
        symbol: str,
        timeframe: TimeFrame,
        since_ms: int | None,
        limit: int,
    ) -> list[OHLCVBar]:
        """
        Execute one CCXT ``fetch_ohlcv`` call, retrying on transient errors.

        Retry policy
        ------------
        - ``ccxt.RateLimitExceeded`` (HTTP 429): exponential backoff with
          jitter.  Respects ``Retry-After`` header if extractable.
        - ``ccxt.NetworkError``: same backoff schedule (transient failures).
        - All other ``ccxt.ExchangeError`` subclasses: mapped to domain
          exceptions and re-raised immediately (not retried).

        Returns
        -------
        list[OHLCVBar]:
            Parsed bars, sorted ascending by timestamp.
        """
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= _MAX_RETRIES:
            try:
                async with self._semaphore:
                    await self._throttle()
                    raw: list[list[int | float]] = await self._exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe.value,
                        since=since_ms,
                        limit=limit,
                    )
                    self._last_request_time = time.monotonic()

                if not raw:
                    return []

                bars = [
                    self._parse_raw_bar(row, symbol, timeframe)
                    for row in raw
                ]
                bars.sort(key=lambda b: b.timestamp)
                return bars

            except ccxt_async.RateLimitExceeded as exc:
                retry_after = self._extract_retry_after(exc)
                wait = retry_after if retry_after else self._backoff(attempt)
                self._log.warning(
                    "rate_limit_hit",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    attempt=attempt,
                    wait_seconds=round(wait, 2),
                )
                if attempt >= _MAX_RETRIES:
                    raise RateLimitError(
                        f"Rate limit exceeded for {symbol!r} after {_MAX_RETRIES} retries",
                        retry_after=retry_after,
                    ) from exc
                await asyncio.sleep(wait)
                last_exc = exc

            except ccxt_async.ExchangeNotAvailable as exc:
                raise MarketDataError(
                    f"Exchange '{self._exchange_id}' is currently unavailable: {exc}"
                ) from exc

            except ccxt_async.NetworkError as exc:
                wait = self._backoff(attempt)
                self._log.warning(
                    "network_error_retrying",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    attempt=attempt,
                    wait_seconds=round(wait, 2),
                    error=str(exc),
                )
                if attempt >= _MAX_RETRIES:
                    raise MarketDataError(
                        f"Network error fetching {symbol!r}: {exc}"
                    ) from exc
                await asyncio.sleep(wait)
                last_exc = exc

            except ccxt_async.BadSymbol as exc:
                raise DataNotAvailableError(
                    f"Symbol {symbol!r} is not available on exchange "
                    f"'{self._exchange_id}': {exc}"
                ) from exc

            except ccxt_async.BadRequest as exc:
                raise DataNotAvailableError(
                    f"Bad request for {symbol!r} / {timeframe.value}: {exc}"
                ) from exc

            except ccxt_async.ExchangeError as exc:
                raise MarketDataError(
                    f"Exchange error fetching OHLCV for {symbol!r}: {exc}"
                ) from exc

            attempt += 1

        # Should be unreachable, but keeps type-checker happy.
        raise MarketDataError(
            f"Failed to fetch OHLCV for {symbol!r} after exhausting retries"
        ) from last_exc

    # ------------------------------------------------------------------
    # Rate-limit / throttle helpers
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """
        Enforce minimum inter-request spacing to stay inside exchange limits.

        Implemented as a simple "wait until min interval has elapsed"
        strategy.  The asyncio Semaphore (acquired by the caller) prevents
        more than ``max_concurrency`` simultaneous sleeps from firing at once.
        """
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    @staticmethod
    def _backoff(attempt: int) -> float:
        """
        Compute truncated binary exponential backoff with uniform jitter.

        Formula: ``min(base * 2^attempt + U(0, jitter), cap)``
        """
        raw = _BACKOFF_BASE * math.pow(2, attempt) + random.uniform(0, _BACKOFF_JITTER)
        return min(raw, _BACKOFF_CAP)

    @staticmethod
    def _extract_retry_after(exc: ccxt_async.RateLimitExceeded) -> float | None:
        """
        Attempt to parse a ``Retry-After`` value from the exception message.

        CCXT embeds the HTTP response body in the exception string.  A
        numeric value in seconds is extracted when present.  Returns ``None``
        when extraction fails so callers fall back to the backoff formula.
        """
        match = re.search(r"retry.after[\"'\s:=]+(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # L1 in-process TTL cache
    # ------------------------------------------------------------------

    def _cache_get(self, key: tuple[Any, ...]) -> list[OHLCVBar] | None:
        """
        Return the cached value for ``key`` if it exists and has not expired.

        Returns ``None`` on a miss or when caching is disabled
        (``cache_ttl_seconds == 0``).
        """
        if self._cache_ttl_seconds == 0:
            return None

        entry = self._cache.get(key)
        if entry is None:
            return None

        bars, expiry = entry
        if time.monotonic() > expiry:
            # Expired — remove stale entry and treat as miss.
            del self._cache[key]
            return None

        return bars

    def _cache_set(self, key: tuple[Any, ...], bars: list[OHLCVBar]) -> None:
        """
        Store ``bars`` in the L1 cache under ``key`` with TTL expiry.

        No-op when caching is disabled (``cache_ttl_seconds == 0``).
        """
        if self._cache_ttl_seconds == 0:
            return

        expiry = time.monotonic() + self._cache_ttl_seconds
        self._cache[key] = (bars, expiry)

    def invalidate_cache(
        self,
        symbol: str | None = None,
        timeframe: TimeFrame | None = None,
    ) -> int:
        """
        Invalidate L1 cache entries matching the given filters.

        Parameters
        ----------
        symbol:
            When provided, only entries for this symbol are removed.
        timeframe:
            When provided, only entries for this timeframe are removed.
            Has no effect unless ``symbol`` is also given (or alone).
        Returns
        -------
        int:
            Number of entries removed.
        """
        if symbol is None and timeframe is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        to_delete = []
        tf_value = timeframe.value if timeframe is not None else None

        for key in self._cache:
            key_symbol, key_tf, *_ = key
            symbol_match = symbol is None or key_symbol == symbol
            tf_match = tf_value is None or key_tf == tf_value
            if symbol_match and tf_match:
                to_delete.append(key)

        for key in to_delete:
            del self._cache[key]

        return len(to_delete)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_symbol(self, symbol: str) -> None:
        """
        Raise ``DataNotAvailableError`` if ``symbol`` is not in loaded markets.

        Skips validation when markets have not been loaded yet (defensive
        against calling fetch_ohlcv before connect() in tests).
        """
        if not self._markets:
            # Markets not loaded — allow the request through; CCXT will error.
            return

        if symbol not in self._markets:
            raise DataNotAvailableError(
                f"Symbol {symbol!r} is not listed on exchange '{self._exchange_id}'. "
                f"Call get_supported_symbols() to see available pairs."
            )

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """
        Raise ``DataNotAvailableError`` if ``timeframe`` is not supported by
        the exchange.

        Skips validation when markets have not been loaded yet.
        """
        if not self._exchange_timeframes:
            return

        if timeframe.value not in self._exchange_timeframes:
            raise DataNotAvailableError(
                f"Timeframe {timeframe.value!r} is not supported by exchange "
                f"'{self._exchange_id}'. "
                f"Supported: {sorted(self._exchange_timeframes)}"
            )

    # ------------------------------------------------------------------
    # Static / class utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _datetime_to_ms(dt: datetime) -> int:
        """
        Convert a UTC-aware ``datetime`` to a millisecond Unix timestamp.

        Parameters
        ----------
        dt:
            Timezone-aware datetime.  Naive datetimes are assumed UTC.

        Returns
        -------
        int:
            Milliseconds since Unix epoch.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _timeframe_duration_ms(timeframe: TimeFrame) -> int:
        """
        Return the candle duration in milliseconds for a given timeframe.

        Used to advance the pagination cursor by exactly one candle width.

        Parameters
        ----------
        timeframe:
            A ``TimeFrame`` enum member.

        Returns
        -------
        int:
            Duration in milliseconds.
        """
        duration = _TIMEFRAME_DURATION_MS.get(timeframe.value)
        if duration is None:
            raise ValueError(
                f"Unknown timeframe {timeframe.value!r}. "
                "Add it to _TIMEFRAME_DURATION_MS at module level in ccxt_market_data.py."
            )
        return duration

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CCXTMarketDataService("
            f"exchange={self._exchange_id!r}, "
            f"cache_ttl={self._cache_ttl_seconds}s, "
            f"page_limit={self._page_limit})"
        )
