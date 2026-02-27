"""
packages/data/market_data.py
-----------------------------
MarketDataService — OHLCV candle fetching, normalisation, and caching.

Architecture
------------
- Exchange access is via CCXT's async API.
- Responses are normalised to UTC datetimes and Decimal prices.
- A two-level cache is used:
    L1: In-process LRU (per symbol/timeframe, bounded size)
    L2: PostgreSQL candle cache (persistent across restarts)
- Rate limiting uses a token-bucket pattern with exponential backoff on 429s.
- All public methods are async.

This module defines the abstract interface only.
Concrete implementation: packages/data/services/ccxt_market_data.py (TBI).
"""

from __future__ import annotations

import abc
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from common.types import TimeFrame
from common.models import OHLCVBar

__all__ = [
    "BaseMarketDataService",
    "MarketDataError",
    "RateLimitError",
    "DataNotAvailableError",
]

logger = structlog.get_logger(__name__)


class MarketDataError(Exception):
    """Base exception for all market data errors."""


class RateLimitError(MarketDataError):
    """
    Raised when the exchange rate limit is hit.

    The ``retry_after`` attribute is the suggested wait time in seconds,
    extracted from the 429 response header when available.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DataNotAvailableError(MarketDataError):
    """
    Raised when requested OHLCV data does not exist for the given
    symbol / timeframe / date range combination.
    """


class BaseMarketDataService(abc.ABC):
    """
    Abstract interface for OHLCV market data acquisition and caching.

    Implementations must handle:
    - Pagination of large historical ranges (CCXT limits per-request rows)
    - Timestamp normalisation to UTC
    - Automatic retry with exponential backoff on transient errors
    - Local caching to avoid redundant exchange round-trips

    Parameters
    ----------
    exchange_id:
        CCXT exchange identifier, e.g. ``"kraken"``.
    cache_ttl_seconds:
        How long cached candle data is considered fresh.
        Set to 0 to disable L1 in-process caching.
    """

    def __init__(
        self,
        exchange_id: str,
        cache_ttl_seconds: int = 60,
    ) -> None:
        self._exchange_id = exchange_id
        self._cache_ttl_seconds = cache_ttl_seconds
        self._log = structlog.get_logger(__name__).bind(exchange=exchange_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """
        Fetch OHLCV candle data for a symbol from the exchange.

        Parameters
        ----------
        symbol:
            Trading pair in CCXT format, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle duration as a ``TimeFrame`` enum member.
        since:
            Start of the requested range in UTC. If None, fetches the
            most recent ``limit`` candles.
        limit:
            Maximum number of candles to return. Subject to exchange
            per-request caps (typically 500–1000).

        Returns
        -------
        list[OHLCVBar]:
            Candles sorted by timestamp ascending.

        Raises
        ------
        RateLimitError:
            On HTTP 429 from the exchange.
        DataNotAvailableError:
            When the symbol or timeframe is not supported by the exchange.
        MarketDataError:
            On other exchange communication errors.
        """
        ...

    @abc.abstractmethod
    async def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """
        Fetch all OHLCV candles for the complete ``[start, end)`` range.

        This method handles pagination transparently — it issues as many
        requests as needed and returns the complete merged result.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle duration.
        start:
            Range start (inclusive) in UTC.
        end:
            Range end (exclusive) in UTC.

        Returns
        -------
        list[OHLCVBar]:
            All available candles within the range, sorted ascending.
        """
        ...

    @abc.abstractmethod
    async def get_latest_bar(self, symbol: str, timeframe: TimeFrame) -> OHLCVBar:
        """
        Return the most recently completed candle for a symbol.

        In paper/live mode this is polled on each bar interval.
        Implementations SHOULD cache the result for ``cache_ttl_seconds``.

        Parameters
        ----------
        symbol:
            Trading pair.
        timeframe:
            Candle duration.

        Returns
        -------
        OHLCVBar:
            The last closed candle.
        """
        ...

    @abc.abstractmethod
    async def get_supported_symbols(self) -> list[str]:
        """
        Return all trading pairs supported by the configured exchange.

        Returns
        -------
        list[str]:
            Symbols in CCXT format (e.g. ``["BTC/USDT", "ETH/USDT", ...]``).
        """
        ...

    @abc.abstractmethod
    async def get_supported_timeframes(self) -> list[TimeFrame]:
        """
        Return the timeframes supported by the configured exchange.

        Returns
        -------
        list[TimeFrame]:
            Supported timeframe enum members.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def connect(self) -> None:
        """
        Initialise the exchange connection and load markets.

        Must be called before any data fetching methods.
        Typically invoked from the FastAPI lifespan handler.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the exchange connection and release resources.

        Must be called on graceful shutdown.
        """
        ...

    # ------------------------------------------------------------------
    # Normalisation helpers (shared, non-abstract)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_timestamp(ts_ms: int) -> datetime:
        """
        Convert a CCXT millisecond Unix timestamp to a UTC datetime.

        Parameters
        ----------
        ts_ms:
            Milliseconds since Unix epoch as returned by CCXT.

        Returns
        -------
        datetime:
            Timezone-aware UTC datetime.
        """
        return datetime.fromtimestamp(ts_ms / 1000, tz=UTC)

    @staticmethod
    def _to_decimal(value: float | int | str) -> Decimal:
        """
        Convert a raw OHLCV value from the exchange to Decimal.

        Avoids float precision issues by converting via str.
        """
        return Decimal(str(value))

    @staticmethod
    def _parse_raw_bar(
        raw: list[int | float],
        symbol: str,
        timeframe: TimeFrame,
    ) -> OHLCVBar:
        """
        Parse a single CCXT OHLCV list into an OHLCVBar.

        CCXT format: ``[timestamp_ms, open, high, low, close, volume]``

        Parameters
        ----------
        raw:
            6-element list from CCXT.
        symbol:
            Trading pair string.
        timeframe:
            Candle timeframe.

        Returns
        -------
        OHLCVBar:
            Fully validated OHLCVBar instance.
        """
        ts_ms, open_, high, low, close, volume = raw
        return OHLCVBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=BaseMarketDataService._normalise_timestamp(int(ts_ms)),
            open=BaseMarketDataService._to_decimal(open_),
            high=BaseMarketDataService._to_decimal(high),
            low=BaseMarketDataService._to_decimal(low),
            close=BaseMarketDataService._to_decimal(close),
            volume=BaseMarketDataService._to_decimal(volume),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exchange={self._exchange_id!r})"
