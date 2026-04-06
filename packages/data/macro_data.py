"""
packages/data/macro_data.py
-----------------------------
FRED (Federal Reserve Economic Data) macro-economic data client.

Provides a lightweight async client for the St. Louis Fed FRED API.
Fetches two key macro series that historically correlate with crypto
risk appetite:

- **FEDFUNDS**: Effective Federal Funds Rate — the short-term benchmark
  interest rate set by the FOMC.  High rates compress risk-asset
  valuations by raising the discount rate.
- **T10Y2Y**: 10-Year minus 2-Year Treasury Yield Spread.  Negative values
  (yield curve inversion) historically precede recessions and risk-off moves.

Results are cached for 24 hours (86400 s) because FRED data is released
monthly (FEDFUNDS) or daily (T10Y2Y) and intra-day re-fetches add no value.

Usage
-----
::

    from data.macro_data import FREDClient, set_global_client, get_global_client

    client = FREDClient(api_key="your_fred_api_key")
    snapshot = await client.get_latest()
    if snapshot:
        print(snapshot.fed_funds_rate, snapshot.yield_curve_spread)
    await client.close()

Global singleton pattern (used by StrategyEngine)::

    set_global_client(FREDClient(api_key=settings.fred_api_key))
    client = get_global_client()
    if client:
        snapshot = await client.get_latest()

API
---
FRED REST API: https://fred.stlouisfed.org/docs/api/fred/
Register for a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

__all__ = [
    "MacroSnapshot",
    "FREDClient",
    "set_global_client",
    "get_global_client",
]

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_TTL_SECONDS = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_global_client: FREDClient | None = None


def set_global_client(client: FREDClient) -> None:
    """Register a FREDClient as the module-level singleton."""
    global _global_client
    _global_client = client


def get_global_client() -> FREDClient | None:
    """Return the module-level FREDClient singleton, or None if not set."""
    return _global_client


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class MacroSnapshot(BaseModel):
    """
    A single FRED macro-economic data snapshot.

    Both fields are optional because FRED data may have observation gaps
    or the API call for a specific series may fail independently.

    Fields
    ------
    fed_funds_rate:
        Effective Federal Funds Rate (percent).  Higher values raise the
        opportunity cost of holding risk assets like crypto.
    yield_curve_spread:
        10-Year Treasury minus 2-Year Treasury yield spread (percent).
        Negative = inverted curve (historically precedes recession).
        Values below -0.5 signal elevated recession risk.
        Values above +1.0 signal a healthy, risk-on macro environment.
    timestamp:
        UTC timestamp when this snapshot was captured locally.
    """

    model_config = {"frozen": True}

    fed_funds_rate: float | None = Field(
        default=None, description="Effective federal funds rate (percent)"
    )
    yield_curve_spread: float | None = Field(
        default=None, description="10Y-2Y Treasury yield spread (percent)"
    )
    timestamp: datetime = Field(description="UTC timestamp of this snapshot")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FREDClient:
    """
    Async HTTP client for the St. Louis Federal Reserve FRED API.

    Features
    --------
    - Lazy aiohttp session (only created on first request)
    - 24-hour in-process cache (FRED data updates at most daily)
    - Stale-cache fallback: returns last known value on network failure
    - Partial success: a snapshot is returned even if only one series fetches
    - Graceful degradation: returns None rather than raising on total failure

    Parameters
    ----------
    api_key:
        FRED API key.  Register at https://fred.stlouisfed.org/docs/api/api_key.html
        The free tier provides 120 requests/minute — well above our 1/24h usage.
    session:
        Optional pre-existing aiohttp.ClientSession.  When None, a session is
        created lazily on the first request.
    cache_ttl_seconds:
        How long to cache responses before re-fetching.  Default 24 hours.
    """

    def __init__(
        self,
        api_key: str,
        session: Any | None = None,
        cache_ttl_seconds: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl = cache_ttl_seconds
        self._session: Any | None = session  # aiohttp.ClientSession, lazy

        # Cache: (snapshot, fetched_at_monotonic)
        self._latest_cache: tuple[MacroSnapshot, float] | None = None

        self._log = structlog.get_logger(__name__).bind(
            component="fred_client"
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Return (or create) the underlying aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15),
                )
            except ImportError as exc:
                raise ImportError(
                    "aiohttp is required for FREDClient. "
                    "Add aiohttp>=3.9 to packages/data/pyproject.toml."
                ) from exc
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self._log.debug("fred_client.closed")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, fetched_at: float) -> bool:
        return (time.monotonic() - fetched_at) < self._cache_ttl

    @property
    def cached_value(self) -> MacroSnapshot | None:
        """Return the most recent cached snapshot, or None if no cache."""
        if self._latest_cache is not None:
            return self._latest_cache[0]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_latest(self) -> MacroSnapshot | None:
        """
        Fetch the most recent FEDFUNDS and T10Y2Y observations from FRED.

        Returns the cached value if it is younger than the TTL.

        Returns
        -------
        MacroSnapshot or None
            None if both series fail to fetch and no cached data is available.
            A partial snapshot (with None fields) is returned if only one
            series fails.
        """
        if self._latest_cache is not None:
            snapshot, fetched_at = self._latest_cache
            if self._is_cache_valid(fetched_at):
                self._log.debug("fred_client.cache_hit")
                return snapshot

        try:
            fed_funds_rate = await self._fetch_series("FEDFUNDS")
            yield_curve_spread = await self._fetch_series("T10Y2Y")

            snapshot = MacroSnapshot(
                fed_funds_rate=fed_funds_rate,
                yield_curve_spread=yield_curve_spread,
                timestamp=datetime.now(UTC),
            )
            self._latest_cache = (snapshot, time.monotonic())
            self._log.info(
                "fred_client.fetched",
                fed_funds_rate=fed_funds_rate,
                yield_curve_spread=yield_curve_spread,
            )
            return snapshot

        except Exception as exc:
            self._log.warning(
                "fred_client.fetch_failed",
                error=str(exc),
            )
            if self._latest_cache is not None:
                self._log.debug("fred_client.stale_cache_used")
                return self._latest_cache[0]
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_series(self, series_id: str) -> float | None:
        """
        Fetch the latest observation for a single FRED series.

        Parameters
        ----------
        series_id:
            FRED series identifier (e.g. ``"FEDFUNDS"``, ``"T10Y2Y"``).

        Returns
        -------
        float or None
            The most recent observation value, or None if unavailable or
            if the observation value is the FRED sentinel ``"."``.
        """
        try:
            session = await self._get_session()
            params = {
                "series_id": series_id,
                "sort_order": "desc",
                "limit": "1",
                "api_key": self._api_key,
                "file_type": "json",
            }
            async with session.get(_BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            observations = data.get("observations", [])
            if not observations:
                self._log.debug("fred_client.no_observations", series_id=series_id)
                return None

            raw_value = observations[0].get("value", ".")
            # FRED uses "." as a sentinel for missing/not-yet-released values
            if raw_value == ".":
                self._log.debug("fred_client.missing_value", series_id=series_id)
                return None

            return float(raw_value)

        except Exception as exc:
            self._log.warning(
                "fred_client.series_fetch_failed",
                series_id=series_id,
                error=str(exc),
            )
            return None
