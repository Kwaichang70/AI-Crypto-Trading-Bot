"""
packages/data/market_signals.py
---------------------------------
CoinGecko market data client for BTC dominance and market structure signals.

Provides a lightweight async client for the CoinGecko Global Market Data API.
Results are cached for 30 minutes (1800 s) to respect the CoinGecko free-tier
rate limit of ~10-30 calls/minute and avoid hammering the public endpoint.

Usage
-----
::

    from data.market_signals import CoinGeckoClient, set_global_client, get_global_client

    client = CoinGeckoClient()
    snapshot = await client.get_latest()
    if snapshot:
        print(snapshot.btc_dominance, snapshot.market_cap_change_24h)
    await client.close()

Global singleton pattern (used by StrategyEngine)::

    set_global_client(CoinGeckoClient())
    client = get_global_client()
    if client:
        snapshot = await client.get_latest()

API
---
CoinGecko Global Market Data: https://api.coingecko.com/api/v3/global
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

__all__ = [
    "CoinGeckoSnapshot",
    "CoinGeckoClient",
    "set_global_client",
    "get_global_client",
]

logger = structlog.get_logger(__name__)

_ENDPOINT = "https://api.coingecko.com/api/v3/global"
_CACHE_TTL_SECONDS = 1800  # 30 minutes

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_global_client: CoinGeckoClient | None = None


def set_global_client(client: CoinGeckoClient) -> None:
    """Register a CoinGeckoClient as the module-level singleton."""
    global _global_client
    _global_client = client


def get_global_client() -> CoinGeckoClient | None:
    """Return the module-level CoinGeckoClient singleton, or None if not set."""
    return _global_client


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class CoinGeckoSnapshot(BaseModel):
    """
    A single CoinGecko global market data snapshot.

    Fields
    ------
    btc_dominance:
        Bitcoin market cap dominance as a percentage in [0, 100].
        High values (>55%) signal BTC-driven risk-off for altcoins.
        Low values (<45%) signal alt-season / diversified liquidity.
    market_cap_change_24h:
        Total crypto market cap percentage change over the last 24 hours.
        Positive = expanding market, negative = contracting.
    total_volume_change_24h:
        Total 24h trading volume percentage change vs. prior period.
        High positive = increasing participation / momentum.
    timestamp:
        UTC timestamp when this snapshot was captured locally (not the
        CoinGecko server timestamp, which is not provided by the endpoint).
    """

    model_config = {"frozen": True}

    btc_dominance: float = Field(ge=0.0, le=100.0, description="BTC dominance percentage [0,100]")
    market_cap_change_24h: float = Field(description="Total market cap 24h % change")
    total_volume_change_24h: float = Field(description="Total volume 24h % change")
    timestamp: datetime = Field(description="UTC timestamp of this snapshot")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CoinGeckoClient:
    """
    Async HTTP client for the CoinGecko Global Market Data API.

    Features
    --------
    - Lazy aiohttp session (only created on first request)
    - 30-minute in-process cache
    - Stale-cache fallback: returns last known value on network failure
    - Graceful degradation: returns None rather than raising on persistent errors

    Parameters
    ----------
    cache_ttl_seconds:
        How long to cache responses before re-fetching.  Default 30 minutes.
    """

    def __init__(self, cache_ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._session: Any | None = None  # aiohttp.ClientSession, lazy

        # Cache: (snapshot, fetched_at_monotonic)
        self._latest_cache: tuple[CoinGeckoSnapshot, float] | None = None

        self._log = structlog.get_logger(__name__).bind(
            component="coingecko_client"
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
                    timeout=aiohttp.ClientTimeout(total=10),
                )
            except ImportError as exc:
                raise ImportError(
                    "aiohttp is required for CoinGeckoClient. "
                    "Add aiohttp>=3.9 to packages/data/pyproject.toml."
                ) from exc
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self._log.debug("coingecko_client.closed")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, fetched_at: float) -> bool:
        return (time.monotonic() - fetched_at) < self._cache_ttl

    @property
    def cached_value(self) -> CoinGeckoSnapshot | None:
        """Return the most recent cached snapshot, or None if no cache."""
        if self._latest_cache is not None:
            return self._latest_cache[0]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_latest(self) -> CoinGeckoSnapshot | None:
        """
        Fetch the most recent CoinGecko global market snapshot.

        Returns the cached value if it is younger than the TTL.

        Returns
        -------
        CoinGeckoSnapshot or None
            None if the API is unreachable and no cached data is available.
        """
        if self._latest_cache is not None:
            snapshot, fetched_at = self._latest_cache
            if self._is_cache_valid(fetched_at):
                self._log.debug("coingecko_client.cache_hit")
                return snapshot

        try:
            session = await self._get_session()
            async with session.get(_ENDPOINT) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            snapshot = self._parse_snapshot(data)
            self._latest_cache = (snapshot, time.monotonic())
            self._log.info(
                "coingecko_client.fetched",
                btc_dominance=snapshot.btc_dominance,
                market_cap_change_24h=snapshot.market_cap_change_24h,
            )
            return snapshot

        except Exception as exc:
            self._log.warning(
                "coingecko_client.fetch_failed",
                error=str(exc),
            )
            # Return stale cache rather than None when available
            if self._latest_cache is not None:
                self._log.debug("coingecko_client.stale_cache_used")
                return self._latest_cache[0]
            return None

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_snapshot(self, response: dict[str, Any]) -> CoinGeckoSnapshot:
        """
        Parse a CoinGecko /global API response into a CoinGeckoSnapshot.

        Parameters
        ----------
        response:
            The full JSON dict returned by the /global endpoint.

        Returns
        -------
        CoinGeckoSnapshot

        Raises
        ------
        KeyError, TypeError
            If required fields are absent or malformed.
        """
        payload: dict[str, Any] = response["data"]

        # BTC dominance is nested under market_cap_percentage
        btc_dominance = float(payload["market_cap_percentage"].get("btc", 0.0))

        market_cap_change_24h = float(
            payload.get("market_cap_change_percentage_24h_usd", 0.0)
        )

        # total_volume_change_24h is not provided directly by the endpoint;
        # the best proxy is the change encoded in the response metadata.
        # CoinGecko does not expose a total_volume_change field, so we default
        # to 0.0 here and leave the field as informational for now.
        total_volume_change_24h = float(
            payload.get("total_volume_change_percentage_24h", 0.0)
        )

        return CoinGeckoSnapshot(
            btc_dominance=btc_dominance,
            market_cap_change_24h=market_cap_change_24h,
            total_volume_change_24h=total_volume_change_24h,
            timestamp=datetime.now(UTC),
        )
