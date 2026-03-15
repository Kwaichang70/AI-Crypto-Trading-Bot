"""
packages/data/sentiment.py
---------------------------
Fear & Greed Index client for adaptive learning (Sprint 32).

Provides a lightweight async client for the Alternative.me Crypto Fear &
Greed Index API. Results are cached for 6 hours to avoid hammering the
public endpoint during backtest replays or high-frequency paper runs.

Usage
-----
::

    from data.sentiment import FearGreedClient, set_global_client, get_global_client

    client = FearGreedClient()
    snapshot = await client.get_latest()
    print(snapshot.value, snapshot.classification, snapshot.regime_boost)
    await client.close()

Global singleton pattern (used by StrategyEngine)::

    set_global_client(FearGreedClient())
    client = get_global_client()
    if client:
        snapshot = await client.get_latest()

API
---
Alternative.me Fear & Greed Index: https://api.alternative.me/fng/?limit=30
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "FearGreedSnapshot",
    "FearGreedClient",
    "set_global_client",
    "get_global_client",
]

logger = structlog.get_logger(__name__)

_FNG_API_URL = "https://api.alternative.me/fng/"
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_global_client: FearGreedClient | None = None


def set_global_client(client: FearGreedClient) -> None:
    """Register a FearGreedClient as the module-level singleton."""
    global _global_client
    _global_client = client


def get_global_client() -> FearGreedClient | None:
    """Return the module-level FearGreedClient singleton, or None if not set."""
    return _global_client


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class FearGreedSnapshot(BaseModel):
    """
    A single Fear & Greed Index data point.

    ``value`` is an integer in [0, 100]:
    - 0 --24:  Extreme Fear
    - 25 --44: Fear
    - 45 --55: Neutral
    - 56 --75: Greed
    - 76 --100: Extreme Greed

    ``regime_boost`` provides a contrarian confidence modifier for strategies:
    positive in fearful regimes (markets tend to be oversold), negative in
    extreme greed (contrarian signal that market is overextended).
    """

    model_config = {"frozen": True}

    value: int = Field(ge=0, le=100, description="Fear & Greed index value (0 --100)")
    classification: str = Field(description="Human-readable classification")
    timestamp: datetime = Field(description="UTC timestamp of the data point")

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_unix_timestamp(cls, v: Any) -> datetime:
        """Accept Unix epoch int/str as well as datetime objects."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v.astimezone(UTC)
        # Unix timestamp (int or string)
        try:
            return datetime.fromtimestamp(int(v), tz=UTC)
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"Cannot coerce timestamp {v!r}: {exc}") from exc

    @property
    def regime_boost(self) -> float:
        """
        Contrarian confidence modifier based on the index value.

        Mapping:
        - 0 --24  (Extreme Fear):  +0.15 boost  -- market likely oversold
        - 25 --44 (Fear):          +0.05 boost
        - 45 --55 (Neutral):        0.00 no adjustment
        - 56 --75 (Greed):         +0.05 boost  -- momentum continuation
        - 76 --100 (Extreme Greed): -0.10 penalty  -- contrarian at extreme

        Returns
        -------
        float
            A signed confidence modifier to be added to strategy confidence.
        """
        if self.value <= 24:
            return 0.15
        elif self.value <= 44:
            return 0.05
        elif self.value <= 55:
            return 0.0
        elif self.value <= 75:
            return 0.05
        else:
            return -0.10


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FearGreedClient:
    """
    Async HTTP client for the Alternative.me Fear & Greed Index API.

    Features
    --------
    - Lazy aiohttp session (only created on first request)
    - 6-hour in-process cache for both latest snapshot and history
    - Graceful error handling: returns None on network failure rather
      than raising (strategies must handle None gracefully)

    Parameters
    ----------
    cache_ttl_seconds:
        How long to cache responses before re-fetching. Default 6 hours.
    """

    def __init__(self, cache_ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._session: Any | None = None  # aiohttp.ClientSession, lazy

        # Cache: (data, fetched_at_monotonic)
        self._latest_cache: tuple[FearGreedSnapshot, float] | None = None
        self._history_cache: tuple[list[FearGreedSnapshot], float] | None = None

        self._log = structlog.get_logger(__name__).bind(
            component="fear_greed_client"
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
                    "aiohttp is required for FearGreedClient. "
                    "Add aiohttp>=3.9 to packages/data/pyproject.toml."
                ) from exc
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self._log.debug("fear_greed_client.closed")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, fetched_at: float) -> bool:
        return (time.monotonic() - fetched_at) < self._cache_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_latest(self) -> FearGreedSnapshot | None:
        """
        Fetch the most recent Fear & Greed Index snapshot.

        Returns the cached value if it is younger than the TTL.

        Returns
        -------
        FearGreedSnapshot or None
            None if the API is unreachable or returns invalid data.
        """
        if self._latest_cache is not None:
            snapshot, fetched_at = self._latest_cache
            if self._is_cache_valid(fetched_at):
                self._log.debug("fear_greed_client.cache_hit", type="latest")
                return snapshot

        try:
            session = await self._get_session()
            async with session.get(
                _FNG_API_URL,
                params={"limit": 1, "format": "json"},
            ) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            snapshot = self._parse_snapshot(data["data"][0])
            self._latest_cache = (snapshot, time.monotonic())
            self._log.info(
                "fear_greed_client.fetched",
                value=snapshot.value,
                classification=snapshot.classification,
            )
            return snapshot

        except Exception as exc:
            self._log.warning(
                "fear_greed_client.fetch_failed",
                error=str(exc),
                type="latest",
            )
            # Return stale cache if available rather than None
            if self._latest_cache is not None:
                self._log.debug("fear_greed_client.stale_cache_used")
                return self._latest_cache[0]
            return None

    async def get_history(self, limit: int = 30) -> list[FearGreedSnapshot]:
        """
        Fetch the last ``limit`` Fear & Greed Index data points.

        Returns the cached value if it is younger than the TTL.

        Parameters
        ----------
        limit:
            Number of historical data points to retrieve. Maximum 30.

        Returns
        -------
        list[FearGreedSnapshot]
            Empty list if the API is unreachable or returns invalid data.
        """
        if self._history_cache is not None:
            history, fetched_at = self._history_cache
            if self._is_cache_valid(fetched_at):
                self._log.debug("fear_greed_client.cache_hit", type="history")
                return history

        try:
            session = await self._get_session()
            async with session.get(
                _FNG_API_URL,
                params={"limit": min(limit, 30), "format": "json"},
            ) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            history = [self._parse_snapshot(item) for item in data.get("data", [])]
            self._history_cache = (history, time.monotonic())
            self._log.debug(
                "fear_greed_client.history_fetched",
                count=len(history),
            )
            return history

        except Exception as exc:
            self._log.warning(
                "fear_greed_client.fetch_failed",
                error=str(exc),
                type="history",
            )
            if self._history_cache is not None:
                return self._history_cache[0]
            return []

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @property
    def cached_value(self) -> int | None:
        """Return the most recent cached FGI value, or None if no cache."""
        if self._latest_cache is not None:
            return self._latest_cache[0].value
        return None

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_snapshot(self, item: dict[str, Any]) -> FearGreedSnapshot:
        """
        Parse one Alternative.me API response dict into a FearGreedSnapshot.

        Parameters
        ----------
        item:
            A single entry from the ``data`` array in the API response.

        Returns
        -------
        FearGreedSnapshot
        """
        return FearGreedSnapshot(
            value=int(item["value"]),
            classification=str(item.get("value_classification", "Unknown")),
            timestamp=item["timestamp"],
        )
