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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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
# Module-level singleton — DEPRECATED: sunset by Sprint 41.
#
# Since Sprint 40 Stap 1c the canonical owner is
# ``app.state.container.services.fgi_client``; main.py's lifespan writes to
# both the container AND this module global during the transition period.
# New code should prefer the container lookup (see apps/api/deps.py helpers
# or apps/api/routers/signals.py::_resolve_service).
# ---------------------------------------------------------------------------
_global_client: FearGreedClient | None = None


def set_global_client(client: FearGreedClient) -> None:
    """Register a FearGreedClient as the module-level singleton.

    .. deprecated:: Sprint 40
       Callers should register the client on
       ``AppContainer.services.fgi_client`` via the API lifespan instead.
       This shim is retained for backwards compatibility until Sprint 41.
    """
    global _global_client
    _global_client = client


def get_global_client() -> FearGreedClient | None:
    """Return the module-level FearGreedClient singleton, or None if not set.

    .. deprecated:: Sprint 40
       Prefer reading from
       ``request.app.state.container.services.fgi_client`` in request-scoped
       code.  This shim is removed in Sprint 41.
    """
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

    async def get_history_as_series(
        self, limit: int = 30,
    ) -> "pd.Series":
        """Return the historical FGI values as a pandas Series indexed by
        timestamp.

        Thin adapter over :meth:`get_history` -- no new I/O, no new caching.
        The returned Series uses ``int64`` values and a UTC-aware
        ``DatetimeIndex`` sorted ascending so it can be passed directly
        to :func:`build_extended_feature_matrix(fgi_series=...)`.

        Parameters
        ----------
        limit:
            Number of historical points to fetch (capped at 30 by the
            underlying API).  The ``limit`` parameter only takes effect
            on a cache miss; a cached response with more points than
            ``limit`` will return all cached points (see ``get_history``
            cache contract).

        Returns
        -------
        pd.Series:
            Empty Series with ``dtype=int64`` and UTC-aware index when the
            API is unreachable.  Callers must NOT rely on the result being
            non-empty; the feature matrix builder gracefully handles an
            empty series by falling back to neutral defaults.
        """
        # Lazy pandas import keeps sentiment.py free of the ~150ms / 60MB
        # startup cost in processes that never call this method.
        import pandas as pd

        snapshots = await self.get_history(limit=limit)
        if not snapshots:
            return pd.Series(
                dtype="int64",
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
                name="fear_greed_index",
            )

        # API returns newest-first; sort ascending so shift(freq=...) in
        # the feature builder operates on a monotonically increasing index.
        ordered = sorted(snapshots, key=lambda s: s.timestamp)
        index = pd.DatetimeIndex(
            [s.timestamp for s in ordered],
            tz="UTC",
            name="timestamp",
        )
        return pd.Series(
            [int(s.value) for s in ordered],
            index=index,
            dtype="int64",
            name="fear_greed_index",
        )

    async def get_value_at_offset(self, days_ago: int) -> int | None:
        """Return the FGI value approximately ``days_ago`` days before
        ``utcnow()``, or None when no usable history is available.

        Used by :meth:`StrategyEngine._build_mtf_context` to populate
        ``MultiTimeframeContext.fear_greed_index_7d_ago`` for the v2
        feature builder (see QT-007 Sprint 46).

        Selection rule
        --------------
        The method picks the snapshot whose timestamp is closest to
        ``utcnow() - days_ago`` (in absolute seconds).  This handles
        alternative.me's sometimes-irregular daily publishing schedule
        (skipped days / different posting times) without falsely
        returning None when an exact match is absent.

        API ceiling
        -----------
        The Alternative.me API returns at most 30 data points (roughly
        the last 30 days).  Values of ``days_ago`` greater than ~28 may
        return ``None`` not due to a data gap but due to the hard API
        limit; a WARNING is logged in that case so the caller can
        distinguish capability limits from real gaps.

        Cache caveat
        ------------
        ``get_history``'s cache is not keyed by ``limit``.  If a prior
        call populated the cache with a smaller ``limit`` than required
        here, the closest-match search may miss and return ``None`` --
        a DEBUG log records the cache shortfall when this happens.

        Parameters
        ----------
        days_ago:
            Non-negative number of days back from now.  Must be >= 1
            (calling with 0 should use :attr:`cached_value` instead).

        Returns
        -------
        int | None:
            FGI value in [0, 100], or None when history is empty,
            unreachable, or the closest match is more than 2 days off
            (suggesting a data gap that the caller should treat as
            "missing" rather than substitute a stale value).
        """
        if days_ago < 1:
            raise ValueError(f"days_ago must be >= 1, got {days_ago}")

        # API ceiling guard: warn the caller when the request exceeds
        # what the upstream service can ever return.
        if days_ago + 2 > 30:
            self._log.warning(
                "fear_greed_client.value_at_offset_beyond_api_limit",
                days_ago=days_ago,
                api_max_days=30,
            )

        # Fetch one more than days_ago so we have headroom for the closest-
        # match search even if recent days were skipped.  Capped at 30.
        fetch_limit = min(30, days_ago + 2)
        snapshots = await self.get_history(limit=fetch_limit)
        if not snapshots:
            return None

        # Cache-length diagnostic: a prior get_history(limit=K) populated
        # the global cache with K points; if K < days_ago we may silently
        # miss the match -- emit a DEBUG log so operators can spot this
        # in incident postmortems.
        if len(snapshots) < days_ago:
            self._log.debug(
                "fear_greed_client.value_at_offset_insufficient_cache",
                cache_len=len(snapshots),
                days_ago=days_ago,
            )

        target = datetime.now(tz=UTC) - timedelta(days=days_ago)
        best = min(
            snapshots,
            key=lambda s: abs((s.timestamp - target).total_seconds()),
        )
        delta_days = abs((best.timestamp - target).total_seconds()) / 86400.0
        # 2-day tolerance: alternative.me occasionally skips a publish day.
        # Returning a value that is 3+ days off would mask a real data gap.
        if delta_days > 2.0:
            self._log.debug(
                "fear_greed_client.value_at_offset_no_match",
                days_ago=days_ago,
                closest_delta_days=round(delta_days, 2),
            )
            return None
        return int(best.value)

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
