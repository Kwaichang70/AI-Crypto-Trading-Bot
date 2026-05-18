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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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
_MARKET_CAP_CHART_ENDPOINT = "https://api.coingecko.com/api/v3/global/market_cap_chart"
_CACHE_TTL_SECONDS = 1800  # 30 minutes

# ---------------------------------------------------------------------------
# Module-level singleton — DEPRECATED: sunset by Sprint 41.
#
# Since Sprint 40 Stap 1c the canonical owner is
# ``app.state.container.services.coingecko_client``; main.py's lifespan
# writes to both the container AND this module global during the transition
# period.  New code should prefer the container lookup (see
# apps/api/deps.py helpers or apps/api/routers/signals.py::_resolve_service).
# ---------------------------------------------------------------------------
_global_client: CoinGeckoClient | None = None


def set_global_client(client: CoinGeckoClient) -> None:
    """Register a CoinGeckoClient as the module-level singleton.

    .. deprecated:: Sprint 40
       Callers should register the client on
       ``AppContainer.services.coingecko_client`` via the API lifespan.
       Retained for backwards compatibility until Sprint 41.
    """
    global _global_client
    _global_client = client


def get_global_client() -> CoinGeckoClient | None:
    """Return the module-level CoinGeckoClient singleton, or None if not set.

    .. deprecated:: Sprint 40
       Prefer reading from
       ``request.app.state.container.services.coingecko_client`` in
       request-scoped code.  Removed in Sprint 41.
    """
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

        # QT-007e (Sprint 46): Historical BTC dominance series state.
        # ``_btc_dom_history_is_manual = True`` means the series was injected
        # via ``set_btc_dominance_history`` and is sticky (no TTL expiry,
        # no overwrite by failed network fetch).  A successful network
        # fetch via the PRO endpoint clears the flag and replaces the data.
        self._btc_dom_history: pd.Series | None = None
        self._btc_dom_history_fetched_at: float | None = None
        self._btc_dom_history_is_manual: bool = False

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

    def set_btc_dominance_history(self, series: pd.Series) -> None:
        """Inject a manually-curated historical BTC dominance series.

        Free-tier CoinGecko does not expose ``/global/market_cap_chart``
        (PRO subscription only).  Operators who want the v2
        ``btc_dom_delta_7d`` feature without upgrading can load a CSV
        snapshot (e.g. from CoinMarketCap, Glassnode, or a manual export)
        and inject it here.

        Manual overrides are STICKY: they survive TTL expiry and are
        only replaced by another call to ``set_btc_dominance_history``
        OR a successful network fetch via ``fetch_btc_dominance_history``
        on a PRO-enabled account.

        Expected series shape:
          - Index: UTC-aware DatetimeIndex (daily granularity typical).
          - Values: float dominance percentage in [0, 100].
          - Sorted ascending; monotonic.

        Parameters
        ----------
        series:
            Pre-prepared dominance series.  Stored by reference; callers
            should not mutate it after handing off.

        Raises
        ------
        ValueError:
            If the index is not a tz-aware monotonically-increasing
            DatetimeIndex.
        """
        import pandas as pd

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series.index must be a DatetimeIndex")
        if series.index.tz is None:
            raise ValueError("series.index must be tz-aware (UTC)")
        if not series.index.is_monotonic_increasing:
            raise ValueError("series.index must be monotonically increasing")

        self._btc_dom_history = series
        self._btc_dom_history_fetched_at = time.monotonic()
        self._btc_dom_history_is_manual = True
        self._log.info(
            "coingecko_client.btc_dom_history_set",
            n_points=len(series),
            source="manual",
        )

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

    async def fetch_btc_dominance_history(
        self,
        days: int = 30,
    ) -> pd.Series:
        """Fetch the BTC dominance percentage time-series for the last
        ``days`` days.

        Resolution order
        ----------------
        1. If a manual series was injected via
           ``set_btc_dominance_history``, return it WITHOUT TTL expiry --
           manual overrides are sticky until a subsequent call to
           ``set_btc_dominance_history`` or a successful network fetch
           replaces them.
        2. If a previous network fetch is within the 30-minute TTL,
           return the cached series.
        3. Try the PRO endpoint ``/global/market_cap_chart``.  Returns a
           ``pd.Series`` indexed by UTC timestamp.  On success the
           manual-override flag (if any) is cleared.
        4. Free tier: the PRO endpoint returns HTTP 401 / 403 / 404 / 422.
           Log a WARNING and return an empty Series so the v2 feature
           pipeline gracefully falls back to the neutral default
           (``btc_dom_delta_7d = 0.0``).

        Rate-limit handling: HTTP 429 is NOT degraded silently -- it
        raises through ``raise_for_status`` and is caught by the broad
        exception handler, which returns stale cache (if any) or empty.

        Parameters
        ----------
        days:
            Number of days of history to fetch.  CoinGecko PRO supports
            up to 365 days on the free Demo plan, longer on paid plans.

        Returns
        -------
        pd.Series:
            Empty Series with ``dtype=float64`` and UTC-aware
            ``DatetimeIndex`` when the endpoint is unreachable.
        """
        import pandas as pd  # noqa: F401  # used by type checker via TYPE_CHECKING

        # 1. Manual override -- sticky, no TTL.
        if self._btc_dom_history is not None and self._btc_dom_history_is_manual:
            self._log.debug(
                "coingecko_client.btc_dom_history_cache_hit",
                source="manual",
            )
            return self._btc_dom_history

        # 2. Network cache hit -- TTL applies.
        if (
            self._btc_dom_history is not None
            and self._btc_dom_history_fetched_at is not None
            and self._is_cache_valid(self._btc_dom_history_fetched_at)
        ):
            self._log.debug(
                "coingecko_client.btc_dom_history_cache_hit",
                source="network_cached",
            )
            return self._btc_dom_history

        # 3. PRO endpoint
        try:
            session = await self._get_session()
            async with session.get(
                _MARKET_CAP_CHART_ENDPOINT,
                params={"vs_currency": "usd", "days": str(days)},
            ) as response:
                if response.status in (401, 403, 404, 422):
                    self._log.warning(
                        "coingecko_client.btc_dom_history_pro_only",
                        status=response.status,
                        endpoint=_MARKET_CAP_CHART_ENDPOINT,
                        msg=(
                            "Historical BTC dominance requires CoinGecko PRO. "
                            "Use set_btc_dominance_history(series) to inject "
                            "a manual CSV, or accept the v2 btc_dom_delta_7d "
                            "feature defaulting to 0.0."
                        ),
                    )
                    return self._empty_dom_series()
                response.raise_for_status()
                payload = await response.json(content_type=None)

            series = self._parse_dom_history(payload, log=self._log)
            # Log if we are replacing a manual override with network data.
            if self._btc_dom_history_is_manual:
                self._log.info(
                    "coingecko_client.btc_dom_history_replaced_manual_override",
                    n_points=len(series),
                    days_requested=days,
                )
            self._btc_dom_history = series
            self._btc_dom_history_fetched_at = time.monotonic()
            self._btc_dom_history_is_manual = False
            self._log.info(
                "coingecko_client.btc_dom_history_fetched",
                n_points=len(series),
                days_requested=days,
            )
            return series

        except Exception as exc:
            self._log.warning(
                "coingecko_client.btc_dom_history_fetch_failed",
                error=str(exc),
            )
            if self._btc_dom_history is not None:
                self._log.debug("coingecko_client.btc_dom_history_stale_cache_used")
                return self._btc_dom_history
            return self._empty_dom_series()

    @staticmethod
    def _empty_dom_series() -> pd.Series:
        """Return an empty UTC-indexed float64 Series."""
        import pandas as pd
        return pd.Series(
            dtype="float64",
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            name="btc_dominance",
        )

    @staticmethod
    def _parse_dom_history(
        payload: dict[str, Any],
        *,
        log: Any | None = None,
    ) -> pd.Series:
        """Parse CoinGecko ``/global/market_cap_chart`` response into a
        dominance percentage time-series.

        Response schema (PRO endpoint)::

            {
              "market_cap_chart": {
                "market_cap": [[ts_ms, total_usd], ...],
                "btc_dominance": [[ts_ms, percentage], ...]
              }
            }

        Falls back to deriving dominance from raw BTC market cap and
        total market cap when ``btc_dominance`` is absent (some PRO
        tiers omit this convenience field).  Misaligned timestamps are
        dropped via ``dropna`` -- the count is logged when a logger is
        supplied so production debugging can spot data-quality issues.
        """
        import pandas as pd

        chart: dict[str, Any] = payload.get("market_cap_chart") or payload

        # Path A: btc_dominance series present (preferred).
        dom_rows = chart.get("btc_dominance")
        if isinstance(dom_rows, list) and dom_rows:
            timestamps = pd.to_datetime(
                [int(row[0]) for row in dom_rows],
                unit="ms",
                utc=True,
            )
            values = [float(row[1]) for row in dom_rows]
            return pd.Series(
                values,
                index=pd.DatetimeIndex(timestamps, name="timestamp"),
                dtype="float64",
                name="btc_dominance",
            ).sort_index()

        # Path B: derive from BTC market cap / total market cap.
        market_cap_rows = chart.get("market_cap") or []
        btc_market_cap_rows = chart.get("btc_market_cap") or []
        if not market_cap_rows or not btc_market_cap_rows:
            return pd.Series(
                dtype="float64",
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
                name="btc_dominance",
            )

        total = pd.Series(
            [float(row[1]) for row in market_cap_rows],
            index=pd.to_datetime(
                [int(row[0]) for row in market_cap_rows],
                unit="ms", utc=True,
            ),
            dtype="float64",
        )
        btc = pd.Series(
            [float(row[1]) for row in btc_market_cap_rows],
            index=pd.to_datetime(
                [int(row[0]) for row in btc_market_cap_rows],
                unit="ms", utc=True,
            ),
            dtype="float64",
        )
        joined = pd.concat([btc.rename("btc"), total.rename("total")], axis=1)
        rows_before = len(joined)
        joined = joined.dropna()
        rows_after = len(joined)
        if log is not None and rows_after < rows_before:
            log.warning(
                "coingecko_client.btc_dom_history_path_b_rows_dropped",
                dropped=rows_before - rows_after,
                total_before=rows_before,
            )
        joined["dominance"] = (joined["btc"] / joined["total"]) * 100.0
        result = joined["dominance"].sort_index()
        result.index = pd.DatetimeIndex(result.index, name="timestamp")
        result.name = "btc_dominance"
        return result

    def btc_dominance_at_offset_from_cache(self, days_ago: int) -> float | None:
        """Synchronous accessor: return the BTC dominance percentage
        approximately ``days_ago`` days before now, reading ONLY from
        ``_btc_dom_history``.  Never performs I/O.

        Used by :meth:`StrategyEngine._build_mtf_context` for the
        ``btc_dominance_7d_ago`` field of :class:`MultiTimeframeContext`.

        Returns ``None`` when no history is cached or when the closest
        cached datum is more than 2 days off the target.

        Parameters
        ----------
        days_ago:
            Non-negative number of days back from now.  Must be >= 1.

        Returns
        -------
        float | None:
            Dominance percentage [0, 100], or None.
        """
        if days_ago < 1:
            raise ValueError(f"days_ago must be >= 1, got {days_ago}")
        if self._btc_dom_history is None or self._btc_dom_history.empty:
            return None
        # pandas is a heavy import; sys.modules cache amortises repeated
        # lookups on this hot path (called once per bar from StrategyEngine).
        import pandas as pd

        target = datetime.now(tz=UTC) - timedelta(days=days_ago)
        target_ts = pd.Timestamp(target)

        idx = self._btc_dom_history.index
        deltas = pd.Series(
            (idx - target_ts).total_seconds(), index=idx,
        ).abs()
        closest_pos = int(deltas.argmin())
        closest_ts = idx[closest_pos]
        delta_days = abs((closest_ts - target_ts).total_seconds()) / 86400.0
        if delta_days > 2.0:
            return None
        return float(self._btc_dom_history.iloc[closest_pos])

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
