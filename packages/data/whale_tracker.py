"""
packages/data/whale_tracker.py
--------------------------------
Whale Alert API client for large on-chain transaction monitoring.

Monitors transactions above $1M USD to detect large capital flows between
exchanges and private wallets.  The directional net flow provides a proxy
for institutional accumulation vs. distribution:

- **Negative net_flow** (outflow from exchanges) → whales moving assets to
  cold storage / self-custody.  Historically associated with accumulation
  and reduced near-term sell pressure.
- **Positive net_flow** (inflow to exchanges) → whales depositing assets to
  exchanges.  Historically associated with distribution and heightened
  sell pressure.

Results are cached for 1 hour (3600 s) to balance freshness with API costs.
The Whale Alert free tier allows 10 calls/minute.

Usage
-----
::

    from data.whale_tracker import WhaleAlertClient, set_global_client, get_global_client

    client = WhaleAlertClient(api_key="your_whale_alert_api_key")
    snapshot = await client.get_latest()
    if snapshot:
        print(snapshot.net_flow, snapshot.large_tx_count)
    await client.close()

Global singleton pattern (used by StrategyEngine)::

    set_global_client(WhaleAlertClient(api_key=settings.whale_alert_api_key))
    client = get_global_client()
    if client:
        snapshot = await client.get_latest()

API
---
Whale Alert REST API: https://docs.whale-alert.io/
Register for a free API key at: https://whale-alert.io/
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

__all__ = [
    "WhaleFlowSnapshot",
    "WhaleAlertClient",
    "set_global_client",
    "get_global_client",
]

logger = structlog.get_logger(__name__)

_ENDPOINT = "https://api.whale-alert.io/v1/transactions"
_CACHE_TTL_SECONDS = 3600  # 1 hour
_MIN_VALUE_USD = 1_000_000  # Only consider transactions >= $1M
_LOOKBACK_SECONDS = 3600  # Last 1 hour of transactions

# Exchange address labels used by Whale Alert (partial; best-effort)
_EXCHANGE_LABELS = frozenset({
    "exchange",
    "binance",
    "coinbase",
    "kraken",
    "bitfinex",
    "huobi",
    "okx",
    "bybit",
    "kucoin",
    "gate.io",
    "bitstamp",
    "gemini",
})

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_global_client: WhaleAlertClient | None = None


def set_global_client(client: WhaleAlertClient) -> None:
    """Register a WhaleAlertClient as the module-level singleton."""
    global _global_client
    _global_client = client


def get_global_client() -> WhaleAlertClient | None:
    """Return the module-level WhaleAlertClient singleton, or None if not set."""
    return _global_client


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class WhaleFlowSnapshot(BaseModel):
    """
    A single whale transaction flow snapshot for a lookback window.

    Fields
    ------
    net_flow:
        Signed net USD value of large transactions in the lookback window.
        Positive = net inflow to exchanges (sell pressure).
        Negative = net outflow from exchanges (accumulation signal).
        Zero = balanced or no qualifying transactions.
    large_tx_count:
        Number of individual transactions above the minimum value threshold.
    timestamp:
        UTC timestamp when this snapshot was captured locally.
    """

    model_config = {"frozen": True}

    net_flow: float = Field(description="Net USD flow (positive=to exchange, negative=from exchange)")
    large_tx_count: int = Field(ge=0, description="Number of large transactions observed")
    timestamp: datetime = Field(description="UTC timestamp of this snapshot")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WhaleAlertClient:
    """
    Async HTTP client for the Whale Alert transactions API.

    Features
    --------
    - Lazy aiohttp session (only created on first request)
    - 1-hour in-process cache
    - Stale-cache fallback: returns last known value on network failure
    - Graceful degradation: returns None rather than raising on failure
    - Best-effort exchange detection via Whale Alert address type labels

    Parameters
    ----------
    api_key:
        Whale Alert API key from https://whale-alert.io/
    session:
        Optional pre-existing aiohttp.ClientSession.
    cache_ttl_seconds:
        How long to cache responses before re-fetching.  Default 1 hour.
    min_value_usd:
        Minimum USD transaction value to include.  Default $1,000,000.
    """

    def __init__(
        self,
        api_key: str,
        session: Any | None = None,
        cache_ttl_seconds: float = _CACHE_TTL_SECONDS,
        min_value_usd: int = _MIN_VALUE_USD,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl = cache_ttl_seconds
        self._min_value_usd = min_value_usd
        self._session: Any | None = session  # aiohttp.ClientSession, lazy

        # Cache: (snapshot, fetched_at_monotonic)
        self._latest_cache: tuple[WhaleFlowSnapshot, float] | None = None

        self._log = structlog.get_logger(__name__).bind(
            component="whale_alert_client"
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
                    "aiohttp is required for WhaleAlertClient. "
                    "Add aiohttp>=3.9 to packages/data/pyproject.toml."
                ) from exc
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and release resources."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self._log.debug("whale_alert_client.closed")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, fetched_at: float) -> bool:
        return (time.monotonic() - fetched_at) < self._cache_ttl

    @property
    def cached_value(self) -> WhaleFlowSnapshot | None:
        """Return the most recent cached snapshot, or None if no cache."""
        if self._latest_cache is not None:
            return self._latest_cache[0]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_latest(self) -> WhaleFlowSnapshot | None:
        """
        Fetch large transactions from the last hour and compute net flow.

        Returns the cached value if it is younger than the TTL.

        The net_flow is computed as::

            net_flow = sum(tx.amount_usd for tx going TO exchange)
                     - sum(tx.amount_usd for tx coming FROM exchange)

        A positive net_flow means more USD is flowing INTO exchanges
        (potential sell pressure); negative means outflow (accumulation).

        Returns
        -------
        WhaleFlowSnapshot or None
            None if the API is unreachable and no cached data is available.
        """
        if self._latest_cache is not None:
            snapshot, fetched_at = self._latest_cache
            if self._is_cache_valid(fetched_at):
                self._log.debug("whale_alert_client.cache_hit")
                return snapshot

        try:
            now_unix = int(datetime.now(UTC).timestamp())
            start_unix = now_unix - _LOOKBACK_SECONDS

            session = await self._get_session()
            params = {
                "api_key": self._api_key,
                "min_value": str(self._min_value_usd),
                "start": str(start_unix),
                "limit": "100",
            }
            async with session.get(_ENDPOINT, params=params) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            snapshot = self._aggregate_flow(data)
            self._latest_cache = (snapshot, time.monotonic())
            self._log.info(
                "whale_alert_client.fetched",
                net_flow=snapshot.net_flow,
                large_tx_count=snapshot.large_tx_count,
            )
            return snapshot

        except Exception as exc:
            self._log.warning(
                "whale_alert_client.fetch_failed",
                error=str(exc),
            )
            if self._latest_cache is not None:
                self._log.debug("whale_alert_client.stale_cache_used")
                return self._latest_cache[0]
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_exchange_address(self, address_info: dict[str, Any] | None) -> bool:
        """
        Return True when an address belongs to a known exchange.

        Whale Alert includes an ``owner_type`` field (``"exchange"``) and an
        ``owner`` field (exchange name).  We check both to maximise coverage.
        """
        if address_info is None:
            return False
        owner_type = str(address_info.get("owner_type", "")).lower()
        owner = str(address_info.get("owner", "")).lower()
        if owner_type == "exchange":
            return True
        # Fallback: check if the owner name contains an exchange keyword
        return any(exch in owner for exch in _EXCHANGE_LABELS)

    def _aggregate_flow(self, response: dict[str, Any]) -> WhaleFlowSnapshot:
        """
        Aggregate a Whale Alert transactions response into a flow snapshot.

        Parameters
        ----------
        response:
            The full JSON dict returned by the /v1/transactions endpoint.

        Returns
        -------
        WhaleFlowSnapshot
            net_flow = inflow_to_exchanges - outflow_from_exchanges (USD).
        """
        transactions: list[dict[str, Any]] = response.get("transactions", [])

        inflow: float = 0.0   # USD moving TO exchanges
        outflow: float = 0.0  # USD moving FROM exchanges
        count: int = 0

        for tx in transactions:
            try:
                amount_usd = float(tx.get("amount_usd", 0.0))
                if amount_usd < self._min_value_usd:
                    continue

                to_addr: dict[str, Any] | None = tx.get("to")
                from_addr: dict[str, Any] | None = tx.get("from")

                to_exchange = self._is_exchange_address(to_addr)
                from_exchange = self._is_exchange_address(from_addr)

                if to_exchange:
                    inflow += amount_usd
                elif from_exchange:
                    outflow += amount_usd

                count += 1
            except (TypeError, ValueError):
                # Skip malformed transaction records
                continue

        return WhaleFlowSnapshot(
            net_flow=round(inflow - outflow, 2),
            large_tx_count=count,
            timestamp=datetime.now(UTC),
        )
