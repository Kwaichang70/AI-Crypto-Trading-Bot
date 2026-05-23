"""
packages/data/fx_service.py
----------------------------
M6 MVP (Sprint 49) -- FX rate service skeleton with 24-hour TTL cache.

This module provides ``FxService``, a typed cache for foreign-exchange rates
between ``QuoteCurrency`` values.  In M6 MVP the ``get_rate`` method is a
stub that always returns ``None`` on cache miss; the in-memory cache and TTL
logic are implemented so M6b can wire in real CCXT calls without touching the
cache layer.

M6b will:
  - Implement ``_fetch_rate_from_ccxt()`` using the live exchange client.
  - Call ``_fetch_rate_from_ccxt()`` inside ``get_rate()`` on cache miss.
  - Wire ``FxCacheWarmer._tick()`` to pre-populate common pairs at startup.

Cache design
------------
* Key: ``(base, quote, date)`` — ``date`` defaults to today (UTC) when
  ``at_date`` is ``None``, so intraday calls collapse to a single entry.
* TTL: 24 hours (``_FX_CACHE_TTL_SECONDS``).  Historical rates are immutable
  once the date has passed, so the TTL guard only matters for today's rate.
* Thread safety: single-process asyncio; no locking required.

MIXED sentinel note (CR-006)
-----------------------------
``QuoteCurrency.MIXED`` is a sentinel for heterogeneous multi-symbol runs.
Passing it as ``base`` or ``quote`` to ``get_rate()`` is not meaningful;
behaviour for the MIXED sentinel is undefined. Callers should check for MIXED
before calling ``get_rate()``.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import NamedTuple

import structlog

from common.types import QuoteCurrency

__all__ = ["FxService", "FxRateUnavailableError"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (AR-M4-004 / M5 CR-002 naming convention)
# ---------------------------------------------------------------------------

_FX_CACHE_TTL_SECONDS: float = 86_400.0  # 24 hours


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class FxRateUnavailableError(Exception):
    """Raised when an FX rate cannot be obtained from any source.

    In M6 MVP this is never raised (``get_rate`` returns ``None`` on miss).
    M6b callers should catch this when they require a rate and cannot
    proceed without it.
    """


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------


class _CacheEntry(NamedTuple):
    rate: Decimal
    fetched_at: float  # time.monotonic() timestamp


# ---------------------------------------------------------------------------
# FxService
# ---------------------------------------------------------------------------


class FxService:
    """FX rate provider with 24-hour in-memory TTL cache.

    Parameters
    ----------
    ttl_seconds : float
        Cache time-to-live in seconds.  Defaults to ``_FX_CACHE_TTL_SECONDS``
        (24 hours).  Overridable in tests for fast TTL expiry.

    M6 MVP behaviour
    ----------------
    ``get_rate`` always returns ``None`` on cache miss because no real network
    source is wired yet.  The cache dict is populated via ``seed_rate()`` to
    verify TTL logic in tests, and will be populated by ``FxCacheWarmer`` in M6b.
    """

    def __init__(self, *, ttl_seconds: float = _FX_CACHE_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._cache: dict[tuple[QuoteCurrency, QuoteCurrency, date], _CacheEntry] = {}
        self._log = logger.bind(component="fx_service")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_rate(
        self,
        base: QuoteCurrency,
        quote: QuoteCurrency,
        at_date: date | None = None,
    ) -> Decimal | None:
        """Return the exchange rate for ``base`` expressed in ``quote`` units.

        Returns rate such that: 1 base = rate × quote.
        Example: ``get_rate(USDT, USD)`` → ``Decimal("0.9995")`` means
        1 USDT buys 0.9995 USD.

        Parameters
        ----------
        base:
            Source currency (the currency being converted FROM).
        quote:
            Target currency (the currency being converted TO).
        at_date:
            Date for the rate.  Defaults to today (UTC).  Historical dates
            return immutable rates; today's rate is subject to TTL.

        Returns
        -------
        Decimal | None
            The exchange rate, or ``None`` when unavailable.
            M6 MVP always returns ``None`` on cache miss (no network source wired).

        Notes
        -----
        If ``base is quote`` the identity rate ``Decimal("1")`` is returned
        immediately without a cache lookup.

        Passing ``QuoteCurrency.MIXED`` as ``base`` or ``quote`` is not
        meaningful; behaviour is undefined for the MIXED sentinel.
        Callers should check for MIXED before invoking this method.
        """
        if base is quote:
            return Decimal("1")

        effective_date = at_date or datetime.now(tz=UTC).date()
        cache_key = (base, quote, effective_date)

        cached = self._cache.get(cache_key)
        if cached is not None:
            age = time.monotonic() - cached.fetched_at
            if age < self._ttl_seconds:
                self._log.debug(
                    "fx_service.cache_hit",
                    base=base,
                    quote=quote,
                    rate=str(cached.rate),
                    age_seconds=round(age, 1),
                )
                return cached.rate
            # TTL expired — evict stale entry
            del self._cache[cache_key]
            self._log.debug(
                "fx_service.cache_expired",
                base=base,
                quote=quote,
                age_seconds=round(age, 1),
            )

        # M6 MVP: no network fetch implemented.  Return None.
        # M6b: call self._fetch_rate_from_ccxt(base, quote, effective_date) here.
        self._log.debug(
            "fx_service.cache_miss_no_source",
            base=base,
            quote=quote,
            at_date=str(effective_date),
        )
        return None

    def seed_rate(
        self,
        base: QuoteCurrency,
        quote: QuoteCurrency,
        rate: Decimal,
        at_date: date | None = None,
        *,
        fetched_at: float | None = None,
    ) -> None:
        """Manually seed a rate into the cache.

        Used by ``FxCacheWarmer`` (M6b) and by tests to pre-populate rates
        without touching the network.

        Parameters
        ----------
        fetched_at:
            ``time.monotonic()`` timestamp to use as the cache-entry age.
            Defaults to ``time.monotonic()`` (now).  Pass a past value in
            tests to simulate TTL expiry.
        """
        effective_date = at_date or datetime.now(tz=UTC).date()
        cache_key = (base, quote, effective_date)
        self._cache[cache_key] = _CacheEntry(
            rate=rate,
            fetched_at=fetched_at if fetched_at is not None else time.monotonic(),
        )
        self._log.debug(
            "fx_service.rate_seeded",
            base=base,
            quote=quote,
            rate=str(rate),
            at_date=str(effective_date),
        )

    def clear_cache(self) -> None:
        """Evict all cached entries.  Useful in tests."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the cache (including possibly-expired)."""
        return len(self._cache)
