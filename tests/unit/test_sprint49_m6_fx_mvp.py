"""
tests/unit/test_sprint49_m6_fx_mvp.py
----------------------------------------
Sprint 49 M6 MVP: FxService cache TTL, QuoteCurrency enum,
BacktestRunner quote detection, FxCacheWarmer lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from common.types import QuoteCurrency
from data.fx_service import FxRateUnavailableError, FxService
from trading.backtest import _detect_quote_currency


# ---------------------------------------------------------------------------
# Class 1: QuoteCurrency enum
# ---------------------------------------------------------------------------


class TestQuoteCurrencyEnum:
    def test_known_values_exist(self) -> None:
        assert QuoteCurrency.USDT == "USDT"
        assert QuoteCurrency.USD == "USD"
        assert QuoteCurrency.USDC == "USDC"
        assert QuoteCurrency.EUR == "EUR"
        assert QuoteCurrency.MIXED == "MIXED"

    def test_is_str_enum(self) -> None:
        assert isinstance(QuoteCurrency.USD, str)

    def test_mixed_membership(self) -> None:
        """MIXED is a member of QuoteCurrency."""
        assert QuoteCurrency.MIXED in QuoteCurrency

    def test_all_values_uppercase(self) -> None:
        """All enum values are uppercase strings (not auto() lowercase)."""
        for member in QuoteCurrency:
            assert member.value == member.value.upper(), (
                f"{member.name}.value should be uppercase, got {member.value!r}"
            )


# ---------------------------------------------------------------------------
# Class 2: FxService — identity, cache hit/miss, TTL expiry
# ---------------------------------------------------------------------------


class TestFxServiceCache:
    @pytest.fixture
    def svc(self) -> FxService:
        return FxService(ttl_seconds=60.0)

    @pytest.mark.asyncio
    async def test_same_currency_returns_one(self, svc: FxService) -> None:
        rate = await svc.get_rate(QuoteCurrency.USD, QuoteCurrency.USD)
        assert rate == Decimal("1")

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none_in_mvp(self, svc: FxService) -> None:
        rate = await svc.get_rate(QuoteCurrency.USDT, QuoteCurrency.USD)
        assert rate is None

    @pytest.mark.asyncio
    async def test_seeded_rate_is_returned(self, svc: FxService) -> None:
        today = datetime.now(tz=UTC).date()
        svc.seed_rate(QuoteCurrency.USDT, QuoteCurrency.USD, Decimal("0.9998"))
        rate = await svc.get_rate(QuoteCurrency.USDT, QuoteCurrency.USD, at_date=today)
        assert rate == Decimal("0.9998")

    @pytest.mark.asyncio
    async def test_expired_cache_returns_none(self, svc: FxService) -> None:
        # Seed with a fetched_at timestamp 120 s in the past (TTL=60 s)
        past_time = time.monotonic() - 120.0
        svc.seed_rate(
            QuoteCurrency.EUR,
            QuoteCurrency.USD,
            Decimal("1.08"),
            fetched_at=past_time,
        )
        rate = await svc.get_rate(QuoteCurrency.EUR, QuoteCurrency.USD)
        assert rate is None  # expired; no refetch in M6 MVP

    def test_cache_size_reflects_entries(self, svc: FxService) -> None:
        svc.seed_rate(QuoteCurrency.GBP, QuoteCurrency.USD, Decimal("1.27"))
        assert svc.cache_size == 1
        svc.clear_cache()
        assert svc.cache_size == 0


# ---------------------------------------------------------------------------
# Class 3: FxRateUnavailableError
# ---------------------------------------------------------------------------


class TestFxServiceUnavailable:
    def test_fx_rate_unavailable_error_is_exception(self) -> None:
        assert issubclass(FxRateUnavailableError, Exception)

    def test_fx_rate_unavailable_error_is_importable(self) -> None:
        from data.fx_service import FxRateUnavailableError as _E  # noqa: PLC0415
        assert _E is FxRateUnavailableError


# ---------------------------------------------------------------------------
# Class 4: _detect_quote_currency (CR-004 extracted function)
# ---------------------------------------------------------------------------


class TestDetectQuoteCurrency:
    def test_empty_symbols_returns_none(self) -> None:
        assert _detect_quote_currency([]) is None

    def test_single_usdt_symbol(self) -> None:
        assert _detect_quote_currency(["BTC/USDT"]) == "USDT"

    def test_single_usd_symbol(self) -> None:
        assert _detect_quote_currency(["BTC/USD"]) == "USD"

    def test_multiple_same_quote(self) -> None:
        assert _detect_quote_currency(["BTC/USDT", "ETH/USDT"]) == "USDT"

    def test_mixed_quotes_returns_mixed(self) -> None:
        result = _detect_quote_currency(["BTC/USDT", "ETH/USD"])
        assert result == QuoteCurrency.MIXED
        assert result == "MIXED"

    def test_symbol_without_slash_returns_none(self) -> None:
        assert _detect_quote_currency(["BTCUSD"]) is None

    def test_unknown_ticker_passes_through_verbatim(self) -> None:
        # BRL is a valid ISO-4217 ticker not in the enum — should pass through
        result = _detect_quote_currency(["BTC/BRL"])
        assert result == "BRL"

    def test_mixed_slash_and_no_slash_skips_no_slash(self) -> None:
        # BTCUSD skipped; BTC/USDT provides the quote
        result = _detect_quote_currency(["BTCUSD", "ETH/USDT"])
        assert result == "USDT"


# ---------------------------------------------------------------------------
# Class 5: FxCacheWarmer lifecycle
# ---------------------------------------------------------------------------


class TestFxCacheWarmerLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running_true(self) -> None:
        from api.services.fx_cache_warmer import FxCacheWarmer  # noqa: PLC0415

        warmer = FxCacheWarmer(startup_delay_seconds=9999.0)
        await warmer.start()
        assert warmer.running is True
        await warmer.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        from api.services.fx_cache_warmer import FxCacheWarmer  # noqa: PLC0415

        warmer = FxCacheWarmer(startup_delay_seconds=9999.0)
        await warmer.start()
        await warmer.stop()
        assert warmer.running is False

    @pytest.mark.asyncio
    async def test_tick_updates_diagnostics(self) -> None:
        from api.services.fx_cache_warmer import FxCacheWarmer  # noqa: PLC0415

        warmer = FxCacheWarmer()
        assert warmer.last_run_at is None
        await warmer._tick()
        assert warmer.last_run_at is not None
        assert warmer.last_rates_cached == 0  # M6 MVP stub

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self) -> None:
        from api.services.fx_cache_warmer import FxCacheWarmer  # noqa: PLC0415

        warmer = FxCacheWarmer(startup_delay_seconds=9999.0)
        await warmer.start()
        task_id = id(warmer._task)
        await warmer.start()  # second call should be a no-op
        assert id(warmer._task) == task_id
        await warmer.stop()
