"""
tests/unit/test_wp15a_live_gate.py
------------------------------------
WP1.5a -- the LIVE engine feeds the daily-loss gate from the portfolio's
``risk_snapshot()`` (mark-to-market, latched, fixed day-start basis).

    WP15-T-16  day-1 MTM loss rejects a BUY; the next UTC day (no fill) does not
               (red@HEAD: HEAD's accumulator only rolls on a fill -- C5)
    WP15-T-17  profit basis: NAV 2x initial -> threshold 8% of initial; a
               ticker spike never writes the portfolio's peak or latch (I-6)

Real ``LiveExecutionEngine`` + real ``PortfolioAccounting`` (injected clock) +
a mock CCXT exchange, exactly like ``test_live_nav_sizing.py``.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, SignalDirection
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import D1, Clock, D, make_fill
from tests.unit.test_live_nav_sizing import _make_engine, _make_signal, _market
from trading.models import Order, RiskCheckResult
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters, live_risk_parameters
from trading.risk_manager import DefaultRiskManager

BTC = "BTC/EUR"
ETH = "ETH/EUR"
_SYMBOLS = [BTC, ETH]
_MARKETS = {BTC: _market("BTC"), ETH: _market("ETH")}


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant)


def _install_clock(portfolio: PortfolioAccounting, clock: Clock) -> None:
    portfolio._clock = clock


def _fill(symbol: str, side: OrderSide, qty: str, price: str) -> Any:
    return make_fill(symbol=symbol, side=side, quantity=qty, price=price)


class _SpyRiskManager(DefaultRiskManager):
    """Approves everything; records what the live engine feeds the gate."""

    def __init__(self) -> None:
        super().__init__(run_id="wp15-live-spy")
        self.calls: list[dict[str, Any]] = []

    def pre_trade_check(self, order: Order, **kwargs: Any) -> RiskCheckResult:  # type: ignore[override]
        self.calls.append(dict(kwargs))
        return RiskCheckResult(
            approved=True, adjusted_quantity=order.quantity, rejection_reasons=[], warnings=[]
        )


def _reasons(logs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for entry in logs:
        if entry["event"] == "live.signal_rejected":
            out.extend(entry["reasons"])
    return out


# ---------------------------------------------------------------------------
# WP15-T-16  (red@HEAD)
# ---------------------------------------------------------------------------


async def test_wp15_t16_mtm_loss_blocks_buy_and_next_utc_day_unblocks_it() -> None:
    clock = Clock(D)
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=DefaultRiskManager(run_id="wp15-live", params=live_risk_parameters()),
        fetch_ticker_response={"last": "60"},
    )
    _install_clock(portfolio, clock)

    # day D: 3 BTC bought at 100, one sold at 60 (realised -40), mark falls to 60
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "3", "100"), Decimal("100"))
    portfolio.update_position(_fill(BTC, OrderSide.SELL, "1", "60"), Decimal("60"))
    portfolio.update_market_prices({BTC: Decimal("60")})
    assert portfolio.current_equity == Decimal("880")  # -120 = -12% on the day
    assert engine._calculate_daily_pnl() == Decimal("-40")  # realised display figure

    buy = _make_signal(symbol=ETH, direction=SignalDirection.BUY, target=Decimal("50"))
    with capture_logs() as logs:
        assert await engine.process_signal(buy) == []
    assert any("daily loss" in r.lower() for r in _reasons(logs)), logs

    # UTC midnight passes; NO fill and NO mark happens: the read already rolls
    clock.at = D1
    assert engine._calculate_daily_pnl() == Decimal("0")
    with capture_logs() as logs:
        orders = await engine.process_signal(buy)
    assert len(orders) == 1, _reasons(logs)
    assert portfolio.risk_snapshot().day_start_source == "roll"


async def test_wp15_t16_unrecovered_latch_still_blocks_after_a_partial_recovery() -> None:
    clock = Clock(D)
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=DefaultRiskManager(run_id="wp15-live", params=live_risk_parameters()),
        fetch_ticker_response={"last": "97"},
    )
    _install_clock(portfolio, clock)
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "3", "100"), Decimal("100"))
    portfolio.update_market_prices({BTC: Decimal("60")})  # -120: latched
    portfolio.update_market_prices({BTC: Decimal("97")})  # NAV back to -9
    assert portfolio.risk_snapshot().nav_pnl == Decimal("-9")

    buy = _make_signal(symbol=ETH, direction=SignalDirection.BUY, target=Decimal("50"))
    with capture_logs() as logs:
        assert await engine.process_signal(buy) == []
    assert any("daily loss" in r.lower() for r in _reasons(logs))


# ---------------------------------------------------------------------------
# WP15-T-17
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mark", "blocked"), [("284", True), ("284.4", False)])
async def test_wp15_t17_profit_basis_is_capped_at_initial_capital(mark: str, blocked: bool) -> None:
    clock = Clock(D)
    relaxed = RiskParameters(
        max_position_size_pct=1.0, max_portfolio_exposure_pct=1.0, max_cluster_exposure_pct=1.0
    )
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=DefaultRiskManager(run_id="wp15-live", params=relaxed),
        fetch_ticker_response={"last": mark},
    )
    _install_clock(portfolio, clock)
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "5", "100"), Decimal("100"))
    portfolio.update_market_prices({BTC: Decimal("300")})  # 500 cash + 1500 = 2000 = 2x initial
    clock.at = D1
    portfolio.update_market_prices({BTC: Decimal("300")})  # roll: day start 2000
    snap = portfolio.risk_snapshot()
    assert snap.nav_day_start == Decimal("2000")
    assert snap.daily_loss_basis == Decimal("1000")  # threshold = 8% of INITIAL = 80
    portfolio.update_market_prices({BTC: Decimal(mark)})  # NAV 1920 / 1922 (-80 / -78)

    buy = _make_signal(symbol=ETH, direction=SignalDirection.BUY, target=Decimal("50"))
    with capture_logs() as logs:
        orders = await engine.process_signal(buy)
    if blocked:  # -80 == -threshold: the boundary blocks
        assert orders == []
        assert any("daily loss" in r.lower() for r in _reasons(logs))
    else:  # -78 is inside the limit
        assert len(orders) == 1, _reasons(logs)


async def test_wp15_t17_ticker_spike_never_writes_peak_or_latch() -> None:
    spy = _SpyRiskManager()
    clock = Clock(D)
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=spy,
        fetch_ticker_response={"last": "400"},  # spike: portfolio mark is 300
    )
    _install_clock(portfolio, clock)
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "5", "100"), Decimal("100"))
    portfolio.update_market_prices({BTC: Decimal("300")})
    peak_before = portfolio.get_peak_equity()
    worst_before = portfolio._worst_nav_pnl
    state_before = (
        portfolio._nav_day_start, portfolio._daily_pnl, len(portfolio.get_equity_curve())
    )

    buy = _make_signal(symbol=BTC, direction=SignalDirection.BUY, target=Decimal("50"))
    await engine.process_signal(buy)

    call = spy.calls[0]
    assert call["current_equity"] == Decimal("2500")  # ticker-adjusted NAV was gated
    assert call["peak_equity"] == Decimal("2500")
    assert call["daily_loss_basis"] == Decimal("1000")  # basis from day start, not the spike
    assert portfolio.get_peak_equity() == peak_before == Decimal("2000")
    assert portfolio._worst_nav_pnl == worst_before
    assert (
        portfolio._nav_day_start, portfolio._daily_pnl, len(portfolio.get_equity_curve())
    ) == state_before


async def test_wp15_t17_sell_gate_value_uses_the_portfolio_nav() -> None:
    spy = _SpyRiskManager()
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=spy,
        fetch_ticker_response={"last": "60"},
    )
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "3", "100"), Decimal("100"))
    portfolio.update_market_prices({BTC: Decimal("60")})  # -120 latched

    sell = _make_signal(symbol=BTC, direction=SignalDirection.SELL, target=Decimal("0"))
    orders = await engine.process_signal(sell)
    assert len(orders) == 1  # the spy approves; what matters is the gate input
    call = spy.calls[0]
    assert call["daily_pnl"] == Decimal("-120")
    assert call["daily_loss_basis"] == Decimal("1000")
    assert call["current_equity"] == Decimal("880")


def test_wp15_t19_equity_snapshot_and_risk_snapshot_reads_never_write() -> None:
    clock = Clock(D)
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"),
        symbols=_SYMBOLS,
        markets=_MARKETS,
        risk_manager=_SpyRiskManager(),
    )
    _install_clock(portfolio, clock)
    portfolio.update_position(_fill(BTC, OrderSide.BUY, "5", "100"), Decimal("100"))
    portfolio.update_market_prices({BTC: Decimal("90")})
    state = (
        portfolio._peak_equity, portfolio._worst_nav_pnl, portfolio._nav_day_start,
        portfolio._daily_pnl, portfolio._daily_pnl_date, len(portfolio.get_equity_curve()),
    )
    for last in ("1", "90", "5000"):  # incl. a wild ticker
        for _ in range(3):
            assert engine._equity_snapshot(BTC, Decimal(last)) is not None or last == "1"
            portfolio.risk_snapshot().gate_daily_pnl(Decimal(last))
    assert state == (
        portfolio._peak_equity, portfolio._worst_nav_pnl, portfolio._nav_day_start,
        portfolio._daily_pnl, portfolio._daily_pnl_date, len(portfolio.get_equity_curve()),
    )
