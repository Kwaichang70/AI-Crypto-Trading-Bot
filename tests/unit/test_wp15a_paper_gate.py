"""
tests/unit/test_wp15a_paper_gate.py
-------------------------------------
WP1.5a -- the paper engine takes sizing and risk-gate inputs from the
portfolio's ``risk_snapshot()`` (D-WP15-A / D-WP15-B).

    WP15-T-13  spy on pre_trade_check / calculate_position_size; no-source path
    WP15-T-14  F3 regression: portfolio marked up, paper mark stale -> no false drawdown
    WP15-T-31  the paper gate honours the resumed peak_equity_hint        (red@HEAD)

The wiring goes through the real ``StrategyEngine`` constructor where HEAD
behaviour matters, so that a HEAD run fails on the *behaviour* and not on a
missing attribute.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, SignalDirection, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import D1, SYMBOL, Clock, D, make_fill, mark, portfolio_with_position
from trading.engines.paper import PaperExecutionEngine
from trading.models import Signal
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import StrategyEngine

_RUN = "wp15-paper"
_PRICE = Decimal("1000")
_RELAXED = {"max_portfolio_exposure_pct": 1.0, "max_cluster_exposure_pct": 1.0}


class _NoopStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="wp15_noop", description="test-only")
    requires_exit_manager = False

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        return []


class _SpyRiskManager(DefaultRiskManager):
    """Records what the engine feeds sizing and the gate; delegates to the real thing."""

    def __init__(self, params: RiskParameters | None = None) -> None:
        super().__init__(run_id=_RUN, params=params)
        self.gate_calls: list[dict[str, Any]] = []
        self.sizing_equities: list[Decimal] = []

    def pre_trade_check(self, **kwargs: Any):  # type: ignore[override]
        self.gate_calls.append(dict(kwargs))
        return super().pre_trade_check(**kwargs)

    def calculate_position_size(self, equity: Decimal, *args: Any, **kwargs: Any) -> Decimal:
        self.sizing_equities.append(equity)
        return super().calculate_position_size(equity, *args, **kwargs)


def _buy(target: str = "1000") -> Signal:
    return Signal(
        strategy_id="wp15",
        symbol=SYMBOL,
        direction=SignalDirection.BUY,
        target_position=Decimal(target),
        confidence=1.0,
    )


def _wire(
    portfolio: PortfolioAccounting,
    *,
    params: RiskParameters | None = None,
    attach_via_strategy_engine: bool = True,
) -> tuple[PaperExecutionEngine, _SpyRiskManager]:
    rm = _SpyRiskManager(params)
    execution = PaperExecutionEngine(_RUN, rm, initial_cash=portfolio.initial_cash)
    execution.set_last_price(SYMBOL, _PRICE)
    if attach_via_strategy_engine:
        StrategyEngine(
            strategies=[_NoopStrategy("wp15-noop")],
            execution_engine=execution,
            risk_manager=rm,
            market_data=MagicMock(),
            portfolio=portfolio,
            symbols=[SYMBOL],
            timeframe=TimeFrame.ONE_HOUR,
            run_mode=RunMode.PAPER,
        )
    return execution, rm


def _rejection_reasons(logs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for entry in logs:
        if entry["event"] == "paper.signal_rejected":
            out.extend(entry["reasons"])
    return out


# ---------------------------------------------------------------------------
# WP15-T-13
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nav_cash", "expected_sizing"),
    [
        (Decimal("20000"), Decimal("10000")),  # profit: sizes at initial (no compounding)
        (Decimal("10000"), Decimal("10000")),  # at par
        (Decimal("7000"), Decimal("7000")),  # drawdown: sizes at NAV
    ],
)
async def test_wp15_t13_gate_and_sizing_read_the_portfolio_snapshot(
    nav_cash: Decimal, expected_sizing: Decimal
) -> None:
    clock = Clock(D)
    portfolio = PortfolioAccounting(_RUN, Decimal("10000"), clock=clock)
    portfolio.update_market_prices({})  # day D
    portfolio._cash = nav_cash  # the portfolio's NAV differs from the engine-local ledger
    clock.at = D1
    portfolio.update_market_prices({})  # roll -> day start == nav_cash

    execution, rm = _wire(portfolio, params=RiskParameters(**_RELAXED, max_drawdown_pct=0.5))
    # an engine-LOCAL position the portfolio does not know about (C-09)
    execution.restore_from_fills([make_fill(quantity="1", price="500", executed_at=D)])
    local_positions = [p for p in execution._positions.values() if not p.is_flat]
    assert local_positions

    snap = portfolio.risk_snapshot()
    await execution.process_signal(_buy())

    assert len(rm.gate_calls) == 1, "exactly one risk-state read feeds the gate"
    call = rm.gate_calls[0]
    assert call["current_equity"] == snap.nav == nav_cash
    assert call["peak_equity"] == snap.peak_equity
    assert call["daily_pnl"] == snap.gate_daily_pnl(snap.nav)
    assert call["daily_loss_basis"] == snap.daily_loss_basis
    assert call["open_positions"] == local_positions  # engine-local (C-09)
    # BUY sizing basis: min(nav, initial_capital)  (D-WP15-B)
    assert rm.sizing_equities == [expected_sizing]


async def test_wp15_t13_sell_gate_uses_the_full_nav() -> None:
    clock = Clock(D)
    portfolio = portfolio_with_position(clock)
    mark(portfolio, "2000")  # NAV 20 000, initial 10 000
    execution, rm = _wire(portfolio, params=RiskParameters(**_RELAXED))
    execution.restore_from_fills([make_fill(quantity="10", price="1000", executed_at=D)])
    execution.set_last_price(SYMBOL, Decimal("2000"))

    sell = Signal(
        strategy_id="wp15",
        symbol=SYMBOL,
        direction=SignalDirection.SELL,
        target_position=Decimal("0"),
        confidence=1.0,
    )
    orders = await execution.process_signal(sell)
    assert len(orders) == 1
    assert rm.gate_calls[0]["current_equity"] == Decimal("20000")


async def test_wp15_t13_without_a_source_legacy_path_and_one_warning() -> None:
    rm = _SpyRiskManager(RiskParameters(**_RELAXED))
    execution = PaperExecutionEngine(_RUN, rm, initial_cash=Decimal("10000"))
    execution.set_last_price(SYMBOL, _PRICE)
    execution._cash = Decimal("20000")  # engine-local equity 20 000 > initial

    with capture_logs() as logs:
        await execution.process_signal(_buy("100"))
        await execution.process_signal(_buy("100"))

    missing = [e for e in logs if e["event"] == "paper.risk_state_source_missing"]
    assert len(missing) == 1
    assert missing[0]["log_level"] == "warning"
    # legacy gate call shape: no daily_loss_basis kwarg
    assert all("daily_loss_basis" not in c for c in rm.gate_calls)
    # D-WP15-B is mode-wide: BUY sizing is capped at initial capital here too
    assert rm.sizing_equities[0] == Decimal("10000")
    # the legacy gate input is still the engine-local equity
    assert rm.gate_calls[0]["current_equity"] == Decimal("20000")


# ---------------------------------------------------------------------------
# WP15-T-14  (F3)
# ---------------------------------------------------------------------------


async def test_wp15_t14_stale_paper_mark_does_not_create_a_false_drawdown() -> None:
    clock = Clock(D)
    portfolio = portfolio_with_position(clock)  # 10 x 1000, NAV 10 000
    mark(portfolio, "1300")  # +30%: NAV and peak 13 000
    execution, rm = _wire(
        portfolio,
        params=RiskParameters(**_RELAXED, max_drawdown_pct=0.15, max_position_size_pct=1.0),
    )
    # The paper engine's own ledger still values the position at the 1000 fill
    # price (it is never re-marked): engine-local equity 10 000.
    execution.restore_from_fills([make_fill(quantity="10", price="1000", executed_at=D)])
    execution.set_last_price(SYMBOL, Decimal("1300"))
    assert execution._get_current_equity() == Decimal("10000")

    with capture_logs() as logs:
        orders = await execution.process_signal(_buy("1300"))

    call = rm.gate_calls[0]
    assert call["current_equity"] == Decimal("13000")
    assert call["peak_equity"] == Decimal("13000")
    assert not [r for r in _rejection_reasons(logs) if "drawdown" in r.lower()]
    assert len(orders) == 1


# ---------------------------------------------------------------------------
# WP15-T-31  (red@HEAD)
# ---------------------------------------------------------------------------


async def test_wp15_t31_paper_gate_honours_the_resumed_peak_hint() -> None:
    portfolio = PortfolioAccounting.from_fills(
        _RUN,
        Decimal("10000"),
        [],
        peak_equity_hint=Decimal("15000"),  # 33% above the rebuilt NAV of 10 000
    )
    assert portfolio.get_peak_equity() == Decimal("15000")
    execution, _rm = _wire(portfolio, params=RiskParameters(**_RELAXED))

    with capture_logs() as logs:
        orders = await execution.process_signal(_buy("100"))

    assert orders == []
    assert any("drawdown" in r.lower() for r in _rejection_reasons(logs)), logs


async def test_wp15_t31_hint_below_the_limit_still_trades() -> None:
    portfolio = PortfolioAccounting.from_fills(
        _RUN, Decimal("10000"), [], peak_equity_hint=Decimal("12000")  # 16.7% < 30%
    )
    execution, _rm = _wire(portfolio, params=RiskParameters(**_RELAXED))
    assert len(await execution.process_signal(_buy("100"))) == 1
