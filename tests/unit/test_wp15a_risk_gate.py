"""
tests/unit/test_wp15a_risk_gate.py
------------------------------------
WP1.5a (Verbeterplan v2 row 1.5) -- the daily-loss gate arithmetic and R-02.

    WP15-T-09  _check_daily_loss: inclusive boundary, fixed basis, fail-closed
    WP15-T-10  latch behaviour is BUY-only (WP1.2 _ENTRY_ONLY_RULES unchanged)
    WP15-T-11  live_risk_parameters() (R-02): live 0.20, class default 0.30
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from common.types import OrderSide
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit.test_risk import (
    _make_limit_order,
    _make_manager,
    _make_position,
)
from trading.risk import (
    LIVE_MAX_DRAWDOWN_PCT,
    BaseRiskManager,
    RiskParameters,
    live_risk_parameters,
)
from trading.risk_manager import DefaultRiskManager

_EIGHT_PCT = {"max_daily_loss_pct": 0.08}


# ---------------------------------------------------------------------------
# WP15-T-09
# ---------------------------------------------------------------------------


def test_wp15_t09_equality_blocks_via_pre_trade_check() -> None:
    """red@HEAD: HEAD's strict ``<`` let a loss of exactly -threshold through."""
    manager = _make_manager(**_EIGHT_PCT)
    result = manager.pre_trade_check(
        order=_make_limit_order(),
        current_equity=Decimal("10000"),
        open_positions=[],
        daily_pnl=Decimal("-800"),  # exactly 8% of 10 000
        peak_equity=Decimal("10000"),
    )
    assert result.approved is False
    assert any("daily loss" in r.lower() for r in result.rejection_reasons)


def test_wp15_t09_one_satoshi_inside_the_boundary_passes() -> None:
    manager = _make_manager(**_EIGHT_PCT)
    basis = Decimal("10000")
    assert manager._check_daily_loss(Decimal("-800"), basis, basis) is not None
    assert manager._check_daily_loss(Decimal("-800") + Decimal("0.00000001"), basis, basis) is None


def test_wp15_t09_basis_wins_over_current_equity() -> None:
    manager = _make_manager(**_EIGHT_PCT)
    # NAV doubled (20 000) but the day-start basis is capped at 10 000
    basis = Decimal("10000")
    assert manager._check_daily_loss(Decimal("-900"), Decimal("20000"), basis) is not None
    assert manager._check_daily_loss(Decimal("-700"), Decimal("20000"), Decimal("10000")) is None
    # legacy: no basis -> current_equity is the basis (threshold 1 600)
    assert manager._check_daily_loss(Decimal("-900"), Decimal("20000")) is None
    assert manager._check_daily_loss(Decimal("-1600"), Decimal("20000")) is not None


@pytest.mark.parametrize("basis", [Decimal("0"), Decimal("-5")])
def test_wp15_t09_non_positive_basis_is_fail_closed(basis: Decimal) -> None:
    manager = _make_manager(**_EIGHT_PCT)
    violation = manager._check_daily_loss(Decimal("0"), Decimal("10000"), basis)
    assert violation is not None
    assert violation.rule == "max_daily_loss"
    assert violation.blocking is True
    assert "daily loss" in violation.message.lower()
    # the legacy path (basis None) is fail-closed on equity <= 0 as well (RK-13)
    assert manager._check_daily_loss(Decimal("0"), basis) is not None


@pytest.mark.parametrize("basis", [Decimal("0"), Decimal("-5")])
def test_wp15_t09_non_positive_basis_blocks_even_when_the_day_is_in_profit(
    basis: Decimal,
) -> None:
    v = _make_manager(**_EIGHT_PCT)._check_daily_loss(Decimal("50"), Decimal("10000"), basis)
    assert v is not None and v.rule == "max_daily_loss" and v.blocking is True


def test_wp15_t09_rejection_text_and_rule() -> None:
    violation = _make_manager(**_EIGHT_PCT)._check_daily_loss(
        Decimal("-801"), Decimal("10000"), Decimal("10000")
    )
    assert violation is not None
    assert violation.rule == "max_daily_loss"
    assert "daily loss" in violation.message.lower()
    assert "-801.00" in violation.message and "800.00" in violation.message


def test_wp15_t09_pre_trade_check_forwards_the_basis_keyword() -> None:
    manager = _make_manager(**_EIGHT_PCT)
    order = _make_limit_order()
    kwargs = {
        "order": order,
        "current_equity": Decimal("20000"),
        "open_positions": [],
        "daily_pnl": Decimal("-900"),
        "peak_equity": Decimal("20000"),
    }
    assert manager.pre_trade_check(**kwargs).approved is True  # legacy basis 20 000
    assert manager.pre_trade_check(**kwargs, daily_loss_basis=Decimal("10000")).approved is False


# ---------------------------------------------------------------------------
# WP15-T-10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["max_daily_loss", "max_drawdown"])
def test_wp15_t10_latched_gates_block_buys_but_never_reducing_sells(gate: str) -> None:
    manager = _make_manager(max_daily_loss_pct=0.08, max_drawdown_pct=0.30)
    held = _make_position(quantity=Decimal("0.1"), current_price=Decimal("50000"))
    if gate == "max_daily_loss":
        gate_kwargs = {
            "current_equity": Decimal("9000"),
            "daily_pnl": Decimal("-900"),
            "peak_equity": Decimal("9500"),
            "daily_loss_basis": Decimal("10000"),
        }
    else:
        gate_kwargs = {
            "current_equity": Decimal("6500"),
            "daily_pnl": Decimal("0"),
            "peak_equity": Decimal("10000"),
        }

    buy = manager.pre_trade_check(
        order=_make_limit_order(symbol="ETH/USDT", side=OrderSide.BUY, price=Decimal("100"),
                                quantity=Decimal("1")),
        open_positions=[held],
        **gate_kwargs,
    )
    assert buy.approved is False
    assert buy.rejection_reasons

    sell = manager.pre_trade_check(
        order=_make_limit_order(side=OrderSide.SELL, quantity=Decimal("0.1")),
        open_positions=[held],
        **gate_kwargs,
    )
    assert sell.approved is True
    assert any("bypassed (exposure-reducing SELL)" in w for w in sell.warnings)

    naked_sell = manager.pre_trade_check(
        order=_make_limit_order(symbol="ETH/USDT", side=OrderSide.SELL, quantity=Decimal("1")),
        open_positions=[held],  # nothing held in ETH
        **gate_kwargs,
    )
    assert naked_sell.approved is False
    assert any("sell_without_position" in r for r in naked_sell.rejection_reasons)


def test_wp15_t10_both_rules_remain_entry_only() -> None:
    assert {"max_daily_loss", "max_drawdown"} <= BaseRiskManager._ENTRY_ONLY_RULES


# ---------------------------------------------------------------------------
# WP15-T-11  (R-02)
# ---------------------------------------------------------------------------


def test_wp15_t11_live_parameters_differ_only_in_drawdown() -> None:
    live = live_risk_parameters()
    assert live == dataclasses.replace(RiskParameters(), max_drawdown_pct=0.20)
    assert live.max_drawdown_pct == LIVE_MAX_DRAWDOWN_PCT == 0.20
    for field in dataclasses.fields(RiskParameters):
        if field.name != "max_drawdown_pct":
            assert getattr(live, field.name) == getattr(RiskParameters(), field.name)


def test_wp15_t11_class_default_and_paper_backtest_stay_030() -> None:
    assert RiskParameters().max_drawdown_pct == 0.30
    assert DefaultRiskManager(run_id="paper").params.max_drawdown_pct == 0.30
    live_manager = DefaultRiskManager(run_id="live", params=live_risk_parameters())
    assert live_manager.params.max_drawdown_pct == 0.20


def test_wp15_t11_smoke_g5_ceiling_is_unchanged() -> None:
    # smoke_guard G-5 reads RiskParameters().max_position_size_pct only
    assert RiskParameters().max_position_size_pct == 0.15
    assert live_risk_parameters().max_position_size_pct == 0.15
