"""
tests/unit/test_wp16_engine_cb.py
-----------------------------------
WP1.6 -- circuit breaker through the REAL ``StrategyEngine._process_bar``
(spec §5.4).  The portfolio is a real ``PortfolioAccounting`` holding 10 BTC
bought at 1000 (NAV == 10 000 == initial cash), so a BTC close of 800 is a 20%
drawdown, 880 is 12%, 900 is 10% (loss of 1 000 > the 8% daily limit).

    WP16-T-16  ``_resolve_order_quantity`` x ``Signal.entry_size_multiplier``
    WP16-T-17  C23: HALT / DAILY_LIMIT drop BUYs only; SELLs still reach the engine (red@HEAD)
    WP16-T-18  D4: HALT never stops the run (paper + live)                 (red@HEAD)
    WP16-T-19  REDUCE halves the FINAL BUY quantity, target untouched      (red@HEAD)
    WP16-T-20  fail-safe: a breaker failure drops that bar's BUYs only      (red@HEAD)
    WP16-T-21  single reason, fixed order, gate not consulted
    WP16-T-22  cadence: one evaluation per processed bar, after the marks
    WP16-T-23  gauge 0/1/2/3, zero after stop(), no legacy suppression event
    WP16-T-24  payload allow-lists
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from common.metrics import metrics
from common.types import OrderSide, RunMode, SignalDirection
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp16_support import (
    DAILY_ONLY,
    HALT_ONLY,
    HELD,
    OTHER,
    T0,
    Clock,
    RecordingPaper,
    bar,
    build_engine,
    buy,
    holding_portfolio,
    sell,
    skipped_reasons,
)
from trading.engines.paper import PaperExecutionEngine
from trading.models import Signal
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters, live_risk_parameters
from trading.risk_manager import DefaultRiskManager
from trading.safety import CircuitBreaker, CircuitBreakerResponse, build_run_circuit_breaker

GAUGE_KEY = 'circuit_breaker_state{run_id="wp16-run"}'


def _engine(
    params: RiskParameters,
    signals: Any,
    *,
    mode: RunMode = RunMode.PAPER,
    cb: CircuitBreaker | None = None,
    **kw: Any,
) -> tuple[Any, Any, PortfolioAccounting, CircuitBreaker]:
    breaker = cb if cb is not None else build_run_circuit_breaker("wp16", params)
    engine, ex, pf = build_engine(cb=breaker, signals=signals, params=params, mode=mode, **kw)
    return engine, ex, pf, breaker


def _both(_i: int) -> list[Signal]:
    return [buy(OTHER), sell(HELD)]


# ---------------------------------------------------------------------------
# T-16  sizing contract
# ---------------------------------------------------------------------------


def _resolve(m: str, *, target: str = "2000", side: OrderSide = OrderSide.BUY,
             held: str | None = None, direction: SignalDirection | None = None) -> Decimal:
    rm = DefaultRiskManager("t16")
    ex = PaperExecutionEngine("t16", rm, initial_cash=Decimal("10000"))
    sig = Signal(
        strategy_id="a",
        symbol=OTHER,
        direction=direction
        or (SignalDirection.BUY if side == OrderSide.BUY else SignalDirection.SELL),
        target_position=Decimal(target),
        entry_size_multiplier=Decimal(m),
    )
    return ex._resolve_order_quantity(
        signal=sig,
        side=side,
        last_price=Decimal("100"),
        equity=Decimal("10000"),
        held_quantity=Decimal(held) if held is not None else None,
    )


def test_wp16_t16_multiplier_acts_on_the_final_quantity_after_the_ceiling() -> None:
    # target 2000 @100 -> base 20, ceiling 15 (15% of 10 000): HEAD's target-halving gave 10.
    assert _resolve("1") == Decimal("15")
    assert _resolve("0.5") == Decimal("7.5")


def test_wp16_t16_the_buy_plus_zero_fallback_is_halved_too() -> None:
    assert _resolve("1", target="0") == Decimal("15")
    assert _resolve("0.5", target="0") == Decimal("7.5")


def test_wp16_t16_sell_is_identical_for_every_multiplier() -> None:
    assert _resolve("1", side=OrderSide.SELL, held="3", target="0") == Decimal("3")
    assert _resolve("0.5", side=OrderSide.SELL, held="3", target="0") == Decimal("3")
    assert _resolve("1", side=OrderSide.SELL, held="3", target="200") == Decimal("2")
    assert _resolve("0.5", side=OrderSide.SELL, held="3", target="200") == Decimal("2")


@pytest.mark.parametrize("bad", ["0", "2", "-0.1", "1.0000001"])
def test_wp16_t16_signal_validation_rejects_out_of_range(bad: str) -> None:
    with pytest.raises(ValueError):
        Signal(strategy_id="a", symbol=OTHER, direction=SignalDirection.BUY,
               target_position=Decimal(1), entry_size_multiplier=Decimal(bad))


@pytest.mark.parametrize("bad", ["0", "2"])
def test_wp16_t16_model_copy_bypass_is_caught_by_the_resolver(bad: str) -> None:
    rm = DefaultRiskManager("t16")
    ex = PaperExecutionEngine("t16", rm, initial_cash=Decimal("10000"))
    sig = buy(OTHER, "2000").model_copy(update={"entry_size_multiplier": Decimal(bad)})
    with pytest.raises(ValueError, match="entry_size_multiplier"):
        ex._resolve_order_quantity(signal=sig, side=OrderSide.BUY, last_price=Decimal("100"),
                                   equity=Decimal("10000"), held_quantity=None)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_wp16_t16_non_finite_multiplier_never_sizes_a_buy(bad: str) -> None:
    """WP16-SEC-04 / R-4: a non-finite multiplier is rejected at both layers (fail-closed).

    NaN reaches the resolver's range check as ``InvalidOperation`` (a Decimal NaN
    comparison), the infinities as ``ValueError``; both are caught per signal.
    """
    with pytest.raises(ValueError):
        Signal(strategy_id="a", symbol=OTHER, direction=SignalDirection.BUY,
               target_position=Decimal(1), entry_size_multiplier=Decimal(bad))
    rm = DefaultRiskManager("t16")
    ex = PaperExecutionEngine("t16", rm, initial_cash=Decimal("10000"))
    sig = buy(OTHER, "2000").model_copy(update={"entry_size_multiplier": Decimal(bad)})
    with pytest.raises((ValueError, InvalidOperation)):
        ex._resolve_order_quantity(signal=sig, side=OrderSide.BUY, last_price=Decimal("100"),
                                   equity=Decimal("10000"), held_quantity=None)


def test_wp16_t16_default_is_one_and_strategies_cannot_depend_on_it() -> None:
    assert buy().entry_size_multiplier == Decimal(1)


# ---------------------------------------------------------------------------
# T-17  C23
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "held_close", "reason"),
    [
        (HALT_ONLY, "800", "circuit_breaker_halt"),
        (DAILY_ONLY, "900", "circuit_breaker_daily_limit"),
    ],
    ids=["halt", "daily_limit"],
)
async def test_wp16_t17_buys_are_dropped_but_the_held_sell_still_reaches_the_engine(
    params: RiskParameters, held_close: str, reason: str
) -> None:
    engine, ex, _pf, cb = _engine(params, _both)
    await bar(engine, held_close)
    assert cb.current_response.value == reason.removeprefix("circuit_breaker_")
    assert [s.direction for s in ex.signals] == [SignalDirection.SELL]  # BUY A dropped, SELL B kept
    assert ex.signals[0].symbol == HELD and ex.signals[0].entry_size_multiplier == Decimal(1)
    assert skipped_reasons(engine) == [reason]  # exactly one skip record, for the BUY


async def test_wp16_t17_reduce_keeps_both_signals() -> None:
    engine, ex, _pf, cb = _engine(HALT_ONLY, _both)
    await bar(engine, "880")  # 12% drawdown
    assert cb.current_response == CircuitBreakerResponse.REDUCE
    assert sorted(s.direction.value for s in ex.signals) == ["buy", "sell"]
    assert skipped_reasons(engine) == []


# ---------------------------------------------------------------------------
# T-18  D4: no auto-stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
async def test_wp16_t18_halt_never_stops_the_run(mode: RunMode) -> None:
    engine, ex, _pf, cb = _engine(HALT_ONLY, lambda i: [buy(OTHER)], mode=mode)
    await engine.start("wp16-run")
    with capture_logs() as logs:
        for _ in range(3):
            await bar(engine, "800")
    assert cb.current_response == CircuitBreakerResponse.HALT
    assert not engine._stop_event.is_set()
    assert engine.auto_stop_reason is None
    assert engine._bar_count == 3  # the next bar is processed
    assert ex.signals == []  # every BUY dropped, run kept polling
    events = [e["event"] for e in logs]
    assert events.count("circuit_breaker.halt_entered") == 1
    assert "circuit_breaker.halt_auto_stop_requested" not in events
    assert metrics.get_all()["gauges"][GAUGE_KEY] == 3.0


# ---------------------------------------------------------------------------
# T-19  REDUCE
# ---------------------------------------------------------------------------


async def test_wp16_t19_reduce_marks_the_buy_and_leaves_target_position_alone() -> None:
    engine, ex, _pf, _cb = _engine(HALT_ONLY, _both)
    await bar(engine, "880")
    by_dir = {s.direction: s for s in ex.signals}
    assert by_dir[SignalDirection.BUY].entry_size_multiplier == Decimal("0.5")
    assert by_dir[SignalDirection.BUY].target_position == Decimal("1000")  # HEAD halved this
    assert by_dir[SignalDirection.SELL].entry_size_multiplier == Decimal(1)


async def test_wp16_t19_end_to_end_reduce_quantity_is_half_the_control() -> None:
    """Same market, same NAV: REDUCE order == 0.5 x the no-breaker control order."""
    sizes: dict[str, Decimal] = {}
    for label, with_cb in (("control", False), ("reduce", True)):
        clock = Clock(T0)
        pf = holding_portfolio(clock)
        rm = DefaultRiskManager("wp16", params=HALT_ONLY)
        paper = RecordingPaper("wp16", rm, initial_cash=Decimal("10000"))
        cb = build_run_circuit_breaker("wp16", HALT_ONLY) if with_cb else None
        engine, _, _ = build_engine(
            cb=cb, signals=lambda i: [buy(OTHER, "2000")], execution=paper,
            portfolio=pf, risk_manager=rm,
        )
        await bar(engine, "880")
        assert len(paper.orders_out) == 1, label
        sizes[label] = paper.orders_out[0].quantity
    assert sizes["control"] > 0
    assert sizes["reduce"] * 2 == sizes["control"]  # ceiling-binding: 13.2 vs 6.6


# ---------------------------------------------------------------------------
# T-20  fail-safe
# ---------------------------------------------------------------------------


class _RaisingEvaluate(CircuitBreaker):
    def evaluate(self, snapshot: Any) -> Any:
        raise RuntimeError("cb boom")


class _RaisingSnapshotPortfolio(PortfolioAccounting):
    fail = False

    def risk_snapshot(self) -> Any:
        if self.fail:
            raise RuntimeError("snapshot boom")
        return super().risk_snapshot()


class _NanSnapshotPortfolio(PortfolioAccounting):
    nan = False

    def risk_snapshot(self) -> Any:
        snap = super().risk_snapshot()
        return replace(snap, drawdown_pct=Decimal("NaN")) if self.nan else snap


def _failing(kind: str) -> tuple[Any, Any, Any, CircuitBreaker]:
    from tests.unit._wp15_support import make_fill

    clock = Clock(T0)
    cls = {"snapshot": _RaisingSnapshotPortfolio, "nan": _NanSnapshotPortfolio,
           "evaluate": PortfolioAccounting}[kind]
    pf = cls("wp16", Decimal("10000"), clock=clock)
    pf.update_position(
        make_fill(quantity="10", price="1000", symbol=HELD), current_price=Decimal("1000")
    )
    cb: CircuitBreaker
    if kind == "evaluate":
        base = build_run_circuit_breaker("wp16", HALT_ONLY)
        cb = _RaisingEvaluate(base.config, "wp16", thresholds=base.thresholds)
    else:
        cb = build_run_circuit_breaker("wp16", HALT_ONLY)
    engine, ex, _ = build_engine(cb=cb, signals=_both, portfolio=pf, params=HALT_ONLY)
    return engine, ex, pf, cb


@pytest.mark.parametrize("kind", ["snapshot", "nan", "evaluate"])
async def test_wp16_t20_a_breaker_failure_drops_buys_but_exits_run(kind: str) -> None:
    engine, ex, pf, cb = _failing(kind)
    await engine.start("wp16-run")
    await bar(engine, "800")  # a healthy HALT bar first (evaluate-kind raises from the start)
    evals_before = cb.state.evaluations
    level_before = cb.current_response
    ex.signals.clear()
    engine._skip_logger.clear()
    if kind == "snapshot":
        pf.fail = True
    elif kind == "nan":
        pf.nan = True
    with capture_logs() as logs:
        await bar(engine, "790")
    assert [s.direction for s in ex.signals] == [SignalDirection.SELL]  # exits still run
    assert skipped_reasons(engine) == ["circuit_breaker_eval_failed"]
    errors = [e for e in logs if e["event"] == "engine.circuit_breaker_eval_failed"]
    assert len(errors) == 1 and errors[0]["log_level"] == "error"
    assert any(e["event"] == "engine.circuit_breaker_entries_blocked" for e in logs)
    assert cb.state.evaluations == evals_before  # CB state untouched
    assert cb.current_response == level_before
    assert not engine._stop_event.is_set()


async def test_wp16_t20_gauge_is_unchanged_on_a_failing_bar() -> None:
    engine, _ex, pf, _cb = _failing("snapshot")
    await engine.start("wp16-run")
    await bar(engine, "800")
    assert metrics.get_all()["gauges"][GAUGE_KEY] == 3.0
    pf.fail = True
    await bar(engine, "1000")  # a calm price, but the breaker is blind: last level stays
    assert metrics.get_all()["gauges"][GAUGE_KEY] == 3.0


# ---------------------------------------------------------------------------
# T-21  single reason, fixed order, gate not consulted
# ---------------------------------------------------------------------------


async def test_wp16_t21_kill_switch_wins_over_halt_and_keeps_its_event_name() -> None:
    engine, ex, _pf, _cb = _engine(HALT_ONLY, _both)
    engine._risk_manager.trigger_kill_switch("test")
    with capture_logs() as logs:
        await bar(engine, "800")
    assert skipped_reasons(engine) == ["kill_switch"]
    events = [e["event"] for e in logs]
    assert "engine.kill_switch_entries_blocked" in events
    assert "engine.circuit_breaker_entries_blocked" not in events
    assert [s.direction for s in ex.signals] == [SignalDirection.SELL]


async def test_wp16_t21_protective_mode_wins_over_daily_limit() -> None:
    engine, _ex, _pf, _cb = _engine(DAILY_ONLY, _both, protective_mode=True)
    await bar(engine, "900")
    assert skipped_reasons(engine) == ["protective_mode"]


async def test_wp16_t21_halt_alone_uses_the_breaker_event_once_per_bar_and_never_the_gate() -> None:
    clock = Clock(T0)
    pf = holding_portfolio(clock)
    rm = DefaultRiskManager("wp16", params=HALT_ONLY)
    spy = MagicMock(wraps=rm.pre_trade_check)
    rm.pre_trade_check = spy  # type: ignore[method-assign]
    paper = RecordingPaper("wp16", rm, initial_cash=Decimal("10000"))
    cb = build_run_circuit_breaker("wp16", HALT_ONLY)
    engine, _, _ = build_engine(
        cb=cb, signals=lambda i: [buy(OTHER), buy(OTHER, "500")], execution=paper,
        portfolio=pf, risk_manager=rm,
    )
    with capture_logs() as logs:
        await bar(engine, "800")
    blocked = [e for e in logs if e["event"] == "engine.circuit_breaker_entries_blocked"]
    assert len(blocked) == 1 and blocked[0]["dropped_buy_count"] == 2
    assert blocked[0]["skip_reason"] == "circuit_breaker_halt"
    assert skipped_reasons(engine) == ["circuit_breaker_halt"] * 2  # one record per BUY
    spy.assert_not_called()  # pre_trade_check is never asked about a CB-dropped BUY
    assert paper.signals == []


# ---------------------------------------------------------------------------
# T-22  cadence
# ---------------------------------------------------------------------------


async def test_wp16_t22_one_evaluation_per_processed_bar_after_the_marks() -> None:
    events: list[str] = []

    class _OrderedPortfolio(PortfolioAccounting):
        def update_market_prices(self, prices: dict[str, Decimal]) -> None:
            events.append("marks")
            super().update_market_prices(prices)

    class _OrderedCb(CircuitBreaker):
        def evaluate(self, snapshot: Any) -> Any:
            events.append("evaluate")
            return super().evaluate(snapshot)

    from tests.unit._wp15_support import make_fill

    pf = _OrderedPortfolio("wp16", Decimal("10000"), clock=Clock(T0))
    pf.update_position(
        make_fill(quantity="10", price="1000", symbol=HELD), current_price=Decimal("1000")
    )
    base = build_run_circuit_breaker("wp16", HALT_ONLY)
    cb = _OrderedCb(base.config, "wp16", thresholds=base.thresholds)
    engine, _, _ = build_engine(cb=cb, signals=lambda i: [], portfolio=pf, params=HALT_ONLY)
    for _ in range(3):
        await bar(engine, "1000")
    assert events == ["marks", "evaluate"] * 3
    assert cb.state.evaluations == 3


# ---------------------------------------------------------------------------
# T-23  gauge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "close", "value"),
    [
        (HALT_ONLY, "1000", 0.0),
        (HALT_ONLY, "880", 1.0),
        (DAILY_ONLY, "900", 2.0),
        (HALT_ONLY, "800", 3.0),
    ],
    ids=["ok", "reduce", "daily_limit", "halt"],
)
async def test_wp16_t23_gauge_follows_the_level_and_is_zeroed_by_stop(
    params: RiskParameters, close: str, value: float
) -> None:
    engine, _ex, _pf, _cb = _engine(params, _both)
    await engine.start("wp16-run")
    with capture_logs() as logs:
        await bar(engine, close)
    assert metrics.get_all()["gauges"][GAUGE_KEY] == value
    assert "engine.daily_limit_signals_suppressed" not in [e["event"] for e in logs]
    await engine.stop()
    assert metrics.get_all()["gauges"][GAUGE_KEY] == 0.0


async def test_wp16_t23_no_breaker_means_gauge_zero() -> None:
    engine, _ex, _pf = build_engine(cb=None, signals=_both, params=HALT_ONLY)
    await engine.start("wp16-run")
    await bar(engine, "800")
    assert metrics.get_all()["gauges"][GAUGE_KEY] == 0.0


# ---------------------------------------------------------------------------
# T-24  payload allow-lists
# ---------------------------------------------------------------------------

_FORBIDDEN = ("balance", "api_key", "secret", "token", "passphrase", "account", "free")


async def test_wp16_t24_log_events_and_state_inputs_contain_only_allow_listed_keys() -> None:
    engine, _ex, _pf, cb = _engine(HALT_ONLY, _both)
    await engine.start("wp16-run")
    with capture_logs() as logs:
        await bar(engine, "800")
        await bar(engine, "1000")
    prefixes = ("circuit_breaker.", "engine.circuit_breaker")
    cb_logs = [e for e in logs if e["event"].startswith(prefixes)]
    assert cb_logs
    for entry in cb_logs:
        for key in entry:
            assert not any(f in key.lower() for f in _FORBIDDEN), (entry["event"], key)
    st = cb.state
    assert st.inputs is not None
    assert set(st.inputs) == {
        "as_of", "day", "nav", "peak_equity", "drawdown_pct", "gate_daily_pnl",
        "daily_loss_basis", "daily_threshold", "day_start_source", "state_unknown",
        "open_position_count",
    }
    assert set(st.thresholds) == {
        "max_daily_loss_pct", "halt_entry", "halt_exit", "reduce_entry", "reduce_exit",
        "reduce_multiplier", "deescalation_evaluations",
    }
    assert isinstance(live_risk_parameters(), RiskParameters)
