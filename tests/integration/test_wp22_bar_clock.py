"""
tests/integration/test_wp22_bar_clock.py
-----------------------------------------
WP2.2 P2b: the bar-time clock, end to end (hermetic: no Postgres, no network).

    WP22-T-02  DoD: every persisted time column of a backtest is in {t0} U {close_i}
    WP22-T-03  quant expected values (F1 123 points, AT-2 holding 24 h, AT-3 stop-loss exit)
    WP22-T-04  day attribution: stamp day == the WP1.5 risk day for every mark
    WP22-T-05  multi-symbol, misaligned bars: the max rule
    WP22-T-06  no wall-clock read in the backtest path (raising tripwires)
    WP22-T-08  determinism under two poisoned wall clocks (uuid-normalised rows)
    WP22-T-09  paper / live unchanged, guards (R1/R2/R8), tripwires (R6), isolation (R8), R10
    WP22-T-15  the WP2.5 contract C1-C5
    WP22-T-16  the CF-26 re-baseline record is byte-identical to the one recorded at 117236f
    WP22-T-18  at most one Fill per Order (guards SY-22-06)
    WP22-T-19  runner edges (no-clock engine, :450, portfolio == BarClock after every bar)

``snapshot_at`` of ``position_snapshots`` is NOT asserted here: it is stamped in
``apps/api/services/run_persistence.py`` (SY-22-15), which belongs to commit P3.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect as pyinspect
import tempfile
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from sqlalchemy import inspect as sa_inspect

from api.db.models import (
    EquitySnapshotORM,
    FillORM,
    OrderORM,
    PositionSnapshotORM,
)
from api.services.run_persistence import persist_backtest_results, persist_paper_results
from common.models import OHLCVBar
from common.types import OrderSide, OrderStatus, OrderType, RunMode, SignalDirection, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import TIMEFRAME_DELTA, scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.integration.test_wp15b_resume_seed_wiring import _session_factory
from tests.integration.test_wp16_cb_api_orchestrator import (
    SYM as _ORCH_SYM,
)
from tests.integration.test_wp16_cb_api_orchestrator import (
    _CapturingEngine,
    _empty_resume,
    _quiet_side_tasks,  # noqa: F401  (autouse fixture of the orchestrator harness)
    _settings,
)
from tests.unit.test_wp21c_rebaseline import _series, rb
from trading import backtest as backtest_mod
from trading import clock as clock_mod
from trading.backtest import BacktestRunner
from trading.clock import (
    WALL_CLOCK,
    BarClock,
    WallClock,
    ambient_clock,
    use_clock,
    utc_now,
)
from trading.costs import get_cost_profile
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine
from trading.models import Order, Signal
from trading.portfolio import PortfolioAccounting, _utc_day
from trading.risk_manager import DefaultRiskManager
from trading.strategy_engine import EngineState, StrategyEngine

LEGACY = get_cost_profile(
    "legacy_backtest_v0"
)  # the profile of the 2.1a goldens / quant fixture F1
SYM = "BTC/USDT"
H = timedelta(hours=1)
F1_FIRST_OPEN = datetime(2026, 3, 7, 22, 0, tzinfo=UTC)  # quant fixture F1, bar 0
POST_WARMUP_OPEN = F1_FIRST_OPEN + 50 * H  # open of the first post-warmup bar
FIRST_OPEN = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

# The CF-26 record digest recorded on pristine 117236f (SY-22-19: never edited).
REBASELINE_DIGEST_AT_117236F = "1c30203bdee86f90598b8f7cbc99e4782a7d5d56fcd09ef31ef55affa7eadcec"

_BRACKET = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.02,
    "bracket_take_profit_pct": 0.04,
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@dataclass
class Run:
    runner: BacktestRunner
    result: Any
    bars: dict[str, list[OHLCVBar]]
    tf: timedelta

    @property
    def symbols(self) -> list[str]:
        return list(self.bars)

    @property
    def n_bars(self) -> int:
        return min(len(b) for b in self.bars.values())

    def close(self, i: int) -> datetime:
        """close_i = max_s(ts_i) + tf (the SY-22-01 definition)."""
        return max(b[i].timestamp for b in self.bars.values()) + self.tf

    @property
    def t0(self) -> datetime:
        return max(b[0].timestamp for b in self.bars.values())

    @property
    def closes(self) -> list[datetime]:
        return [self.close(i) for i in range(self.n_bars)]

    @property
    def curve(self) -> list[Any]:
        return list(self.result.equity_curve)

    @property
    def fills(self) -> list[Any]:
        assert self.runner.last_execution_engine is not None
        return self.runner.last_execution_engine.get_all_fills()


def _strategy(script: dict[int, str], *, targets: dict[int, str] | None = None) -> Any:
    return ScriptedSequenceStrategy(
        "wp22",
        {"script": script, "target_notional": "3000", "targets": targets or {}},
    )


async def _run(
    closes: Sequence[int | str],
    script: dict[int, str],
    *,
    first_open: datetime = FIRST_OPEN,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    bracket: dict[str, Any] | None = None,
    trailing: float | None = None,
    breaker: bool = True,
    symbols: tuple[str, ...] = (SYM,),
    bars: dict[str, list[OHLCVBar]] | None = None,
) -> Run:
    runner = BacktestRunner(
        strategies=[_strategy(script)],
        symbols=list(symbols),
        timeframe=timeframe,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=bracket,
        trailing_stop_pct=trailing,
        allow_pyramiding=False,
        enable_circuit_breaker=breaker,
        cost_profile=LEGACY,
    )
    if bars is None:
        bars = {
            s: scripted_path(closes, first_open=first_open, symbol=s, timeframe=timeframe)
            for s in symbols
        }
    result = await runner.run(bars)
    return Run(runner, result, bars, TIMEFRAME_DELTA[timeframe])


async def _dod_run() -> Run:
    """T-02 scenario: fixed bracket + trailing stop + breaker on, two exits, one open lot."""
    return await _run(
        [100, 101, 102, 104, 105, 100, 99, 98, 97, 96, 100, 101],
        {0: "buy", 5: "buy", 10: "buy"},
        bracket=_BRACKET,
        trailing=0.03,
        breaker=True,
    )


class _Recorder:
    """A recording fake of the AsyncSession surface ``persist_backtest_results`` uses."""

    def __init__(self) -> None:
        self.rows: list[Any] = []

    def add_all(self, rows: Any) -> None:
        self.rows.extend(rows)

    def add(self, row: Any) -> None:
        self.rows.append(row)

    async def flush(self) -> None:
        return None


async def _persist(run: Run) -> tuple[_Recorder, Any]:
    rec = _Recorder()
    run_orm = SimpleNamespace(config={}, n_closed_trades=None)
    with patch("sqlalchemy.orm.attributes.flag_modified"):
        await persist_backtest_results(
            rec,
            uuid.uuid4(),
            run_orm,
            run.result,
            structlog.get_logger("wp22"),
            execution_engine=run.runner.last_execution_engine,
            portfolio=run.runner.last_portfolio,
        )
    return rec, run_orm


def _row_values(row: Any) -> dict[str, Any]:
    return {a.key: getattr(row, a.key) for a in sa_inspect(row).mapper.column_attrs}


def _by_type(rec: _Recorder, cls: type) -> list[dict[str, Any]]:
    return [_row_values(r) for r in rec.rows if isinstance(r, cls)]


_TIME_COLUMNS_EXCL_SNAPSHOT_AT = ("timestamp", "created_at", "updated_at", "executed_at")


def _time_values(rec: _Recorder) -> list[tuple[str, str, datetime]]:
    """Every datetime in the persisted rows except ``snapshot_at`` (P3, SY-22-15)."""
    out: list[tuple[str, str, datetime]] = []
    for row in rec.rows:
        values = _row_values(row)
        for col, value in values.items():
            if isinstance(value, datetime) and col != "snapshot_at":
                out.append((type(row).__name__, col, value))
    return out


# ---------------------------------------------------------------------------
# T-02  DoD: persisted stamps equal bar times
# ---------------------------------------------------------------------------


class TestT02PersistedStampsEqualBarTimes:
    async def test_every_time_column_is_t0_or_a_bar_close(self) -> None:
        run = await _dod_run()
        rec, _ = await _persist(run)
        allowed = {run.t0, *run.closes}
        stamps = _time_values(rec)
        assert stamps, "nothing persisted: the test would be vacuous"
        assert {t for t, _c, _v in stamps} >= {
            "EquitySnapshotORM",
            "FillORM",
            "OrderORM",
            "TradeORM",
            "PositionSnapshotORM",
        }
        outside = [(t, c, v) for t, c, v in stamps if v not in allowed]
        assert outside == []
        # nothing is even near the (2026-11) wall clock of this suite
        assert all(v < datetime(2026, 4, 1, tzinfo=UTC) for _t, _c, v in stamps)

    async def test_equity_stamps_are_non_decreasing_in_bar_index_order(self) -> None:
        run = await _dod_run()
        rec, _ = await _persist(run)
        rows = sorted(_by_type(rec, EquitySnapshotORM), key=lambda r: r["bar_index"])
        assert [r["bar_index"] for r in rows] == list(range(len(rows)))
        stamps = [r["timestamp"] for r in rows]
        assert stamps == sorted(stamps)
        assert stamps[0] == run.t0

    async def test_each_processed_bar_has_exactly_one_mark_and_it_is_first(self) -> None:
        run = await _dod_run()
        curve = run.curve
        fills_per_stamp = Counter(f.executed_at for f in run.fills)
        # 1 seed + one mark per bar (warmup included) + one point per fill
        assert len(curve) == 1 + run.n_bars + len(run.fills)
        by_stamp: dict[datetime, list[int]] = {}
        for idx, point in enumerate(curve[1:], start=1):
            by_stamp.setdefault(point.timestamp, []).append(idx)
        assert sorted(by_stamp) == run.closes  # every bar has a mark; nothing else
        for stamp, idxs in by_stamp.items():
            # the mark comes first, fill points follow it
            assert len(idxs) == 1 + fills_per_stamp[stamp]
            assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))

    async def test_trade_stamps_equal_the_opening_and_closing_fill_stamps(self) -> None:
        run = await _dod_run()
        buys = [f for f in run.fills if f.side == OrderSide.BUY]
        sells = [f for f in run.fills if f.side == OrderSide.SELL]
        trades = run.result.trades
        assert len(trades) == 2 and len(buys) == 3 and len(sells) == 2
        for k, trade in enumerate(trades):
            assert trade.entry_at == buys[k].executed_at
            assert trade.exit_at == sells[k].executed_at
            assert trade.exit_at >= trade.entry_at  # ck_trades_exit_after_entry
        assert {t.exit_reason for t in trades} <= {"stop_loss", "take_profit", "trailing_stop"}

    async def test_orders_fills_and_open_lot_are_stamped_with_bar_closes(self) -> None:
        run = await _dod_run()
        rec, _ = await _persist(run)
        closes = set(run.closes)
        orders = _by_type(rec, OrderORM)
        assert orders
        for o in orders:
            assert o["created_at"] in closes and o["updated_at"] in closes
            assert o["updated_at"] >= o["created_at"]
        assert {f["executed_at"] for f in _by_type(rec, FillORM)} <= closes
        positions = _by_type(rec, PositionSnapshotORM)
        assert positions and all(p["opened_at"] in closes for p in positions)
        mtm = run.result.open_positions_mtm
        assert mtm and all(m.opened_at in closes for m in mtm)
        # the open lot was entered at the last BUY (call 10 == bar 60)
        assert mtm[0].opened_at == run.close(60)

    async def test_a_still_resting_order_is_canceled_at_stop_with_the_final_close(self) -> None:
        t0 = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
        clock = BarClock(t0)
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"), clock=clock)
        engine.set_last_price(SYM, Decimal("100"))
        clock.advance(t0 + H)
        order = Order(
            client_order_id="c2",
            run_id="r",
            symbol=SYM,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("90"),
        )
        await engine.submit_order(order)
        clock.advance(t0 + 9 * H)  # the final bar close
        await engine.on_stop()  # engine.stop() runs AFTER the loop, outside use_clock
        canceled = await engine.get_order(order.order_id)
        assert canceled.status == OrderStatus.CANCELED
        assert canceled.updated_at == t0 + 9 * H

    async def test_a_resting_limit_order_is_stamped_with_the_creation_and_fill_bars(self) -> None:
        """The resting-order half of the DoD scenario, on the engine itself."""
        t0 = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
        clock = BarClock(t0)
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"), clock=clock)
        engine.set_last_price(SYM, Decimal("100"))
        clock.advance(t0 + H)
        with use_clock(clock):
            order = Order(
                client_order_id="c1",
                run_id="r",
                symbol=SYM,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1"),
                price=Decimal("95"),
            )
            assert order.created_at == t0 + H  # ambient fallback only
        rested = await engine.submit_order(order)
        assert rested.status == OrderStatus.OPEN and rested.updated_at == t0 + H
        clock.advance(t0 + 4 * H)
        filled = await engine.check_resting_orders(SYM, Decimal("94"))
        assert [o.order_id for o in filled] == [order.order_id]
        assert filled[0].created_at == t0 + H  # creation bar
        assert filled[0].updated_at == t0 + 4 * H  # fill bar
        (fill,) = engine.get_all_fills()
        assert fill.executed_at == t0 + 4 * H
        assert engine.positions[SYM].opened_at == t0 + 4 * H


# ---------------------------------------------------------------------------
# T-03  quant expected values
# ---------------------------------------------------------------------------

_FLAT_72 = [100] * 72


class TestT03QuantExpectedValues:
    async def test_f1_no_trades_has_123_points_stamped_one_hour_apart(self) -> None:
        run = await _run(_FLAT_72, {}, first_open=POST_WARMUP_OPEN)
        assert run.n_bars == 122 and run.bars[SYM][0].timestamp == F1_FIRST_OPEN
        curve = run.curve
        assert len(curve) == 123  # AT-1: 1 seed + 50 warmup + 72
        assert [p.timestamp for p in curve] == [F1_FIRST_OPEN + k * H for k in range(123)]
        assert curve[-1].timestamp == datetime(2026, 3, 13, 0, 0, tzinfo=UTC)
        assert all(a.timestamp < b.timestamp for a, b in pairwise(curve))
        # trading window (last stamp - close of the last warmup bar) = 72 h = 3.000 days
        assert (curve[-1].timestamp - curve[50].timestamp) == timedelta(hours=72)
        assert curve[50].timestamp == datetime(2026, 3, 10, 0, 0, tzinfo=UTC)

    async def test_at2_buy_then_sell_24_bars_later(self) -> None:
        run = await _run(_FLAT_72, {0: "buy", 24: "sell"}, first_open=POST_WARMUP_OPEN)
        buy_at = datetime(2026, 3, 10, 1, 0, tzinfo=UTC)
        sell_at = datetime(2026, 3, 11, 1, 0, tzinfo=UTC)
        assert [f.executed_at for f in run.fills] == [buy_at, sell_at]
        (trade,) = run.result.trades
        assert trade.entry_at == buy_at and trade.exit_at == sell_at
        assert trade.exit_at - trade.entry_at == timedelta(hours=24)  # 24 bars x 1h
        orders = run.runner.last_execution_engine.get_all_orders()
        buy_order = next(o for o in orders if o.side == OrderSide.BUY)
        assert buy_order.created_at == buy_at
        curve = run.curve
        assert len(curve) == 125
        at_buy = [i for i, p in enumerate(curve) if p.timestamp == buy_at]
        assert len(at_buy) == 2  # the mark, then the fill point
        mark, post_fill = curve[at_buy[0]], curve[at_buy[1]]
        assert post_fill.equity < mark.equity  # the entry fee / slippage is booked
        # flat price: the next bar's mark equals the post-fill equity
        assert curve[at_buy[1] + 1].equity == post_fill.equity

    async def test_at3_stop_loss_exit_is_stamped_with_the_trigger_bar_close(self) -> None:
        path: list[int | str] = [100, 101, 99, 98, 97, *[97] * 67]
        run = await _run(
            path,
            {0: "buy"},
            first_open=POST_WARMUP_OPEN,
            bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.02},
        )
        (trade,) = run.result.trades
        assert trade.exit_reason == "stop_loss"
        stop = trade.entry_price * Decimal("0.98")  # all-in basis, frozen at entry
        trigger = next(j for j in range(1, len(path)) if Decimal(path[j]) <= stop)
        assert trigger == 3  # the first close at or below the stop (98 <= ~98.2)
        expected = run.close(50 + trigger)
        assert trade.exit_at == expected
        assert run.fills[-1].side == OrderSide.SELL and run.fills[-1].executed_at == expected
        assert trade.exit_at - trade.entry_at == timedelta(hours=trigger)


# ---------------------------------------------------------------------------
# T-04  day attribution
# ---------------------------------------------------------------------------


class _DaySpy:
    """Class-level spy: after every mark, stamp day vs ``risk_snapshot().day``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.checked: list[tuple[datetime, date]] = []
        original = PortfolioAccounting.update_market_prices
        spy = self

        def wrapped(self_: PortfolioAccounting, prices: Any) -> None:
            original(self_, prices)
            stamp = self_._equity_curve[-1][0]
            spy.checked.append((stamp, self_.risk_snapshot().day))

        monkeypatch.setattr(PortfolioAccounting, "update_market_prices", wrapped)


class TestT04DayAttribution:
    async def test_daily_bars_bar0_is_stamped_the_next_midnight_and_days_agree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _DaySpy(monkeypatch)
        first = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        run = await _run(
            [100] * 12,
            {},
            first_open=first + 50 * timedelta(days=1),
            timeframe=TimeFrame.ONE_DAY,
        )
        assert run.bars[SYM][0].timestamp == first
        assert run.curve[0].timestamp == first  # t0 = D 00:00
        assert run.curve[1].timestamp == datetime(2026, 1, 2, 0, 0, tzinfo=UTC)  # AT-4
        assert len(spy.checked) >= run.n_bars
        assert all(_utc_day(stamp) == day for stamp, day in spy.checked)

    async def test_the_23h_bar_belongs_to_the_next_day_like_the_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _DaySpy(monkeypatch)
        run = await _run([100] * 40, {}, first_open=POST_WARMUP_OPEN)
        assert len(spy.checked) >= run.n_bars
        assert all(_utc_day(stamp) == day for stamp, day in spy.checked)
        midnight = [(s, d) for s, d in spy.checked if s.hour == 0 and s.minute == 0]
        assert midnight, "the series must contain a 23:00 bar (stamped 00:00 next day)"
        for stamp, day in midnight:
            assert day == stamp.date()  # the 00:00 stamp opens the NEW day (WP15-T-24)


# ---------------------------------------------------------------------------
# T-05  multi-symbol, misaligned
# ---------------------------------------------------------------------------


def _offset_bars(symbol: str, first_open: datetime, n: int) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol,
            timeframe=TimeFrame.ONE_HOUR,
            timestamp=first_open + k * H,
            open=Decimal(100),
            high=Decimal(100),
            low=Decimal(100),
            close=Decimal(100),
            volume=Decimal(10),
        )
        for k in range(n)
    ]


class TestT05MisalignedMultiSymbol:
    async def test_stamps_follow_the_max_rule_and_increase_strictly(self) -> None:
        other = "ETH/USDT"
        lead = _offset_bars(SYM, F1_FIRST_OPEN + timedelta(minutes=30), 70)
        lag = _offset_bars(other, F1_FIRST_OPEN, 70)  # opens 30 min EARLIER than SYM
        run = await _run([], {}, symbols=(SYM, other), bars={SYM: lead, other: lag})
        assert run.t0 == lead[0].timestamp  # max of the first opens
        expected = [lead[i].timestamp + H for i in range(70)]
        assert [p.timestamp for p in run.curve[1:]] == expected
        assert run.curve[0].timestamp == run.t0
        stamps = [p.timestamp for p in run.curve]
        assert all(a < b for a, b in pairwise(stamps))

    async def test_the_lagging_symbol_orders_carry_the_leading_close(self) -> None:
        other = "ETH/USDT"
        lead = _offset_bars(SYM, F1_FIRST_OPEN + timedelta(minutes=30), 60)
        lag = _offset_bars(other, F1_FIRST_OPEN, 60)
        runner = BacktestRunner(
            strategies=[_strategy({0: "buy"})],
            symbols=[SYM, other],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=LEGACY,
        )
        result = await runner.run({SYM: lead, other: lag})
        run = Run(runner, result, {SYM: lead, other: lag}, H)
        assert run.fills, "the scripted BUY must fill"
        assert all(f.executed_at in set(run.closes) for f in run.fills)


# ---------------------------------------------------------------------------
# T-06  no wall-clock read on the backtest path
# ---------------------------------------------------------------------------


def _install_wall_tripwires(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: list[str] = []

    def boom(name: str) -> Any:
        def _raise(*_a: Any, **_k: Any) -> Any:
            reads.append(name)
            raise AssertionError(f"wall clock read in the backtest path: {name}")

        return _raise

    monkeypatch.setattr(clock_mod, "_wall_now", boom("trading.clock._wall_now"))
    monkeypatch.setattr("trading.portfolio._utc_now", boom("trading.portfolio._utc_now"))
    return reads


def _uninstall_global_data_clients(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Make the precondition of T-06 true, whatever an earlier test installed.

    The API lifespan installs process-global data clients (sentiment, CoinGecko,
    whale alert, FRED) that the MTF context reads with the WALL clock (C10, WP2.4).
    A suite-wide run can leave one installed, so the test removes them (restored by
    ``monkeypatch``) and then asserts they are gone.
    """
    from data import macro_data, market_signals, sentiment, whale_tracker

    modules = (sentiment, market_signals, whale_tracker, macro_data)
    for module in modules:
        monkeypatch.setattr(module, "_global_client", None)
    return [m.get_global_client() for m in modules]


class TestT06NoWallClockInTheBacktestPath:
    async def test_scenario_and_persist_complete_with_raising_wall_seams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _uninstall_global_data_clients(monkeypatch) == [None] * 4  # C10 / WP2.4 limit
        reads = _install_wall_tripwires(monkeypatch)
        run = await _dod_run()
        rec, _ = await _persist(run)
        assert reads == []
        assert len(run.result.trades) == 2 and rec.rows

    async def test_control_the_tripwire_does_fire_on_a_wall_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_wall_tripwires(monkeypatch)
        with pytest.raises(AssertionError, match="wall clock read"):
            WALL_CLOCK.now()
        with pytest.raises(AssertionError, match="wall clock read"):
            PortfolioAccounting("r")  # a clock-less portfolio seeds through ``_utc_now``


# ---------------------------------------------------------------------------
# T-08  determinism
# ---------------------------------------------------------------------------


def _normalised_rows(rec: _Recorder) -> list[tuple[str, tuple[Any, ...]]]:
    """Persisted rows with uuid / random-id columns mapped to first-appearance ordinals."""
    ordinals: dict[Any, int] = {}

    def norm(value: Any, column: str) -> Any:
        if isinstance(value, uuid.UUID) or column == "client_order_id":
            return ("id", ordinals.setdefault(value, len(ordinals)))
        return value

    out = []
    for row in rec.rows:
        values = _row_values(row)
        values.pop("snapshot_at", None)  # P3 (SY-22-15): still wall time until then
        out.append((type(row).__name__, tuple((k, norm(v, k)) for k, v in sorted(values.items()))))
    return out


class TestT08Determinism:
    async def test_two_runs_under_different_poisoned_walls_persist_identical_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = []
        for poison in (datetime(2031, 6, 1, tzinfo=UTC), datetime(1999, 1, 1, tzinfo=UTC)):
            monkeypatch.setattr(clock_mod, "_wall_now", lambda p=poison: p)
            monkeypatch.setattr("trading.portfolio._utc_now", lambda p=poison: p)
            run = await _dod_run()
            rec, _ = await _persist(run)
            results.append((_normalised_rows(rec), run, poison))
        (rows_a, _ra, poison_a), (rows_b, _rb, poison_b) = results
        assert rows_a == rows_b and len(rows_a) > 50
        for rows, poison in ((rows_a, poison_a), (rows_b, poison_b)):
            years = {v.year for _t, vals in rows for _k, v in vals if isinstance(v, datetime)}
            assert years == {2026}, years  # bar time only; the poisoned wall year never leaks
            assert poison.year not in years

    async def test_control_the_comparison_detects_a_stamp_difference(self) -> None:
        run_a = await _dod_run()
        run_b = await _run(
            [100, 101, 102, 104, 105, 100, 99, 98, 97, 96, 100, 101],
            {0: "buy", 5: "buy", 10: "buy"},
            first_open=FIRST_OPEN + H,  # one hour later: every stamp shifts
            bracket=_BRACKET,
            trailing=0.03,
        )
        rec_a, _ = await _persist(run_a)
        rec_b, _ = await _persist(run_b)
        assert _normalised_rows(rec_a) != _normalised_rows(rec_b)


# ---------------------------------------------------------------------------
# T-15  the WP2.5 contract (C1-C5)
# ---------------------------------------------------------------------------


class TestT15Wp25Contract:
    async def test_c1_c2_c3_stamps_and_ordinals_on_a_traded_run(self) -> None:
        run = await _run(_FLAT_72, {0: "buy", 24: "sell"}, first_open=POST_WARMUP_OPEN)
        rec, _ = await _persist(run)
        w = 50
        curve = run.curve
        assert curve[0].timestamp == run.t0  # C1: point 0 is t0
        assert curve[w].timestamp == run.close(w - 1)  # C3: curve[W] == close_{W-1}
        assert curve[w].timestamp == datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
        post = [p for p in curve if p.timestamp > curve[w].timestamp]
        assert len({p.timestamp for p in post}) == run.n_bars - w  # N - W distinct stamps
        # C1: the last point per stamp is the end-of-bar equity (flat price: = next mark)
        last_per_stamp: dict[datetime, Any] = {}
        for p in curve:
            last_per_stamp[p.timestamp] = p
        for stamp in (run.close(50), run.close(74)):  # the buy bar and the sell bar
            assert last_per_stamp[stamp].equity == last_per_stamp[stamp + H].equity
        # C2: bar_index is the point ordinal; (timestamp, bar_index) is the total order
        rows = _by_type(rec, EquitySnapshotORM)
        assert [r["bar_index"] for r in rows] == list(range(len(curve)))
        keys = [(r["timestamp"], r["bar_index"]) for r in rows]
        assert keys == sorted(keys) and len(set(keys)) == len(keys)

    async def test_c3_oos_boundary_stamp_is_the_close_before_the_oos_bar(self) -> None:
        runner = BacktestRunner(
            strategies=[_strategy({0: "buy", 6: "sell"})],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=LEGACY,
        )
        bars = {SYM: scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)}
        t0 = bars[SYM][0].timestamp
        clock = BarClock(t0)
        engine, portfolio, *_ = runner._build_engine("oos", clock=clock)
        await engine.start("oos")
        oos = 70
        try:
            summary = await engine.run_backtest(bars, oos_start_index=oos)
        finally:
            await engine.stop()
        offset = summary["oos_equity_curve_offset"]
        curve = portfolio.get_equity_curve()
        assert curve[offset - 1][0] == bars[SYM][oos - 1].timestamp + H  # close_{oos-1}

    async def test_c5_risk_day_equals_the_stamp_day_for_every_mark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _DaySpy(monkeypatch)
        run = await _run(_FLAT_72, {0: "buy", 30: "sell"}, first_open=POST_WARMUP_OPEN)
        assert len(spy.checked) == run.n_bars
        assert all(_utc_day(s) == d for s, d in spy.checked)


# ---------------------------------------------------------------------------
# T-16  CF-26 re-baseline unaffected
# ---------------------------------------------------------------------------


class TestT16RebaselineUnaffected:
    async def test_the_record_json_matches_the_digest_recorded_at_117236f(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv.gz"
            digest = rb.write_fixture(path, _series())
            series, sha = rb.load_fixture(path, digest)
            baselines = rb._select(
                rb.BASELINES, ["momentum_breakout:BTC/EUR", "rsi_mean_reversion:ETH/EUR"]
            )
            record = await rb.run_matrix(series, fixture_sha256=sha, baselines=baselines)
        got = hashlib.sha256(rb.render_json(record).encode()).hexdigest()
        assert got == REBASELINE_DIGEST_AT_117236F


# ---------------------------------------------------------------------------
# T-18  at most one Fill per Order
# ---------------------------------------------------------------------------


class TestT18OneFillPerOrder:
    async def test_runner_scenario_has_at_most_one_fill_per_order(self) -> None:
        run = await _dod_run()
        counts = Counter(f.order_id for f in run.fills)
        assert counts and max(counts.values()) == 1
        order_ids = {o.order_id for o in run.runner.last_execution_engine.get_all_orders()}
        assert set(counts) <= order_ids

    async def test_resting_limit_order_gets_exactly_one_fill(self) -> None:
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"))
        engine.set_last_price(SYM, Decimal("100"))
        order = Order(
            client_order_id="c",
            run_id="r",
            symbol=SYM,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("95"),
        )
        await engine.submit_order(order)
        await engine.check_resting_orders(SYM, Decimal("94"))
        await engine.check_resting_orders(SYM, Decimal("93"))  # filled orders are not re-filled
        assert Counter(f.order_id for f in engine.get_all_fills()) == {order.order_id: 1}


# ---------------------------------------------------------------------------
# T-19  runner edges (+ Amendment A)
# ---------------------------------------------------------------------------


class TestT19RunnerEdges:
    async def test_build_engine_without_a_clock_runs_and_only_the_seed_is_wall(self) -> None:
        runner = BacktestRunner(
            strategies=[_strategy({0: "buy", 6: "sell"})],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=LEGACY,
        )
        bars = {SYM: scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)}
        before = datetime.now(tz=UTC)
        engine, portfolio, _rm, exec_engine = runner._build_engine("noclock")
        assert engine._clock is WALL_CLOCK and exec_engine._clock is WALL_CLOCK
        await engine.start("noclock")
        try:
            await engine.run_backtest(bars)
        finally:
            await engine.stop()
        closes = {b.timestamp + H for b in bars[SYM]}
        curve = portfolio.get_equity_curve()
        assert before <= curve[0][0] <= datetime.now(tz=UTC)  # documented: the seed is wall
        assert {stamp for stamp, _eq in curve[1:]} == closes
        assert {f.executed_at for f in exec_engine.get_all_fills()} <= closes
        # WP22-P2B-CR-01: the run-scoped fallback BarClock is restored afterwards
        assert engine._clock is WALL_CLOCK and exec_engine._clock is WALL_CLOCK
        assert ambient_clock() is WALL_CLOCK

    async def _direct_engine(
        self, tag: str, script: dict[int, str] | None = None
    ) -> tuple[Any, Any, Any]:
        runner = BacktestRunner(
            strategies=[_strategy(script or {0: "buy", 6: "sell"})],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            enable_circuit_breaker=False,
            cost_profile=LEGACY,
        )
        engine, portfolio, _rm, exec_engine = runner._build_engine(tag)
        return engine, portfolio, exec_engine

    async def test_fallback_clock_is_restored_after_the_run_and_after_a_raise(self) -> None:
        """WP22-P2B-CR-01: the run-scoped fallback BarClock never outlives ``run_backtest``."""
        engine, _portfolio, exec_engine = await self._direct_engine("restore")
        bars = {SYM: scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)}
        await engine.start("restore")
        try:
            await engine.run_backtest(bars)
        finally:
            await engine.stop()
        assert engine._clock is WALL_CLOCK and exec_engine._clock is WALL_CLOCK
        assert ambient_clock() is WALL_CLOCK

        engine2, _p2, exec2 = await self._direct_engine("restore-raise")
        await engine2.start("restore-raise")
        try:
            with patch.object(engine2, "_process_bar", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    await engine2.run_backtest(bars)
        finally:
            await engine2.stop()
        assert engine2._clock is WALL_CLOCK and exec2._clock is WALL_CLOCK
        assert ambient_clock() is WALL_CLOCK

    async def test_two_backtests_on_one_engine_with_an_earlier_second_start_stamp_correctly(
        self,
    ) -> None:
        """WP22-P2B-CR-01: the second run does not inherit the first run's forward-only clock."""
        # the scripted strategy counts on_bar calls across runs: the second run is 72..143
        script = {0: "buy", 6: "sell", 72: "buy", 78: "sell"}
        engine, _portfolio, exec_engine = await self._direct_engine("twice", script)
        late = scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN + 200 * H, symbol=SYM)
        early = scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)
        await engine.start("twice")
        try:
            await engine.run_backtest({SYM: late})
            first = list(exec_engine.get_all_fills())
            await engine.run_backtest({SYM: early})
        finally:
            await engine.stop()
        first_ids = {f.fill_id for f in first}
        second = [f for f in exec_engine.get_all_fills() if f.fill_id not in first_ids]
        first_closes = {b.timestamp + H for b in late}
        second_closes = {b.timestamp + H for b in early}
        assert first and second
        assert {f.executed_at for f in first} <= first_closes
        assert {f.executed_at for f in second} <= second_closes
        assert min(f.executed_at for f in second) < min(f.executed_at for f in first)

    def test_the_empty_bars_branch_raises_and_never_reads_the_wall_clock(self) -> None:
        src = Path(pyinspect.getsourcefile(backtest_mod) or "").read_text()
        tree = ast.parse(src)
        assert "datetime.now" not in src and "UTC" not in src.replace("QuoteCurrency", "")
        branch = next(
            n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "all_bars"
        )
        assert any(isinstance(s, ast.Raise) for s in branch.orelse)

    async def test_empty_bars_fail_before_any_clock_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads = _install_wall_tripwires(monkeypatch)
        runner = BacktestRunner(
            strategies=[_strategy({})],
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            seed=7,
            cost_profile=LEGACY,
        )
        with pytest.raises(ValueError, match="Empty bar list"):
            await runner.run({SYM: []})
        assert reads == []

    async def test_portfolio_and_bar_clock_agree_after_every_bar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[BarClock] = []
        real = backtest_mod.BarClock

        def factory(start: datetime) -> BarClock:
            captured.append(real(start))
            return captured[-1]

        monkeypatch.setattr(backtest_mod, "BarClock", factory)
        checks: list[tuple[datetime, datetime]] = []
        original = PortfolioAccounting.advance_clock

        def wrapped(self_: PortfolioAccounting, at: datetime) -> None:
            original(self_, at)
            checks.append((self_._now(), captured[0].now()))

        monkeypatch.setattr(PortfolioAccounting, "advance_clock", wrapped)
        run = await _run(_FLAT_72, {0: "buy", 5: "sell"}, first_open=POST_WARMUP_OPEN)
        assert len(captured) == 1 and len(checks) == run.n_bars
        assert all(a == b for a, b in checks)
        assert [a for a, _b in checks] == run.closes
        assert captured[0].now() == run.closes[-1]


# ---------------------------------------------------------------------------
# T-09  paper / live unchanged, guards, tripwires, isolation
# ---------------------------------------------------------------------------

_T = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def _paper_engine(run_mode: RunMode = RunMode.PAPER, **kw: Any) -> StrategyEngine:
    strategy = MagicMock()
    strategy.strategy_id = "s"
    strategy.min_bars_required = 20
    strategy.htf_timeframes = None
    execution = AsyncMock()
    return StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=MagicMock(),
        market_data=AsyncMock(),
        portfolio=MagicMock(),
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=run_mode,
        **kw,
    )


def _live_engine() -> LiveExecutionEngine:
    return LiveExecutionEngine("live", DefaultRiskManager("live"), MagicMock())


class _SecretRepr:
    def now(self) -> datetime:
        return _T

    def __repr__(self) -> str:
        return "SECRET-REPR-DO-NOT-LEAK"


def _candidates() -> list[tuple[str, Any]]:
    bar = BarClock(_T)
    return [
        ("BarClock", bar),
        ("duck", _SecretRepr()),
        ("magicmock", MagicMock()),
        ("bound-now", SimpleNamespace(now=bar.now)),
        ("fresh-WallClock", WallClock()),
    ]


_CANDIDATE_IDS = [c[0] for c in _candidates()]


class TestT09GuardsAndMatrix:
    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    @pytest.mark.parametrize("name", _CANDIDATE_IDS)
    def test_strategy_engine_rejects_every_non_singleton_outside_backtest(
        self, mode: RunMode, name: str
    ) -> None:
        candidate = dict(_candidates())[name]
        with pytest.raises(ValueError, match="accepts only the wall clock") as err:
            _paper_engine(mode, clock=candidate)
        assert "SECRET-REPR" not in str(err.value)  # type names only (R11)

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    def test_strategy_engine_accepts_none_and_the_singleton_outside_backtest(
        self, mode: RunMode
    ) -> None:
        for clock in (None, WALL_CLOCK):
            engine = _paper_engine(mode, clock=clock)
            assert engine._clock is WALL_CLOCK
            engine._execution_engine.attach_clock.assert_not_called()

    @pytest.mark.parametrize("name", _CANDIDATE_IDS)
    def test_live_attach_clock_rejects_every_non_singleton(self, name: str) -> None:
        live = _live_engine()
        candidate = dict(_candidates())[name]
        with pytest.raises(ValueError, match="accepts only the wall clock") as err:
            live.attach_clock(candidate)
        assert "SECRET-REPR" not in str(err.value)
        assert live._clock is WALL_CLOCK

    def test_live_attach_clock_rejects_none_and_accepts_the_singleton(self) -> None:
        live = _live_engine()
        with pytest.raises(ValueError, match="NoneType"):
            live.attach_clock(None)
        live.attach_clock(WALL_CLOCK)
        assert live._clock is WALL_CLOCK

    @pytest.mark.parametrize("name", ["duck", "magicmock", "bound-now", "fresh-WallClock"])
    def test_backtest_strategy_engine_accepts_only_a_bar_clock(self, name: str) -> None:
        candidate = dict(_candidates())[name]
        with pytest.raises(ValueError, match="accepts only a BarClock") as err:
            _paper_engine(RunMode.BACKTEST, clock=candidate)
        assert "SECRET-REPR" not in str(err.value)

    def test_backtest_strategy_engine_accepts_none_and_a_bar_clock_and_attaches_it(self) -> None:
        assert _paper_engine(RunMode.BACKTEST)._clock is WALL_CLOCK
        bar = BarClock(_T)
        engine = _paper_engine(RunMode.BACKTEST, clock=bar)
        assert engine._clock is bar
        engine._execution_engine.attach_clock.assert_called_once_with(bar)

    def test_base_attach_clock_rejects_non_clocks_with_a_type_error(self) -> None:
        paper = PaperExecutionEngine("r", DefaultRiskManager("r"))
        for bad in (None, 42, object(), BarClock):
            with pytest.raises(TypeError, match="attach_clock requires a Clock"):
                paper.attach_clock(bad)
        paper.attach_clock(BarClock(_T))  # the paper/backtest engine is mode-agnostic
        assert isinstance(paper._clock, BarClock)

    def test_neither_base_nor_live_engine_takes_a_clock_constructor_argument(self) -> None:
        with pytest.raises(TypeError):
            LiveExecutionEngine("l", DefaultRiskManager("l"), MagicMock(), clock=BarClock(_T))
        assert (
            PaperExecutionEngine("p", DefaultRiskManager("p"), clock=BarClock(_T))._clock.now()
            == _T
        )

    def test_a_bound_bar_clock_now_reaches_a_portfolio_only_through_the_forbidden_clock_kwarg(
        self,
    ) -> None:
        """Documents SY-22-R2-03: the portfolio callable is unguarded; R7 (zero production
        ``clock=`` in apps/, AST-pinned) is the control, not a runtime check."""
        bar = BarClock(_T)
        portfolio = PortfolioAccounting("r", clock=bar.now)
        assert portfolio._now() == _T
        assert PortfolioAccounting("r")._clock is None  # R10: the orchestrator shape
        rebuilt = PortfolioAccounting.from_fills(run_id="r", initial_cash=Decimal("10"), fills=[])
        assert rebuilt._clock is None  # R10: the orchestrator's resume shape

    def test_class_level_default_survives_a_subclass_that_skips_super_init(self) -> None:
        class Bare(PaperExecutionEngine):
            def __init__(self) -> None:
                pass

        assert Bare()._clock is WALL_CLOCK


class TestT09Tripwires:
    async def test_paper_start_fails_closed_under_a_non_wall_ambient_clock(self) -> None:
        engine = _paper_engine(RunMode.PAPER)
        with use_clock(BarClock(_T)), pytest.raises(RuntimeError, match="non-wall ambient clock"):
            await engine.start("run")

    async def test_start_fails_closed_for_a_real_exec_engine_with_a_non_wall_clock(self) -> None:
        paper = PaperExecutionEngine("r", DefaultRiskManager("r"))
        paper.attach_clock(BarClock(_T))  # what no production path may do
        engine = StrategyEngine(
            strategies=[MagicMock(strategy_id="s", min_bars_required=20, htf_timeframes=None)],
            execution_engine=paper,
            risk_manager=MagicMock(),
            market_data=AsyncMock(),
            portfolio=MagicMock(),
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            run_mode=RunMode.PAPER,
        )
        with pytest.raises(RuntimeError, match="execution engine clock is not the wall clock"):
            await engine.start("run")

    async def test_start_fails_closed_for_a_non_wall_engine_clock_and_stays_idle(self) -> None:
        """WP22-P2B-SEC-07 (a) + (c): the own-clock branch trips, and before any state change."""
        engine = _paper_engine(RunMode.PAPER)
        engine._clock = BarClock(_T)  # what the R1 constructor guard makes unreachable
        with pytest.raises(RuntimeError, match=r"engine clock is not the wall clock \(BarClock\)"):
            await engine.start("run")
        assert engine.state is EngineState.IDLE

    async def test_ambient_and_exec_clock_trips_leave_the_engine_idle(self) -> None:
        """WP22-P2B-SEC-07 (c): the trip precedes ``STARTING`` for every branch."""
        ambient = _paper_engine(RunMode.PAPER)
        with use_clock(BarClock(_T)), pytest.raises(RuntimeError):
            await ambient.start("run")
        assert ambient.state is EngineState.IDLE
        paper = PaperExecutionEngine("r", DefaultRiskManager("r"))
        paper.attach_clock(BarClock(_T))
        real = _paper_engine(RunMode.PAPER)
        real._execution_engine = paper
        with pytest.raises(RuntimeError, match="execution engine clock"):
            await real.start("run")
        assert real.state is EngineState.IDLE

    async def test_exec_clock_tripwire_message_never_leaks_the_clock_repr(self) -> None:
        """WP22-P2B-SEC-07 (b) / R11: type name only, never ``repr(clock)``."""
        paper = PaperExecutionEngine("r", DefaultRiskManager("r"))
        paper.attach_clock(_SecretRepr())  # a Clock by protocol; the tripwire must refuse it
        engine = _paper_engine(RunMode.PAPER)
        engine._execution_engine = paper
        with pytest.raises(RuntimeError, match="execution engine clock") as err:
            await engine.start("run")
        assert "SECRET" not in str(err.value)
        assert "_SecretRepr" in str(err.value)

    async def test_mock_exec_engines_stay_green_the_type_gate(self) -> None:
        engine = _paper_engine(RunMode.PAPER)  # AsyncMock exec: _clock is a child Mock
        await engine.start("run")
        assert engine._execution_engine._clock is not WALL_CLOCK  # the hazard, documented

    async def test_start_in_backtest_mode_is_not_tripped(self) -> None:
        engine = _paper_engine(RunMode.BACKTEST, clock=BarClock(_T))
        with use_clock(BarClock(_T)):
            await engine.start("run")  # no tripwire in BACKTEST

    async def test_run_live_loop_entry_tripwire(self) -> None:
        engine = _paper_engine(RunMode.PAPER)
        await engine.start("run")
        # wait_for: a missing tripwire would start the endless poll loop; fail, do not hang
        with use_clock(BarClock(_T)), pytest.raises(RuntimeError, match="non-wall ambient clock"):
            await asyncio.wait_for(engine.run_live_loop(), timeout=3)

    async def test_live_on_start_tripwires_before_any_io(self) -> None:
        live = _live_engine()
        with use_clock(BarClock(_T)), pytest.raises(RuntimeError, match="non-wall ambient clock"):
            await live.on_start()
        live._clock = BarClock(_T)  # bypass attach_clock on purpose
        with pytest.raises(RuntimeError, match="clock is not the wall clock"):
            await live.on_start()
        live._exchange.load_markets.assert_not_called()

    @pytest.mark.parametrize("mode", ["paper", "live"])
    async def test_orchestrator_entry_fails_closed_before_constructing_anything(
        self, mode: str
    ) -> None:
        from api.services import run_orchestrator as orch

        _CapturingEngine.instances.clear()
        common: dict[str, Any] = {
            "run_id_str": str(uuid.uuid4()),
            "strategy_cls": MagicMock,
            "strategy_name": "grid_trading",
            "strategy_params": {},
            "symbols": [_ORCH_SYM],
            "timeframe": TimeFrame.ONE_HOUR,
            "initial_capital": "10000",
            "cost_profile": LEGACY,
            "resume": _empty_resume() if mode == "live" else None,
        }
        engine_cls = MagicMock()
        patches = [
            patch("api.db.session.get_session_factory", _session_factory()),
            patch("api.config.get_settings", return_value=_settings()),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.strategy_engine.StrategyEngine", _CapturingEngine),
            patch.object(orch, "make_circuit_breaker_alert_listener", MagicMock()),
            patch.object(orch, "record_audit_event", AsyncMock()),
            patch.object(orch, "_auto_retry_paper_run", AsyncMock()),
            patch.object(orch, "build_live_ccxt_exchange", MagicMock()),
            patch("trading.engines.paper.PaperExecutionEngine", engine_cls),
            patch("trading.engines.live.LiveExecutionEngine", engine_cls),
        ]
        runner = orch.run_paper_engine if mode == "paper" else orch.run_live_engine
        if mode == "live":
            common["protective_mode"] = False
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with use_clock(BarClock(_T)):
                await runner(**common)  # the generic handler marks the run errored; no raise
        assert _CapturingEngine.instances == []  # the engine was never built
        engine_cls.assert_not_called()

    @pytest.mark.parametrize("trip", [True, False], ids=["tripwire", "control-other-error"])
    async def test_paper_tripwire_does_not_schedule_an_auto_retry(self, trip: bool) -> None:
        """WP22-P2B-SEC-06: a retry would inherit the poisoned ambient clock.

        The control (an unrelated error) proves the harness does schedule a retry.
        """
        import contextlib

        from api.services import run_orchestrator as orch

        _CapturingEngine.instances.clear()
        retry_mock = AsyncMock()
        patches = [
            patch("api.db.session.get_session_factory", _session_factory()),
            patch("api.config.get_settings", return_value=_settings()),
            patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
            patch("trading.portfolio.PortfolioAccounting", MagicMock()),
            patch("trading.strategy_engine.StrategyEngine", _CapturingEngine),
            patch.object(orch, "make_circuit_breaker_alert_listener", MagicMock()),
            patch.object(orch, "record_audit_event", AsyncMock()),
            patch.object(orch, "_auto_retry_paper_run", retry_mock),
            patch("trading.engines.paper.PaperExecutionEngine", MagicMock()),
        ]
        if not trip:
            patches.append(patch.object(orch, "validate_smoke_run", side_effect=RuntimeError("x")))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with use_clock(BarClock(_T)) if trip else contextlib.nullcontext():
                await orch.run_paper_engine(
                    run_id_str=str(uuid.uuid4()),
                    strategy_cls=MagicMock,
                    strategy_name="grid_trading",
                    strategy_params={},
                    symbols=[_ORCH_SYM],
                    timeframe=TimeFrame.ONE_HOUR,
                    initial_capital="10000",
                    cost_profile=LEGACY,
                    resume=None,
                )
            await asyncio.sleep(0)  # let a fire-and-forget retry task start
        assert retry_mock.call_count == (0 if trip else 1)


class TestT09StampsAndIsolation:
    async def test_paper_stamps_are_wall_time_through_the_one_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"))
        portfolio = PortfolioAccounting("r")
        engine.attach_risk_state_source(portfolio)
        engine.set_last_price(SYM, Decimal("100"))
        before = datetime.now(tz=UTC)
        orders = await engine.process_signal(
            Signal(
                strategy_id="s",
                symbol=SYM,
                direction=SignalDirection.BUY,
                target_position=Decimal("1000"),
            )
        )
        after = datetime.now(tz=UTC)
        assert orders, "the paper BUY must be accepted"
        (order,) = orders
        (fill,) = engine.get_all_fills()
        assert before <= fill.executed_at <= after
        assert before <= order.created_at <= order.updated_at <= after
        assert before <= engine.positions[SYM].opened_at <= after
        portfolio.update_position(fill, Decimal("100"))
        portfolio.update_market_prices({SYM: Decimal("101")})
        assert all(
            before - timedelta(seconds=5) <= ts <= after + timedelta(seconds=5)
            for ts, _eq in portfolio.get_equity_curve()
        )
        # S18 / R14: the only supported seam is trading.clock._wall_now
        frozen = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        monkeypatch.setattr(clock_mod, "_wall_now", lambda: frozen)
        assert WALL_CLOCK.now() == frozen and utc_now() == frozen
        engine.set_last_price(SYM, Decimal("110"))
        (sell,) = await engine.process_signal(
            Signal(
                strategy_id="s",
                symbol=SYM,
                direction=SignalDirection.SELL,
                target_position=Decimal("0"),
            )
        )
        assert sell.created_at == frozen and sell.updated_at == frozen
        assert engine.get_all_fills()[-1].executed_at == frozen
        # WP22-P2B1-03: the portfolio's seed, position and curve stamps follow the
        # SAME seam (``portfolio._utc_now`` delegates to ``clock._wall_now``).
        fresh = PortfolioAccounting("r2")
        assert fresh.get_equity_curve()[0][0] == frozen  # seed point
        fresh.update_position(engine.get_all_fills()[0], Decimal("100"))
        position = fresh.get_position(SYM)
        assert position is not None and position.opened_at == frozen
        fresh.update_market_prices({SYM: Decimal("102")})
        assert {ts for ts, _eq in fresh.get_equity_curve()} == {frozen}

    async def test_explicit_stamps_do_not_rely_on_the_ambient_fallback(self) -> None:
        """SY-22-05 (c): every persisted field is stamped from the ENGINE clock.

        The engine holds a BarClock while the ambient clock is the wall clock (no
        ``use_clock``): if any site fell back to the ``utc_now`` default it would
        stamp wall time and this test would see it.
        """
        clock = BarClock(_T)
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"), clock=clock)
        engine.attach_risk_state_source(PortfolioAccounting("r", clock=clock.now))
        engine.set_last_price(SYM, Decimal("100"))
        assert ambient_clock() is WALL_CLOCK
        (buy,) = await engine.process_signal(
            Signal(
                strategy_id="s",
                symbol=SYM,
                direction=SignalDirection.BUY,
                target_position=Decimal("1000"),
            )
        )
        assert buy.created_at == _T and buy.updated_at == _T
        clock.advance(_T + H)
        (sell,) = await engine.process_signal(
            Signal(
                strategy_id="s",
                symbol=SYM,
                direction=SignalDirection.SELL,
                target_position=Decimal("0"),
            )
        )
        assert sell.created_at == _T + H and sell.updated_at == _T + H
        assert [f.executed_at for f in engine.get_all_fills()] == [_T, _T + H]
        assert engine.positions[SYM].opened_at == _T
        assert engine.positions[SYM].updated_at == _T + H
        for order in engine.get_all_orders():
            assert order.updated_at in {_T, _T + H}  # _transition / fill-site stamps

    async def test_paper_strategy_engine_trade_exit_at_is_wall_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frozen = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        monkeypatch.setattr(clock_mod, "_wall_now", lambda: frozen)
        engine = _paper_engine(RunMode.PAPER)
        assert engine._clock.now() == frozen
        assert engine._clock is WALL_CLOCK

    async def test_persist_paper_results_snapshot_at_is_wall_time(self) -> None:
        portfolio = PortfolioAccounting("r")
        engine = PaperExecutionEngine("r", DefaultRiskManager("r"))
        engine.attach_risk_state_source(portfolio)
        engine.set_last_price(SYM, Decimal("100"))
        (order,) = await engine.process_signal(
            Signal(
                strategy_id="s",
                symbol=SYM,
                direction=SignalDirection.BUY,
                target_position=Decimal("1000"),
            )
        )
        portfolio.update_position(engine.get_all_fills()[0], Decimal("100"))
        db = MagicMock()
        db.add_all = MagicMock()
        db.commit = AsyncMock()
        db.flush = AsyncMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=db)
        ctx.__aexit__ = AsyncMock(return_value=False)
        before = datetime.now(tz=UTC)
        with patch("api.db.session.get_session_factory", return_value=MagicMock(return_value=ctx)):
            await persist_paper_results(
                run_id_str=str(uuid.uuid4()),
                portfolio=portfolio,
                execution_engine=engine,
                log=structlog.get_logger("wp22"),
            )
        after = datetime.now(tz=UTC)
        rows = [r for call in db.add_all.call_args_list for r in call.args[0]]
        snaps = [r for r in rows if isinstance(r, PositionSnapshotORM)]
        assert snaps and all(before <= s.snapshot_at <= after for s in snaps)
        assert order.created_at <= order.updated_at

    async def test_utc_now_is_wall_time_after_return_after_raise_and_after_cancel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frozen = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        monkeypatch.setattr(clock_mod, "_wall_now", lambda: frozen)
        await _run(_FLAT_72, {}, first_open=POST_WARMUP_OPEN)  # returns
        assert ambient_clock() is WALL_CLOCK and utc_now() == frozen

        for exc in (RuntimeError("boom"), asyncio.CancelledError()):
            runner = BacktestRunner(
                strategies=[_strategy({})],
                symbols=[SYM],
                timeframe=TimeFrame.ONE_HOUR,
                seed=7,
                cost_profile=LEGACY,
            )
            bars = {SYM: scripted_path(_FLAT_72, first_open=POST_WARMUP_OPEN, symbol=SYM)}
            engine, *_ = runner._build_engine("x", clock=BarClock(bars[SYM][0].timestamp))
            await engine.start("x")
            seen: list[bool] = []

            async def failing(
                *_a: Any,
                _exc: BaseException = exc,
                _seen: list[bool] = seen,
                **_k: Any,
            ) -> None:
                _seen.append(ambient_clock() is not WALL_CLOCK)  # inside: the bar clock
                raise _exc

            monkeypatch.setattr(engine, "_process_bar", failing)
            with pytest.raises(type(exc)):
                await engine.run_backtest(bars)
            assert seen == [True]  # the clock WAS ambient inside the loop ...
            assert ambient_clock() is WALL_CLOCK and utc_now() == frozen  # ... and is restored
            await engine.stop()

    async def test_concurrent_backtests_and_a_paper_task_never_see_each_others_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: dict[int, list[tuple[datetime, datetime]]] = {}
        original = StrategyEngine._process_bar

        async def interleaving(self_: StrategyEngine, *a: Any, **k: Any) -> Any:
            await asyncio.sleep(0)  # force the three tasks to interleave per bar
            observed.setdefault(id(self_), []).append((utc_now(), self_._clock.now()))
            return await original(self_, *a, **k)

        monkeypatch.setattr(StrategyEngine, "_process_bar", interleaving)
        paper_seen: list[tuple[bool, datetime]] = []
        stop = asyncio.Event()

        async def paper_task() -> None:
            while not stop.is_set():
                paper_seen.append((ambient_clock() is WALL_CLOCK, utc_now()))
                await asyncio.sleep(0)

        first_a = POST_WARMUP_OPEN
        first_b = datetime(2024, 5, 1, 0, 0, tzinfo=UTC)
        background = asyncio.ensure_future(paper_task())
        run_a, run_b = await asyncio.gather(
            _run(_FLAT_72, {0: "buy", 10: "sell"}, first_open=first_a),
            _run(_FLAT_72, {0: "buy", 10: "sell"}, first_open=first_b),
        )
        stop.set()
        await background
        assert len(observed) == 2 and all(len(v) == 72 for v in observed.values())
        for samples in observed.values():
            assert all(ambient == own for ambient, own in samples)  # ambient == own BarClock
        assert paper_seen and all(is_wall for is_wall, _ in paper_seen)
        floor = datetime.now(tz=UTC) - timedelta(minutes=5)
        assert all(ts >= floor for _w, ts in paper_seen)  # wall time, never a bar time
        assert {f.executed_at for f in run_a.fills} <= set(run_a.closes)
        assert {f.executed_at for f in run_b.fills} <= set(run_b.closes)
        assert run_a.closes[0] != run_b.closes[0]

    def test_other_threads_never_inherit_the_backtest_context(self) -> None:
        # complements R13: on 3.11-3.13 a new thread starts with an empty context
        import threading

        seen: list[Any] = []
        with use_clock(BarClock(_T)):
            t = threading.Thread(target=lambda: seen.append(ambient_clock()))
            t.start()
            t.join()
        assert seen == [WALL_CLOCK]
