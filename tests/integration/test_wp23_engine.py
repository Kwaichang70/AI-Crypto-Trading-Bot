"""
tests/integration/test_wp23_engine.py
---------------------------------------
WP2.3 P2: ``StrategyEngine.run_backtest`` walks the aligned timeline (hermetic: no Postgres,
no network).  The engine is driven directly (no ``BacktestRunner``): P2 changes the engine only,
and its input contract is "several symbols start and end on a full row".

    WP23-T-05  QD fixture (12 x 1h; A complete, B missing {4,5,6}, C missing {9}): an absent
               symbol gets no on_bar, no mark, no bracket / trailing check, no history step;
               a held symbol keeps its last mark and still counts as exposed
    WP23-T-08  engine half: untrimmed / off-grid multi-symbol input is rejected up front
               (``bar_alignment_not_trimmed`` / ``bar_alignment_off_grid``), before any state
    WP23-T-12  seeded random gaps (B 2 %, C 5 %): every fill / mark uses its own symbol's bar of
               the step; stamps in {t0} U {step + tf}, strictly increasing
    WP23-T-13  a symbol's signal stream does not depend on the other symbols; no synthetic bar
    WP23-T-14  WP2.2 clock pins: ``close_i`` unchanged, one ``build_timeline`` call, the
               reconstructed ``Step.ts`` is never read (P1-CR-06)
    I-1        single symbol: plain index walk, no ``build_timeline``, no validation

``tests/unit/test_wp22_ast_pins.py`` is not edited (WP23-T-14).
"""

from __future__ import annotations

import ast
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import OrderSide, RunMode, SignalDirection, TimeFrame
from data.bar_alignment import BarAlignmentError
from trading import strategy_engine as engine_mod
from trading.clock import WALL_CLOCK, BarClock
from trading.engines.paper import PaperExecutionEngine
from trading.models import Signal
from trading.portfolio import PortfolioAccounting
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import StrategyEngine

A = "BTC/USDT"
B = "ETH/USDT"
C = "SOL/USDT"
H = timedelta(hours=1)
T0 = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)  # the QD fixture start (on the 1h grid)
_ENGINE_SRC = Path(engine_mod.__file__)

_BRACKET_AND_TRAILING: dict[str, Any] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.02,
    "bracket_take_profit_pct": 0.04,
    "trailing_stop_pct": 0.03,
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _bar(symbol: str, k: int, close: Decimal | int | str, *, minutes: int = 0) -> OHLCVBar:
    c = Decimal(str(close))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=T0 + k * H + timedelta(minutes=minutes),
        open=c,
        high=c,
        low=c,
        close=c,
        volume=Decimal(10),
    )


def _series(symbol: str, base: int, ks: Sequence[int]) -> list[OHLCVBar]:
    """One bar per ``k`` in ``ks``; ``close = base + k`` (distinct per symbol and k)."""
    return [_bar(symbol, k, base + k) for k in ks]


def _qd_bars(*, b_close: dict[int, int] | None = None) -> dict[str, list[OHLCVBar]]:
    """QD fixture: 12 x 1h from 2026-03-02T00:00Z; B misses {4,5,6}, C misses {9}."""
    b_close = b_close or {}
    return {
        A: _series(A, 100, range(12)),
        B: [_bar(B, k, b_close.get(k, 200 + k)) for k in (0, 1, 2, 3, 7, 8, 9, 10, 11)],
        C: _series(C, 300, [k for k in range(12) if k != 9]),
    }


class _Rec(PortfolioAccounting):
    """Records the clock pushes and every mark (stamp, prices, price held BEFORE the mark)."""

    def __init__(self, clock: BarClock, symbols: Sequence[str]) -> None:
        super().__init__("wp23", Decimal("10000"), clock=clock.now)
        self.symbols = tuple(symbols)
        self.pushes: list[datetime] = []
        self.curve_len_at_push: list[int] = []
        self.marks: list[tuple[datetime, dict[str, Decimal], dict[str, Decimal]]] = []

    def advance_clock(self, at: datetime) -> None:
        self.pushes.append(at)
        self.curve_len_at_push.append(len(self.get_equity_curve()))
        super().advance_clock(at)

    def update_market_prices(self, prices: dict[str, Decimal]) -> None:
        before = {
            s: p.current_price for s in self.symbols if (p := self.get_position(s)) is not None
        }
        self.marks.append((self.pushes[-1], dict(prices), before))
        super().update_market_prices(prices)


class _Spy(BaseStrategy):
    """Records every ``on_bar`` call; BUYs / SELLs on scripted ``(symbol, bar ts)`` keys."""

    metadata = StrategyMetadata(name="wp23_spy", description="test-only")
    requires_exit_manager = False

    def __init__(
        self,
        clock: BarClock | None = None,
        *,
        buys: Sequence[tuple[str, int]] = (),
        rng: random.Random | None = None,
    ) -> None:
        super().__init__("wp23-spy")
        self.clock = clock
        self.buys = {(s, T0 + k * H) for s, k in buys}
        self.rng = rng
        self.held: set[str] = set()
        #: (symbol, bar ts, history length, clock.now() at the call)
        self.calls: list[tuple[str, datetime, int, datetime | None]] = []
        self.histories: list[Sequence[OHLCVBar]] = []

    @property
    def min_bars_required(self) -> int:
        return 1

    def _signal(self, symbol: str, direction: SignalDirection, notional: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            target_position=Decimal(notional),
            confidence=1.0,
        )

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        last = bars[-1]
        stamp = self.clock.now() if self.clock is not None else None
        self.calls.append((last.symbol, last.timestamp, len(bars), stamp))
        self.histories.append(bars)
        if self.rng is not None:  # seeded random entries and exits (T-12)
            u = self.rng.random()
            if last.symbol not in self.held and u < 0.10:
                self.held.add(last.symbol)
                return [self._signal(last.symbol, SignalDirection.BUY, "300")]
            if last.symbol in self.held and u < 0.20:
                self.held.discard(last.symbol)
                return [self._signal(last.symbol, SignalDirection.SELL, "0")]
            return []
        if (last.symbol, last.timestamp) in self.buys:
            return [self._signal(last.symbol, SignalDirection.BUY, "1000")]
        return []


class _Decider(BaseStrategy):
    """A pure function of the symbol's own history (T-13): close above its 5-bar mean."""

    metadata = StrategyMetadata(name="wp23_decider", description="test-only")
    requires_exit_manager = False

    def __init__(self, symbol: str) -> None:
        super().__init__("wp23-decider")
        self.symbol = symbol
        self.stream: list[tuple[datetime, int, bool]] = []

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        if bars[-1].symbol == self.symbol:
            window = [b.close for b in bars[-5:]]
            self.stream.append(
                (bars[-1].timestamp, len(bars), bars[-1].close > sum(window) / len(window))
            )
        return []


class Run:
    def __init__(
        self,
        engine: StrategyEngine,
        portfolio: _Rec,
        execution: PaperExecutionEngine,
        clock: BarClock | None,
        spy: BaseStrategy,
    ) -> None:
        self.engine = engine
        self.portfolio = portfolio
        self.execution = execution
        self.clock = clock
        self.spy = spy
        self.summary: dict[str, Any] = {}
        self.exit_checks: list[tuple[str, str, datetime, Decimal]] = []

    @property
    def fills(self) -> list[Any]:
        return self.execution.get_all_fills()


def _build(
    symbols: Sequence[str],
    strategy: BaseStrategy,
    *,
    t0: datetime = T0,
    warmup: int = 1,
    config: dict[str, Any] | None = None,
    with_clock: bool = True,
) -> Run:
    clock = BarClock(start=t0) if with_clock else None
    if clock is not None and isinstance(strategy, _Spy):
        strategy.clock = clock  # the spy stamps each call with the step's clock reading
    rm = DefaultRiskManager(run_id="wp23")
    execution = PaperExecutionEngine("wp23", rm, initial_cash=Decimal("10000"))
    portfolio = _Rec(clock if clock is not None else BarClock(start=t0), symbols)
    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=rm,
        market_data=MagicMock(),
        portfolio=portfolio,
        symbols=list(symbols),
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
        config={"warmup_bars": warmup, **(config or {})},
        clock=clock,
    )
    run = Run(engine, portfolio, execution, clock, strategy)
    if clock is not None:
        for kind, manager in (
            ("bracket", engine._bracket_exit),
            ("trailing", engine._trailing_stop),
        ):
            if manager is None:
                continue

            def spy_check(
                *args: Any, _orig: Any = manager.check, _kind: str = kind, **kwargs: Any
            ) -> Any:
                run.exit_checks.append(
                    (_kind, kwargs["symbol"], clock.now(), kwargs["current_price"])
                )
                return _orig(*args, **kwargs)

            manager.check = spy_check  # type: ignore[method-assign]
    return run


async def _run(
    symbols: Sequence[str],
    bars: dict[str, list[OHLCVBar]],
    strategy: BaseStrategy,
    *,
    oos_start_index: int | None = None,
    **kw: Any,
) -> Run:
    run = _build(symbols, strategy, **kw)
    await run.engine.start("wp23")
    try:
        run.summary = await run.engine.run_backtest(bars, oos_start_index=oos_start_index)
    finally:
        await run.engine.stop()
    return run


def _stamp(k: int) -> datetime:
    """The bar-close stamp of step k of the 1h fixture grid."""
    return T0 + (k + 1) * H


# ---------------------------------------------------------------------------
# T-05  the QD fixture through the engine
# ---------------------------------------------------------------------------


class TestT05AbsentSymbolIsSkipped:
    async def test_an_absent_symbol_gets_no_on_bar_and_its_history_skips_the_hole(self) -> None:
        spy = _Spy()
        run = await _run([A, B, C], _qd_bars(), spy)
        by_sym = {s: [(c[1], c[2]) for c in spy.calls if c[0] == s] for s in (A, B, C)}
        # warmup_bars = 1: step 0 is not shown to the strategies
        assert by_sym[A] == [(T0 + k * H, k + 1) for k in range(1, 12)]
        assert by_sym[B] == [
            (T0 + k * H, n)
            for k, n in ((1, 2), (2, 3), (3, 4), (7, 5), (8, 6), (9, 7), (10, 8), (11, 9))
        ]
        assert by_sym[C] == [
            (T0 + k * H, k + 1 if k < 9 else k) for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
        ]
        assert run.engine._bar_count == 11, "steps processed: 12 timeline steps minus 1 warmup"

    async def test_no_step_pairs_bars_of_different_timestamps(self) -> None:
        """HEAD oracle: num_bars 9 and A@04:00 paired with B@07:00 (index pairing)."""
        spy = _Spy()
        await _run([A, B, C], _qd_bars(), spy)
        assert len(spy.calls) == 11 + 8 + 10
        for symbol, ts, _n, at in spy.calls:
            assert at == ts + H, (symbol, ts, at)
        # the HEAD pairing would show B@07:00 in a step stamped A@04:00 + 1h
        assert (B, T0 + 7 * H, 5, _stamp(4)) not in spy.calls

    async def test_one_mark_per_step_and_the_absent_symbol_is_not_marked(self) -> None:
        spy = _Spy()
        run = await _run([A, B, C], _qd_bars(), spy)
        marks = run.portfolio.marks
        assert [m[0] for m in marks] == [_stamp(k) for k in range(12)], "one mark per step"
        for k, (_, prices, _before) in enumerate(marks):
            expected = {A} | ({B} if k not in (4, 5, 6) else set()) | ({C} if k != 9 else set())
            assert set(prices) == expected, k
            assert prices[A] == Decimal(100 + k)

    async def test_clock_stamps_are_the_timeline_closes_in_order(self) -> None:
        run = await _run([A, B, C], _qd_bars(), _Spy())
        assert run.portfolio.pushes == [T0 + H + k * H for k in range(12)]  # 01:00 + k h
        assert all(
            a < b for a, b in zip(run.portfolio.pushes, run.portfolio.pushes[1:], strict=False)
        )
        assert run.portfolio.get_equity_curve()[0][0] == T0, "t0 == timeline[0]"

    async def test_a_held_absent_symbol_keeps_its_last_mark_and_gets_no_exit_check(self) -> None:
        bars = _qd_bars(b_close={7: 150, 8: 150, 9: 150, 10: 150, 11: 150})
        spy = _Spy(buys=[(B, 2)])
        run = await _run([A, B, C], bars, spy, config=_BRACKET_AND_TRAILING)
        # the exit managers see B only on its own present steps (>= warmup)
        present_b = {_stamp(k) for k in (1, 2, 3, 7, 8, 9, 10, 11)}
        for kind in ("bracket", "trailing"):
            seen = [at for k_, s, at, _p in run.exit_checks if k_ == kind and s == B]
            assert set(seen) == present_b, kind
            assert len(seen) == len(present_b), kind
        for kind in ("bracket", "trailing"):
            for sym, absent in ((C, {_stamp(9)}), (A, set())):
                seen = {at for k_, s, at, _p in run.exit_checks if k_ == kind and s == sym}
                assert not seen & absent, (kind, sym)
                assert len(seen) == 11 - len(absent), (kind, sym)
        # the portfolio keeps B's last mark (close at k=3) through the three absent steps
        before = {at: held.get(B) for at, _p, held in run.portfolio.marks}
        for k in (4, 5, 6, 7):
            assert before[_stamp(k)] == Decimal(203), k
        assert before[_stamp(8)] == Decimal(150)
        # B's stop-loss exit happens on the FIRST present bar after the hole (k = 7)
        sells = [f for f in run.fills if f.symbol == B and f.side == OrderSide.SELL]
        buys = [f for f in run.fills if f.symbol == B and f.side == OrderSide.BUY]
        assert [f.executed_at for f in buys] == [_stamp(2)]
        assert [f.executed_at for f in sells] == [_stamp(7)]

    async def test_a_held_absent_symbol_counts_as_exposed_through_its_absent_steps(self) -> None:
        bars = _qd_bars(b_close={7: 150, 8: 150, 9: 150, 10: 150, 11: 150})
        run = await _run([A, B, C], bars, _Spy(buys=[(B, 2)]), config=_BRACKET_AND_TRAILING)
        # held after the processing of steps 2..6 (4, 5, 6 are B's absent steps)
        assert run.summary["exposure_bars_per_symbol"] == {A: 0, B: 5, C: 0}
        assert run.summary["exposure_bars_total"] == 5

    async def test_oos_start_index_is_an_ordinal_into_the_timeline(self) -> None:
        run = await _run([A, B, C], _qd_bars(), _Spy(), oos_start_index=5)
        assert run.summary["oos_equity_curve_offset"] == run.portfolio.curve_len_at_push[5]
        # 10 > min(len) = 9 (HEAD never reached it); 12 is past the 12-step timeline
        run10 = await _run([A, B, C], _qd_bars(), _Spy(), oos_start_index=10)
        assert run10.summary["oos_equity_curve_offset"] == run10.portfolio.curve_len_at_push[10]
        run12 = await _run([A, B, C], _qd_bars(), _Spy(), oos_start_index=12)
        assert run12.summary["oos_equity_curve_offset"] is None

    async def test_warmup_counts_timeline_steps(self) -> None:
        spy = _Spy()
        run = await _run([A, B, C], _qd_bars(), spy, warmup=6)
        assert [c[1] for c in spy.calls if c[0] == A] == [T0 + k * H for k in range(6, 12)]
        assert [m[0] for m in run.portfolio.marks] == [_stamp(k) for k in range(12)]

    async def test_a_direct_engine_without_a_clock_seeds_from_the_full_first_row(self) -> None:
        run = await _run([A, B, C], _qd_bars(), _Spy(buys=[(A, 1)]), with_clock=False)
        assert [f.executed_at for f in run.fills] == [_stamp(1)]
        assert run.engine._clock is WALL_CLOCK  # the fallback clock is scoped to the call


# ---------------------------------------------------------------------------
# T-08 (engine half)  the multi-symbol input guard
# ---------------------------------------------------------------------------


def _assert_untouched(run: Run, spy: _Spy) -> None:
    assert spy.calls == []
    assert run.portfolio.pushes == [] and run.portfolio.marks == []
    assert run.engine._bar_count == 0
    assert run.engine._backtest_running is False


class TestT08EngineGuard:
    @pytest.mark.parametrize(
        "case",
        ["late_first_row", "early_last_row", "late_first_row_of_first_symbol"],
    )
    async def test_untrimmed_input_is_rejected_as_not_trimmed(self, case: str) -> None:
        bars = {A: _series(A, 100, range(12)), B: _series(B, 200, range(12))}
        if case == "late_first_row":
            bars[B] = _series(B, 200, range(1, 12))
        elif case == "early_last_row":
            bars[B] = _series(B, 200, range(11))
        else:
            bars[A] = _series(A, 100, range(2, 12))
        spy = _Spy()
        run = _build([A, B], spy)
        await run.engine.start("wp23")
        with pytest.raises(BarAlignmentError) as ei:
            await run.engine.run_backtest(bars)
        assert ei.value.code == "bar_alignment_not_trimmed"
        assert str(ei.value) == "bar alignment check failed (bar_alignment_not_trimmed)"
        _assert_untouched(run, spy)

    @pytest.mark.parametrize("case", ["phase", "duplicate", "non_increasing"])
    async def test_off_grid_input_is_rejected_as_off_grid(self, case: str) -> None:
        bars = {A: _series(A, 100, range(12)), B: _series(B, 200, range(12))}
        if case == "phase":
            bars[B] = [_bar(B, k, 200 + k, minutes=30) for k in range(12)]
        elif case == "duplicate":
            bars[B] = [*bars[B][:5], bars[B][4], *bars[B][5:]]
        else:
            bars[B][5], bars[B][6] = bars[B][6], bars[B][5]
        spy = _Spy()
        run = _build([A, B], spy)
        await run.engine.start("wp23")
        with pytest.raises(BarAlignmentError) as ei:
            await run.engine.run_backtest(bars)
        assert ei.value.code == "bar_alignment_off_grid"
        assert str(ei.value) == "bar alignment check failed (bar_alignment_off_grid)"
        assert [v.symbol for v in ei.value.violations] == [B]
        _assert_untouched(run, spy)

    async def test_a_rejection_leaves_the_engine_usable(self) -> None:
        spy = _Spy()
        run = _build([A, B], spy)
        await run.engine.start("wp23")
        bad = {A: _series(A, 100, range(12)), B: _series(B, 200, range(1, 12))}
        with pytest.raises(BarAlignmentError):
            await run.engine.run_backtest(bad)
        good = {A: _series(A, 100, range(12)), B: _series(B, 200, range(12))}
        summary = await run.engine.run_backtest(good)
        assert summary["exposure_bars_total"] == 0
        assert run.engine._bar_count == 11

    async def test_the_message_never_carries_data(self) -> None:
        bars = {A: _series(A, 100, range(12)), B: [_bar(B, k, 777, minutes=30) for k in range(12)]}
        run = _build([A, B], _Spy())
        await run.engine.start("wp23")
        with pytest.raises(BarAlignmentError) as ei:
            await run.engine.run_backtest(bars)
        text = f"{ei.value} {ei.value!s}"
        assert B not in text and "777" not in text and "2026" not in text


# ---------------------------------------------------------------------------
# I-1  single symbol: the plain index walk
# ---------------------------------------------------------------------------


class TestSingleSymbolFastPath:
    async def test_the_walk_is_the_index_walk_and_never_builds_a_timeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise AssertionError("build_timeline must not run for one symbol")

        monkeypatch.setattr(engine_mod, "build_timeline", boom)
        bars = {A: _series(A, 100, range(20))}
        spy = _Spy()
        run = await _run([A], bars, spy, warmup=3)
        assert [(c[1], c[2]) for c in spy.calls] == [(T0 + k * H, k + 1) for k in range(3, 20)]
        assert run.portfolio.pushes == [_stamp(k) for k in range(20)]
        assert all(h[-1] is bars[A][len(h) - 1] for h in spy.histories)

    async def test_a_non_monotone_series_is_not_validated(self) -> None:
        """test_wp13b_fix.py:505-512 feeds the engine an unsorted series with repeats."""
        ks = [*range(10), *range(4, 14), 20, 15, 15]
        bars = {A: [_bar(A, k, 100 + i) for i, k in enumerate(ks)]}
        spy = _Spy()
        run = await _run([A], bars, spy, warmup=2)
        assert run.engine._bar_count == len(ks) - 2
        assert [c[1] for c in spy.calls] == [bars[A][i].timestamp for i in range(2, len(ks))]

    async def test_off_grid_single_symbol_bars_run_unchanged(self) -> None:
        bars = {A: [_bar(A, k, 100 + k, minutes=17) for k in range(8)]}
        run = await _run([A], bars, _Spy(), warmup=1)
        assert run.engine._bar_count == 7


# ---------------------------------------------------------------------------
# T-12  seeded random gaps (QD S2: B 2 %, C 5 %, seed 20251)
# ---------------------------------------------------------------------------

_N = 300


def _random_walk(symbol: str, base: int, rng: random.Random, keep: set[int]) -> list[OHLCVBar]:
    price = Decimal(base)
    out: list[OHLCVBar] = []
    for k in range(_N):
        price = (price * (Decimal(1) + Decimal(rng.gauss(0.0, 0.01)))).quantize(Decimal("0.000001"))
        if k in keep:
            out.append(_bar(symbol, k, price))
    return out


def _gapped_universe(seed: int) -> dict[str, list[OHLCVBar]]:
    rng = random.Random(seed)  # noqa: S311 -- seeded PRNG for a deterministic fixture
    interior = list(range(1, _N - 1))  # first and last rows stay full
    drop_b = set(rng.sample(interior, round(0.02 * _N)))
    drop_c = set(rng.sample(interior, round(0.05 * _N)))
    return {
        A: _random_walk(A, 1000, rng, set(range(_N))),
        B: _random_walk(B, 2000, rng, set(range(_N)) - drop_b),
        C: _random_walk(C, 3000, rng, set(range(_N)) - drop_c),
    }


class TestT12RandomGaps:
    async def test_every_fill_mark_and_stamp_belongs_to_its_own_symbols_bar(self) -> None:
        bars = _gapped_universe(20251)
        assert len(bars[B]) < _N and len(bars[C]) < len(bars[B])
        run = await _run(
            [A, B, C],
            bars,
            _Spy(rng=random.Random(7)),  # noqa: S311 -- seeded, deterministic
            warmup=5,
            config=_BRACKET_AND_TRAILING,
        )
        union = sorted({b.timestamp for s in bars.values() for b in s})
        assert len(union) == _N and union[0] == T0
        own_ts = {s: {b.timestamp for b in series} for s, series in bars.items()}
        close_at = {(s, b.timestamp): b.close for s, series in bars.items() for b in series}

        assert run.fills, "the seeded strategy must trade"
        assert {f.symbol for f in run.fills} == {A, B, C}
        for f in run.fills:  # a fill uses its own symbol's bar: open == stamp - tf
            assert f.executed_at - H in own_ts[f.symbol], (f.symbol, f.executed_at)

        n_partial = 0
        for stamp, prices, _before in run.portfolio.marks:  # no mark from a bar later than the step
            for symbol, price in prices.items():
                assert close_at[(symbol, stamp - H)] == price
            n_partial += len(prices) < 3
        assert n_partial > 0, "the fixture must contain absent steps"
        assert len(run.portfolio.marks) == _N, "one mark per step"

        assert run.portfolio.pushes == [t + H for t in union]
        assert all(
            a < b for a, b in zip(run.portfolio.pushes, run.portfolio.pushes[1:], strict=False)
        )
        allowed = {union[0], *(t + H for t in union)}
        assert {ts for ts, _eq in run.portfolio.get_equity_curve()} <= allowed
        assert run.portfolio.get_equity_curve()[0][0] == union[0] == T0


# ---------------------------------------------------------------------------
# T-13  independence of the other symbols; no synthetic bar
# ---------------------------------------------------------------------------


class TestT13Independence:
    async def test_a_signal_stream_does_not_depend_on_a_gapped_companion(self) -> None:
        bars = _gapped_universe(20251)
        alone, together = _Decider(A), _Decider(A)
        await _run([A], {A: bars[A]}, alone, warmup=5)
        await _run([A, B, C], bars, together, warmup=5)
        assert len(alone.stream) == _N - 5
        assert together.stream == alone.stream

    async def test_every_history_element_is_an_input_bar_object(self) -> None:
        bars = _gapped_universe(20251)
        spy = _Spy()
        await _run([A, B, C], bars, spy, warmup=5)
        pool = {id(b) for series in bars.values() for b in series}
        assert spy.histories
        for history in spy.histories:
            assert all(id(b) in pool for b in history)
        # the gapped symbols' histories are shorter than the timeline, never padded
        b_lens = [len(h) for h in spy.histories if h[-1].symbol == B]
        assert max(b_lens) == len(bars[B]) < _N


# ---------------------------------------------------------------------------
# T-14  the WP2.2 clock lines stay textually unchanged (P1-CR-06)
# ---------------------------------------------------------------------------


def _run_backtest_fn() -> ast.AsyncFunctionDef:
    tree = ast.parse(_ENGINE_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StrategyEngine")
    return next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_backtest"
    )


class TestT14ClockLinesUnchanged:
    def test_close_i_is_read_from_the_bars_own_timestamps(self) -> None:
        fn = _run_backtest_fn()
        assigns = [
            ast.unparse(n)
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "close_i" for t in n.targets)
        ]
        expected = ast.unparse(
            ast.parse(
                "close_i = max((b.timestamp for b in current_bars.values())) "
                "+ timedelta(seconds=_TIMEFRAME_SECONDS[self._timeframe])"
            )
        )
        assert assigns == [expected]

    def test_the_reconstructed_step_time_is_never_read_by_the_engine(self) -> None:
        fn = _run_backtest_fn()
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert not attrs & {"ts", "ts_ms"}, attrs & {"ts", "ts_ms"}
        assert "present" in attrs

    def test_one_build_timeline_call_and_none_for_a_single_symbol_branch(self) -> None:
        fn = _run_backtest_fn()
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "build_timeline"
        ]
        assert len(calls) == 1
        ifs = [
            n
            for n in fn.body
            if isinstance(n, ast.If) and "len(self._symbols) == 1" in ast.unparse(n.test)
        ]
        assert len(ifs) == 1
        assert not [n for n in ast.walk(ifs[0]) if n in calls and n in ifs[0].body], (
            "the single-symbol branch must not call build_timeline"
        )
        assert calls[0] in list(ast.walk(ifs[0])), "the call sits in the multi-symbol branch"

    def test_build_timeline_comes_from_the_data_leaf(self) -> None:
        tree = ast.parse(_ENGINE_SRC.read_text(encoding="utf-8"))
        origins = [
            n.module
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and any(a.name == "build_timeline" for a in n.names)
        ]
        assert origins == ["data.bar_alignment"]
