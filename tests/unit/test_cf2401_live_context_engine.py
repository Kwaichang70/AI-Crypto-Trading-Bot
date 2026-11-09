"""CF-24-01 P4a -- the engine's live source: LiveSignals over LIVE_BOOK, the stage fence, consumer
registration (T-15..T-17, T-19 (P4a), T-20..T-25; SY-01-21a/d/h/j, ER2-01, ER3-05, AR4-01c).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from structlog.testing import capture_logs

import trading.clock as clock_mod
import trading.mtf_signals as mtf
from common.models import OHLCVBar
from common.types import RunMode, TimeFrame
from data.live_context import (
    LIVE_BOOK,
    LIVE_CONSUMERS,
    ConsumerRegistry,
    LiveContextBook,
    fgi_observations,
    snapshot_observation,
    t10y2y_observation,
)
from data.point_in_time import FIELD_MAP, PitSeries, to_ms
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit.test_sprint51_cycle1_sizing_contract import (
    _INITIAL_CASH,
    _LAST_PRICE,
    _make_real_paper_engine,
    _make_signal,
)
from trading.mtf_signals import LIVE_SIGNALS, PitSignals, SignalValues, live_served_fields
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading.strategy_engine import EngineState, StrategyEngine

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
MIN, HOUR, DAY = 60_000, 3_600_000, 86_400_000
FGI, T10, DOM, WHALE = (
    "alternative_me:fgi",
    "fred:T10Y2Y",
    "coingecko:btc_dominance",
    "whale:net_flow_1h",
)
SIX = frozenset(FIELD_MAP)
_BANNED_IN_SIGNALS = ("structlog", "getLogger", "logging", "print(", "exc_log_fields")  # G-17
# Transcribed from bd93c27 tests/unit/test_wp24_live_parity.py:122-139 (the frozen HEAD oracle's
# non-None key order; the oracle itself is deleted, SY-01-21d).
_HEAD_NON_NONE_ORDER = (
    "fear_greed_index", "btc_dominance", "market_cap_change_24h", "total_volume_change_24h",
    "fed_funds_rate", "yield_curve_spread", "whale_net_flow", "fear_greed_index_7d_ago",
    "btc_dominance_7d_ago",
)  # fmt: skip


class Wall:
    """The wall clock, pinned through the WP2.2 seam ``trading.clock._wall_now``."""

    def __init__(self, at: datetime) -> None:
        self.at = at

    def ms(self) -> int:
        return to_ms(self.at)


@pytest.fixture
def wall(monkeypatch: pytest.MonkeyPatch) -> Wall:
    w = Wall(NOW)
    monkeypatch.setattr(clock_mod, "_wall_now", lambda: w.at)
    return w


def fresh_book(now_ms: int) -> None:
    """All six fields servable at ``now_ms`` (and 7d back)."""
    day = now_ms - now_ms % DAY
    LIVE_BOOK.publish(
        FGI, fgi_observations([(day - k * DAY, 30 + k) for k in range(1, 10)], now_ms), now_ms
    )
    LIVE_BOOK.publish(
        T10, [t10y2y_observation(day - k * DAY, 0.4, now_ms) for k in range(1, 8)], now_ms
    )
    for sid, raw in ((DOM, 52.5), (WHALE, 1234.5)):
        for fetch in (now_ms - 7 * DAY - 30 * MIN, now_ms - 10 * MIN):
            LIVE_BOOK.publish(sid, [snapshot_observation(sid, raw, fetch)], fetch)


def random_book(rng: random.Random, now_ms: int) -> None:
    """A random but valid book whose high-water mark is below ``now_ms``."""
    LIVE_BOOK.reset()
    day = now_ms - now_ms % DAY
    days = rng.sample(range(1, 12), rng.randint(0, 6))
    LIVE_BOOK.publish(
        FGI, fgi_observations([(day - k * DAY, rng.randint(0, 100)) for k in days], now_ms), now_ms
    )
    days = rng.sample(range(1, 12), rng.randint(0, 6))
    LIVE_BOOK.publish(
        T10, [t10y2y_observation(day - k * DAY, rng.uniform(-3, 3), now_ms) for k in days], now_ms
    )
    for sid in (DOM, WHALE):
        for fetch in sorted(now_ms - rng.randint(1, 8 * DAY) for _ in range(rng.randint(0, 5))):
            LIVE_BOOK.publish(sid, [snapshot_observation(sid, rng.uniform(0, 99), fetch)], fetch)


def make_engine(
    mode: RunMode = RunMode.PAPER, *, fields: Any = SIX, source: Any = None
) -> StrategyEngine:
    strategy = MagicMock()
    strategy.strategy_id = "s"
    strategy.min_bars_required = 20
    strategy.htf_timeframes = None
    strategy.mtf_signal_fields.return_value = fields if fields is None else frozenset(fields)
    return StrategyEngine(
        strategies=[strategy], execution_engine=AsyncMock(), risk_manager=MagicMock(),
        market_data=AsyncMock(), portfolio=MagicMock(), symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR, run_mode=mode, mtf_signals=source,
    )  # fmt: skip


def shadow_events(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == "engine.mtf_context_shadow"]


# --- T-15: the reader ---


class TestT15Reader:
    def test_live_signals_equals_the_pit_join_over_random_books(self, wall: Wall) -> None:
        rng = random.Random(24)  # noqa: S311  # seed control, not security
        served = 0
        for _ in range(240):
            random_book(rng, wall.ms())
            expected = PitSignals(LIVE_BOOK.view(wall.ms()), mtf._LIVE_FIELDS).values_at(NOW)
            for stamp in (None, datetime(2001, 1, 1, tzinfo=UTC), NOW, NOW + timedelta(days=400)):
                assert LIVE_SIGNALS.values_at(stamp) == expected  # decision_at is ignored
            served += expected != SignalValues()
        assert served > 150  # the draws really serve values

    def test_a_raising_view_or_join_is_all_none_and_nothing_escapes(
        self, wall: Wall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert mtf._LIVE_READ_FAILED == SignalValues()  # T-15: the failure value is all None
        fresh_book(wall.ms())
        assert LIVE_SIGNALS.values_at() != SignalValues()
        engine = make_engine(source=None)
        for target, name in ((LiveContextBook, "view"), (mtf.PitSignals, "values_at")):
            with monkeypatch.context() as m:
                m.setattr(target, name, MagicMock(side_effect=RuntimeError("boom")))
                assert LIVE_SIGNALS.values_at() is mtf._LIVE_READ_FAILED
                assert engine._build_mtf_context(NOW) is None

    def test_the_reader_is_synchronous_and_the_never_served_fields_are_none(
        self, wall: Wall
    ) -> None:
        assert not inspect.iscoroutinefunction(LIVE_SIGNALS.values_at)  # T-23
        fresh_book(wall.ms())
        values = LIVE_SIGNALS.values_at()
        for name in (
            "market_cap_change_24h",
            "total_volume_change_24h",
            "fed_funds_rate",
        ):  # CF-2401-30
            assert getattr(values, name) is None
            assert getattr(values.restricted(mtf.ALL_MTF_SIGNAL_FIELDS), name) is None
        assert {n for n in SIX if getattr(values, n) is None} == set()

    def test_the_reader_passes_wall_clock_never_a_bar_or_decision_time(self, wall: Wall) -> None:
        fresh_book(wall.ms())  # CF-2401-66
        LIVE_SIGNALS.values_at(NOW + timedelta(days=3650))
        assert LIVE_BOOK.heartbeat(wall.ms()) is False and LIVE_BOOK.regressed_reads == 0

    def test_a_base_exception_from_the_book_propagates(self, wall: Wall) -> None:
        with patch.object(LiveContextBook, "view", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                LIVE_SIGNALS.values_at()


# --- T-25 (reader clause): ER3-05 as amended by AR4-01c ---


class TestT25Reader:
    def test_a_backward_step_below_a_poller_raised_mark_is_all_none(self, wall: Wall) -> None:
        fresh_book(wall.ms())
        LIVE_BOOK.heartbeat(wall.ms())
        wall.at = NOW - timedelta(minutes=6)
        assert LIVE_SIGNALS.values_at() == SignalValues() and LIVE_BOOK.regressed_reads == 1
        wall.at = NOW - timedelta(minutes=5)  # the tolerance is inclusive
        assert LIVE_SIGNALS.values_at() != SignalValues()

    def test_a_mark_raised_only_by_reader_views_still_guards_a_dead_poller(
        self, wall: Wall
    ) -> None:
        fresh_book(wall.ms())
        wall.at = NOW + timedelta(hours=1)
        LIVE_SIGNALS.values_at()  # no heartbeat ever
        wall.at = NOW + timedelta(minutes=30)
        assert LIVE_SIGNALS.values_at() == SignalValues()

    def test_a_forward_step_ages_recorded_fields_out(self, wall: Wall) -> None:
        fresh_book(wall.ms())
        assert LIVE_SIGNALS.values_at().whale_net_flow == 1234.5
        wall.at = NOW + timedelta(hours=3)  # the whale cap is 2 h
        values = LIVE_SIGNALS.values_at()
        assert values.whale_net_flow is None and values.fear_greed_index is not None

    def test_a_held_estimated_lag_row_is_revealed_by_a_forward_step(self, wall: Wall) -> None:
        day = wall.ms() - wall.ms() % DAY  # ER3-04: documented, up to the step size early
        LIVE_BOOK.publish(
            FGI, fgi_observations([(day, 61)], wall.ms()), wall.ms()
        )  # visible at day+24h
        assert LIVE_SIGNALS.values_at().fear_greed_index is None
        wall.at = NOW + timedelta(hours=13)
        assert LIVE_SIGNALS.values_at().fear_greed_index == 61

    def test_a_corrected_forward_step_is_neutral_and_no_step_row_is_served(
        self, wall: Wall
    ) -> None:
        fetch = wall.ms() + HOUR  # stamped in the stepped frame
        LIVE_BOOK.publish(WHALE, [snapshot_observation(WHALE, 7.0, fetch)], fetch)
        wall.at = NOW + timedelta(hours=1)
        assert LIVE_SIGNALS.values_at().whale_net_flow == 7.0
        wall.at = NOW + timedelta(minutes=2)  # corrected 2 min after the step began (< S - 5 min)
        assert LIVE_SIGNALS.values_at() == SignalValues()
        wall.at = NOW + timedelta(minutes=56)  # now + 5 min >= the mark: the guard lifts
        assert LIVE_SIGNALS.values_at().whale_net_flow is None  # the step row was dropped

    def test_r3_res_2_a_long_offset_keeps_its_rows_which_read_an_hour_too_young(
        self, wall: Wall
    ) -> None:
        base = to_ms(datetime(2026, 3, 8, tzinfo=UTC))  # an hour boundary
        steps = [base + k * HOUR + 2 * MIN + HOUR for k in range(48)]  # +60 min offset held 48 h
        for k, fetch in enumerate(steps):
            LIVE_BOOK.heartbeat(fetch)
            assert (
                LIVE_BOOK.publish(WHALE, [snapshot_observation(WHALE, float(k), fetch)], fetch) == 1
            )
        corrected = steps[-1] - HOUR + MIN  # NTP fixes it 1 min after the last step fetch
        assert LIVE_BOOK.heartbeat(corrected) is True  # drops only the last (stamped > now + 5 min)
        extra = corrected + MIN  # a real sighting in the correction's bucket
        assert LIVE_BOOK.publish(WHALE, [snapshot_observation(WHALE, 99.0, extra)], extra) == 0
        lift = steps[-1] - 5 * MIN  # the guard lifts when now + 5 min >= the mark
        assert LIVE_BOOK.freshness(lift).regressed is False
        wall.at = datetime.fromtimestamp(lift / 1000, UTC)
        assert LIVE_SIGNALS.values_at().whale_net_flow == 46.0  # the bucket's step row
        real_age = lift - (steps[46] - HOUR)
        assert LIVE_BOOK.freshness(lift).series[WHALE].age_ms == real_age - HOUR


# --- T-16: field selection ---


@pytest.mark.parametrize(
    ("declared", "served"),
    [
        ([frozenset()], frozenset()),
        ([frozenset({"fear_greed_index_7d_ago"})], {"fear_greed_index_7d_ago", "fear_greed_index"}),
        ([frozenset({"btc_dominance_7d_ago", "whale_net_flow"})],
         {"btc_dominance_7d_ago", "btc_dominance", "whale_net_flow"}),
        ([frozenset({"fed_funds_rate", "market_cap_change_24h"})], frozenset()),
        ([frozenset(), frozenset({"yield_curve_spread"})], {"yield_curve_spread"}),
        ([object()], SIX),  # a non-frozenset declares everything (SY-24-06)
        ([MagicMock()], SIX),
        ([], frozenset()),
    ],
)  # fmt: skip
def test_t16_live_served_fields(declared: list[object], served: frozenset[str]) -> None:
    assert live_served_fields(declared) == frozenset(served)


def test_t16_the_engine_selects_per_run_and_only_in_paper_and_live() -> None:
    assert (
        make_engine(fields=MomentumBreakoutStrategy.consumes_mtf_signals)._live_fields
        == frozenset()
    )
    rsi = make_engine(fields=RSIMeanReversionStrategy.consumes_mtf_signals)._live_fields
    assert rsi == RSIMeanReversionStrategy.consumes_mtf_signals & SIX and len(rsi) == 4
    assert make_engine(fields=None)._live_fields == SIX  # MagicMock return value: not a frozenset
    backtest = make_engine(RunMode.BACKTEST)
    assert backtest._live_fields is None and backtest._live_stage is None


# --- T-17 / T-20: the stale baseline and the sizing bound ---


def _path(kind: str) -> list[OHLCVBar]:
    sign = (
        -2 if kind == "down" else 2
    )  # 50 flat-ish bars, then a straight run: oversold / overbought
    closes = [100 + (i % 3) for i in range(50)] + [100 + sign * k for k in range(1, 25)]
    return [
        OHLCVBar(
            symbol="BTC/USDT", timeframe=TimeFrame.ONE_HOUR, timestamp=NOW + timedelta(hours=i),
            open=Decimal(c), high=Decimal(c), low=Decimal(c), close=Decimal(c), volume=Decimal(1),
        )
        for i, c in enumerate(closes)
    ]  # fmt: skip


def _signals(cls: Any, ctx: Any, kind: str) -> list[tuple[Any, ...]]:
    strategy, bars = cls("s", {}), _path(kind)  # a fresh instance: no state carries over
    strategy.on_start("t17")  # dca needs it, rsi ignores it
    return [
        (s.direction, s.target_position, s.confidence, s.metadata)
        for n in range(45, len(bars) + 1)
        for s in strategy.on_bar(bars[:n], mtf_context=ctx)
    ]


# Each value LOWERS the conviction of its own direction and lies strictly beyond the boost
# thresholds (both strategies test whale net flow with a strict ``< -5e6`` / ``> 5e6``).
_T17_VALUES = {  # kind -> (fear_greed, dominance, t10y2y, whale net flow)
    "down": (90, 70.0, -2.0, 6e6),
    "up": (10, 40.0, 1.5, -6e6),
}
_T17_STALE = (3 * DAY, 12 * DAY, 7 * HOUR, 3 * HOUR)  # held in the book, past every cap
_T17_FRESH = (1 * DAY, 5 * DAY, 10 * MIN, 10 * MIN)  # the power control: same values, in the cap


def _t17_publish(now: int, kind: str, ages: tuple[int, int, int, int]) -> list[int]:
    """Publish the vector with the given ages (fgi, t10y2y, dominance, whale); the verdicts."""
    day, (fgi, dom, t10, whale) = now - now % DAY, _T17_VALUES[kind]
    fgi_age, t10_age, dom_age, whale_age = ages
    return [
        LIVE_BOOK.publish(FGI, fgi_observations([(day - fgi_age, fgi)], now), now),
        LIVE_BOOK.publish(T10, [t10y2y_observation(day - t10_age, t10, now)], now),
        *(
            LIVE_BOOK.publish(sid, [snapshot_observation(sid, raw, now - age)], now - age)
            for sid, raw, age in ((DOM, dom, dom_age), (WHALE, whale, whale_age))
        ),
    ]


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
@pytest.mark.parametrize("cls", [RSIMeanReversionStrategy, DCARSIHybridStrategy])
@pytest.mark.parametrize("kind", ["down", "up"])
def test_t17_stale_context_gives_the_unavailable_backtest_signals(
    wall: Wall, monkeypatch: pytest.MonkeyPatch, mode: RunMode, cls: Any, kind: str
) -> None:
    """Held but stale rows are not applied: signals equal the UNAVAILABLE backtest's, and the
    same values, fresh, do change the confidence (so the equality is not vacuous)."""
    monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")  # before construction: test staleness
    now = wall.ms()
    assert _t17_publish(now, kind, _T17_STALE) == [1, 1, 1, 1]  # held, not trimmed at publish
    live = make_engine(mode, fields=cls.consumes_mtf_signals)
    reference = make_engine(RunMode.BACKTEST)  # the UNAVAILABLE context
    for htf in (None, {}):
        live._htf_bars = reference._htf_bars = htf
        assert live._build_mtf_context(NOW) == reference._build_mtf_context(NOW, decision_at=NOW)
    live._htf_bars = reference._htf_bars = None
    stale = _signals(cls, live._build_mtf_context(NOW), kind)
    ref_ctx = reference._build_mtf_context(NOW, decision_at=NOW)
    assert stale and stale == _signals(cls, ref_ctx, kind)
    LIVE_BOOK.reset()  # power control: the same values, inside every cap
    assert _t17_publish(now, kind, _T17_FRESH) == [1, 1, 1, 1]
    fresh_ctx = make_engine(mode, fields=cls.consumes_mtf_signals)._build_mtf_context(NOW)
    assert fresh_ctx is not None
    fresh = _signals(cls, fresh_ctx, kind)
    assert any(f[2] < s[2] for f, s in zip(fresh, stale, strict=True))  # confidence, not metadata


def test_t20_context_never_sizes_above_the_bound() -> None:
    """1000 seeded trials: BUY <= min(target, 0.15 E, 10 000) / P x m_CB; SELL <= held."""
    engine, _ = _make_real_paper_engine()
    rng = random.Random(2401)  # noqa: S311  # seed control, not security
    for _ in range(1000):
        conf, target = rng.uniform(-0.5, 1.5), Decimal(rng.randint(1, 50_000))
        equity, mult = (
            Decimal(rng.randint(1_000, 200_000)),
            Decimal(str(rng.choice([1, 0.5, 0.25]))),
        )
        signal = _make_signal(target_position=target, confidence=min(1.0, max(0.0, conf)))
        signal = signal.model_copy(update={"entry_size_multiplier": mult})
        qty = engine._resolve_order_quantity(
            signal=signal,
            side=mtf_side("BUY"),
            last_price=_LAST_PRICE,
            equity=equity,
            held_quantity=None,
        )
        assert (
            qty * _LAST_PRICE <= min(target, Decimal("0.15") * equity, Decimal(10_000)) * mult + 1
        )
        held = Decimal(rng.randint(0, 5))
        sell = engine._resolve_order_quantity(
            signal=_make_signal(direction=mtf_side("SELL", direction=True), target_position=target,
                                confidence=min(1.0, max(0.0, conf))),
            side=mtf_side("SELL"), last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=held,
        )  # fmt: skip
        assert sell <= held


def mtf_side(name: str, *, direction: bool = False) -> Any:
    from common.types import OrderSide, SignalDirection

    return (SignalDirection if direction else OrderSide)[name]


# --- T-21: the stage, the shadow event and the fence ---


class TestT21Stage:
    def test_the_stage_constant_is_neutral_shadow(self) -> None:
        assert mtf.LIVE_CONTEXT_STAGE == "neutral_shadow"

    def test_consumers_get_the_neutral_context_and_the_shadow_carries_fresh_values(
        self, wall: Wall
    ) -> None:
        engine = make_engine()
        fresh_book(wall.ms())
        with capture_logs() as logs:
            assert engine._build_mtf_context(NOW) is None
        (event,) = shadow_events(logs)
        bound = set(structlog.get_context(engine._log))  # component, run_mode, timeframe, ...
        assert set(event) - bound == {"event", "log_level", "values", "states"}
        assert set(event["values"]) == set(event["states"]) == set(engine._live_fields)
        assert event["values"] == {
            "btc_dominance": 52.5, "btc_dominance_7d_ago": 52.5, "fear_greed_index": 31,
            "fear_greed_index_7d_ago": 38, "whale_net_flow": 1234.5, "yield_curve_spread": 0.4,
        }  # fmt: skip
        assert set(event["states"].values()) == {"served"} and "age" not in str(event)  # ER2-01
        LIVE_BOOK.reset()
        with capture_logs() as logs:
            assert engine._build_mtf_context(NOW) is None
        assert set(shadow_events(logs)[0]["states"].values()) == {"none"}

    def test_a_non_consumer_emits_nothing_and_reads_no_field(self, wall: Wall) -> None:
        engine = make_engine(fields=MomentumBreakoutStrategy.consumes_mtf_signals)
        fresh_book(wall.ms())
        with capture_logs() as logs:
            assert engine._build_mtf_context(NOW) is None
        assert shadow_events(logs) == [] and engine._live_fields == frozenset()

    def test_stage_fresh_serves_only_the_consumed_fields_and_no_shadow(
        self, wall: Wall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")  # patched BEFORE construction
        engine = make_engine(fields={"fear_greed_index_7d_ago", "whale_net_flow"})
        fresh_book(wall.ms())
        with capture_logs() as logs:
            ctx = engine._build_mtf_context(NOW)
        assert ctx is not None and shadow_events(logs) == []
        assert (ctx.fear_greed_index, ctx.fear_greed_index_7d_ago, ctx.whale_net_flow) == (
            31,
            38,
            1234.5,
        )
        assert ctx.btc_dominance is None and ctx.yield_curve_spread is None

    def test_stage_fresh_a_raising_book_gives_no_context(
        self, wall: Wall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")  # patched BEFORE construction
        fresh_book(wall.ms())
        engine = make_engine()
        assert engine._build_mtf_context(NOW) is not None  # the control: the book serves
        for target, name in ((LiveContextBook, "view"), (mtf.PitSignals, "values_at")):
            with monkeypatch.context() as m, capture_logs() as logs:
                m.setattr(target, name, MagicMock(side_effect=RuntimeError("boom")))
                assert engine._build_mtf_context(NOW) is None  # the failure value serves nothing
            assert shadow_events(logs) == []

    def test_the_stage_is_read_once_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")
            engine = make_engine()
        assert engine._live_stage == "fresh"
        engine = make_engine()
        monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")  # after construction: no effect
        assert engine._live_stage == "neutral_shadow"

    def test_a_backtest_engine_is_never_staged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        day = to_ms(NOW) - to_ms(NOW) % DAY
        series = PitSeries(FGI, fgi_observations([(day - DAY, 44)], to_ms(NOW)))
        results = []
        for stage in ("neutral_shadow", "fresh"):
            with monkeypatch.context() as m:
                m.setattr(mtf, "LIVE_CONTEXT_STAGE", stage)
                engine = make_engine(RunMode.BACKTEST, source=PitSignals({FGI: series}, SIX))
            results.append(engine._build_mtf_context(NOW, decision_at=NOW))
        assert (
            results[0] == results[1]
            and results[0] is not None
            and results[0].fear_greed_index == 44
        )


# --- CF-2401-57 / RK-P3-10: consumer registration ---


async def _run_loop(
    mode: RunMode,
    fields: Any,
    *,
    poll: Callable[[StrategyEngine], Any] | None = None,
    stop: Callable[[], Awaitable[None]] | None = None,
) -> tuple[StrategyEngine, list[dict[str, int]], list[str]]:
    engine = make_engine(mode, fields=fields)
    engine._state = EngineState.RUNNING
    seen: list[dict[str, int]] = []
    calls: list[str] = []

    async def _noop() -> None:
        return None

    async def _poll() -> None:
        calls.append("poll")
        seen.append(dict(LIVE_CONSUMERS.counts()))
        if poll is not None:
            await poll(engine)
        engine._stop_event.set()

    async def _stop_task() -> None:
        calls.append("stop_intrabar")

    engine._wall_clock_tripwire = lambda: None  # type: ignore[method-assign]
    engine._warmup_bar_windows = _noop  # type: ignore[method-assign]
    engine._start_intrabar_task = lambda: calls.append("start_intrabar")  # type: ignore[method-assign]
    engine._stop_intrabar_task = stop or _stop_task  # type: ignore[method-assign]
    engine._poll_and_process = _poll  # type: ignore[method-assign]
    try:
        await engine.run_live_loop()
    finally:
        calls.append("done")
    return engine, seen, calls


class TestRegistration:
    @pytest.mark.parametrize(("mode", "key"), [(RunMode.PAPER, "paper"), (RunMode.LIVE, "live")])
    async def test_a_consumer_registers_before_the_first_bar_and_releases_in_finally(
        self, mode: RunMode, key: str
    ) -> None:
        with capture_logs() as logs:
            engine, seen, calls = await _run_loop(mode, SIX)
        (marker,) = [e for e in logs if e["event"] == "engine.live_context"]
        assert (marker["version"], marker["stage"]) == ("cf2401-1", "neutral_shadow")
        bound = set(structlog.get_context(engine._log))
        assert set(marker) - bound == {"event", "log_level", "version", "stage"}
        assert seen == [{"paper": int(key == "paper"), "live": int(key == "live")}]
        assert LIVE_CONSUMERS.counts() == {"paper": 0, "live": 0}
        assert calls == ["start_intrabar", "poll", "stop_intrabar", "done"]

    async def test_a_non_consumer_never_registers(self) -> None:
        with capture_logs() as logs:
            _, seen, _ = await _run_loop(RunMode.LIVE, frozenset())
        assert seen == [{"paper": 0, "live": 0}]
        assert [e for e in logs if e["event"] == "engine.live_context"] == []  # no marker either

    async def test_the_registration_is_released_when_the_loop_dies(self) -> None:
        async def _die(_: StrategyEngine) -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _run_loop(RunMode.LIVE, SIX, poll=_die)
        assert LIVE_CONSUMERS.counts() == {"paper": 0, "live": 0}

    @pytest.mark.parametrize("exc", [RuntimeError, asyncio.CancelledError])
    async def test_the_registration_is_released_when_the_intrabar_stop_raises(
        self, exc: type[BaseException]
    ) -> None:
        async def _stop() -> None:
            raise exc("stop")

        with pytest.raises(exc):
            await _run_loop(RunMode.LIVE, SIX, stop=_stop)
        assert LIVE_CONSUMERS.counts() == {"paper": 0, "live": 0}  # CF-2401-57 / RK-P3-10

    async def test_a_raising_started_skips_neither_the_protective_paths_nor_the_bar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(self: ConsumerRegistry, mode: str) -> int:
            raise ValueError("SECRET-TEXT")

        monkeypatch.setattr(ConsumerRegistry, "started", _boom)
        with capture_logs() as logs:
            _, _, calls = await _run_loop(RunMode.LIVE, SIX)
        assert calls == ["start_intrabar", "poll", "stop_intrabar", "done"]
        (err,) = [e for e in logs if e["event"] == "engine.live_consumer_register_failed"]
        assert err["error_type"] == "ValueError" and "SECRET" not in str(err)


# --- T-22: the pin ledger (SY-01-21d) ---

_PARITY = Path(__file__).with_name("test_wp24_live_parity.py")


def test_t22_the_retired_pins_are_gone_and_the_adapted_ones_remain() -> None:
    tree = ast.parse(_PARITY.read_text())
    names = {getattr(n, "name", None) for n in ast.walk(tree)}
    names |= {
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    retired = {
        "TestT15FrozenText", "TestT15SourceMoveIsVerbatim", "_MOVED_BLOCK_SHA256", "_MOVED_MARKER",
        "_MOVED_LINES", "_HEAD_ORACLE_SHA256", "_head_signal_block", "_head_context", "_SCENARIOS",
        "_installed", "_fgi", "_cg", "_fred", "_whale",
        "test_engine_context_equals_the_frozen_head_oracle",
        "test_with_an_htf_dict_present_the_signal_fields_are_the_oracles",
        "test_signal_reads_are_the_same_statements_in_the_same_order",
        "test_the_moved_block_references_only_the_four_lazy_imports",
        "test_the_moved_block_is_byte_identical_to_head", "test_the_head_oracle_is_frozen",
    }  # fmt: skip
    assert not retired & names
    assert {"test_decision_at_is_ignored_on_the_live_path",
            "test_a_base_exception_is_not_swallowed_by_either_implementation"} <= names  # fmt: skip


def test_the_non_none_rule_is_the_oracles_in_the_head_order() -> None:
    node = next(n for n in ast.walk(ast.parse(inspect.getsource(SignalValues)))
                if isinstance(n, ast.FunctionDef) and n.name == "to_context_kwargs")  # fmt: skip
    keys = [
        str(s.body[0].targets[0].slice.value)  # type: ignore[attr-defined]
        for s in node.body
        if isinstance(s, ast.If)
    ]
    assert tuple(keys) == _HEAD_NON_NONE_ORDER


def test_the_moved_code_logs_nothing_new(wall: Wall, caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level("DEBUG"),
        patch.object(LiveContextBook, "view", side_effect=RuntimeError("x")),
    ):
        assert make_engine(RunMode.LIVE)._build_mtf_context(NOW) is None
    assert caplog.records == []
    src = inspect.getsource(mtf)  # G-17: comments and docstrings count
    assert [b for b in _BANNED_IN_SIGNALS if b in src] == []


def test_t22_the_reexported_registry_is_the_book_module_singleton() -> None:
    import data.live_context as lc

    assert mtf.LIVE_CONSUMERS is lc.LIVE_CONSUMERS and "LIVE_CONSUMERS" in mtf.__all__


def test_t23_a_backtest_still_refuses_the_live_source() -> None:
    with pytest.raises(ValueError, match="backtest signal source"):
        make_engine(RunMode.BACKTEST, source=LIVE_SIGNALS)


def test_t23_a_poller_mutating_the_book_cannot_reach_a_backtest(wall: Wall) -> None:
    day = wall.ms() - wall.ms() % DAY
    series = PitSeries(FGI, fgi_observations([(day - DAY, 44)], wall.ms()))
    engine = make_engine(RunMode.BACKTEST, source=PitSignals({FGI: series}, SIX))
    before = engine._build_mtf_context(NOW, decision_at=NOW)
    with patch.multiple("data.sentiment", get_global_client=MagicMock(side_effect=AssertionError)):
        fresh_book(wall.ms())  # a fake poller writes the book mid-run
        LIVE_BOOK.heartbeat(wall.ms() + 30 * DAY)
        assert engine._build_mtf_context(NOW, decision_at=NOW) == before
    assert "context_poller" not in inspect.getsource(mtf)  # neither side imports the poller
