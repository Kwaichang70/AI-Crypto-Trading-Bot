"""
tests/unit/test_wp15a_engine_wiring.py
----------------------------------------
WP1.5a -- ``StrategyEngine`` wiring of the risk clock and the risk-state source.

    WP15-T-15  run_backtest pushes the bar CLOSE time once per bar (warmup
               included) BEFORE any mark; every TimeFrame has a duration; the
               risk-state source is attached in every mode; test doubles that
               lack the new hooks do not crash the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.conftest import make_bars
from trading.engines.paper import PaperExecutionEngine
from trading.models import Signal
from trading.portfolio import PortfolioAccounting
from trading.risk_manager import DefaultRiskManager
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import _TIMEFRAME_SECONDS, EngineState, StrategyEngine

_SYMBOL = "BTC/USDT"


class _NoopStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="wp15_noop_wiring", description="test-only")
    requires_exit_manager = False

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        return []


def _engine(
    *,
    portfolio: PortfolioAccounting,
    execution: object | None = None,
    mode: RunMode = RunMode.BACKTEST,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
) -> StrategyEngine:
    rm = DefaultRiskManager(run_id="wp15-wiring")
    return StrategyEngine(
        strategies=[_NoopStrategy("wp15-noop")],
        execution_engine=execution
        if execution is not None
        else PaperExecutionEngine("wp15-wiring", rm),  # type: ignore[arg-type]
        risk_manager=rm,
        market_data=MagicMock(),
        portfolio=portfolio,
        symbols=[_SYMBOL],
        timeframe=timeframe,
        run_mode=mode,
        config={"warmup_bars": 3},
    )


class _RecordingPortfolio(PortfolioAccounting):
    def __init__(self, events: list[tuple[str, object]]) -> None:
        super().__init__("wp15-wiring", Decimal("10000"))
        self.events = events

    def advance_clock(self, at: datetime) -> None:
        self.events.append(("advance_clock", at))
        super().advance_clock(at)

    def update_market_prices(self, prices: dict[str, Decimal]) -> None:
        self.events.append(("marks", None))
        super().update_market_prices(prices)


async def test_wp15_t15_run_backtest_pushes_bar_close_before_marks_every_bar() -> None:
    events: list[tuple[str, object]] = []
    portfolio = _RecordingPortfolio(events)
    engine = _engine(portfolio=portfolio)
    original = engine._update_engine_prices

    def spy(current_bars: dict[str, OHLCVBar]) -> None:
        events.append(("engine_prices", None))
        original(current_bars)

    engine._update_engine_prices = spy  # type: ignore[method-assign]

    bars = make_bars(8, seed=3, symbol=_SYMBOL)
    await engine.start("wp15-wiring")
    await engine.run_backtest({_SYMBOL: bars})

    pushes = [e for e in events if e[0] == "advance_clock"]
    assert len(pushes) == len(bars), "one push per bar, warmup included"
    for (_, at), bar in zip(pushes, bars, strict=True):
        assert at == bar.timestamp + timedelta(hours=1), "bar CLOSE time"

    # order: for every bar the push comes before the engine-price update and
    # before the first mark of that bar
    seq = [name for name, _ in events]
    i = 0
    while "advance_clock" in seq[i:]:
        i = seq.index("advance_clock", i)
        window = seq[i + 1 : i + 3]
        assert window[0] == "engine_prices", seq[i : i + 4]
        assert "marks" in window
        i += 1
    assert portfolio.risk_snapshot().as_of == bars[-1].timestamp + timedelta(hours=1)


async def test_wp15_t15_clock_value_is_the_max_timestamp_across_symbols() -> None:
    events: list[tuple[str, object]] = []
    portfolio = _RecordingPortfolio(events)
    rm = DefaultRiskManager(run_id="wp15-wiring")
    engine = StrategyEngine(
        strategies=[_NoopStrategy("wp15-noop")],
        execution_engine=PaperExecutionEngine("wp15-wiring", rm),
        risk_manager=rm,
        market_data=MagicMock(),
        portfolio=portfolio,
        symbols=[_SYMBOL, "ETH/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
        config={"warmup_bars": 1},
    )
    btc = make_bars(3, seed=1, symbol=_SYMBOL)
    eth = [
        bar.model_copy(update={"timestamp": bar.timestamp + timedelta(minutes=30)})
        for bar in make_bars(3, seed=2, symbol="ETH/USDT")
    ]
    await engine.start("wp15-wiring")
    await engine.run_backtest({_SYMBOL: btc, "ETH/USDT": eth})
    pushes = [at for name, at in events if name == "advance_clock"]
    assert pushes == [e.timestamp + timedelta(hours=1) for e in eth]


def test_wp15_t15_every_timeframe_has_a_duration() -> None:
    assert set(TimeFrame) <= set(_TIMEFRAME_SECONDS)


@pytest.mark.parametrize("mode", [RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE])
def test_wp15_t15_risk_state_source_is_attached_in_every_mode(mode: RunMode) -> None:
    portfolio = PortfolioAccounting("wp15-wiring", Decimal("10000"))
    execution = MagicMock()
    _engine(portfolio=portfolio, execution=execution, mode=mode)
    execution.attach_risk_state_source.assert_called_once_with(portfolio)


def test_wp15_t15_real_paper_engine_holds_the_portfolio_as_risk_source() -> None:
    portfolio = PortfolioAccounting("wp15-wiring", Decimal("10000"))
    rm = DefaultRiskManager(run_id="wp15-wiring")
    execution = PaperExecutionEngine("wp15-wiring", rm)
    assert execution._risk_state_source is None
    _engine(portfolio=portfolio, execution=execution, mode=RunMode.PAPER)
    assert execution._risk_state_source is portfolio
    execution.attach_risk_state_source(portfolio)  # idempotent
    assert execution._risk_state_source is portfolio


def test_wp15_t15_attach_risk_state_source_is_idempotent_and_logs_once() -> None:
    p = PortfolioAccounting("r", Decimal("1000"))
    engine = PaperExecutionEngine("r", DefaultRiskManager(run_id="r"), initial_cash=Decimal("1000"))
    with capture_logs() as logs:
        engine.attach_risk_state_source(p)
        engine.attach_risk_state_source(p)
    assert engine._risk_state_source is p
    assert (
        len([e for e in logs if e["event"] == "execution_engine.risk_state_source_attached"]) == 1
    )


class _BareExecution:
    """An execution-engine double with NO attach_* hooks at all."""

    def set_last_price(self, symbol: str, price: Decimal) -> None:
        return None

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    async def process_signal(self, signal: Signal) -> list[object]:
        return []


async def test_wp15_t15_doubles_without_the_new_hooks_do_not_crash() -> None:
    class _NoClockPortfolio(PortfolioAccounting):
        advance_clock = None  # type: ignore[assignment]

    rm = DefaultRiskManager(run_id="wp15-wiring")
    bare_execution = _BareExecution()
    portfolio = _NoClockPortfolio("wp15-wiring", Decimal("10000"))
    engine = StrategyEngine(
        strategies=[_NoopStrategy("wp15-noop")],
        execution_engine=bare_execution,  # type: ignore[arg-type]
        risk_manager=rm,
        market_data=MagicMock(),
        portfolio=portfolio,
        symbols=[_SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
        config={"warmup_bars": 3},
    )
    await engine.start("wp15-wiring")
    summary = await engine.run_backtest({_SYMBOL: make_bars(6, seed=5, symbol=_SYMBOL)})
    assert summary["current_equity"] == "10000"
    assert engine._state in (EngineState.RUNNING, EngineState.STOPPED)
    assert datetime(2024, 1, 1, tzinfo=UTC) < portfolio._now()  # no sim clock was pushed
