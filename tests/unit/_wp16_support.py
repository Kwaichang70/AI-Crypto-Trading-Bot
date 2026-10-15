"""
tests/unit/_wp16_support.py
----------------------------
Shared helpers for the WP1.6 unit tests (circuit-breaker wiring).

Not collected by pytest (leading underscore).  Everything is deterministic:
hand-built ``RiskSnapshot`` values, injected clocks, no wall-clock reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import RunMode, SignalDirection, TimeFrame
from tests.conftest import make_bar
from trading.engines.paper import PaperExecutionEngine
from trading.models import Signal
from trading.portfolio import DayStartSource, PortfolioAccounting, RiskSnapshot
from trading.risk import RiskParameters, live_risk_parameters
from trading.risk_manager import DefaultRiskManager
from trading.safety import (
    DEESCALATION_EVALUATIONS,
    CircuitBreaker,
    CircuitBreakerDecision,
    build_run_circuit_breaker,
)
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import StrategyEngine

D1 = date(2026, 3, 10)
D2 = date(2026, 3, 11)
D3 = date(2026, 3, 12)
T0 = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)

#: A large peak so a drawdown ``dd`` maps to an exact ``Decimal`` NAV and the
#: snapshot's ``drawdown_pct`` is *computed* exactly as the WP1.5 gate does.
PEAK = Decimal("100000000")
BASIS = Decimal("10000")

LIVE = live_risk_parameters()  # drawdown limit 0.20, daily 0.08
PAPER = RiskParameters()  # drawdown limit 0.30, daily 0.08


def snap(
    dd: str | Decimal = "0",
    *,
    day: date = D1,
    worst: str | Decimal = "0",
    basis: str | Decimal = BASIS,
    source: DayStartSource = "run_start",
    at: datetime | None = None,
    open_positions: int = 0,
) -> RiskSnapshot:
    """A ``RiskSnapshot`` with drawdown ``dd`` and a day-loss latch ``worst``.

    ``drawdown_pct`` is derived as ``(peak - nav) / peak`` (the gate's formula).
    ``nav_day_start`` equals ``basis`` so ``daily_loss_basis == basis`` whenever
    ``basis <= initial_cash`` semantics are not needed.
    """
    dd_d = Decimal(str(dd))
    nav = PEAK * (Decimal(1) - dd_d)
    basis_d = Decimal(str(basis))
    worst_d = Decimal(str(worst))
    return RiskSnapshot(
        as_of=at if at is not None else T0 + timedelta(days=(day - D1).days),
        day=day,
        nav=nav,
        peak_equity=PEAK,
        drawdown_pct=(PEAK - nav) / PEAK,
        nav_day_start=basis_d,
        daily_loss_basis=basis_d,
        nav_pnl=nav - basis_d,
        worst_nav_pnl=worst_d,
        realised_pnl=Decimal(0),
        day_start_source=source,
        open_position_count=open_positions,
    )


def live_cb(run_id: str = "wp16-live") -> CircuitBreaker:
    return build_run_circuit_breaker(run_id, LIVE)


def paper_cb(run_id: str = "wp16-paper") -> CircuitBreaker:
    return build_run_circuit_breaker(run_id, PAPER)


def warm(cb: CircuitBreaker, *, day: date = D1) -> CircuitBreaker:
    """Consume the restart watch with 2N calm snapshots (an uninterrupted breaker)."""
    for _ in range(2 * DEESCALATION_EVALUATIONS):
        cb.evaluate(snap("0", day=day))
    return cb


def run(
    cb: CircuitBreaker, dds: Sequence[str | float], *, day: date = D1
) -> list[CircuitBreakerDecision]:
    """Evaluate a drawdown series (same day) and return every decision."""
    return [cb.evaluate(snap(str(dd), day=day)) for dd in dds]


def levels(decisions: Sequence[CircuitBreakerDecision]) -> list[str]:
    return [d.level.value for d in decisions]


# ---------------------------------------------------------------------------
# Engine-level helpers (T-17 .. T-24)
# ---------------------------------------------------------------------------

HELD = "BTC/USDT"  # the symbol the portfolio holds (SELL target)
OTHER = "ETH/USDT"  # a symbol with no position (BUY target; no pyramiding clash)


class Clock:
    """A mutable injected risk clock."""

    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


class ScriptedSignals(BaseStrategy):
    """Returns ``params['signals']`` (a callable ``bar_index -> list[Signal]``).

    The engine calls ``on_bar`` once per symbol per bar; a scripted signal is
    emitted only on the call for ITS symbol, so each signal appears once per bar.
    """

    metadata = StrategyMetadata(name="wp16_scripted_signals", description="test-only")
    requires_exit_manager = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._fn = (params or {}).get("signals", lambda i: [])
        self.calls = 0  # distinct bars seen
        self._last_ts: datetime | None = None

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
    ) -> list[Signal]:
        ts = bars[-1].timestamp
        if ts != self._last_ts:
            self._last_ts = ts
            self.calls += 1
        symbol = bars[-1].symbol
        return [s for s in self._fn(self.calls - 1) if s.symbol == symbol]


def buy(symbol: str = OTHER, target: str = "1000") -> Signal:
    return Signal(
        strategy_id="wp16",
        symbol=symbol,
        direction=SignalDirection.BUY,
        target_position=Decimal(target),
    )


def sell(symbol: str = HELD) -> Signal:
    return Signal(
        strategy_id="wp16",
        symbol=symbol,
        direction=SignalDirection.SELL,
        target_position=Decimal(0),
    )


class SpyExecution:
    """Execution double: records signals, never fills."""

    def __init__(self) -> None:
        self.signals: list[Signal] = []

    async def process_signal(self, signal: Signal) -> list[Any]:
        self.signals.append(signal)
        return []

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    def get_open_orders(self) -> list[Any]:
        return []

    async def cancel_order(self, order_id: Any) -> None:
        return None


class RecordingPaper(PaperExecutionEngine):
    """A real paper engine that also records every signal and resulting order."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.signals: list[Signal] = []
        self.orders_out: list[Any] = []

    async def process_signal(self, signal: Signal) -> list[Any]:
        self.signals.append(signal)
        orders = await super().process_signal(signal)
        self.orders_out.extend(orders)
        return orders


def holding_portfolio(
    clock: Clock, *, quantity: str = "10", price: str = "1000"
) -> PortfolioAccounting:
    """Portfolio holding ``quantity`` HELD bought at ``price`` (NAV == 10 000)."""
    from tests.unit._wp15_support import make_fill

    p = PortfolioAccounting("wp16", Decimal("10000"), clock=clock)
    p.update_position(
        make_fill(quantity=quantity, price=price, symbol=HELD), current_price=Decimal(price)
    )
    return p


def bars_at(held_close: str, *, other_close: str = "100", at: datetime = T0) -> dict[str, OHLCVBar]:
    ts = at
    return {
        HELD: make_bar(symbol=HELD, timestamp=ts, close=held_close, open_=held_close,
                       high=held_close, low=held_close),
        OTHER: make_bar(symbol=OTHER, timestamp=ts, close=other_close, open_=other_close,
                        high=other_close, low=other_close),
    }


def history_of(current: dict[str, OHLCVBar]) -> dict[str, list[OHLCVBar]]:
    return {sym: [bar] for sym, bar in current.items()}


def build_engine(
    *,
    cb: CircuitBreaker | None,
    signals: Any,
    execution: Any | None = None,
    portfolio: PortfolioAccounting | None = None,
    risk_manager: DefaultRiskManager | None = None,
    mode: RunMode = RunMode.PAPER,
    protective_mode: bool = False,
    config: dict[str, Any] | None = None,
    params: RiskParameters | None = None,
) -> tuple[StrategyEngine, SpyExecution | Any, PortfolioAccounting]:
    """A real ``StrategyEngine`` over a spy execution engine and a real portfolio."""
    clock = Clock(T0)
    pf = portfolio if portfolio is not None else holding_portfolio(clock)
    rm = risk_manager if risk_manager is not None else DefaultRiskManager("wp16", params=params)
    ex = execution if execution is not None else SpyExecution()
    from unittest.mock import AsyncMock, MagicMock

    market_data = MagicMock()
    market_data.connect = AsyncMock()
    market_data.close = AsyncMock()
    engine = StrategyEngine(
        strategies=[ScriptedSignals("wp16-s", {"signals": signals})],
        execution_engine=ex,
        risk_manager=rm,
        market_data=market_data,
        portfolio=pf,
        symbols=[HELD, OTHER],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=mode,
        config=config or {"warmup_bars": 1},
        circuit_breaker=cb,
        protective_mode=protective_mode,
    )
    return engine, ex, pf


async def bar(engine: StrategyEngine, held_close: str, **kw: Any) -> None:
    """Process one bar exactly as ``run_backtest`` / the poll loop would."""
    kw.setdefault("at", T0 + timedelta(hours=engine._bar_count))
    cur = bars_at(held_close, **kw)
    engine._update_engine_prices(cur)
    await engine._process_bar(cur, history_of(cur))


#: Raise the daily-loss limit so a drawdown test isolates HALT / REDUCE.
HALT_ONLY = replace(live_risk_parameters(), max_daily_loss_pct=0.5)
#: Raise the drawdown limit so a daily-loss test isolates DAILY_LIMIT.
DAILY_ONLY = replace(live_risk_parameters(), max_drawdown_pct=0.5)


def skipped_reasons(engine: StrategyEngine) -> list[str]:
    return [s.skip_reason for s in engine._skip_logger._skipped]
