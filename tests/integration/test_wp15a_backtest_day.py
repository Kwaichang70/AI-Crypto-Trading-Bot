"""
tests/integration/test_wp15a_backtest_day.py
----------------------------------------------
WP1.5a (Verbeterplan v2 row 1.5, C5) -- daily-PnL & peak semantics through the
REAL ``BacktestRunner`` stack (``StrategyEngine`` + ``PaperExecutionEngine`` +
``DefaultRiskManager`` + ``PortfolioAccounting``; nothing mocked).

    WP15-T-20  an 8%-style loss on day 1 does not block day 2      (red@HEAD)
    WP15-T-21  the peak follows NAV: a fall from the high blocks BUYs (red@HEAD, variant A)
    WP15-T-22  5% losses per day: daily gate never blocks, drawdown gate does
    WP15-T-23  determinism: two different wall clocks, identical results
    WP15-T-24  day attribution: bar CLOSE time labels the risk day
    WP15-T-41  a pre-midnight outlier close does not depress the next day's
               start (SYN-01 / Option B)                                  (red@1.5a)

Scenario clock convention (spec §5.2): the risk day of a bar is the UTC day of
its CLOSE time (``timestamp + timeframe``).  An hourly bar opening 23:00 on D
closes at 00:00 on D+1 and belongs to D+1; every 1d bar is its own day.

The ``Trace`` fixture records, without altering behaviour, the risk-state
snapshot after every bar's marks, every fill (simulated time, side, qty,
price) and every ``pre_trade_check`` (side, verdict, reasons, gate inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import trading.portfolio as portfolio_module
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.models import Fill, Order, RiskCheckResult
from trading.portfolio import PortfolioAccounting
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.trade_journal import TradeSkipLogger

if TYPE_CHECKING:
    from trading.portfolio import RiskSnapshot

SYMBOL = "BTC/USDT"
INITIAL = Decimal("10000")
D = date(2026, 3, 10)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class GateCall:
    side: str
    approved: bool
    reasons: list[str]
    daily_pnl: Decimal
    current_equity: Decimal
    peak_equity: Decimal
    daily_loss_basis: Decimal | None
    bar_close: datetime | None


@dataclass
class Trace:
    snapshots: list[RiskSnapshot] = field(default_factory=list)
    fills: list[tuple[datetime | None, str, Decimal, Decimal]] = field(default_factory=list)
    gate_calls: list[GateCall] = field(default_factory=list)
    equity_values: list[Decimal] = field(default_factory=list)

    def rejections(self, needle: str | None = None) -> list[GateCall]:
        out = [g for g in self.gate_calls if not g.approved]
        if needle is not None:
            out = [g for g in out if any(needle in r.lower() for r in g.reasons)]
        return out

    def fills_of(self, side: str) -> list[tuple[datetime | None, str, Decimal, Decimal]]:
        return [f for f in self.fills if f[1] == side]

    def comparable(self) -> tuple[Any, ...]:
        """Everything that must be identical across wall clocks."""
        return (
            self.fills,
            [
                (g.side, g.approved, g.reasons, g.daily_pnl, g.current_equity,
                 g.peak_equity, g.daily_loss_basis, g.bar_close)
                for g in self.gate_calls
            ],
            self.snapshots,
        )


def _install_trace(mp: pytest.MonkeyPatch) -> Trace:
    """Record (without altering behaviour) per-bar snapshots, fills and gate calls."""
    t = Trace()

    original_marks = PortfolioAccounting.update_market_prices
    original_update = PortfolioAccounting.update_position
    original_check = DefaultRiskManager.pre_trade_check

    # (``getattr`` keeps the recorder runnable against code that predates the
    # risk-state API, so these scenarios can be run red-before-green.)
    def marks(self: PortfolioAccounting, prices: dict[str, Decimal]) -> None:
        original_marks(self, prices)
        reader = getattr(self, "risk_snapshot", None)
        if reader is not None:
            t.snapshots.append(reader())

    def update(self: PortfolioAccounting, fill: Fill, current_price: Decimal) -> None:
        now = getattr(self, "_now", None)
        t.fills.append((now() if now else None, fill.side.value, fill.quantity, fill.price))
        original_update(self, fill, current_price)

    def check(self: DefaultRiskManager, order: Order, *args: Any, **kwargs: Any) -> RiskCheckResult:
        result = original_check(self, order, *args, **kwargs)
        bound = dict(zip(
            ("current_equity", "open_positions", "daily_pnl", "peak_equity", "market_price"),
            args, strict=False,
        ))
        bound.update(kwargs)
        t.gate_calls.append(
            GateCall(
                side=order.side.value,
                approved=result.approved,
                reasons=list(result.rejection_reasons),
                daily_pnl=bound["daily_pnl"],
                current_equity=bound["current_equity"],
                peak_equity=bound["peak_equity"],
                daily_loss_basis=bound.get("daily_loss_basis"),
                bar_close=t.snapshots[-1].as_of if t.snapshots else None,
            )
        )
        return result

    mp.setattr(PortfolioAccounting, "update_market_prices", marks)
    mp.setattr(PortfolioAccounting, "update_position", update)
    mp.setattr(DefaultRiskManager, "pre_trade_check", check)
    return t


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    return _install_trace(monkeypatch)


@pytest.fixture
def skips(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The ``skip_reason`` of every skipped trade, in order (the breaker twins, WP21C-D-10)."""
    reasons: list[str] = []
    original = TradeSkipLogger.log_skip

    def spy(
        self: TradeSkipLogger, symbol: str, skip_reason: str, *args: Any, **kwargs: Any
    ) -> Any:
        reasons.append(skip_reason)
        return original(self, symbol, skip_reason, *args, **kwargs)

    monkeypatch.setattr(TradeSkipLogger, "log_skip", spy)
    return reasons


async def _run(
    closes: list[str | int],
    *,
    first_open: datetime,
    script: dict[int, str],
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    risk: RiskParameters | None = None,
    bracket: dict[str, object] | None = None,
    pyramiding: bool = False,
    target: str = "1500",
    lead_strategies: list[ScriptedSequenceStrategy] | None = None,
    breaker: bool = False,
) -> BacktestRunner:
    # ``lead_strategies`` run (and their signals are processed) BEFORE the main one.
    strategies = [
        *(lead_strategies or []),
        ScriptedSequenceStrategy("wp15-main", {"script": script, "target_notional": target}),
    ]
    runner = BacktestRunner(
        strategies=strategies,
        symbols=[SYMBOL],
        timeframe=timeframe,
        initial_capital=INITIAL,
        risk_params=risk,
        seed=7,
        bracket_config=bracket,
        allow_pyramiding=pyramiding,
        # WP2.1c: these scenarios pin the WP1.5 RISK GATE (daily-loss / drawdown at
        # ``pre_trade_check``).  The default breaker would drop the BUY before the gate
        # sees it (K-10) and hide exactly what is asserted, so it is pinned off.  The
        # ``cb_on`` twins (``breaker=True``) pin the reporting layer the other way round.
        enable_circuit_breaker=breaker,
    )
    bars = scripted_path(closes, first_open=first_open, symbol=SYMBOL, timeframe=timeframe)
    await runner.run({SYMBOL: bars})
    return runner


# ---------------------------------------------------------------------------
# WP15-T-20: "8% loss on day 1 does not block day 2"
# ---------------------------------------------------------------------------

#: 2% daily limit (200 on 10 000) so a 15% position that loses 20% (-300)
#: latches the day without needing an 8% loss on a 15%-sized book.
_DAY_LOSS_RISK = RiskParameters(max_daily_loss_pct=0.02, max_position_size_pct=0.15)
_SL = {"bracket_stop_loss_pct": 0.05}


async def test_wp15_t20_day1_loss_blocks_rest_of_day_but_not_day2_hourly(trace: Trace) -> None:
    # Bar j opens at 06:00 + j h on D; j=17 opens 23:00 and CLOSES 00:00 D+1.
    closes = [100, 80] + [80] * 16
    await _run(
        closes,
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 2: "buy", 17: "buy"},
        risk=_DAY_LOSS_RISK,
        bracket=_SL,
    )

    # SL fired on the loss bar: BUY, SELL, then (day 2) BUY again
    assert [f[1] for f in trace.fills] == ["buy", "sell", "buy"]

    day_rejections = trace.rejections("daily loss")
    assert len(day_rejections) == 1, trace.gate_calls
    assert day_rejections[0].bar_close == datetime(2026, 3, 10, 9, 0, tzinfo=UTC)  # j=2

    # The second BUY filled on the first bar of D+1 (close 00:00 UTC)
    buy_times = [f[0] for f in trace.fills_of("buy")]
    assert buy_times[0].date() == D
    assert buy_times[1] == datetime(2026, 3, 11, 0, 0, tzinfo=UTC)
    approved_d1_buy = [g for g in trace.gate_calls if g.approved and g.bar_close == buy_times[1]]
    assert len(approved_d1_buy) == 1
    # day-2 basis is the (lower) day-start NAV, not the initial 10 000
    assert approved_d1_buy[0].daily_loss_basis is not None
    assert approved_d1_buy[0].daily_loss_basis < INITIAL
    assert approved_d1_buy[0].daily_pnl == Decimal("0")


async def test_wp15_t20_day1_loss_does_not_block_the_next_1d_bar(trace: Trace) -> None:
    await _run(
        [100, 80, 80],
        first_open=datetime(2026, 3, 10, 0, 0, tzinfo=UTC),
        timeframe=TimeFrame.ONE_DAY,
        script={0: "buy", 2: "buy"},
        risk=_DAY_LOSS_RISK,
        bracket=_SL,
    )
    assert [f[1] for f in trace.fills] == ["buy", "sell", "buy"]
    assert trace.rejections() == []


# ---------------------------------------------------------------------------
# WP15-T-21: the peak follows NAV
# ---------------------------------------------------------------------------

_PEAK_RISK = RiskParameters(
    max_position_size_pct=0.4, max_daily_loss_pct=0.5, max_drawdown_pct=0.10
)


async def test_wp15_t21_peak_follows_nav_strategy_exit_still_fills(trace: Trace) -> None:
    """Variant A (red@HEAD): the position rises 50% then falls back.  The
    BUY at j=3 is rejected on the portfolio's REAL peak; the strategy SELL
    at j=4 still fills under the drawdown block."""
    await _run(
        [100, 150, 100, 100, 100],
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 3: "buy", 4: "sell"},
        risk=_PEAK_RISK,
        pyramiding=True,
        target="3900",
    )
    dd = trace.rejections("drawdown")
    assert len(dd) == 1, trace.gate_calls
    assert dd[0].peak_equity > Decimal("11800")  # ~ 6100 cash + 39 x 150
    assert [f[1] for f in trace.fills] == ["buy", "sell"]
    # the exit was gated by the same block but bypassed (WP1.2)
    sell_check = next(g for g in trace.gate_calls if g.side == "sell")
    assert sell_check.approved is True


async def test_wp15_t21_peak_follows_nav_bracket_exit_still_fills(trace: Trace) -> None:
    """Variant B: the bracket SL fires at j=2 while the drawdown limit is
    already exceeded; the exit fills, the later BUY is rejected."""
    await _run(
        [100, 150, 70, 70],
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 3: "buy"},
        risk=_PEAK_RISK,
        bracket={"bracket_stop_loss_pct": 0.20},
        pyramiding=True,
        target="3900",
    )
    assert [f[1] for f in trace.fills] == ["buy", "sell"]
    assert len(trace.rejections("drawdown")) == 1
    sell_check = next(g for g in trace.gate_calls if g.side == "sell")
    assert sell_check.approved is True


# ---------------------------------------------------------------------------
# WP15-T-22: -5% per day for several days
# ---------------------------------------------------------------------------


async def test_wp15_t22_daily_gate_never_blocks_drawdown_gate_does(trace: Trace) -> None:
    # Every 1d bar closes 12.5% lower; a 40% position that is closed and
    # re-opened each bar loses ~5% of NAV per day (never reaching the 8% limit).
    n = 14
    closes: list[Decimal] = []
    price = Decimal(100)
    for _ in range(n):
        closes.append(price)
        price *= Decimal("0.875")
    sells = ScriptedSequenceStrategy("wp15-sells", {"script": dict.fromkeys(range(1, n), "sell")})
    await _run(
        closes,
        first_open=datetime(2026, 3, 10, tzinfo=UTC),
        timeframe=TimeFrame.ONE_DAY,
        script=dict.fromkeys(range(n), "buy"),
        target="4000",
        risk=RiskParameters(
            max_position_size_pct=0.4,
            loss_streak_count=1000,  # keep the loss-streak cooldown out of the way
        ),
        lead_strategies=[sells],  # SELL first, then re-BUY, on the same bar
    )

    assert trace.rejections("daily loss") == [], "a ~5% day never reaches the 8% limit"
    dd_rejections = trace.rejections("drawdown")
    assert dd_rejections, "cumulative losses must eventually trip the 30% drawdown gate"

    first_block = dd_rejections[0]
    assert first_block.side == "buy"
    drawdown_at_block = (
        first_block.peak_equity - first_block.current_equity
    ) / first_block.peak_equity
    assert drawdown_at_block >= Decimal("0.30")
    # every BUY decided before that point saw a drawdown < 30% and was approved
    earlier = trace.gate_calls[: trace.gate_calls.index(first_block)]
    earlier_buys = [g for g in earlier if g.side == "buy"]
    assert earlier_buys and all(g.approved for g in earlier_buys)
    assert all(
        (g.peak_equity - g.current_equity) / g.peak_equity < Decimal("0.30") for g in earlier_buys
    )
    # no daily loss was ever near the limit at a BUY decision
    assert all(
        -g.daily_pnl < Decimal("0.08") * (g.daily_loss_basis or INITIAL)
        for g in trace.gate_calls
        if g.side == "buy"
    )


# ---------------------------------------------------------------------------
# WP15-T-23: determinism across wall clocks
# ---------------------------------------------------------------------------


async def _t23_scenario(wall: datetime) -> Trace:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(portfolio_module, "_utc_now", lambda: wall)
        t = _install_trace(mp)
        runner = await _run(
            [100, 80] + [80] * 16,
            first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
            script={0: "buy", 2: "buy", 17: "buy"},
            risk=_DAY_LOSS_RISK,
            bracket=_SL,
        )
        assert runner.last_portfolio is not None
        t.equity_values = [v for _, v in runner.last_portfolio.get_equity_curve()][1:]
    return t


async def test_wp15_t23_backtest_is_independent_of_the_wall_clock() -> None:
    near_midnight = datetime(2026, 1, 1, 23, 59, 59, tzinfo=UTC)
    midday = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

    a = await _t23_scenario(near_midnight)
    b = await _t23_scenario(midday)

    assert a.fills, "scenario must trade"
    assert a.rejections("daily loss"), "scenario must exercise the latch"
    assert a.comparable() == b.comparable()
    assert a.equity_values == b.equity_values and a.equity_values
    assert len(a.snapshots) == 50 + 18  # one per bar, warmup included


# ---------------------------------------------------------------------------
# WP15-T-24: day attribution (close time)
# ---------------------------------------------------------------------------


async def test_wp15_t24_hourly_bar_opening_2300_belongs_to_the_next_day(trace: Trace) -> None:
    # processed bars open 20:00 .. 02:00 across midnight
    await _run(
        [100] * 7,
        first_open=datetime(2026, 3, 10, 20, 0, tzinfo=UTC),
        script={},
    )
    by_close = {s.as_of: s for s in trace.snapshots}
    open_22 = by_close[datetime(2026, 3, 10, 23, 0, tzinfo=UTC)]  # bar 22:00-23:00
    open_23 = by_close[datetime(2026, 3, 11, 0, 0, tzinfo=UTC)]  # bar 23:00-00:00
    assert open_22.day == D
    assert open_23.day == D + timedelta(days=1)
    assert open_23.day_start_source == "roll"
    # max(NAV at the previous marks, NAV at the first D+1 mark): flat prices
    assert open_23.nav_day_start == open_22.nav
    # and it holds for the rest of that UTC day
    assert by_close[datetime(2026, 3, 11, 1, 0, tzinfo=UTC)].day == D + timedelta(days=1)


async def test_wp15_t24_every_1d_bar_is_its_own_day(trace: Trace) -> None:
    await _run(
        [100] * 5,
        first_open=datetime(2026, 3, 10, tzinfo=UTC),
        timeframe=TimeFrame.ONE_DAY,
        script={},
    )
    processed = trace.snapshots[50:]
    assert [s.day for s in processed] == [D + timedelta(days=k) for k in range(1, 6)]
    assert all(s.day_start_source == "roll" for s in processed)
    assert len({s.day for s in trace.snapshots}) == len(trace.snapshots)


# ---------------------------------------------------------------------------
# WP15-T-41: a pre-midnight outlier close must not depress the next day's start
# ---------------------------------------------------------------------------


#: As T-20 (2% daily limit) but room for a second 15% BUY, so the late BUY is
#: decided by the daily-loss gate and nothing else.
_T41_RISK = RiskParameters(
    max_daily_loss_pct=0.02,
    max_position_size_pct=0.4,
    max_portfolio_exposure_pct=1.0,
    max_cluster_exposure_pct=1.0,
)


def _t41_closes(decline: str | int) -> list[str | int]:
    # j=0 opens 06:00 on D.  j=16 (opens 22:00) closes 23:00 on D at an outlier
    # low; j=17 (opens 23:00) closes 00:00 D+1 at the normal price (the first
    # D+1 observation); j=18 is the real D+1 move; j=19 carries the late BUY.
    closes: list[str | int] = [100] * 16
    return [*closes, 50, 100, decline, decline]


async def test_wp15_t41_outlier_close_before_midnight_does_not_cushion_the_next_day(
    trace: Trace,
) -> None:
    await _run(
        _t41_closes(80),  # a real D+1 decline: 15 x -20 = -300 >= 2% of 10 000
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 19: "buy"},
        risk=_T41_RISK,
        pyramiding=True,
    )
    by_close = {s.as_of: s for s in trace.snapshots}
    last_d = by_close[datetime(2026, 3, 10, 23, 0, tzinfo=UTC)]
    first_d1 = by_close[datetime(2026, 3, 11, 0, 0, tzinfo=UTC)]
    # the outlier really depressed the last D mark (~ -750 on 10 000)
    assert last_d.nav < first_d1.nav - Decimal("700")
    assert first_d1.nav_day_start == first_d1.nav  # re-based, not the outlier NAV
    assert first_d1.worst_nav_pnl == Decimal("0")

    assert [f[1] for f in trace.fills] == ["buy"], trace.fills
    rejections = trace.rejections("daily loss")
    assert len(rejections) == 1, trace.gate_calls
    assert rejections[0].bar_close == datetime(2026, 3, 11, 2, 0, tzinfo=UTC)  # j=19


async def test_wp15_t41_control_without_the_decline_the_late_buy_fills(trace: Trace) -> None:
    await _run(
        _t41_closes(100),
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 19: "buy"},
        risk=_T41_RISK,
        pyramiding=True,
    )
    assert trace.rejections() == [], trace.gate_calls
    assert [f[1] for f in trace.fills] == ["buy", "buy"]


# ---------------------------------------------------------------------------
# WP21C-D-10: the ``cb_on`` twins of T-20 (daily) and T-22 (drawdown)
#
# The scenarios above pin the WP1.5 gate with the breaker OFF.  P-WP16-3 / FS-04 ask for the
# same scenario with the breaker ON, which pins the reporting layer: the breaker drops the BUY
# first (a ``circuit_breaker_*`` skip, K-10), the gate is never asked about it, and so the
# gate records no rejection.
# ---------------------------------------------------------------------------


def _buy_gate_calls(trace: Trace) -> list[GateCall]:
    return [g for g in trace.gate_calls if g.side == "buy"]


async def test_wp15_t20_cb_on_the_breaker_drops_the_daily_loss_buy_before_the_gate(
    trace: Trace, skips: list[str]
) -> None:
    """Twin of ``test_wp15_t20_day1_loss_blocks_rest_of_day_but_not_day2_hourly``."""
    await _run(
        [100, 80] + [80] * 16,
        first_open=datetime(2026, 3, 10, 6, 0, tzinfo=UTC),
        script={0: "buy", 2: "buy", 17: "buy"},
        risk=_DAY_LOSS_RISK,
        bracket=_SL,
        breaker=True,
    )
    # the j=2 BUY (bar close 09:00) is the one the breaker dropped: a circuit_breaker_* skip ...
    cb_skips = [r for r in skips if r.startswith("circuit_breaker_")]
    assert cb_skips == ["circuit_breaker_daily_limit"], skips
    # ... and pre_trade_check was never called for it: 3 BUY signals = 2 gate calls + 1 skip
    assert len(_buy_gate_calls(trace)) + len(cb_skips) == 3, trace.gate_calls
    dropped_bar = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)
    assert [g for g in _buy_gate_calls(trace) if g.bar_close == dropped_bar] == []
    # ... so the gate records no rejection at all (P-WP16-3)
    assert trace.rejections() == [], trace.gate_calls


async def test_wp15_t22_cb_on_the_breaker_halts_entries_before_the_drawdown_gate(
    trace: Trace, skips: list[str]
) -> None:
    """Twin of ``test_wp15_t22_daily_gate_never_blocks_drawdown_gate_does``."""
    n = 14
    closes: list[Decimal] = []
    price = Decimal(100)
    for _ in range(n):
        closes.append(price)
        price *= Decimal("0.875")
    sells = ScriptedSequenceStrategy("wp15-sells", {"script": dict.fromkeys(range(1, n), "sell")})
    await _run(
        closes,
        first_open=datetime(2026, 3, 10, tzinfo=UTC),
        timeframe=TimeFrame.ONE_DAY,
        script=dict.fromkeys(range(n), "buy"),
        target="4000",
        risk=RiskParameters(
            max_position_size_pct=0.4,
            loss_streak_count=1000,  # keep the loss-streak cooldown out of the way
        ),
        lead_strategies=[sells],  # SELL first, then re-BUY, on the same bar
        breaker=True,
    )
    # the same path that trips the 30% drawdown gate with the breaker off ends in HALT here
    cb_skips = [r for r in skips if r.startswith("circuit_breaker_")]
    assert cb_skips and set(cb_skips) == {"circuit_breaker_halt"}, skips
    # pre_trade_check was never called for a dropped BUY: n BUY signals = gate calls + skips
    assert len(_buy_gate_calls(trace)) + len(cb_skips) == n, trace.gate_calls
    # ... so neither the drawdown gate nor any other gate rejected anything (P-WP16-3)
    assert trace.rejections() == [], trace.gate_calls
    assert trace.rejections("drawdown") == []
