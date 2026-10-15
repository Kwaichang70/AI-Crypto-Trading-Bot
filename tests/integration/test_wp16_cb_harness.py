"""
tests/integration/test_wp16_cb_harness.py
-------------------------------------------
WP1.6 -- the circuit breaker through the REAL paper and live stacks
(``StrategyEngine`` + ``PaperExecutionEngine`` / ``LiveExecutionEngine`` +
``PortfolioAccounting`` + ``DefaultRiskManager`` + ``CCXTMarketDataService``)
over ``FakeCCXTExchange``.  Bars go through ``StrategyEngine._poll_and_process``
(the exact per-iteration body of ``run_live_loop``).

    WP16-T-34  live: HALT, then the bracket SL still fires; the run keeps going   (red@HEAD)
    WP16-T-35  live: HALT + a strategy SELL goes through; the same-bar BUY is dropped (red@HEAD)
    WP16-T-36  live: HALT + trailing stop fires; HALT + kill switch + flatten()
    WP16-T-37  live: U-1 recovery, gate coherence, flat-run persistence (SYN-03)
    WP16-T-38  live: DAILY_LIMIT across 00:00 UTC
    WP16-T-39  paper + live: REDUCE sizing is 0.5x the control, full size after recovery (red@HEAD)
    WP16-T-40  paper twins of T-34 / T-35                                         (red@HEAD)
    WP16-T-41  live resume: band initialisation -> HALT on the first bar
    WP16-T-42  protective resume + CB: single reason, SL fires, GET readable
    WP16-T-43  harness parity: default stacks carry the production breaker
    WP16-T-22b cadence on the real loop: idle polls and flatten() never evaluate

Numbers.  A 150 BUY on a 1 000 run is a 15% position, so a price fall of x% is a
NAV drawdown of ~0.15x%.  The tests that must reach HALT/REDUCE with a bracket SL
still ahead use a tiny drawdown limit (``max_drawdown_pct = 0.01``): HALT at a
1% NAV drawdown (~7% price fall), REDUCE from 0.5%, HALT exit below 0.9%, REDUCE
exit below 0.4%.  ``price_for_dd`` solves the price for an exact target drawdown.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from common.types import SignalDirection, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    ManualClock,
    PaperStack,
    build_live_stack,
    build_paper_stack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.models import Signal
from trading.risk import RiskParameters, live_risk_parameters
from trading.safety import CircuitBreakerResponse
from trading.strategy import BaseStrategy, StrategyMetadata

SYMBOL = "BTC/EUR"
TF = "1h"
TIMEFRAME = TimeFrame.ONE_HOUR
PRICE = Decimal("100")
INITIAL = Decimal("1000")

D_2105 = datetime(2026, 3, 10, 21, 5, tzinfo=UTC)
D_2300 = datetime(2026, 3, 10, 23, 0, tzinfo=UTC)
D_2330 = datetime(2026, 3, 10, 23, 30, tzinfo=UTC)
D1_0005 = datetime(2026, 3, 11, 0, 5, tzinfo=UTC)

_ROOM = {
    "max_position_size_pct": 0.5,
    "max_portfolio_exposure_pct": 1.0,
    "max_cluster_exposure_pct": 1.0,
}
#: 1% drawdown limit (HALT at 1%, exit < 0.9%, REDUCE 0.5% / 0.4%); 8% daily loss untouched.
LIVE_TINY = dataclasses.replace(live_risk_parameters(), max_drawdown_pct=0.01, **_ROOM)
PAPER_TINY = dataclasses.replace(RiskParameters(), max_drawdown_pct=0.01, **_ROOM)
_SL10 = {"bracket_stop_loss_pct": 0.10, "allow_pyramiding": True}


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("100000"))
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TF)
    patch_exchange_factory(monkeypatch, ex)
    return ex


class ScriptedMulti(BaseStrategy):
    """Emit several scripted signals on one bar: ``{call_index: ["buy", "sell", ...]}``."""

    metadata = StrategyMetadata(name="wp16_scripted_multi", description="WP1.6 harness-only")
    requires_exit_manager = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._script: dict[int, list[str]] = {
            int(k): list(v) for k, v in (self._params.get("script") or {}).items()
        }
        self._target = Decimal(str(self._params.get("target_notional", "150")))
        self._n = -1

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(self, bars: Sequence[Any], *, mtf_context: Any = None) -> list[Signal]:
        self._n += 1
        out: list[Signal] = []
        for action in self._script.get(self._n, []):
            sell = action == "sell"
            out.append(
                Signal(
                    strategy_id=self.strategy_id,
                    symbol=bars[-1].symbol,
                    direction=SignalDirection.SELL if sell else SignalDirection.BUY,
                    target_position=Decimal(0) if sell else self._target,
                )
            )
        return out


def _seq(script: dict[int, str], target: str = "150") -> ScriptedSequenceStrategy:
    return ScriptedSequenceStrategy("wp16-seq", {"script": script, "target_notional": target})


def _multi(script: dict[int, list[str]], target: str = "150") -> ScriptedMulti:
    return ScriptedMulti("wp16-multi", {"script": script, "target_notional": target})


async def _stack(
    kind: str,
    exchange: FakeCCXTExchange,
    strategy: BaseStrategy,
    *,
    config: dict[str, object] | None = None,
    params: RiskParameters | None = None,
    clock: ManualClock | None = None,
    circuit_breaker: bool = True,
) -> LiveStack | PaperStack:
    builder = build_live_stack if kind == "live" else build_paper_stack
    default = LIVE_TINY if kind == "live" else PAPER_TINY
    stack = await builder(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL,
        run_id=f"wp16-{kind}",
        engine_config=config if config is not None else dict(_SL10),
        risk_params=params if params is not None else default,
        clock=clock,
        circuit_breaker=circuit_breaker,
    )
    await start_and_warmup(stack, f"wp16-{kind}")
    return stack


def _sides(stack: LiveStack | PaperStack, exchange: FakeCCXTExchange) -> list[str]:
    if isinstance(stack, LiveStack):
        return [o["side"] for o in exchange.order_log]
    return [f.side.value for f in stack.execution.get_all_fills()]


def _qtys(stack: LiveStack | PaperStack, exchange: FakeCCXTExchange, side: str) -> list[Decimal]:
    if isinstance(stack, LiveStack):
        return [Decimal(str(o["amount"])) for o in exchange.order_log if o["side"] == side]
    return [f.quantity for f in stack.execution.get_all_fills() if f.side.value == side]


def price_for_dd(stack: LiveStack | PaperStack, dd: str) -> Decimal:
    """The BTC price at which the run's NAV is exactly ``dd`` below its peak."""
    snap = stack.portfolio.risk_snapshot()
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat
    cash = Decimal(stack.portfolio.get_summary()["current_cash"])
    target_nav = snap.peak_equity * (Decimal(1) - Decimal(dd))
    return ((target_nav - cash) / position.quantity).quantize(Decimal("0.0001"))


def _cb_level(stack: LiveStack | PaperStack) -> CircuitBreakerResponse:
    assert stack.circuit_breaker is not None
    return stack.circuit_breaker.current_response


def _skips(stack: LiveStack | PaperStack) -> list[str]:
    return [s.skip_reason for s in stack.engine._skip_logger._skipped]


def _spy_gate(stack: LiveStack | PaperStack) -> MagicMock:
    spy = MagicMock(wraps=stack.risk_manager.pre_trade_check)
    stack.risk_manager.pre_trade_check = spy  # type: ignore[method-assign]
    return spy


# ---------------------------------------------------------------------------
# T-34 / T-40  HALT, then the SL still fires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["live", "paper"])
async def test_wp16_t34_t40_halt_then_the_sl_still_fires_and_the_run_keeps_going(
    exchange: FakeCCXTExchange, kind: str
) -> None:
    stack = await _stack(kind, exchange, _seq({0: "buy", 1: "buy"}))
    gate = _spy_gate(stack)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # BUY fills (1.5 BTC)
    assert _sides(stack, exchange) == ["buy"]
    assert gate.call_count == 1

    # -7% on a 15% position: NAV drawdown ~1.05% >= 1% -> HALT on this bar; the
    # scripted second BUY is dropped by the circuit breaker, not by the gate.
    await step_bar(stack, SYMBOL, Decimal("93"), timeframe=TF)
    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert _sides(stack, exchange) == ["buy"]
    assert _skips(stack) == ["circuit_breaker_halt"]  # exactly one skip record
    assert gate.call_count == 1, "pre_trade_check must not be asked about a CB-dropped BUY"

    # the SL (-10% of entry) is breached on the next bar: the SELL fills, flat
    await step_bar(stack, SYMBOL, Decimal("89"), timeframe=TF)
    assert _sides(stack, exchange) == ["buy", "sell"]
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and position.is_flat

    # D4: the run is still RUNNING and the loop keeps polling
    assert not stack.engine._stop_event.is_set()
    assert stack.engine.auto_stop_reason is None
    before = stack.engine._bar_count
    await step_bar(stack, SYMBOL, Decimal("89"), timeframe=TF)
    assert stack.engine._bar_count == before + 1
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-35 / T-40  HALT + strategy SELL goes through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["live", "paper"])
async def test_wp16_t35_t40_halt_still_lets_the_strategy_sell_fill_and_drops_the_same_bar_buy(
    exchange: FakeCCXTExchange, kind: str
) -> None:
    stack = await _stack(kind, exchange, _multi({0: ["buy"], 1: ["buy", "sell"]}))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    held = stack.portfolio.get_position(SYMBOL)
    assert held is not None and not held.is_flat
    held_qty = held.quantity

    await step_bar(stack, SYMBOL, Decimal("93"), timeframe=TF)  # HALT + [BUY, SELL] on the bar

    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert _sides(stack, exchange) == ["buy", "sell"]  # the BUY was dropped, the SELL filled
    assert _qtys(stack, exchange, "sell")[0] == pytest.approx(held_qty, rel=Decimal("1e-6"))
    assert _skips(stack) == ["circuit_breaker_halt"]
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and position.is_flat
    assert not stack.engine._stop_event.is_set()
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-36  trailing stop + flatten under HALT
# ---------------------------------------------------------------------------


async def test_wp16_t36_halt_trailing_stop_fires(exchange: FakeCCXTExchange) -> None:
    stack = await _stack(
        "live", exchange, _seq({0: "buy"}),
        config={"trailing_stop_pct": 0.05, "allow_pyramiding": True},
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    await step_bar(stack, SYMBOL, Decimal("93"), timeframe=TF)  # -7%: HALT, and below the 5% trail
    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert _sides(stack, exchange) == ["buy", "sell"]
    await stack.engine.stop()


async def test_wp16_t36_halt_plus_kill_switch_flatten_still_flattens(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _stack(
        "live", exchange, _seq({0: "buy"}), config={"allow_pyramiding": True}
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    await step_bar(stack, SYMBOL, Decimal("93"), timeframe=TF)
    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert stack.circuit_breaker is not None
    evaluations = stack.circuit_breaker.state.evaluations

    stack.risk_manager.trigger_kill_switch("stop_in_progress")
    result = await stack.engine.flatten(reason="emergency_stop", timeout_s=30.0)

    assert result.outcome == "flattened"
    assert _sides(stack, exchange) == ["buy", "sell"]
    assert stack.circuit_breaker.state.evaluations == evaluations  # flatten never evaluates (T-22)
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-22b  cadence on the real loop
# ---------------------------------------------------------------------------


async def test_wp16_t22b_idle_polls_do_not_evaluate_new_bars_do(exchange: FakeCCXTExchange) -> None:
    stack = await _stack("live", exchange, _seq({}))
    assert stack.circuit_breaker is not None
    cb = stack.circuit_breaker
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    assert cb.state.evaluations == 1
    for _ in range(3):  # no new bar on the exchange: _poll_and_process returns early
        await stack.engine._poll_and_process()
    assert cb.state.evaluations == 1
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    assert cb.state.evaluations == 2
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-37  U-1 recovery and gate coherence
# ---------------------------------------------------------------------------


async def test_wp16_t37_halt_auto_recovers_through_reduce_and_never_bypasses_the_gate(
    exchange: FakeCCXTExchange,
) -> None:
    script = dict.fromkeys(range(8), "buy")
    stack = await _stack("live", exchange, _seq(script))
    assert stack.circuit_breaker is not None
    limit = Decimal("0.01")
    gate = _spy_gate(stack)
    buys_seen = 0
    gate_blocked_bars: list[int] = []

    async def bar_at(price: Decimal) -> Any:
        nonlocal buys_seen
        await step_bar(stack, SYMBOL, price, timeframe=TF)
        snap = stack.portfolio.risk_snapshot()
        new_buys = _sides(stack, exchange).count("buy") - buys_seen
        buys_seen += new_buys
        if snap.drawdown_pct >= limit:  # the WP1.5 gate predicate
            gate_blocked_bars.append(len(gate_blocked_bars))
            assert new_buys == 0, f"a BUY reached the exchange at dd={snap.drawdown_pct}"
        return snap, new_buys

    # idx0: BUY fills at 100 (calm)
    await bar_at(PRICE)
    # 1. holding with dd >= L -> HALT
    snap, _ = await bar_at(price_for_dd(stack, "0.0105"))
    assert _cb_level(stack) == CircuitBreakerResponse.HALT and snap.drawdown_pct >= limit
    # 2. marks recover into [0.9 L, L): the gate would ALLOW, the CB still blocks
    snap, buys = await bar_at(price_for_dd(stack, "0.0095"))
    assert Decimal("0.009") <= snap.drawdown_pct < limit
    assert _cb_level(stack) == CircuitBreakerResponse.HALT and buys == 0
    assert _skips(stack).count("circuit_breaker_halt") == 2
    # 3. three bars below 0.9 L -> REDUCE on the third; its BUY fills at half size
    for expected in ("halt", "halt"):
        _, buys = await bar_at(price_for_dd(stack, "0.004"))
        assert _cb_level(stack).value == expected and buys == 0
    price = price_for_dd(stack, "0.004")
    _, buys = await bar_at(price)
    assert _cb_level(stack) == CircuitBreakerResponse.REDUCE and buys == 1
    fill_qty = _qtys(stack, exchange, "buy")[-1]
    assert fill_qty == pytest.approx(Decimal("150") / price / 2, rel=Decimal("1e-4"))
    assert gate_blocked_bars, "the scenario must contain gate-blocked bars"
    # the gate was consulted only for the two BUYs that reached it (idx0 and the REDUCE BUY)
    assert gate.call_count == 2
    await stack.engine.stop()


async def test_wp16_t37_flat_run_at_the_limit_never_recovers(exchange: FakeCCXTExchange) -> None:
    """SYN-03: SL closes the position under HALT -> NAV constant -> HALT persists."""
    stack = await _stack("live", exchange, _seq(dict.fromkeys(range(1, 8), "buy")))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # idx0: no scripted BUY (script starts at 1)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # idx1: BUY fills
    assert _sides(stack, exchange) == ["buy"]
    await step_bar(stack, SYMBOL, Decimal("93"), timeframe=TF)  # HALT
    await step_bar(stack, SYMBOL, Decimal("89"), timeframe=TF)  # SL closes
    assert _sides(stack, exchange) == ["buy", "sell"]
    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    for price in ("89", "100", "120", "100", "90"):  # a flat run: marks cannot move the NAV
        await step_bar(stack, SYMBOL, Decimal(price), timeframe=TF)
        assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert _sides(stack, exchange) == ["buy", "sell"]  # no BUY ever again
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-38  DAILY_LIMIT across 00:00 UTC
# ---------------------------------------------------------------------------

_DAY_LOSS = dataclasses.replace(live_risk_parameters(), max_daily_loss_pct=0.02)


async def test_wp16_t38_daily_limit_blocks_until_the_utc_day_rolls(
    exchange: FakeCCXTExchange,
) -> None:
    clock = ManualClock(D_2105)
    stack = await _stack(
        "live", exchange, _seq({0: "buy", 2: "buy", 3: "buy"}),
        config={"bracket_stop_loss_pct": 0.05}, params=_DAY_LOSS, clock=clock,
    )
    gate = _spy_gate(stack)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # D 21:05 BUY 150 @ 100
    assert gate.call_count == 1

    clock.set(D_2300)
    await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # -30 > 20: DAILY_LIMIT; SL sells
    assert _cb_level(stack) == CircuitBreakerResponse.DAILY_LIMIT
    assert _sides(stack, exchange) == ["buy", "sell"]

    clock.set(D_2330)
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # scripted BUY: dropped
    assert _sides(stack, exchange) == ["buy", "sell"]
    assert _skips(stack)[-1] == "circuit_breaker_daily_limit"  # not `max_daily_loss`
    assert not [e for e in logs if e["event"] == "live.signal_rejected"]
    assert gate.call_count == 2  # only the SL SELL reached the gate since the first BUY

    clock.set(D1_0005)
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # D+1: cleared, BUY fills
    assert _cb_level(stack) == CircuitBreakerResponse.OK
    assert _sides(stack, exchange) == ["buy", "sell", "buy"]
    cleared = [e for e in logs if e["event"] == "circuit_breaker.daily_limit_cleared"]
    assert len(cleared) == 1
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-39  REDUCE sizing vs a no-breaker control
# ---------------------------------------------------------------------------


def _fresh_exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("100000"))
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TF)
    patch_exchange_factory(monkeypatch, ex)
    return ex


@pytest.mark.parametrize("kind", ["live", "paper"])
async def test_wp16_t39_reduce_halves_the_buy_and_full_size_returns_after_recovery(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The CONTROL stack opts out of the breaker (``circuit_breaker=False``): it is the
    no-breaker baseline the REDUCE quantity is compared against (K-15 written reason)."""
    script = {0: "buy", 2: "buy", 5: "buy"}

    # 1. breaker stack: bar0 BUY; bar1/2 in the REDUCE band; bars 3-5 below 0.4 L
    ex_cb = _fresh_exchange(monkeypatch)
    cb_stack = await _stack(kind, ex_cb, _seq(script))
    prices: list[Decimal] = [PRICE]
    await step_bar(cb_stack, SYMBOL, PRICE, timeframe=TF)
    for _ in range(2):  # dd 0.7% in [0.5%, 1%) -> REDUCE
        prices.append(price_for_dd(cb_stack, "0.007"))
        await step_bar(cb_stack, SYMBOL, prices[-1], timeframe=TF)
        assert _cb_level(cb_stack) == CircuitBreakerResponse.REDUCE
    assert cb_stack.circuit_breaker is not None
    assert cb_stack.circuit_breaker.state.entry_size_multiplier == "0.5"
    for i in range(3):  # three bars < 0.4% (0.4 L): REDUCE, REDUCE, then OK on the third
        prices.append(price_for_dd(cb_stack, "0.002"))
        await step_bar(cb_stack, SYMBOL, prices[-1], timeframe=TF)
        assert _cb_level(cb_stack) == (
            CircuitBreakerResponse.OK if i == 2 else CircuitBreakerResponse.REDUCE
        )
    cb_qty = _qtys(cb_stack, ex_cb, "buy")
    await cb_stack.engine.stop()

    # 2. control: identical price path, no breaker
    ex_ctl = _fresh_exchange(monkeypatch)
    ctl_stack = await _stack(kind, ex_ctl, _seq(script), circuit_breaker=False)
    for price in prices:
        await step_bar(ctl_stack, SYMBOL, price, timeframe=TF)
    ctl_qty = _qtys(ctl_stack, ex_ctl, "buy")
    await ctl_stack.engine.stop()

    assert len(cb_qty) == len(ctl_qty) == 3
    rel = Decimal("1e-6")
    assert cb_qty[0] == pytest.approx(ctl_qty[0], rel=rel)  # OK level: full size
    assert cb_qty[1] == pytest.approx(ctl_qty[1] / 2, rel=rel)  # REDUCE: exactly 0.5x
    assert cb_qty[2] == pytest.approx(ctl_qty[2], rel=rel)  # recovered: full size again


# ---------------------------------------------------------------------------
# T-41 / T-42  resume: band initialisation, protective mode
# ---------------------------------------------------------------------------

RESUME_RUN = "wp16-resume"


async def _persisted_buy(exchange: FakeCCXTExchange) -> list[Any]:
    """A first stack buys 150 at 100; return its fills as a resume would re-read them."""
    clock = ManualClock(D_2105)
    before = await build_live_stack(
        exchange=exchange, strategy=_seq({0: "buy"}), symbol=SYMBOL, timeframe=TIMEFRAME,
        initial_capital=INITIAL, run_id=RESUME_RUN,
        engine_config={"allow_pyramiding": True}, clock=clock,
    )
    await start_and_warmup(before, RESUME_RUN)
    await step_bar(before, SYMBOL, PRICE, timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    return [f.model_copy(update={"executed_at": D_2105}) for f in before.execution.get_all_fills()]


async def _resumed(
    exchange: FakeCCXTExchange,
    fills: list[Any],
    script: dict[int, str],
    *,
    protective: bool = False,
) -> LiveStack:
    stack = await build_resumed_live_stack(
        exchange=exchange, strategy=_seq(script), symbol=SYMBOL, timeframe=TIMEFRAME,
        initial_capital=INITIAL, run_id=RESUME_RUN, fills=fills,
        engine_config=dict(_SL10), protective_mode=protective,
        # a persisted high ~19.4% above the rebuilt NAV: inside [0.9 L, L) of the live 20% limit
        peak_equity_hint=Decimal("1240"), clock=ManualClock(D_2105),
    )
    await start_and_warmup(stack, RESUME_RUN)
    return stack


async def test_wp16_t41_a_resume_in_the_halt_exit_band_starts_in_halt_and_the_sl_still_fires(
    exchange: FakeCCXTExchange,
) -> None:
    fills = await _persisted_buy(exchange)
    stack = await _resumed(exchange, fills, {0: "buy"})
    snap = stack.portfolio.risk_snapshot()
    assert Decimal("0.18") <= snap.drawdown_pct < Decimal("0.20"), snap.drawdown_pct
    assert stack.circuit_breaker is not None and stack.circuit_breaker.state.evaluations == 0

    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # first bar: band initialisation -> HALT
    assert _cb_level(stack) == CircuitBreakerResponse.HALT
    assert [o["side"] for o in exchange.order_log] == ["buy"]  # the scripted BUY was dropped
    assert _skips(stack) == ["circuit_breaker_halt"]

    await step_bar(stack, SYMBOL, Decimal("89"), timeframe=TF)  # SL (-10%) fires under HALT
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    assert not stack.engine._stop_event.is_set()
    await stack.engine.stop()


async def test_wp16_t42_protective_resume_with_the_breaker_has_a_single_reason_and_readable_state(
    exchange: FakeCCXTExchange,
) -> None:
    from api.routers.circuit_breaker import get_circuit_breaker_state
    from api.services.run_orchestrator import _RUN_ENGINES

    fills = await _persisted_buy(exchange)
    stack = await _resumed(exchange, fills, {0: "buy"}, protective=True)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)

    assert _cb_level(stack) == CircuitBreakerResponse.HALT  # the breaker still evaluates...
    assert _skips(stack) == ["protective_mode"]  # ...but protective mode is the single reason
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    await step_bar(stack, SYMBOL, Decimal("89"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]  # the SL fires

    _RUN_ENGINES[RESUME_RUN] = stack.engine
    try:
        state = await get_circuit_breaker_state(RESUME_RUN)
    finally:
        _RUN_ENGINES.pop(RESUME_RUN, None)
    assert state.level == "halt" and state.evaluations == 2 and state.halt_count == 1
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# T-43  harness parity
# ---------------------------------------------------------------------------


async def test_wp16_t43_default_stacks_carry_the_production_breaker(
    exchange: FakeCCXTExchange,
) -> None:
    live = await build_live_stack(
        exchange=exchange, strategy=_seq({}), symbol=SYMBOL, timeframe=TIMEFRAME,
        initial_capital=INITIAL, run_id="wp16-par-live",
    )
    paper = await build_paper_stack(
        exchange=exchange, strategy=_seq({}), symbol=SYMBOL, timeframe=TIMEFRAME,
        initial_capital=INITIAL, run_id="wp16-par-paper",
    )
    off = await build_live_stack(
        exchange=exchange, strategy=_seq({}), symbol=SYMBOL, timeframe=TIMEFRAME,
        initial_capital=INITIAL, run_id="wp16-par-off", circuit_breaker=False,
    )
    assert live.circuit_breaker is not None and live.engine.circuit_breaker is live.circuit_breaker
    assert paper.circuit_breaker is not None
    assert paper.engine.circuit_breaker is paper.circuit_breaker
    assert live.circuit_breaker.thresholds is not None
    assert paper.circuit_breaker.thresholds is not None
    assert live.circuit_breaker.thresholds.halt_entry == Decimal("0.2")  # live 20%
    assert paper.circuit_breaker.thresholds.halt_entry == Decimal("0.3")  # paper 30%
    assert off.circuit_breaker is None and off.engine.circuit_breaker is None
