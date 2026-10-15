"""
tests/integration/test_wp15a_paper_live_harness.py
----------------------------------------------------
WP1.5a (Verbeterplan v2 row 1.5, C5) -- daily-PnL & peak semantics across
00:00 UTC through the REAL paper and live stacks (``StrategyEngine`` +
``PaperExecutionEngine`` / ``LiveExecutionEngine`` + ``PortfolioAccounting`` +
``DefaultRiskManager`` + ``CCXTMarketDataService``) over ``FakeCCXTExchange``.

    WP15-T-25  paper: day-1 latch, then a day-2 BUY fills; the peak follows marks (red@HEAD)
    WP15-T-26  live:  SL at D 23:00, BUY at 23:30 rejected, BUY at D+1 00:05 fills
               sized at min(NAV, initial)                                        (red@HEAD)
    WP15-T-27  live:  under a forced daily latch SL, TP, trailing, strategy
               SELL and the WP1.7 flatten all fill (AC-04)
    WP15-T-28  R-02:  ~28% drawdown -> live BUY rejected, paper BUY allowed;
               the live SL still fires under the latch
    WP15-T-42  live:  an outlier-low close at D 23:00 does not cushion D+1; a
               real D+1 decline rejects the BUY, the SL still fills     (red@1.5a)
    WP15-T-29  smoke: worst-case smoke loss is far below both gates (V9); the
               smoke suites are run unchanged as the regression gate

The paper / live risk-day clock is the wall clock UTC; the tests drive it with
a ``ManualClock`` injected into the portfolio and move it across 00:00 UTC
between two bars.  Bars are processed through ``StrategyEngine._poll_and_process``
(the exact per-iteration body of ``run_live_loop``) -- no shortcuts.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from structlog.testing import capture_logs

from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    ManualClock,
    PaperStack,
    build_live_stack,
    build_paper_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.risk import RiskParameters, live_risk_parameters

SYMBOL = "BTC/EUR"
TIMEFRAME = TimeFrame.ONE_HOUR
TF = "1h"
PRICE = Decimal("100")
INITIAL = Decimal("1000")

D_2105 = datetime(2026, 3, 10, 21, 5, tzinfo=UTC)
D_2300 = datetime(2026, 3, 10, 23, 0, tzinfo=UTC)
D_2330 = datetime(2026, 3, 10, 23, 30, tzinfo=UTC)
D1_0005 = datetime(2026, 3, 11, 0, 5, tzinfo=UTC)

#: 2% daily limit (20 on 1 000) so a 15% position that loses 20% latches the day.
_DAY_LOSS = dataclasses.replace(live_risk_parameters(), max_daily_loss_pct=0.02)
_SL = {"bracket_stop_loss_pct": 0.05}


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("100000"))  # run capital (1 000) << account
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TF)
    patch_exchange_factory(monkeypatch, ex)
    return ex


def _script(
    script: dict[int, str], target: str = "150", targets: dict[int, str] | None = None
) -> ScriptedSequenceStrategy:
    return ScriptedSequenceStrategy(
        "wp15-seq", {"script": script, "target_notional": target, "targets": targets or {}}
    )


async def _live(
    exchange: FakeCCXTExchange,
    strategy: ScriptedSequenceStrategy,
    clock: ManualClock,
    *,
    risk_params: RiskParameters | None = None,
    engine_config: dict[str, object] | None = None,
    circuit_breaker: bool = True,
) -> LiveStack:
    stack = await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL,
        run_id="wp15-live",
        engine_config=engine_config,
        risk_params=risk_params,
        clock=clock,
        circuit_breaker=circuit_breaker,
    )
    await start_and_warmup(stack, "wp15-live")
    return stack


async def _paper(
    exchange: FakeCCXTExchange,
    strategy: ScriptedSequenceStrategy,
    clock: ManualClock,
    *,
    risk_params: RiskParameters | None = None,
    engine_config: dict[str, object] | None = None,
    circuit_breaker: bool = True,
) -> PaperStack:
    stack = await build_paper_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL,
        run_id="wp15-paper",
        engine_config=engine_config,
        risk_params=risk_params,
        clock=clock,
        circuit_breaker=circuit_breaker,
    )
    await start_and_warmup(stack, "wp15-paper")
    return stack


#: WP1.6: with the harness circuit breaker on (production parity, K-15) the SAME
#: predicates that the WP1.5 gate enforces block the BUY one layer earlier, in the
#: engine's single BUY filter.  The skip record's reason replaces the gate's
#: ``signal_rejected`` text (a CB-dropped BUY never reaches the gate: one reason).
_CB_SKIP_REASON = {"daily loss": "circuit_breaker_daily_limit", "drawdown": "circuit_breaker_halt"}


def _rejections(logs: list[dict[str, Any]], prefix: str, needle: str) -> list[str]:
    out: list[str] = []
    for entry in logs:
        if entry["event"] == f"{prefix}.signal_rejected":
            out.extend(r for r in entry["reasons"] if needle in r.lower())
        elif (
            entry["event"] == "trade_skip_logger.skip_recorded"
            and entry.get("skip_reason") == _CB_SKIP_REASON.get(needle)
        ):
            out.append(entry["skip_reason"])
    return out


def _gate_rejections(logs: list[dict[str, Any]], prefix: str, needle: str) -> list[str]:
    """Only the WP1.5 gate's own ``signal_rejected`` reasons (pre-WP1.6 semantics)."""
    return [
        r
        for entry in logs
        if entry["event"] == f"{prefix}.signal_rejected"
        for r in entry["reasons"]
        if needle in r.lower()
    ]


def _cb_skips(logs: list[dict[str, Any]]) -> list[str]:
    return [
        e["skip_reason"]
        for e in logs
        if e["event"] == "trade_skip_logger.skip_recorded"
        and str(e.get("skip_reason", "")).startswith("circuit_breaker")
    ]


def _assert_layer(logs: list[dict[str, Any]], prefix: str, needle: str, cb: bool) -> None:
    """WP16-CR-03: pin WHICH layer blocked the BUY (K-10: exactly one reason).

    ``cb=True``: the breaker drops it and the gate is never asked.
    ``cb=False`` (gate-only twin): the WP1.5 gate rejects it, as the second layer.
    """
    if cb:
        assert _cb_skips(logs) == [_CB_SKIP_REASON[needle]], logs
        assert not _gate_rejections(logs, prefix, needle), logs
    else:
        assert _gate_rejections(logs, prefix, needle), logs
        assert not _cb_skips(logs), logs


#: WP16-CR-03 twin parametrization.  ``gate_only`` passes ``circuit_breaker=False``
#: for a written reason: it isolates the WP1.5 gate as the second layer, which the
#: production breaker otherwise pre-empts for these predicates.
_LAYERS = pytest.mark.parametrize("cb", [True, False], ids=["cb_on", "gate_only"])


# ---------------------------------------------------------------------------
# WP15-T-26  (live)
# ---------------------------------------------------------------------------


@_LAYERS
async def test_wp15_t26_live_day1_latch_then_day2_buy_sized_at_min_nav_initial(
    exchange: FakeCCXTExchange, cb: bool,
) -> None:
    clock = ManualClock(D_2105)
    stack = await _live(
        exchange, _script({0: "buy", 2: "buy", 3: "buy"}), clock,
        risk_params=_DAY_LOSS, engine_config=_SL, circuit_breaker=cb,
    )

    # D 21:05 -- BUY 150 at 100
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    # D 23:00 -- the bar closes -20%: the bracket stop-loss sells; the day loses > 2%
    clock.set(D_2300)
    await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]

    # D 23:30 -- a BUY on the same UTC day is rejected by the daily-loss latch
    clock.set(D_2330)
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)
    assert _rejections(logs, "live", "daily loss"), logs
    _assert_layer(logs, "live", "daily loss", cb)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]

    # D+1 00:05 -- a new UTC day: the BUY fills (HEAD kept blocking: the live
    # accumulator only rolled on a fill)
    clock.set(D1_0005)
    nav_before = stack.portfolio.current_equity
    await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell", "buy"]

    # sized at min(NAV, initial): the 15% ceiling of a NAV (< 1 000) basis
    assert nav_before < INITIAL
    notional = exchange.order_log[-1]["amount"] * exchange.order_log[-1]["price"]
    assert notional <= Decimal("0.15") * nav_before + Decimal("0.01")
    assert notional < Decimal("0.15") * INITIAL
    # the day state: a roll captured the NAV at the previous marks
    day2 = stack.portfolio.risk_snapshot()
    assert day2.day_start_source == "roll"
    assert day2.nav_day_start == nav_before
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-42  (live, SYN-01 / Option B)
# ---------------------------------------------------------------------------

#: 2% daily limit (20 on 1 000), room for a second BUY (so only the daily gate
#: decides it) and a stop-loss far enough away not to fire on the outlier.
_T42_RISK = dataclasses.replace(
    live_risk_parameters(),
    max_daily_loss_pct=0.02,
    max_position_size_pct=1.0,
    max_portfolio_exposure_pct=1.0,
    max_cluster_exposure_pct=1.0,
)
_T42_CONFIG = {"bracket_stop_loss_pct": 0.30, "allow_pyramiding": True}


async def test_wp15_t42_live_outlier_close_does_not_cushion_the_next_day(
    exchange: FakeCCXTExchange,
) -> None:
    clock = ManualClock(D_2105)
    stack = await _live(
        exchange, _script({0: "buy", 3: "buy"}), clock,
        risk_params=_T42_RISK, engine_config=_T42_CONFIG,
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # D 21:05 BUY 150 @ 100
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    clock.set(D_2300)  # D 23:00: an outlier-low close (-25%), SL (-30%) not hit
    await step_bar(stack, SYMBOL, Decimal("75"), timeframe=TF)
    outlier_nav = stack.portfolio.current_equity
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    clock.set(D1_0005)  # D+1 00:05: a normal mark -- the first observation of D+1
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    snap = stack.portfolio.risk_snapshot()
    assert snap.day_start_source == "roll"
    assert snap.nav_day_start == snap.nav > outlier_nav + Decimal("30")

    with capture_logs() as logs:  # a real D+1 decline: -22.5 >= 20 (2% of ~1 000)
        await step_bar(stack, SYMBOL, Decimal("85"), timeframe=TF)  # + scripted BUY
    assert _rejections(logs, "live", "daily loss"), logs
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    # AC-04 under the B latch: the SL on the held position still fills
    await step_bar(stack, SYMBOL, Decimal("65"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    await stack.engine.stop()


async def test_wp15_t42_control_without_the_decline_the_late_buy_fills(
    exchange: FakeCCXTExchange,
) -> None:
    clock = ManualClock(D_2105)
    stack = await _live(
        exchange, _script({0: "buy", 3: "buy"}), clock,
        risk_params=_T42_RISK, engine_config=_T42_CONFIG,
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    clock.set(D_2300)
    await step_bar(stack, SYMBOL, Decimal("75"), timeframe=TF)
    clock.set(D1_0005)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # scripted BUY
    assert not _rejections(logs, "live", "daily loss"), logs
    assert [o["side"] for o in exchange.order_log] == ["buy", "buy"]
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-25  (paper)
# ---------------------------------------------------------------------------


@_LAYERS
async def test_wp15_t25_paper_day1_latch_then_day2_buy_fills_and_peak_follows_marks(
    exchange: FakeCCXTExchange, cb: bool,
) -> None:
    clock = ManualClock(D_2105)
    stack = await _paper(
        exchange, _script({0: "buy", 3: "buy", 4: "buy"}), clock,
        risk_params=dataclasses.replace(RiskParameters(), max_daily_loss_pct=0.02),
        engine_config=_SL, circuit_breaker=cb,
    )
    portfolio = stack.portfolio

    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # BUY 150 @ 100 (1.5 BTC)
    assert portfolio.get_open_positions()

    clock.set(datetime(2026, 3, 10, 22, 0, tzinfo=UTC))
    await step_bar(stack, SYMBOL, Decimal("120"), timeframe=TF)  # mark +30
    peak = portfolio.get_peak_equity()
    assert peak > INITIAL, "the peak follows the portfolio's marks"
    assert peak == portfolio.current_equity

    clock.set(D_2300)
    await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # SL fires: -60 on the run
    assert not portfolio.get_open_positions()
    assert portfolio.get_peak_equity() == peak, "a fall never lowers the peak"

    clock.set(D_2330)
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # BUY: latched
    assert _rejections(logs, "paper", "daily loss"), logs
    _assert_layer(logs, "paper", "daily loss", cb)
    assert len(stack.execution.get_all_fills()) == 2

    clock.set(D1_0005)
    await step_bar(stack, SYMBOL, Decimal("80"), timeframe=TF)  # BUY on D+1 fills
    fills = stack.execution.get_all_fills()
    assert [f.side.value for f in fills] == ["buy", "sell", "buy"]
    assert portfolio.risk_snapshot().day_start_source == "roll"
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-27  (live exits under a forced daily latch, AC-04)
# ---------------------------------------------------------------------------

_EXIT_CASES = [
    # id, engine_config, script, price path after the BUY at 100
    ("stop_loss", {"bracket_stop_loss_pct": 0.05}, {0: "buy"}, ["90"]),
    ("take_profit", {"bracket_take_profit_pct": 0.05}, {0: "buy"}, ["110"]),
    ("trailing", {"trailing_stop_pct": 0.05}, {0: "buy"}, ["120", "110"]),
    ("strategy_sell", None, {0: "buy", 1: "sell"}, ["100"]),
    ("flatten", None, {0: "buy"}, []),
]


@pytest.mark.parametrize(
    ("case", "config", "script", "path"), _EXIT_CASES, ids=[c[0] for c in _EXIT_CASES]
)
async def test_wp15_t27_every_exit_fills_under_a_forced_daily_latch(
    exchange: FakeCCXTExchange,
    case: str,
    config: dict[str, object] | None,
    script: dict[int, str],
    path: list[str],
) -> None:
    clock = ManualClock(D_2105)
    stack = await _live(exchange, _script(script, "100"), clock, engine_config=config)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # BUY
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    # force the latch: worst NAV change -100 vs an 80 threshold (8% of 1 000)
    stack.portfolio._worst_nav_pnl = Decimal("-100")
    snap = stack.portfolio.risk_snapshot()
    assert snap.gate_daily_pnl(snap.nav) <= -snap.daily_loss_basis * Decimal("0.08")

    with capture_logs() as logs:
        for price in path:
            await step_bar(stack, SYMBOL, Decimal(price), timeframe=TF)
        if case == "flatten":
            stack.risk_manager.trigger_kill_switch("stop_in_progress")
            result = await stack.engine.flatten(reason="emergency_stop", timeout_s=30.0)
            assert result.outcome == "flattened"

    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"], exchange.order_log
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and position.is_flat
    bypasses = [e for e in logs if e["event"] == "risk.exposure_reducing_bypass"]
    assert bypasses and "max_daily_loss" in bypasses[-1]["bypassed_rules"]
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-28  (R-02)
# ---------------------------------------------------------------------------

#: A big position (BUY 700 of a 1 000 run) so a 40% price fall is a ~28% NAV
#: drawdown: past the 20% live limit, inside the 30% paper limit.
_BIG = {
    "max_position_size_pct": 1.0,
    "max_portfolio_exposure_pct": 1.0,
    "max_cluster_exposure_pct": 1.0,
    "max_daily_loss_pct": 0.5,
}


@_LAYERS
async def test_wp15_t28_live_blocks_buys_past_20pct_drawdown_but_stop_loss_still_fires(
    exchange: FakeCCXTExchange, cb: bool,
) -> None:
    clock = ManualClock(D_2105)
    params = dataclasses.replace(live_risk_parameters(), **_BIG)
    assert params.max_drawdown_pct == 0.20
    stack = await _live(
        exchange, _script({0: "buy", 1: "buy"}, "700"), clock,
        risk_params=params,
        engine_config={"bracket_stop_loss_pct": 0.45, "allow_pyramiding": True},
        circuit_breaker=cb,
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # BUY ~700
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("60"), timeframe=TF)  # NAV -28%: BUY probe
    assert _rejections(logs, "live", "drawdown"), logs
    _assert_layer(logs, "live", "drawdown", cb)
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    snap = stack.portfolio.risk_snapshot()
    assert Decimal("0.20") < snap.drawdown_pct < Decimal("0.30")

    await step_bar(stack, SYMBOL, Decimal("50"), timeframe=TF)  # SL (45%) under the latch
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    await stack.engine.stop()


async def test_wp15_t28_paper_allows_a_drawdown_between_20_and_30pct(
    exchange: FakeCCXTExchange,
) -> None:
    clock = ManualClock(D_2105)
    params = dataclasses.replace(RiskParameters(), **_BIG)
    assert params.max_drawdown_pct == 0.30
    stack = await _paper(
        exchange, _script({0: "buy", 1: "buy"}, "700", {1: "5"}), clock,
        risk_params=params,
        engine_config={"allow_pyramiding": True},
        circuit_breaker=False,
    )
    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    assert len(stack.execution.get_all_fills()) == 1

    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("60"), timeframe=TF)
    assert not _rejections(logs, "paper", "drawdown"), logs
    snap = stack.portfolio.risk_snapshot()
    assert Decimal("0.20") < snap.drawdown_pct < Decimal("0.30")
    # the RISK GATE approved the probe BUY (clamped to the remaining position room)
    approved = [
        e for e in logs
        if e["event"] == "risk.pre_trade_check.approved" and e["side"] == "buy"
    ]
    assert len(approved) == 1
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-29  (smoke regression: V9 arithmetic)
# ---------------------------------------------------------------------------


def test_wp15_t29_smoke_worst_case_loss_is_far_below_both_limits() -> None:
    """One smoke round trip: notional <= 9.50, SL <= 8%, capital >= 60."""
    notional, sl, capital = Decimal("9.50"), Decimal("0.08"), Decimal("60")
    fees = notional * Decimal("0.006") * 2
    worst = notional * sl + fees  # ~0.87
    assert worst / capital < Decimal("0.02")
    assert worst / capital < Decimal(str(live_risk_parameters().max_drawdown_pct)) / 5
    assert worst < capital * Decimal(str(RiskParameters().max_daily_loss_pct))  # 4.80
