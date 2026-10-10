"""
tests/integration/test_wp15b_live_resume_harness.py
-----------------------------------------------------
WP1.5b T-34 -- the resume daily-risk-state seed through the REAL live stack
(``StrategyEngine`` + ``LiveExecutionEngine`` + ``PortfolioAccounting`` rebuilt
by the production ``from_fills`` + ``DefaultRiskManager`` +
``CCXTMarketDataService``) over ``FakeCCXTExchange``.

    WP15-T-34a  AC-10: BUY + a loss today, restart with a snapshot seed ->
                BUYs are blocked for the rest of the UTC day while the stop-loss
                still fires; on D+1 a BUY fills
    WP15-T-34b  R-5: a depressed pre-midnight hint plus a normal first-of-today
                snapshot -> ``nav_day_start = day_first``; a real loss blocks BUYs
                while the SL fires (control: without ``day_first`` the cushion
                returns and the BUY fills)
    WP15-T-34c  R-5: no snapshots yet today -> the first post-resume mark
                re-bases the provisional day start
    WP15-T-34d  R-2 / R-1: an INVALID seed (float ``day_first_equity``, ``day_min``
                None -- SEC-208) never aborts a resume, a protective resume
                included: the resume completes, BUYs are blocked, the SL fires

The risk-day clock is the injected ``ManualClock`` (passed to ``from_fills``); bars go
through ``StrategyEngine._poll_and_process`` (``step_bar``) -- no shortcuts.
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
    build_live_stack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.models import Fill
from trading.recovery import ResumeDailySeed, ResumeSnapshot
from trading.risk import live_risk_parameters

SYMBOL = "BTC/EUR"
TIMEFRAME = TimeFrame.ONE_HOUR
TF = "1h"
PRICE = Decimal("100")
INITIAL = Decimal("1000")
RUN_ID = "wp15b-live"

D_2105 = datetime(2026, 3, 10, 21, 5, tzinfo=UTC)
D_2305 = datetime(2026, 3, 10, 23, 5, tzinfo=UTC)
D1_0005 = datetime(2026, 3, 11, 0, 5, tzinfo=UTC)
D1_0030 = datetime(2026, 3, 11, 0, 30, tzinfo=UTC)
_STARTED = datetime(2026, 3, 5, 8, 0, tzinfo=UTC)

#: 2% daily limit (20 on 1 000); room for a second BUY so only the daily gate decides it;
#: the 30% stop-loss (70 on a 100 entry) is far enough away not to fire on a -15% mark.
_RISK = dataclasses.replace(
    live_risk_parameters(),
    max_daily_loss_pct=0.02,
    max_position_size_pct=1.0,
    max_portfolio_exposure_pct=1.0,
    max_cluster_exposure_pct=1.0,
)
_CONFIG = {"bracket_stop_loss_pct": 0.30, "allow_pyramiding": True}


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


def _script(script: dict[int, str]) -> ScriptedSequenceStrategy:
    return ScriptedSequenceStrategy("wp15b-seq", {"script": script, "target_notional": "150"})


async def _persisted_buy(exchange: FakeCCXTExchange) -> list[Fill]:
    """Run a first stack that buys 150 at 100 on D 21:05; return its fills as persisted."""
    clock = ManualClock(D_2105)
    before = await build_live_stack(
        exchange=exchange,
        strategy=_script({0: "buy"}),
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        engine_config=_CONFIG,
        risk_params=_RISK,
        clock=clock,
    )
    await start_and_warmup(before, RUN_ID)
    await step_bar(before, SYMBOL, PRICE, timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    # a real restart re-reads the persisted rows: stamp them inside the scenario's day
    return [f.model_copy(update={"executed_at": D_2105}) for f in before.execution.get_all_fills()]


async def _resume(
    exchange: FakeCCXTExchange,
    fills: list[Fill],
    script: dict[int, str],
    clock: ManualClock,
    seed: ResumeDailySeed | Any,
    *,
    protective: bool = False,
) -> tuple[LiveStack, list[dict[str, Any]]]:
    with capture_logs() as logs:
        stack = await build_resumed_live_stack(
            exchange=exchange,
            strategy=_script(script),
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            initial_capital=INITIAL,
            run_id=RUN_ID,
            fills=fills,
            engine_config=_CONFIG,
            risk_params=_RISK,
            protective_mode=protective,
            clock=clock,
            daily_seed=seed,
        )
    await start_and_warmup(stack, RUN_ID)
    return stack, logs


def _daily_loss_rejections(logs: list[dict[str, Any]]) -> list[str]:
    # WP1.6: with the harness circuit breaker on (production parity, K-15) the same
    # daily-loss predicate drops the BUY one layer earlier (skip reason
    # ``circuit_breaker_daily_limit``); the gate is then never asked (single reason).
    gate = [
        r
        for e in logs
        if e["event"] == "live.signal_rejected"
        for r in e["reasons"]
        if "daily loss" in r.lower()
    ]
    cb = [
        e["skip_reason"]
        for e in logs
        if e["event"] == "trade_skip_logger.skip_recorded"
        and e.get("skip_reason") == "circuit_breaker_daily_limit"
    ]
    return gate + cb


def _snapshot(as_of: datetime, **hints: Any) -> ResumeSnapshot:
    """The ``ResumeSnapshot`` ``load_resume_snapshot`` would build for a run (no fills here:
    the harness feeds the fills to ``from_fills`` itself)."""
    return ResumeSnapshot(
        run_id=RUN_ID,
        initial_cash=INITIAL,
        fills=[],
        orders=[],
        peak_equity_hint=None,
        max_bar_index=-1,
        daily_hints_day=as_of.date(),
        run_started_at=_STARTED,
        **hints,
    )


def _seed(
    as_of: datetime,
    *,
    start: str | None,
    day_min: str | None,
    day_first: str | None,
) -> ResumeDailySeed:
    """The seed through the 1.5b object under test: ``ResumeSnapshot.daily_seed``."""
    seed = _snapshot(
        as_of,
        day_start_equity_hint=Decimal(start) if start is not None else None,
        day_min_equity_hint=Decimal(day_min) if day_min is not None else None,
        day_first_equity_hint=Decimal(day_first) if day_first is not None else None,
    ).daily_seed
    assert seed is not None
    return seed


# ---------------------------------------------------------------------------
# WP15-T-34a  loss today -> BUYs blocked for the rest of the day, SL fires, D+1 BUY fills
# ---------------------------------------------------------------------------


async def test_wp15_t34a_resumed_loss_day_blocks_buys_until_midnight_but_the_sl_fires(
    exchange: FakeCCXTExchange,
) -> None:
    fills = await _persisted_buy(exchange)
    clock = ManualClock(D_2305)
    # today's snapshot minimum 975 vs a 1 000 day start: worst -25 <= -20 (2% of 1 000)
    seed = _seed(D_2305, start="1000", day_min="975", day_first="1000")
    stack, logs = await _resume(exchange, fills, {0: "buy", 2: "buy"}, clock, seed)

    snap = stack.portfolio.risk_snapshot()
    assert snap.day_start_source == "snapshot"
    assert snap.nav_day_start == Decimal("1000") and snap.worst_nav_pnl == Decimal("-25")
    assert [e for e in logs if e["event"] == "resume.risk_state_seeded"]
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat  # the pre-restart entry survived

    # D 23:05 -- the scripted BUY is rejected by the daily-loss latch (flat mark: no new loss)
    with capture_logs() as bar_logs:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    assert _daily_loss_rejections(bar_logs), bar_logs
    assert [o["side"] for o in exchange.order_log] == ["buy"]

    # D 23:05 -- the stop-loss (70) still fires under the latch (exits are never blocked)
    await step_bar(stack, SYMBOL, Decimal("65"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    flat = stack.portfolio.get_position(SYMBOL)
    assert flat is not None and flat.is_flat

    # D+1 00:05 -- a new UTC day: the BUY fills
    clock.set(D1_0005)
    with capture_logs() as day2_logs:
        await step_bar(stack, SYMBOL, Decimal("65"), timeframe=TF)
    assert not _daily_loss_rejections(day2_logs), day2_logs
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell", "buy"]
    assert stack.portfolio.risk_snapshot().day_start_source == "roll"
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-34b  a depressed pre-midnight hint + a normal first-of-today snapshot
# ---------------------------------------------------------------------------


async def test_wp15_t34b_first_of_today_hardens_a_depressed_pre_midnight_hint(
    exchange: FakeCCXTExchange,
) -> None:
    fills = await _persisted_buy(exchange)
    clock = ManualClock(D1_0030)
    # the last pre-midnight snapshot was an outlier-low 850; the first one today is 1 000
    seed = _seed(D1_0030, start="850", day_min="1000", day_first="1000")
    stack, _ = await _resume(exchange, fills, {0: "buy"}, clock, seed)

    snap = stack.portfolio.risk_snapshot()
    assert snap.day_start_source == "snapshot"
    assert snap.nav_day_start == Decimal("1000"), "the day start is max(hint, first-of-today)"
    assert stack.portfolio._day_start_pending is False

    # a real decline (-15% on a 15% position = -22.5 >= 20): the scripted BUY is rejected
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("85"), timeframe=TF)
    assert _daily_loss_rejections(logs), logs
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    assert stack.portfolio.risk_snapshot().worst_nav_pnl <= Decimal("-20")

    # AC-04: the SL still fills under the latch
    await step_bar(stack, SYMBOL, Decimal("65"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    await stack.engine.stop()


async def test_wp15_t34b_control_without_first_of_today_the_cushion_returns(
    exchange: FakeCCXTExchange,
) -> None:
    """Pins what ``day_first_equity`` is FOR: with the depressed hint alone (and a
    today-minimum above it), the same real decline is cushioned and the BUY fills."""
    fills = await _persisted_buy(exchange)
    clock = ManualClock(D1_0030)
    seed = _seed(D1_0030, start="850", day_min="1000", day_first=None)
    stack, _ = await _resume(exchange, fills, {0: "buy"}, clock, seed)
    assert stack.portfolio.risk_snapshot().nav_day_start == Decimal("850")

    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("85"), timeframe=TF)
    assert not _daily_loss_rejections(logs), logs
    assert [o["side"] for o in exchange.order_log] == ["buy", "buy"]
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-34c  no snapshots yet today -> the first post-resume mark re-bases
# ---------------------------------------------------------------------------


async def test_wp15_t34c_no_snapshot_today_first_mark_rebases_the_provisional_start(
    exchange: FakeCCXTExchange,
) -> None:
    fills = await _persisted_buy(exchange)
    clock = ManualClock(D1_0030)
    seed = _seed(D1_0030, start="900", day_min=None, day_first=None)
    stack, _ = await _resume(exchange, fills, {2: "buy"}, clock, seed)

    assert stack.portfolio.risk_snapshot().nav_day_start == Decimal("900")
    assert stack.portfolio._day_start_pending is True

    await step_bar(stack, SYMBOL, PRICE, timeframe=TF)  # the first observation of the day
    after_first_mark = stack.portfolio.risk_snapshot()
    assert after_first_mark.nav_day_start > Decimal("990"), "re-based to the first mark's NAV"
    assert stack.portfolio._day_start_pending is False

    await step_bar(stack, SYMBOL, Decimal("85"), timeframe=TF)  # the real decline
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal("85"), timeframe=TF)  # scripted BUY (index 2)
    assert _daily_loss_rejections(logs), logs
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    await stack.engine.stop()


# ---------------------------------------------------------------------------
# WP15-T-34d  an invalid seed never aborts a resume, a protective resume included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protective", [False, True], ids=["normal", "protective"])
async def test_wp15_t34d_invalid_seed_never_aborts_the_resume_and_the_sl_still_fires(
    exchange: FakeCCXTExchange, protective: bool
) -> None:
    fills = await _persisted_buy(exchange)
    clock = ManualClock(D_2305)
    # SEC-208: a float day_first_equity with day_min None and a valid Decimal day start,
    # passed through the snapshot builder unmodified (no coercion anywhere)
    seed = _snapshot(
        D_2305,
        day_start_equity_hint=Decimal("1000"),
        day_min_equity_hint=None,
        day_first_equity_hint=1000.0,
    ).daily_seed
    assert seed is not None and seed.day_first_equity == 1000.0
    assert isinstance(seed.day_first_equity, float)
    stack, logs = await _resume(exchange, fills, {0: "buy"}, clock, seed, protective=protective)

    # the resume completed; the day state is unknown (fail closed)
    events = [e for e in logs if e["event"] == "resume.daily_state_unknown"]
    assert len(events) == 1
    assert events[0]["reason"] == "seed_invalid" and events[0]["field"] == "day_first_equity"
    snap = stack.portfolio.risk_snapshot()  # does not raise
    assert snap.day_start_source == "unknown" and snap.state_unknown
    assert isinstance(snap.nav_day_start, Decimal)
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat

    # the scripted BUY never reaches the exchange (daily-loss gate / protective filter)
    with capture_logs() as bar_logs:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy"]
    if not protective:
        assert _daily_loss_rejections(bar_logs), bar_logs  # blocked by the unknown-state gate

    # the stop-loss fires on the resumed position (mark + SELL fill paths work)
    await step_bar(stack, SYMBOL, Decimal("65"), timeframe=TF)
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]
    flat = stack.portfolio.get_position(SYMBOL)
    assert flat is not None and flat.is_flat
    after = stack.portfolio.risk_snapshot()
    assert isinstance(after.nav, Decimal) and isinstance(after.worst_nav_pnl, Decimal)
    await stack.engine.stop()
