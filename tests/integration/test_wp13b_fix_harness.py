"""
tests/integration/test_wp13b_fix_harness.py
---------------------------------------------
WP1.3b-a FIX DELTA (``reports/vp2-wp1.3b/final-synthesis-1.3b-a.md`` section 7) through the
REAL live stack -- ``StrategyEngine`` + ``LiveExecutionEngine`` + ``PortfolioAccounting`` +
``CCXTMarketDataService`` over ``FakeCCXTExchange`` (the helpers of
``test_wp13b_anchor_harness.py`` are reused).

    FX-01  period 14 under an exchange page cap < W: the exact window can never be verified
    FX-03  (a) 3 attempts then `entry_unverified`   (b) one failure then final `entry`
           (c) SEC P2 liveness: SL armed bar after a 1-bar outage / under a permanent outage
           (d) SEC P1: resume fallback is the interim, recovers to the live stop
           (e) ARB-13ba-01: a deferred anchor targets the rule bar (interim, then unverified)
    FX-04  a closed lot spends no fetch and logs nothing
    FX-05  bracket / trailing restore isolation; a malformed fetch result
    FX-06  `bracket_exit.lot_unprotected` and its metric
    FX-08  `trailing_stop.seed_partial` for an empty warmup window
    AC-13B-18  the delta's new events carry `error_type` only, never exception text
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import ccxt
import pytest
from structlog.testing import capture_logs

from common.types import TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.integration.test_wp13b_anchor_harness import (
    _LEAKY_ERROR_TEXT,
    ATR_CFG,
    ATR_SL_ONLY,
    FIXED_CFG,
    INITIAL,
    RUN_ID,
    SYMBOL,
    _fail_since_fetches,
    _gapped_entry,
    _live,
    _new_site_logs,
    _old_lot,
    _persisted,
    _resume,
    _sells,
    _set_now_after_close,
    _step,
    _wavy,
)
from trading.models import Fill, Order
from trading.strategy_engine import StrategyEngine, atr_for_bars

TRAIL = 0.20


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
    patch_exchange_factory(monkeypatch, ex)
    return ex


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Outage:
    """Switchable OHLCV outage on the fake: ``warmup_down`` fails the ``since=None`` calls,
    ``since_down`` the ``since=...`` calls (every raw call, so the service's own retries
    fail too).  Engine attempts are counted by ``engine.exit_window_fetch_failed`` logs."""

    def __init__(self, ex: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch) -> None:
        self.warmup_down = False
        self.since_down = False
        original = ex.fetch_ohlcv

        async def _flaky(
            symbol: str,
            timeframe: str = "1h",
            since: int | None = None,
            limit: int | None = None,
            params: dict[str, Any] | None = None,
        ) -> list[list[Any]]:
            if since is None and self.warmup_down:
                raise ccxt.NetworkError("simulated warmup outage")
            if since is not None and self.since_down:
                raise ccxt.NetworkError("simulated since-fetch outage")
            return await original(symbol, timeframe, since, limit, params)

        monkeypatch.setattr(ex, "fetch_ohlcv", _flaky)


async def _resume_blind(
    ex: FakeCCXTExchange,
    cfg: dict[str, object],
    fills: list[Fill],
    orders: list[Order],
    outage: _Outage,
) -> LiveStack:
    """A resume whose WARMUP fetch fails: the engine starts with an empty bar window."""
    ex.pad_forming_bar = False
    stack = await build_resumed_live_stack(
        exchange=ex,
        strategy=ScriptedSequenceStrategy("wp13b-fix", {"script": {}}),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id=RUN_ID,
        fills=fills,
        orders=orders,
        engine_config=cfg,
    )
    outage.warmup_down = True
    await start_and_warmup(stack, RUN_ID)
    outage.warmup_down = False
    ex.pad_forming_bar = True
    return stack


async def _poll(stack: LiveStack) -> None:
    """Process the newest closed bar without pushing a new one."""
    _set_now_after_close(stack.exchange)
    await stack.engine._poll_and_process()


def _anchor(stack: LiveStack) -> Any:
    return stack.engine._bracket_exit.get_anchor(SYMBOL)  # type: ignore[union-attr]


def _events(logs: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == name]


def _fetch_failures(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _events(logs, "engine.exit_window_fetch_failed")


@pytest.fixture
def metric_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str]]]:
    from common.metrics import metrics as registry

    seen: list[tuple[str, dict[str, str]]] = []
    original = registry.increment

    def _spy(name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        seen.append((name, dict(labels or {})))
        original(name, value, labels)

    monkeypatch.setattr(registry, "increment", _spy)
    return seen


# ---------------------------------------------------------------------------
# FX-01 (f)  an exchange page cap below W: the exact window can never be verified
# ---------------------------------------------------------------------------


async def test_fx01f_page_cap_below_w_live_entry_ends_unverified_after_three_attempts(
    exchange: FakeCCXTExchange,
) -> None:
    """Period 14 (W = 85) under ``max_ohlcv_limit=60`` models the ccxt cap the policy bound
    prevents (kills M24: the short-window check).  Entry `entry_provisional`; 3 passes ->
    3 `IncompleteWindow` failures (got=60, expected=85); then `entry_unverified` with the
    interim levels unchanged."""
    exchange.max_ohlcv_limit = 60
    _wavy(exchange)
    stack = await _live(exchange, ATR_CFG)
    await _step(stack, 101)
    prov = _anchor(stack)
    assert prov is not None and prov.source == "entry_provisional" and not prov.final
    assert prov.stop_price is not None

    with capture_logs() as logs:
        for _ in range(3):
            await _step(stack, 100)
    failures = _fetch_failures(logs)
    assert len(failures) == 3
    assert all(
        e["error_type"] == "IncompleteWindow" and e["got"] == 60 and e["expected"] == 85
        for e in failures
    )
    final = _anchor(stack)
    assert final.final and final.source == "entry_unverified"
    assert (final.stop_price, final.take_profit_price, final.atr) == (
        prov.stop_price,
        prov.take_profit_price,
        prov.atr,
    )


async def test_fx01f_page_cap_below_w_resume_is_interim_then_final_after_two_more_passes(
    exchange: FakeCCXTExchange,
) -> None:
    exchange.max_ohlcv_limit = 60
    _wavy(exchange)
    live = await _live(exchange, ATR_CFG)
    await _step(live, 101)
    fills, orders = _persisted(live)

    resumed = await _resume(exchange, ATR_CFG, fills, orders)
    first = _anchor(resumed)
    assert first.source == "resume_fallback_current_atr" and not first.final
    assert first.stop_price is not None
    with capture_logs() as logs:
        await _step(resumed, 100)  # attempt 2
        assert not _anchor(resumed).final
        await _step(resumed, 100)  # attempt 3
    final = _anchor(resumed)
    assert final.final and final.source == "resume_fallback_current_atr"
    assert (final.stop_price, final.atr) == (first.stop_price, first.atr)
    assert len(_fetch_failures(logs)) == 2
    assert _events(logs, "bracket_exit.resume_fallback_final")[0]["log_level"] == "error"


# ---------------------------------------------------------------------------
# FX-03 (a)/(b)  the attempt bound on an entry-origin pending anchor
# ---------------------------------------------------------------------------


async def test_fx03a_permanent_since_failure_makes_exactly_three_attempts_over_five_passes(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills the 'attempt cap removed' mutant (the M11 analogue): a gapped entry whose exact
    window can never be fetched -> 3 engine attempts (one per pass), `entry_unverified` with
    the interim levels, and NO attempt on passes 4-5."""
    stack = await _gapped_entry(exchange)
    prov = _anchor(stack)
    assert prov.source == "entry_provisional" and not prov.final
    _fail_since_fetches(exchange, monkeypatch)

    per_pass: list[int] = []
    sources: list[str] = []
    all_logs: list[dict[str, Any]] = []
    for _ in range(5):
        with capture_logs() as logs:
            await _step(stack, 100)
        per_pass.append(len(_fetch_failures(logs)))
        sources.append(_anchor(stack).source)
        all_logs += logs
    assert per_pass == [1, 1, 1, 0, 0]
    assert sources == ["entry_provisional", "entry_provisional"] + ["entry_unverified"] * 3
    final = _anchor(stack)
    assert final.final
    assert (final.stop_price, final.take_profit_price) == (prov.stop_price, prov.take_profit_price)
    unverified = _events(all_logs, "bracket_exit.entry_unverified")
    assert len(unverified) == 1 and unverified[0]["log_level"] == "error"
    assert unverified[0]["attempts"] == 3 and unverified[0]["error_type"] == "MarketDataError"
    assert "SECRET" not in repr(unverified)


async def test_fx03b_one_failed_attempt_then_the_final_entry_anchor_equals_the_resume_anchor(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = await _gapped_entry(exchange)
    outage = _Outage(exchange, monkeypatch)
    outage.since_down = True
    with capture_logs() as logs:
        await _step(stack, 100)  # attempt 1 fails
    assert len(_fetch_failures(logs)) == 1
    assert not _anchor(stack).final
    outage.since_down = False
    await _step(stack, 100)  # attempt 2 succeeds on pass 2
    final = _anchor(stack)
    assert final.final and final.source == "entry"

    fills, orders = _persisted(stack)
    resumed = await _resume(exchange, ATR_CFG, fills, orders)
    again = _anchor(resumed)
    assert again.source == "resume_recomputed" and again.final
    assert (again.atr, again.stop_price, again.take_profit_price, again.anchor_bar_ts) == (
        final.atr,
        final.stop_price,
        final.take_profit_price,
        final.anchor_bar_ts,
    )


# ---------------------------------------------------------------------------
# FX-03 (c)  SEC P2: liveness of the stop after a resume with the warmup failing
# ---------------------------------------------------------------------------


async def _p2_lot(
    exchange: FakeCCXTExchange,
) -> tuple[Any, list[Fill], list[Order]]:
    _wavy(exchange)
    live = await _live(exchange, ATR_SL_ONLY)  # ATR stop-loss only, NO trailing stop
    await _step(live, 101)
    anchor = _anchor(live)
    assert anchor.source == "entry" and anchor.final and anchor.stop_price is not None
    fills, orders = _persisted(live)
    return anchor, fills, orders


async def test_fx03c_p2_stop_is_armed_on_the_second_pass_after_a_one_pass_outage(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, metric_spy: list[Any]
) -> None:
    live_anchor, fills, orders = await _p2_lot(exchange)
    outage = _Outage(exchange, monkeypatch)
    stack = await _resume_blind(exchange, ATR_SL_ONLY, fills, orders, outage)
    assert stack.engine._bar_windows[SYMBOL] == []  # the warmup really failed

    outage.since_down = True
    with capture_logs() as pass1:
        await _poll(stack)  # pass 1: attempt 1 fails, no ATR estimate possible yet
    assert _anchor(stack).stop_price is None  # no stop yet (HEAD: none before p+1 bars either)
    assert len(_events(pass1, "bracket_exit.lot_unprotected")) == 1

    outage.since_down = False
    await _step(stack, 100)  # pass 2: attempt 2 succeeds
    armed = _anchor(stack)
    assert armed.final and armed.source == "resume_recomputed"
    assert armed.stop_price == live_anchor.stop_price  # Decimal ==, the live level
    assert armed.atr == live_anchor.atr

    await _step(stack, 60)  # crash below the stop
    assert len(_sells(exchange)) == 1


async def test_fx03c_p2_permanent_outage_arms_the_stop_at_pass_p_plus_1_like_head(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, metric_spy: list[Any]
) -> None:
    """HEAD armed the ATR stop as soon as more than ``period`` (14) bars existed, i.e. at
    pass 15.  The patch as submitted waited 84 bars.  Now: pass 15 (parity), as the tagged
    fallback `resume_fallback_current_atr`, final, with an error log; FX-06 makes the 14
    unprotected passes loud (error logs on bars 1 and 11, 14 metric increments)."""
    live_anchor, fills, orders = await _p2_lot(exchange)
    outage = _Outage(exchange, monkeypatch)
    stack = await _resume_blind(exchange, ATR_SL_ONLY, fills, orders, outage)
    outage.since_down = True

    all_logs: list[dict[str, Any]] = []
    stop_by_pass: list[Any] = []
    for k in range(1, 16):
        with capture_logs() as logs:
            if k == 1:
                await _poll(stack)
            else:
                await _step(stack, 100 + (k % 2) * 2)
        all_logs += logs
        stop_by_pass.append(_anchor(stack).stop_price)
    assert [s is None for s in stop_by_pass] == [True] * 14 + [False]  # armed at pass 15

    armed = _anchor(stack)
    assert armed.final and armed.source == "resume_fallback_current_atr"
    newest = atr_for_bars(stack.engine._bar_windows[SYMBOL][-85:], 14)
    assert newest is not None
    # I-2: never looser than HEAD's per-bar stop on that bar
    assert armed.stop_price >= armed.entry_basis - Decimal("1.5") * newest
    final_log = _events(all_logs, "bracket_exit.resume_fallback_final")
    assert len(final_log) == 1 and final_log[0]["log_level"] == "error"
    assert final_log[0]["attempts"] == 3

    # FX-06: loud while unprotected, quiet once armed
    unprotected = _events(all_logs, "bracket_exit.lot_unprotected")
    assert [e["bars_unprotected"] for e in unprotected] == [1, 11]
    assert all(e["log_level"] == "error" for e in unprotected)
    assert sum(1 for n, _ in metric_spy if n == "lot_unprotected_bars_total") == 14

    await _step(stack, 60)
    assert len(_sells(exchange)) == 1  # exactly one SELL
    assert live_anchor.stop_price is not None


# ---------------------------------------------------------------------------
# FX-03 (d)  SEC P1: the resume fallback is the interim, not the (looser) current ATR
# ---------------------------------------------------------------------------


async def _p1_lot(
    exchange: FakeCCXTExchange,
) -> tuple[Any, list[Fill], list[Order]]:
    """Entry ATR ~1.93, then 20 violent bars (ATR rises to ~9): SL only, no trailing."""
    _wavy(exchange)
    live = await _live(exchange, ATR_SL_ONLY)
    await _step(live, 101)
    anchor = _anchor(live)
    for i in range(20):
        await _step(live, 112 if i % 2 else 100)  # stays above the stop (98.7)
    assert _anchor(live) == anchor
    assert not live.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]
    fills, orders = _persisted(live)
    return anchor, fills, orders


async def test_fx03d_p1_recovery_after_the_restore_attempt_ends_at_the_live_stop(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_anchor, fills, orders = await _p1_lot(exchange)
    outage = _Outage(exchange, monkeypatch)
    outage.since_down = True
    resumed = await _resume(exchange, ATR_SL_ONLY, fills, orders)
    interim = _anchor(resumed)
    assert interim.source == "resume_fallback_current_atr" and not interim.final
    outage.since_down = False
    with capture_logs() as logs:
        await _step(resumed, 100)  # attempt 2 succeeds
    final = _anchor(resumed)
    assert final.final and final.source == "resume_recomputed"
    assert final.stop_price == live_anchor.stop_price  # == live.stop_price
    assert final.atr == live_anchor.atr
    assert len(_events(logs, "bracket_exit.anchor_finalized")) == 1


async def test_fx03d_p1_permanent_outage_keeps_a_stop_no_looser_than_heads_and_exits(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The patch as submitted anchored on the CURRENT ATR (~8): stop 89.6 vs live 98.7, and a
    close in between did not exit.  Now the interim is min(rule-bar window, newest)."""
    live_anchor, fills, orders = await _p1_lot(exchange)
    outage = _Outage(exchange, monkeypatch)
    outage.since_down = True
    resumed = await _resume(exchange, ATR_SL_ONLY, fills, orders)
    first = _anchor(resumed)
    newest = atr_for_bars(resumed.engine._bar_windows[SYMBOL][-85:], 14)
    assert newest is not None and newest > live_anchor.atr * 2  # the market really got wilder
    assert first.stop_price >= first.entry_basis - Decimal("1.5") * newest  # >= HEAD's stop
    assert first.stop_price > live_anchor.stop_price - first.entry_basis / 100  # within 1% of live
    await _step(resumed, 100)
    await _step(resumed, 100)
    final = _anchor(resumed)
    assert final.final and final.source == "resume_fallback_current_atr"
    assert final.stop_price == first.stop_price
    await _step(resumed, 95)  # between the old fallback (89.6) and the live stop (98.7)
    assert len(_sells(exchange)) == 1


# ---------------------------------------------------------------------------
# FX-03 (e)  ARB-13ba-01: a deferred anchor targets the rule bar like every other anchor
# ---------------------------------------------------------------------------


async def test_fx03e_deferred_entry_anchor_gets_interim_levels_then_settles_as_unverified(
    exchange: FakeCCXTExchange,
) -> None:
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe="1h")
    cfg = {**ATR_SL_ONLY, "trailing_stop_pct": TRAIL}  # trailing keeps the ATR-less entry allowed
    stack = await _live(exchange, cfg)
    await _step(stack, 100)  # flat market: no ATR -> deferred, trailing stop only
    deferred = _anchor(stack)
    assert deferred.source == "entry_deferred" and not deferred.final
    assert deferred.stop_price is None

    with capture_logs() as logs:
        await _step(stack, 103)  # the first volatile bar
    names = [e["event"] for e in logs]
    assert "bracket_exit.interim_armed" in names and "bracket_exit.entry_unverified" in names
    assert names.index("bracket_exit.interim_armed") < names.index("bracket_exit.entry_unverified")
    armed = _events(logs, "bracket_exit.interim_armed")[0]
    assert armed["log_level"] == "warning" and armed["source"] == "entry_provisional"
    final = _anchor(stack)
    # the OLD behaviour finalised it as a silent `entry_deferred` on the newest window
    assert final.final and final.source == "entry_unverified"
    assert final.stop_price is not None and final.anchor_bar_ts is not None


# ---------------------------------------------------------------------------
# FX-04  a closed lot spends no fetch
# ---------------------------------------------------------------------------


async def test_fx04_a_flattened_lot_with_a_pending_anchor_costs_no_fetch_and_no_log(
    exchange: FakeCCXTExchange,
) -> None:
    stack = await _gapped_entry(exchange)
    prov = _anchor(stack)
    assert prov is not None and not prov.final  # pending, and its window needs a `since` fetch
    stack.risk_manager.trigger_kill_switch("stop_in_progress")
    result = await stack.engine.flatten("fx04")
    assert result.complete
    assert stack.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]

    before = len(exchange.ohlcv_calls)
    with capture_logs() as logs:
        await _step(stack, 100)
    assert [c for c in exchange.ohlcv_calls[before:] if c[2] is not None] == []
    names = {e["event"] for e in logs}
    assert not names & {
        "bracket_exit.entry_unverified",
        "bracket_exit.anchor_finalized",
        "bracket_exit.interim_armed",
        "engine.exit_window_fetch_failed",
    }
    assert stack.engine._anchor_fetch_attempts == {}


# ---------------------------------------------------------------------------
# FX-05  restore fault isolation and a malformed fetch result
# ---------------------------------------------------------------------------


BOTH_CFG: dict[str, object] = {**ATR_SL_ONLY, "trailing_stop_pct": TRAIL}


async def _both_run(exchange: FakeCCXTExchange) -> tuple[LiveStack, list[Fill], list[Order]]:
    """ATR stop-loss + trailing stop (no TP, so the 108 spike does not exit); one spike early
    on, outside the part of the history the warmup window will cover."""
    _wavy(exchange)
    stack = await _live(exchange, BOTH_CFG)
    await _step(stack, 101)
    for i in range(130):
        await _step(stack, 108 if i == 20 else 100 + (i % 2) * 2)
    fills, orders = _persisted(stack)
    return stack, fills, orders


async def test_fx05a_p3_a_failing_bracket_restore_does_not_skip_the_trailing_restore(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack, fills, orders = await _both_run(exchange)
    live_peak = stack.engine._trailing_stop.peak_prices[SYMBOL]  # type: ignore[union-attr]
    assert live_peak == Decimal("108")

    async def _boom(self: StrategyEngine, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated ATR maths fault")

    monkeypatch.setattr(StrategyEngine, "_resume_atr", _boom)
    with capture_logs() as logs:
        resumed = await _resume(exchange, BOTH_CFG, fills, orders)
    failed = _events(logs, "engine.exit_restore_failed")
    assert [(e["part"], e["error_type"]) for e in failed] == [("bracket", "RuntimeError")]
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] == live_peak  # type: ignore[union-attr]
    assert _anchor(resumed) is None  # the lazy `late` anchor protects the lot at the next bar


async def test_fx05b_a_failing_trailing_restore_seeds_heads_rule_and_says_so(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HEAD's WP1.8a rule: ``max(gross entry VWAP, max close of the warmup bars at/after the
    position's opened_at)`` -- here 102, NOT the whole-window max and not the live 108."""
    _stack, fills, orders = await _both_run(exchange)

    async def _boom(self: StrategyEngine, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated trailing restore fault")

    monkeypatch.setattr(StrategyEngine, "_resume_trailing_peak", _boom)
    with capture_logs() as logs:
        resumed = await _resume(exchange, BOTH_CFG, fills, orders)
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] == Decimal("102")  # type: ignore[union-attr]
    partial = _events(logs, "trailing_stop.seed_partial")
    assert len(partial) == 1
    assert partial[0]["log_level"] == "error" and partial[0]["reason"] == "restore_failed"
    assert [(e["part"], e["error_type"]) for e in _events(logs, "engine.exit_restore_failed")] == [
        ("trailing", "RuntimeError")
    ]
    anchor = _anchor(resumed)  # the bracket half was untouched
    assert anchor is not None and anchor.source == "resume_recomputed"


@pytest.mark.parametrize("garbage", [[None], None, "not a list", ({"a": 1},)])
async def test_fx05c_a_malformed_fetch_result_is_a_logged_fallback_never_an_exception(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, garbage: Any
) -> None:
    _live_stack, fills, orders = await _old_lot(exchange, ATR_CFG)
    original = CCXTMarketDataService.fetch_ohlcv

    async def _garbage(
        self: CCXTMarketDataService,
        symbol: str,
        timeframe: TimeFrame,
        since: Any = None,
        limit: int = 500,
    ) -> Any:
        if since is not None:
            return garbage
        return await original(self, symbol, timeframe, since, limit)

    monkeypatch.setattr(CCXTMarketDataService, "fetch_ohlcv", _garbage)
    with capture_logs() as logs:
        resumed = await _resume(exchange, ATR_CFG, fills, orders)
    bad = [e for e in _fetch_failures(logs) if e["error_type"] == "MalformedWindow"]
    assert bad, [e.get("error_type") for e in _fetch_failures(logs)]
    anchor = _anchor(resumed)
    assert anchor.source == "resume_fallback_current_atr" and not anchor.final
    assert anchor.stop_price is not None
    assert not _events(logs, "engine.exit_restore_failed")  # handled inside, not an escape


# ---------------------------------------------------------------------------
# FX-06  a fixed-mode lot never logs `lot_unprotected`
# ---------------------------------------------------------------------------


async def test_fx06_a_fixed_stop_lot_and_a_trailing_protected_lot_never_log_unprotected(
    exchange: FakeCCXTExchange,
) -> None:
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe="1h")
    fixed = await _live(exchange, FIXED_CFG)
    with capture_logs() as logs:
        await _step(fixed, 100)
        for _ in range(12):
            await _step(fixed, 100)
    assert not fixed.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]
    assert _events(logs, "bracket_exit.lot_unprotected") == []


async def test_fx06_a_deferred_atr_lot_protected_by_the_trailing_stop_is_not_unprotected(
    exchange: FakeCCXTExchange,
) -> None:
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe="1h")
    stack = await _live(exchange, {**ATR_SL_ONLY, "trailing_stop_pct": TRAIL})
    with capture_logs() as logs:
        await _step(stack, 100)  # deferred anchor, no ATR at all
        assert _anchor(stack).stop_price is None
        await _step(stack, 100)
    assert _events(logs, "bracket_exit.lot_unprotected") == []


async def test_cr201_a_take_profit_only_bracket_is_never_reported_unprotected(
    exchange: FakeCCXTExchange, metric_spy: list[Any]
) -> None:
    """WP13BA-CR-201: a TP-only bracket has no stop-loss leg by design (the operator's
    choice; live needs the downside waiver), so it is not an unprotected-lot incident."""
    exchange.seed_flat_bars(SYMBOL, count=100, price=Decimal("100"), timeframe="1h")
    tp_only = await _live(exchange, {"bracket_mode": "fixed", "bracket_take_profit_pct": 0.50})
    with capture_logs() as logs:
        await _step(tp_only, 100)
        for _ in range(12):
            await _step(tp_only, 100)
    assert not tp_only.portfolio.get_position(SYMBOL).is_flat  # type: ignore[union-attr]
    anchor = _anchor(tp_only)
    assert anchor.final and anchor.stop_price is None and anchor.take_profit_price is not None
    assert _events(logs, "bracket_exit.lot_unprotected") == []
    assert not [n for n, _ in metric_spy if n == "lot_unprotected_bars_total"]


# ---------------------------------------------------------------------------
# FX-08  an empty warmup window is reported, not silent
# ---------------------------------------------------------------------------


async def test_fx08_resume_with_an_empty_warmup_window_logs_seed_partial_and_seeds_hwm0(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch, metric_spy: list[Any]
) -> None:
    _wavy(exchange)
    live = await _live(exchange, {"trailing_stop_pct": TRAIL})
    await _step(live, 101)
    hwm0 = live.engine._trailing_stop.peak_prices[SYMBOL]  # type: ignore[union-attr]
    assert hwm0 == Decimal("101")
    fills, orders = _persisted(live)

    outage = _Outage(exchange, monkeypatch)
    with capture_logs() as logs:
        resumed = await _resume_blind(exchange, {"trailing_stop_pct": TRAIL}, fills, orders, outage)
    partial = _events(logs, "trailing_stop.seed_partial")
    assert len(partial) == 1
    assert partial[0]["log_level"] == "warning" and partial[0]["reason"] == "no_warmup_bars"
    assert resumed.engine._trailing_stop.peak_prices[SYMBOL] == hwm0  # type: ignore[union-attr]
    assert any(n == "trailing_seed_partial_total" for n, _ in metric_spy)


# ---------------------------------------------------------------------------
# AC-13B-18  log hygiene of the delta's new events
# ---------------------------------------------------------------------------


async def test_ac13b18_entry_attempt_cycle_logs_carry_no_exception_text(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = await _gapped_entry(exchange)
    _fail_since_fetches(exchange, monkeypatch, _LEAKY_ERROR_TEXT)
    with capture_logs() as logs:
        for _ in range(3):
            await _step(stack, 100)
    mine = _new_site_logs(logs)
    names = {e["event"] for e in mine}
    assert {
        "engine.exit_window_fetch_failed",
        "bracket_exit.entry_unverified",
        "bracket_exit.anchor_finalized",
    } <= names
    rendered = repr(mine)
    assert "SECRET" not in rendered and "api.exchange.example" not in rendered


async def test_ac13b18_resume_attempt_cycle_logs_carry_no_exception_text(
    exchange: FakeCCXTExchange, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_stack, fills, orders = await _old_lot(exchange, ATR_CFG)
    _fail_since_fetches(exchange, monkeypatch, _LEAKY_ERROR_TEXT)
    with capture_logs() as logs:
        resumed = await _resume(exchange, ATR_CFG, fills, orders)
        await _step(resumed, 100)
        await _step(resumed, 100)
    mine = _new_site_logs(logs)
    names = {e["event"] for e in mine}
    assert {
        "bracket_exit.interim_armed",
        "bracket_exit.resume_fallback_final",
        "bracket_exit.anchor_finalized",
    } <= names
    rendered = repr(mine)
    assert "SECRET" not in rendered and "api.exchange.example" not in rendered
