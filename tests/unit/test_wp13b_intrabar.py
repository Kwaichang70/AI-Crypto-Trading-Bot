"""
tests/unit/test_wp13b_intrabar.py
-----------------------------------
WP1.3b-b (live intra-bar exit check) -- pure and engine-level unit coverage.
WP13B-T-34..T-39, T-44..T-46, T-67 and the 1.3b-a carry-forwards routed to this WP
(CF-13b-22 streak, CF-13b-23 metric label, CF-13b-24 lot order ids, CF-13b-27
origin tag).  The real-stack scenarios (T-40..T-43, T-47..T-66, T-68) live in
``tests/integration/test_wp13b_intrabar_harness.py``.

    T-34  resolver table + Settings bounds
    T-35  W8 only when the check is NOT effective; now also 1h
    T-36  engine construction bounds
    T-37  the bar-poll timing is identical with the feature on or off
    T-38  task cadence, fast retry, stop interrupts
    T-39  a tick exception does not kill the task; task death does not touch the bar loop
    T-44  AST pins: the tick is read-only (CF-04 / CF-30) and never holds the lock across I/O
    T-45  trailing intra-bar mode
    T-46  bracket intra-bar mode
    T-67  fetch_ticker_quote: no cache, no retry, hygiene
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import textwrap
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock
from uuid import uuid4

import ccxt
import pytest
from structlog.testing import capture_logs

from api.config import Settings
from common.types import OrderSide, OrderStatus, OrderType, RunMode, TimeFrame
from data.market_data import (
    BaseMarketDataService,
    DataNotAvailableError,
    MarketDataError,
    RateLimitError,
    TickerQuote,
)
from data.services.ccxt_market_data import CCXTMarketDataService
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.live_harness import IntrabarClock
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading import strategy_engine as se
from trading.bracket_exit import BracketExitManager
from trading.exit_config import (
    EXIT_CHECK_INTERVAL_MAX_S,
    EXIT_CHECK_INTERVAL_MIN_S,
    ExitCheckResolution,
    ExitConfigError,
    resolve_exit_check_interval,
    resolve_run_exit_check,
    validate_run_exit_config,
)
from trading.models import Fill, Order, Position
from trading.portfolio import PortfolioAccounting
from trading.strategy import BaseStrategy
from trading.strategy_engine import StrategyEngine
from trading.trailing_stop import TrailingStopManager

SYM = "BTC/EUR"
LOT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# T-34  resolver + Settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "mode", "tf_s", "smoke", "protective", "expected", "reason"),
    [
        (300, "live", 3600, False, True, 300, "enabled"),
        (300, RunMode.LIVE, 3600, False, True, 300, "enabled"),
        (300, "live", 14400, False, True, 300, "enabled"),
        (60, "live", 300, False, True, 60, "enabled"),  # 5m with a 60 s interval is fine
        (300.0, "live", 3600, False, True, 300, "enabled"),  # an integral float
        (300, "live", 300, False, True, None, "interval_ge_timeframe"),  # 5m / 300
        (300, "live", 60, False, True, None, "interval_ge_timeframe"),  # 1m
        (3600, "live", 3600, False, True, None, "interval_ge_timeframe"),
        (300, "paper", 3600, False, True, None, "not_live"),
        (300, "backtest", 3600, False, True, None, "not_live"),
        (300, "live", 3600, True, True, None, "smoke_run"),
        (60, "live", 300, True, True, None, "smoke_run"),  # smoke 5m / 60 s: forced off
        (0, "live", 3600, False, True, None, "disabled"),
        (300, "live", 3600, False, False, None, "no_protective_exit"),
    ],
)
def test_t34_resolver_table(
    configured: object,
    mode: object,
    tf_s: int,
    smoke: bool,
    protective: bool,
    expected: int | None,
    reason: str,
) -> None:
    got = resolve_exit_check_interval(
        configured,
        mode=mode,
        timeframe_s=tf_s,
        is_smoke=smoke,
        has_protective_exit=protective,  # type: ignore[arg-type]
    )
    assert got == ExitCheckResolution(expected, reason)


@pytest.mark.parametrize(
    "bad",
    [59, 3601, -1, 12.5, float("nan"), float("inf"), True, False, "300", None, [300], 1],
)
def test_t34_resolver_rejects_out_of_range_and_non_integral_values(bad: object) -> None:
    with pytest.raises(ExitConfigError) as exc:
        resolve_exit_check_interval(
            bad, mode="live", timeframe_s=3600, is_smoke=False, has_protective_exit=True
        )
    assert exc.value.code == "invalid_exit_config"
    assert exc.value.issues[0].field == "live_exit_check_interval_s"


def test_t34_a_bad_setting_is_an_error_in_every_mode() -> None:
    with pytest.raises(ExitConfigError):
        resolve_exit_check_interval(
            59, mode="paper", timeframe_s=3600, is_smoke=True, has_protective_exit=False
        )


def test_t34_bounds_are_the_documented_ones() -> None:
    assert (EXIT_CHECK_INTERVAL_MIN_S, EXIT_CHECK_INTERVAL_MAX_S) == (60, 3600)


def test_t34_run_level_resolver_derives_the_run_facts() -> None:
    sl = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    tp_only = {"bracket_mode": "fixed", "bracket_take_profit_pct": 0.10}

    def run(**kw: Any) -> ExitCheckResolution:
        args: dict[str, Any] = {
            "mode": RunMode.LIVE,
            "timeframe": TimeFrame.ONE_HOUR,
            "strategy_name": "momentum_breakout",
            "bracket": sl,
            "trailing_stop_pct": None,
        }
        args.update(kw)
        return resolve_run_exit_check(300, **args)

    assert run().interval_s == 300
    assert run(trailing_stop_pct=0.1, bracket=None).interval_s == 300
    assert run(bracket=tp_only).reason == "no_protective_exit"  # TP is not a downside exit
    assert run(bracket=None).reason == "no_protective_exit"
    assert run(bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 9}).reason == (
        "no_protective_exit"
    )  # unparsable -> no managers
    assert run(strategy_name="smoke_roundtrip").reason == "smoke_run"
    assert run(strategy_name="Smoke-Roundtrip").reason == "smoke_run"
    assert run(timeframe=TimeFrame.FIVE_MINUTES).reason == "interval_ge_timeframe"
    assert run(timeframe="5m").reason == "interval_ge_timeframe"
    assert run(timeframe="weird").reason == "interval_ge_timeframe"  # unknown tf: off
    assert run(exit_config_waived=True).reason == "no_protective_exit"  # protective resume
    assert run(mode=RunMode.PAPER).reason == "not_live"


_DB = "postgresql+asyncpg://u:p@h:5432/d"


def test_t34_settings_default_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVE_EXIT_CHECK_INTERVAL_S", raising=False)
    assert Settings(database_url=_DB).live_exit_check_interval_s == 300  # ON by default
    for ok in (0, 60, 300, 3600):
        assert (
            Settings(database_url=_DB, live_exit_check_interval_s=ok).live_exit_check_interval_s
            == ok
        )
    for bad in (59, 3601, -1, 12.5, True, "abc", "12.5", float("nan")):
        with pytest.raises(ValueError, match="live_exit_check_interval_s"):
            Settings(database_url=_DB, live_exit_check_interval_s=bad)


def test_t34_settings_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_EXIT_CHECK_INTERVAL_S", "120")
    assert Settings(database_url=_DB).live_exit_check_interval_s == 120
    monkeypatch.setenv("LIVE_EXIT_CHECK_INTERVAL_S", "10")
    with pytest.raises(ValueError, match="live_exit_check_interval_s"):
        Settings(database_url=_DB)


# ---------------------------------------------------------------------------
# T-35  W8
# ---------------------------------------------------------------------------


class _NoExitRequired(BaseStrategy):
    requires_exit_manager: ClassVar[bool] = False

    def on_bar(self, bars: Any, *, mtf_context: Any = None) -> list[Any]:
        return []

    @property
    def min_bars_required(self) -> int:
        return 1


def _w8(timeframe: str, *, intrabar: bool, mode: RunMode = RunMode.LIVE) -> bool:
    verdict = validate_run_exit_config(
        _NoExitRequired,
        bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
        trailing_stop_pct=None,
        mode=mode,
        allow_pyramiding=None,
        timeframe=timeframe,
        intrabar_exit_checks=intrabar,
    )
    return any(w.code == "bar_close_exit_checks_only" for w in verdict.warnings)


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d", "1w"])
def test_t35_w8_is_shown_when_the_intrabar_check_is_not_effective_now_also_for_1h(
    timeframe: str,
) -> None:
    assert _w8(timeframe, intrabar=False) is True


@pytest.mark.parametrize("timeframe", ["1h", "4h", "1d", "1w"])
def test_t35_w8_is_not_shown_when_the_intrabar_check_is_effective(timeframe: str) -> None:
    assert _w8(timeframe, intrabar=True) is False


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "30m"])
def test_t35_w8_never_applies_to_short_timeframes(timeframe: str) -> None:
    assert _w8(timeframe, intrabar=False) is False


def test_t35_w8_is_live_only_and_the_default_is_the_conservative_one() -> None:
    assert _w8("1h", intrabar=False, mode=RunMode.PAPER) is False
    verdict = validate_run_exit_config(
        _NoExitRequired,
        bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
        trailing_stop_pct=None,
        mode=RunMode.LIVE,
        allow_pyramiding=None,
        timeframe="4h",
    )
    assert any(w.code == "bar_close_exit_checks_only" for w in verdict.warnings)
    text = next(w.message for w in verdict.warnings if w.code == "bar_close_exit_checks_only")
    assert "1.3b" not in text  # the stale "not detected until 1.3b" wording is gone


# WP13BB-CR-02: the router helper that feeds W8 at create / promote / resume
_FIXED_SL = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}


@pytest.mark.parametrize(
    ("setting", "timeframe", "strategy", "expected"),
    [
        (300, "1h", "wp13bb", True),  # live, 1h, fixed SL, default setting
        (0, "1h", "wp13bb", False),  # switched off
        (300, "5m", "wp13bb", False),  # interval >= timeframe
        (300, "1h", "smoke_roundtrip", False),  # smoke is forced off
    ],
)
def test_cr02_router_w8_flag_follows_the_effective_check(
    monkeypatch: pytest.MonkeyPatch, setting: int, timeframe: str, strategy: str, expected: bool
) -> None:
    import api.config as api_config
    from api.routers.runs import _intrabar_exit_checks_effective

    monkeypatch.setattr(
        api_config, "get_settings", lambda: SimpleNamespace(live_exit_check_interval_s=setting)
    )
    got = _intrabar_exit_checks_effective(
        mode=RunMode.LIVE,
        timeframe=timeframe,
        strategy_name=strategy,
        bracket=_FIXED_SL,
        trailing_stop_pct=None,
    )
    assert got is expected


def test_cr02_router_w8_flag_is_false_when_the_settings_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.config as api_config
    from api.routers.runs import _intrabar_exit_checks_effective

    def _boom() -> Any:
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(api_config, "get_settings", _boom)
    assert (
        _intrabar_exit_checks_effective(
            mode=RunMode.LIVE,
            timeframe="1h",
            strategy_name="wp13bb",
            bracket=_FIXED_SL,
            trailing_stop_pct=None,
        )
        is False
    )


# ---------------------------------------------------------------------------
# engine helpers (a real StrategyEngine over light doubles)
# ---------------------------------------------------------------------------


class _QuoteService:
    """Duck-typed market data double with ticker quotes."""

    supports_ticker_quotes = True

    def __init__(self) -> None:
        self.quotes: list[Any] = []
        self.calls = 0

    async def fetch_ticker_quote(self, symbol: str) -> Any:
        self.calls += 1
        out = self.quotes.pop(0) if self.quotes else TickerQuote(symbol, None, None, None, None)
        if isinstance(out, Exception):
            raise out
        return out


class _PlainService:
    supports_ticker_quotes = False


def _engine(
    *,
    config: dict[str, Any] | None = None,
    mode: RunMode = RunMode.LIVE,
    market_data: Any = None,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
) -> StrategyEngine:
    cfg = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    cfg.update(config or {})
    return StrategyEngine(
        strategies=[ScriptedSequenceStrategy("t", {"script": {}})],
        execution_engine=MagicMock(),
        risk_manager=MagicMock(),
        market_data=market_data if market_data is not None else _QuoteService(),
        portfolio=PortfolioAccounting(run_id="r", initial_cash=Decimal("1000")),
        symbols=[SYM],
        timeframe=timeframe,
        run_mode=mode,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# T-36  engine construction
# ---------------------------------------------------------------------------


def test_t36_a_valid_live_interval_is_accepted_and_exposed() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    assert engine.exit_check_interval_s == 300
    assert _engine().exit_check_interval_s is None


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.BACKTEST])
def test_t36_a_non_live_engine_refuses_the_key(mode: RunMode) -> None:
    with pytest.raises(ValueError, match="only valid for LIVE"):
        _engine(config={"exit_check_interval_s": 300}, mode=mode)


@pytest.mark.parametrize("value", [60 - 1, 3601, 0, -5, True, 12.5, "300"])
def test_t36_out_of_range_or_mistyped_values_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match="exit_check_interval_s"):
        _engine(config={"exit_check_interval_s": value})


def test_t36_an_interval_not_below_the_poll_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="below the poll interval"):
        _engine(config={"exit_check_interval_s": 300, "poll_interval_seconds": 300})
    with pytest.raises(ValueError, match="below the poll interval"):
        _engine(config={"exit_check_interval_s": 3600})  # 1h poll = 3600
    with pytest.raises(ValueError, match="below the poll interval"):
        _engine(config={"exit_check_interval_s": 300}, timeframe=TimeFrame.FIVE_MINUTES)
    _engine(config={"exit_check_interval_s": 299, "poll_interval_seconds": 300})


def test_t36_live_without_ticker_support_is_refused() -> None:
    with pytest.raises(ValueError, match="ticker quotes"):
        _engine(config={"exit_check_interval_s": 300}, market_data=_PlainService())
    with pytest.raises(ValueError, match="ticker quotes"):
        _engine(config={"exit_check_interval_s": 300}, market_data=MagicMock())  # truthy != True
    _engine(market_data=_PlainService())  # no key: nothing to refuse


# ---------------------------------------------------------------------------
# T-37  the bar loop is untouched
# ---------------------------------------------------------------------------


async def _run_bar_loop(
    monkeypatch: pytest.MonkeyPatch, *, feature_on: bool
) -> tuple[list[float], list[float]]:
    """Drive the REAL ``run_live_loop`` on virtual time; return (poll times, sleep timeouts)."""
    vt = {"now": 1000.0}
    polls: list[float] = []
    timeouts: list[float] = []
    engine = _engine(config={"exit_check_interval_s": 300} if feature_on else {})
    overruns = [10.0, 5000.0, 0.0, 3599.0, 1.0, 0.0]  # incl. polls longer than the interval

    async def _poll() -> None:
        polls.append(vt["now"])
        vt["now"] += overruns[len(polls) - 1]
        await asyncio.sleep(0)  # a real yield: the sibling task interleaves
        if len(polls) >= len(overruns):
            engine._stop_event.set()

    async def _warmup() -> None:
        return None

    async def _fake_wait_for(aw: Any, timeout: float | None = None) -> Any:
        timeouts.append(float(timeout))  # type: ignore[arg-type]
        vt["now"] += float(timeout)  # type: ignore[arg-type]
        aw.close()
        raise TimeoutError

    async def _quiet_tick() -> None:
        await asyncio.sleep(0)

    async def _spin_wait(seconds: float) -> bool:
        await asyncio.sleep(0)
        return engine._stop_event.is_set()

    monkeypatch.setattr(engine, "_poll_and_process", _poll)
    monkeypatch.setattr(engine, "_warmup_bar_windows", _warmup)
    monkeypatch.setattr(engine, "_intrabar_exit_tick", _quiet_tick)
    monkeypatch.setattr(engine, "_intrabar_wait", _spin_wait)
    monkeypatch.setattr(se, "time", SimpleNamespace(monotonic=lambda: vt["now"]))
    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)
    engine._state = se.EngineState.RUNNING
    await engine.run_live_loop()
    return polls, timeouts


async def test_t37_bar_poll_timing_is_identical_with_the_feature_on_or_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off = await _run_bar_loop(monkeypatch, feature_on=False)
    on = await _run_bar_loop(monkeypatch, feature_on=True)
    assert on == off
    polls, timeouts = off
    assert len(polls) == 6
    assert timeouts[0] == 3600.0 - 10.0  # the sleep is interval - elapsed, as before
    assert all(t > 0 for t in timeouts)  # an overrun skips the sleep (no zero waits)


def test_t37_the_while_loop_of_run_live_loop_is_the_pre_1_3b_b_loop() -> None:
    """Structural pin: the loop body polls, never awaits the intra-bar task, and the only
    additions to ``run_live_loop`` are the task start and the ``finally``."""
    src = textwrap.dedent(inspect.getsource(StrategyEngine.run_live_loop))
    tree = ast.parse(src)
    fn = tree.body[0]
    assert isinstance(fn, ast.AsyncFunctionDef)
    whiles = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert len(whiles) == 1
    body_names = {n.attr for n in ast.walk(whiles[0]) if isinstance(n, ast.Attribute)}
    assert not any("intrabar" in name for name in body_names)
    tries = [n for n in fn.body if isinstance(n, ast.Try)]
    assert len(tries) == 1 and tries[0].finalbody and not tries[0].handlers


# ---------------------------------------------------------------------------
# T-38 / T-39  the task
# ---------------------------------------------------------------------------


async def test_t38_first_tick_at_min_60_and_interval_then_every_interval() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    clock = IntrabarClock(engine, horizon=1500.0)
    engine._intrabar_wait = clock.wait  # type: ignore[method-assign]
    ticks: list[float] = []

    async def _tick() -> None:
        ticks.append(clock.now)

    engine._intrabar_exit_tick = _tick  # type: ignore[method-assign]
    engine._state = se.EngineState.RUNNING
    await engine._intrabar_exit_loop()
    assert ticks == [60.0, 360.0, 660.0, 960.0, 1260.0]
    assert clock.waits[:2] == [60.0, 300.0]


async def test_t38_a_short_interval_makes_the_first_tick_wait_the_interval() -> None:
    engine = _engine(config={"exit_check_interval_s": 60})
    clock = IntrabarClock(engine, horizon=200.0)
    engine._intrabar_wait = clock.wait  # type: ignore[method-assign]

    async def _tick() -> None:
        return None

    engine._intrabar_exit_tick = _tick  # type: ignore[method-assign]
    await engine._intrabar_exit_loop()
    assert clock.waits[:2] == [60.0, 60.0]


async def test_t38_a_busy_lock_earns_exactly_one_fast_retry() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    clock = IntrabarClock(engine, horizon=1000.0)
    engine._intrabar_wait = clock.wait  # type: ignore[method-assign]
    busy = {"left": 2}

    async def _tick() -> None:
        if busy["left"] > 0:
            busy["left"] -= 1
            engine._intrabar_retry_soon = True

    engine._intrabar_exit_tick = _tick  # type: ignore[method-assign]
    await engine._intrabar_exit_loop()
    # tick@60 busy -> +30 (fast retry) busy again -> back to 300 (no second fast retry)
    assert clock.waits[:4] == [60.0, 30.0, 300.0, 300.0]


async def test_t38_the_real_wait_is_interrupted_by_the_stop_event() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        engine._stop_event.set()

    stopper = asyncio.create_task(_stop_soon())
    started = asyncio.get_running_loop().time()
    assert await engine._intrabar_wait(30.0) is True
    assert asyncio.get_running_loop().time() - started < 5.0
    await stopper
    assert await engine._intrabar_wait(30.0) is True  # already stopped: returns at once
    engine._stop_event.clear()
    assert await engine._intrabar_wait(0.01) is False  # a plain timeout


async def test_t38_stopping_the_run_ends_the_task_within_one_wait() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    engine._state = se.EngineState.RUNNING
    engine._start_intrabar_task()
    task = engine._intrabar_task
    assert task is not None and not task.done()
    await asyncio.sleep(0)  # let it reach its first wait
    await asyncio.wait_for(engine._stop_intrabar_task(), timeout=5.0)
    assert task.done() and engine._intrabar_task is None
    assert engine._stop_event.is_set()


async def test_t38_a_tick_never_runs_after_stop() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    engine._stop_event.set()
    assert engine._state != se.EngineState.RUNNING
    svc = engine._market_data
    await engine._intrabar_exit_tick()
    assert svc.calls == 0


async def test_t39_a_tick_exception_is_logged_by_type_counted_and_the_task_continues() -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    clock = IntrabarClock(engine, horizon=1000.0)
    engine._intrabar_wait = clock.wait  # type: ignore[method-assign]
    calls = {"n": 0}

    async def _tick() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("secret-api-key-123 https://example.invalid/x")

    engine._intrabar_exit_tick = _tick  # type: ignore[method-assign]
    metrics: list[str] = []
    engine._metric = lambda name, **labels: metrics.append(name)  # type: ignore[method-assign]
    with capture_logs() as logs:
        await engine._intrabar_exit_loop()
    assert calls["n"] >= 3  # it kept ticking
    errs = [e for e in logs if e["event"] == "engine.intrabar_tick_error"]
    assert len(errs) == 1 and errs[0]["error_type"] == "RuntimeError"
    assert "secret" not in repr(logs) and "example.invalid" not in repr(logs)
    assert metrics == ["intrabar_tick_errors_total"]


async def test_t39_task_death_is_logged_and_does_not_stop_the_bar_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(config={"exit_check_interval_s": 300})
    polls: list[int] = []

    async def _poll() -> None:
        polls.append(1)
        await asyncio.sleep(0)
        if len(polls) >= 3:
            engine._stop_event.set()

    async def _warmup() -> None:
        return None

    async def _dying_wait(seconds: float) -> bool:
        raise RuntimeError("boom")

    async def _fast_wait_for(aw: Any, timeout: float | None = None) -> Any:
        aw.close()
        raise TimeoutError

    monkeypatch.setattr(engine, "_poll_and_process", _poll)
    monkeypatch.setattr(engine, "_warmup_bar_windows", _warmup)
    monkeypatch.setattr(engine, "_intrabar_wait", _dying_wait)
    monkeypatch.setattr(asyncio, "wait_for", _fast_wait_for)
    metrics: list[str] = []
    monkeypatch.setattr(engine, "_metric", lambda name, **labels: metrics.append(name))
    engine._state = se.EngineState.RUNNING
    with capture_logs() as logs:
        await engine.run_live_loop()
    assert len(polls) == 3
    died = [e for e in logs if e["event"] == "engine.intrabar_loop_died"]
    assert len(died) == 1 and died[0]["error_type"] == "RuntimeError"
    assert metrics.count("intrabar_loop_died_total") == 1  # WP13BB-SEC-06: alertable


async def test_t38_a_paper_engine_never_starts_a_task() -> None:
    engine = _engine(mode=RunMode.PAPER)
    engine._start_intrabar_task()
    assert engine._intrabar_task is None
    await engine._stop_intrabar_task()  # no-op


# ---------------------------------------------------------------------------
# T-44  AST pins (CF-04 / CF-30 / lock discipline)
# ---------------------------------------------------------------------------

_ENGINE_SRC = Path(se.__file__).read_text(encoding="utf-8")
_FORBIDDEN_NAMES = {
    "update_market_prices",
    "evaluate",
    "tick_cooldown",
    "_call_strategy_on_bar",
    "open_lot",
    "finalize_anchor",
    "arm_interim",
    "restore_anchor",
    "seed_peak",
    "_excursion_tracker",
    "_circuit_breaker",
    "_bar_count",
    "_drop_entry_signals",
}


def _intrabar_functions() -> dict[str, ast.AST]:
    tree = ast.parse(_ENGINE_SRC)
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("_intrabar")
            or node.name in ("_start_intrabar_task", "_stop_intrabar_task")
        ):
            out[node.name] = node
    return out


def _referenced(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def test_t44_the_intrabar_functions_exist() -> None:
    fns = _intrabar_functions()
    assert {
        "_intrabar_exit_loop",
        "_intrabar_exit_tick",
        "_intrabar_fire",
        "_intrabar_wait",
        "_intrabar_read_bid",
    } <= set(fns)


def test_t44_the_tick_is_read_only_and_never_a_bar_pass() -> None:
    for name, node in _intrabar_functions().items():
        hit = _referenced(node) & _FORBIDDEN_NAMES
        assert not hit, f"{name} references {sorted(hit)}"
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "BUY":
                raise AssertionError(f"{name} references a BUY direction")
            if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Attribute):
                assert sub.target.attr != "_bar_count", name


def test_t44_nothing_that_waits_or_fetches_runs_while_the_lock_is_held() -> None:
    """The lock is taken in ``_intrabar_exit_tick`` right before ``_intrabar_fire`` and released
    in the ``finally``: ``_intrabar_fire`` is the only code in the critical section and it must
    neither fetch a quote nor sleep / wait on the stop event."""
    fns = _intrabar_functions()
    fire = _referenced(fns["_intrabar_fire"])
    assert not fire & {
        "fetch_ticker_quote",
        "_intrabar_read_bid",
        "_intrabar_wait",
        "sleep",
        "wait_for",
        "_acquire_cycle_lock",
    }
    tick = fns["_intrabar_exit_tick"]
    assert isinstance(tick, ast.AsyncFunctionDef)
    tries = [n for n in tick.body if isinstance(n, ast.Try)]
    assert len(tries) == 1
    critical = ast.Module(body=tries[0].body, type_ignores=[])
    assert (
        _referenced(critical) & {"_intrabar_read_bid", "_intrabar_wait", "fetch_ticker_quote"}
        == set()
    )
    assert any(
        isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "release"
        for n in tries[0].finalbody
    )
    # and it is never an ``async with self._cycle_lock`` around an await of a reading
    assert not [n for n in ast.walk(tick) if isinstance(n, ast.AsyncWith)]


def test_t44_the_tick_reaches_the_exchange_only_through_the_quote_service_and_process_signal() -> (
    None
):
    fns = _intrabar_functions()
    refs = set().union(*(_referenced(n) for n in fns.values()))
    assert "fetch_ticker_quote" in refs and "_submit_exit_signal" in refs
    assert not refs & {"create_order", "fetch_ticker", "fetch_ohlcv", "get_latest_bar"}


# ---------------------------------------------------------------------------
# T-45 / T-46  manager intra-bar modes
# ---------------------------------------------------------------------------


def _pos(qty: str = "1", entry: str = "100", opened: datetime = LOT) -> Position:
    return Position(
        symbol=SYM,
        run_id="r",
        quantity=Decimal(qty),
        average_entry_price=Decimal(entry),
        current_price=Decimal(entry),
        opened_at=opened,
        updated_at=opened,
    )


def _trail() -> TrailingStopManager:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("100"), anchor_bar_ts=None)
    return mgr


def test_t45_an_intrabar_reading_above_the_mark_does_not_raise_it() -> None:
    mgr = _trail()
    assert mgr.check(SYM, Decimal("130"), _pos(), intrabar=True) is None
    assert mgr.peak_prices[SYM] == Decimal("100")  # read-only (U-8)
    mgr.check(SYM, Decimal("130"), _pos(), bar_ts=datetime(2026, 3, 1, 13, tzinfo=UTC))
    assert mgr.peak_prices[SYM] == Decimal("130")  # only a bar close ratchets


def test_t45_a_reading_at_or_below_the_trail_triggers_with_provenance() -> None:
    mgr = _trail()
    assert mgr.check(SYM, Decimal("90.01"), _pos(), intrabar=True) is None
    sig = mgr.check(SYM, Decimal("90"), _pos(), intrabar=True)
    assert sig is not None
    assert sig.metadata["trigger"] == "trailing_stop"
    assert sig.metadata["exit_path"] == "intrabar" and sig.metadata["trigger_source"] == "bid"
    assert sig.metadata["trigger_price"] == "90"
    assert mgr.peak_prices[SYM] == Decimal("100")  # the mark is kept


def test_t45_the_bar_close_signal_has_no_exit_path() -> None:
    mgr = _trail()
    sig = mgr.check(SYM, Decimal("85"), _pos(), bar_ts=datetime(2026, 3, 1, 13, tzinfo=UTC))
    assert sig is not None and "exit_path" not in sig.metadata


def test_t45_no_mark_means_no_signal_and_nothing_is_seeded() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    assert mgr.check(SYM, Decimal("1"), _pos(), intrabar=True) is None
    assert mgr.peak_prices == {}  # no lazy seed from a ticker reading


def test_t45_flat_position_is_not_cleaned_up_by_an_intrabar_check() -> None:
    mgr = _trail()
    assert mgr.check(SYM, Decimal("1"), None, intrabar=True) is None
    assert mgr.peak_prices[SYM] == Decimal("100")


def test_t45_pending_is_never_aged_by_ticks_and_a_quantity_change_reprotects() -> None:
    mgr = _trail()
    assert mgr.check(SYM, Decimal("80"), _pos(), intrabar=True) is not None
    assert mgr.has_unchanged_pending(SYM, _pos()) is True
    for _ in range(10):  # any number of ticks: no aging, no re-emission
        assert mgr.check(SYM, Decimal("80"), _pos(), intrabar=True) is None
    assert mgr._pending_age[SYM] == 0
    # bar-close checks DO age it: ttl 3
    t = datetime(2026, 3, 1, 13, tzinfo=UTC)
    assert mgr.check(SYM, Decimal("80"), _pos(), bar_ts=t) is None
    assert mgr.check(SYM, Decimal("80"), _pos(), bar_ts=t) is None
    assert mgr.check(SYM, Decimal("80"), _pos(), bar_ts=t) is not None  # ttl expired, re-emitted
    # a partial fill (quantity change) re-protects the residual from a tick
    assert mgr.has_unchanged_pending(SYM, _pos("0.4")) is False
    assert mgr.check(SYM, Decimal("80"), _pos("0.4"), intrabar=True) is not None


def _bracket(**kw: Any) -> BracketExitManager:
    args: dict[str, Any] = {
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "bracket_mode": "fixed",
    }
    args.update(kw)
    return BracketExitManager(**args)


def _anchor(mgr: BracketExitManager, *, lot_key: datetime = LOT) -> None:
    mgr.open_lot(
        SYM,
        lot_key=lot_key,
        opening_order_id=None,
        anchor_at=None,
        anchor_bar_ts=None,
        entry_basis=Decimal("100"),
        basis_qty=Decimal("1"),
        atr=None,
        source="entry",
        final=True,
    )


def test_t46_an_intrabar_check_fires_on_the_stop_only_never_on_the_take_profit() -> None:
    mgr = _bracket()
    _anchor(mgr)
    assert mgr.check(SYM, Decimal("95.01"), _pos(), intrabar=True) is None
    assert mgr.check(SYM, Decimal("111"), _pos(), intrabar=True) is None  # above the TP: ignored
    assert mgr.pending_exit_symbols == set()
    sig = mgr.check(SYM, Decimal("95"), _pos(), intrabar=True)
    assert sig is not None and sig.metadata["stop_loss"] is True
    assert "take_profit" not in sig.metadata
    assert sig.metadata["exit_path"] == "intrabar" and sig.metadata["trigger_source"] == "bid"
    assert sig.metadata["trigger_price"] == "95" and sig.metadata["anchor_source"] == "entry"


def test_t46_the_take_profit_still_fires_at_the_bar_close() -> None:
    mgr = _bracket()
    _anchor(mgr)
    sig = mgr.check(SYM, Decimal("111"), _pos())
    assert sig is not None and sig.metadata["take_profit"] is True
    assert "exit_path" not in sig.metadata


def test_t46_without_an_anchor_nothing_is_created_or_dropped() -> None:
    mgr = _bracket()
    assert mgr.check(SYM, Decimal("1"), _pos(), intrabar=True) is None
    assert mgr.get_anchor(SYM) is None  # a tick never creates a (late) anchor
    _anchor(mgr, lot_key=datetime(2026, 1, 1, tzinfo=UTC))  # another lot's anchor
    assert mgr.check(SYM, Decimal("1"), _pos(), intrabar=True) is None
    assert mgr.get_anchor(SYM) is not None  # and does not drop a mismatching one


def test_t46_a_tick_never_cleans_up_a_flat_position() -> None:
    mgr = _bracket()
    _anchor(mgr)
    assert mgr.check(SYM, Decimal("1"), None, intrabar=True) is None
    assert mgr.get_anchor(SYM) is not None


def test_t46_ticks_never_age_the_pending_ttl_so_it_still_expires_after_three_bar_checks() -> None:
    mgr = _bracket()
    _anchor(mgr)
    assert mgr.check(SYM, Decimal("90"), _pos(), intrabar=True) is not None
    for _ in range(25):  # any number of ticks in between
        assert mgr.check(SYM, Decimal("90"), _pos(), intrabar=True) is None
    assert mgr._pending_age[SYM] == 0
    assert mgr.check(SYM, Decimal("90"), _pos()) is None  # bar check 1
    assert mgr.check(SYM, Decimal("90"), _pos()) is None  # bar check 2
    assert mgr.check(SYM, Decimal("90"), _pos()) is not None  # bar check 3: TTL expired


def test_t46_a_partial_fill_reprotects_the_residual_from_a_tick() -> None:
    mgr = _bracket()
    _anchor(mgr)
    assert mgr.check(SYM, Decimal("90"), _pos("1"), intrabar=True) is not None
    assert mgr.has_unchanged_pending(SYM, _pos("1")) is True
    assert mgr.has_unchanged_pending(SYM, _pos("0.4")) is False
    sig = mgr.check(SYM, Decimal("90"), _pos("0.4"), intrabar=True)
    assert sig is not None


def test_t46_pure_reads_do_not_touch_state() -> None:
    mgr = _bracket()
    _anchor(mgr)
    before = (mgr.get_anchor(SYM), set(mgr.pending_exit_symbols))
    assert mgr.protective_level(SYM, _pos()) == Decimal("95.00")
    assert mgr.has_unchanged_pending(SYM, _pos()) is False
    assert (mgr.get_anchor(SYM), set(mgr.pending_exit_symbols)) == before


# ---------------------------------------------------------------------------
# T-67  fetch_ticker_quote
# ---------------------------------------------------------------------------


class _StubExchange:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def fetch_ticker(self, symbol: str) -> Any:
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _service(*outcomes: Any) -> tuple[CCXTMarketDataService, _StubExchange]:
    svc = CCXTMarketDataService(exchange_id="coinbase", cache_ttl_seconds=60)
    stub = _StubExchange(*outcomes)
    svc._exchange = stub  # type: ignore[assignment]
    return svc, stub


_GOOD = {"symbol": SYM, "bid": 99.5, "ask": 100.5, "last": 100.0, "timestamp": 1_767_225_600_000}


async def test_t67_two_calls_are_two_exchange_calls_no_cache() -> None:
    svc, stub = _service(dict(_GOOD), dict(_GOOD))
    a = await svc.fetch_ticker_quote(SYM)
    b = await svc.fetch_ticker_quote(SYM)
    assert stub.calls == 2 and a == b
    assert a.bid == Decimal("99.5") and a.ask == Decimal("100.5") and a.last == Decimal("100.0")
    assert a.exchange_ts == datetime(2026, 1, 1, tzinfo=UTC)
    assert svc.supports_ticker_quotes is True


async def test_t67_a_network_error_is_not_retried_and_nothing_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def _sleep(delay: float, result: object = None) -> object:
        slept.append(delay)
        return result

    svc, stub = _service(ccxt.NetworkError("down"), dict(_GOOD))
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    with pytest.raises(MarketDataError):
        await svc.fetch_ticker_quote(SYM)
    assert stub.calls == 1  # the confirming reading is the retry; the service does not retry
    assert [d for d in slept if d >= 0.5] == []  # no backoff sleep (a 100 ms throttle may sleep)


@pytest.mark.parametrize("exc_type", [ccxt.RateLimitExceeded, ccxt.DDoSProtection])
async def test_t67_rate_limit_answers_map_to_rate_limit_error_with_one_call(
    exc_type: type[Exception],
) -> None:
    svc, stub = _service(exc_type("429 too many"))
    with pytest.raises(RateLimitError):
        await svc.fetch_ticker_quote(SYM)
    assert stub.calls == 1


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ccxt.BadSymbol("nope"), DataNotAvailableError),
        (ccxt.ExchangeNotAvailable("maintenance"), MarketDataError),
        (ccxt.RequestTimeout("slow"), MarketDataError),
        (ccxt.ExchangeError("boom"), MarketDataError),
        (ccxt.AuthenticationError("bad key"), MarketDataError),
        # WP13BB-SEC-01: the rest of the ccxt tree and non-ccxt errors are mapped too
        (ccxt.OperationFailed("failed"), MarketDataError),
        (ccxt.BaseError("base"), MarketDataError),
        (TypeError("parse"), MarketDataError),
        (KeyError("bid"), MarketDataError),
    ],
)
async def test_t67_error_mapping_and_hygiene(exc: Exception, expected: type[Exception]) -> None:
    leaky = type(exc)("https://api.example/v3/ticker?api_key=SECRET-KEY-1 " + str(exc))
    svc, _stub = _service(leaky)
    with pytest.raises(expected) as info:
        await svc.fetch_ticker_quote(SYM)
    assert "SECRET" not in str(info.value) and "api.example" not in str(info.value)
    assert info.value.__cause__ is None  # the exchange's text is not chained either
    assert info.value.__suppress_context__ is True


async def test_sec01_a_cancellation_is_never_mapped() -> None:
    """WP13BB-SEC-01: the catch-all maps ``Exception`` only; a cancel (stop / shutdown)
    still propagates as ``CancelledError``."""
    svc, stub = _service(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await svc.fetch_ticker_quote(SYM)
    assert stub.calls == 1


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "abc", "NaN", "Infinity", 0, -1, "0", True, None, [], {}],
)
async def test_t67_an_unusable_price_becomes_none(raw: object) -> None:
    svc, _ = _service({"symbol": SYM, "bid": raw, "ask": raw, "last": raw})
    quote = await svc.fetch_ticker_quote(SYM)
    assert quote.bid is None and quote.ask is None and quote.last is None


async def test_t67_numeric_strings_and_a_missing_timestamp() -> None:
    svc, _ = _service({"symbol": SYM, "bid": "99.50", "ask": 100, "last": "1e2"})
    quote = await svc.fetch_ticker_quote(SYM)
    assert (quote.bid, quote.ask, quote.last) == (Decimal("99.50"), Decimal("100"), Decimal("1e2"))
    assert quote.exchange_ts is None


async def test_t67_a_crossed_book_is_passed_through_for_the_engine_to_judge() -> None:
    svc, _ = _service({"symbol": SYM, "bid": 101, "ask": 100, "last": 100})
    quote = await svc.fetch_ticker_quote(SYM)
    assert quote.bid == Decimal("101") and quote.ask == Decimal("100")


@pytest.mark.parametrize("payload", [None, [], "x", 5])
async def test_t67_a_payload_that_is_not_a_dict_is_an_error(payload: object) -> None:
    svc, _ = _service(payload)
    with pytest.raises(MarketDataError, match="Malformed"):
        await svc.fetch_ticker_quote(SYM)


async def test_t67_the_base_service_does_not_support_quotes() -> None:
    class _Base(BaseMarketDataService):
        pass

    assert BaseMarketDataService.supports_ticker_quotes.fget(object()) is False  # type: ignore[attr-defined]
    with pytest.raises(NotImplementedError):
        await BaseMarketDataService.fetch_ticker_quote(object(), SYM)  # type: ignore[arg-type]


async def test_t67_the_engine_bounds_one_reading_and_reports_the_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Hanging(_QuoteService):
        async def fetch_ticker_quote(self, symbol: str) -> Any:
            await asyncio.sleep(3600)

    monkeypatch.setattr(se, "_INTRABAR_QUOTE_TIMEOUT_S", 0.01)
    engine = _engine(config={"exit_check_interval_s": 300}, market_data=_Hanging())
    assert await engine._intrabar_read_bid(SYM) == (None, "TimeoutError", {})


async def test_t67_the_engine_judges_validity_not_the_service() -> None:
    svc = _QuoteService()
    engine = _engine(config={"exit_check_interval_s": 300}, market_data=svc)
    d = Decimal
    svc.quotes = [
        TickerQuote(SYM, d("99"), d("100"), d("99.5"), None),
        TickerQuote(SYM, d("99"), None, None, None),  # no ask: fine
        TickerQuote(SYM, d("100"), d("99"), None, None),  # crossed
        TickerQuote(SYM, None, d("99"), d("99"), None),  # no bid
        TickerQuote(SYM, d("0"), d("99"), d("99"), None),
        TickerQuote(SYM, d("NaN"), d("99"), d("99"), None),
        MarketDataError("boom"),
    ]
    got = [(await engine._intrabar_read_bid(SYM))[:2] for _ in range(7)]
    assert got == [
        (d("99"), None),
        (d("99"), None),
        (None, "invalid_quote"),
        (None, "invalid_quote"),
        (None, "invalid_quote"),
        (None, "invalid_quote"),
        (None, "MarketDataError"),
    ]
    svc.quotes = [RateLimitError("429")]
    with pytest.raises(RateLimitError):
        await engine._intrabar_read_bid(SYM)


async def test_sec03_a_usable_reading_carries_its_ask_and_age_for_the_log_only() -> None:
    """WP13BB-SEC-03 / SY-13b-06: the ask and the exchange-timestamp age are carried
    as numeric strings (or None) and never change the bid the engine judges."""
    svc = _QuoteService()
    engine = _engine(config={"exit_check_interval_s": 300}, market_data=svc)
    d = Decimal
    now = datetime.now(UTC)
    svc.quotes = [
        TickerQuote(SYM, d("99"), d("99.5"), None, now - timedelta(seconds=5)),
        TickerQuote(SYM, d("99"), None, None, None),
        TickerQuote(
            SYM, d("99"), d("99.5"), None, (now - timedelta(seconds=5)).replace(tzinfo=None)
        ),
        TickerQuote(SYM, d("100"), d("99"), None, now),  # crossed: unusable, no context
    ]
    bid, reason, ctx = await engine._intrabar_read_bid(SYM)
    assert (bid, reason, ctx["ask"]) == (d("99"), None, "99.5")
    assert ctx["quote_age_s"] is not None and 4.5 <= float(ctx["quote_age_s"]) < 60
    assert await engine._intrabar_read_bid(SYM) == (
        d("99"),
        None,
        {"ask": None, "quote_age_s": None},
    )
    _, _, naive = await engine._intrabar_read_bid(SYM)  # a naive timestamp is read as UTC
    assert naive["quote_age_s"] is not None and 4.5 <= float(naive["quote_age_s"]) < 60
    assert await engine._intrabar_read_bid(SYM) == (None, "invalid_quote", {})


# ---------------------------------------------------------------------------
# carry-forwards routed to 1.3b-b
# ---------------------------------------------------------------------------


def _order(status: OrderStatus = OrderStatus.REJECTED) -> Order:
    return Order(
        client_order_id="r-abc",
        run_id="r",
        symbol=SYM,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        status=status,
    )


def _fill(side: OrderSide, order_id: Any, qty: str, price: str) -> Fill:
    return Fill(
        order_id=order_id,
        symbol=SYM,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        fee=Decimal("0"),
        fee_currency="EUR",
        executed_at=LOT,
    )


def test_cf22_streak_levels_metric_and_reset() -> None:
    engine = _engine()
    seen: list[tuple[str, dict[str, str]]] = []
    engine._metric = lambda name, **labels: seen.append((name, dict(labels)))  # type: ignore[method-assign]
    levels: list[str] = []
    for _ in range(12):
        with capture_logs() as logs:
            engine._note_exit_not_submitted(SYM, "bracket_stop_loss", "no_orders")
        levels.append(logs[0]["log_level"])
    # warning 1,2 / error at 3 / warning 4..9 / error at 10 / warning 11, 12
    assert levels == (
        ["warning", "warning", "error"] + ["warning"] * 6 + ["error"] + ["warning"] * 2
    )
    assert engine._exit_not_submitted_streak[SYM] == 12
    assert (
        seen.count(
            ("exit_not_submitted_total", {"trigger": "bracket_stop_loss", "reason": "no_orders"})
        )
        == 12
    )
    engine._forget_exit_check_state(SYM)  # flat
    assert SYM not in engine._exit_not_submitted_streak


def test_cf22_a_raised_process_signal_is_always_error_level() -> None:
    engine = _engine()
    with capture_logs() as logs:
        engine._note_exit_not_submitted(SYM, "trailing_stop", "process_signal_raised")
    assert logs[0]["log_level"] == "error" and logs[0]["streak"] == 1

    class _Exec:  # WP13BB-CR-05: an older SELL in flight never downgrades a raise
        def pending_sell_quantity(self, symbol: str) -> Decimal:
            return Decimal("0.5")

    engine._execution_engine = _Exec()  # type: ignore[assignment]
    with capture_logs() as logs:
        engine._note_exit_not_submitted(SYM, "trailing_stop", "process_signal_raised")
    assert logs[0]["log_level"] == "error"
    assert logs[0]["reason"] == "process_signal_raised" and logs[0]["streak"] == 2


def test_cf22_an_inflight_sell_is_info_and_outside_the_streak() -> None:
    engine = _engine()

    class _Exec:
        def pending_sell_quantity(self, symbol: str) -> Decimal:
            return Decimal("0.5")

    engine._execution_engine = _Exec()  # type: ignore[assignment]
    assert engine._pending_sell_quantity(SYM) == Decimal("0.5")
    for _ in range(5):
        with capture_logs() as logs:
            engine._note_exit_not_submitted(SYM, "bracket_stop_loss", "no_orders")
        assert logs[0]["log_level"] == "info" and logs[0]["reason"] == "inflight_exit_pending"
    assert SYM not in engine._exit_not_submitted_streak


def test_cf22_the_pending_sell_accessor_ignores_engines_without_it() -> None:
    engine = _engine()  # a MagicMock execution engine: no accessor on its TYPE
    assert engine._pending_sell_quantity(SYM) == Decimal(0)

    class _Broken:
        def pending_sell_quantity(self, symbol: str) -> Decimal:
            raise RuntimeError("boom")

    engine._execution_engine = _Broken()  # type: ignore[assignment]
    assert engine._pending_sell_quantity(SYM) == Decimal(0)


async def test_cf22_an_accepted_order_resets_the_streak() -> None:
    engine = _engine()
    engine._exit_not_submitted_streak[SYM] = 2
    accepted = _order(OrderStatus.OPEN)

    async def _process(signal: Any) -> list[Order]:
        return [accepted]

    async def _get_fills(order_id: Any) -> list[Fill]:
        return []

    engine._execution_engine = SimpleNamespace(process_signal=_process, get_fills=_get_fills)  # type: ignore[assignment]
    manager = TrailingStopManager(trailing_stop_pct=0.1)
    sig = manager._emit(SYM, Decimal("100"), Decimal("90"), Decimal("89"), _pos())
    await engine._submit_exit_signal(manager, sig, {})
    assert SYM not in engine._exit_not_submitted_streak


def test_cf24_a_closed_lots_add_orders_leave_the_signal_bar_map() -> None:
    engine = _engine()
    opening, add1, add2, other = uuid4(), uuid4(), uuid4(), uuid4()
    ts = datetime(2026, 3, 1, tzinfo=UTC)
    for oid in (opening, add1, add2, other):
        engine._order_signal_bar_ts[oid] = ts
    lot = se._EngineLot(
        opening_order_id=opening,
        order=None,
        fills=[],
        entry=SimpleNamespace(),  # type: ignore[arg-type]
        add_order_ids={add1, add2},
    )
    engine._lots[SYM] = lot
    sell = _fill(OrderSide.SELL, uuid4(), "2", "90")
    pre = _pos("2")
    engine._on_fill_routed_impl(None, sell, pre)
    assert set(engine._order_signal_bar_ts) == {other}
    assert SYM not in engine._lots


def test_cf24_an_add_fill_is_remembered_on_the_lot() -> None:
    engine = _engine()
    opening, add = uuid4(), uuid4()
    entry = SimpleNamespace(
        entry_basis=Decimal("100"), basis_qty=Decimal("1"), entry_gross_vwap=Decimal("100")
    )
    lot = se._EngineLot(opening_order_id=opening, order=None, fills=[], entry=entry)  # type: ignore[arg-type]
    engine._lots[SYM] = lot
    engine._portfolio.update_position(_fill(OrderSide.BUY, opening, "1", "100"), Decimal("100"))
    add_fill = _fill(OrderSide.BUY, add, "1", "101")
    pre = engine._portfolio.get_position(SYM)
    engine._portfolio.update_position(add_fill, Decimal("101"))
    engine._on_fill_routed_impl(None, add_fill, pre)
    assert lot.add_order_ids == {add}


async def test_cf24_an_exact_finalisation_clears_the_engine_side_pending_state() -> None:
    """Pins CR-205 mutant J (``_clear_anchor_pending`` after an exact finalisation): once the
    exact window finalises a pending anchor its attempt counter, origin tag and last error go."""
    from tests.integration.fakes.bar_paths import scripted_path

    engine = _engine(
        config={
            "bracket_mode": "atr",
            "bracket_stop_loss_pct": None,
            "bracket_atr_sl_multiplier": 1.5,
            "bracket_atr_period": 14,
        }
    )
    bracket = engine._bracket_exit
    assert bracket is not None
    history = scripted_path(
        [100 + (i % 3) for i in range(100)], first_open=LOT, symbol=SYM, warmup=0
    )
    rule_ts = history[-1].timestamp
    anchor = bracket.open_lot(
        SYM,
        lot_key=LOT,
        opening_order_id=None,
        anchor_at=None,
        anchor_bar_ts=rule_ts,
        entry_basis=Decimal("100"),
        basis_qty=Decimal("1"),
        atr=Decimal("1"),
        source="entry_provisional",
        final=False,
    )
    engine._anchor_fetch_attempts[SYM] = 2
    engine._anchor_origin[SYM] = "entry"
    engine._last_exit_fetch_error[SYM] = "TimeoutError"
    done = await engine._finalize_one_anchor(SYM, anchor, history, 14)
    assert done is not None and done.final and done.source == "entry"
    assert SYM not in engine._anchor_fetch_attempts
    assert SYM not in engine._anchor_origin
    assert SYM not in engine._last_exit_fetch_error


def test_cf23_the_anchor_source_metric_carries_the_final_label() -> None:
    src = _ENGINE_SRC
    tree = ast.parse(src)
    sites = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_metric"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "bracket_anchor_source_total"
    ]
    assert len(sites) == 3
    for call in sites:
        assert {kw.arg for kw in call.keywords} == {"source", "final"}


# ---------------------------------------------------------------------------
# WP1.3b-b fix delta, Part B (test-only): CR-06 backoff clamp, CR-07 wiring, CR-11 tables
# ---------------------------------------------------------------------------


def test_cr06_the_rate_limit_backoff_stays_one_second_below_the_timeframe() -> None:
    """WP13BB-CR-06 (mutant M48): on a 15m run (900 s) with a 300 s interval, repeated
    429s double the interval up to ``tf_s - 1`` = 899 s and never reach the timeframe."""
    engine = _engine(config={"exit_check_interval_s": 300}, timeframe=TimeFrame.FIFTEEN_MINUTES)
    assert engine._tf_s == 900 and engine._intrabar_interval_now_s == 300.0
    seen: list[float] = []
    for _ in range(4):
        engine._intrabar_on_rate_limit()
        seen.append(engine._intrabar_interval_now_s)
    assert seen == [600.0, 899.0, 899.0, 899.0]


def test_cr07_the_live_exit_check_interval_is_wired_into_env_example_and_the_api_service() -> None:
    """WP13BB-CR-07: dropping the variable from ``.env.example`` or from the compose ``api``
    service would silently leave operators on the code default."""
    root = Path(__file__).resolve().parents[2]
    env_lines = (root / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "LIVE_EXIT_CHECK_INTERVAL_S=300" in env_lines
    compose = (root / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
    block = re.search(r"\n  api:\n(.*?)(?=\n  [A-Za-z0-9_-]+:)", compose, re.DOTALL)
    assert block is not None, "no `api:` service in infra/docker-compose.yml"
    assert "${LIVE_EXIT_CHECK_INTERVAL_S:-300}" in block.group(1)


def test_cr11_both_timeframe_second_tables_cover_every_timeframe_and_agree() -> None:
    """WP13BB-CR-11: the resolver table (``exit_config``) and the engine table are
    duplicates (CF-13b-31); an unknown timeframe resolves to 0 and silently turns the
    check off, so every ``TimeFrame`` must be present, positive and equal in both."""
    from trading.exit_config import _TIMEFRAME_SECONDS_BY_VALUE

    assert set(_TIMEFRAME_SECONDS_BY_VALUE) == {tf.value for tf in TimeFrame}
    assert set(se._TIMEFRAME_SECONDS) == set(TimeFrame)
    for tf in TimeFrame:
        assert _TIMEFRAME_SECONDS_BY_VALUE[tf.value] == se._TIMEFRAME_SECONDS[tf] > 0
