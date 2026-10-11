"""
tests/unit/test_wp13b_fix.py
------------------------------
WP1.3b-a FIX DELTA (``reports/vp2-wp1.3b/final-synthesis-1.3b-a.md`` section 7,
FX-01..FX-12): unit-level coverage.  The harness-level tests of the same delta are
in ``tests/integration/test_wp13b_fix_harness.py``.

New API is reached through the module / instance (``se.<name>``, ``engine.<attr>``)
so this file imports on the ``wp1.3b-a.patch`` tree and fails test by test there
(red-before-green) instead of failing at collection.

    FX-01  ATR period cap 49 (W = 6p+1 <= 300)       engine, strategy params, precheck
    FX-02  interim ATR = min(rule-bar window, newest window)
    FX-03  attempt budget, per-pass fetch budget, ``arm_interim``
    FX-07  ``bracket_exit.anchor_finalized`` with prev_stop_price / delta_bps
    FX-09  the same-pass guard (``_anchor_created_bar``)
    FX-12  ``_order_signal_bar_ts`` pruned when the lot closes
"""

from __future__ import annotations

import importlib.util
import uuid
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

import trading.strategy_engine as se
from common.models import OHLCVBar
from common.types import RunMode, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from tests.unit.test_wp13b_anchor import (
    LOT,
    SYM,
    T0,
    _apply,
    _atr_mgr,
    _bt_runner,
    _engine,
    _fill,
    _open,
    _order,
    _pos,
    _wavy_bars,
)
from trading.backtest import BacktestRunner
from trading.bracket_exit import BracketExitManager
from trading.exit_config import (
    POLICY_ATR_PERIOD_MAX,
    ExitConfigError,
    parse_exit_config,
)
from trading.portfolio import PortfolioAccounting

SYM2 = "ETH/USD"
TF = timedelta(hours=1)
ATR_BRACKET: dict[str, Any] = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_tp_multiplier": 3.0,
    "bracket_atr_period": 14,
}


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _path(closes: list[int], *, symbol: str = SYM) -> list[OHLCVBar]:
    return scripted_path(closes, first_open=T0, symbol=symbol, warmup=0, warmup_price=closes[0])


def _calm(n: int) -> list[int]:
    return [100 + (i % 2) for i in range(n)]  # true range ~ 1


def _wild(n: int) -> list[int]:
    return [100 + (i % 2) * 10 for i in range(n)]  # true range ~ 10


async def _paper_engine(
    bracket: dict[str, Any] | None = None, *, symbols: tuple[str, ...] = (SYM,)
) -> se.StrategyEngine:
    """A started engine whose anchor logic runs the PAPER/LIVE branch."""
    runner = BacktestRunner(
        strategies=[ScriptedSequenceStrategy("fx", {"script": {}, "target_notional": "1000"})],
        symbols=list(symbols),
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=bracket if bracket is not None else ATR_BRACKET,
        trailing_stop_pct=None,
        allow_pyramiding=False,
    )
    engine, _pf, _rm, _ex = runner._build_engine("fx")
    await engine.start("fx")
    engine._run_mode = RunMode.PAPER
    return engine


def _open_lot_at(
    engine: se.StrategyEngine,
    history: dict[str, list[OHLCVBar]],
    symbol: str = SYM,
    *,
    rule_bar_index: int | None = None,
) -> None:
    """Open a lot whose ``anchor_at`` maps to the rule bar ``history[symbol][rule_bar_index]``
    (default: the newest bar), through the real fill hook."""
    bars = history[symbol]
    idx = len(bars) - 1 if rule_bar_index is None else rule_bar_index
    at = bars[idx].timestamp + TF + timedelta(seconds=2)
    oid = uuid4()
    order = _order(oid, created_at=at).model_copy(update={"symbol": symbol})
    engine._current_history = history
    _apply(engine, order, _fill(oid, price="100", at=at, symbol=symbol))


def _gapped(n: int = 100, *, symbol: str = SYM) -> tuple[list[OHLCVBar], list[OHLCVBar]]:
    """``(full, gapped)``: ``n`` wavy bars and the same bars minus the one at index 60."""
    full = _path(_calm(n), symbol=symbol)
    return full, full[:60] + full[61:]


# ---------------------------------------------------------------------------
# FX-01  the ATR period cap
# ---------------------------------------------------------------------------


def test_fx01_window_always_fits_one_exchange_call() -> None:
    """``W = 6*period+1`` for the largest allowed period must fit ccxt Coinbase's single
    ``fetch_ohlcv`` call (maxLimit 300) -- otherwise resume and finalisation could never
    re-fetch the exact window."""
    assert POLICY_ATR_PERIOD_MAX == 49
    assert se._EXCHANGE_OHLCV_MAX_LIMIT == 300
    assert POLICY_ATR_PERIOD_MAX * se._ATR_WINDOW_MULTIPLIER + 1 <= se._EXCHANGE_OHLCV_MAX_LIMIT
    assert (POLICY_ATR_PERIOD_MAX + 1) * se._ATR_WINDOW_MULTIPLIER + 1 > (
        se._EXCHANGE_OHLCV_MAX_LIMIT
    ), "the cap must be tight: 50 would not fit"


def test_fx01_strategy_engine_refuses_period_50() -> None:
    """ARB-13ba-03: the engine's own ``parse_exit_config`` raises first, so a 'log instead of
    raise' guard for period >= 50 would be unreachable."""

    def build(period: int) -> se.StrategyEngine:
        return se.StrategyEngine(
            strategies=[ScriptedSequenceStrategy("fx", {"script": {}})],
            execution_engine=MagicMock(),
            risk_manager=MagicMock(),
            market_data=MagicMock(),
            portfolio=PortfolioAccounting(run_id="r", initial_cash=Decimal("1000")),
            symbols=[SYM],
            timeframe=TimeFrame.ONE_HOUR,
            run_mode=RunMode.PAPER,
            config={**ATR_BRACKET, "bracket_atr_period": period},
        )

    assert build(49)._bracket_exit.atr_period == 49  # type: ignore[union-attr]
    with pytest.raises(ExitConfigError) as exc:
        build(50)
    issue = exc.value.issues[0]
    assert (issue.field, issue.reason, issue.max) == ("bracket_atr_period", "out_of_range", 49)


def test_fx01_strategy_param_models_cap_the_period_at_49() -> None:
    from pydantic import ValidationError

    from trading.strategies.momentum_breakout import _MomentumBreakoutParams
    from trading.strategies.sl_tp_reversion import _SLTPReversionParams

    for model in (_MomentumBreakoutParams, _SLTPReversionParams):
        assert model(bracket_atr_period=49).bracket_atr_period == 49
        with pytest.raises(ValidationError):
            model(bracket_atr_period=50)


def test_fx01_precheck_flags_a_persisted_period_60_run() -> None:
    """DS-1: the WP1.3a pre-deploy precheck uses the same validator, so a persisted run with
    ``bracket_atr_period`` 50..99 is reported before the 1.3b-a image is deployed."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "wp13a_exit_config_precheck.py"
    spec = importlib.util.spec_from_file_location("wp13a_exit_config_precheck_fx", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(period: int) -> dict[str, Any]:
        return module._check_one_run(  # type: ignore[no-any-return]
            uuid.uuid4(),
            "paper",
            "running",
            {
                "strategy_name": "momentum_breakout",
                "strategy_params": {},
                "timeframe": "1h",
                "bracket_config": {
                    "bracket_mode": "atr",
                    "bracket_atr_sl_multiplier": 1.5,
                    "bracket_atr_period": period,
                },
            },
        )

    bad = run(60)
    assert bad["valid"] is False
    assert "bracket_atr_period:out_of_range" in bad["note"]
    assert run(49)["valid"] is True


def test_fx01_parse_reports_the_new_bound_in_the_message() -> None:
    with pytest.raises(ExitConfigError) as exc:
        parse_exit_config(
            bracket={
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 1.5,
                "bracket_atr_period": 50,
            },
            trailing_stop_pct=None,
        )
    assert "49" in exc.value.issues[0].message


# ---------------------------------------------------------------------------
# FX-02  the interim ATR
# ---------------------------------------------------------------------------


def test_fx02_interim_is_the_min_when_the_rule_bar_window_is_calmer() -> None:
    hist = _path(_calm(70) + _wild(30))
    rule = hist[69].timestamp
    best = se.atr_for_bars(hist[:70], 14)
    newest = se.atr_for_bars(hist[-85:], 14)
    assert best is not None and newest is not None and best < newest  # a `max` mutant fails
    assert se.StrategyEngine._interim_atr(hist, rule, 14) == best


def test_fx02_interim_is_the_min_when_the_newest_window_is_calmer() -> None:
    hist = _path(_wild(30) + _calm(70))
    rule = hist[29].timestamp
    best = se.atr_for_bars(hist[:30], 14)
    newest = se.atr_for_bars(hist[-85:], 14)
    assert best is not None and newest is not None and newest < best  # a `max` mutant fails
    assert se.StrategyEngine._interim_atr(hist, rule, 14) == newest


def test_fx02_rule_bar_window_with_at_most_period_bars_falls_back_to_the_newest() -> None:
    hist = _path(_calm(100))
    # rule bar at index 3 -> 4 bars <= period 14 -> ATR_best is None
    assert se.atr_for_bars(hist[:4], 14) is None
    assert se.StrategyEngine._interim_atr(hist, hist[3].timestamp, 14) == se.atr_for_bars(
        hist[-85:], 14
    )
    # a rule bar older than the whole window behaves the same
    assert se.StrategyEngine._interim_atr(hist, hist[0].timestamp - TF, 14) == se.atr_for_bars(
        hist[-85:], 14
    )
    # and no rule bar at all -> HEAD's per-bar ATR
    assert se.StrategyEngine._interim_atr(hist, None, 14) == se.atr_for_bars(hist[-85:], 14)


def test_fx02_interim_is_none_when_neither_estimate_is_computable() -> None:
    assert se.StrategyEngine._interim_atr([], None, 14) is None
    short = _path(_calm(14))  # 14 bars <= period
    assert se.StrategyEngine._interim_atr(short, short[-1].timestamp, 14) is None
    flat = _path([100] * 60)  # ATR 0 is not computable
    assert se.StrategyEngine._interim_atr(flat, flat[-1].timestamp, 14) is None


@pytest.mark.parametrize("pattern", ["calm_then_wild", "wild_then_calm"])
def test_fx02_i2_the_interim_stop_is_never_looser_than_heads_per_bar_stop(
    pattern: str,
) -> None:
    """I-2: ``I <= ATR_newest`` so ``S_I >= E - k * ATR(history)`` at every arming point."""
    closes = _calm(70) + _wild(30) if pattern == "calm_then_wild" else _wild(30) + _calm(70)
    hist = _path(closes)
    head_atr = se.atr_for_bars(hist, 14)  # HEAD's per-bar ATR (last W bars)
    assert head_atr is not None
    entry = Decimal("100")
    for idx in (3, 10, 20, 50, 80, 99):
        interim = se.StrategyEngine._interim_atr(hist, hist[idx].timestamp, 14)
        assert interim is not None and interim <= head_atr
        assert entry - Decimal("1.5") * interim >= entry - Decimal("1.5") * head_atr


async def test_fx02_cr03_rule_bar_at_window_index_3_is_provisional_not_deferred() -> None:
    """The CR-03 case: the rule bar sits within the first ``period`` bars of a 100-bar window
    although the newest window has an ATR.  HEAD-of-patch made it `entry_deferred` (no levels)."""
    engine = await _paper_engine()
    hist = _path(_calm(100))
    _open_lot_at(engine, {SYM: hist}, rule_bar_index=3)
    anchor = engine._bracket_exit.get_anchor(SYM)  # type: ignore[union-attr]
    interim = se.atr_for_bars(hist[-85:], 14)
    assert interim is not None
    assert anchor is not None and anchor.source == "entry_provisional" and not anchor.final
    assert anchor.anchor_bar_ts == hist[3].timestamp
    assert anchor.atr == interim
    assert anchor.stop_price == anchor.entry_basis - Decimal("1.5") * interim  # Decimal ==
    assert anchor.take_profit_price == anchor.entry_basis + Decimal("3.0") * interim


# ---------------------------------------------------------------------------
# FX-03  arm_interim, the per-anchor attempt budget and the per-pass fetch budget
# ---------------------------------------------------------------------------


def _arm(mgr: BracketExitManager, atr: str, rule: Any = None) -> Any:
    return mgr.arm_interim(SYM, atr=Decimal(atr), anchor_bar_ts=rule, source="entry_provisional")


def test_fx03g_arm_interim_arms_once_and_refuses_final_and_armed_anchors() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    assert _arm(mgr, "1") is None  # no anchor at all
    _open(mgr, entry="100", atr=None, final=False, source="entry_deferred")
    rule = LOT - TF
    armed = _arm(mgr, "1", rule)
    assert armed is not None and not armed.final and armed.source == "entry_provisional"
    assert armed.atr == Decimal("1") and armed.anchor_bar_ts == rule
    assert (armed.stop_price, armed.take_profit_price) == (Decimal("98"), Decimal("103"))
    # I-1: set once.  A second arming (even tighter) is refused and changes nothing
    assert _arm(mgr, "0.1", rule) is None
    assert mgr.get_anchor(SYM) == armed
    # a final anchor is never armed again
    done = mgr.finalize_anchor(SYM, atr=Decimal("2"), anchor_bar_ts=rule, source="entry")
    assert done is not None and done.final
    assert _arm(mgr, "0.1", rule) is None
    assert mgr.get_anchor(SYM) == done


def test_fx03g_arm_interim_refuses_an_anchor_that_already_has_levels() -> None:
    mgr = _atr_mgr()
    first = _open(mgr, entry="100", atr="1", final=False, source="entry_provisional")
    assert _arm(mgr, "0.5") is None
    assert mgr.get_anchor(SYM) == first


async def _two_pending(
    engine: se.StrategyEngine,
) -> tuple[dict[str, list[OHLCVBar]], dict[str, list[OHLCVBar]]]:
    """Two provisional anchors (one per symbol); returns ``(gapped, full)`` histories."""
    full1, gapped1 = _gapped(symbol=SYM)
    full2, gapped2 = _gapped(symbol=SYM2)
    _open_lot_at(engine, {SYM: gapped1, SYM2: gapped2}, SYM)
    _open_lot_at(engine, {SYM: gapped1, SYM2: gapped2}, SYM2)
    bracket = engine._bracket_exit
    assert bracket is not None
    for symbol in (SYM, SYM2):
        anchor = bracket.get_anchor(symbol)
        assert anchor is not None and anchor.source == "entry_provisional" and not anchor.final
    return {SYM: gapped1, SYM2: gapped2}, {SYM: full1, SYM2: full2}


async def test_fx03f_one_finalisation_fetch_per_pass_engine_wide() -> None:
    """SEC-06: at most ONE exact-window fetch per ``_finalize_pending_anchors`` call across all
    symbols, so a pass is never held for more than one fetch timeout.  The symbol skipped for
    budget spends no attempt and is served on the next pass."""
    engine = await _paper_engine(symbols=(SYM, SYM2))
    gapped, full = await _two_pending(engine)
    calls: list[str] = []

    async def fetch(symbol: str, end_ts: Any) -> list[OHLCVBar] | None:
        calls.append(symbol)
        return full[symbol][-85:]

    engine._fetch_exact_window = fetch  # type: ignore[method-assign]
    bracket = engine._bracket_exit
    assert bracket is not None

    engine._bar_count += 1  # pass 1
    await engine._finalize_pending_anchors(gapped)
    assert calls == [SYM]
    assert bracket.get_anchor(SYM).final and not bracket.get_anchor(SYM2).final  # type: ignore[union-attr]
    assert engine._anchor_fetch_attempts.get(SYM2, 0) == 0  # skipped, no attempt consumed

    engine._bar_count += 1  # pass 2
    await engine._finalize_pending_anchors(gapped)
    assert calls == [SYM, SYM2]
    assert bracket.get_anchor(SYM2).final  # type: ignore[union-attr]


async def test_fx03f_failed_fetches_spend_one_attempt_per_pass_and_stop_at_three() -> None:
    engine = await _paper_engine(symbols=(SYM, SYM2))
    gapped, _full = await _two_pending(engine)
    calls: list[str] = []

    async def failing(symbol: str, end_ts: Any) -> None:
        calls.append(symbol)
        return None

    engine._fetch_exact_window = failing  # type: ignore[method-assign]
    bracket = engine._bracket_exit
    assert bracket is not None
    for expected_calls in (1, 2, 3):
        engine._bar_count += 1
        await engine._finalize_pending_anchors(gapped)
        assert len(calls) == expected_calls  # exactly one fetch per pass
    assert calls == [SYM, SYM, SYM]  # first symbol first; attempts, not symbols, are capped
    assert engine._anchor_fetch_attempts.get(SYM2, 0) == 0
    done = bracket.get_anchor(SYM)
    assert done is not None and done.final and done.source == "entry_unverified"
    still = bracket.get_anchor(SYM2)
    assert still is not None and not still.final  # never starved of its interim levels
    assert still.stop_price is not None
    engine._bar_count += 1
    await engine._finalize_pending_anchors(gapped)
    assert calls[3:] == [SYM2]  # the budget now serves the second symbol


# ---------------------------------------------------------------------------
# FX-07  anchor_finalized observability
# ---------------------------------------------------------------------------


def _finalized_events(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == "bracket_exit.anchor_finalized"]


def test_fx07_a_provisional_tighter_than_final_logs_a_warning_with_negative_delta_bps() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry="100", atr="1", final=False, source="entry_provisional")  # stop 98
    with capture_logs() as logs:
        mgr.finalize_anchor(SYM, atr=Decimal("2"), anchor_bar_ts=None, source="entry")  # stop 96
    (event,) = _finalized_events(logs)
    assert event["log_level"] == "warning"
    assert event["prev_stop_price"] == "98.00" or Decimal(event["prev_stop_price"]) == Decimal("98")
    assert Decimal(event["stop_price"]) == Decimal("96")
    assert event["delta_bps"] == pytest.approx(-200.0)  # (96 - 98) / 100 * 1e4


def test_fx07_an_unchanged_stop_logs_info_with_zero_delta() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry="100", atr="1", final=False, source="entry_provisional")
    with capture_logs() as logs:
        mgr.finalize_anchor(SYM, atr=None, anchor_bar_ts=None, source="entry_unverified")
    (event,) = _finalized_events(logs)
    assert event["log_level"] == "info"
    assert event["delta_bps"] == 0


def test_fx07_a_tighter_final_stop_and_a_small_loosening_stay_at_info() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry="100", atr="1", final=False, source="entry_provisional")  # stop 98
    with capture_logs() as logs:  # final stop 98.2: moved UP 20 bps
        mgr.finalize_anchor(SYM, atr=Decimal("0.9"), anchor_bar_ts=None, source="entry")
    (event,) = _finalized_events(logs)
    assert event["log_level"] == "info" and event["delta_bps"] > 0

    mgr2 = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr2, entry="100", atr="1", final=False, source="entry_provisional")  # stop 98
    with capture_logs() as logs2:  # final stop 97.9: moved DOWN 10 bps (< 25)
        mgr2.finalize_anchor(SYM, atr=Decimal("1.05"), anchor_bar_ts=None, source="entry")
    (event2,) = _finalized_events(logs2)
    assert event2["log_level"] == "info" and -25 < event2["delta_bps"] < 0


def test_fx07_a_missing_stop_gives_a_none_delta() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry="100", atr=None, final=False, source="entry_deferred")  # no stop yet
    with capture_logs() as logs:
        mgr.finalize_anchor(SYM, atr=Decimal("1"), anchor_bar_ts=None, source="entry_deferred")
    (event,) = _finalized_events(logs)
    assert event["delta_bps"] is None and event["prev_stop_price"] is None


# ---------------------------------------------------------------------------
# FX-09  the same-pass guard
# ---------------------------------------------------------------------------


async def test_fx09_an_anchor_created_in_this_pass_is_never_finalised_in_it() -> None:
    """Kills M10 (guard removed): ``_open_exit_lot`` then ``_finalize_pending_anchors`` at the
    same ``_bar_count`` -> no finalisation and no fetch; the next pass does fetch."""
    engine = await _paper_engine()
    _full, gapped = _gapped()
    _open_lot_at(engine, {SYM: gapped})
    bracket = engine._bracket_exit
    assert bracket is not None
    before = bracket.get_anchor(SYM)
    assert before is not None and not before.final
    fetch = AsyncMock(return_value=None)
    engine._fetch_exact_window = fetch  # type: ignore[method-assign]

    await engine._finalize_pending_anchors({SYM: gapped})  # same _bar_count
    assert fetch.await_count == 0
    assert bracket.get_anchor(SYM) == before

    engine._bar_count += 1
    await engine._finalize_pending_anchors({SYM: gapped})
    assert fetch.await_count == 1


# ---------------------------------------------------------------------------
# FX-12  _order_signal_bar_ts pruned
# ---------------------------------------------------------------------------


async def test_fx12_a_closed_lot_leaves_no_signal_bar_entry_behind() -> None:
    fixed = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    runner = _bt_runner({0: "buy", 3: "sell"}, bracket=fixed)
    engine = await _engine(runner)
    bars = scripted_path(
        [100] * 20, first_open=T0 + timedelta(hours=60), symbol=SYM, warmup=0, warmup_price=100
    )
    await engine.run_backtest({SYM: [*_wavy_bars(50), *bars]})
    assert engine._order_signal_bar_ts == {}
    assert engine._lots == {}
    assert engine._bracket_exit.get_anchor(SYM) is None  # type: ignore[union-attr]


async def test_fx12_the_entry_survives_while_the_lot_is_open() -> None:
    fixed = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    runner = _bt_runner({0: "buy"}, bracket=fixed)
    engine = await _engine(runner)
    await engine.run_backtest({SYM: _wavy_bars(60)})
    assert len(engine._order_signal_bar_ts) == 1  # the lot is still open


# ---------------------------------------------------------------------------
# WP13BA-CR-201  FX-06 is PAPER/LIVE only
# ---------------------------------------------------------------------------


async def test_cr201_backtest_never_reports_an_unprotected_lot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In BACKTEST a held lot without a stop is in-model, not an incident: no error event
    and no `lot_unprotected_bars_total` tick (the counter would otherwise feed the
    paper/live G-LIVE L-8 alert).  An end-to-end BACKTEST lot cannot be left without a
    stop here (the entry is skipped, `engine.entry_skipped_exit_level_unavailable`), so
    the gate is pinned on the 5a hook itself; the PAPER control proves the probe is live."""
    from common.metrics import metrics as registry

    seen: list[str] = []
    original = registry.increment

    def _spy(name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        seen.append(name)
        original(name, value, labels)

    monkeypatch.setattr(registry, "increment", _spy)
    engine = await _engine(_bt_runner({}, bracket=ATR_BRACKET))  # SL leg, no trailing
    assert engine._bracket_exit is not None and engine._bracket_exit.has_stop_loss
    assert not _atr_mgr(sl=None).has_stop_loss  # an ATR TP-only bracket has no SL leg
    held = _pos()  # held, and no anchor for it -> no protective level
    with capture_logs() as logs:
        for _ in range(12):
            engine._track_lot_protection(SYM, held)
    assert not [e for e in logs if e["event"] == "bracket_exit.lot_unprotected"]
    assert "lot_unprotected_bars_total" not in seen
    assert engine._unprotected_bars == {}

    engine._run_mode = RunMode.PAPER  # control: the same hook reports in PAPER
    with capture_logs() as logs:
        engine._track_lot_protection(SYM, held)
    reported = [e for e in logs if e["event"] == "bracket_exit.lot_unprotected"]
    assert [e["bars_unprotected"] for e in reported] == [1]
    assert seen.count("lot_unprotected_bars_total") == 1


# ---------------------------------------------------------------------------
# stop() clears the new state
# ---------------------------------------------------------------------------


async def test_fx03_stop_clears_the_pending_anchor_state() -> None:
    engine = await _paper_engine()
    _full, gapped = _gapped()
    _open_lot_at(engine, {SYM: gapped})
    engine._unprotected_bars[SYM] = 3
    engine._last_exit_fetch_error[SYM] = "NetworkError"
    assert engine._anchor_origin == {SYM: "entry"}
    assert engine._anchor_fetch_attempts == {SYM: 0}
    await engine.stop()
    assert engine._anchor_origin == {}
    assert engine._anchor_fetch_attempts == {}
    assert engine._last_exit_fetch_error == {}
    assert engine._unprotected_bars == {}


def test_fx03_constants_are_the_specified_bounds() -> None:
    assert se._ANCHOR_FETCH_MAX_ATTEMPTS == 3
    assert se._ANCHOR_FETCHES_PER_PASS == 1
    assert hasattr(BracketExitManager, "arm_interim")
