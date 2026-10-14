"""
tests/integration/test_wp16_cb_backtest.py
--------------------------------------------
WP1.6 / U-2 -- the circuit breaker in backtests is OPT-IN (default off).

    WP16-T-32  the default ``BacktestRunner`` builds no breaker and a reference
               backtest is byte-identical to pre-WP1.6 (golden digest taken at HEAD 206eb12;
               RE-PINNED once by WP2.1a-1 for the C25 fix, see GOLDEN_REFERENCE_DIGEST)
    WP16-T-33  ``enable_circuit_breaker=True`` through DAILY_LIMIT -> REDUCE -> OK
               and HALT -> REDUCE -> OK across 00:00 UTC: identical transitions, fills,
               equity and ``halt_since`` under two different wall clocks

The breaker's clock is the portfolio's risk clock: ``StrategyEngine.run_backtest``
pushes the bar CLOSE time (``advance_clock``), so the wall clock is never read.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

import trading.backtest as backtest_module
import trading.portfolio as portfolio_module
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.risk import RiskParameters
from trading.safety import CircuitBreaker, CircuitBreakerTransition

SYMBOL = "BTC/USDT"
INITIAL = Decimal("10000")
FIRST_OPEN = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

#: 30% position that loses 52% (-15.6% NAV): DAILY_LIMIT on D (>8%), REDUCE after the roll
#: (>= 15%), recovery to OK; later BUYs sized so the cluster cap never decides (``_RISK`` sets it
#: to 1.0).  Since the C25 fix (WP2.1a-1) ``BacktestRunner`` forwards EVERY ``risk_params`` field,
#: including ``max_cluster_exposure_pct``; before it the cap was silently reset to 0.40.
_RD_CLOSES = [100, 100, 80, 48] + [48] * 10 + [48, 100] + [100] * 8
_RD_SCRIPT = {0: "buy", 14: "buy", 21: "buy"}
_RD_TARGETS = {0: "3000", 14: "300", 21: "600"}

#: 38% position that loses 85% (-32% NAV): HALT (paper limit 30%), then recovery.
_HALT_CLOSES = [100, 100, 15] + [15] * 5 + [100] * 16
_HALT_SCRIPT = {0: "buy", 5: "buy", 14: "buy"}
_HALT_TARGETS = {0: "3800", 5: "300", 14: "300"}

_RISK = RiskParameters(
    max_position_size_pct=0.9, max_portfolio_exposure_pct=1.0, max_cluster_exposure_pct=1.0
)

SCENARIOS: dict[str, tuple[list[int], dict[int, str], dict[int, str]]] = {
    "reduce_daily": (_RD_CLOSES, _RD_SCRIPT, _RD_TARGETS),
    "halt": (_HALT_CLOSES, _HALT_SCRIPT, _HALT_TARGETS),
}


def _runner(scenario: str, *, enable: bool) -> tuple[BacktestRunner, dict[str, list[Any]]]:
    closes, script, targets = SCENARIOS[scenario]
    strategy = ScriptedSequenceStrategy(
        "wp16-bt", {"script": script, "target_notional": "3000", "targets": targets}
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        risk_params=_RISK,
        seed=7,
        allow_pyramiding=True,
        # WP2.1: the goldens below are calibrated on the pre-WP2.1 backtest costs
        # (maker 10 / taker 15 / slippage 5 bps); pinned explicitly because the default
        # profile moved to coinbase_advanced_1.  Both T-32 scenarios and every T-33
        # assertion run under this one profile, so the C25 re-pin stays single-cause.
        cost_profile=get_cost_profile("legacy_backtest_v0"),
        **({"enable_circuit_breaker": True} if enable else {}),
    )
    bars = scripted_path(closes, first_open=FIRST_OPEN, symbol=SYMBOL, timeframe=TimeFrame.ONE_HOUR)
    return runner, {SYMBOL: bars}


async def _run(
    scenario: str, *, enable: bool, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    transitions: list[CircuitBreakerTransition] = []
    breakers: list[CircuitBreaker] = []
    original = backtest_module.build_run_circuit_breaker

    def _spy(run_id: str, params: RiskParameters) -> CircuitBreaker:
        cb = original(run_id, params)
        cb.add_transition_listener(transitions.append)
        breakers.append(cb)
        return cb

    monkeypatch.setattr(backtest_module, "build_run_circuit_breaker", _spy)
    runner, bars = _runner(scenario, enable=enable)
    result = await runner.run(bars)
    assert runner.last_execution_engine is not None
    fills = [
        (f.side.value, str(f.quantity), str(f.price))
        for f in runner.last_execution_engine.get_all_fills()
    ]
    return {
        "fills": fills,
        # equity VALUES only: the portfolio stamps its curve points from the wall clock
        # (pre-existing; WP2.2 moves them onto the risk clock), the values are deterministic.
        "curve": [str(p.equity) for p in result.equity_curve],
        "final_equity": str(result.final_equity),
        "transitions": [
            (t.from_level.value, t.to_level.value, str(t.at), str(t.drawdown_pct), t.reason)
            for t in transitions
        ],
        "breakers": breakers,
    }


def _digest(payload: dict[str, Any]) -> str:
    stable = {k: payload[k] for k in ("fills", "curve", "final_equity")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


# Golden digest of the reference backtest (default: no breaker).
#
# History: taken at HEAD 206eb12 (BEFORE any WP1.6 change) as
# ``b130c79103652a460a093d4c28997de2ce2d56e3f1de5a434c7f8295b21f9eaa``.  RE-PINNED ONCE, with a
# single cause, by WP2.1a-1 (C25): the old ``BacktestRunner`` rebuilt a caller-supplied
# ``RiskParameters`` by hand and dropped ``max_cluster_exposure_pct`` (reset to 0.40), so the
# 6.25-unit REDUCE-sized BUY at 48.024 was rejected by the cap.  With the field forwarded that
# BUY is filled -- an extra fill and a different final equity (9992.7973 -> 10317.1970750).
# The explicit fill lists below are asserted BEFORE the digest is compared, so the re-pin cannot
# hide an unintended change.  If this fails, the default backtest changed (U-2).
GOLDEN_REFERENCE_DIGEST = "5990ddc54bec8ff1bf9c3a46a52b66b3575cb4584b26d86650f7bca3bddae48d"

#: Explicit post-C25 fills (side, quantity, price) of the two scenarios with the breaker OFF.
_REDUCE_DAILY_FILLS_BREAKER_OFF = [
    ("buy", "30.00000000", "100.05000000"),
    ("buy", "6.25000000", "48.02400000"),
    ("buy", "6.00000000", "100.05000000"),
]
_HALT_FILLS_BREAKER_OFF = [
    ("buy", "38.00000000", "100.05000000"),
    ("buy", "3.00000000", "100.05000000"),
]


async def test_wp16_t32_default_runner_builds_no_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _bars = _runner("reduce_daily", enable=False)
    engine, _pf, _rm, _ex = runner._build_engine("t32")
    assert engine.circuit_breaker is None
    runner_on, _ = _runner("reduce_daily", enable=True)
    engine_on, *_ = runner_on._build_engine("t32-on")
    assert engine_on.circuit_breaker is not None
    t = engine_on.circuit_breaker.thresholds
    assert t is not None and t.halt_entry == Decimal("0.3")  # paper/backtest default limit


async def test_wp16_t32_reference_backtest_matches_the_c25_repinned_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _run("reduce_daily", enable=False, monkeypatch=monkeypatch)
    assert out["breakers"] == [] and out["transitions"] == []
    # C25: the explicit fills first (the extra 6.25 @ 48.024 BUY is the cluster cap no longer
    # being reset), THEN the digest.
    assert out["fills"] == _REDUCE_DAILY_FILLS_BREAKER_OFF
    assert out["final_equity"] == "10317.1970750000000000"
    assert _digest(out) == GOLDEN_REFERENCE_DIGEST
    again = await _run("reduce_daily", enable=False, monkeypatch=monkeypatch)
    assert _digest(again) == _digest(out)  # deterministic run to run


async def test_wp16_t32_halt_scenario_breaker_off_fills_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C25: with the breaker off the halt scenario also gains the 3 @ 100.05 BUY."""
    out = await _run("halt", enable=False, monkeypatch=monkeypatch)
    assert out["breakers"] == [] and out["transitions"] == []
    assert out["fills"] == _HALT_FILLS_BREAKER_OFF
    assert out["final_equity"] == "9991.7969250000000000"


def _clock(at: datetime) -> Any:
    return lambda: at


@pytest.mark.parametrize("scenario", ["reduce_daily", "halt"])
async def test_wp16_t33_enabled_breaker_is_deterministic_under_any_wall_clock(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    walls = (
        datetime(2031, 6, 1, 23, 59, 59, tzinfo=UTC),
        datetime(1999, 1, 1, 12, 0, tzinfo=UTC),
    )
    for wall in walls:
        with monkeypatch.context() as mp:
            mp.setattr(portfolio_module, "_utc_now", _clock(wall))
            runs.append(await _run(scenario, enable=True, monkeypatch=mp))
    a, b = runs
    assert a["transitions"] == b["transitions"] and a["transitions"]
    assert a["fills"] == b["fills"]
    assert a["curve"] == b["curve"]
    assert a["final_equity"] == b["final_equity"]
    state_a = a["breakers"][0].state.model_dump(mode="json")
    state_b = b["breakers"][0].state.model_dump(mode="json")
    assert state_a == state_b  # includes halt_since / last_halt_reason / evaluations
    assert state_a["halt_since"] == state_b["halt_since"]


async def test_wp16_t33_reduce_daily_scenario_crosses_the_levels_and_halves_the_reduce_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _run("reduce_daily", enable=True, monkeypatch=monkeypatch)
    path = [(t[0], t[1]) for t in out["transitions"]]
    assert path == [("ok", "daily_limit"), ("daily_limit", "reduce"), ("reduce", "ok")]
    # the DAILY_LIMIT transition is on day D, the clear exactly at the 00:00 UTC bar close
    assert out["transitions"][0][2].startswith("2026-03-10 14:00")
    assert out["transitions"][1][2].startswith("2026-03-11 00:00")
    fills = out["fills"]
    assert [f[0] for f in fills] == ["buy", "buy", "buy"]
    # the REDUCE-band BUY (target 300 at price 48 = 6.25) is halved: 3.125
    assert Decimal(fills[1][1]) == Decimal("3.125")
    assert Decimal(fills[2][1]) == Decimal("6")  # recovered: full size (600 / 100)


async def test_wp16_t33_halt_scenario_halts_recovers_and_never_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _run("halt", enable=True, monkeypatch=monkeypatch)
    path = [(t[0], t[1]) for t in out["transitions"]]
    assert path[0] == ("ok", "halt")
    assert ("halt", "ok") not in path  # HALT never clears straight to OK
    assert path[-1][1] == "ok"  # ... but it does clear by itself (U-1)
    cb = out["breakers"][0]
    assert cb.current_response.value == "ok"
    assert cb.state.halt_count == 1 and cb.state.evaluations > 20  # processing continued
