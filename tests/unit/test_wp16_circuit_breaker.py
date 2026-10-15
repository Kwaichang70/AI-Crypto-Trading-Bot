"""
tests/unit/test_wp16_circuit_breaker.py
-----------------------------------------
WP1.6 -- ``CircuitBreaker.evaluate(RiskSnapshot)`` state machine (spec §5.2).

    WP16-T-01  thresholds: exact Decimal bands from the run's RiskParameters
    WP16-T-02  validation (band order, multiplier, N, D, non-finite)
    WP16-T-03  HALT boundary and the inverted top band (red@HEAD)
    WP16-T-04  parity with the WP1.5 gates + property "gate blocks => CB blocks"
    WP16-T-05  day reset only on ``snapshot.day`` (red@HEAD: wall clock) + SYN-01
    WP16-T-06  state_unknown / basis <= 0 block immediately
    WP16-T-07  E-6: the daily threshold is recomputed from every snapshot
    WP16-T-08  REDUCE hysteresis
    WP16-T-09  HALT auto-recovery (U-1): steps down to REDUCE, never to OK
    WP16-T-10  immediate escalation, anti-flap
    WP16-T-11  precedence and ``active_conditions``
    WP16-T-12  band initialisation on the first evaluation
    WP16-T-13  non-mutation of the portfolio
    WP16-T-14  type contract and transactionality (SYN-06)
    WP16-T-15  transitions and listeners
"""

from __future__ import annotations

import copy
import random
from datetime import date, datetime
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import Clock, portfolio_with_position
from tests.unit._wp15_support import D as WP15_D
from tests.unit._wp16_support import (
    BASIS,
    D1,
    D2,
    D3,
    LIVE,
    PAPER,
    PEAK,
    levels,
    live_cb,
    paper_cb,
    run,
    snap,
    warm,
)
from trading.portfolio import PortfolioAccounting, RiskSnapshot
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.safety import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResponse,
    CircuitBreakerThresholds,
    CircuitBreakerTransition,
    build_run_circuit_breaker,
)

OK = CircuitBreakerResponse.OK
REDUCE = CircuitBreakerResponse.REDUCE
DAILY = CircuitBreakerResponse.DAILY_LIMIT
HALT = CircuitBreakerResponse.HALT


def _thr(**over: object) -> CircuitBreakerThresholds:
    base: dict[str, object] = {
        "max_daily_loss_pct": Decimal("0.08"),
        "halt_entry": Decimal("0.20"),
        "halt_exit": Decimal("0.18"),
        "reduce_entry": Decimal("0.10"),
        "reduce_exit": Decimal("0.08"),
        "reduce_multiplier": Decimal("0.5"),
        "deescalation_evaluations": 3,
    }
    base.update(over)
    return CircuitBreakerThresholds(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T-01
# ---------------------------------------------------------------------------


def test_wp16_t01_thresholds_live_and_paper_are_exact_decimals() -> None:
    live = CircuitBreakerThresholds.from_risk_parameters(LIVE)
    assert live == _thr()
    assert live.halt_exit == Decimal("0.18")  # SYN-02: float 0.2 * 0.9 is 0.18000000000000002
    assert str(live.halt_exit) == "0.18"
    paper = CircuitBreakerThresholds.from_risk_parameters(PAPER)
    assert (paper.halt_entry, paper.halt_exit) == (Decimal("0.30"), Decimal("0.27"))
    assert (paper.reduce_entry, paper.reduce_exit) == (Decimal("0.15"), Decimal("0.12"))
    assert paper.max_daily_loss_pct == Decimal("0.08")
    assert paper.reduce_multiplier == Decimal("0.5")
    assert paper.deescalation_evaluations == 3


def test_wp16_t01_changed_limit_propagates_and_helper_sets_the_config_mirror() -> None:
    from dataclasses import replace

    params = replace(RiskParameters(), max_drawdown_pct=0.25, max_daily_loss_pct=0.04)
    t = CircuitBreakerThresholds.from_risk_parameters(params)
    assert (t.halt_entry, t.halt_exit) == (Decimal("0.25"), Decimal("0.225"))
    assert (t.reduce_entry, t.reduce_exit) == (Decimal("0.125"), Decimal("0.100"))
    assert t.max_daily_loss_pct == Decimal("0.04")

    cb = build_run_circuit_breaker("run-x", params)
    assert cb.thresholds == t
    assert cb.config.max_drawdown_pct == 0.25
    assert cb.config.max_daily_loss_pct == 0.04
    assert cb.config.reduce_drawdown_pct == float(t.reduce_entry)
    assert cb.config.reduce_position_multiplier == 0.5
    assert cb.state.thresholds["halt_exit"] == "0.225"


# ---------------------------------------------------------------------------
# T-02
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"halt_exit": Decimal("0.20")},  # halt_exit == halt_entry
        {"halt_exit": Decimal("0.21")},  # halt_exit > halt_entry
        {"reduce_entry": Decimal("0.18")},  # reduce_entry == halt_exit
        {"reduce_exit": Decimal("0.10")},  # reduce_exit == reduce_entry
        {"reduce_exit": Decimal("0")},  # not > 0
        {"halt_entry": Decimal("0.51"), "halt_exit": Decimal("0.5")},  # > 0.50
        {"max_daily_loss_pct": Decimal("0.51")},
        {"max_daily_loss_pct": Decimal("0")},
        {"reduce_multiplier": Decimal("0")},
        {"reduce_multiplier": Decimal("1.01")},
        {"deescalation_evaluations": 0},
        {"deescalation_evaluations": 51},
        {"deescalation_evaluations": True},
        {"halt_entry": Decimal("NaN")},
        {"reduce_multiplier": Decimal("Infinity")},
        {"max_daily_loss_pct": 0.08},  # float, not Decimal
    ],
)
def test_wp16_t02_invalid_thresholds_raise(over: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _thr(**over)


def test_wp16_t02_default_ratios_validate_for_every_legal_limit() -> None:
    for pct in (0.01, 0.05, 0.2, 0.3, 0.5):
        t = CircuitBreakerThresholds.from_risk_parameters(
            RiskParameters() if pct == 0.3 else _params(pct)
        )
        assert t.reduce_exit < t.reduce_entry < t.halt_exit < t.halt_entry


def _params(limit: float) -> RiskParameters:
    from dataclasses import replace

    return replace(RiskParameters(), max_drawdown_pct=limit)


def test_wp16_t02_evaluate_without_thresholds_raises_runtime_error() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(), run_id="bare")
    with pytest.raises(RuntimeError, match="build_run_circuit_breaker"):
        cb.evaluate(snap("0"))


def test_wp16_t02_out_of_range_daily_loss_fails_loudly_at_construction() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        build_run_circuit_breaker("r", replace(RiskParameters(), max_daily_loss_pct=0.6))


# ---------------------------------------------------------------------------
# T-03  (red@HEAD: check_graduated(drawdown >= max) falls through to OK)
# ---------------------------------------------------------------------------


def test_wp16_t03_drawdown_above_the_limit_is_halt_not_ok() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0.40"))
    assert d.level == HALT and d.entries_blocked and d.entry_size_multiplier == 0


def test_wp16_t03_drawdown_exactly_at_the_limit_is_halt() -> None:
    cb = warm(live_cb())
    assert cb.evaluate(snap("0.20")).level == HALT


def test_wp16_t03_just_below_the_limit_is_reduce_from_an_ok_history() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0.19999999"))
    assert d.level == REDUCE and not d.entries_blocked
    assert d.entry_size_multiplier == Decimal("0.5")


# ---------------------------------------------------------------------------
# T-04  parity with the WP1.5 gates
# ---------------------------------------------------------------------------

_DD_GRID = ["0", "0.0999999", "0.1", "0.17999999", "0.18", "0.19999999", "0.2", "0.20000001", "0.4"]
_PNL_GRID = ["0", "-799.99999999", "-800", "-800.00000001", "-5000"]


def _gate_blocks_drawdown(rm: DefaultRiskManager, s: RiskSnapshot) -> bool:
    return rm._check_drawdown(s.nav, s.peak_equity) is not None


def _gate_blocks_daily(rm: DefaultRiskManager, s: RiskSnapshot) -> bool:
    return (
        rm._check_daily_loss(s.gate_daily_pnl(s.nav), s.nav, daily_loss_basis=s.daily_loss_basis)
        is not None
    )


@pytest.mark.parametrize("dd", _DD_GRID)
@pytest.mark.parametrize("params", [LIVE, PAPER], ids=["live", "paper"])
def test_wp16_t04_raw_halt_equals_the_drawdown_gate(dd: str, params: RiskParameters) -> None:
    rm = DefaultRiskManager("parity", params=params)
    cb = warm(build_run_circuit_breaker("parity", params))
    s = snap(dd)
    d = cb.evaluate(s)
    assert (HALT in d.active_conditions) is _gate_blocks_drawdown(rm, s)
    if _gate_blocks_drawdown(rm, s):
        assert d.entries_blocked


@pytest.mark.parametrize("pnl", _PNL_GRID)
@pytest.mark.parametrize("basis", ["10000", "0", "-5"])
@pytest.mark.parametrize("source", ["run_start", "unknown"])
def test_wp16_t04_raw_daily_equals_the_daily_loss_gate(pnl: str, basis: str, source: str) -> None:
    rm = DefaultRiskManager("parity", params=LIVE)
    cb = warm(live_cb())
    s = snap("0", worst=pnl, basis=basis, source=source)  # type: ignore[arg-type]
    d = cb.evaluate(s)
    gate = _gate_blocks_daily(rm, s)
    assert (DAILY in d.active_conditions) is gate
    if gate:
        assert d.entries_blocked


def test_wp16_t04_property_gate_blocks_implies_cb_blocks_over_random_sequences() -> None:
    rng = random.Random(1606)  # noqa: S311 - seeded, deterministic test data
    rm = DefaultRiskManager("prop", params=LIVE)
    for _ in range(200):
        cb = live_cb()
        day = D1
        for _step in range(30):
            if rng.random() < 0.1:
                day = date.fromordinal(day.toordinal() + 1)
            dd = Decimal(
                rng.choice(["0", "0.05", "0.09", "0.1", "0.15", "0.18", "0.19", "0.2", "0.25"])
            )
            worst = Decimal(rng.choice(["0", "-100", "-700", "-800", "-900"]))
            s = snap(dd, day=day, worst=worst)
            d = cb.evaluate(s)
            if _gate_blocks_drawdown(rm, s) or _gate_blocks_daily(rm, s):
                assert d.entries_blocked, (dd, worst, day, d.level)


# ---------------------------------------------------------------------------
# T-05  day reset  (red@HEAD: the legacy reset reads the wall clock)
# ---------------------------------------------------------------------------


def test_wp16_t05a_daily_limit_clears_only_when_the_snapshot_day_advances() -> None:
    cb = warm(live_cb())
    assert cb.evaluate(snap("0", worst="-800")).level == DAILY
    # raw predicate false again on the same day -> still latched (running-min latch)
    assert cb.evaluate(snap("0", worst="0")).level == DAILY
    assert cb.evaluate(snap("0", worst="0", day=D2)).level == OK  # day roll


def test_wp16_t05b_the_wall_clock_has_no_influence(monkeypatch: pytest.MonkeyPatch) -> None:
    import trading.safety as safety

    class _FarFuture(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return cls(2099, 1, 1, tzinfo=tz)

    monkeypatch.setattr(safety, "datetime", _FarFuture)
    cb = warm(live_cb())
    assert cb.evaluate(snap("0", worst="-800")).level == DAILY
    assert cb.evaluate(snap("0", worst="0")).level == DAILY  # far-future wall clock: no reset
    assert cb.evaluate(snap("0", worst="0", day=D2)).level == OK


def test_wp16_t05c_syn01_a_retrip_on_day_two_is_not_cleared_intraday() -> None:
    cb = warm(live_cb())
    assert cb.evaluate(snap("0", worst="-800", day=D1)).level == DAILY
    # day 2: raw predicate true again -> day-2 latch (daily_limit_day follows the CURRENT day)
    assert cb.evaluate(snap("0", worst="-800", day=D2)).level == DAILY
    assert cb.state.daily_limit_day == D2
    # later on day 2 the raw predicate turns false: must NOT clear intraday
    assert cb.evaluate(snap("0", worst="0", day=D2)).level == DAILY
    assert cb.evaluate(snap("0", worst="0", day=D3)).level == OK


# ---------------------------------------------------------------------------
# T-06
# ---------------------------------------------------------------------------


def test_wp16_t06_state_unknown_blocks_immediately_and_clears_at_the_roll() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0", source="unknown"))
    assert d.level == DAILY and d.inputs.state_unknown
    assert cb.evaluate(snap("0", source="roll", day=D2)).level == OK


@pytest.mark.parametrize("basis", ["0", "-1"])
def test_wp16_t06_non_positive_basis_blocks_immediately(basis: str) -> None:
    cb = warm(live_cb())
    assert cb.evaluate(snap("0", basis=basis)).level == DAILY


# ---------------------------------------------------------------------------
# T-07  E-6
# ---------------------------------------------------------------------------


def test_wp16_t07_threshold_is_recomputed_from_every_snapshot() -> None:
    cb = warm(live_cb())
    d1 = cb.evaluate(snap("0", worst="-70", basis="1000"))
    assert d1.inputs.daily_threshold == Decimal("80.00") and d1.level == OK
    # same-day step-up of the day-start basis (E-6): the threshold follows it
    d2 = cb.evaluate(snap("0", worst="-90", basis="2000"))
    assert d2.inputs.daily_threshold == Decimal("160.00")
    assert d2.level == OK  # a cached 80 would have latched here
    d3 = cb.evaluate(snap("0", worst="-160", basis="2000"))
    assert d3.level == DAILY  # boundary inclusive, like the gate


# ---------------------------------------------------------------------------
# T-08 / T-09 / T-10
# ---------------------------------------------------------------------------


def test_wp16_t08_reduce_enters_immediately_holds_in_band_and_needs_three_below() -> None:
    cb = warm(live_cb())
    assert levels(run(cb, ["0.10"])) == ["reduce"]  # entry 0.5 L in one evaluation
    assert levels(run(cb, ["0.09", "0.0801"])) == ["reduce", "reduce"]  # [0.4L, 0.5L) holds
    assert cb.state.reduce_exit_progress == 0
    assert levels(run(cb, ["0.079", "0.079"])) == ["reduce", "reduce"]
    assert cb.state.reduce_exit_progress == 2
    assert levels(run(cb, ["0.085"])) == ["reduce"]  # back in the band: streak reset
    assert cb.state.reduce_exit_progress == 0
    assert levels(run(cb, ["0.079", "0.079", "0.079"])) == ["reduce", "reduce", "ok"]


def test_wp16_t08_exit_band_value_is_exclusive_below() -> None:
    cb = warm(live_cb())
    run(cb, ["0.10"])
    # exactly reduce_exit (0.08) is NOT below it -> streak does not advance
    assert levels(run(cb, ["0.08", "0.08", "0.08", "0.08"])) == ["reduce"] * 4


def test_wp16_t09_halt_auto_recovers_and_steps_down_to_reduce_never_to_ok() -> None:
    cb = warm(live_cb())
    assert levels(run(cb, ["0.20"])) == ["halt"]  # entry L in one evaluation
    # [0.9 L, L) holds: stricter than the gate, which would allow
    assert levels(run(cb, ["0.19", "0.18"])) == ["halt", "halt"]
    assert cb.state.halt_exit_progress == 0
    assert levels(run(cb, ["0.179", "0.179"])) == ["halt", "halt"]
    assert cb.state.halt_exit_progress == 2
    assert levels(run(cb, ["0.19"])) == ["halt"]  # streak reset
    assert cb.state.halt_exit_progress == 0
    # three consecutive below 0.9 L -> REDUCE, even though dd is far below 0.4 L
    assert levels(run(cb, ["0.05", "0.05", "0.05"])) == ["halt", "halt", "reduce"]
    st = cb.state
    assert st.level == "reduce" and st.halt_since is None and st.last_halt_reason is not None
    # then three below 0.4 L -> OK
    assert levels(run(cb, ["0.05", "0.05", "0.05"])) == ["reduce", "reduce", "ok"]


def test_wp16_t09_halt_implies_reduce_with_held_streak() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0.20"))
    assert d.active_conditions == (HALT, REDUCE)
    d = cb.evaluate(snap("0.05"))
    assert cb.state.reduce_exit_progress == 0  # invariant: streak held at 0 while HALT


def test_wp16_t10_escalation_takes_one_evaluation_for_every_level() -> None:
    assert warm(live_cb()).evaluate(snap("0.20")).level == HALT
    assert warm(live_cb()).evaluate(snap("0", worst="-800")).level == DAILY
    cb = warm(live_cb())
    assert cb.evaluate(snap("0.12")).level == REDUCE
    assert cb.evaluate(snap("0.25")).level == HALT  # REDUCE -> HALT in one evaluation


def test_wp16_t10_alternating_around_the_limit_produces_one_halt_entry() -> None:
    cb = warm(live_cb())
    ds = run(cb, ["0.201", "0.199"] * 10)
    assert len(ds) == 20 and set(levels(ds)) == {"halt"}
    assert cb.state.halt_count == 1
    transitions = [d.transition for d in ds if d.transition is not None]
    assert [(t.from_level, t.to_level) for t in transitions] == [(OK, HALT)]


# ---------------------------------------------------------------------------
# T-11
# ---------------------------------------------------------------------------


def test_wp16_t11_precedence_and_active_conditions_across_the_day_roll() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0.20", worst="-800"))
    assert d.level == HALT
    assert d.active_conditions == (HALT, DAILY, REDUCE)
    d = cb.evaluate(snap("0.20", worst="0", day=D2))
    assert d.level == HALT and d.active_conditions == (HALT, REDUCE)  # DAILY dropped, HALT holds
    assert cb.state.active_conditions == ["halt", "reduce"]


def test_wp16_t11_daily_limit_outranks_reduce() -> None:
    cb = warm(live_cb())
    d = cb.evaluate(snap("0.12", worst="-800"))
    assert d.level == DAILY and d.active_conditions == (DAILY, REDUCE)
    assert d.entries_blocked and d.entry_size_multiplier == 0


# ---------------------------------------------------------------------------
# T-12  band initialisation
# ---------------------------------------------------------------------------


def test_wp16_t12_fresh_cb_in_the_halt_exit_band_starts_in_halt() -> None:
    d = live_cb().evaluate(snap("0.19"))
    assert d.level == HALT
    assert d.active_conditions == (HALT, REDUCE)


def test_wp16_t12_fresh_cb_in_the_reduce_exit_band_starts_in_reduce() -> None:
    assert live_cb().evaluate(snap("0.09")).level == REDUCE
    assert live_cb().evaluate(snap("0.0799")).level == OK
    assert live_cb().evaluate(snap("0.18")).level == HALT  # exactly 0.9 L


def test_wp16_t12_fresh_cb_is_never_laxer_than_the_gate_nor_the_memoryless_bands() -> None:
    """A restart must never bypass the breaker (K-16).

    NOTE (producer deviation D-1, see producer-report): spec T-12 also states
    "never laxer than the UNINTERRUPTED CB".  That is false: after a HALT
    entered at ``dd >= L`` an uninterrupted CB holds HALT until 3 consecutive
    evaluations < 0.9 L, whereas a fresh CB whose first reading is already
    < 0.9 L starts in REDUCE (see the counter-example test below).  What does
    hold, and what is asserted here, is the memoryless floor: HALT for
    ``dd >= 0.9 L``, REDUCE for ``dd >= 0.4 L`` and never below the WP1.5 gate.
    """
    rng = random.Random(7)  # noqa: S311 - seeded, deterministic test data
    rm = DefaultRiskManager("restart", params=LIVE)
    for _ in range(300):
        series = [Decimal(str(round(rng.uniform(0, 0.3), 3))) for _ in range(rng.randint(1, 12))]
        for k in range(len(series)):
            fresh = live_cb().evaluate(snap(series[k]))
            if series[k] >= Decimal("0.18"):
                assert fresh.level == HALT
            elif series[k] >= Decimal("0.08"):
                assert fresh.level in (REDUCE, DAILY)
            if _gate_blocks_drawdown(rm, snap(series[k])):
                assert fresh.entries_blocked


def test_wp16_t12_documented_counter_example_fresh_can_be_laxer_than_uninterrupted() -> None:
    uninterrupted = warm(live_cb())
    run(uninterrupted, ["0.21"])  # HALT
    held = uninterrupted.evaluate(snap("0.17"))  # first reading < 0.9 L: streak 1 of 3
    fresh = live_cb().evaluate(snap("0.17"))
    assert held.level == HALT and fresh.level == REDUCE
    # bounded (WP1.6 fix2): from the 2N-th evaluation on, never laxer (T-46, T-48)
    # ... but the gate allows 0.17 either way, so the WP1.5 gate is never bypassed
    assert not _gate_blocks_drawdown(DefaultRiskManager("x", params=LIVE), snap("0.17"))


# ---------------------------------------------------------------------------
# T-13  non-mutation
# ---------------------------------------------------------------------------


def _portfolio_state(p: PortfolioAccounting) -> dict[str, object]:
    return {
        "peak": p._peak_equity,
        "worst": p._worst_nav_pnl,
        "day": p._daily_pnl_date,
        "source": p._day_start_source,
        "start": p._nav_day_start,
        "pending": p._day_start_pending,
        "summary": p.get_summary(),
        "snapshot": p.risk_snapshot(),
    }


def test_wp16_t13_hundred_evaluations_do_not_mutate_the_portfolio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock(WP15_D)
    p = portfolio_with_position(clock)
    p.update_market_prices({"BTC/USDT": Decimal("900")})
    clock.at = clock.at.replace(day=11, hour=0, minute=5)  # a pending (un-rolled) day roll
    before = copy.deepcopy(_portfolio_state(p))
    cb = live_cb()

    def _boom(self: PortfolioAccounting) -> object:
        raise AssertionError("the circuit breaker must never read get_summary()")

    monkeypatch.setattr(PortfolioAccounting, "get_summary", _boom)
    for _ in range(100):
        cb.evaluate(p.risk_snapshot())
    monkeypatch.undo()
    assert _portfolio_state(p) == before
    assert cb.state.evaluations == 100


def test_wp16_t13_evaluate_takes_only_the_snapshot() -> None:
    import inspect

    params = list(inspect.signature(CircuitBreaker.evaluate).parameters)
    assert params == ["self", "snapshot"]


# ---------------------------------------------------------------------------
# T-14  type contract + transactionality
# ---------------------------------------------------------------------------


def _frozen_state(cb: CircuitBreaker) -> dict[str, object]:
    return {k: copy.copy(v) for k, v in vars(cb).items() if k not in ("_log", "_listeners")}


def test_wp16_t14_a_dict_is_rejected() -> None:
    cb = warm(live_cb())
    with pytest.raises(TypeError):
        cb.evaluate({"drawdown_pct": 0.5})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["drawdown_pct", "nav", "peak_equity", "daily_loss_basis", "worst_nav_pnl"]
)
def test_wp16_t14_non_finite_input_raises_and_leaves_every_field_unchanged(field: str) -> None:
    import dataclasses

    cb = warm(live_cb())
    cb.evaluate(snap("0.20"))  # non-trivial state: HALT, counters
    before = _frozen_state(cb)
    bad = dataclasses.replace(snap("0.05"), **{field: Decimal("NaN")})
    with pytest.raises(ValueError):
        cb.evaluate(bad)
    assert _frozen_state(cb) == before
    assert cb.state.evaluations == 7  # 2N warm-up + 1


def test_wp16_t14_halt_since_is_the_risk_clock_not_the_wall_clock() -> None:
    cb = warm(live_cb())
    s = snap("0.20")
    d = cb.evaluate(s)
    assert cb.state.halt_since == s.as_of
    assert d.transition is not None and d.transition.at == s.as_of
    assert cb.state.last_evaluated_at == s.as_of


# ---------------------------------------------------------------------------
# T-15  transitions + listeners
# ---------------------------------------------------------------------------


def test_wp16_t15_exactly_one_transition_per_level_change() -> None:
    cb = live_cb()
    seen: list[CircuitBreakerTransition] = []
    cb.add_transition_listener(seen.append)
    ds = run(cb, ["0", "0", "0.12", "0.12", "0.25", "0.25", "0.05", "0.05", "0.05"])
    assert [(t.from_level, t.to_level) for t in seen] == [
        (OK, REDUCE),
        (REDUCE, HALT),
        (HALT, REDUCE),
    ]
    assert [d.transition for d in ds if d.transition is not None] == seen
    assert all(t.run_id == "wp16-live" and t.thresholds == cb.thresholds for t in seen)
    assert seen[1].at == snap("0.25").as_of and seen[1].drawdown_pct == Decimal("0.25")


def test_wp16_t15_a_raising_listener_is_logged_and_does_not_affect_the_decision() -> None:
    cb = live_cb()
    ran: list[str] = []

    def bad(_t: CircuitBreakerTransition) -> None:
        raise RuntimeError("listener boom")

    cb.add_transition_listener(bad)
    cb.add_transition_listener(lambda t: ran.append(t.to_level.value))
    with capture_logs() as logs:
        d = cb.evaluate(snap("0.25"))
    assert d.level == HALT and ran == ["halt"]  # decision unaffected, other listeners still run
    failed = [e for e in logs if e["event"] == "circuit_breaker.listener_failed"]
    assert len(failed) == 1 and failed[0]["log_level"] == "error"
    assert "payload" not in failed[0]


def test_wp16_t15_halt_events_are_emitted_once_per_entry_and_exit() -> None:
    cb = warm(live_cb())
    with capture_logs() as logs:
        run(cb, ["0.25", "0.25", "0.25", "0.05", "0.05", "0.05"])
    events = [e["event"] for e in logs]
    assert events.count("circuit_breaker.halt_entered") == 1
    assert events.count("circuit_breaker.halt_cleared") == 1
    entered = next(e for e in logs if e["event"] == "circuit_breaker.halt_entered")
    assert entered["log_level"] == "critical" and entered["alert"] == "ENTRIES_BLOCKED"
    cleared = next(e for e in logs if e["event"] == "circuit_breaker.halt_cleared")
    assert cleared["next_level"] == "reduce" and cleared["log_level"] == "warning"


def test_wp16_t15_state_exposes_the_typed_contract_with_legacy_fields_intact() -> None:
    cb = warm(live_cb())
    cb.evaluate(snap("0.25"))
    st = cb.state
    assert st.is_tripped is True and st.graduated_response == "halt" and st.level == "halt"
    assert st.entries_blocked and st.entry_size_multiplier == "0"
    assert st.halt_count == 1 and st.evaluations == 7  # 2N warm-up + 1
    assert Decimal(st.thresholds["halt_entry"]) == Decimal("0.20") and st.inputs is not None
    assert Decimal(st.inputs["drawdown_pct"]) == Decimal("0.25")
    assert st.trip_count == 0 and st.trip_reason is None  # legacy manual-trip meaning kept


def test_wp16_t15_paper_limits_are_the_papers() -> None:
    cb = warm(paper_cb())
    assert cb.evaluate(snap("0.25")).level == REDUCE  # 0.25 < paper 0.30
    assert cb.evaluate(snap("0.30")).level == HALT
    assert PEAK > BASIS  # support-module sanity: nav stays positive for dd < 1
