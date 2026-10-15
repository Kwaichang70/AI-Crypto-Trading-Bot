"""
tests/unit/test_wp16_restart_watch.py
-------------------------------------
WP1.6 fix2 (final synthesis round 2, CF-35 user decision "stricter"): the
restart watch of ``CircuitBreaker.evaluate``.  Live bands: L = 0.20,
halt_exit 0.18, reduce_entry 0.10, reduce_exit 0.08, N = 3.

    T-46  property: after any history and a restart at any point, a fresh
          breaker is never laxer than the uninterrupted one from the 2N-th
          post-restart evaluation on; before that only at dd < 0.9 L, for at
          most 2N-1 evaluations; never laxer than the WP1.5 gate
    T-47  band boundaries of the restart watch (entry, HALT watch, REDUCE
          watch, expiry, no re-arm, DAILY_LIMIT, state field)
    T-48  the WP16-SEC-206 counter-examples A, B, C and the ARB-01 monotone
          case, pinned
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from tests.unit._wp16_support import live_cb, run, snap, warm
from trading.safety import CircuitBreaker, CircuitBreakerDecision, CircuitBreakerResponse

OK = CircuitBreakerResponse.OK
REDUCE = CircuitBreakerResponse.REDUCE
DAILY = CircuitBreakerResponse.DAILY_LIMIT
HALT = CircuitBreakerResponse.HALT

N = 3
L, HALT_EXIT, REDUCE_EXIT = Decimal("0.20"), Decimal("0.18"), Decimal("0.08")

#: Band representatives and every band edge (both sides), live limits.
_VALUES = [
    "0",
    "0.05",
    "0.0799",
    "0.08",
    "0.09",
    "0.0999",
    "0.10",
    "0.12",
    "0.1799",
    "0.18",
    "0.19",
    "0.1999",
    "0.20",
    "0.21",
    "0.25",
]


def _levels(cb: CircuitBreaker, dds: list[str]) -> list[str]:
    return [d.level.value for d in run(cb, dds)]


# ---------------------------------------------------------------------------
# T-46  property
# ---------------------------------------------------------------------------


def _evaluate(cb: CircuitBreaker, dd: str, worst: str) -> CircuitBreakerDecision:
    return cb.evaluate(snap(dd, worst=worst))


def test_wp16_t46_fresh_breaker_is_never_laxer_than_uninterrupted_after_2n_minus_1() -> None:
    rng = random.Random(1606)  # noqa: S311 - seeded, deterministic test data
    laxer_seen = 0
    for _ in range(600):
        history = [rng.choice(_VALUES) for _ in range(rng.randint(0, 12))]
        after = [rng.choice(_VALUES) for _ in range(rng.randint(1, 12))]
        after += [rng.choice(_VALUES)] * rng.randint(0, 8)  # a steady tail
        # day-loss latch: worst-of-day only falls (as PortfolioAccounting keeps it)
        worst, worsts = Decimal(0), []
        for _ in history + after:
            worst = min(worst, Decimal(rng.choice(["0", "0", "0", "-500", "-800"])))
            worsts.append(str(worst))
        uninterrupted = live_cb()
        if rng.random() < 0.8:  # mostly a long-running breaker; sometimes a fresh one
            warm(uninterrupted)
        for dd, w in zip(history, worsts, strict=False):
            _evaluate(uninterrupted, dd, w)
        fresh = live_cb()
        laxer = 0
        for j, dd in enumerate(after):
            w = worsts[len(history) + j]
            u = _evaluate(uninterrupted, dd, w)
            f = _evaluate(fresh, dd, w)
            ctx = (history, after, j)
            if Decimal(dd) >= L:  # never laxer than the WP1.5 gate
                assert f.entries_blocked, ctx
            if f.entry_size_multiplier > u.entry_size_multiplier:
                laxer += 1
                assert j < 2 * N - 1, ctx  # only during the first 2N-1 evaluations
                assert Decimal(dd) < HALT_EXIT, ctx
                if f.entry_size_multiplier == 1:
                    assert Decimal(dd) < REDUCE_EXIT, ctx
        assert laxer <= 2 * N - 1
        laxer_seen += laxer
    assert laxer_seen > 0  # the bounded transient is exercised, not vacuous


# ---------------------------------------------------------------------------
# T-47  band boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dd", "level"),
    [("0.20", HALT), ("0.18", HALT), ("0.1799", REDUCE), ("0.08", REDUCE), ("0.0799", OK)],
)
def test_wp16_t47_first_evaluation_uses_the_recovery_bands(dd: str, level: object) -> None:
    assert live_cb().evaluate(snap(dd)).level == level


@pytest.mark.parametrize(
    ("dd", "level"),
    [("0.18", HALT), ("0.1799", REDUCE), ("0.08", REDUCE), ("0.0799", OK)],
)
def test_wp16_t47_halt_watch_still_active_on_the_nth_evaluation(dd: str, level: object) -> None:
    cb = live_cb()
    run(cb, ["0.05"] * (N - 1))  # N-1 calm evaluations: the HALT watch is still on
    assert cb.state.restart_watch == "halt"
    assert cb.evaluate(snap(dd)).level == level


def test_wp16_t47_halt_watch_ends_after_n_evaluations() -> None:
    cb = live_cb()
    run(cb, ["0.1799"] * N)  # REDUCE (watch); the HALT watch ends after N evaluations
    assert cb.state.restart_watch == "reduce"
    assert cb.evaluate(snap("0.1999")).level == REDUCE  # entry semantics again
    assert cb.evaluate(snap("0.20")).level == HALT


def test_wp16_t47_halt_entered_in_the_watch_is_held_by_the_ordinary_hysteresis() -> None:
    cb = live_cb()
    assert _levels(cb, ["0.05", "0.05", "0.18"]) == ["ok", "ok", "halt"]  # N-th evaluation
    assert cb.state.restart_watch == "reduce"  # the HALT watch has ended ...
    # ... but the HALT holds until N evaluations below 0.9 L, then steps down
    assert _levels(cb, ["0.19", "0.1799", "0.1799", "0.1799"]) == ["halt"] * 3 + ["reduce"]
    assert cb.state.restart_watch == "none"


def test_wp16_t47_reduce_watch_lasts_2n_evaluations() -> None:
    cb = live_cb()
    run(cb, ["0.05"] * N)  # HALT watch ends after the N-th evaluation
    assert cb.state.restart_watch == "reduce"
    run(cb, ["0.05"] * (N - 1))  # 2N-1 evaluations done
    assert cb.state.restart_watch == "reduce"
    assert cb.evaluate(snap("0.08")).level == REDUCE  # 0.4 L still enters REDUCE


@pytest.mark.parametrize(
    ("dd", "level"),
    [("0.08", OK), ("0.0999", OK), ("0.10", REDUCE), ("0.1999", REDUCE), ("0.20", HALT)],
)
def test_wp16_t47_after_2n_calm_evaluations_the_entry_bands_apply(dd: str, level: object) -> None:
    cb = live_cb()
    run(cb, ["0.05"] * (2 * N))
    assert cb.state.restart_watch == "none"
    assert cb.evaluate(snap(dd)).level == level


def test_wp16_t47_an_ended_watch_is_never_re_armed() -> None:
    cb = live_cb()
    run(cb, ["0.05"] * (2 * N))
    assert _levels(cb, ["0.25", "0.05", "0.05", "0.05"]) == ["halt", "halt", "halt", "reduce"]
    assert cb.state.restart_watch == "none"
    run(cb, ["0.05"] * N)  # REDUCE -> OK
    assert _levels(cb, ["0.09", "0.19"]) == ["ok", "reduce"]  # entry bands, not exit bands


def test_wp16_t47_reduce_watch_outlasts_the_halt_watch() -> None:
    """An uninterrupted HALT steps down to REDUCE with streak 0, so REDUCE needs 2N."""
    cb = live_cb()
    run(cb, ["0.05"] * N)  # a REDUCE watch as short as the HALT watch would have ended
    assert cb.evaluate(snap("0.09")).level == REDUCE


def test_wp16_t47_daily_limit_on_the_first_evaluation_follows_the_gate_state() -> None:
    d = live_cb().evaluate(snap("0.05", worst="-800"))
    assert d.level == DAILY and d.entries_blocked
    assert live_cb().evaluate(snap("0.05", worst="-799.99")).level == OK
    d = live_cb().evaluate(snap("0.09", worst="-800"))
    assert d.level == DAILY and d.active_conditions == (DAILY, REDUCE)


def test_wp16_t47_reason_names_the_restart_watch() -> None:
    cb = live_cb()
    d = cb.evaluate(snap("0.19"))
    assert d.reason == "drawdown 0.19 >= halt exit band 0.18 (restart watch)"
    assert cb.state.last_halt_reason == d.reason
    assert d.transition is not None and d.transition.reason == d.reason
    assert live_cb().evaluate(snap("0.20")).reason == "drawdown 0.20 >= limit 0.2"


def test_wp16_t47_state_reports_the_watch() -> None:
    cb = live_cb()
    assert cb.state.restart_watch == "halt"
    assert warm(cb).state.restart_watch == "none"


# ---------------------------------------------------------------------------
# T-48  WP16-SEC-206 counter-examples and the ARB-01 monotone case
# ---------------------------------------------------------------------------


def _pair(history: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    uninterrupted = warm(live_cb())
    run(uninterrupted, history)
    return _levels(uninterrupted, after), _levels(live_cb(), after)


def test_wp16_t48_case_a_reduce_instead_of_halt_is_closed() -> None:
    u, f = _pair(["0.25"], ["0.12"] + ["0.19"] * 20)
    assert u == ["halt"] * 21
    assert f == ["reduce"] + ["halt"] * 20  # 1 transient evaluation, then equal


def test_wp16_t48_case_b_ok_instead_of_reduce_is_closed() -> None:
    u, f = _pair(["0.25", "0.05", "0.05", "0.05"], ["0.05"] + ["0.09"] * 20)
    assert u == ["reduce"] * 21
    assert f == ["ok"] + ["reduce"] * 20


def test_wp16_t48_case_c_ok_instead_of_reduce_without_halt_is_closed() -> None:
    u, f = _pair(["0.12"], ["0.07"] + ["0.09"] * 20)
    assert u == ["reduce"] * 21
    assert f == ["ok"] + ["reduce"] * 20


def test_wp16_t48_arb01_monotone_recovery_is_the_bounded_transient() -> None:
    u, f = _pair(["0.21"], ["0.05"] * 8)
    assert u == ["halt", "halt", "reduce", "reduce", "reduce", "ok", "ok", "ok"]
    assert f == ["ok"] * 8  # 2N-1 = 5 laxer evaluations, all at dd < 0.4 L, then equal
