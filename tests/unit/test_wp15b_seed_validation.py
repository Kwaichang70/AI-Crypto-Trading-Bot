"""
tests/unit/test_wp15b_seed_validation.py
------------------------------------------
WP1.5b -- strict seed validation and the never-abort guarantee of the resume
daily-risk-state seed (final-synthesis-1.5a.md §A.7 R-1 / R-2; security
SEC-05 / SEC-208).

    WP15-T-43  every seed field is type-validated strictly (NO coercion): a
               float / int / str / bool, a NaN / infinite / negative Decimal,
               a ``datetime`` ``as_of_day``, a naive or non-datetime
               ``run_started_at`` and a non-seed object all degrade to
               ``unknown`` (``reason="seed_invalid"``); a seeding exception
               degrades the same way.  After EVERY case the resume has
               succeeded, ``risk_snapshot()`` works with Decimal money
               fields, a mark works, a reducing SELL fill works and a BUY is
               blocked by the real ``_check_daily_loss``.
    WP15-T-44  ``ResumeSnapshot.daily_seed`` builder: ``None`` iff the hints
               were never read, never raises, never returns ``None`` once
               hints were read, passes values through unmodified and degrades
               a construction failure to a day-only seed.
    WP15-T-45  snapshot -> seed -> ``from_fills`` end to end (AC-10 through
               the ``ResumeSnapshot`` value object).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import INITIAL, SYMBOL, Clock, D, make_fill, mark
from tests.unit.test_wp15a_resume_seed import _gate, _rebuild
from trading import portfolio as portfolio_module
from trading import recovery as recovery_module
from trading.portfolio import PortfolioAccounting
from trading.recovery import ResumeDailySeed, ResumeSnapshot
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager

_TODAY = D.date()
_STARTED = D - timedelta(days=5)
#: Distinctive so a leak of the raw value into a log event is detectable.
_MARK = "123456.789"


def _seed(**overrides: Any) -> ResumeDailySeed:
    fields: dict[str, Any] = {
        "as_of_day": _TODAY,
        "day_start_equity": Decimal("10000"),
        "day_min_equity": Decimal("9900"),
        "run_started_at": _STARTED,
        "day_first_equity": Decimal("10000"),
    }
    fields.update(overrides)
    return ResumeDailySeed(**fields)


def _assert_degraded_and_operable(
    p: PortfolioAccounting, logs: list[dict[str, Any]], *, field: str | None = None
) -> None:
    """The R-2 'every case asserts' block (a resume that merely succeeds is not enough)."""
    # -- the seeding outcome
    assert p._day_start_pending is False
    snap = p.risk_snapshot()  # must not raise
    assert snap.day_start_source == "unknown" and snap.state_unknown
    events = [e for e in logs if e["event"] == "resume.daily_state_unknown"]
    assert len(events) == 1, logs
    assert events[0]["reason"] == "seed_invalid"
    assert events[0]["log_level"] == "error"
    if field is not None:
        assert events[0]["field"] == field
    assert not [e for e in logs if e["event"] == "resume.risk_state_seeded"]
    assert _MARK not in repr(logs), "the offending value must never be logged"

    # -- every money field is a Decimal (a stored float would raise later)
    for name in (
        "nav",
        "peak_equity",
        "drawdown_pct",
        "nav_day_start",
        "daily_loss_basis",
        "nav_pnl",
        "worst_nav_pnl",
        "realised_pnl",
    ):
        assert isinstance(getattr(snap, name), Decimal), name
    assert isinstance(p._nav_day_start, Decimal) and isinstance(p._worst_nav_pnl, Decimal)

    # -- BUYs are blocked by the REAL daily-loss rule (gate = -basis), exits are not
    manager = DefaultRiskManager(run_id="wp15b", params=RiskParameters())
    assert (
        manager._check_daily_loss(snap.gate_daily_pnl(snap.nav), snap.nav, snap.daily_loss_basis)
        is not None
    )
    assert _gate(p, OrderSide.BUY) is False
    assert _gate(p, OrderSide.SELL) is True

    # -- a mark works and the state stays unknown
    mark(p, "1100")
    after_mark = p.risk_snapshot()
    assert after_mark.state_unknown and isinstance(after_mark.nav_pnl, Decimal)

    # -- a stop-loss / reducing SELL fill works (the protective path)
    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="10", price="900"),
        current_price=Decimal("900"),
    )
    position = p.get_position(SYMBOL)
    assert position is not None and position.is_flat
    final = p.risk_snapshot()
    assert final.state_unknown and isinstance(final.worst_nav_pnl, Decimal)


# ---------------------------------------------------------------------------
# WP15-T-43  strict validation, no coercion
# ---------------------------------------------------------------------------

_BAD_MONEY = [
    pytest.param(float(_MARK), id="float"),
    pytest.param(123456, id="int"),
    pytest.param(_MARK, id="str"),
    pytest.param(True, id="bool"),
    pytest.param(Decimal("NaN"), id="nan"),
    pytest.param(Decimal("Infinity"), id="infinity"),
    pytest.param(Decimal("-0.01"), id="negative"),
]
_MONEY_FIELDS = ["day_start_equity", "day_min_equity", "day_first_equity"]


@pytest.mark.parametrize("bad", _BAD_MONEY)
@pytest.mark.parametrize("field", _MONEY_FIELDS)
def test_wp15_t43_wrong_typed_equity_hint_degrades_to_unknown(field: str, bad: object) -> None:
    with capture_logs() as logs:
        p, _ = _rebuild(_seed(**{field: bad}))
    _assert_degraded_and_operable(p, logs, field=field)


def test_wp15_t43_sec208_float_day_first_with_day_min_none() -> None:
    """The mandatory SEC-208 case: ``day_first`` a float, ``day_min`` None, a valid
    Decimal ``day_start``.  Unvalidated, the float compared fine with a Decimal,
    became ``_nav_day_start``, and every later snapshot / mark / SELL raised
    ``TypeError`` -- the C-3 try/except alone would not have caught it."""
    seed = _seed(
        day_start_equity=Decimal("10000"), day_min_equity=None, day_first_equity=12345.5
    )
    with capture_logs() as logs:
        p, _ = _rebuild(seed)
    _assert_degraded_and_operable(p, logs, field="day_first_equity")


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"as_of_day": datetime(2026, 3, 10, tzinfo=UTC)}, "as_of_day", id="as_of_dt"),
        pytest.param({"as_of_day": "2026-03-10"}, "as_of_day", id="as_of_str"),
        pytest.param({"as_of_day": None}, "as_of_day", id="as_of_none"),
        pytest.param({"run_started_at": datetime(2026, 3, 5)}, "run_started_at", id="naive_start"),
        pytest.param({"run_started_at": "2026-03-05"}, "run_started_at", id="str_start"),
        pytest.param({"run_started_at": date(2026, 3, 5)}, "run_started_at", id="date_start"),
    ],
)
def test_wp15_t43_wrong_typed_day_or_start_degrades_to_unknown(
    overrides: dict[str, Any], field: str
) -> None:
    with capture_logs() as logs:
        p, _ = _rebuild(_seed(**overrides))
    _assert_degraded_and_operable(p, logs, field=field)


def test_wp15_t43_invalid_field_wins_over_a_midnight_race_and_a_future_day() -> None:
    """Validation precedes every comparison: an invalid seed is ``unknown`` even
    when ``as_of_day`` would otherwise pick the roll or the future branch."""
    for as_of in (_TODAY - timedelta(days=1), _TODAY + timedelta(days=1)):
        with capture_logs() as logs:
            p, _ = _rebuild(_seed(as_of_day=as_of, day_min_equity=float(_MARK)))
        _assert_degraded_and_operable(p, logs, field="day_min_equity")


def test_wp15_t43_a_non_seed_object_degrades_to_unknown() -> None:
    with capture_logs() as logs:
        p, _ = _rebuild({"as_of_day": _TODAY})  # type: ignore[arg-type]
    _assert_degraded_and_operable(p, logs, field="daily_seed")


def test_wp15_t43_a_seeding_exception_degrades_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-3: any exception inside seeding (here: the validator itself) is caught."""

    def _boom(seed: object) -> None:
        raise RuntimeError(f"boom {_MARK}")

    monkeypatch.setattr(portfolio_module, "_invalid_seed_field", _boom)
    with capture_logs() as logs:
        p, _ = _rebuild(_seed())
    _assert_degraded_and_operable(p, logs, field="seeding")
    [event] = [e for e in logs if e["event"] == "resume.daily_state_unknown"]
    assert event["value_type"] == "RuntimeError"  # the type name only, never the message


def test_wp15_t43_a_valid_seed_is_still_accepted_unchanged() -> None:
    """No false 'unknown' on the normal path (Decimal hints, aware start, strict date)."""
    with capture_logs() as logs:
        p, _ = _rebuild(_seed(day_min_equity=Decimal("9150")))
    snap = p.risk_snapshot()
    assert snap.day_start_source == "snapshot" and not snap.state_unknown
    assert snap.worst_nav_pnl == Decimal("-850")
    assert [e["event"] for e in logs if e["event"].startswith("resume.")] == [
        "resume.risk_state_seeded"
    ]
    # ``Decimal`` subclasses / zero are fine too
    p0, _ = _rebuild(_seed(day_min_equity=Decimal("0"), day_first_equity=Decimal("0")))
    assert p0.risk_snapshot().day_start_source == "snapshot"


# ---------------------------------------------------------------------------
# WP15-T-44  ResumeSnapshot.daily_seed (the builder)
# ---------------------------------------------------------------------------


def _snapshot(**overrides: Any) -> ResumeSnapshot:
    fields: dict[str, Any] = {
        "run_id": "wp15b",
        "initial_cash": INITIAL,
        "fills": [],
        "orders": [],
        "peak_equity_hint": None,
        "max_bar_index": -1,
    }
    fields.update(overrides)
    return ResumeSnapshot(**fields)


def test_wp15_t44_legacy_snapshot_has_no_seed() -> None:
    snap = _snapshot()  # the pre-1.5b constructor call: every new field defaulted
    assert snap.daily_hints_day is None
    assert snap.day_start_equity_hint is None
    assert snap.day_min_equity_hint is None
    assert snap.day_first_equity_hint is None
    assert snap.run_started_at is None
    assert snap.daily_seed is None  # -> replay_estimate, exactly the 1.5a behaviour


def test_wp15_t44_hints_day_alone_still_yields_a_seed() -> None:
    """An empty run (no equity snapshots) still gets a seed: ``None`` would be the
    weaker ``replay_estimate``."""
    seed = _snapshot(daily_hints_day=_TODAY, run_started_at=_STARTED).daily_seed
    assert seed is not None
    assert seed.as_of_day == _TODAY and seed.run_started_at == _STARTED
    assert seed.day_start_equity is None
    assert seed.day_min_equity is None
    assert seed.day_first_equity is None


def test_wp15_t44_maps_every_hint_to_its_seed_field() -> None:
    seed = _snapshot(
        daily_hints_day=_TODAY,
        day_start_equity_hint=Decimal("1"),
        day_min_equity_hint=Decimal("2"),
        day_first_equity_hint=Decimal("3"),
        run_started_at=_STARTED,
    ).daily_seed
    assert seed == ResumeDailySeed(
        as_of_day=_TODAY,
        day_start_equity=Decimal("1"),
        day_min_equity=Decimal("2"),
        run_started_at=_STARTED,
        day_first_equity=Decimal("3"),
    )


def test_wp15_t44_wrong_typed_hints_neither_raise_nor_return_none_nor_get_coerced() -> None:
    bad = float(_MARK)
    snap = _snapshot(
        daily_hints_day=_TODAY,
        day_start_equity_hint=bad,  # type: ignore[arg-type]
        day_min_equity_hint="9",  # type: ignore[arg-type]
        day_first_equity_hint=7,  # type: ignore[arg-type]
        run_started_at=datetime(2026, 3, 5),  # naive
    )
    seed = snap.daily_seed  # must not raise
    assert seed is not None  # None would map to the weaker replay_estimate
    # no coercion: the consumer (``_seed_daily_state``) flags the very same objects
    assert seed.day_start_equity is bad
    assert seed.day_min_equity == "9" and seed.day_first_equity == 7
    assert seed.run_started_at.tzinfo is None  # type: ignore[union-attr]
    with capture_logs() as logs:
        p, _ = _rebuild(seed)
    _assert_degraded_and_operable(p, logs)


def test_wp15_t44_construction_failure_degrades_to_a_day_only_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = recovery_module.ResumeDailySeed
    calls: list[int] = []

    def _flaky(**kwargs: Any) -> ResumeDailySeed:
        calls.append(1)
        if len(calls) == 1:
            raise ValueError(f"cannot build {_MARK}")
        return real(**kwargs)

    monkeypatch.setattr(recovery_module, "ResumeDailySeed", _flaky)
    snap = _snapshot(
        daily_hints_day=_TODAY,
        day_start_equity_hint=Decimal("10000"),
        day_min_equity_hint=Decimal("9000"),
        day_first_equity_hint=Decimal("10000"),
        run_started_at=_STARTED,
    )
    with capture_logs() as logs:
        seed = snap.daily_seed  # must not raise, must not return None
    assert seed is not None
    assert seed.as_of_day == _TODAY
    assert seed.day_start_equity is None and seed.day_min_equity is None
    assert seed.day_first_equity is None and seed.run_started_at is None
    [event] = [e for e in logs if e["event"] == "resume.daily_seed_invalid"]
    assert event["log_level"] == "error" and event["value_type"] == "ValueError"
    assert _MARK not in repr(logs)

    monkeypatch.undo()
    p, _ = _rebuild(seed)
    assert p.risk_snapshot().day_start_source == "unknown"  # day-only seed -> unknown


def test_wp15_t44_construction_failure_with_a_bad_day_still_yields_a_plain_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = recovery_module.ResumeDailySeed
    calls: list[int] = []

    def _flaky(**kwargs: Any) -> ResumeDailySeed:
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("x")
        return real(**kwargs)

    monkeypatch.setattr(recovery_module, "ResumeDailySeed", _flaky)
    seed = _snapshot(daily_hints_day=datetime(2026, 3, 10, tzinfo=UTC)).daily_seed  # type: ignore[arg-type]
    assert seed is not None
    assert type(seed.as_of_day) is date  # a strict date, never a datetime


# ---------------------------------------------------------------------------
# WP15-T-45  snapshot -> seed -> from_fills (AC-10 through the value object)
# ---------------------------------------------------------------------------


def _from_snapshot(snap: ResumeSnapshot) -> PortfolioAccounting:
    clock = Clock(D)
    return PortfolioAccounting.from_fills(
        snap.run_id,
        snap.initial_cash,
        [make_fill(quantity="10", price="1000", executed_at=D - timedelta(days=3))],
        peak_equity_hint=snap.peak_equity_hint,
        now=D,
        clock=clock,
        daily_seed=snap.daily_seed,
    )


def test_wp15_t45_snapshot_minimum_latches_buys_until_midnight() -> None:
    """AC-10(a) via the ResumeSnapshot: today's minimum 9 150, hint 10 000."""
    p = _from_snapshot(
        _snapshot(
            daily_hints_day=_TODAY,
            day_start_equity_hint=Decimal("10000"),
            day_min_equity_hint=Decimal("9150"),
            day_first_equity_hint=Decimal("10000"),
            run_started_at=_STARTED,
        )
    )
    s = p.risk_snapshot()
    assert s.day_start_source == "snapshot" and s.worst_nav_pnl == Decimal("-850")
    assert _gate(p, OrderSide.BUY) is False
    assert _gate(p, OrderSide.SELL) is True


def test_wp15_t45_depressed_pre_midnight_hint_is_hardened_by_the_first_of_today() -> None:
    """SEC-01 through the wiring: the day start is ``max(hint, first-of-today)``."""
    p = _from_snapshot(
        _snapshot(
            daily_hints_day=_TODAY,
            day_start_equity_hint=Decimal("8500"),  # outlier-low pre-midnight close
            day_min_equity_hint=Decimal("10000"),
            day_first_equity_hint=Decimal("10000"),
            run_started_at=_STARTED,
        )
    )
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000") and s.worst_nav_pnl == Decimal("0")
    assert p._day_start_pending is False


def test_wp15_t45_run_started_today_with_no_snapshots_uses_initial_cash() -> None:
    """AC-10(c): an empty run that started today."""
    p = _from_snapshot(
        _snapshot(daily_hints_day=_TODAY, run_started_at=D.replace(hour=1)),
    )
    s = p.risk_snapshot()
    assert s.day_start_source == "run_start" and s.nav_day_start == INITIAL


def test_wp15_t45_run_started_earlier_with_no_snapshots_is_unknown() -> None:
    """AC-10(d): an old run that never flushed a snapshot -> unknown (blocked until 00:00)."""
    with capture_logs() as logs:
        p = _from_snapshot(_snapshot(daily_hints_day=_TODAY, run_started_at=_STARTED))
    assert p.risk_snapshot().state_unknown
    [event] = [e for e in logs if e["event"] == "resume.daily_state_unknown"]
    assert event["reason"] == "no_day_start_hint"
    assert _gate(p, OrderSide.BUY) is False


def test_wp15_t45_midnight_race_applies_roll_semantics() -> None:
    """AC-10(e): the hints were read yesterday (the day rolled before from_fills)."""
    p = _from_snapshot(
        _snapshot(
            daily_hints_day=_TODAY - timedelta(days=1),
            day_start_equity_hint=Decimal("10000"),
            day_min_equity_hint=Decimal("9000"),
            day_first_equity_hint=Decimal("10000"),
            run_started_at=_STARTED,
        )
    )
    s = p.risk_snapshot()
    assert s.day_start_source == "roll" and s.worst_nav_pnl == Decimal("0")
    assert p._day_start_pending is True
