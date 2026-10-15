"""
tests/unit/test_wp15a_portfolio_day.py
----------------------------------------
WP1.5a (Verbeterplan v2 row 1.5, C5) -- PortfolioAccounting daily-PnL & peak
semantics.  Test IDs mirror ``reports/vp2-wp1.5/synthesis-spec.md`` §10:

    WP15-T-01  roll via update_market_prices with an injected clock
    WP15-T-02  reads are pure; D+1 reads report the virtual roll   (red@HEAD)
    WP15-T-03  mark-to-market + latch
    WP15-T-04  advance_clock: bar-days, wall clock pinned, V5
    WP15-T-05  advance_clock edge cases + wall-clock regression warning
    WP15-T-06  running-max peak, drawdown_pct
    WP15-T-07  FIX-07: realised-only get_daily_pnl, S-04 date-None case
    WP15-T-08  daily_loss_basis == BaseExecutionEngine._sizing_basis
    WP15-T-12  RiskSnapshot.gate_daily_pnl
    WP15-T-19  only fills / update_market_prices write peak or worst (R-09);
               extended: the day-start writers (SYN-01)
    WP15-T-36  Option B: a pre-roll outlier close does not depress the day start
    WP15-T-37  Option B: lower first mark, once only, reads stay pure
    WP15-T-38  Option B: fill-first day re-bases on the post-fill NAV
    WP15-T-07* reset_daily_pnl rolls across midnight (CR-01)
    WP15-T-05* first push after a wall-stamped event is a regression (CR-05)
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

import trading.portfolio as portfolio_module
from common.types import OrderSide
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import (
    D1,
    D2,
    INITIAL,
    Clock,
    D,
    make_fill,
    mark,
    portfolio_with_position,
)
from trading.execution import BaseExecutionEngine
from trading.portfolio import PortfolioAccounting, RiskSnapshot
from trading.risk_manager import DefaultRiskManager

_PINNED_WALL = datetime(2030, 1, 1, 0, 0, 1, tzinfo=UTC)


def _state(p: PortfolioAccounting) -> tuple[object, ...]:
    """Every field a pure read must leave untouched."""
    return (
        p._daily_pnl,
        p._daily_pnl_date,
        p._nav_day_start,
        p._worst_nav_pnl,
        p._day_start_source,
        p._day_start_pending,
        p._peak_equity,
        p._sim_now,
        p._regression_logged_day,
        len(p._equity_curve),
        p._cash,
    )


# ---------------------------------------------------------------------------
# WP15-T-01
# ---------------------------------------------------------------------------


def test_wp15_t01_roll_via_update_market_prices_captures_previous_nav() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    # realised loss on day D: sell 5 @ 920 -> (920 - 1000) * 5 = -400
    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="5", price="920"),
        current_price=Decimal("920"),
    )
    mark(p, "900")  # NAV 5 x 900 + 4 600 cash = 9 100: latch at -900 vs the 10 000 start
    nav_before_roll = p.current_equity
    assert p.get_daily_pnl() == Decimal("-400")
    assert p.risk_snapshot().worst_nav_pnl < 0
    assert p.risk_snapshot().day_start_source == "run_start"

    clock.at = D1
    with capture_logs() as logs:
        mark(p, "910")

    snap = p.risk_snapshot()
    assert snap.day_start_source == "roll"
    assert snap.day == D1.date()
    # day start = max(NAV at the PREVIOUS marks, NAV at the first D+1 mark):
    # SYN-01 / Option B re-bases once, at the first observation of the day
    assert snap.nav_day_start == max(nav_before_roll, snap.nav) == Decimal("9150")
    assert snap.realised_pnl == Decimal("0")
    assert p.get_daily_pnl() == Decimal("0")
    assert snap.nav_pnl == Decimal("0")
    # +5 units x +10 price move = +50 -> no loss on the new day, latch reset
    assert snap.worst_nav_pnl == Decimal("0")

    rolls = [e for e in logs if e["event"] == "portfolio.daily_roll"]
    assert len(rolls) == 1
    assert rolls[0]["prev_day"] == str(D.date())
    assert rolls[0]["day"] == str(D1.date())
    assert rolls[0]["nav_day_start"] == str(nav_before_roll)
    assert Decimal(rolls[0]["prev_realised"]) == Decimal("-400")
    # run-level NAV figures only (AC-16): no balance / secret-looking keys
    assert not {"balance", "api_key", "secret"} & set(rolls[0])


# ---------------------------------------------------------------------------
# WP15-T-02  (red@HEAD: no clock, no risk_snapshot at HEAD)
# ---------------------------------------------------------------------------


def test_wp15_t02_reads_are_pure_and_report_virtual_roll_on_next_day() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="5", price="920"),
        current_price=Decimal("920"),
    )
    mark(p, "900")
    assert p.get_daily_pnl() == Decimal("-400")

    clock.at = D1  # new UTC day, but NO mutator ran
    before = _state(p)
    with capture_logs() as logs:
        snap = p.risk_snapshot()
        realised = p.get_daily_pnl()
        summary = p.get_summary()
    assert _state(p) == before, "risk_snapshot/get_daily_pnl/get_summary must not mutate"
    assert not [e for e in logs if e["event"] == "portfolio.daily_roll"]

    assert realised == Decimal("0")
    assert summary["daily_pnl"] == "0"
    assert snap.realised_pnl == Decimal("0")
    assert snap.worst_nav_pnl == Decimal("0")
    assert snap.nav_day_start == snap.nav
    assert snap.day == D1.date()
    assert snap.day_start_source == "roll"
    assert snap.nav_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# WP15-T-03
# ---------------------------------------------------------------------------


def test_wp15_t03_mark_to_market_and_latch() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock, quantity="10", price="100", initial=Decimal("1000"))
    q = Decimal("10")
    mark(p, "92")
    snap = p.risk_snapshot()
    assert snap.nav_pnl == Decimal("-8") * q
    assert snap.worst_nav_pnl == Decimal("-8") * q

    mark(p, "96")  # recovery
    snap = p.risk_snapshot()
    assert snap.nav_pnl == Decimal("-4") * q
    assert snap.worst_nav_pnl == Decimal("-8") * q, "the latch must not recover"
    # realised stays 0 throughout (FIX-07)
    assert snap.realised_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# WP15-T-04
# ---------------------------------------------------------------------------


def test_wp15_t04_advance_clock_bar_days_with_wall_clock_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portfolio_module, "_utc_now", lambda: _PINNED_WALL)
    p = PortfolioAccounting("wp15", INITIAL)
    past = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)

    with capture_logs() as logs:
        # V5: the FIRST push (a past time) sets the day; it is not a roll
        p.advance_clock(past)
        assert p._daily_pnl_date == past.date()
        p.update_position(make_fill(), current_price=Decimal("1000"))
        # fills bucket by SIMULATED time (day 1), not by the pinned wall clock
        assert p._daily_pnl_date == past.date()

        for offset in (1, 2, 3):
            p.advance_clock(past + timedelta(days=offset))
    rolls = [e for e in logs if e["event"] == "portfolio.daily_roll"]
    assert len(rolls) == 3
    assert not [e for e in logs if e["event"] == "portfolio.clock_regression"]
    assert p.risk_snapshot().day == (past + timedelta(days=3)).date()
    assert p.risk_snapshot().as_of == past + timedelta(days=3)

    # a SELL between pushes is attributed to the simulated day
    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="10", price="990"),
        current_price=Decimal("990"),
    )
    assert p.get_daily_pnl() == Decimal("-100")
    p.advance_clock(past + timedelta(days=4))
    assert p.get_daily_pnl() == Decimal("0")
    # the wall clock was never consulted for the risk day
    assert p.risk_snapshot().day == (past + timedelta(days=4)).date()


# ---------------------------------------------------------------------------
# WP15-T-05
# ---------------------------------------------------------------------------


def test_wp15_t05_advance_clock_is_forward_only_and_strict_about_naive() -> None:
    p = PortfolioAccounting("wp15", INITIAL)
    late = datetime(2026, 3, 10, 23, 30, tzinfo=UTC)
    p.advance_clock(late)

    state = _state(p)
    p.advance_clock(late)  # equal: no-op
    p.advance_clock(late - timedelta(hours=5))  # earlier: no-op
    assert _state(p) == state
    assert p.risk_snapshot().as_of == late

    with pytest.raises(ValueError):
        p.advance_clock(datetime(2026, 3, 11, 1, 0))  # naive

    with capture_logs() as logs:
        p.advance_clock(datetime(2026, 3, 11, 2, 0, tzinfo=UTC))  # 23:30 -> 02:00 rolls
    assert len([e for e in logs if e["event"] == "portfolio.daily_roll"]) == 1
    assert p.risk_snapshot().day == datetime(2026, 3, 11, tzinfo=UTC).date()


def test_wp15_t05_wall_clock_regression_never_rolls_back_and_warns_once_per_day() -> None:
    clock = Clock(D1)
    p = portfolio_with_position(clock)
    mark(p, "1000")
    assert p._daily_pnl_date == D1.date()
    start = p._nav_day_start

    clock.at = D  # NTP step back one day
    with capture_logs() as logs:
        mark(p, "990")
        mark(p, "980")
    warnings = [e for e in logs if e["event"] == "portfolio.clock_regression"]
    assert len(warnings) == 1, "at most one regression warning per risk day"
    assert warnings[0]["log_level"] == "warning"
    assert p._daily_pnl_date == D1.date(), "a regression must never roll back"
    assert p._nav_day_start == start

    # next risk day: a regression is announced again (once)
    clock.at = D2
    mark(p, "980")
    clock.at = D1
    with capture_logs() as logs:
        mark(p, "970")
        mark(p, "960")
    assert len([e for e in logs if e["event"] == "portfolio.clock_regression"]) == 1


def test_wp15_t05_first_push_after_a_wall_stamped_event_is_a_regression() -> None:
    # CR-05: a mutator already stamped the risk day from the wall clock, so an
    # earlier pushed bar time is a regression (warning, no re-base).
    wall = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
    p = PortfolioAccounting("wp15", INITIAL, clock=Clock(wall))
    p.update_market_prices({})
    assert p._daily_pnl_date == wall.date()

    with capture_logs() as logs:
        p.advance_clock(datetime(2026, 1, 2, tzinfo=UTC))
    assert p._daily_pnl_date == datetime(2026, 6, 1, tzinfo=UTC).date()
    warnings = [e for e in logs if e["event"] == "portfolio.clock_regression"]
    assert len(warnings) == 1


def test_wp15_t05_first_ever_event_never_arms_the_rebase() -> None:
    p = PortfolioAccounting("wp15", INITIAL, clock=Clock(D))
    p.update_position(make_fill(quantity="10", price="1000"), current_price=Decimal("1200"))
    assert p._day_start_pending is False  # NAV 12 000 must not become the day start
    assert p.risk_snapshot().nav_day_start == INITIAL
    mark(p, "1100")
    assert p.risk_snapshot().nav_day_start == INITIAL and p._worst_nav_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# WP15-T-06
# ---------------------------------------------------------------------------


def test_wp15_t06_peak_is_running_max_of_marks_and_survives_roll() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)  # 10 x 1000, NAV 10 000
    for price in ("1050", "1100", "1020"):  # NAV 10 500 -> 11 000 -> 10 200
        mark(p, price)
    snap = p.risk_snapshot()
    assert snap.peak_equity == Decimal("11000")
    assert p.get_peak_equity() == Decimal("11000")
    assert snap.drawdown_pct == (Decimal("11000") - Decimal("10200")) / Decimal("11000")
    assert round(float(snap.drawdown_pct), 4) == 0.0727  # AC-05

    mark(p, "770")  # NAV 7 700 -> exactly 30.0%
    assert p.risk_snapshot().drawdown_pct == Decimal("0.3")

    clock.at = D1  # a roll never resets the peak
    mark(p, "780")
    assert p.get_peak_equity() == Decimal("11000")

    # snapshot.peak >= nav even if the stored peak is behind
    p._peak_equity = Decimal("1")
    assert p.risk_snapshot().peak_equity == p.current_equity
    assert p.risk_snapshot().drawdown_pct == Decimal("0")


# ---------------------------------------------------------------------------
# WP15-T-07
# ---------------------------------------------------------------------------


def test_wp15_t07_fix07_realised_only_and_rolled_summary() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock, price="50000", quantity="0.01")
    before = p.get_daily_pnl()
    mark(p, "55000")  # +50 unrealised
    assert p.get_daily_pnl() == before == Decimal("0")
    snap = p.risk_snapshot()
    assert snap.nav_pnl == Decimal("50.00")  # NAV moved, realised did not

    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="0.01", price="51000"),
        current_price=Decimal("51000"),
    )
    assert p.get_daily_pnl() == Decimal("10")
    assert Decimal(p.get_summary()["daily_pnl"]) == Decimal("10")

    clock.at = D1
    assert p.get_daily_pnl() == Decimal("0")
    assert p.get_summary()["daily_pnl"] == "0"
    # key set unchanged by WP1.5
    assert {"daily_pnl", "peak_equity", "current_equity", "drawdown_pct"} <= set(
        p.get_summary()
    )


def test_wp15_t07_s04_date_none_returns_the_accumulator() -> None:
    p = PortfolioAccounting("wp15", INITIAL)
    p._daily_pnl = Decimal("42.5")  # date still None: no event yet
    assert p._daily_pnl_date is None
    assert p.get_daily_pnl() == Decimal("42.5")


def test_wp15_t07_reset_daily_pnl_rolls_across_midnight_but_never_lifts_a_same_day_latch() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    mark(p, "900")  # -1000 NAV on day D -> latch
    assert p.risk_snapshot().worst_nav_pnl == Decimal("-1000")
    clock.at = D1
    p.reset_daily_pnl()
    s = p.risk_snapshot()
    assert s.day == D1.date() and s.day_start_source == "roll"
    assert s.nav_day_start == Decimal("9000") and p._worst_nav_pnl == Decimal("0")
    assert p._day_start_pending is True  # provisional until the first mark
    mark(p, "800")  # lower than the provisional start: the start stays
    assert p.risk_snapshot().nav_day_start == Decimal("9000")
    before = p._worst_nav_pnl
    p.reset_daily_pnl()  # same day: latch untouched
    assert p._worst_nav_pnl == before < 0


# ---------------------------------------------------------------------------
# WP15-T-08
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day_start_cash", "expected"),
    [
        (Decimal("20000"), Decimal("10000")),  # profit: capped at initial
        (Decimal("7000"), Decimal("7000")),  # loss: shrinks
        (Decimal("-5"), Decimal("0")),  # negative NAV: floored at 0
    ],
)
def test_wp15_t08_basis_equals_sizing_basis_and_is_constant_intraday(
    day_start_cash: Decimal, expected: Decimal
) -> None:
    clock = Clock(D)
    p = PortfolioAccounting("wp15", INITIAL, clock=clock)
    p.update_market_prices({})  # sets day D
    p._cash = day_start_cash  # a NAV the run reached on day D
    clock.at = D1
    p.update_market_prices({})  # roll: day start = that NAV

    snap = p.risk_snapshot()
    assert snap.nav_day_start == day_start_cash
    assert snap.daily_loss_basis == expected
    assert snap.daily_loss_basis == BaseExecutionEngine._sizing_basis(day_start_cash, INITIAL)

    # constant for the whole day, whatever the NAV does afterwards
    for delta in ("-300", "+500", "-100"):
        p._cash += Decimal(delta)
        p.update_market_prices({})
        assert p.risk_snapshot().daily_loss_basis == expected


# ---------------------------------------------------------------------------
# WP15-T-12
# ---------------------------------------------------------------------------


def _snap(**overrides: object) -> RiskSnapshot:
    base: dict[str, object] = {
        "as_of": D,
        "day": D.date(),
        "nav": Decimal("9500"),
        "peak_equity": Decimal("10000"),
        "drawdown_pct": Decimal("0.05"),
        "nav_day_start": Decimal("10000"),
        "daily_loss_basis": Decimal("10000"),
        "nav_pnl": Decimal("-500"),
        "worst_nav_pnl": Decimal("-700"),
        "realised_pnl": Decimal("0"),
        "day_start_source": "snapshot",
        "open_position_count": 1,
    }
    base.update(overrides)
    return RiskSnapshot(**base)  # type: ignore[arg-type]


def test_wp15_t12_gate_daily_pnl() -> None:
    snap = _snap()
    assert not snap.state_unknown
    # ticker NAV below the marks: the lower value wins
    assert snap.gate_daily_pnl(Decimal("9100")) == Decimal("-900")
    # ticker NAV above the marks: the latch (worst) wins
    assert snap.gate_daily_pnl(Decimal("9800")) == Decimal("-700")
    assert snap.gate_daily_pnl(Decimal("12000")) == Decimal("-700")

    unknown = _snap(day_start_source="unknown")
    assert unknown.state_unknown
    assert unknown.gate_daily_pnl(Decimal("99999")) == Decimal("-10000")


def test_wp15_t12_snapshot_is_immutable() -> None:
    with pytest.raises(AttributeError):
        _snap().nav = Decimal("1")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WP15-T-19 (R-09)
# ---------------------------------------------------------------------------


def test_wp15_t19_reads_never_write_peak_or_worst() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    mark(p, "950")
    state = _state(p)
    snap = p.risk_snapshot()
    for nav in ("1", "5000", "9500", "50000"):  # incl. a ticker spike
        snap.gate_daily_pnl(Decimal(nav))
    for _ in range(5):
        p.risk_snapshot()
        p.get_daily_pnl()
        p.get_peak_equity()
        p.get_summary()
    assert _state(p) == state


def test_wp15_t19_only_fills_and_marks_write_peak_or_worst() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(PortfolioAccounting)))

    def _writes(fn: ast.FunctionDef, attrs: set[str]) -> bool:
        for node in ast.walk(fn):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr in attrs:
                    return True
        return False

    writers = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and _writes(fn, {"_peak_equity", "_worst_nav_pnl"})
    }
    # construction, the shared NAV folder, the day roll (reset), resume seeding
    assert writers <= {
        "__init__",
        "_record_nav",
        "_ensure_daily_pnl_date",
        "from_fills",
        "_seed_daily_state",
    }, writers

    callers = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_record_nav"
            for n in ast.walk(fn)
        )
    }
    assert callers == {"_apply_fill", "update_market_prices"}

    # SYN-01: the day-start reference and its one-shot flag have a closed
    # writer set as well.
    start_writers = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and _writes(fn, {"_nav_day_start", "_day_start_pending"})
    }
    assert start_writers <= {
        "__init__",
        "_ensure_daily_pnl_date",
        "_record_nav",
        "_seed_daily_state",
    }, start_writers


# ---------------------------------------------------------------------------
# WP15-T-36 .. T-38  (SYN-01 / Option B: robust day start, one-shot re-base)
# ---------------------------------------------------------------------------


def _blocked(p: PortfolioAccounting) -> bool:
    """The REAL daily-loss gate (default 8%) fed exactly as the engines feed it."""
    s = p.risk_snapshot()
    violation = DefaultRiskManager(run_id="wp15")._check_daily_loss(
        s.gate_daily_pnl(s.nav), s.nav, s.daily_loss_basis
    )
    return violation is not None and violation.rule == "max_daily_loss"


def test_wp15_t36_pre_roll_outlier_close_does_not_depress_the_day_start() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock, quantity="3")  # cash 7 000, NAV 10 000
    mark(p, "1000")
    mark(p, "500")  # outlier close: NAV 8 500, day D latched
    assert p.risk_snapshot().nav == Decimal("8500")

    clock.at = D1
    with capture_logs() as logs:
        mark(p, "1000")  # the normal first mark of D+1: NAV 10 000
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("0")
    assert s.daily_loss_basis == Decimal("10000")
    rolls = [e for e in logs if e["event"] == "portfolio.daily_roll"]
    assert len(rolls) == 1 and rolls[0]["nav_day_start"] == "8500"  # pre-roll capture
    rebased = [e for e in logs if e["event"] == "portfolio.day_start_rebased"]
    assert len(rebased) == 1
    assert rebased[0]["prev_nav_day_start"] == "8500"
    assert rebased[0]["nav_day_start"] == "10000"
    assert not {"balance", "api_key", "secret"} & set(rebased[0])
    assert _blocked(p) is False

    mark(p, "700")  # a real D+1 decline: NAV 9 100
    s = p.risk_snapshot()
    assert s.nav == Decimal("9100")
    assert s.worst_nav_pnl == Decimal("-900")
    assert _blocked(p) is True


def test_wp15_t37_lower_first_mark_keeps_the_pre_roll_start_and_logs_no_rebase() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    mark(p, "1000")
    clock.at = D1
    with capture_logs() as logs:
        mark(p, "930")  # NAV 9 300 < 10 000
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("-700")
    assert not [e for e in logs if e["event"] == "portfolio.day_start_rebased"]


def test_wp15_t37_the_rebase_happens_once_only() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    mark(p, "850")  # NAV 8 500 on day D
    clock.at = D1
    mark(p, "1000")
    assert p._day_start_pending is False
    assert p.risk_snapshot().nav_day_start == Decimal("10000")
    mark(p, "1050")  # a later, higher mark must not raise the start again
    assert p.risk_snapshot().nav_day_start == Decimal("10000")
    mark(p, "960")  # NAV 9 600
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("-400")


def test_wp15_t37_reads_stay_pure_while_a_rebase_is_pending() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)
    mark(p, "900")  # NAV 9 000
    clock.at = D1
    p.reset_daily_pnl()  # rolls (CR-01) without any observation
    assert p._day_start_pending is True
    assert p.risk_snapshot().nav_day_start == Decimal("9000")

    state = _state(p)
    for _ in range(5):
        p.risk_snapshot()
        p.get_daily_pnl()
        p.get_summary()
    assert _state(p) == state
    assert p._day_start_pending is True

    mark(p, "1000")  # the next observation consumes the flag
    assert p._day_start_pending is False
    assert p.risk_snapshot().nav_day_start == Decimal("10000")


def test_wp15_t38_fill_first_day_rebases_on_the_post_fill_nav() -> None:
    clock = Clock(D)
    p = portfolio_with_position(clock)  # 10 @ 1 000, cash 0
    mark(p, "900")  # NAV 9 000
    clock.at = D1
    p.update_position(
        make_fill(side=OrderSide.SELL, quantity="5", price="1000"),
        current_price=Decimal("1000"),
    )  # NAV 10 000
    assert p.risk_snapshot().nav_day_start == Decimal("10000")
    assert p._day_start_pending is False

    mark(p, "900")  # NAV 9 500
    s = p.risk_snapshot()
    assert s.nav == Decimal("9500")
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("-500")
