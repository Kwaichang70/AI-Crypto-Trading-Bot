"""
tests/unit/test_wp15a_resume_seed.py
--------------------------------------
WP1.5a -- ``PortfolioAccounting.from_fills(daily_seed=...)``: the resume
day-start seeding logic (spec §5.7; the DB wiring that fills the seed is 1.5b).

    WP15-T-30  AC-10 (a)-(f), the rebuilt NAV never reaches ``worst``, and
               both resume events.
    WP15-T-39  midnight-race seed re-bases on the first mark (SEC-10)
    WP15-T-40  the seed table of ``final-synthesis-1.5a.md`` §3.1 (pending flag)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from structlog.testing import capture_logs

from common.types import OrderSide
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp15_support import D1, INITIAL, SYMBOL, Clock, D, make_fill, mark
from tests.unit.test_risk import _make_limit_order, _make_position
from trading.portfolio import PortfolioAccounting
from trading.recovery import ResumeDailySeed
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager

_TODAY = D.date()
_OLD_BUY = make_fill(quantity="10", price="1000", executed_at=D - timedelta(days=3))
_RELAXED = RiskParameters(
    max_daily_loss_pct=0.08,
    max_portfolio_exposure_pct=1.0,
    max_cluster_exposure_pct=1.0,
    max_position_size_pct=1.0,
)


def _seed(
    *,
    as_of: object = _TODAY,
    start: str | None = None,
    day_min: str | None = None,
    started: datetime | None = D - timedelta(days=5),
    day_first: str | None = None,
) -> ResumeDailySeed:
    return ResumeDailySeed(
        as_of_day=as_of,  # type: ignore[arg-type]
        day_start_equity=Decimal(start) if start is not None else None,
        day_min_equity=Decimal(day_min) if day_min is not None else None,
        run_started_at=started,
        day_first_equity=Decimal(day_first) if day_first is not None else None,
    )


def _rebuild(
    seed: ResumeDailySeed | None,
    *,
    fills: list[object] | None = None,
    peak_hint: str | None = None,
) -> tuple[PortfolioAccounting, Clock]:
    clock = Clock(D)
    portfolio = PortfolioAccounting.from_fills(
        "wp15-resume",
        INITIAL,
        fills if fills is not None else [_OLD_BUY],  # type: ignore[arg-type]
        peak_equity_hint=Decimal(peak_hint) if peak_hint is not None else None,
        now=D,
        clock=clock,
        daily_seed=seed,
    )
    return portfolio, clock


def _gate(portfolio: PortfolioAccounting, side: OrderSide) -> bool:
    """Run the REAL risk manager on a BUY/SELL against the portfolio's state."""
    snap = portfolio.risk_snapshot()
    manager = DefaultRiskManager(run_id="wp15-resume", params=_RELAXED)
    # Only the DAILY gate is under test: the BUY sees no open positions (no
    # exposure / concentration noise); the SELL exits the held BTC.
    held = (
        [_make_position(quantity=Decimal("10"), current_price=Decimal("1000"))]
        if side == OrderSide.SELL
        else []
    )
    order = _make_limit_order(
        symbol=SYMBOL,
        side=side,
        quantity=Decimal("0.01") if side == OrderSide.BUY else Decimal("10"),
        price=Decimal("1000"),
    )
    return manager.pre_trade_check(
        order=order,
        current_equity=snap.nav,
        open_positions=held,
        daily_pnl=snap.gate_daily_pnl(snap.nav),
        peak_equity=snap.peak_equity,
        daily_loss_basis=snap.daily_loss_basis,
    ).approved


# ---------------------------------------------------------------------------
# AC-10 (a): today's minimum 9 150, hint 10 000
# ---------------------------------------------------------------------------


def test_wp15_t30_a_snapshot_minimum_latches_buys_until_midnight_exits_work() -> None:
    with capture_logs() as logs:
        p, clock = _rebuild(_seed(start="10000", day_min="9150"))
    snap = p.risk_snapshot()
    assert snap.day_start_source == "snapshot"
    assert snap.nav_day_start == Decimal("10000")
    assert p._worst_nav_pnl == Decimal("-850")
    assert snap.worst_nav_pnl == Decimal("-850")
    assert _gate(p, OrderSide.BUY) is False
    assert _gate(p, OrderSide.SELL) is True  # exits are never latched

    clock.at = D1  # 00:00 passes: the day state clears (virtual roll)
    assert p.risk_snapshot().day_start_source == "roll"
    assert _gate(p, OrderSide.BUY) is True

    seeded = [e for e in logs if e["event"] == "resume.risk_state_seeded"]
    assert len(seeded) == 1
    assert seeded[0]["source"] == "snapshot"
    assert seeded[0]["worst"] == "-850"
    assert not [e for e in logs if e["event"] == "resume.daily_state_unknown"]


def test_wp15_t30_a_rebuilt_nav_never_reaches_worst() -> None:
    # The hint says the run was at 12 000 at day start; the replay (old fill
    # prices) only reaches 10 000.  Feeding the rebuilt NAV into the latch
    # would fabricate a -2 000 loss; the seed gives -100.
    p, _ = _rebuild(_seed(start="12000", day_min="11900"))
    assert p.current_equity == Decimal("10000")
    assert p._worst_nav_pnl == Decimal("-100")
    # after the first FRESH mark the portfolio is consistent again
    mark(p, "1190")  # 10 x 1190 = 11 900
    assert p.risk_snapshot().nav_pnl == Decimal("-100")
    assert p._worst_nav_pnl == Decimal("-100")


# ---------------------------------------------------------------------------
# AC-10 (b): only yesterday had a 9% loss
# ---------------------------------------------------------------------------


def test_wp15_t30_b_yesterdays_loss_does_not_block_today() -> None:
    p, _ = _rebuild(_seed(start="9100", day_min="9100"))
    snap = p.risk_snapshot()
    assert snap.worst_nav_pnl == Decimal("0")
    assert snap.daily_loss_basis == Decimal("9100")
    assert _gate(p, OrderSide.BUY) is True


# ---------------------------------------------------------------------------
# AC-10 (c): the run started today
# ---------------------------------------------------------------------------


def test_wp15_t30_c_run_started_today_uses_initial_cash_as_day_start() -> None:
    p, _ = _rebuild(_seed(start=None, day_min="9900", started=D - timedelta(hours=9)))
    snap = p.risk_snapshot()
    assert snap.day_start_source == "run_start"
    assert snap.nav_day_start == INITIAL
    assert p._worst_nav_pnl == Decimal("-100")


def test_wp15_t30_c_run_started_today_day_start_is_initial_even_with_todays_fills() -> None:
    fills = [
        make_fill(quantity="10", price="1000", executed_at=D - timedelta(hours=8)),
        make_fill(
            side=OrderSide.SELL, quantity="10", price="950", executed_at=D - timedelta(hours=4)
        ),
    ]
    p, _ = _rebuild(_seed(start=None, day_min=None, started=D - timedelta(hours=9)), fills=fills)
    s = p.risk_snapshot()
    assert s.day_start_source == "run_start"
    assert s.nav_day_start == INITIAL  # not the rebuilt 9 500
    assert s.nav_pnl == Decimal("-500")


# ---------------------------------------------------------------------------
# AC-10 (d): run older than today, no snapshot before today -> unknown
# ---------------------------------------------------------------------------


def test_wp15_t30_d_unknown_day_start_fails_closed_until_midnight() -> None:
    with capture_logs() as logs:
        p, clock = _rebuild(_seed(start=None, day_min=None, started=D - timedelta(days=5)))
    snap = p.risk_snapshot()
    assert snap.state_unknown and snap.day_start_source == "unknown"
    assert snap.gate_daily_pnl(snap.nav) == -snap.daily_loss_basis
    assert _gate(p, OrderSide.BUY) is False
    assert _gate(p, OrderSide.SELL) is True

    errors = [e for e in logs if e["event"] == "resume.daily_state_unknown"]
    assert len(errors) == 1 and errors[0]["log_level"] == "error"
    assert not [e for e in logs if e["event"] == "resume.risk_state_seeded"]

    clock.at = D1
    assert not p.risk_snapshot().state_unknown
    assert _gate(p, OrderSide.BUY) is True


def test_wp15_t30_d_run_without_start_time_is_also_unknown() -> None:
    p, _ = _rebuild(_seed(start=None, day_min="9900", started=None))
    assert p.risk_snapshot().state_unknown


def test_wp15_t30_seed_from_the_future_is_unknown() -> None:
    p, _ = _rebuild(_seed(as_of=_TODAY + timedelta(days=1), start="10000", day_min="9000"))
    assert p.risk_snapshot().state_unknown


# ---------------------------------------------------------------------------
# AC-10 (e): midnight race -> roll semantics
# ---------------------------------------------------------------------------


def test_wp15_t30_e_midnight_race_applies_roll_semantics() -> None:
    p, _ = _rebuild(_seed(as_of=_TODAY - timedelta(days=1), start="10000", day_min="9000"))
    snap = p.risk_snapshot()
    assert snap.day_start_source == "roll"
    assert snap.nav_day_start == p.current_equity
    assert snap.worst_nav_pnl == Decimal("0")
    assert _gate(p, OrderSide.BUY) is True


# ---------------------------------------------------------------------------
# AC-10 (f): no seed -> replay_estimate.  The gate VALUE is never weaker than
# HEAD's realised-today value; the threshold uses the fixed day-start basis
# (C-02) and mark drift since the fills counts toward the resume day (E-1).
# ---------------------------------------------------------------------------


def test_wp15_t30_f_no_seed_gate_value_never_weaker_than_realised_today() -> None:
    sell_today = make_fill(
        side=OrderSide.SELL, quantity="5", price="800", executed_at=D.replace(hour=8)
    )
    p, _ = _rebuild(None, fills=[_OLD_BUY, sell_today])
    snap = p.risk_snapshot()
    assert snap.day_start_source == "replay_estimate"
    assert snap.realised_pnl == Decimal("-1000")
    assert snap.nav_day_start == p.current_equity - snap.realised_pnl  # 8 000 + 1 000
    assert snap.worst_nav_pnl == Decimal("-1000")
    assert snap.gate_daily_pnl(snap.nav) <= snap.realised_pnl
    assert _gate(p, OrderSide.BUY) is False  # HEAD blocked this too

    # a run with no losses today: floor is 0 and nothing is blocked
    p2, _ = _rebuild(None)
    assert p2.risk_snapshot().worst_nav_pnl == Decimal("0")
    assert p2.risk_snapshot().nav_day_start == p2.current_equity
    assert _gate(p2, OrderSide.BUY) is True


def test_wp15_t30_f_replay_estimate_counts_mark_drift_since_fill() -> None:
    # CR-06 / E-1: BUY 40 @ 100 a week ago, no seed.  The rebuilt NAV is at the
    # fill price (10 000); drift since the fill counts toward the resume day.
    fills = [make_fill(quantity="40", price="100", executed_at=D - timedelta(days=7))]
    p, _ = _rebuild(None, fills=fills)
    assert p.risk_snapshot().day_start_source == "replay_estimate"
    mark(p, "80")  # NAV 9 200
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("-800") == p._worst_nav_pnl
    assert _gate(p, OrderSide.BUY) is False  # -800 == 8% of 10 000: inclusive

    # fresh rebuild, the price rose instead: no cushion from the drift
    p2, _ = _rebuild(None, fills=fills)
    mark(p2, "120")  # NAV 10 800
    s2 = p2.risk_snapshot()
    assert s2.nav_day_start == Decimal("10800")
    assert s2.worst_nav_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# WP15-T-39  (SEC-10): midnight-race seed re-bases on the first mark
# ---------------------------------------------------------------------------


def _midnight_race() -> PortfolioAccounting:
    fills = [make_fill(quantity="10", price="1000", executed_at=D - timedelta(days=2))]
    seed = _seed(
        as_of=(D - timedelta(days=1)).date(),
        start="10000",
        day_min=None,
        started=D - timedelta(days=5),
    )
    p, _ = _rebuild(seed, fills=fills)
    return p


def test_wp15_t39_midnight_race_seed_rebases_on_the_first_mark() -> None:
    p = _midnight_race()
    s = p.risk_snapshot()
    assert s.day_start_source == "roll" and s.nav_day_start == Decimal("10000")
    assert p._day_start_pending is True

    mark(p, "1200")  # fresh price above the fill price: no cushion
    assert p.risk_snapshot().nav_day_start == Decimal("12000")
    mark(p, "1110")  # NAV 11 100
    s = p.risk_snapshot()
    assert s.worst_nav_pnl == Decimal("-900")
    assert _gate(p, OrderSide.BUY) is False  # 900 >= 8% of min(12 000, 10 000)


def test_wp15_t39_midnight_race_lower_first_mark_counts_drift_since_fills() -> None:
    p = _midnight_race()
    mark(p, "900")  # NAV 9 000: below the rebuilt start, which stays
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("10000")
    assert s.worst_nav_pnl == Decimal("-1000")  # fail-closed, documented
    assert _gate(p, OrderSide.BUY) is False


# ---------------------------------------------------------------------------
# WP15-T-40: the seed table (final-synthesis-1.5a.md §3.1)
# ---------------------------------------------------------------------------


def test_wp15_t40_1_replay_estimate_with_a_realised_loss_never_rebases() -> None:
    fills = [
        make_fill(quantity="10", price="1000", executed_at=D - timedelta(hours=8)),
        make_fill(
            side=OrderSide.SELL, quantity="5", price="850", executed_at=D - timedelta(hours=4)
        ),
    ]
    p, _ = _rebuild(None, fills=fills)
    s = p.risk_snapshot()
    assert s.day_start_source == "replay_estimate"
    assert s.nav_day_start == Decimal("9250") and p._worst_nav_pnl == Decimal("-750")
    assert p._day_start_pending is False

    mark(p, "1300")  # NAV 10 750: a re-base here would relax the latch
    s = p.risk_snapshot()
    assert s.nav_day_start == Decimal("9250")
    assert s.worst_nav_pnl == Decimal("-750")
    assert _gate(p, OrderSide.BUY) is False


def test_wp15_t40_1_replay_estimate_without_a_loss_arms_the_rebase() -> None:
    p, _ = _rebuild(None)
    assert p._day_start_pending is True


def test_wp15_t40_2_snapshot_with_day_min_does_not_rebase() -> None:
    p, _ = _rebuild(_seed(start="10000", day_min="9900"))
    assert p._day_start_pending is False
    mark(p, "1100")  # NAV 11 000, higher than the hint
    assert p.risk_snapshot().nav_day_start == Decimal("10000")


def test_wp15_t40_2_snapshot_with_day_min_at_the_hint_does_not_rebase() -> None:
    p, _ = _rebuild(_seed(start="10000", day_min="10000"))  # worst == 0, but a snapshot exists
    assert p._day_start_pending is False
    mark(p, "1100")
    assert p.risk_snapshot().nav_day_start == Decimal("10000")


def test_wp15_t40_3_snapshot_day_first_hardens_a_depressed_day_start() -> None:
    p, _ = _rebuild(_seed(start="9000", day_first="9600", day_min="9100"))
    s = p.risk_snapshot()
    assert s.day_start_source == "snapshot"
    assert s.nav_day_start == Decimal("9600")
    assert s.worst_nav_pnl == Decimal("-500") == p._worst_nav_pnl
    assert p._day_start_pending is False


def test_wp15_t40_3_snapshot_day_first_below_the_hint_keeps_the_hint() -> None:
    p, _ = _rebuild(_seed(start="10000", day_first="9600", day_min="9100"))
    assert p.risk_snapshot().nav_day_start == Decimal("10000")
    assert p._worst_nav_pnl == Decimal("-900")


def test_wp15_t40_4_snapshot_with_neither_arms_the_rebase() -> None:
    p, _ = _rebuild(_seed(start="9000", day_min=None, day_first=None))
    assert p._day_start_pending is True
    assert p.risk_snapshot().nav_day_start == Decimal("9000")
    mark(p, "1100")  # NAV 11 000
    assert p.risk_snapshot().nav_day_start == Decimal("11000")
    assert p._day_start_pending is False


def test_wp15_t40_5_run_start_does_not_rebase() -> None:
    p, _ = _rebuild(_seed(start=None, day_min=None, started=D - timedelta(hours=9)))
    assert p.risk_snapshot().day_start_source == "run_start"
    assert p._day_start_pending is False


def test_wp15_t40_6_unknown_never_rebases_and_still_gates_at_minus_basis() -> None:
    for seed in (
        _seed(start=None, day_min=None, started=D - timedelta(days=5)),
        _seed(as_of=_TODAY + timedelta(days=1), start="10000", day_min="9000"),
    ):
        p, _ = _rebuild(seed)
        s = p.risk_snapshot()
        assert s.state_unknown
        assert p._day_start_pending is False
        assert s.gate_daily_pnl(s.nav) == -s.daily_loss_basis
        mark(p, "1300")
        assert p.risk_snapshot().state_unknown
        assert _gate(p, OrderSide.BUY) is False


# ---------------------------------------------------------------------------
# peak (AC-06) and seed bookkeeping
# ---------------------------------------------------------------------------


def test_wp15_t30_peak_hint_still_only_rises() -> None:
    p, _ = _rebuild(_seed(start="10000", day_min="10000"), peak_hint="12000")
    assert p.get_peak_equity() == Decimal("12000")
    assert p.risk_snapshot().drawdown_pct == Decimal("2000") / Decimal("12000")
    # a lower hint never lowers the rebuilt peak
    p2, _ = _rebuild(_seed(start="10000", day_min="10000"), peak_hint="500")
    assert p2.get_peak_equity() == Decimal("10000")


def test_wp15_t30_seeded_day_is_today_and_clock_is_injectable() -> None:
    p, clock = _rebuild(_seed(start="10000", day_min="10000"))
    assert p._daily_pnl_date == _TODAY
    assert p._now() == D
    clock.at = datetime(2030, 1, 1, tzinfo=UTC)
    assert p._now() == clock.at
