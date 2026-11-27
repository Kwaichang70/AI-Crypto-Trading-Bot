"""WP2.5 / P0: pure equity-sampling leaf ``packages/trading/equity_sampling.py``.

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-04, SY-25-21; WP25-T-01..04, T-06).
The AST pin (T-05) lives in ``test_wp25_ast_pins.py``.

Reference values are the closed-form REF-A numbers of the quant design (exact ``Fraction``
arithmetic), compared at rel 1e-9 / 1e-12 as stated in the spec.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from trading.equity_sampling import (
    EQUITY_SAMPLING_CODES,
    BarSample,
    EquitySamplingError,
    last_per_stamp,
    sample_equity_per_bar,
    simple_returns,
)
from trading.metrics import EquityCurvePoint, compute_returns_from_equity, compute_sharpe

H = 3600
P_1H = 8766.0  # 365.25 * 24
T0 = datetime(2026, 1, 1, tzinfo=UTC)
FINAL_REF_A = 16284.270878242956


@dataclass(frozen=True, slots=True)
class Pt:
    timestamp: datetime
    equity: Decimal


def ts(k: int, tf: int = H, t0: datetime = T0) -> datetime:
    return t0 + timedelta(seconds=k * tf)


def ref_a_equity() -> list[Decimal]:
    """101 equity values: 10000 then 100 alternating +2% / -1% steps (exact rationals)."""
    eq = Fraction(10000)
    out = [Decimal(eq.numerator) / Decimal(eq.denominator)]
    for k in range(100):
        eq *= Fraction(102, 100) if k % 2 == 0 else Fraction(99, 100)
        out.append(Decimal(eq.numerator) / Decimal(eq.denominator))
    return out


def ref_a_points(skip: frozenset[int] = frozenset()) -> list[Pt]:
    return [Pt(ts(k), e) for k, e in enumerate(ref_a_equity()) if k not in skip]


# ---------------------------------------------------------------------------
# WP25-T-01  last_per_stamp
# ---------------------------------------------------------------------------


class TestLastPerStamp:
    def test_duplicates_collapse_to_the_last(self) -> None:
        pts = [
            Pt(ts(0), Decimal(1)),
            Pt(ts(1), Decimal(2)),
            Pt(ts(1), Decimal(3)),
            Pt(ts(1), Decimal(4)),
            Pt(ts(2), Decimal(5)),
            Pt(ts(2), Decimal(6)),
        ]
        out = last_per_stamp(pts)
        assert [(p.timestamp, p.equity) for p in out] == [
            (ts(0), Decimal(1)),
            (ts(1), Decimal(4)),
            (ts(2), Decimal(6)),
        ]
        assert out[1] is pts[3]

    def test_decreasing_stamp_raises_closed_code(self) -> None:
        pts = [Pt(ts(0), Decimal(1)), Pt(ts(2), Decimal(1)), Pt(ts(1), Decimal(1))]
        with pytest.raises(EquitySamplingError) as ei:
            last_per_stamp(pts)
        assert str(ei.value) == "equity sampling failed (equity_sampling_unordered)"
        assert ei.value.code == "equity_sampling_unordered"
        assert isinstance(ei.value, ValueError)

    def test_input_not_mutated(self) -> None:
        pts = [Pt(ts(0), Decimal(1)), Pt(ts(0), Decimal(2)), Pt(ts(1), Decimal(3))]
        snapshot = list(pts)
        last_per_stamp(pts)
        assert pts == snapshot
        assert len(pts) == 3

    def test_empty_and_single(self) -> None:
        assert last_per_stamp([]) == []
        one = Pt(ts(0), Decimal(1))
        assert last_per_stamp([one]) == [one]

    def test_error_codes_are_closed(self) -> None:
        assert EquitySamplingError.CODES == EQUITY_SAMPLING_CODES
        assert set(EQUITY_SAMPLING_CODES) == {
            "equity_sampling_unordered",
            "equity_sampling_no_base_point",
        }
        with pytest.raises(ValueError, match="unknown equity sampling code"):
            EquitySamplingError("anything else")


# ---------------------------------------------------------------------------
# WP25-T-02  REF-A truth
# ---------------------------------------------------------------------------


class TestRefATruth:
    def test_gap_free_reference(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        assert s.n_returns == 100
        assert s.mode == "calendar_asof"
        assert s.carried_slots == 0
        assert s.start_at == T0
        assert s.end_at == ts(100)
        assert s.timeframe_s == H
        sharpe = compute_sharpe(s.returns(), P_1H)
        assert sharpe == pytest.approx(31.052536128310, rel=1e-12)
        assert sharpe == pytest.approx(math.sqrt(0.99 * 8766) / 3, rel=1e-12)
        assert float(s.equity[-1]) == pytest.approx(FINAL_REF_A, rel=1e-12)

    def test_curve_point_type_is_accepted(self) -> None:
        """The pydantic ``EquityCurvePoint`` satisfies the duck type (no import in the leaf)."""
        pts = [EquityCurvePoint(timestamp=p.timestamp, equity=p.equity) for p in ref_a_points()]
        s = sample_equity_per_bar(pts, start_at=T0, timeframe_s=H)
        assert s.n_returns == 100
        assert (
            s.returns()
            == sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H).returns()
        )


# ---------------------------------------------------------------------------
# WP25-T-03  REF-A with holes {10, 11, 40}
# ---------------------------------------------------------------------------

HOLES = frozenset({10, 11, 40})


class TestRefAHoles:
    def test_calendar_asof(self) -> None:
        s = sample_equity_per_bar(ref_a_points(HOLES), start_at=T0, timeframe_s=H)
        assert (s.mode, s.n_returns, s.carried_slots) == ("calendar_asof", 100, 3)
        assert s.carried_stamps == tuple(ts(k) for k in sorted(HOLES))
        assert compute_sharpe(s.returns(), P_1H) == pytest.approx(31.734024926469, rel=1e-9)
        assert float(s.equity[-1]) == pytest.approx(FINAL_REF_A, rel=1e-12)
        # carried slots repeat the previous value
        assert s.equity[10] == s.equity[9]
        assert s.equity[11] == s.equity[9]
        assert s.equity[40] == s.equity[39]

    def test_present_steps(self) -> None:
        s = sample_equity_per_bar(
            ref_a_points(HOLES), start_at=T0, timeframe_s=H, mode="present_steps"
        )
        assert (s.mode, s.n_returns, s.carried_slots) == ("present_steps", 97, 0)
        assert compute_sharpe(s.returns(), P_1H) == pytest.approx(32.273953452686, rel=1e-9)
        assert float(s.equity[-1]) == pytest.approx(FINAL_REF_A, rel=1e-12)
        assert len(s.stamps) == 98

    def test_default_mode_is_calendar(self) -> None:
        s = sample_equity_per_bar(ref_a_points(HOLES), start_at=T0, timeframe_s=H)
        assert s.mode == "calendar_asof"

    def test_hole_at_first_slot_after_base_is_carried(self) -> None:
        """Slot 1 is the first slot the ``k >= 1`` carried rule covers."""
        s = sample_equity_per_bar(ref_a_points(frozenset({1})), start_at=T0, timeframe_s=H)
        assert s.mode == "calendar_asof"
        assert s.carried_slots == 1
        assert s.carried_stamps == (ts(1),)
        assert s.equity[1] == s.equity[0]
        assert s.n_returns == 100

    def test_hole_at_last_but_one_slot_is_carried(self) -> None:
        s = sample_equity_per_bar(ref_a_points(frozenset({99})), start_at=T0, timeframe_s=H)
        assert s.carried_slots == 1
        assert s.carried_stamps == (ts(99),)
        assert s.equity[99] == s.equity[98]
        assert s.n_returns == 100


# ---------------------------------------------------------------------------
# WP25-T-04  warmup prefix and duplicate fill points do not matter
# ---------------------------------------------------------------------------


def with_warmup_and_fills(warmup: int, dups: int) -> tuple[list[Pt], datetime]:
    """Flat warmup marks, then REF-A where each stamp gets ``dups`` earlier fill points.

    The end-of-bar mark is always the LAST point of its stamp; the fill points carry
    deliberately different (intrabar) equities that must never reach the sample.
    """
    eq = ref_a_equity()
    base = eq[0]
    start = ts(warmup)
    pts = [Pt(ts(k), base) for k in range(warmup)]
    for k, e in enumerate(eq):
        for d in range(dups):
            pts.append(Pt(ts(warmup + k), e + Decimal(d + 1) * Decimal(7)))
        pts.append(Pt(ts(warmup + k), e))
    return pts, start


class TestWarmupAndFillInvariance:
    @pytest.mark.parametrize("dups", [0, 1, 2, 3, 4, 5])
    def test_sample_identical_to_truth_for_any_fill_count(self, dups: int) -> None:
        truth = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        pts, start = with_warmup_and_fills(206, dups)
        s = sample_equity_per_bar(pts, start_at=start, timeframe_s=H)
        assert s.n_returns == 100
        assert s.equity == truth.equity
        assert s.returns() == truth.returns()
        assert (s.mode, s.carried_slots) == ("calendar_asof", 0)
        assert s.stamps[0] == start
        assert compute_sharpe(s.returns(), P_1H) == pytest.approx(31.052536128310, rel=1e-12)

    def test_whole_sample_equal_modulo_origin(self) -> None:
        truth = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        pts, start = with_warmup_and_fills(206, 3)
        s = sample_equity_per_bar(pts, start_at=start, timeframe_s=H)
        shift = start - T0
        assert replace(s, start_at=T0, end_at=truth.end_at,
                       stamps=tuple(x - shift for x in s.stamps)) == truth  # fmt: skip

    def test_warmup_zero(self) -> None:
        pts, start = with_warmup_and_fills(0, 2)
        assert start == T0
        s = sample_equity_per_bar(pts, start_at=start, timeframe_s=H)
        assert s.n_returns == 100


# ---------------------------------------------------------------------------
# WP25-T-06  base point, empty, off-grid, from_stamp
# ---------------------------------------------------------------------------


class TestBasePointAndEdges:
    def test_missing_base_point_fails_closed(self) -> None:
        pts = [Pt(ts(k), Decimal(100 + k)) for k in range(1, 6)]
        with pytest.raises(EquitySamplingError) as ei:
            sample_equity_per_bar(pts, start_at=ts(0), timeframe_s=H)
        assert str(ei.value) == "equity sampling failed (equity_sampling_no_base_point)"

    def test_base_point_between_stamps_fails_closed(self) -> None:
        pts = [Pt(ts(0), Decimal(1)), Pt(ts(2), Decimal(2))]
        with pytest.raises(EquitySamplingError, match="no_base_point"):
            sample_equity_per_bar(pts, start_at=ts(1), timeframe_s=H)

    def test_nothing_at_or_after_start_is_empty(self) -> None:
        pts = [Pt(ts(k), Decimal(100)) for k in range(5)]
        s = sample_equity_per_bar(pts, start_at=ts(9), timeframe_s=H)
        assert s.n_returns == 0
        assert s.stamps == ()
        assert s.equity == ()
        assert s.end_at is None
        assert s.carried_slots == 0
        assert s.returns() == []

    def test_empty_input_is_empty_sample(self) -> None:
        s = sample_equity_per_bar([], start_at=T0, timeframe_s=H)
        assert s.n_returns == 0
        assert s.mode == "calendar_asof"
        assert s.start_at == T0
        assert s.end_at is None
        assert s.timeframe_s == H

    def test_empty_input_present_steps_reports_requested_mode(self) -> None:
        s = sample_equity_per_bar([], start_at=T0, timeframe_s=H, mode="present_steps")
        assert s.n_returns == 0
        assert s.mode == "present_steps"
        assert s.start_at == T0
        assert s.end_at is None
        assert s.timeframe_s == H

    def test_single_base_point_has_no_returns(self) -> None:
        s = sample_equity_per_bar([Pt(T0, Decimal(5))], start_at=T0, timeframe_s=H)
        assert s.n_returns == 0
        assert s.equity == (Decimal(5),)

    def test_off_grid_stamp_gives_present_steps(self) -> None:
        pts = [
            Pt(ts(0), Decimal(100)),
            Pt(ts(1), Decimal(101)),
            Pt(ts(1) + timedelta(seconds=7), Decimal(102)),
            Pt(ts(3), Decimal(103)),
        ]
        s = sample_equity_per_bar(pts, start_at=ts(0), timeframe_s=H)
        assert s.mode == "present_steps"
        assert s.carried_slots == 0
        assert s.stamps == (ts(0), ts(1), ts(1) + timedelta(seconds=7), ts(3))
        assert s.n_returns == 3

    @pytest.mark.parametrize("tf", [900, 3600, 86400])
    def test_half_step_offset_gives_present_steps(self, tf: int) -> None:
        stamps = [ts(0, tf), ts(1, tf), ts(1, tf) + timedelta(seconds=tf // 2), ts(3, tf)]
        pts = [Pt(t, Decimal(100 + i)) for i, t in enumerate(stamps)]
        s = sample_equity_per_bar(pts, start_at=ts(0, tf), timeframe_s=tf)
        assert s.mode == "present_steps"
        assert s.carried_slots == 0
        assert s.stamps == tuple(stamps)
        assert s.n_returns == 3

    def test_ninety_minute_offset_on_hourly_grid_gives_present_steps(self) -> None:
        stamps = [ts(0), ts(1), T0 + timedelta(minutes=90), ts(3)]
        pts = [Pt(t, Decimal(100 + i)) for i, t in enumerate(stamps)]
        s = sample_equity_per_bar(pts, start_at=T0, timeframe_s=H)
        assert s.mode == "present_steps"
        assert s.carried_slots == 0
        assert s.stamps == tuple(stamps)

    def test_daily_grid_with_hole_is_calendar_asof(self) -> None:
        tf = 86400
        pts = [Pt(ts(k, tf), Decimal(100 + k)) for k in (0, 1, 3, 4)]
        s = sample_equity_per_bar(pts, start_at=ts(0, tf), timeframe_s=tf)
        assert s.mode == "calendar_asof"
        assert s.carried_slots == 1
        assert s.carried_stamps == (ts(2, tf),)
        assert s.stamps == tuple(ts(k, tf) for k in range(5))
        assert s.equity[2] == s.equity[1]
        assert s.n_returns == 4

    def test_pre_start_off_grid_stamps_are_ignored(self) -> None:
        pts = [Pt(ts(0) - timedelta(seconds=5), Decimal(1)), Pt(ts(0), Decimal(2)),
               Pt(ts(1), Decimal(3))]  # fmt: skip
        s = sample_equity_per_bar(pts, start_at=ts(0), timeframe_s=H)
        assert s.mode == "calendar_asof"
        assert s.equity == (Decimal(2), Decimal(3))

    def test_unordered_input_raises_from_sampler(self) -> None:
        pts = [Pt(ts(1), Decimal(1)), Pt(ts(0), Decimal(1))]
        with pytest.raises(EquitySamplingError, match="unordered"):
            sample_equity_per_bar(pts, start_at=ts(0), timeframe_s=H)

    @pytest.mark.parametrize("tf", [0, -3600])
    def test_non_positive_timeframe_rejected(self, tf: int) -> None:
        with pytest.raises(ValueError, match="timeframe_s must be positive"):
            sample_equity_per_bar([Pt(T0, Decimal(1))], start_at=T0, timeframe_s=tf)

    def test_trailing_gap_is_not_extended(self) -> None:
        """The grid ends at the last stamp: no value is invented past the curve end."""
        s = sample_equity_per_bar(ref_a_points(frozenset({100})), start_at=T0, timeframe_s=H)
        assert s.end_at == ts(99)
        assert s.n_returns == 99

    def test_input_not_mutated(self) -> None:
        pts = ref_a_points(HOLES)
        snapshot = list(pts)
        sample_equity_per_bar(pts, start_at=T0, timeframe_s=H)
        assert pts == snapshot

    def test_sample_is_frozen(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        with pytest.raises(FrozenInstanceError):
            s.mode = "present_steps"  # type: ignore[misc]


class TestFromStamp:
    def test_base_is_asof_value_at_stamp(self) -> None:
        s = sample_equity_per_bar(ref_a_points(HOLES), start_at=T0, timeframe_s=H)
        sub = s.from_stamp(ts(11))  # slot 11 is carried: base == value at slot 9
        assert sub.start_at == ts(11)
        assert sub.stamps[0] == ts(11)
        assert sub.equity[0] == s.equity[9] == s.equity[11]
        assert sub.equity[1:] == s.equity[12:]
        assert sub.n_returns == s.n_returns - 11
        assert sub.end_at == s.end_at
        assert sub.mode == "calendar_asof"

    def test_carried_slots_recounted(self) -> None:
        s = sample_equity_per_bar(ref_a_points(HOLES), start_at=T0, timeframe_s=H)
        assert s.from_stamp(ts(11)).carried_slots == 1  # only slot 40 remains
        assert s.from_stamp(ts(10)).carried_slots == 2
        assert s.from_stamp(ts(41)).carried_slots == 0

    def test_from_first_stamp_is_identity(self) -> None:
        s = sample_equity_per_bar(ref_a_points(HOLES), start_at=T0, timeframe_s=H)
        assert s.from_stamp(T0) == s

    def test_returns_match_tail_of_full_returns(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        assert s.from_stamp(ts(30)).returns() == s.returns()[30:]

    def test_at_last_stamp_has_no_returns(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        sub = s.from_stamp(ts(100))
        assert (sub.n_returns, sub.equity) == (0, (s.equity[-1],))

    def test_after_end_is_empty(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=T0, timeframe_s=H)
        sub = s.from_stamp(ts(101))
        assert sub.n_returns == 0
        assert sub.stamps == ()
        assert sub.start_at == ts(101)
        assert sub.end_at is None
        assert sub.mode == "calendar_asof"

    def test_after_end_is_empty_present_steps(self) -> None:
        s = sample_equity_per_bar(
            ref_a_points(HOLES), start_at=T0, timeframe_s=H, mode="present_steps"
        )
        sub = s.from_stamp(ts(101))
        assert sub.n_returns == 0
        assert sub.stamps == ()
        assert sub.start_at == ts(101)
        assert sub.end_at is None
        assert sub.mode == "present_steps"

    def test_before_start_fails_closed(self) -> None:
        s = sample_equity_per_bar(ref_a_points(), start_at=ts(5), timeframe_s=H)
        with pytest.raises(EquitySamplingError, match="no_base_point"):
            s.from_stamp(ts(4))

    def test_empty_sample_stays_empty(self) -> None:
        s = sample_equity_per_bar([], start_at=T0, timeframe_s=H)
        assert s.from_stamp(ts(3)) is s

    def test_present_steps_from_stamp(self) -> None:
        s = sample_equity_per_bar(
            ref_a_points(HOLES), start_at=T0, timeframe_s=H, mode="present_steps"
        )
        sub = s.from_stamp(ts(10))  # stamp 10 is absent: as-of value is stamp 9's
        assert sub.equity[0] == s.equity[s.stamps.index(ts(9))]
        assert sub.mode == "present_steps"
        assert sub.stamps[0] == ts(10)
        assert sub.start_at == ts(10)
        assert sub.stamps[1:] == s.stamps[s.stamps.index(ts(9)) + 1 :]


# ---------------------------------------------------------------------------
# simple_returns: one formula, byte-identical to HEAD
# ---------------------------------------------------------------------------


class TestSimpleReturns:
    def test_matches_compute_returns_from_equity_exactly(self) -> None:
        import random

        rng = random.Random(25)  # noqa: S311 - seeded test data
        eq = [Decimal(10000)]
        for _ in range(500):
            eq.append(eq[-1] * (Decimal(1) + Decimal(rng.randint(-300, 300)) / Decimal(10000)))
        curve = [EquityCurvePoint(timestamp=ts(i), equity=e) for i, e in enumerate(eq)]
        assert simple_returns(eq) == compute_returns_from_equity(curve)

    def test_short_inputs(self) -> None:
        assert simple_returns([]) == []
        assert simple_returns([Decimal(1)]) == []

    def test_non_positive_previous_gives_zero(self) -> None:
        r = simple_returns([Decimal(0), Decimal(5), Decimal(-1), Decimal(2)])
        assert r == [0.0, -1.2, 0.0]
        assert all(type(x) is float for x in r)

    def test_barsample_returns_delegates(self) -> None:
        s = BarSample(
            start_at=T0, end_at=ts(2), timeframe_s=H, stamps=(T0, ts(1), ts(2)),
            equity=(Decimal(100), Decimal(110), Decimal(99)),
            mode="calendar_asof", carried_slots=0,
        )  # fmt: skip
        assert s.returns() == simple_returns(list(s.equity))
        assert s.n_returns == 2
