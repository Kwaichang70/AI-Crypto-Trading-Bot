"""WP2.4 / P0: point-in-time leaf ``packages/data/point_in_time.py``.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-01, -03, -04, -05, -17,
WP24-T-01..T-12; T-11 lives in ``test_wp24_ast_pins.py``). The vectors are ported from the
quant oracle
(``pit_ref.py`` / ``test_pit_ref.py``) with Decimal values. Two things differ from the oracle on
purpose: the T10Y2Y fallback is ``observed + 2 business days + 1 day`` (SY-24-04; the expected
values below are recomputed), and the staleness age counts from the FIRST availability of the
selected observation (SY-24-04), not from the latest revision.
"""

from __future__ import annotations

import json
import pickle
import random
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

import data.point_in_time as pit
from data.point_in_time import (
    BACK_7D_MS,
    FIELD_MAP,
    MIN_COVERAGE_PPM,
    NOT_SERVED_FIELDS,
    PIT_VERSION,
    SERIES_RULES,
    Observation,
    PitError,
    PitSeries,
    available_at_estimated,
    available_at_vintage,
    parse_jsonl_lines,
    series_digest,
    to_ms,
)

H = 3_600_000
DAY = 24 * H
D0 = to_ms(datetime(2024, 3, 4, tzinfo=UTC))  # Monday 00:00 UTC
FGI_ID = "alternative_me:fgi"
T10_ID = "fred:T10Y2Y"
DOM_ID = "coingecko:btc_dominance"
WHALE_ID = "whale:net_flow_1h"
FGI_STALE = 48 * H


def ms(*args: int) -> int:
    return to_ms(datetime(*args, tzinfo=UTC))


def fgi_series(values: list[int], start: int = D0) -> PitSeries:
    rows = tuple(
        Observation(
            start + i * DAY,
            available_at_estimated(FGI_ID, start + i * DAY),
            Decimal(v),
            "estimated_lag",
        )
        for i, v in enumerate(values)
    )
    return PitSeries(FGI_ID, rows)


FGI = fgi_series([20, 40, 60, 80, 55, 30, 10, 45, 70])  # Mon 4 Mar .. Tue 12 Mar


def asof_fgi(series: PitSeries, t: int) -> Decimal | None:
    return series.asof(t, max_age_ms=FGI_STALE)


def ref_asof(rows: list[Observation], t: int, max_age: int) -> Decimal | None:
    """Naive reference: linear scan, first availability taken over all rows of the same time."""
    visible = [o for o in rows if o.available_ms <= t]
    if not visible:
        return None
    best = max(visible, key=lambda o: (o.observed_ms, o.available_ms))
    first = min(o.available_ms for o in rows if o.observed_ms == best.observed_ms)
    return best.value if t - first <= max_age else None


# ---------------------------------------------------------------------------
# WP24-T-01: no look-ahead
# ---------------------------------------------------------------------------


class TestT01NoLookAhead:
    def test_future_perturbation_invariance_1000_trials(self) -> None:
        rng = random.Random(24)  # noqa: S311 - deterministic test data
        for _ in range(1000):
            n = rng.randint(1, 30)
            rows: list[Observation] = []
            for i in range(n):
                observed = D0 + i * DAY
                first = observed + rng.randint(0, 3 * DAY)
                rows.append(Observation(observed, first, Decimal(rng.randint(0, 100)), "recorded"))
                if rng.random() < 0.3:  # a revision, always later than the first sighting
                    rows.append(
                        Observation(
                            observed,
                            first + rng.randint(1, 2 * DAY),
                            Decimal(rng.randint(0, 100)),
                            "recorded",
                        )
                    )
            t = D0 + rng.randint(0, (n + 4) * DAY)
            mutated = [
                o
                if o.available_ms <= t
                else Observation(o.observed_ms, o.available_ms, Decimal(999), o.basis)
                for o in rows
            ]
            dropped = [o for o in rows if o.available_ms <= t]
            got = PitSeries("x", tuple(rows)).asof(t, max_age_ms=FGI_STALE)
            assert got == PitSeries("x", tuple(mutated)).asof(t, max_age_ms=FGI_STALE)
            assert got == PitSeries("x", tuple(dropped)).asof(t, max_age_ms=FGI_STALE)
            assert got == ref_asof(rows, t, FGI_STALE)
            assert got != Decimal(999)

    def test_prefix_invariance_over_a_216_bar_timeline(self) -> None:
        closes = [D0 + (k + 1) * H for k in range(24 * 9)]
        assert len(closes) == 216
        full = [asof_fgi(FGI, c) for c in closes]
        for k in (0, 23, 24, 25, 100, len(closes) - 1):
            cut = PitSeries(
                FGI_ID, tuple(o for o in FGI.observations if o.available_ms <= closes[k])
            )
            assert [asof_fgi(cut, c) for c in closes[: k + 1]] == full[: k + 1]


# ---------------------------------------------------------------------------
# WP24-T-02: inclusive boundary, FGI intraday and daily tables
# ---------------------------------------------------------------------------


class TestT02Boundary:
    @pytest.mark.parametrize(
        ("close_i", "expected"),
        [
            (D0 + 5 * H, None),  # Mon 05:00: FGI(Mon) is not usable until Tue 00:00
            (D0 + DAY - H, None),  # Mon 23:00
            (D0 + DAY, 20),  # Tue 00:00: inclusive boundary
            (D0 + DAY + 5 * H, 20),  # Tue 05:00
            (D0 + 2 * DAY + 13 * H, 40),  # Wed 13:00 -> FGI(Tue)
            (D0 + 4 * DAY, 80),  # Fri 00:00 -> FGI(Thu)
        ],
    )
    def test_fgi_intraday(self, close_i: int, expected: int | None) -> None:
        got = asof_fgi(FGI, close_i)
        assert got == (None if expected is None else Decimal(expected))

    def test_exactly_equal_is_visible_one_ms_earlier_is_not(self) -> None:
        for obs in FGI.observations:
            assert asof_fgi(FGI, obs.available_ms) == obs.value
        assert asof_fgi(FGI, FGI.observations[0].available_ms - 1) is None
        assert asof_fgi(FGI, FGI.observations[3].available_ms - 1) == FGI.observations[2].value

    @pytest.mark.parametrize(("day", "expected"), [(0, 20), (1, 40), (2, 60), (3, 80), (8, 70)])
    def test_fgi_daily_bars_see_their_own_day(self, day: int, expected: int) -> None:
        close_i = D0 + day * DAY + DAY  # bar D opens D 00:00 and closes D+1 00:00
        assert asof_fgi(FGI, close_i) == Decimal(expected)


# ---------------------------------------------------------------------------
# WP24-T-03: revisions
# ---------------------------------------------------------------------------


class TestT03Revision:
    def _series(self) -> PitSeries:
        return PitSeries(
            "x",
            (
                Observation(D0, D0 + 10 * H, Decimal(40), "recorded"),
                Observation(D0, D0 + 30 * H, Decimal(55), "recorded"),  # the revision
            ),
        )

    def test_revision_invisible_before_and_wins_from_its_availability(self) -> None:
        s = self._series()
        big = 100 * DAY
        assert s.asof(D0 + 10 * H, max_age_ms=big) == Decimal(40)
        assert s.asof(D0 + 30 * H - 1, max_age_ms=big) == Decimal(40)
        assert s.asof(D0 + 30 * H, max_age_ms=big) == Decimal(55)
        assert s.asof(D0 + 60 * H, max_age_ms=big) == Decimal(55)

    def test_revision_does_not_refresh_the_staleness_age(self) -> None:
        s = self._series()
        cap = 48 * H  # counted from the FIRST availability (D0 + 10h)
        assert s.asof(D0 + 58 * H, max_age_ms=cap) == Decimal(55)  # exactly 48h old
        assert s.asof(D0 + 58 * H + 1, max_age_ms=cap) is None  # revised at +30h, still expired

    def test_a_newer_observation_beats_a_later_revision_of_an_older_one(self) -> None:
        s = PitSeries(
            "x",
            (
                Observation(D0, D0 + H, Decimal(1), "recorded"),
                Observation(D0 + DAY, D0 + DAY + H, Decimal(2), "recorded"),
                Observation(D0, D0 + DAY + 5 * H, Decimal(9), "recorded"),  # old time, revised late
            ),
        )
        assert s.asof(D0 + DAY + 6 * H, max_age_ms=10 * DAY) == Decimal(2)


# ---------------------------------------------------------------------------
# WP24-T-04: staleness (inclusive) and the pinned constants
# ---------------------------------------------------------------------------


class TestT04Staleness:
    def test_gap_turns_the_value_off(self) -> None:
        gappy = PitSeries(
            FGI_ID, tuple(o for o in FGI.observations if o.observed_ms < D0 + 2 * DAY)
        )
        assert asof_fgi(gappy, D0 + 2 * DAY) == Decimal(40)  # Wed 00:00, 0h old
        assert asof_fgi(gappy, D0 + 4 * DAY) == Decimal(40)  # Fri 00:00, 48h old: inclusive
        assert asof_fgi(gappy, D0 + 4 * DAY + H) is None  # 49h old

    def test_zero_cap_means_only_the_availability_instant(self) -> None:
        obs = FGI.observations[0]
        assert FGI.asof(obs.available_ms, max_age_ms=0) == obs.value
        assert FGI.asof(obs.available_ms + 1, max_age_ms=0) is None

    def test_constants_are_pinned_by_value(self) -> None:
        assert PIT_VERSION == "wp24-1"
        assert BACK_7D_MS == 604_800_000
        assert MIN_COVERAGE_PPM == 990_000
        assert {k: v.max_age_ms for k, v in SERIES_RULES.items()} == {
            FGI_ID: 48 * H,
            T10_ID: 7 * DAY,
            DOM_ID: 6 * H,
            WHALE_ID: 2 * H,
        }
        assert all(r.in_backtest for r in SERIES_RULES.values())
        assert {k: v.fields for k, v in SERIES_RULES.items()} == {
            FGI_ID: ("fear_greed_index", "fear_greed_index_7d_ago"),
            T10_ID: ("yield_curve_spread",),
            DOM_ID: ("btc_dominance", "btc_dominance_7d_ago"),
            WHALE_ID: ("whale_net_flow",),
        }
        assert {k: (v.series_id, v.back_ms) for k, v in FIELD_MAP.items()} == {
            "fear_greed_index": (FGI_ID, 0),
            "fear_greed_index_7d_ago": (FGI_ID, BACK_7D_MS),
            "yield_curve_spread": (T10_ID, 0),
            "btc_dominance": (DOM_ID, 0),
            "btc_dominance_7d_ago": (DOM_ID, BACK_7D_MS),
            "whale_net_flow": (WHALE_ID, 0),
        }
        assert NOT_SERVED_FIELDS == {
            "fed_funds_rate",
            "market_cap_change_24h",
            "total_volume_change_24h",
        }
        assert not NOT_SERVED_FIELDS & set(FIELD_MAP)
        assert len(FIELD_MAP) + len(NOT_SERVED_FIELDS) == 9  # the nine MTF signal fields
        assert pit.SERIES_RULES is SERIES_RULES

    def test_rule_tables_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            SERIES_RULES["x"] = SERIES_RULES[FGI_ID]  # type: ignore[index]
        with pytest.raises(TypeError):
            FIELD_MAP["x"] = FIELD_MAP["fear_greed_index"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# WP24-T-05: "7 days ago" is asof(close_i - 7d)
# ---------------------------------------------------------------------------


def _leak_no_lag(s: PitSeries, t: int) -> Decimal | None:
    """Mutant: ignores availability and uses the observation time."""
    seen = [o for o in s.observations if o.observed_ms <= t]
    return max(seen, key=lambda o: o.observed_ms).value if seen else None


def _leak_nearest(s: PitSeries, t: int) -> Decimal:
    """Mutant: the nearest snapshot (like the live ``value_at_offset_from_cache``)."""
    return min(s.observations, key=lambda o: abs(o.observed_ms - t)).value


class TestT05SevenDaysAgo:
    def test_now_and_seven_days_ago(self) -> None:
        t = D0 + 8 * DAY + 10 * H  # Tue 12 Mar 10:00
        assert asof_fgi(FGI, t) == Decimal(45)  # FGI(Mon 11)
        # t - 7d = Tue 5 Mar 10:00: FGI(Tue 5) is the nearest but is only usable on Wed 6
        assert asof_fgi(FGI, t - BACK_7D_MS) == Decimal(20)

    def test_same_staleness_is_applied_at_the_earlier_time(self) -> None:
        gappy = PitSeries(FGI_ID, FGI.observations[:2])  # Mon, Tue only; last usable from Wed 00:00
        t = D0 + 9 * DAY  # Wed 13 Mar 00:00
        assert asof_fgi(gappy, t) is None  # 7 days old
        assert asof_fgi(gappy, t - BACK_7D_MS) == Decimal(40)

    def test_the_nearest_snapshot_and_ignore_lag_mutants_are_caught(self) -> None:
        t = D0 + 8 * DAY + 10 * H
        assert _leak_nearest(FGI, t - BACK_7D_MS) == Decimal(40)
        assert _leak_nearest(FGI, t - BACK_7D_MS) != asof_fgi(FGI, t - BACK_7D_MS)
        mon_5 = D0 + 5 * H
        assert _leak_no_lag(FGI, mon_5) == Decimal(20)
        assert asof_fgi(FGI, mon_5) is None
        assert _leak_nearest(FGI, D0 + DAY - H) == Decimal(40)  # FGI(Tue) at Mon 23:00
        assert asof_fgi(FGI, D0 + DAY - H) is None


# ---------------------------------------------------------------------------
# WP24-T-06: no history; recorded snapshots
# ---------------------------------------------------------------------------


class TestT06NoHistoryAndRecorded:
    def test_no_history_is_none_everywhere(self) -> None:
        empty = PitSeries(FGI_ID, ())
        assert all(empty.asof(D0 + k * H, max_age_ms=FGI_STALE) is None for k in range(200))
        assert empty.observations == ()

    def test_recorded_snapshot_window(self) -> None:
        fetched = D0 + 10 * H + 7 * 60_000  # 10:07
        s = PitSeries(WHALE_ID, (Observation(fetched, fetched, Decimal(-6_000_000), "recorded"),))
        cap = SERIES_RULES[WHALE_ID].max_age_ms
        assert (
            s.asof(D0 + 10 * H, max_age_ms=cap) is None
        )  # the bar closing 10:00 precedes the fetch
        assert s.asof(D0 + 11 * H, max_age_ms=cap) == Decimal(-6_000_000)
        assert s.asof(D0 + 12 * H, max_age_ms=cap) == Decimal(-6_000_000)  # 1h53m old
        assert s.asof(D0 + 13 * H, max_age_ms=cap) is None  # 2h53m old


# ---------------------------------------------------------------------------
# WP24-T-07: ingestion-side availability
# ---------------------------------------------------------------------------


class TestT07IngestionAvailability:
    def test_fgi_is_observed_plus_24h(self) -> None:
        assert available_at_estimated(FGI_ID, D0) == D0 + DAY
        assert _code_of(lambda: available_at_estimated(FGI_ID, D0 + 5 * H)) == "pit_bad_value"

    def test_t10y2y_fallback_values_recomputed(self) -> None:
        # observed + 2 business days (Mon-Fri) + 1 day.  QD's (oracle) rule gave Thu -> Sat 00:00
        # and Fri -> Tue 00:00; the binding rule gives one business day more:
        mon, tue, wed = ms(2024, 3, 4), ms(2024, 3, 5), ms(2024, 3, 6)
        thu, fri = ms(2024, 3, 7), ms(2024, 3, 8)
        assert available_at_estimated(T10_ID, mon) == ms(2024, 3, 7)  # Wed + 1d = Thu 00:00
        assert available_at_estimated(T10_ID, tue) == ms(2024, 3, 8)  # Thu + 1d = Fri 00:00
        assert available_at_estimated(T10_ID, wed) == ms(2024, 3, 9)  # Fri + 1d = Sat 00:00
        assert available_at_estimated(T10_ID, thu) == ms(2024, 3, 12)  # Mon(bd2) + 1d = Tue 00:00
        assert available_at_estimated(T10_ID, fri) == ms(2024, 3, 13)  # Tue(bd2) + 1d = Wed 00:00
        assert available_at_estimated(T10_ID, ms(2024, 3, 9)) == ms(2024, 3, 13)  # Sat obs
        assert available_at_estimated(T10_ID, ms(2024, 3, 10)) == ms(2024, 3, 13)  # Sun obs

    def test_t10y2y_weekend_series_through_the_join(self) -> None:
        thu, fri = ms(2024, 3, 7), ms(2024, 3, 8)
        s = PitSeries(
            T10_ID,
            (
                Observation(
                    thu, available_at_estimated(T10_ID, thu), Decimal("-0.30"), "estimated_lag"
                ),
                Observation(
                    fri, available_at_estimated(T10_ID, fri), Decimal("-0.35"), "estimated_lag"
                ),
            ),
        )
        cap = SERIES_RULES[T10_ID].max_age_ms
        assert s.asof(ms(2024, 3, 9), max_age_ms=cap) is None  # QD: -0.30 here; now still hidden
        assert s.asof(ms(2024, 3, 11, 23), max_age_ms=cap) is None
        assert s.asof(ms(2024, 3, 12), max_age_ms=cap) == Decimal("-0.30")
        assert s.asof(ms(2024, 3, 12, 23), max_age_ms=cap) == Decimal("-0.30")
        assert s.asof(ms(2024, 3, 13), max_age_ms=cap) == Decimal("-0.35")
        assert s.asof(ms(2024, 3, 13) + cap, max_age_ms=cap) == Decimal("-0.35")  # 7d inclusive
        assert s.asof(ms(2024, 3, 13) + cap + 1, max_age_ms=cap) is None

    def test_fallback_matches_a_calendar_walk_for_every_weekday_of_a_year(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(366):
            day = start + timedelta(days=i)
            cursor, left = day, 2
            while left:
                cursor += timedelta(days=1)
                if cursor.weekday() < 5:
                    left -= 1
            expected = to_ms(cursor + timedelta(days=1))
            assert available_at_estimated(T10_ID, to_ms(day)) == expected

    @pytest.mark.parametrize("sid", [FGI_ID, T10_ID])
    @pytest.mark.parametrize(
        "observed",
        [
            ms(2024, 3, 7, 12),  # Thu 12:00Z (was CR-03: Tue 12:00); now rejected
            ms(2024, 3, 9, 18),  # Sat 18:00Z (was CR-03: Wed 18:00); now rejected
            to_ms(datetime(2024, 3, 7, tzinfo=timezone(timedelta(hours=1)))),  # Thu +01:00
            to_ms(datetime(2024, 3, 7, tzinfo=timezone(timedelta(hours=14)))),  # Thu +14:00
            to_ms(datetime(2024, 3, 7, tzinfo=timezone(timedelta(hours=-5)))),  # west: rejected
            ms(2024, 3, 7) + 1,
            ms(2024, 3, 7) - 1,
        ],
    )
    def test_estimated_and_vintage_reject_non_utc_midnight(self, sid: str, observed: int) -> None:
        # QR-08 / SY-24-04: inputs are day D 00:00 UTC; +01:00 Thu used to give Sat 23:00Z.
        assert _code_of(lambda: available_at_estimated(sid, observed)) == "pit_bad_value"
        assert _code_of(lambda: available_at_vintage(observed)) == "pit_bad_value"

    def test_utc_midnight_is_accepted_before_the_epoch_too(self) -> None:
        assert available_at_vintage(ms(1969, 12, 31)) == 0
        assert available_at_estimated(FGI_ID, ms(1969, 12, 31)) == 0

    def test_vintage_path_is_realtime_start_plus_one_day(self) -> None:
        assert available_at_vintage(ms(2024, 3, 11)) == ms(2024, 3, 12)
        assert available_at_vintage(ms(2024, 3, 8)) == ms(2024, 3, 9)

    def test_other_series_and_bad_input_are_rejected(self) -> None:
        for sid in (DOM_ID, WHALE_ID, "fred:FEDFUNDS", "nope", ""):
            with pytest.raises(PitError) as err:
                available_at_estimated(sid, D0)
            assert err.value.code == "pit_unknown_series"
        for bad in (1.5, True, "1"):
            with pytest.raises(PitError) as err:
                available_at_estimated(FGI_ID, bad)  # type: ignore[arg-type]
            assert err.value.code == "pit_bad_value"
        with pytest.raises(PitError):
            available_at_vintage(1.5)  # type: ignore[arg-type]

    def test_join_takes_no_lag_argument(self) -> None:
        with pytest.raises(TypeError):
            FGI.asof(D0, max_age_ms=FGI_STALE, lag_ms=0)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            FGI.coverage_ppm([D0], back_ms=0, max_age_ms=FGI_STALE, lag_ms=0)  # type: ignore[call-arg]
        # the lag is already in available_ms: the join reports exactly what ingestion stored
        assert FGI.observations[0].available_ms == D0 + DAY


# ---------------------------------------------------------------------------
# WP24-T-08: errors, Decimal-only values
# ---------------------------------------------------------------------------


def _code_of(thunk: object) -> str:
    assert callable(thunk)
    with pytest.raises(PitError) as err:
        thunk()
    return err.value.code


class TestT08Errors:
    def test_closed_code_set_and_fixed_messages(self) -> None:
        assert PitError.CODES == (
            "pit_naive_time",
            "pit_available_before_observed",
            "pit_duplicate",
            "pit_bad_value",
            "pit_unknown_series",
        )
        for code in PitError.CODES:
            assert str(PitError(code)) == f"point-in-time check failed ({code})"
            assert PitError(code).code == code
        with pytest.raises(ValueError, match="unknown point-in-time code"):
            PitError("pit_other")
        assert isinstance(PitError("pit_duplicate"), ValueError)

    def test_pickle_round_trip(self) -> None:
        clone = pickle.loads(pickle.dumps(PitError("pit_bad_value")))  # noqa: S301
        assert isinstance(clone, PitError)
        assert clone.code == "pit_bad_value"
        assert str(clone) == "point-in-time check failed (pit_bad_value)"

    def test_each_code_from_one_input(self) -> None:
        assert _code_of(lambda: to_ms(datetime(2024, 1, 1))) == "pit_naive_time"
        assert (
            _code_of(lambda: Observation(D0 + 1, D0, Decimal(1), "recorded"))
            == "pit_available_before_observed"
        )
        row = Observation(D0, D0, Decimal(1), "recorded")
        assert (
            _code_of(lambda: PitSeries("x", (row, Observation(D0, D0, Decimal(2), "recorded"))))
            == "pit_duplicate"
        )
        assert _code_of(lambda: Observation(D0, D0, 1.5, "recorded")) == "pit_bad_value"  # type: ignore[arg-type]
        assert (
            _code_of(lambda: parse_jsonl_lines([_line("nope", D0, D0, "1", "recorded")]))
            == "pit_unknown_series"
        )

    @pytest.mark.parametrize(
        "value",
        [1.5, 3, "1", None, Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"),
         Decimal("0.123456789"), Decimal("1e16"), Decimal("-1e16"),
         Decimal("1e999999999999"), Decimal("-1e999999999999")],
    )  # fmt: skip
    def test_values_must_be_finite_eight_decimal_decimals(self, value: object) -> None:
        assert _code_of(lambda: Observation(D0, D0, value, "recorded")) == "pit_bad_value"  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value",
        [
            Decimal("0"),
            Decimal("-0.30"),
            Decimal("0.12345678"),
            Decimal("9999999999999999.99999999"),
        ],
    )
    def test_valid_values(self, value: Decimal) -> None:
        assert Observation(D0, D0, value, "recorded").value == value

    @pytest.mark.parametrize(
        ("observed", "available"), [(1.0, 2), (1, 2.0), (True, 2), (1, True), ("1", 2)]
    )
    def test_times_must_be_plain_ints(self, observed: object, available: object) -> None:
        thunk = lambda: Observation(observed, available, Decimal(1), "recorded")  # type: ignore[arg-type]  # noqa: E731
        assert _code_of(thunk) == "pit_bad_value"

    def test_basis_and_series_id_and_row_types(self) -> None:
        assert _code_of(lambda: Observation(D0, D0, Decimal(1), "guess")) == "pit_bad_value"  # type: ignore[arg-type]
        assert _code_of(lambda: PitSeries("", ())) == "pit_bad_value"
        assert _code_of(lambda: PitSeries(3, ())) == "pit_bad_value"  # type: ignore[arg-type]
        assert _code_of(lambda: PitSeries("x", ((D0, D0),))) == "pit_bad_value"  # type: ignore[arg-type]

    def test_non_integral_fgi_is_rejected_integral_accepted(self) -> None:
        bad = Observation(D0, D0 + DAY, Decimal("45.5"), "estimated_lag")
        assert _code_of(lambda: PitSeries(FGI_ID, (bad,))) == "pit_bad_value"
        assert PitSeries(FGI_ID, (Observation(D0, D0 + DAY, Decimal("45.00"), "estimated_lag"),))
        assert PitSeries(T10_ID, (Observation(D0, D0 + DAY, Decimal("0.5"), "estimated_lag"),))

    def test_naive_datetime_and_non_datetime(self) -> None:
        assert _code_of(lambda: to_ms("2024-01-01")) == "pit_naive_time"  # type: ignore[arg-type]
        assert to_ms(datetime(1970, 1, 1, 0, 0, 0, 1999, tzinfo=UTC)) == 1  # floor to ms
        plus2 = datetime(2024, 3, 4, 2, tzinfo=UTC).astimezone(timezone(timedelta(hours=2)))
        assert to_ms(plus2) == D0 + 2 * H

    def test_join_arguments_are_validated(self) -> None:
        for bad in (1.5, True, "1", None):
            assert _code_of(lambda b=bad: FGI.asof(b, max_age_ms=1)) == "pit_bad_value"  # type: ignore[misc]
            assert _code_of(lambda b=bad: FGI.asof(0, max_age_ms=b)) == "pit_bad_value"  # type: ignore[misc]
        assert _code_of(lambda: FGI.asof(0, max_age_ms=-1)) == "pit_bad_value"
        assert _code_of(lambda: FGI.coverage_ppm([0], back_ms=-1, max_age_ms=1)) == "pit_bad_value"

    def test_series_are_immutable(self) -> None:
        with pytest.raises(AttributeError):
            FGI.series_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            FGI.observations[0].value = Decimal(0)  # type: ignore[misc]
        assert isinstance(FGI.observations, tuple)

    def test_construction_sorts_and_accepts_any_iterable(self) -> None:
        rows = list(reversed(FGI.observations))
        built = PitSeries(FGI_ID, rows)  # type: ignore[arg-type]
        assert built == FGI
        assert built.observations == FGI.observations


# ---------------------------------------------------------------------------
# WP24-T-09: coverage_ppm
# ---------------------------------------------------------------------------


class TestT09Coverage:
    DECISIONS = tuple(k * H for k in range(1000))
    BIG = 10_000 * DAY

    def _single(self, available_ms: int) -> PitSeries:
        return PitSeries("x", (Observation(available_ms, available_ms, Decimal(1), "recorded"),))

    def test_990000_boundary_on_and_off(self) -> None:
        on = self._single(10 * H).coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG)
        off = self._single(11 * H).coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG)
        assert on == 990_000 == MIN_COVERAGE_PPM
        assert off == 989_000
        assert on >= MIN_COVERAGE_PPM > off

    def test_back_leg_must_also_be_covered(self) -> None:
        s = self._single(-14 * H)  # now-leg covered everywhere, 24h-ago leg from k >= 10
        assert s.coverage_ppm(self.DECISIONS, back_ms=24 * H, max_age_ms=self.BIG) == 990_000
        assert (
            self._single(-13 * H).coverage_ppm(self.DECISIONS, back_ms=24 * H, max_age_ms=self.BIG)
            == 989_000
        )
        assert s.coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG) == 1_000_000

    def test_a_series_starting_at_half_the_window_is_off(self) -> None:
        got = self._single(500 * H).coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG)
        assert got == 500_000
        assert got < MIN_COVERAGE_PPM

    def test_staleness_reduces_coverage(self) -> None:
        s = self._single(0)
        assert s.coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=99 * H) == 100_000

    def test_zero_decisions_and_empty_series_are_zero(self) -> None:
        assert self._single(0).coverage_ppm([], back_ms=0, max_age_ms=self.BIG) == 0
        assert PitSeries("x", ()).coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG) == 0

    def test_full_coverage_and_floor_rounding(self) -> None:
        assert (
            self._single(0).coverage_ppm(self.DECISIONS, back_ms=0, max_age_ms=self.BIG)
            == 1_000_000
        )
        three = (0, H, 2 * H)
        assert self._single(H).coverage_ppm(three, back_ms=0, max_age_ms=self.BIG) == 666_666

    def test_fgi_daily_series_over_a_window_matches_field_use(self) -> None:
        decisions = [D0 + (k + 1) * H for k in range(24 * 9)]
        rule = SERIES_RULES[FGI_ID]
        now_ppm = FGI.coverage_ppm(decisions, back_ms=0, max_age_ms=rule.max_age_ms)
        week_ppm = FGI.coverage_ppm(decisions, back_ms=BACK_7D_MS, max_age_ms=rule.max_age_ms)
        assert now_ppm == 893_518  # 193 of 216 (the first 23 closes precede Tue 00:00)
        assert week_ppm == 115_740  # 25 of 216: the 7d field needs seven more days of history


# ---------------------------------------------------------------------------
# WP24-T-10: series_digest
# ---------------------------------------------------------------------------


def _two_series() -> list[PitSeries]:
    return [
        FGI,
        PitSeries(
            DOM_ID,
            (
                Observation(D0 + H, D0 + H, Decimal("52.1"), "recorded"),
                Observation(D0 + 2 * H, D0 + 2 * H, Decimal("52.3"), "recorded"),
            ),
        ),
    ]


class TestT10Digest:
    def test_known_answer_is_an_independent_hash_of_the_canonical_text(self) -> None:
        import hashlib

        s = PitSeries(DOM_ID, (Observation(5, 7, Decimal("52.1"), "recorded"),))
        t = PitSeries(FGI_ID, (Observation(1, 2, Decimal("-0"), "estimated_lag"),))
        text = (
            f"{FGI_ID}|1|2|0.00000000|estimated_lag\n"  # sorted by series id
            f"{DOM_ID}|5|7|52.10000000|recorded\n"
        )
        assert series_digest([s, t]) == hashlib.sha256(text.encode()).hexdigest()

    def test_stable_across_input_order_and_reload(self) -> None:
        a, b = _two_series()
        base = series_digest([a, b])
        assert series_digest([b, a]) == base
        assert series_digest(iter([a, b])) == base
        shuffled = PitSeries(a.series_id, tuple(reversed(a.observations)))
        assert series_digest([shuffled, b]) == base
        reloaded = parse_jsonl_lines(_to_lines([a, b]))
        assert series_digest(reloaded.values()) == base
        assert series_digest([]) == series_digest([PitSeries("x", ())])

    def test_value_text_ignores_the_decimal_exponent(self) -> None:
        one = PitSeries("x", (Observation(1, 2, Decimal("1.5"), "recorded"),))
        two = PitSeries("x", (Observation(1, 2, Decimal("1.50000"), "recorded"),))
        assert series_digest([one]) == series_digest([two])

    def test_changes_on_any_row_change(self) -> None:
        a, b = _two_series()
        base = series_digest([a, b])
        row = b.observations[0]
        variants = [
            Observation(row.observed_ms, row.available_ms, Decimal("52.2"), row.basis),
            Observation(row.observed_ms, row.available_ms + 1, row.value, row.basis),
            Observation(row.observed_ms - 1, row.available_ms, row.value, row.basis),
            Observation(row.observed_ms, row.available_ms, row.value, "estimated_lag"),
        ]
        for variant in variants:
            changed = PitSeries(b.series_id, (variant, *b.observations[1:]))
            assert series_digest([a, changed]) != base
        added = PitSeries(
            b.series_id,
            (*b.observations, Observation(D0 + 3 * H, D0 + 3 * H, Decimal(1), "recorded")),
        )
        assert series_digest([a, added]) != base
        assert series_digest([a, PitSeries("coingecko:other", b.observations)]) != base
        assert series_digest([a]) != base

    def test_unchanged_by_excluded_rows(self) -> None:
        """Rows with ``available_ms > end`` are filtered at load time: appends cannot move it."""
        end = D0 + 5 * DAY
        store = list(FGI.observations)
        loaded = PitSeries(FGI_ID, tuple(o for o in store if o.available_ms <= end))
        grown = [*store, Observation(D0 + 20 * DAY, D0 + 21 * DAY, Decimal(5), "estimated_lag")]
        reloaded = PitSeries(FGI_ID, tuple(o for o in grown if o.available_ms <= end))
        assert series_digest([loaded]) == series_digest([reloaded])
        assert series_digest([loaded]) != series_digest([FGI])

    def test_a_series_id_may_appear_once(self) -> None:
        assert _code_of(lambda: series_digest([FGI, FGI])) == "pit_duplicate"


# ---------------------------------------------------------------------------
# WP24-T-12: parse_jsonl_lines
# ---------------------------------------------------------------------------


def _iso(ms_value: int) -> str:
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms_value)).isoformat()


def _line(sid: str, observed: int, available: int, value: str, basis: str) -> str:
    return json.dumps(
        {
            "series_id": sid,
            "observed_at": _iso(observed),
            "available_at": _iso(available),
            "value": value,
            "basis": basis,
        }
    )


def _raw(**over: object) -> str:
    """A valid FGI line with raw JSON overrides (``as_number``: emit ``value`` as a JSON number)."""
    as_number = over.pop("as_number", False)
    row: dict[str, object] = {
        "series_id": FGI_ID,
        "observed_at": "2024-03-04T00:00:00Z",
        "available_at": "2024-03-05T00:00:00Z",
        "value": "1",
        "basis": "recorded",
    }
    row.update(over)
    text = json.dumps(row, separators=(",", ":"), default=str)
    if as_number:
        text = text.replace(f'"value":"{row["value"]}"', f'"value":{row["value"]}')
    return text


def _to_lines(series: list[PitSeries]) -> list[str]:
    return [
        _line(s.series_id, o.observed_ms, o.available_ms, format(o.value, "f"), o.basis)
        for s in series
        for o in s.observations
    ]


class TestT12Jsonl:
    def test_round_trip_equals_the_digest(self) -> None:
        series = _two_series()
        lines = _to_lines(series)
        parsed = parse_jsonl_lines(lines)
        assert set(parsed) == {FGI_ID, DOM_ID}
        assert parsed[FGI_ID] == series[0]
        assert parsed[DOM_ID] == series[1]
        assert series_digest(parsed.values()) == series_digest(series)
        # shuffled lines, blank lines and a generator input give the same result
        mixed = [*lines[::-1], "", "   \n"]
        assert series_digest(parse_jsonl_lines(iter(mixed)).values()) == series_digest(series)

    def test_json_numbers_and_z_suffix_are_accepted(self) -> None:
        line = _raw(value=Decimal(45), as_number=True, basis="estimated_lag")
        assert '"value":45' in line
        s = parse_jsonl_lines([line])[FGI_ID]
        assert s.observations == (Observation(D0, D0 + DAY, Decimal(45), "estimated_lag"),)
        decimal_line = (
            line.replace('"value":45', '"value":-0.30')
            .replace(FGI_ID, T10_ID)
            .replace('"estimated_lag"', '"vintage"')  # Mon + 1d = Tue: the vintage floor
        )
        assert parse_jsonl_lines([decimal_line])[T10_ID].observations[0].value == Decimal("-0.30")

    def test_empty_input(self) -> None:
        assert parse_jsonl_lines([]) == {}
        assert parse_jsonl_lines(["", " "]) == {}

    SECRET = "SECRET-LINE-CONTENT"  # noqa: S105

    @pytest.mark.parametrize(
        ("line", "code"),
        [
            ("{not json SECRET-LINE-CONTENT", "pit_bad_value"),
            ('["SECRET-LINE-CONTENT"]', "pit_bad_value"),
            ('"SECRET-LINE-CONTENT"', "pit_bad_value"),
            ("{" * 100_000, "pit_bad_value"),
            ('{"series_id": "SECRET-LINE-CONTENT"}', "pit_bad_value"),
            (_raw(x="SECRET-LINE-CONTENT"), "pit_bad_value"),  # extra key
            (_line("SECRET-LINE-CONTENT", D0, D0 + DAY, "1", "recorded"), "pit_unknown_series"),
            (_line("fred:FEDFUNDS", D0, D0 + DAY, "1", "recorded"), "pit_unknown_series"),
            (_line(FGI_ID, D0, D0 + DAY, "SECRET-LINE-CONTENT", "recorded"), "pit_bad_value"),
            (_line(FGI_ID, D0, D0 + DAY, "NaN", "recorded"), "pit_bad_value"),
            (_line(FGI_ID, D0, D0 + DAY, "45.5", "recorded"), "pit_bad_value"),
            (_line(FGI_ID, D0, D0 + DAY, "1", "SECRET-LINE-CONTENT"), "pit_bad_value"),
            (_line(FGI_ID, D0 + DAY, D0, "1", "recorded"), "pit_available_before_observed"),
            (
                _line(FGI_ID, D0, D0 + DAY, "1", "recorded").replace("+00:00", ""),
                "pit_naive_time",
            ),
            (_raw(observed_at="2024-03-04", available_at="2024-03-05"), "pit_naive_time"),
            (_raw(observed_at="SECRET-LINE-CONTENT"), "pit_bad_value"),
            (_raw(observed_at=5), "pit_bad_value"),
            (_raw(value=None), "pit_bad_value"),
            (_raw(value="1e999999999999"), "pit_bad_value"),  # CR-01: Overflow
            (_raw(value="1e999999999999", as_number=True), "pit_bad_value"),
            (_raw(value="1E+999999999999999999999"), "pit_bad_value"),
            (_raw(value="1E+999999999999999999999", as_number=True), "pit_bad_value"),
            (_raw(available_at="2024-03-04T01:00:00Z", basis="estimated_lag"),
             "pit_available_before_observed"),  # QR-09: 23h early
            (_raw(available_at="2024-03-04T23:59:59.999Z", basis="estimated_lag"),
             "pit_available_before_observed"),
            (_raw(series_id=T10_ID, available_at="2024-03-06T23:59:59.999Z", basis="estimated_lag"),
             "pit_available_before_observed"),  # Mon needs Thu 00:00
            (_raw(series_id=T10_ID, available_at="2024-03-04T23:59:59.999Z", basis="vintage"),
             "pit_available_before_observed"),
            (_raw(observed_at="2024-03-04T05:00:00Z", available_at="2024-03-06T00:00:00Z",
                  basis="estimated_lag"), "pit_bad_value"),  # QR-08 via JSONL
            (_raw(basis="vintage", available_at="2024-03-05T00:00:00Z"), "pit_bad_value"),  # FGI
            (_raw(series_id=T10_ID, basis="recorded"), "pit_bad_value"),
            (_raw(series_id=DOM_ID, basis="estimated_lag"), "pit_bad_value"),
            (_raw(series_id=WHALE_ID, basis="vintage"), "pit_bad_value"),
        ],
    )  # fmt: skip
    def test_malformed_lines_raise_without_echoing_content(self, line: str, code: str) -> None:
        with pytest.raises(PitError) as err:
            parse_jsonl_lines([line])
        assert err.value.code == code
        assert self.SECRET not in str(err.value)
        assert self.SECRET not in repr(err.value)
        assert err.value.__cause__ is None
        assert err.value.__context__ is None or err.value.__suppress_context__ is True  # CR-02
        assert err.value.args == (f"point-in-time check failed ({code})",)

    def test_duplicate_rows_and_non_str_lines(self) -> None:
        line = _line(FGI_ID, D0, D0 + DAY, "1", "recorded")
        assert _code_of(lambda: parse_jsonl_lines([line, line])) == "pit_duplicate"
        assert _code_of(lambda: parse_jsonl_lines([b"{}"])) == "pit_bad_value"  # type: ignore[list-item]


class TestP0ReviewEdits:
    def test_jsonl_floor_accepts_exact_and_later_stamps(self) -> None:  # QR-09
        exact = _raw(basis="estimated_lag")
        later = _raw(basis="estimated_lag", available_at="2024-03-05T00:00:00.001Z")
        assert parse_jsonl_lines([exact])[FGI_ID].observations[0].available_ms == D0 + DAY
        assert parse_jsonl_lines([later])[FGI_ID].observations[0].available_ms == D0 + DAY + 1
        thu = _raw(series_id=T10_ID, basis="estimated_lag", available_at="2024-03-07T00:00:00Z")
        assert parse_jsonl_lines([thu])[T10_ID].observations[0].available_ms == ms(2024, 3, 7)

    def test_early_recorded_sighting_is_accepted(self) -> None:  # QR-09 / SY-24-09
        line = _raw(available_at="2024-03-04T00:00:00Z", basis="recorded")
        assert parse_jsonl_lines([line])[FGI_ID].observations[0].available_ms == D0

    def test_allowed_bases_follow_the_rule_table(self) -> None:
        assert dict(pit._ALLOWED_BASES) == {
            FGI_ID: frozenset({"estimated_lag", "recorded"}),
            T10_ID: frozenset({"vintage", "estimated_lag"}),
            DOM_ID: frozenset({"recorded"}),
            WHALE_ID: frozenset({"recorded"}),
        }
        assert set(pit._ALLOWED_BASES) == set(SERIES_RULES)

    @pytest.mark.parametrize("sid", ["a|b", "a\nb", "a\rb", "a\tb", "\u00e9", "x\x00"])
    def test_series_id_must_be_printable_ascii_without_pipe(self, sid: str) -> None:  # CR-05
        assert _code_of(lambda: PitSeries(sid, ())) == "pit_bad_value"

    def test_observations_order_is_observed_then_available(self) -> None:  # CR-04
        rows = (
            Observation(D0 + DAY, D0 + DAY, Decimal(2), "recorded"),
            Observation(D0, D0 + 3 * DAY, Decimal(3), "recorded"),  # late older-time row
            Observation(D0, D0 + 2 * H, Decimal(1), "recorded"),
            Observation(D0, D0 + 5 * DAY, Decimal(4), "recorded"),  # revision
        )
        s = PitSeries(DOM_ID, rows)
        assert s.observations == tuple(sorted(rows, key=lambda o: (o.observed_ms, o.available_ms)))
        assert s.observations != tuple(sorted(rows, key=lambda o: (o.available_ms, o.observed_ms)))
