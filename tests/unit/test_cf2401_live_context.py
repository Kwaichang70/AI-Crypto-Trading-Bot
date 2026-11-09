"""
tests/unit/test_cf2401_live_context.py
---------------------------------------
CF-24-01 P3 (unwired): LiveContextBook, converters, ConsumerRegistry and the cadence policy.
T-10 (publish), T-11 (converters), T-12 (caps), T-13 (7d), T-14 (policy), T-14b (isolation),
T-25 (clock guard, book part) and the P3-B1 never-served vector. The AST pins are in
``test_cf2401_ast_pins.py``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, Inexact, localcontext
from types import MappingProxyType

import pytest

import data.live_context_policy as pol
from data.live_context import (
    CLOCK_REGRESSION_TOLERANCE_MS as TOL,
)
from data.live_context import (
    LIVE_BOOK,
    LIVE_CONSUMERS,
    ConsumerRegistry,
    LiveContextBook,
    _value,
    fgi_observations,
    snapshot_observation,
    t10y2y_observation,
    t10y2y_observations,
)
from data.macro_data import MacroSnapshot
from data.market_signals import CoinGeckoSnapshot
from data.point_in_time import (
    BACK_7D_MS,
    FIELD_MAP,
    SERIES_RULES,
    Observation,
    PitError,
    available_at_estimated,
    to_ms,
)
from trading.mtf_signals import PitSignals, SignalValues

H = 3_600_000
DAY = 24 * H
MIN = 60_000
D0 = to_ms(datetime(2024, 3, 4, tzinfo=UTC))  # Monday 00:00 UTC
B = D0 + 10 * H  # a snapshot bucket start
FGI, FRED, DOM, WHALE = (
    "alternative_me:fgi",
    "fred:T10Y2Y",
    "coingecko:btc_dominance",
    "whale:net_flow_1h",
)


def _dt(ms: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)


def _code(thunk: object) -> str:
    with pytest.raises(PitError) as err:
        thunk()
    return err.value.code


def _values(book: LiveContextBook, now: int, fields: set[str]) -> SignalValues:
    return PitSignals(book.view(now), frozenset(fields)).values_at(_dt(now))


def _val(book: LiveContextBook, now: int, field: str) -> object:
    return getattr(_values(book, now, {field}), field)


def _rows(book: LiveContextBook, sid: str) -> tuple[tuple[int, int, Decimal], ...]:
    return tuple(
        (o.observed_ms, o.available_ms, o.value) for o in book.snapshot()[sid].observations
    )


# ---------------------------------------------------------------------------
# T-10: publish
# ---------------------------------------------------------------------------
class TestT10Publish:
    def test_dedup_and_copy_on_write(self) -> None:
        book = LiveContextBook()
        rows = fgi_observations([(D0, 40), (D0 + DAY, 50)], D0 + DAY + H)
        empty = book.snapshot()
        assert book.publish(FGI, rows, D0 + DAY + H) == 2
        first = book.snapshot()
        assert len(empty) == 0 and first is not empty
        assert (
            book.publish(FGI, rows, D0 + DAY + H) == 0
        )  # identical rows: no-op, returns 0 (ER3-02 rule 3)
        assert book.snapshot() is first  # a no-op leaves the very same mapping
        more = fgi_observations([(D0 + 2 * DAY, 60)], D0 + 2 * DAY + H)
        assert book.publish(FGI, more, D0 + 2 * DAY + H) == 1
        assert len(first[FGI].observations) == 2  # the old snapshot is unchanged
        assert len(book.snapshot()[FGI].observations) == 3

    def test_first_sighting_per_hour_wins(self) -> None:
        book = LiveContextBook()
        t1 = D0 + 10 * H + 5 * 60_000
        assert book.publish(DOM, [snapshot_observation(DOM, 52, t1)], t1) == 1
        later = snapshot_observation(DOM, 53, t1 + 20 * 60_000)  # same hour
        assert book.publish(DOM, [later], t1 + 20 * 60_000) == 0
        assert _rows(book, DOM) == ((D0 + 10 * H, t1, Decimal(52)),)
        nxt = D0 + 11 * H + 1
        assert book.publish(DOM, [snapshot_observation(DOM, 53, nxt)], nxt) == 1
        assert len(_rows(book, DOM)) == 2

    def test_v1_a_held_snapshot_bucket_row_is_never_replaced_by_publish(self) -> None:
        """E-1 (ER3-02 rule 4): a later sighting stamped earlier (backward step) cannot replace."""
        book, f1, g = LiveContextBook(), B + 40 * MIN, B + 20 * MIN
        assert book.publish(DOM, [snapshot_observation(DOM, 50, f1)], f1) == 1
        assert book.publish(DOM, [snapshot_observation(DOM, 60, g)], g) == 0  # the old rule: 1
        assert _rows(book, DOM) == ((B, f1, Decimal(50)),)

    def test_not_yet_visible_is_replaceable_and_visible_is_immutable(self) -> None:
        book = LiveContextBook()
        key = (D0, 40), (D0, 41), (D0, 42)
        v40, v41, v42 = (fgi_observations([k], D0 + H) for k in key)
        assert book.publish(FGI, v40, D0 + H) == 1
        assert book.publish(FGI, v41, D0 + H) == 1  # available D0+24h is still in the future
        assert _rows(book, FGI) == ((D0, D0 + DAY, Decimal(41)),)
        assert book.publish(FGI, v42, D0 + DAY) == 0  # visible now: immutable
        assert _rows(book, FGI)[0][2] == Decimal(41)

    def test_trim_window_is_inclusive_and_moves_with_now(self) -> None:
        book = LiveContextBook()
        now = D0 + 20 * DAY
        cutoff = now - BACK_7D_MS - SERIES_RULES[DOM].max_age_ms
        assert book.publish(DOM, [snapshot_observation(DOM, 51, cutoff - 1)], now) == 0
        assert book.publish(DOM, [snapshot_observation(DOM, 50, cutoff)], now) == 1
        assert [r[1] for r in _rows(book, DOM)] == [cutoff]  # == the cutoff is kept
        assert book.publish(DOM, [snapshot_observation(DOM, 52, now)], now + H) == 1
        assert [r[1] for r in _rows(book, DOM)] == [now]  # the old row fell out of the window

    def test_a_bad_publish_leaves_the_book_unchanged(self) -> None:
        book = LiveContextBook()
        book.publish(FGI, fgi_observations([(D0, 40)], D0 + H), D0 + H)
        before = book.snapshot()
        half = fgi_observations([(D0 + DAY, Decimal("40.5"))], D0 + DAY + H)  # FGI is an integer
        assert _code(lambda: book.publish(FGI, half, D0 + DAY + H)) == "pit_bad_value"
        assert _code(lambda: book.publish("nope", [], D0)) == "pit_unknown_series"
        assert _code(lambda: book.publish(FGI, [(D0, 1)], D0)) == "pit_bad_value"
        assert book.snapshot() is before


# ---------------------------------------------------------------------------
# T-11: converters (and the future-date checklist, P3-B2)
# ---------------------------------------------------------------------------
class TestT11Converters:
    def test_fgi_is_estimated_lag_and_rejects_a_non_midnight_day(self) -> None:
        (obs,) = fgi_observations([(D0, 40)], D0 + H)
        assert (obs.observed_ms, obs.available_ms, obs.basis) == (D0, D0 + DAY, "estimated_lag")
        assert obs.available_ms == available_at_estimated(FGI, D0)
        assert _code(lambda: fgi_observations([(D0 + H, 40)], D0 + 2 * H)) == "pit_bad_value"
        assert _code(lambda: fgi_observations([(D0, 40)], D0 - TOL - 1)) == "pit_bad_value"
        assert _code(lambda: fgi_observations([(D0, 40)], float(D0))) == "pit_bad_value"  # type: ignore[arg-type]

    def test_t10y2y_uses_the_observation_date(self) -> None:
        friday = D0 + 4 * DAY
        obs = t10y2y_observation(friday, 0.41, friday + 5 * H)
        assert obs.available_ms == available_at_estimated(FRED, friday) == friday + 5 * DAY
        assert (obs.basis, obs.value) == ("estimated_lag", Decimal("0.41000000"))

    def test_snapshot_series_use_the_hour_bucket_and_the_fetch_stamp(self) -> None:
        fetch = D0 + 7 * H + 1_234_567
        for sid in (DOM, WHALE):
            obs = snapshot_observation(sid, 1, fetch)
            assert (obs.observed_ms, obs.available_ms, obs.basis) == (D0 + 7 * H, fetch, "recorded")
        assert _code(lambda: snapshot_observation(FGI, 1, fetch)) == "pit_unknown_series"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.123456785, "0.12345679"),
            (0.123456784, "0.12345678"),
            (Decimal("1.000000005"), "1.00000001"),
            (Decimal("-0.000000005"), "-0.00000001"),
            (7, "7.00000000"),
        ],
    )
    def test_values_are_quantised_half_up_to_8_decimals(self, raw: object, expected: str) -> None:
        assert format(snapshot_observation(DOM, raw, D0).value, "f") == expected

    @pytest.mark.parametrize(
        "raw",
        [True, None, "52", float("nan"), float("inf"), 1e30, 10**5000]
        + [Decimal(x) for x in ("1e16", "-1e16", "NaN", "sNaN", "1e999999")],
        # explicit ids: pytest's str() of 10**5000 exceeds CPython's int-to-str limit (G-P3-3)
        ids="bool none str nan inf 1e30 int5000 1e16 -1e16 dNaN sNaN huge".split(),
    )
    def test_bad_values_are_pit_bad_value(self, raw: object) -> None:
        assert _code(lambda: snapshot_observation(DOM, raw, D0)) == "pit_bad_value"

    @pytest.mark.parametrize(
        "raw", [10**5000, *map(Decimal, "NaN sNaN 1e999999".split())], ids=range(4)
    )
    def test_value_itself_raises_the_closed_code_never_value_error(self, raw: object) -> None:
        assert _code(lambda: _value(raw)) == "pit_bad_value"

    @pytest.mark.parametrize(
        "raw", [-0.0, Decimal("-0.00"), -1e-9, 0.123456785, Decimal("-0.000000005"), 7, 12345.67]
    )
    def test_value_equals_the_store_quantiser_including_the_sign_of_zero(self, raw: object) -> None:
        from api.services.context_store import quantize_store_value

        got, want = _value(raw), quantize_store_value(raw)  # type: ignore[arg-type]
        assert (got, str(got)) == (want, str(want))

    def test_the_ambient_decimal_context_is_ignored(self) -> None:
        with localcontext() as ctx:
            ctx.prec = 12
            ctx.traps[Inexact] = True
            assert str(_value(12345.67)) == "12345.67000000"
            assert str(_value(0.123456785)) == "0.12345679"

    def test_t10y2y_page_converts_every_row_and_serves_at_a_cold_start(self) -> None:
        """QR-P3-05: a Mon..Fri page fetched next Monday serves Wednesday; newest-only is None."""
        days = [D0 + i * DAY for i in range(5)]
        fetch = D0 + 7 * DAY + 12 * H
        points = [(d, 0.40 + i / 100) for i, d in enumerate(days)]
        rows = t10y2y_observations(points, fetch)
        assert rows == tuple(t10y2y_observation(d, v, fetch) for d, v in points)  # == the singular
        assert all(r.basis == "estimated_lag" for r in rows)
        assert rows[2].available_ms == to_ms(datetime(2024, 3, 9, tzinfo=UTC))
        book, newest_only = LiveContextBook(), LiveContextBook()
        book.publish(FRED, rows, fetch)
        newest_only.publish(FRED, rows[-1:], fetch)
        assert _val(book, fetch, "yield_curve_spread") == 0.42
        assert _val(newest_only, fetch, "yield_curve_spread") is None
        wed = to_ms(datetime(2024, 3, 13, tzinfo=UTC))
        assert _val(newest_only, wed, "yield_curve_spread") == 0.44

    def test_one_future_dated_row_rejects_the_whole_t10y2y_page(self) -> None:
        future = D0 + 8 * DAY  # a day stamp is midnight, so the fetch is placed around it
        assert len(t10y2y_observations([(D0, 1), (future, 2)], future - TOL)) == 2  # inclusive
        page = [(D0, 1), (future, 2)]
        assert _code(lambda: t10y2y_observations(page, future - TOL - 1)) == "pit_bad_value"

    def test_the_three_never_served_fields_are_none_in_signal_values(self) -> None:
        """P3-B1 / CF-2401-30: non-zero 24h values and a fed funds rate never reach the context."""
        cg = CoinGeckoSnapshot(
            btc_dominance=52.5,
            market_cap_change_24h=3.2,
            total_volume_change_24h=9.9,
            timestamp=_dt(D0 + 9 * H),
        )
        macro = MacroSnapshot(
            fed_funds_rate=5.33,
            yield_curve_spread=0.41,
            timestamp=_dt(D0 + 9 * H),
            yield_curve_observed_on=date(2024, 3, 4),
        )
        book = LiveContextBook()
        obs = t10y2y_observation(
            to_ms(datetime(2024, 3, 4, tzinfo=UTC)), macro.yield_curve_spread, D0 + 9 * H
        )
        now = obs.available_ms + 600_000
        book.publish(FRED, [obs], now)
        book.publish(DOM, [snapshot_observation(DOM, cg.btc_dominance, now)], now)
        values = PitSignals(book.view(now), frozenset(FIELD_MAP)).values_at(_dt(now))
        assert values.btc_dominance == 52.5 and values.yield_curve_spread == 0.41
        assert (values.market_cap_change_24h, values.total_volume_change_24h) == (None, None)
        assert values.fed_funds_rate is None
        for name in ("fed_funds_rate", "market_cap_change_24h", "total_volume_change_24h"):
            assert _code(lambda n=name: PitSignals({}, frozenset({n}))) == "pit_unknown_series"


# ---------------------------------------------------------------------------
# T-12: per-series caps, stalls, future availability
# ---------------------------------------------------------------------------
MID = D0 + H + 1_234_567  # a mid-hour fetch: observed != available for the snapshot series


def _one(sid: str) -> tuple[Observation, str]:
    if sid == FGI:
        return fgi_observations([(D0, 40)], D0 + H)[0], "fear_greed_index"
    if sid == FRED:
        return t10y2y_observation(D0, 1.5, D0 + H), "yield_curve_spread"
    field = "btc_dominance" if sid == DOM else "whale_net_flow"
    return snapshot_observation(sid, 50, MID), field


def _two(sid: str) -> list[Observation]:
    """An older and a newer row; the newer one is the visible value the age is measured from."""
    if sid == FGI:
        return list(fgi_observations([(D0, 40), (D0 + DAY, 41)], D0 + DAY + H))
    if sid == FRED:
        return list(t10y2y_observations([(D0, 1.5), (D0 + DAY, 1.6)], D0 + DAY + H))
    return [snapshot_observation(sid, 50, MID), snapshot_observation(sid, 51, MID + H)]


class TestT12Caps:
    @pytest.mark.parametrize("sid", sorted(SERIES_RULES))
    def test_age_equal_to_the_cap_is_served_and_one_ms_more_is_none(self, sid: str) -> None:
        obs, field = _one(sid)
        cap = SERIES_RULES[sid].max_age_ms
        book = LiveContextBook()
        book.publish(sid, [obs], obs.available_ms)
        at_cap = _values(book, obs.available_ms + cap, {field})
        assert getattr(at_cap, field) is not None
        over = _values(book, obs.available_ms + cap + 1, {field})
        assert getattr(over, field) is None

    @pytest.mark.parametrize("sid", sorted(SERIES_RULES))
    def test_freshness_values_at_the_cap_one_ms_over_and_at_availability(self, sid: str) -> None:
        """CR-P3-01: rows, fresh and age_ms (counted from availability, not from observed)."""
        older, newer = _two(sid)
        cap = SERIES_RULES[sid].max_age_ms
        book = LiveContextBook()
        assert book.publish(sid, [older, newer], newer.available_ms) == 2
        assert older.available_ms < newer.available_ms and newer.observed_ms != newer.available_ms

        def state(now: int) -> tuple[int, bool, int | None]:
            s = book.freshness(now).series[sid]
            return s.rows, s.fresh, s.age_ms

        assert state(newer.available_ms) == (2, True, 0)  # visible exactly at availability
        assert state(newer.available_ms - 1)[0] == 2  # rows counts stored rows, visible or not
        assert state(newer.available_ms + cap) == (2, True, cap)
        assert state(newer.available_ms + cap + 1) == (2, False, cap + 1)

    @pytest.mark.parametrize("sid", [FGI, FRED])
    def test_a_stall_does_not_refresh_a_provider_dated_series(self, sid: str) -> None:
        book = LiveContextBook()
        first, _ = _one(sid)
        book.publish(sid, [first], D0 + H)
        now = D0 + 30 * H
        age = book.freshness(now).series[sid].age_ms
        rows = (
            fgi_observations([(D0, 40)], D0 + 20 * H)
            if sid == FGI
            else [t10y2y_observation(D0, 1.5, D0 + 20 * H)]
        )
        assert book.publish(sid, rows, D0 + 20 * H) == 0  # same observed/value, later fetch
        assert book.freshness(now).series[sid].age_ms == age

    def test_a_frozen_snapshot_value_in_a_new_bucket_does_refresh_the_age(self) -> None:
        book = LiveContextBook()
        t1 = D0 + H
        book.publish(DOM, [snapshot_observation(DOM, 50, t1)], t1)
        t2 = t1 + 3 * H
        book.publish(DOM, [snapshot_observation(DOM, 50, t2)], t2)  # documented behaviour (ARC-09)
        now = t1 + 6 * H + 1  # the first row alone would be stale here
        state = book.freshness(now).series[DOM]
        assert state.fresh and state.age_ms == now - t2

    def test_a_future_available_row_is_invisible(self) -> None:
        book = LiveContextBook()
        book.publish(FGI, fgi_observations([(D0, 40)], D0 + H), D0 + H)
        assert _values(book, D0 + DAY - 1, {"fear_greed_index"}).fear_greed_index is None
        assert _values(book, D0 + DAY, {"fear_greed_index"}).fear_greed_index == 40
        assert book.freshness(D0 + DAY - 1).series[FGI].age_ms is None


# ---------------------------------------------------------------------------
# T-13: 7d = asof(now - 7d)
# ---------------------------------------------------------------------------
class TestT13SevenDays:
    def test_fgi_7d_is_served_right_after_the_first_limit_30_poll(self) -> None:
        fetch = D0 + 29 * DAY + H
        points = [(D0 + i * DAY, 10 + i) for i in range(30)]
        book = LiveContextBook()
        book.publish(FGI, fgi_observations(points, fetch), fetch)
        got = _values(book, fetch, {"fear_greed_index", "fear_greed_index_7d_ago"})
        assert (got.fear_greed_index, got.fear_greed_index_7d_ago) == (38, 31)
        series = book.snapshot()[FGI]
        target = fetch - BACK_7D_MS
        nearest = min(series.observations, key=lambda o: abs(o.observed_ms - target)).value
        no_lag = max(
            (o for o in series.observations if o.observed_ms <= target), key=lambda o: o.observed_ms
        )
        assert nearest == no_lag.value == Decimal(32) != Decimal(31)  # both mutants are caught

    def test_dominance_7d_needs_a_recorded_row_old_enough(self) -> None:
        book = LiveContextBook()
        t0 = D0 + H
        want = {"btc_dominance", "btc_dominance_7d_ago"}
        for i in range(7 * 24):
            book.publish(DOM, [snapshot_observation(DOM, 40 + i % 10, t0 + i * H)], t0 + i * H)
        before = _values(book, t0 + 7 * 24 * H - H, want)
        assert before.btc_dominance_7d_ago is None and before.btc_dominance is not None
        after = _values(book, t0 + 7 * 24 * H, want)
        assert after.btc_dominance_7d_ago == 40.0

    def test_the_cap_is_evaluated_at_the_earlier_time(self) -> None:
        book = LiveContextBook()
        t0 = D0 + H
        book.publish(DOM, [snapshot_observation(DOM, 40, t0)], t0)
        t1 = t0 + 7 * DAY + 7 * H  # now - 7d = t0 + 7h: the t0 row is older than 6h there
        book.publish(DOM, [snapshot_observation(DOM, 41, t1)], t1)
        got = _values(book, t1, {"btc_dominance", "btc_dominance_7d_ago"})
        assert (got.btc_dominance, got.btc_dominance_7d_ago) == (41.0, None)


# ---------------------------------------------------------------------------
# T-14: cadence policy (the AST pins are in test_cf2401_ast_pins.py)
# ---------------------------------------------------------------------------
class TestT14Policy:
    def test_failed_cycles_double_the_interval_up_to_cap_over_4(self) -> None:
        assert [pol.next_interval_s(DOM, n) for n in (0, 3, 4, 5, 6, 7, 8, 50)] == [
            600,
            600,
            1_200,
            2_400,
            4_800,
            5_400,
            5_400,
            5_400,
        ]
        assert pol.next_interval_s(FGI, 4) == 7_200 and pol.next_interval_s(FGI, 99) == 43_200

    @pytest.mark.parametrize("sid", sorted(SERIES_RULES))
    def test_the_interval_always_lies_between_the_floor_and_the_series_max_gap(
        self, sid: str
    ) -> None:
        gaps = [pol.next_interval_s(sid, failed) for failed in range(101)]
        assert pol.FLOOR_S <= min(gaps) and max(gaps) <= pol.max_gap_s(sid)

    @pytest.mark.parametrize(
        ("given", "expected"), [(5, 60), (60, 60), (120, 120), (999_999, 3_600)]
    )
    def test_retry_after_is_clamped_between_the_floor_and_one_hour(
        self, given: int, expected: int
    ) -> None:
        assert pol.retry_after_next_s(given) == expected

    def test_waits_must_leave_room_for_one_more_attempt(self) -> None:
        assert pol.wait_fits_cycle(5, 15) and not pol.wait_fits_cycle(5.1, 15)
        assert pol.wait_fits_cycle(0, 10) and not pol.wait_fits_cycle(0, 9.9)
        assert [pol.backoff_s(n, 0.5) for n in (0, 1, 2, 5, 9)] == [1.5, 2.5, 4.5, 32.5, 60.0]


# ---------------------------------------------------------------------------
# T-25 (book part): clock-regression guard
# ---------------------------------------------------------------------------
class TestT25ClockGuard:
    """Coverage per ER3-01 (r3.1): only ``recorded`` rows stamped more than
    CLOCK_REGRESSION_TOLERANCE_MS beyond the corrected clock are dropped; the V2-V6 vectors correct
    the step less than ``S - TOL`` after the stamped publish. A longer-lasting offset keeps its
    earlier step rows (R3-RES-2; asserted by the P4a T-25 reader clause)."""

    def _book(self) -> LiveContextBook:
        book = LiveContextBook()
        book.publish(FGI, fgi_observations([(D0, 40)], D0 + H), D0 + H)
        return book

    def test_view_returns_empty_when_the_clock_stepped_back_more_than_5_minutes(self) -> None:
        book, hw = self._book(), D0 + 3 * DAY
        assert len(book.view(hw)) == 1
        assert len(book.view(hw - TOL)) == 1  # tolerance inclusive
        assert book.regressed_reads == 0
        assert len(book.view(hw - TOL - 1)) == 0 and book.regressed_reads == 1
        assert book._high_water_ms == hw  # unchanged by a regressed read
        assert len(book.view(hw + 1)) == 1 and book._high_water_ms == hw + 1

    def test_heartbeat_uses_the_same_rule_and_only_raises_the_mark_when_not_regressed(self) -> None:
        book, hw = self._book(), D0 + 3 * DAY
        assert book.heartbeat(hw) is False and book._high_water_ms == hw
        assert book.heartbeat(hw - TOL) is False and book._high_water_ms == hw
        assert book.heartbeat(hw - TOL - 1) is True and book._high_water_ms == hw
        assert book.regressed_reads == 0  # only reader views count
        assert book.heartbeat(hw + 5) is False and book._high_water_ms == hw + 5

    def test_a_mark_raised_by_a_reader_view_alone_also_guards(self) -> None:
        book = self._book()
        book.view(D0 + 9 * DAY)  # dead poller: no heartbeat, only reads
        assert len(book.view(D0 + 9 * DAY - 10 * 60_000)) == 0

    def test_a_forward_step_later_corrected_stays_neutral_until_the_clock_passes_the_mark(
        self,
    ) -> None:
        book, real = self._book(), D0 + DAY + H
        assert len(book.view(real)) == 1
        book.view(real + 2 * DAY)  # clock jumps forward, then NTP corrects it
        assert len(book.view(real + H)) == 0
        assert len(book.view(real + 2 * DAY - TOL)) == 1

    def test_freshness_reports_regressed_but_changes_nothing(self) -> None:
        book, hw = self._book(), D0 + 3 * DAY
        book.view(hw)
        assert book.freshness(hw - TOL).regressed is False
        assert book.freshness(hw - TOL - 1).regressed is True
        book.freshness(hw + 10 * DAY)  # a far-future probe must not raise the mark
        assert (book._high_water_ms, book.regressed_reads) == (hw, 0)
        assert len(book.view(hw)) == 1

    @pytest.mark.parametrize("bad", [True, 1.5, "5", None])
    def test_a_non_int_time_is_rejected(self, bad: object) -> None:
        book = LiveContextBook()
        calls = (book.view, book.heartbeat, book.freshness, lambda t: book.publish(DOM, [], t))
        for call in calls:  # publish too (C04)
            assert _code(lambda c=call: c(bad)) == "pit_bad_value"  # type: ignore[arg-type]

    def test_view_is_a_read_only_mapping_in_both_branches(self) -> None:
        """CR-P3-02: the engine can neither mutate nor diverge from the shared snapshot."""
        book, hw = self._book(), D0 + 3 * DAY
        normal = book.view(hw)
        assert isinstance(normal, MappingProxyType) and normal is book.snapshot()
        regressed = book.view(hw - TOL - 1)
        assert isinstance(regressed, MappingProxyType) and len(regressed) == 0
        assert book.snapshot() is normal  # nothing to drop: no mapping swap (ER3-01)
        for mapping in (normal, regressed):
            with pytest.raises(TypeError):
                mapping["x"] = 1  # type: ignore[index]

    def test_v2_backward_step_drop_then_the_first_sighting_in_the_current_frame_wins(self) -> None:
        book, f1, g = LiveContextBook(), B + 40 * MIN, B + 20 * MIN
        book.publish(DOM, [snapshot_observation(DOM, 50, f1)], f1)
        assert book.heartbeat(g) is True
        assert DOM not in book.snapshot()  # f1 > g + TOL: dropped, and the series key removed
        assert book.publish(DOM, [snapshot_observation(DOM, 60, g)], g) == 1
        assert book.publish(DOM, [snapshot_observation(DOM, 70, g + 10 * MIN)], g + 10 * MIN) == 0
        assert _rows(book, DOM) == ((B, g, Decimal(60)),)
        assert len(book.view(f1 - TOL - 1)) == 0  # still regressed against the mark f1
        assert _val(book, f1 - TOL, "btc_dominance") == 60.0

    @pytest.mark.parametrize("entry", ["heartbeat", "view"])
    @pytest.mark.parametrize(("delta", "kept"), [(-TOL - 1, False), (-TOL, True)])
    def test_v3_the_drop_bound_is_now_plus_tolerance(
        self, entry: str, delta: int, kept: bool
    ) -> None:
        f1 = B + 40 * MIN
        book = LiveContextBook()
        book.publish(DOM, [snapshot_observation(DOM, 50, f1)], f1)
        book.heartbeat(f1 + 2 * TOL)  # the mark is above the row: both probes are regressed
        assert book.freshness(f1 + delta).regressed
        getattr(book, entry)(f1 + delta)
        assert (DOM in book.snapshot()) is kept

    def test_v4_rows_stamped_beyond_the_corrected_clock_plus_tol_are_dropped_by_heartbeat(
        self,
    ) -> None:
        book, real = LiveContextBook(), B + 2 * MIN
        old, fwd = real - 30 * MIN, real + 3 * DAY
        book.publish(WHALE, [snapshot_observation(WHALE, 3, old)], old)
        assert book.heartbeat(fwd) is False
        book.publish(WHALE, [snapshot_observation(WHALE, -5, fwd)], fwd)
        book.publish(
            FGI, fgi_observations([(D0 + 3 * DAY, 40)], fwd), fwd
        )  # available in the future
        fgi_before = book.snapshot()[FGI]
        assert book.heartbeat(real + MIN) is True
        assert _rows(book, WHALE) == ((B - H, old, Decimal(3)),)  # the older real row survives
        assert book.snapshot()[FGI] is fgi_before  # estimated_lag rows are never touched
        late = fwd + MIN  # the real clock reaches the forward-stamped bucket
        assert book.publish(WHALE, [snapshot_observation(WHALE, 7, late)], late) == 1
        assert _val(book, late, "whale_net_flow") == 7.0

    def test_v5_rows_stamped_beyond_the_corrected_clock_plus_tol_are_dropped_by_view_alone(
        self,
    ) -> None:
        book, real = LiveContextBook(), B + 2 * MIN
        fwd = real + 3 * DAY
        book.view(fwd)  # dead poller: reads only
        book.publish(WHALE, [snapshot_observation(WHALE, -5, fwd)], fwd)
        assert len(book.view(real + MIN)) == 0 and book.regressed_reads == 1
        assert WHALE not in book.snapshot()

    def test_v6_publish_alone_raises_the_mark_so_the_correction_is_detected(self) -> None:
        book, real = LiveContextBook(), B + 2 * MIN
        fwd = real + 3 * DAY
        assert book.publish(WHALE, [snapshot_observation(WHALE, -5, fwd)], fwd) == 1
        assert book._high_water_ms == fwd  # no heartbeat or view happened at fwd
        assert len(book.view(real)) == 0 and WHALE not in book.snapshot()
        late = fwd + MIN
        assert book.publish(WHALE, [snapshot_observation(WHALE, 7, late)], late) == 1
        assert _val(book, late, "whale_net_flow") == 7.0

    def test_v7_a_noop_or_failed_publish_leaves_the_mark(self) -> None:
        book, t = LiveContextBook(), B + MIN
        book.publish(DOM, [snapshot_observation(DOM, 50, t)], t)
        far = t + 10 * DAY
        assert book.publish(DOM, [snapshot_observation(DOM, 50, t)], far) == 0
        half = fgi_observations([(D0 + 10 * DAY, Decimal("40.5"))], far)  # FGI is an integer
        assert _code(lambda: book.publish(FGI, half, far)) == "pit_bad_value"
        assert _code(lambda: book.publish(FGI, [(D0, 1)], far)) == "pit_bad_value"  # type: ignore[list-item]
        assert book._high_water_ms == t

    def test_v8_a_recorded_row_stamped_after_now_is_rejected(self) -> None:
        book, t = LiveContextBook(), B + MIN
        before = book.snapshot()
        future = Observation(B, t + 1, Decimal("50.00000000"), "recorded")
        assert _code(lambda: book.publish(DOM, [future], t)) == "pit_bad_value"
        assert book.snapshot() is before and book._high_water_ms == 0
        assert book.publish(DOM, [Observation(B, t, Decimal("50.00000000"), "recorded")], t) == 1

    def test_v9_freshness_and_non_regressed_calls_never_drop(self) -> None:
        book, f1 = LiveContextBook(), B + 40 * MIN
        book.publish(DOM, [snapshot_observation(DOM, 50, f1)], f1)
        snap = book.snapshot()
        assert book.freshness(f1 - TOL - 1).regressed is True
        assert book.snapshot() is snap and book._high_water_ms == f1 and book.regressed_reads == 0
        assert book.heartbeat(f1 - TOL) is False and book.snapshot() is snap
        assert len(book.view(f1 - TOL)) == 1 and book.snapshot() is snap


# ---------------------------------------------------------------------------
# ConsumerRegistry (SY-01-21j) and T-14b isolation
# ---------------------------------------------------------------------------
_EQ = type("Eq", (), {"__eq__": lambda s, o: True})()  # equal to everything, even "paper"


class TestConsumerRegistry:
    def test_counts_per_mode_with_an_idempotent_stop(self) -> None:
        reg = ConsumerRegistry()
        assert dict(reg.counts()) == {"paper": 0, "live": 0}
        a, b, c = reg.started("paper"), reg.started("paper"), reg.started("live")
        assert dict(reg.counts()) == {"paper": 2, "live": 1} and len({a, b, c}) == 3
        reg.stopped(a)
        reg.stopped(a)  # idempotent
        reg.stopped(9999)  # unknown token: no-op
        reg.stopped("x")  # type: ignore[arg-type]
        reg.stopped([1])  # type: ignore[arg-type]  # unhashable: still a no-op
        assert dict(reg.counts()) == {"paper": 1, "live": 1}
        reg.reset()
        assert dict(reg.counts()) == {"paper": 0, "live": 0}

    def test_counts_is_a_read_only_mapping(self) -> None:
        reg = ConsumerRegistry()
        counts = reg.counts()
        assert isinstance(counts, MappingProxyType)
        with pytest.raises(TypeError):
            counts["paper"] = 5  # type: ignore[index]
        assert dict(reg.counts()) == {"paper": 0, "live": 0}

    @pytest.mark.parametrize("mode", ["backtest", "", "LIVE", None, [1], ("paper",), b"paper", _EQ])
    def test_unknown_and_non_str_modes_are_refused(self, mode: object) -> None:
        with pytest.raises(ValueError, match="unknown consumer mode"):
            ConsumerRegistry().started(mode)  # type: ignore[arg-type]


class TestT14bIsolation:
    """Every instance asserts a clean start and then leaves dirt: any order must stay green."""

    @pytest.mark.parametrize("n", range(3))
    def test_the_autouse_fixture_resets_both_singletons_around_every_test(self, n: int) -> None:
        assert len(LIVE_BOOK.snapshot()) == 0
        assert (LIVE_BOOK._high_water_ms, LIVE_BOOK.regressed_reads) == (0, 0)
        assert dict(LIVE_CONSUMERS.counts()) == {"paper": 0, "live": 0}
        LIVE_BOOK.view(10**15)  # a future high-water mark
        LIVE_BOOK.view(1)  # a regressed read
        LIVE_BOOK.publish(FGI, fgi_observations([(D0, 40)], D0 + H), D0 + H)
        LIVE_CONSUMERS.started("live")
        assert LIVE_BOOK.regressed_reads == 1 and len(LIVE_BOOK.snapshot()) == 1

    def test_reset_clears_series_mark_and_counter(self) -> None:
        book = LiveContextBook()
        book.view(10**15)
        book.view(1)
        book.reset()
        assert (len(book.snapshot()), book._high_water_ms, book.regressed_reads) == (0, 0, 0)
