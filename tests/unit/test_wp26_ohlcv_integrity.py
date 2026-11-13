"""WP2.6 / P2: unit tests and AST pin of the OHLCV integrity leaf ``data/ohlcv_integrity.py``.

Binding spec: ``reports/vp2-wp26/spec.md`` (SY-26-02/03/04/08/12). The golden sha256 values were
recomputed independently of the leaf. The AST pin has the same style as
``tests/unit/test_wp24_ast_pins.py`` (import whitelist, banned names, mutated-source self-test).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import pickle
import sys
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

import data.ohlcv_integrity as leaf
from data.ohlcv_integrity import (
    OHLCV_CODES,
    Candle,
    OhlcvDataError,
    SeriesSummary,
    canon_decimal,
    canon_text,
    canonicalise_candle,
    ceil_tf,
    check_closed,
    check_identifiers,
    closed_end_ms,
    floor_tf,
    is_on_grid,
    is_storable,
    run_digest,
    series_digest,
    settle_end_ms,
    storable_end_ms,
    summarise_series,
    tf_ms,
)

_REPO = Path(__file__).resolve().parents[2]
_LEAF = _REPO / "packages" / "data" / "ohlcv_integrity.py"
H = 3_600_000  # one 1h bar in ms

# spec section 3.4 (SY-26-04): inputs and expected digests
BTC_ROWS = [
    (1704067200000, "42000.5", "42100", "41950.25", "42050.0", "12.345678901234567"),
    (1704070800000, "42050", "42075.10", "42000", "42010.000000000001", "-0"),
]
ETH_ROWS = [(1704067200000, "2100", "2110.5", "2095", "2105.25", "3.5")]
BTC_PREIMAGE = (
    "ohlcv-wp26-1|series|coinbase|BTC/EUR|3600\n"
    "1704067200000|42000.5|42100|41950.25|42050|12.345678901235\n"
    "1704070800000|42050|42075.1|42000|42010.000000000001|0\n"
)
BTC_DIGEST = "4f68d163ac60684ce1b739031fc0780a199cf58b730069b7465d60699a56bfe5"
ETH_DIGEST = "0cbceecc7bbb72496d83b3b8d74e94a189d484eaca75e191a580d03125f3550c"
SOL_DIGEST = "772a41a07fbeb56abf6e82a694ccaa692010fef666c03911aae2ac23809c20c4"
DATA_HASH = "607ff058428ac5491d8de7d43ad6970c933bb6cf5d2e4b2865faf92440a52a3f"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _candles(rows: list[tuple[Any, ...]], tf: int = 3600) -> tuple[list[Candle], int]:
    built = [canonicalise_candle(*r, timeframe_s=tf) for r in rows]
    return [c for c, _ in built], sum(n for _, n in built)


def _golden_summaries() -> list[SeriesSummary]:
    btc, _ = _candles(BTC_ROWS)
    eth, _ = _candles(ETH_ROWS)
    return [
        summarise_series("coinbase", "BTC/EUR", 3600, btc),
        summarise_series("coinbase", "ETH/EUR", 3600, eth),
    ]


def _code(excinfo: pytest.ExceptionInfo[OhlcvDataError]) -> str:
    return excinfo.value.code


# ===========================================================================
# canonical text (SY-26-02 / SY-26-03)
# ===========================================================================
CANON_TABLE = [
    ("1E-7", "0.0000001"),
    ("1.10", "1.1"),
    ("100", "100"),
    ("1E+2", "100"),
    ("0", "0"),
    ("-0", "0"),
    ("0E-12", "0"),
    ("0.0000000000005", "0.000000000001"),  # half-up (half-even would give 0)
    ("-0.0000000000005", "-0.000000000001"),
    ("0.00000000000049", "0"),
    ("-0.0000000000004", "0"),  # rounds to -0, printed 0
    ("12.345678901234567", "12.345678901235"),
    ("42010.000000000001", "42010.000000000001"),
    ("123456789012345678.123456789012", "123456789012345678.123456789012"),
]


class TestCanonicalText:
    @pytest.mark.parametrize(("raw", "expected"), CANON_TABLE)
    def test_canon_table(self, raw: str, expected: str) -> None:
        assert canon_text(raw) == expected
        assert canon_text(Decimal(raw)) == expected
        assert canon_decimal(raw) == Decimal(expected)
        assert format(canon_decimal(raw), "f") == expected  # text == format(canonical Decimal)

    def test_int_input_and_idempotence(self) -> None:
        assert canon_text(42) == "42"
        for _raw, expected in CANON_TABLE:
            assert canon_text(expected) == expected

    def test_normalize_trap_is_real_and_avoided(self) -> None:
        trap = Decimal("123456789012345678.123456789012")
        assert str(trap.normalize()) == "123456789012345678.123456789"  # default 28-digit context
        assert canon_text(trap) == "123456789012345678.123456789012"

    def test_canonical_decimal_exponent_is_stripped(self) -> None:
        assert str(canon_decimal("1.500000000000")) == "1.5"
        assert str(canon_decimal("1E+2")) == "100"

    @pytest.mark.parametrize("bad", [1.5, True, None, "abc", "", b"1", [1], (1,), object()])
    def test_bad_type(self, bad: Any) -> None:
        with pytest.raises(OhlcvDataError) as e:
            canon_text(bad)
        assert _code(e) == "bad_type"

    @pytest.mark.parametrize(
        "bad", ["NaN", "sNaN", "Infinity", "-Infinity", Decimal("NaN"), Decimal("-Infinity")]
    )
    def test_not_finite(self, bad: Any) -> None:
        with pytest.raises(OhlcvDataError) as e:
            canon_text(bad)
        assert _code(e) == "not_finite"

    @pytest.mark.parametrize("big", ["1E18", "-1E18", "1E+500", "999999999999999999.9999999999995"])
    def test_canon_text_is_bounded(self, big: str) -> None:
        with pytest.raises(OhlcvDataError) as e:
            canon_text(big)
        assert _code(e) == "out_of_range"


# ===========================================================================
# grid arithmetic and the settle boundary (SY-26-08)
# ===========================================================================
class TestGrid:
    @pytest.mark.parametrize("tf", leaf.CACHEABLE_TIMEFRAMES_S)
    def test_every_cacheable_timeframe_divides_a_day(self, tf: int) -> None:
        assert 86_400 % tf == 0
        assert tf_ms(tf) == tf * 1000
        assert is_on_grid(0, tf) and is_on_grid(86_400_000, tf)
        assert not is_on_grid(1, tf) and not is_on_grid(tf * 1000 - 1, tf)

    def test_floor_and_ceil(self) -> None:
        assert floor_tf(H, 3600) == H and floor_tf(H + 1, 3600) == H
        assert floor_tf(H - 1, 3600) == 0 and floor_tf(0, 3600) == 0
        assert ceil_tf(H, 3600) == H and ceil_tf(H + 1, 3600) == 2 * H
        assert ceil_tf(0, 3600) == 0 and ceil_tf(1, 3600) == H
        assert floor_tf(-1, 3600) == -H  # floor, not truncation
        assert floor_tf(1_700_000_123_456, 180) % 180_000 == 0

    def test_closed_end_settle_end_and_storable_end(self) -> None:
        now = 10 * H + 123_456  # F = 10H
        assert closed_end_ms(100 * H, now, 3600) == 10 * H
        assert closed_end_ms(5 * H, now, 3600) == 5 * H
        assert settle_end_ms(now, 3600) == 9 * H  # F - SETTLE_BARS * tf
        assert storable_end_ms(100 * H, now, 3600) == 9 * H
        assert storable_end_ms(5 * H, now, 3600) == 5 * H
        assert settle_end_ms(now, 3600) == floor_tf(now, 3600) - leaf.SETTLE_BARS * H

    def test_is_storable_boundary(self) -> None:
        now = 10 * H + 123_456
        assert is_storable(8 * H, now, 3600)  # open + tf == F - tf
        assert not is_storable(9 * H, now, 3600)  # the last closed bar is served, not stored
        assert not is_storable(10 * H, now, 3600)  # forming

    def test_check_closed_boundary(self) -> None:
        now = 10 * H + 1
        check_closed(9 * H, now, 3600)  # open + tf == F: closed
        for forming in (10 * H, 11 * H):
            with pytest.raises(OhlcvDataError) as e:
                check_closed(forming, now, 3600)
            assert _code(e) == "forming_or_future"

    @pytest.mark.parametrize("tf", [0, 61, 604_800, -3600, True, "3600", None, 3600.0])
    def test_timeframe_whitelist(self, tf: Any) -> None:
        with pytest.raises(OhlcvDataError) as e:
            tf_ms(tf)
        assert _code(e) == "bad_identifier"

    @pytest.mark.parametrize("ms", [True, 1.0, "1", None])
    def test_ms_must_be_an_int(self, ms: Any) -> None:
        with pytest.raises(OhlcvDataError) as e:
            floor_tf(ms, 3600)
        assert _code(e) == "bad_type"

    def test_whitelist_is_the_engine_map_minus_one_week(self) -> None:
        from trading.strategy_engine import _TIMEFRAME_SECONDS

        assert set(leaf.CACHEABLE_TIMEFRAMES_S) == set(_TIMEFRAME_SECONDS.values()) - {604_800}
        assert list(leaf.CACHEABLE_TIMEFRAMES_S) == sorted(leaf.CACHEABLE_TIMEFRAMES_S)


# ===========================================================================
# identifiers (SY-26-04): the vectors the DB CHECK parity test (P3/P4) reuses
# ===========================================================================
EXCHANGE_ACCEPT = ["coinbase", "binance", "a_1", "a", "x" * 32, "0"]
EXCHANGE_REJECT = [
    "",
    "x" * 33,
    "_\n",
    "Coinbase",
    "coin-base",
    "coin base",
    "coinbase\n",
    "cöin",
    "a|b",
]
SYMBOL_ACCEPT = [
    "BTC/EUR",
    "BTC/EURC",
    "ETH-PERP/USD",
    "BTC/USD:USD",
    "A/B",
    "X" * 15 + "/" + "Y" * 16,
    "1INCH/USDT",
    "BTC/USD:X",
]
SYMBOL_REJECT = [
    "", "/", "/AB", "AB/", "A//B", "A/B/C", "BTC|X/EUR", "A" * 29 + "/EUR", "AB", "BTC",
    "BTCEUR", "BTC//EUR", "BTC/E/UR", "BTC|/EUR", "BTC /EUR", "BTC/EUR\n", "BTC/EÜR",
    "A/" + "B" * 31, "BTC/EUR\x7f", "BTC/EUR\t",
]  # fmt: skip


class TestIdentifiers:
    @pytest.mark.parametrize("exchange", EXCHANGE_ACCEPT)
    def test_exchange_accept(self, exchange: str) -> None:
        check_identifiers(exchange, "BTC/EUR", 3600)

    @pytest.mark.parametrize("symbol", SYMBOL_ACCEPT)
    def test_symbol_accept(self, symbol: str) -> None:
        check_identifiers("coinbase", symbol, 3600)

    @pytest.mark.parametrize(
        ("exchange", "symbol"),
        [(e, "BTC/EUR") for e in EXCHANGE_REJECT]
        + [("coinbase", s) for s in SYMBOL_REJECT]
        + [(None, "BTC/EUR"), ("coinbase", None), (5, "BTC/EUR"), ("coinbase", b"BTC/EUR")],
    )
    def test_reject(self, exchange: Any, symbol: Any) -> None:
        with pytest.raises(OhlcvDataError) as e:
            check_identifiers(exchange, symbol, 3600)
        assert _code(e) == "bad_identifier"

    def test_digest_functions_validate_identifiers(self) -> None:
        with pytest.raises(OhlcvDataError):
            series_digest("Coinbase", "BTC/EUR", 3600, [])
        with pytest.raises(OhlcvDataError):
            series_digest("coinbase", "BTC/EUR", 604_800, [])
        with pytest.raises(OhlcvDataError):
            run_digest([SeriesSummary("coinbase", "BTC|/EUR", 3600, 0, SOL_DIGEST)])


# ===========================================================================
# candle validation (SY-26-02, DE-26-12)
# ===========================================================================
def _row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "open_ms": H, "open_": "100", "high": "110", "low": "90", "close": "105", "volume": "1",
    }  # fmt: skip
    row.update(over)
    return row


def _canon(**over: Any) -> tuple[Candle, int]:
    row = _row(**over)
    return canonicalise_candle(
        row.pop("open_ms"), row.pop("open_"), row.pop("high"), row.pop("low"), row.pop("close"),
        row.pop("volume"), timeframe_s=row.pop("tf", 3600),
    )  # fmt: skip


class TestCandle:
    def test_valid_candle_is_canonical(self) -> None:
        candle, rounded = _canon(open_="99.10", high="1E+2", low="0.5", close="99.9000")
        assert candle.open_ms == H and rounded == 0
        assert (candle.open_, candle.high, candle.low) == (Decimal("99.1"), 100, Decimal("0.5"))
        assert str(candle.high) == "100"

    def test_flat_candle_and_zero_volume_are_valid(self) -> None:
        candle, _ = _canon(open_="5", high="5", low="5", close="5", volume="0")
        assert candle.volume == 0

    @pytest.mark.parametrize("ms", [H + 1, 1, -H, 0 - 1, 2 * H - 1])
    def test_off_grid(self, ms: int) -> None:
        with pytest.raises(OhlcvDataError) as e:
            _canon(open_ms=ms)
        assert _code(e) == "off_grid"

    def test_grid_follows_the_timeframe(self) -> None:
        assert _canon(open_ms=180_000, tf=180)[0].open_ms == 180_000
        assert _canon(open_ms=0)[0].open_ms == 0
        with pytest.raises(OhlcvDataError):
            _canon(open_ms=180_000, tf=300)

    @pytest.mark.parametrize(
        "over",
        [
            {"open_": "80"},  # open below low
            {"open_": "120"},  # open above high
            {"close": "89"},
            {"close": "111"},
            {"low": "0"},
            {"low": "-1", "open_": "100"},
            {"low": "0.0000000000004", "open_": "100"},  # quantises to 0
            {"high": "89", "low": "90"},  # low above high
        ],
    )
    def test_bad_ohlc(self, over: dict[str, Any]) -> None:
        with pytest.raises(OhlcvDataError) as e:
            _canon(**over)
        assert _code(e) == "bad_ohlc"

    def test_bad_volume(self) -> None:
        with pytest.raises(OhlcvDataError) as e:
            _canon(volume="-1")
        assert _code(e) == "bad_volume"
        assert _canon(volume="-0.0000000000004")[0].volume == 0  # rounds to zero: valid

    @pytest.mark.parametrize(
        "over",
        [
            {"high": "1E16"},
            {"high": "10000000000000000"},
            {"high": "9999999999999999.9999999999995"},  # rounds up to the limit
            {"high": "1E+500"},
            {"volume": "1E18"},
            {"volume": "999999999999999999.9999999999995"},
            {"volume": "-1E18"},
        ],
    )
    def test_out_of_range(self, over: dict[str, Any]) -> None:
        with pytest.raises(OhlcvDataError) as e:
            _canon(**over)
        assert _code(e) == "out_of_range"

    def test_range_limits_are_inclusive_of_the_largest_scale_12_value(self) -> None:
        assert _canon(high="9999999999999999.999999999999")[0].high > 9 * 10**15
        assert _canon(volume="999999999999999999.999999999999")[0].volume > 9 * 10**17

    @pytest.mark.parametrize(
        ("over", "code"),
        [
            ({"high": "NaN"}, "not_finite"),
            ({"volume": "Infinity"}, "not_finite"),
            ({"low": 90.5}, "bad_type"),
            ({"close": True}, "bad_type"),
            ({"open_ms": 1.5}, "bad_type"),
            ({"open_ms": True}, "bad_type"),
            ({"volume": None}, "bad_type"),
            ({"tf": 604_800}, "bad_identifier"),
        ],
    )
    def test_type_and_finiteness(self, over: dict[str, Any], code: str) -> None:
        with pytest.raises(OhlcvDataError) as e:
            _canon(**over)
        assert _code(e) == code

    def test_rounded_values_counts_changed_values_only(self) -> None:
        assert _canon(open_="99.10", high="1E+2", low="90.000", close="99", volume="1E+0")[1] == 0
        three = _canon(open_="100.0000000000005", low="89.9999999999994", volume="0.0000000000005")
        assert three[1] == 3
        assert three[0].open_ == Decimal("100.000000000001")
        assert three[0].low == Decimal("89.999999999999")
        assert three[0].volume == Decimal("0.000000000001")


# ===========================================================================
# digests and the data_hash (SY-26-04)
# ===========================================================================
class TestGoldenVectors:
    def test_constants(self) -> None:
        assert leaf.OHLCV_DATA_VERSION == "wp26-1"
        assert leaf.SETTLE_BARS == 1 and leaf.SCALE_DECIMALS == 12

    def test_btc_preimage_and_digest(self) -> None:
        assert _sha(BTC_PREIMAGE) == BTC_DIGEST
        candles, rounded = _candles(BTC_ROWS)
        assert rounded == 1  # only the 17-digit volume is rounded
        assert series_digest("coinbase", "BTC/EUR", 3600, candles) == BTC_DIGEST

    def test_eth_and_empty_series(self) -> None:
        eth, rounded = _candles(ETH_ROWS)
        assert rounded == 0
        assert series_digest("coinbase", "ETH/EUR", 3600, eth) == ETH_DIGEST
        assert series_digest("coinbase", "SOL/EUR", 3600, []) == SOL_DIGEST
        assert _sha("ohlcv-wp26-1|series|coinbase|SOL/EUR|3600\n") == SOL_DIGEST  # header only

    def test_run_preimage_and_data_hash(self) -> None:
        preimage = (
            "ohlcv-wp26-1|run\n"
            f"coinbase|BTC/EUR|3600|2|{BTC_DIGEST}\n"
            f"coinbase|ETH/EUR|3600|1|{ETH_DIGEST}\n"
        )
        assert _sha(preimage) == DATA_HASH
        assert run_digest(_golden_summaries()) == DATA_HASH

    def test_summary_fields(self) -> None:
        btc, eth = _golden_summaries()
        assert (btc.n, btc.digest, eth.n, eth.digest) == (2, BTC_DIGEST, 1, ETH_DIGEST)

    def test_empty_run_hashes_the_header_only(self) -> None:
        assert run_digest([]) == _sha("ohlcv-wp26-1|run\n")


class TestDigestProperties:
    def test_run_digest_is_independent_of_input_order(self) -> None:
        sol = SeriesSummary("coinbase", "SOL/EUR", 3600, 0, SOL_DIGEST)
        both = [*_golden_summaries(), sol]
        hashes = {run_digest(list(p)) for p in permutations(both)}
        assert len(hashes) == 1
        assert run_digest(iter(_golden_summaries()[::-1])) == DATA_HASH  # any iterable

    def test_run_sort_is_by_tuple_not_by_line_bytes(self) -> None:
        d = SOL_DIGEST
        rows = [
            SeriesSummary("coinbase", "BTC/EURC", 3600, 0, d),
            SeriesSummary("coinbase", "BTC/EUR", 3600, 0, d),
            SeriesSummary("coinbase", "BTC/EUR", 900, 0, d),  # int order, not text order
            SeriesSummary("binance", "ZZZ/EUR", 60, 0, d),
        ]
        lines = [f"{s.exchange}|{s.symbol}|{s.timeframe_s}|0|{d}\n" for s in rows]
        by_bytes = sorted(lines)  # the trap: EURC before EUR, and 3600 before 900
        assert by_bytes.index(lines[0]) < by_bytes.index(lines[1])
        assert by_bytes.index(lines[1]) < by_bytes.index(lines[2])
        expected = _sha("ohlcv-wp26-1|run\n" + "".join([lines[3], lines[2], lines[1], lines[0]]))
        assert run_digest(rows) == expected
        assert run_digest(rows[::-1]) == expected

    def test_domain_separation(self) -> None:
        assert DATA_HASH not in {BTC_DIGEST, ETH_DIGEST, SOL_DIGEST}
        btc = _golden_summaries()[0]
        assert run_digest([btc]) != btc.digest  # a one-series run is not its series digest
        assert run_digest([]) != series_digest("coinbase", "SOL/EUR", 3600, [])
        assert BTC_PREIMAGE.startswith("ohlcv-wp26-1|series|")

    def test_no_range_in_the_preimage(self) -> None:
        for fn, params in (
            (series_digest, ["exchange", "symbol", "timeframe_s", "candles"]),
            (run_digest, ["series"]),
            (summarise_series, ["exchange", "symbol", "timeframe_s", "candles"]),
        ):
            assert list(inspect.signature(fn).parameters) == params
        assert list(SeriesSummary.__dataclass_fields__) == [
            "exchange", "symbol", "timeframe_s", "n", "digest",
        ]  # fmt: skip

    def test_series_digest_depends_on_every_field(self) -> None:
        base = [Candle(H, Decimal(1), Decimal(2), Decimal(1), Decimal(2), Decimal(3))]
        reference = series_digest("coinbase", "BTC/EUR", 3600, base)
        changed = [
            ("binance", "BTC/EUR", 3600, base),
            ("coinbase", "ETH/EUR", 3600, base),
            ("coinbase", "BTC/EUR", 900, base),
            ("coinbase", "BTC/EUR", 3600, [Candle(2 * H, *(getattr(base[0], f) for f in _F))]),
            (
                "coinbase",
                "BTC/EUR",
                3600,
                [Candle(H, Decimal(2), *(getattr(base[0], f) for f in _F[1:]))],
            ),
            ("coinbase", "BTC/EUR", 3600, []),
        ]
        digests = {series_digest(*args) for args in changed}
        assert reference not in digests and len(digests) == len(changed)

    def test_non_canonical_candles_are_rejected(self) -> None:
        # QR-26P2-01: equal digest must mean identical engine inputs, so a loose exponent, an
        # unrounded value, a non-Decimal or an invalid bar is refused, never re-canonicalised.
        candles, _ = _candles(BTC_ROWS)
        a = candles[0]
        loose = Candle(a.open_ms, *(getattr(a, f).quantize(Decimal("1E-14")) for f in _F))
        raw = Candle(a.open_ms, a.open_, a.high, a.low, a.close, Decimal("12.345678901234567"))
        sci = Candle(a.open_ms, a.open_, a.high, a.low, a.close, Decimal("1E+1"))
        text = Candle(a.open_ms, "42000.5", a.high, a.low, a.close, a.volume)  # type: ignore[arg-type]
        for bad in (loose, raw, sci, text):
            with pytest.raises(OhlcvDataError) as e:
                series_digest("coinbase", "BTC/EUR", 3600, [bad])
            assert _code(e) == "not_canonical"
        big = Decimal("10000000000000000")  # canonical text, but >= the NUMERIC(28,12) price limit
        for bad, code in (
            (Candle(a.open_ms, a.open_, a.low, a.high, a.close, a.volume), "bad_ohlc"),
            (Candle(a.open_ms, big, big, big, big, a.volume), "out_of_range"),
            (Candle(a.open_ms, a.open_, a.high, a.low, a.close, Decimal(-1)), "bad_volume"),
        ):
            with pytest.raises(OhlcvDataError) as e:
                series_digest("coinbase", "BTC/EUR", 3600, [bad])
            assert _code(e) == code
        assert series_digest("coinbase", "BTC/EUR", 3600, candles) == BTC_DIGEST

    def test_series_must_be_strictly_increasing_and_on_grid(self) -> None:
        a, b = _candles(BTC_ROWS)[0]
        for bad in ([b, a], [a, a]):
            with pytest.raises(OhlcvDataError) as e:
                series_digest("coinbase", "BTC/EUR", 3600, bad)
            assert _code(e) == "not_increasing"
        off = Candle(H + 1, *(getattr(a, f) for f in _F))
        with pytest.raises(OhlcvDataError) as e:
            series_digest("coinbase", "BTC/EUR", 3600, [off])
        assert _code(e) == "off_grid"
        boolean = Candle(True, *(getattr(a, f) for f in _F))  # type: ignore[arg-type]
        with pytest.raises(OhlcvDataError):
            series_digest("coinbase", "BTC/EUR", 3600, [boolean])
        negative = Candle(-H, *(getattr(a, f) for f in _F))  # on the grid, before the epoch
        with pytest.raises(OhlcvDataError) as e:
            series_digest("coinbase", "BTC/EUR", 3600, [negative])
        assert _code(e) == "off_grid"
        zero, _ = canonicalise_candle(0, "1", "1", "1", "1", "0", timeframe_s=3600)
        header = "ohlcv-wp26-1|series|coinbase|BTC/EUR|3600\n"
        assert series_digest("coinbase", "BTC/EUR", 3600, [zero]) == _sha(header + "0|1|1|1|1|0\n")

    def test_duplicate_series_and_bad_summaries(self) -> None:
        one = SeriesSummary("coinbase", "BTC/EUR", 3600, 0, SOL_DIGEST)
        with pytest.raises(OhlcvDataError) as e:
            run_digest([one, one])
        assert _code(e) == "duplicate_series"
        run_digest([one, SeriesSummary("coinbase", "BTC/EUR", 900, 0, SOL_DIGEST)])  # not a dup
        for bad in (
            SeriesSummary("coinbase", "BTC/EUR", 3600, -1, SOL_DIGEST),
            SeriesSummary("coinbase", "BTC/EUR", 3600, True, SOL_DIGEST),  # type: ignore[arg-type]
            SeriesSummary("coinbase", "BTC/EUR", 3600, 0, SOL_DIGEST.upper()),
            SeriesSummary("coinbase", "BTC/EUR", 3600, 0, SOL_DIGEST[:-1]),
            SeriesSummary("coinbase", "BTC/EUR", 3600, 0, SOL_DIGEST[:-1] + "g"),
            SeriesSummary("coinbase", "BTC/EUR", 3600, 0, None),  # type: ignore[arg-type]
        ):
            with pytest.raises(OhlcvDataError) as e:
                run_digest([bad])
            assert _code(e) == "bad_digest"


_F = ("open_", "high", "low", "close", "volume")


class TestError:
    def test_codes_are_closed_and_messages_fixed(self) -> None:
        assert len(OHLCV_CODES) == len(set(OHLCV_CODES)) == 12
        assert {
            "off_grid", "forming_or_future", "bad_ohlc", "bad_volume", "out_of_range",
            "not_finite", "bad_identifier", "duplicate_series",
        } <= set(OHLCV_CODES)  # fmt: skip
        for code in OHLCV_CODES:
            err = OhlcvDataError(code)
            assert isinstance(err, ValueError) and err.code == code
            assert str(err) == f"ohlcv data check failed ({code})"
            clone = pickle.loads(pickle.dumps(err))  # noqa: S301
            assert type(clone) is OhlcvDataError and clone.code == code
        with pytest.raises(ValueError, match="unknown ohlcv code"):
            OhlcvDataError("anything")

    def test_message_never_echoes_input(self) -> None:
        with pytest.raises(OhlcvDataError) as e:
            check_identifiers("coinbase", "SECRET|x/Y", 3600)
        assert "SECRET" not in str(e.value) and "SECRET" not in repr(e.value)


# ===========================================================================
# AST pin of the leaf (same style as test_wp24_ast_pins.py)
# ===========================================================================
_ALLOWED_IMPORTS = frozenset(
    {"__future__", "collections.abc", "dataclasses", "decimal", "hashlib", "re", "typing"}
)
_BANNED_NAMES = frozenset(
    {
        "environ", "getenv", "putenv", "now", "utcnow", "today", "time", "monotonic",
        "perf_counter", "open", "print", "input", "metrics", "increment", "getLogger",
        "logging", "structlog", "random", "uuid4", "__import__", "import_module", "eval",
        "exec", "read_text", "write_text", "urlopen", "os", "sleep", "math", "float",
        "logger", "warnings", "normalize", "datetime", "timedelta", "sorted_bytes",
        "from_float", "getcontext", "setcontext", "localcontext",
    }
)  # fmt: skip
_FORBIDDEN_PREFIXES = ("trading", "common", "data", "apps")


def leaf_problems(src: str) -> list[str]:
    """Purity checker for the leaf source; returns ``"<category>: <detail>"`` problems."""
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"import: {a.name}" for a in node.names if a.name not in _ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or (node.module or "") not in _ALLOWED_IMPORTS:
                out.append(f"import: {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Constant) and type(node.value) is float:
            out.append("float: literal")
        elif isinstance(node, ast.Div):
            out.append("float: operator /")
        elif isinstance(node, ast.Pow):
            out.append("float: operator **")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            out.append("state: global/nonlocal")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    out += [f"banned: {name}" for name in sorted(used & _BANNED_NAMES)]
    return out


class TestLeafAstPin:
    def test_leaf_is_clean(self) -> None:
        assert leaf_problems(_LEAF.read_text()) == []

    def test_imports_are_an_exact_whitelist(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert {"decimal", "hashlib", "dataclasses", "typing"} <= imported <= _ALLOWED_IMPORTS
        assert all(m.split(".")[0] in sys.stdlib_module_names for m in _ALLOWED_IMPORTS)
        assert not [m for m in imported if m and m.split(".")[0] in _FORBIDDEN_PREFIXES]

    def test_not_reexported_from_the_data_package(self) -> None:
        init = (_REPO / "packages" / "data" / "__init__.py").read_text()
        assert "ohlcv_integrity" not in init

    def test_all_exports_exist(self) -> None:
        tree = ast.parse(_LEAF.read_text())
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)} | {
            n.target.id for n in tree.body if isinstance(n, ast.AnnAssign)
            if isinstance(n.target, ast.Name)
        }  # fmt: skip
        exported: list[str] = []
        for n in tree.body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__":
                assert isinstance(n.value, ast.List)
                exported = [ast.literal_eval(e) for e in n.value.elts]
        assert exported and set(exported) <= defined
        assert exported == sorted(
            exported, key=lambda s: (0 if s.isupper() else 1 if s[0].isupper() else 2, s)
        )
        assert {
            "OHLCV_DATA_VERSION", "SETTLE_BARS", "OhlcvDataError", "canon_text", "series_digest",
            "run_digest", "canonicalise_candle", "storable_end_ms", "check_identifiers",
        } <= set(exported)  # fmt: skip

    def test_hash_path_uses_the_private_context_and_never_normalize(self) -> None:
        src = _LEAF.read_text()
        assert "ROUND_HALF_UP" in src and "context=_CTX" in src

    @pytest.mark.parametrize(
        ("addition", "expected"),
        [
            ("import pathlib", "import: pathlib"),
            ("from random import shuffle", "import: random"),
            ("import urllib.request", "import: urllib.request"),
            ("from os import environ", "import: os"),
            ("import logging", "import: logging"),
            ("import structlog", "import: structlog"),
            ("from time import time", "import: time"),
            ("import math", "import: math"),
            ("import json", "import: json"),
            ("from datetime import datetime", "import: datetime"),
            ("from trading import metrics", "import: trading"),
            ("from common import types", "import: common"),
            ("from data import bar_alignment", "import: data"),
            ("from data.point_in_time import to_ms", "import: data.point_in_time"),
            ("from apps.api import main", "import: apps.api"),
            ("from . import sibling", "import: ."),
            ("_x = datetime.now()", "banned: now"),
            ("_t = time.time()", "banned: time"),
            ("_y = open('f')", "banned: open"),
            ("_p = print", "banned: print"),
            ("_e = os.environ", "banned: environ"),
            ("_l = logging.getLogger('x')", "banned: getLogger"),
            ("_n = _QUANTUM.normalize()", "banned: normalize"),
            ("_z = 0.5", "float: literal"),
            ("_w = 2 ** 3", "float: operator **"),
            ("_d = 6 / 3", "float: operator /"),
            ("_v = float(3)", "banned: float"),
            ("_f = Decimal.from_float(3)", "banned: from_float"),
            ("_c = getcontext()", "banned: getcontext"),
            ("def _g() -> None:\n    global _G\n    _G = 1", "state: global/nonlocal"),
        ],
    )
    def test_pin_fails_on_mutated_source(self, addition: str, expected: str) -> None:
        src = _LEAF.read_text()
        assert leaf_problems(src) == []  # the real leaf is clean ...
        problems = leaf_problems(src + f"\n{addition}\n")  # ... and the pin can fail
        assert expected in problems, problems
