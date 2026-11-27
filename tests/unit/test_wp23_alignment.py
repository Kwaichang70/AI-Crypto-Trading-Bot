"""WP2.3 / P1: pure bar-alignment leaf ``packages/data/bar_alignment.py``.

Binding spec: ``reports/vp2-wp2.3/synthesis-spec.md`` (SY-23-01..11, WP23-T-03/04/06..09).
Test ids are ``WP23-P1-NN`` (see the traceability table in the P1 report).

Expected values come from the quant design (QD T-03 / T-06 / T-07 / T-08). The
TOLERANCE DEFAULTS class pins U-01 option (a); if the user picks another option the
constants block in the leaf and ``TestToleranceDefaults`` are the only things to edit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from data import bar_alignment as ba
from data.bar_alignment import (
    BAR_ALIGNMENT_CODES,
    BAR_ALIGNMENT_VERSION,
    DEFAULT_ALIGNMENT_TOLERANCE,
    AlignmentTolerance,
    BarAlignmentError,
    align_bars,
    build_timeline,
    check_alignment,
    fold_check,
    iso_from_ms,
)

_REPO = Path(__file__).resolve().parents[2]
_LEAF = _REPO / "packages" / "data" / "bar_alignment.py"
T0 = datetime(2026, 3, 2, tzinfo=UTC)
H = 3600
D = 86_400
LENIENT = AlignmentTolerance(
    max_missing_ppm=1_000_000,
    warn_missing_ppm=1_000_000,
    max_gap_bars=10_000,
    warn_gap_bars=10_000,
    max_head_gap_bars=10_000,
    max_tail_gap_bars=10_000,
)


@dataclass(frozen=True, slots=True)
class Bar:
    """Duck-typed stand-in (the leaf only reads ``.timestamp`` and ``.volume``)."""

    symbol: str
    timestamp: datetime
    volume: Decimal = Decimal(1)


def ts_of(k: int, tf: int = H, t0: datetime = T0) -> datetime:
    return t0 + timedelta(seconds=k * tf)


def mk(
    sym: str, ks: object, tf: int = H, t0: datetime = T0, zero: frozenset[int] = frozenset()
) -> list[Bar]:
    return [
        Bar(sym, ts_of(k, tf, t0), Decimal(0) if k in zero else Decimal(1))
        for k in ks  # type: ignore[attr-defined]
    ]


def rng_without(n: int, missing: set[int]) -> list[int]:
    return [k for k in range(n) if k not in missing]


def ms_of(dt: datetime) -> int:
    return (dt - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)


def slot_spans(
    runs: tuple[ba.GapRun, ...], kind: str, tf: int = H, t0: datetime = T0
) -> list[tuple[int, int]]:
    return [((g.start_ms - ms_of(t0)) // (tf * 1000), g.bars) for g in runs if g.kind == kind]


def qd_fixture() -> dict[str, list[Bar]]:
    """QD T-03: 12 x 1h; A complete, B missing {4,5,6}, C missing {9}."""
    return {
        "A": mk("A", range(12)),
        "B": mk("B", rng_without(12, {4, 5, 6})),
        "C": mk("C", rng_without(12, {9})),
    }


# ---------------------------------------------------------------------------
# WP23-P1-01  T-03 leaf AST pin
# ---------------------------------------------------------------------------


# Exact import whitelist of the leaf (CR-01): any new import forces an edit here.
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "bisect",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "itertools",
        "json",
        "typing",
    }
)
_BANNED_NAMES = frozenset(
    {
        "environ", "getenv", "putenv", "now", "utcnow", "today", "time", "monotonic",
        "perf_counter", "open", "print", "input", "metrics", "increment", "getLogger",
        "random", "uuid4", "__import__", "import_module", "eval", "exec", "read_text",
        "write_text", "urlopen",
    }
)  # fmt: skip


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
        elif isinstance(node, ast.Name) and node.id == "float":
            out.append("float: name")
        elif isinstance(node, ast.Div | ast.Pow):
            out.append("float: operator")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    out += [f"banned: {name}" for name in sorted(used & _BANNED_NAMES)]
    return out


class TestLeafAstPin:
    def test_imports_are_an_exact_whitelist(self) -> None:  # WP23-P1-01
        src = _LEAF.read_text()
        assert not [p for p in leaf_problems(src) if p.startswith("import:")]
        imported = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names
        }
        assert imported, "leaf imports nothing?"
        assert imported <= _ALLOWED_IMPORTS
        assert all(m.split(".")[0] in sys.stdlib_module_names for m in _ALLOWED_IMPORTS)

    def test_no_env_clock_logging_io_or_metrics(self) -> None:  # WP23-P1-02
        assert not [p for p in leaf_problems(_LEAF.read_text()) if p.startswith("banned:")]

    def test_no_float_arithmetic_in_the_leaf(self) -> None:  # WP23-P1-03 (I-7)
        assert not [p for p in leaf_problems(_LEAF.read_text()) if p.startswith("float:")]

    @pytest.mark.parametrize(
        ("addition", "expected"),
        [
            ("import pathlib", "import: pathlib"),
            ("from random import shuffle", "import: random"),
            ("import urllib.request", "import: urllib.request"),
            ("from os import environ", "import: os"),
            ("import logging", "import: logging"),
            ("from time import time", "import: time"),
            ("from common import models", "import: common"),
            ("from . import sibling", "import: ."),
            ("_x = datetime.now()", "banned: now"),
            ("_y = open('f')", "banned: open"),
            ("_z = 0.5", "float: literal"),
            ("_w = 1 / 2", "float: operator"),
            ("_v = float(3)", "float: name"),
        ],
    )
    def test_pin_fails_on_mutated_source(
        self, addition: str, expected: str
    ) -> None:  # P1-70 (CR-01)
        src = _LEAF.read_text()
        assert leaf_problems(src) == []  # the real leaf is clean ...
        problems = leaf_problems(src + f"\n{addition}\n")  # ... and the pin can fail
        assert expected in problems, problems


# ---------------------------------------------------------------------------
# WP23-P1-04..  constants, tolerance, error
# ---------------------------------------------------------------------------


class TestToleranceDefaults:
    def test_u01_option_a_qd_values(self) -> None:  # WP23-P1-04  (SY-23-03)
        t = DEFAULT_ALIGNMENT_TOLERANCE
        assert t.as_dict() == {
            "max_missing_ppm": 10_000,
            "warn_missing_ppm": 1_000,
            "max_gap_bars": 4,
            "warn_gap_bars": 2,
            "max_head_gap_bars": 0,
            "max_tail_gap_bars": 1,
            "fold_max_missing_floor": 1,
            "fold_max_missing_ppm": 20_000,
        }
        assert AlignmentTolerance() == t

    def test_frozen_and_integers_only(self) -> None:  # WP23-P1-05
        with pytest.raises(AttributeError):
            DEFAULT_ALIGNMENT_TOLERANCE.max_gap_bars = 9  # type: ignore[misc]
        for bad in (0.01, True, -1, "4"):
            with pytest.raises(ValueError, match="non-negative integers"):
                AlignmentTolerance(max_gap_bars=bad)  # type: ignore[arg-type]

    def test_version_marker(self) -> None:  # WP23-P1-06
        assert BAR_ALIGNMENT_VERSION == "wp23-1"
        assert align_bars({"A": mk("A", range(3))}, ["A"], H).report.version == "wp23-1"


class TestErrorContract:
    def test_closed_code_set_in_precedence_order(self) -> None:  # WP23-P1-07
        assert BAR_ALIGNMENT_CODES == (
            "bar_alignment_off_grid",
            "bar_alignment_no_common_row",
            "bar_alignment_head_coverage",
            "bar_alignment_tail_coverage",
            "bar_alignment_missing_bars",
            "bar_alignment_gap_too_long",
            "bar_alignment_fold_tolerance",
            "bar_alignment_not_trimmed",
        )

    def test_fixed_text_valueerror_subclass_and_pickle(self) -> None:  # WP23-P1-08
        assert issubclass(BarAlignmentError, ValueError)
        for code in BAR_ALIGNMENT_CODES:
            exc = BarAlignmentError(code)
            assert str(exc) == f"bar alignment check failed ({code})"
            assert exc.code == code
        clone = pickle.loads(pickle.dumps(BarAlignmentError(BAR_ALIGNMENT_CODES[4])))  # noqa: S301
        assert clone.code == BAR_ALIGNMENT_CODES[4] and str(clone) == str(
            BarAlignmentError(BAR_ALIGNMENT_CODES[4])
        )
        with pytest.raises(ValueError, match="unknown bar alignment code"):
            BarAlignmentError("secret detail")

    def test_message_never_carries_symbol_or_data(self) -> None:  # WP23-P1-09 (I-8)
        bars = {"SECRET/EUR": mk("SECRET/EUR", [0, 1, 2, 9, 10])}
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(align_bars(bars, ["SECRET/EUR"], H).report)
        assert "SECRET" not in str(ei.value) and str(ei.value) == (
            "bar alignment check failed (bar_alignment_missing_bars)"
        )
        assert ei.value.violations[0].symbol == "SECRET/EUR"  # structured, for logs only


# ---------------------------------------------------------------------------
# WP23-P1-10..  union timeline + present mask (T-04)
# ---------------------------------------------------------------------------


class TestUnionTimeline:
    def test_qd_t03_fixture_counts(self) -> None:  # WP23-P1-10  (T-04)
        out = align_bars(qd_fixture(), ["A", "B", "C"], H)
        r = out.report
        assert r.expected == 12
        by = {s.symbol: s for s in r.per_symbol}
        assert (by["A"].expected - by["A"].missing, by["B"].expected - by["B"].missing) == (12, 9)
        assert by["C"].expected - by["C"].missing == 11
        assert r.all_present_steps == 8 and r.steps == 12 and len(out.timeline) == 12
        assert [st.ts_ms for st in out.timeline] == [ms_of(ts_of(k)) for k in range(12)]
        assert [st.ts for st in out.timeline] == [ts_of(k) for k in range(12)]
        assert (by["B"].max_gap_bars, by["C"].max_gap_bars, by["A"].max_gap_bars) == (3, 1, 0)
        assert slot_spans(by["B"].gap_runs, "interior") == [(4, 3)]
        assert r.all_absent_slots == 0 and r.index_gap_runs == ()

    def test_present_mask_and_indices_point_at_the_right_bars(self) -> None:  # WP23-P1-11
        bars = qd_fixture()
        out = align_bars(bars, ["A", "B", "C"], H)
        assert out.timeline[5].present == (("A", 5), ("C", 5))  # B absent
        assert out.timeline[7].present == (("A", 7), ("B", 4), ("C", 7))  # B's 5th bar
        assert out.timeline[9].present == (("A", 9), ("B", 6))  # C absent
        for st in out.timeline:
            for sym, idx in st.present:
                assert out.bars_by_symbol[sym][idx].timestamp == st.ts

    def test_union_not_intersection_no_bar_dropped(self) -> None:  # WP23-P1-12 (mutant: inner join)
        bars = qd_fixture()
        out = align_bars(bars, ["A", "B", "C"], H)
        assert {s: len(v) for s, v in out.bars_by_symbol.items()} == {"A": 12, "B": 9, "C": 11}
        assert out.bars_by_symbol["A"] is bars["A"]  # complete symbol untouched
        assert len(out.timeline) == 12 > 8  # an inner join would give 8

    def test_market_wide_hole_is_an_index_gap_not_a_step(self) -> None:  # WP23-P1-13
        ks = rng_without(12, {5})
        bars = {"A": mk("A", ks), "B": mk("B", ks)}
        r = align_bars(bars, ["A", "B"], H).report
        assert r.expected == 12 and r.steps == 11 and r.all_absent_slots == 1
        assert slot_spans(r.index_gap_runs, "index") == [(5, 1)]
        assert [s.missing for s in r.per_symbol] == [1, 1]  # visible in G

    def test_build_timeline_matches_align_on_trimmed_input(self) -> None:  # WP23-P1-14
        out = align_bars(qd_fixture(), ["A", "B", "C"], H)
        assert build_timeline(out.bars_by_symbol, ["A", "B", "C"], H) == out.timeline

    def test_steps_follow_symbol_order_not_dict_order(self) -> None:  # WP23-P1-15
        bars = qd_fixture()
        rev = {k: bars[k] for k in ("C", "B", "A")}
        assert build_timeline(rev, ["A", "B", "C"], H) == build_timeline(bars, ["A", "B", "C"], H)


class TestSingleSymbolIdentity:
    def test_build_timeline_identity_mapping_no_validation(self) -> None:  # WP23-P1-16 (I-1)
        # non-monotone with a duplicate: the engine fast path must not validate or copy
        weird = [Bar("A", ts_of(k)) for k in (3, 1, 1, 7)]
        steps = build_timeline({"A": weird}, ["A"], H)
        assert [st.present for st in steps] == [(("A", i),) for i in range(4)]
        assert [st.ts for st in steps] == [b.timestamp for b in weird]

    @pytest.mark.parametrize("missing", [set(), {3, 9, 20, 21, 40, 41, 42, 70, 71}])
    def test_align_returns_the_same_list_object(self, missing: set[int]) -> None:  # WP23-P1-17
        lst = mk("A", rng_without(100, missing))
        before = list(lst)
        out = align_bars({"A": lst}, ["A"], H)
        assert out.bars_by_symbol["A"] is lst and lst == before
        assert [st.present for st in out.timeline] == [(("A", i),) for i in range(len(lst))]
        s = out.report.per_symbol[0]
        assert (s.bars_in, s.bars_out, s.trimmed_leading, s.trimmed_trailing) == (
            len(lst), len(lst), 0, 0,
        )  # fmt: skip

    def test_gapped_single_symbol_is_reported_not_raised_by_align(self) -> None:  # WP23-P1-18
        lst = mk("A", rng_without(100, {10, 11, 12, 13, 14, 15}))
        out = align_bars({"A": lst}, ["A"], H)  # must not raise
        assert out.report.violations and out.bars_by_symbol["A"] is lst


# ---------------------------------------------------------------------------
# WP23-P1-19..  window / coverage rules
# ---------------------------------------------------------------------------


class TestWindowAndCoverage:
    def test_head_late_aborts_head_coverage(self) -> None:  # WP23-P1-19  (T-07a)
        bars = {"A": mk("A", range(100)), "B": mk("B", range(5, 100))}
        out = align_bars(bars, ["A", "B"], H)
        b = out.report.per_symbol[1]
        assert (b.head, b.tail, b.missing) == (5, 0, 5)
        assert out.report.violations[0].code == "bar_alignment_head_coverage"
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_head_coverage"
        # execution window starts at the first full row: A's 5 leading bars are trimmed
        assert out.report.per_symbol[0].trimmed_leading == 5 and len(out.bars_by_symbol["A"]) == 95
        assert out.report.exec_start_ms == ms_of(ts_of(5))

    def test_head_is_measured_against_the_requested_window(self) -> None:  # WP23-P1-20 (mutant)
        # both symbols start at slot 3; the observed span is complete (head 0 against it)
        bars = {"A": mk("A", range(3, 50)), "B": mk("B", range(3, 50))}
        no_win = align_bars(bars, ["A", "B"], H).report
        assert [s.head for s in no_win.per_symbol] == [0, 0]
        win = align_bars(bars, ["A", "B"], H, window=(ts_of(0), ts_of(50))).report
        assert [s.head for s in win.per_symbol] == [3, 3]
        assert win.violations[0].code == "bar_alignment_head_coverage"
        assert slot_spans(win.per_symbol[0].gap_runs, "head") == [(0, 3)]

    def test_tail_one_early_passes_two_early_aborts(self) -> None:  # WP23-P1-21  (T-07b/c)
        a = mk("A", range(100))
        one = align_bars({"A": a, "B": mk("B", range(99))}, ["A", "B"], H)
        assert one.report.per_symbol[1].tail == 1 and one.report.violations == ()
        check_alignment(one.report)
        two = align_bars({"A": a, "B": mk("B", range(98))}, ["A", "B"], H)
        assert two.report.per_symbol[1].tail == 2
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(two.report)
        assert ei.value.code == "bar_alignment_tail_coverage"
        assert two.report.per_symbol[0].trimmed_trailing == 2  # E ends at the last full row

    def test_truncated_fetch_299_of_2000_is_tail_coverage(self) -> None:  # WP23-P1-22  (T-07d)
        win = (ts_of(0), ts_of(2000))
        out = align_bars({"A": mk("A", range(299))}, ["A"], H, window=win)
        s = out.report.per_symbol[0]
        assert (s.expected, s.tail, s.head, s.missing) == (2000, 1701, 0, 1701)
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_tail_coverage"

    def test_forming_bar_outside_the_window_is_trimmed_and_counted(self) -> None:  # WP23-P1-23
        lst = mk("A", range(11))  # slot 10 is the forming bar
        out = align_bars({"A": lst}, ["A"], H, window=(ts_of(0), ts_of(10)))
        s = out.report.per_symbol[0]
        assert (s.expected, s.trimmed_outside, s.bars_out, s.tail) == (10, 1, 10, 0)
        assert out.bars_by_symbol["A"] == lst[:10] and out.bars_by_symbol["A"] is not lst
        assert out.report.violations == () and out.report.window_requested is True

    def test_window_start_is_rounded_up_to_the_grid(self) -> None:  # WP23-P1-24
        half = T0 + timedelta(minutes=30)
        out = align_bars({"A": mk("A", range(1, 10))}, ["A"], H, window=(half, ts_of(10)))
        assert out.report.expected == 9 and out.report.per_symbol[0].head == 0

    def test_library_default_window_is_the_union_span(self) -> None:  # WP23-P1-25
        bars = {"A": mk("A", range(2, 20)), "B": mk("B", range(0, 15))}
        r = align_bars(bars, ["A", "B"], H).report
        assert (r.window_start_ms, r.window_end_ms) == (ms_of(ts_of(0)), ms_of(ts_of(20)))
        assert r.expected == 20 and r.window_requested is False

    def test_no_common_row(self) -> None:  # WP23-P1-26
        bars = {"A": mk("A", range(0, 5)), "B": mk("B", range(10, 15))}
        out = align_bars(bars, ["A", "B"], H)
        assert out.timeline == () and out.report.steps == 0 and out.report.exec_start_ms is None
        assert out.bars_by_symbol == {"A": [], "B": []}
        assert out.report.violations[0].code == "bar_alignment_no_common_row"
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_no_common_row"

    def test_empty_symbol_or_missing_key_is_no_common_row(self) -> None:  # WP23-P1-27
        out = align_bars({"A": mk("A", range(5))}, ["A", "B"], H)
        assert out.report.violations[0].code == "bar_alignment_no_common_row"
        assert out.report.per_symbol[1].bars_in == 0

    def test_window_with_no_bars_inside_empties_a_single_symbol(self) -> None:  # WP23-P1-28
        out = align_bars({"A": mk("A", range(5))}, ["A"], H, window=(ts_of(20), ts_of(30)))
        s = out.report.per_symbol[0]
        assert (s.expected, s.head, s.missing, s.trimmed_outside) == (10, 10, 10, 5)
        assert out.bars_by_symbol["A"] == []


# ---------------------------------------------------------------------------
# WP23-P1-29..  T-06 / T-09 gap runs and tolerance
# ---------------------------------------------------------------------------


class TestGapRunsAndTolerance:
    def _one(
        self,
        n: int,
        missing: set[int],
        tf: int = H,
        t0: datetime = T0,
        zero: set[int] | None = None,
    ) -> ba.AlignedBars[Bar]:
        lst = mk("X", rng_without(n, missing), tf, t0, frozenset(zero or ()))
        return align_bars({"X": lst}, ["X"], tf)

    def test_qd_t06_hourly_168(self) -> None:  # WP23-P1-29
        out = self._one(168, {10, 50, 51, 100, 101, 102, 103})
        s = out.report.per_symbol[0]
        assert (s.expected, s.expected - s.missing, s.missing) == (168, 161, 7)
        assert s.missing_ppm == 41_667 and s.max_gap_bars == 4
        assert slot_spans(s.gap_runs, "interior") == [(10, 1), (50, 2), (100, 4)]
        assert [v.code for v in out.report.violations] == ["bar_alignment_missing_bars"]
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_missing_bars"

    def test_qd_t06_adding_slot_104_reports_both_violations(self) -> None:  # WP23-P1-30 (mutant)
        out = self._one(168, {10, 50, 51, 100, 101, 102, 103, 104})
        s = out.report.per_symbol[0]
        assert (s.missing, s.missing_ppm, s.max_gap_bars) == (8, 47_619, 5)
        assert slot_spans(s.gap_runs, "interior")[-1] == (100, 5)
        assert [v.code for v in out.report.violations] == [
            "bar_alignment_missing_bars",
            "bar_alignment_gap_too_long",
        ]
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_missing_bars"  # precedence
        assert [v.code for v in ei.value.violations] == [v.code for v in out.report.violations]

    def test_qd_t06_daily_936_passes_with_two_warnings(self) -> None:  # WP23-P1-31
        out = self._one(936, {300, 600, 601}, tf=D, t0=datetime(2024, 1, 1, tzinfo=UTC))
        s = out.report.per_symbol[0]
        assert (s.expected, s.missing, s.missing_ppm, s.max_gap_bars) == (936, 3, 3205, 2)
        assert out.report.violations == ()
        warnings = check_alignment(out.report)
        assert [w.code for w in warnings] == ["warn_missing_bars", "warn_gap_bars"]
        assert out.report.warnings == warnings

    def test_missing_ppm_boundary_is_inclusive(self) -> None:  # WP23-P1-32 (mutant: >= vs >)

        def scattered(n: int) -> set[int]:  # no adjacent slots
            return {50 + 7 * i for i in range(n)}

        ok = self._one(1000, scattered(10))
        assert ok.report.per_symbol[0].missing_ppm == 10_000 and ok.report.violations == ()
        bad = self._one(1000, scattered(11))
        assert [v.code for v in bad.report.violations] == ["bar_alignment_missing_bars"]

    def test_missing_warn_boundary_is_exclusive(self) -> None:  # WP23-P1-33
        assert self._one(1000, {500}).report.warnings == ()  # exactly 1000 ppm: no warning
        two = self._one(1000, {500, 600})
        assert [w.code for w in two.report.warnings] == ["warn_missing_bars"]

    def test_gap_boundaries(self) -> None:  # WP23-P1-34 (mutant: off-by-one gap length)
        four = self._one(10_000, {100, 101, 102, 103})
        assert four.report.per_symbol[0].max_gap_bars == 4 and four.report.violations == ()
        assert [w.code for w in four.report.warnings] == ["warn_gap_bars"]
        five = self._one(10_000, {100, 101, 102, 103, 104})
        assert five.report.per_symbol[0].max_gap_bars == 5
        assert [v.code for v in five.report.violations] == ["bar_alignment_gap_too_long"]
        one = self._one(10_000, {100})
        assert one.report.per_symbol[0].max_gap_bars == 1 and one.report.warnings == ()
        two = self._one(10_000, {100, 101})
        assert two.report.per_symbol[0].max_gap_bars == 2 and len(two.report.warnings) == 1

    def test_gap_run_lengths_are_exact_for_every_size(self) -> None:  # WP23-P1-35 (mutant)
        for size in range(1, 12):
            gap = set(range(40, 40 + size))
            s = self._one(200, gap).report.per_symbol[0]
            assert s.max_gap_bars == size and slot_spans(s.gap_runs, "interior") == [(40, size)]

    def test_head_tail_boundaries(self) -> None:  # WP23-P1-36
        t = DEFAULT_ALIGNMENT_TOLERANCE
        assert (t.max_head_gap_bars, t.max_tail_gap_bars) == (0, 1)
        win = (ts_of(0), ts_of(100))
        head1 = align_bars({"A": mk("A", range(1, 100))}, ["A"], H, window=win).report
        assert [v.code for v in head1.violations] == ["bar_alignment_head_coverage"]
        tail1 = align_bars({"A": mk("A", range(99))}, ["A"], H, window=win).report
        assert tail1.violations == () and tail1.per_symbol[0].tail == 1
        tail2 = align_bars({"A": mk("A", range(98))}, ["A"], H, window=win).report
        # 2 missing of 100 is also 20 000 ppm; tail precedes missing in the code order
        assert [v.code for v in tail2.violations] == [
            "bar_alignment_tail_coverage",
            "bar_alignment_missing_bars",
        ]

    def test_zero_volume_counts_as_missing_but_stays_in_the_data(self) -> None:  # WP23-P1-37
        zeros = {50 + 7 * i for i in range(10)}
        ok = self._one(1000, set(), zero=zeros)
        s = ok.report.per_symbol[0]
        assert (s.zero_volume, s.missing, s.missing_ppm, s.bars_out) == (10, 10, 10_000, 1000)
        # isolated zero-volume bars are dead runs of 1 (QA-04); gap_runs stay structural
        assert s.max_gap_bars == 1 and s.gap_runs == () and ok.report.violations == ()
        bad = self._one(1000, set(), zero=zeros | {900})
        assert bad.report.per_symbol[0].zero_volume == 11
        assert [v.code for v in bad.report.violations] == ["bar_alignment_missing_bars"]
        assert len(bad.bars_by_symbol["X"]) == 1000 and len(bad.timeline) == 1000

    def test_holes_in_warmup_count_toward_the_tolerance_and_are_reported(
        self,
    ) -> None:  # WP23-P1-38
        lst = mk("X", rng_without(200, {10, 100}))
        out = align_bars({"X": lst}, ["X"], H, warmup_bars=50)
        s = out.report.per_symbol[0]
        assert (s.missing, s.holes_in_warmup, s.holes_in_trading) == (2, 1, 1)
        assert s.missing_ppm == 10_000 and out.report.violations == ()
        three = align_bars(
            {"X": mk("X", rng_without(200, {10, 20, 100}))}, ["X"], H, warmup_bars=50
        )
        assert three.report.per_symbol[0].holes_in_warmup == 2
        assert [v.code for v in three.report.violations] == ["bar_alignment_missing_bars"]
        default = align_bars({"X": lst}, ["X"], H).report.per_symbol[0]
        assert (default.holes_in_warmup, default.holes_in_trading) == (0, 2)

    def test_per_symbol_precedence_follows_symbol_order(self) -> None:  # WP23-P1-39
        bars = {
            "A": mk("A", rng_without(100, {10, 11, 12, 13, 14, 15})),
            "B": mk("B", rng_without(100, set(range(40, 60)))),
        }
        r = align_bars(bars, ["A", "B"], H).report
        got = [(v.code, v.symbol) for v in r.violations]
        assert got == [
            ("bar_alignment_missing_bars", "A"),
            ("bar_alignment_missing_bars", "B"),
            ("bar_alignment_gap_too_long", "A"),
            ("bar_alignment_gap_too_long", "B"),
        ]
        swapped = align_bars(bars, ["B", "A"], H).report
        assert [(v.code, v.symbol) for v in swapped.violations][:2] == [
            ("bar_alignment_missing_bars", "B"),
            ("bar_alignment_missing_bars", "A"),
        ]

    def test_check_with_explicit_tolerance_recomputes(self) -> None:  # WP23-P1-40
        out = self._one(168, {10, 50, 51, 100, 101, 102, 103})
        assert check_alignment(out.report, LENIENT) == ()
        strict = replace(DEFAULT_ALIGNMENT_TOLERANCE, max_gap_bars=3)
        with pytest.raises(BarAlignmentError):
            check_alignment(out.report, strict)


# ---------------------------------------------------------------------------
# WP23-P1-41..  T-08 grid / anchor / guards
# ---------------------------------------------------------------------------


class TestGridAndGuards:
    def test_off_grid_bar_aborts_whatever_the_thresholds(self) -> None:  # WP23-P1-41  (T-08)
        lst = mk("A", range(10))
        lst[4] = Bar("A", ts_of(4) + timedelta(minutes=30))
        out = align_bars({"A": lst}, ["A"], H, tolerance=LENIENT)
        assert out.report.per_symbol[0].off_grid == 1
        assert out.report.violations[0].code == "bar_alignment_off_grid"
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report, LENIENT)
        assert ei.value.code == "bar_alignment_off_grid"
        # passthrough, rejected; a single symbol keeps the identity timeline (CR-02)
        assert out.bars_by_symbol["A"] is lst and out.timeline == build_timeline(
            {"A": lst}, ["A"], H
        )
        assert len(out.timeline) == 10

    def test_phase_offset_fixture_is_off_grid_for_runner_and_engine(self) -> None:  # WP23-P1-42
        a = mk("A", range(10))
        b = mk("B", range(10), t0=T0 + timedelta(minutes=30))
        out = align_bars({"A": a, "B": b}, ["A", "B"], H)
        assert [(v.code, v.symbol) for v in out.report.violations] == [
            ("bar_alignment_off_grid", "B")
        ]
        with pytest.raises(BarAlignmentError) as ei:
            build_timeline({"A": a, "B": b}, ["A", "B"], H)
        assert ei.value.code == "bar_alignment_off_grid"

    def test_sub_millisecond_and_naive_timestamps(self) -> None:  # WP23-P1-43
        lst = mk("A", range(5))
        lst[2] = Bar("A", ts_of(2) + timedelta(microseconds=1))
        assert align_bars({"A": lst}, ["A"], H).report.per_symbol[0].off_grid == 1
        naive = [Bar("A", ts_of(k).replace(tzinfo=None)) for k in range(5)]
        n = align_bars({"A": naive}, ["A"], H)
        assert n.report.per_symbol[0].off_grid == 0 and n.timeline[0].ts == T0

    def test_duplicate_or_decreasing_is_off_grid(self) -> None:  # WP23-P1-44
        for ks in ([0, 1, 1, 2], [0, 2, 1, 3]):
            r = align_bars({"A": mk("A", ks)}, ["A"], H).report
            assert r.per_symbol[0].off_grid == 1
            assert r.violations[0].code == "bar_alignment_off_grid"
            with pytest.raises(BarAlignmentError):
                build_timeline({"A": mk("A", ks), "B": mk("B", range(4))}, ["A", "B"], H)

    def test_weekly_anchor_is_the_phase_of_the_first_symbol(self) -> None:  # WP23-P1-45
        week = 604_800
        monday = datetime(2026, 3, 2, tzinfo=UTC)
        a = mk("A", range(10), week, monday)
        b = mk("B", range(10), week, monday)
        ok = align_bars({"A": a, "B": b}, ["A", "B"], week)
        assert ok.report.violations == () and ok.report.anchor_phase_ms == ms_of(monday) % (
            week * 1000
        )
        tue = mk("B", range(10), week, monday + timedelta(days=1))
        bad = align_bars({"A": a, "B": tue}, ["A", "B"], week)
        assert [v.symbol for v in bad.report.violations] == ["B"]
        # epoch-anchored weeks (Thursday) are off-grid against a Monday-phased first symbol
        thu = mk("B", range(10), week, datetime(1970, 1, 1, tzinfo=UTC) + timedelta(weeks=2900))
        assert align_bars({"A": a, "B": thu}, ["A", "B"], week).report.violations[0].symbol == "B"

    def test_engine_timeframe_table_assumption(self) -> None:  # WP23-P1-46
        """SY-23-02: every engine timeframe divides a day, except 1w (phase anchored)."""
        from trading.strategy_engine import _TIMEFRAME_SECONDS

        not_dividing = {tf for tf, s in _TIMEFRAME_SECONDS.items() if D % s}
        assert {tf.value for tf in not_dividing} == {"1w"}
        for tf, secs in _TIMEFRAME_SECONDS.items():
            bars = {"A": mk("A", range(5), secs), "B": mk("B", range(5), secs)}
            r = align_bars(bars, ["A", "B"], secs).report
            assert r.violations == () and r.steps == 5, tf

    def test_untrimmed_multi_symbol_input_is_not_trimmed(self) -> None:  # WP23-P1-47  (T-08)
        head = {"A": mk("A", range(10)), "B": mk("B", range(3, 10))}
        with pytest.raises(BarAlignmentError) as ei:
            build_timeline(head, ["A", "B"], H)
        assert ei.value.code == "bar_alignment_not_trimmed"
        assert ei.value.violations[0].symbol == "B"
        tail = {"A": mk("A", range(10)), "B": mk("B", range(7))}
        with pytest.raises(BarAlignmentError) as ej:
            build_timeline(tail, ["A", "B"], H)
        assert ej.value.code == "bar_alignment_not_trimmed"

    def test_interior_gaps_are_allowed_by_build_timeline(self) -> None:  # WP23-P1-48
        assert len(build_timeline(qd_fixture(), ["A", "B", "C"], H)) == 12

    def test_argument_validation_is_fixed_text(self) -> None:  # WP23-P1-49
        for tf in (0, -5, 1.5, True):
            with pytest.raises(ValueError, match="timeframe_s must be a positive integer"):
                align_bars({"A": []}, ["A"], tf)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one symbol"):
            build_timeline({}, [], H)
        assert len(align_bars({"A": mk("A", range(4))}, ["A", "A"], H).report.per_symbol) == 1


# ---------------------------------------------------------------------------
# WP23-P1-50..  idempotence, purity, determinism, report content
# ---------------------------------------------------------------------------


def _walk_no_floats(obj: object, path: str = "") -> None:
    assert type(obj) is not float, path
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), path
            _walk_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_no_floats(v, f"{path}[{i}]")
    else:
        assert obj is None or isinstance(obj, int | str | bool), (path, type(obj))


class TestReportDeterminism:
    def _report(self, order: tuple[str, ...] = ("A", "B", "C")) -> ba.AlignmentReport:
        bars = qd_fixture()
        shuffled = {k: bars[k] for k in order}
        return align_bars(shuffled, ["A", "B", "C"], H, warmup_bars=3).report

    def test_metadata_is_ints_and_strings_only(self) -> None:  # WP23-P1-50  (mutant: float)
        meta = self._report().to_metadata()
        _walk_no_floats(meta)
        assert set(meta) >= {
            "version", "policy", "tolerance", "window", "exec_window", "steps",
            "all_present_steps", "all_absent_slots", "symbols", "presence_digest",
            "fingerprint", "index_gap_spans", "violations", "warnings",
        }  # fmt: skip
        assert meta["version"] == "wp23-1" and meta["policy"] == "union_present_mask"
        sym_keys = set(meta["symbols"][0])  # type: ignore[index]
        assert {
            "bars_in", "bars_out", "expected", "missing", "missing_ppm", "zero_volume",
            "max_gap_bars", "head", "tail", "trimmed_outside", "trimmed_leading",
            "trimmed_trailing", "holes_in_warmup", "holes_in_trading", "gap_spans",
            "gap_spans_truncated",
        } <= sym_keys  # fmt: skip

    def test_fingerprint_is_sha256_of_the_canonical_json_without_itself(self) -> None:  # P1-51
        r = self._report()
        meta = r.to_metadata()
        core = {k: v for k, v in meta.items() if k != "fingerprint"}
        text = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert meta["fingerprint"] == hashlib.sha256(text.encode()).hexdigest() == r.fingerprint
        assert json.loads(r.to_canonical_json()) == meta
        assert len(r.fingerprint) == 64

    def test_dict_order_and_repeat_runs_give_identical_bytes(self) -> None:  # WP23-P1-52
        a = self._report(("A", "B", "C"))
        assert a.to_canonical_json() == self._report(("C", "A", "B")).to_canonical_json()
        assert a.to_canonical_json() == self._report().to_canonical_json()
        assert a.presence_digest == self._report(("B", "C", "A")).presence_digest

    def test_presence_digest_is_sorted_symbol_ts_lines_of_the_output(self) -> None:  # WP23-P1-53
        bars = {"B": mk("B", range(4)), "A": mk("A", range(4))}
        out = align_bars(bars, ["B", "A"], H)
        lines = "".join(f"{s}|{ms_of(ts_of(k))}\n" for s in ("A", "B") for k in range(4))
        assert out.report.presence_digest == hashlib.sha256(lines.encode()).hexdigest()
        assert align_bars(bars, ["A", "B"], H).report.presence_digest == out.report.presence_digest
        dropped = {"B": mk("B", [0, 1, 3]), "A": mk("A", range(4))}
        # B loses slot 2, A keeps it: digest changes (it hashes presence, not just steps)
        assert (
            align_bars(dropped, ["B", "A"], H, tolerance=LENIENT).report.presence_digest
            != out.report.presence_digest
        )

    def test_digest_covers_only_the_bars_handed_to_the_engine(self) -> None:  # WP23-P1-54
        bars = {"A": mk("A", range(10)), "B": mk("B", range(2, 10))}
        out = align_bars(bars, ["A", "B"], H)
        text = "".join(f"A|{ms_of(ts_of(k))}\n" for k in range(2, 10)) + "".join(
            f"B|{ms_of(ts_of(k))}\n" for k in range(2, 10)
        )
        assert out.report.presence_digest == hashlib.sha256(text.encode()).hexdigest()

    def test_idempotent_and_no_input_mutation(self) -> None:  # WP23-P1-55
        bars = {"A": mk("A", range(3, 40)), "B": mk("B", range(0, 36)), "C": mk("C", range(1, 50))}
        snapshot = {k: list(v) for k, v in bars.items()}
        first = align_bars(bars, ["A", "B", "C"], H, tolerance=LENIENT)
        assert bars == snapshot and all(bars[k] is not first.bars_by_symbol[k] for k in "ABC")
        again = align_bars(first.bars_by_symbol, ["A", "B", "C"], H, tolerance=LENIENT)
        assert again.bars_by_symbol == first.bars_by_symbol and again.timeline == first.timeline
        assert all(again.bars_by_symbol[k] is first.bars_by_symbol[k] for k in "ABC")  # untrimmed
        assert again.report.presence_digest == first.report.presence_digest

    def test_untrimmed_multi_symbol_returns_the_same_list_objects(self) -> None:  # WP23-P1-56
        bars = qd_fixture()
        out = align_bars(bars, ["A", "B", "C"], H)
        assert all(out.bars_by_symbol[k] is bars[k] for k in bars)

    def test_ordinary_ohlcv_bars_work(self) -> None:  # WP23-P1-57
        from common.models import OHLCVBar
        from common.types import TimeFrame

        def bar(sym: str, k: int, vol: str = "5") -> OHLCVBar:
            return OHLCVBar(
                symbol=sym, timeframe=TimeFrame.ONE_HOUR, timestamp=ts_of(k),
                open=Decimal(1), high=Decimal(2), low=Decimal(1), close=Decimal(1),
                volume=Decimal(vol),
            )  # fmt: skip

        bars = {
            "A": [bar("A", k) for k in range(10)],
            "B": [bar("B", k, "0" if k == 4 else "5") for k in range(10) if k != 7],
        }
        out = align_bars(bars, ["A", "B"], H)
        b = out.report.per_symbol[1]
        assert (b.zero_volume, b.missing, b.bars_out) == (1, 2, 9) and out.report.steps == 10
        assert out.bars_by_symbol["A"] is bars["A"]


# ---------------------------------------------------------------------------
# WP23-P1-58..  seeded property loops (hypothesis is not a dependency)
# ---------------------------------------------------------------------------


def _random_case(
    seed: int,
) -> tuple[list[str], dict[str, list[Bar]], tuple[datetime, datetime] | None, int]:
    rnd = random.Random(seed)  # noqa: S311 - deterministic test data
    n_sym = rnd.randint(1, 4)
    n = rnd.randint(5, 80)
    syms = [f"S{i}" for i in range(n_sym)]
    bars: dict[str, list[Bar]] = {}
    for s in syms:
        lo = rnd.randint(0, 6) if rnd.random() < 0.5 else 0
        hi = n - (rnd.randint(0, 6) if rnd.random() < 0.5 else 0)
        keep = [k for k in range(lo, max(hi, lo + 1)) if rnd.random() > 0.15]
        zero = frozenset(k for k in keep if rnd.random() < 0.05)
        bars[s] = mk(s, keep or [lo], zero=zero)
    window = None
    if rnd.random() < 0.5:
        a = rnd.randint(-3, 5)
        window = (ts_of(a), ts_of(a + rnd.randint(1, n + 5)))
    return syms, bars, window, rnd.randint(0, 20)


@pytest.mark.parametrize("seed", range(60))
def test_property_invariants(seed: int) -> None:  # WP23-P1-58
    syms, bars, window, warm = _random_case(seed)
    snapshot = {k: list(v) for k, v in bars.items()}
    out = align_bars(bars, syms, H, window=window, tolerance=LENIENT, warmup_bars=warm)
    r = out.report
    assert bars == snapshot  # no mutation
    n = len(syms)
    # accounting per symbol
    for sa in r.per_symbol:
        src, kept = bars[sa.symbol], out.bars_by_symbol[sa.symbol]
        assert sa.bars_in == len(src) and sa.bars_out == len(kept)
        assert (
            sa.bars_in
            == sa.bars_out + sa.trimmed_outside + sa.trimmed_leading + sa.trimmed_trailing
        )
        if kept:  # contiguous slice of the input, same objects
            i = src.index(kept[0])
            assert all(a is b for a, b in zip(kept, src[i : i + len(kept)], strict=True))
        run_total = sum(g.bars for g in sa.gap_runs)
        assert run_total + sa.zero_volume == sa.missing
        assert sa.holes_in_warmup + sa.holes_in_trading <= sa.missing
    if r.steps == 0:
        assert out.timeline == ()
        return
    # timeline: strictly increasing union of the output, mask exact, edges full
    ts = [st.ts_ms for st in out.timeline]
    assert ts == sorted(set(ts))
    union = {ms_of(b.timestamp) for v in out.bars_by_symbol.values() for b in v}
    assert set(ts) == union
    for st in out.timeline:
        assert [s for s, _ in st.present] == [
            s for s in syms if any(ms_of(b.timestamp) == st.ts_ms for b in out.bars_by_symbol[s])
        ]
        for sym, idx in st.present:
            assert ms_of(out.bars_by_symbol[sym][idx].timestamp) == st.ts_ms
    assert len(out.timeline[0].present) == n and len(out.timeline[-1].present) == n
    assert r.exec_start_ms == ts[0] and r.exec_end_ms == ts[-1]
    assert r.steps + r.all_absent_slots == (ts[-1] - ts[0]) // 3_600_000 + 1
    # build_timeline on the trimmed output agrees, and align is idempotent
    assert build_timeline(out.bars_by_symbol, syms, H) == out.timeline
    again = align_bars(out.bars_by_symbol, syms, H, tolerance=LENIENT)
    assert again.bars_by_symbol == out.bars_by_symbol and again.timeline == out.timeline


@pytest.mark.parametrize("seed", range(20))
def test_property_symbol_stream_is_independent_of_the_basket(seed: int) -> None:  # WP23-P1-59
    """SY-23-01: A's bars as seen by the engine do not depend on who else is in the run."""
    case_seed = seed + 1000
    syms, bars, _, _ = _random_case(case_seed)
    while len(syms) < 2:  # deterministic re-draw: this property needs a basket
        case_seed += 100
        syms, bars, _, _ = _random_case(case_seed)
    full = align_bars(bars, syms, H, tolerance=LENIENT)
    for sym in syms:
        solo = align_bars({sym: bars[sym]}, [sym], H, tolerance=LENIENT)
        assert solo.bars_by_symbol[sym] is bars[sym]  # alone: nothing trimmed
        kept = full.bars_by_symbol[sym]
        assert all(b in bars[sym] for b in kept)  # basket only ever trims the edges


@pytest.mark.parametrize("seed", range(20))
def test_property_fingerprint_independent_of_dict_order(seed: int) -> None:  # WP23-P1-60
    syms, bars, window, _ = _random_case(seed + 2000)
    rnd = random.Random(seed)  # noqa: S311
    keys = list(bars)
    rnd.shuffle(keys)
    a = align_bars(bars, syms, H, window=window, tolerance=LENIENT).report
    b = align_bars({k: bars[k] for k in keys}, syms, H, window=window, tolerance=LENIENT).report
    assert a.to_canonical_json() == b.to_canonical_json()
    _walk_no_floats(a.to_metadata())


# ---------------------------------------------------------------------------
# WP23-P1-61..  fold_check (SY-23-03 last row, SY-23-17)
# ---------------------------------------------------------------------------


class TestFoldCheck:
    def _bars(self, n: int, missing: set[int]) -> dict[str, list[Bar]]:
        return {"A": mk("A", rng_without(n, missing))}

    def test_clean_folds_pass(self) -> None:  # WP23-P1-61
        bars = self._bars(200, set())
        fold_check(bars, ["A"], H, [(ts_of(0), ts_of(100)), (ts_of(100), ts_of(200))])

    def test_small_fold_allows_one_missing_not_two(self) -> None:  # WP23-P1-62
        fold = [(ts_of(0), ts_of(24))]  # max(1, floor(2% * 24) = 0) = 1
        fold_check(self._bars(24, {10}), ["A"], H, fold)
        with pytest.raises(BarAlignmentError) as ei:
            fold_check(self._bars(24, {10, 15}), ["A"], H, fold)
        assert ei.value.code == "bar_alignment_fold_tolerance"
        assert str(ei.value) == "bar alignment check failed (bar_alignment_fold_tolerance)"

    def test_two_percent_boundary_is_inclusive(self) -> None:  # WP23-P1-63 (mutant)
        fold = [(ts_of(0), ts_of(100))]  # floor(2%) = 2
        fold_check(self._bars(100, {10, 30}), ["A"], H, fold)
        with pytest.raises(BarAlignmentError):
            fold_check(self._bars(100, {10, 30, 50}), ["A"], H, fold)

    def test_fold_gap_longest_run_four_passes_five_fails(self) -> None:  # WP23-P1-64
        fold = [(ts_of(0), ts_of(1000))]  # allowed missing 20, so only the gap rule bites
        fold_check(self._bars(1000, {5, 6, 7, 8}), ["A"], H, fold)
        with pytest.raises(BarAlignmentError):
            fold_check(self._bars(1000, {5, 6, 7, 8, 9}), ["A"], H, fold)

    def test_fold_edges_count_as_gaps(self) -> None:  # WP23-P1-65
        bars = {"A": mk("A", range(5, 100))}  # head of 5 inside the fold
        with pytest.raises(BarAlignmentError):
            fold_check(bars, ["A"], H, [(ts_of(0), ts_of(1000))])
        with pytest.raises(BarAlignmentError):
            fold_check({"A": []}, ["A"], H, [(ts_of(0), ts_of(10))])

    def test_violation_reports_fold_index_and_symbol_order(self) -> None:  # WP23-P1-66
        bars = {"A": mk("A", range(200)), "B": mk("B", rng_without(200, {120, 130, 140}))}
        folds = [(ts_of(0), ts_of(100)), (ts_of(100), ts_of(200))]
        with pytest.raises(BarAlignmentError) as ei:
            fold_check(bars, ["A", "B"], H, folds)
        assert [(v.fold, v.symbol, v.observed, v.limit) for v in ei.value.violations] == [
            (1, "B", 3, 2)
        ]

    def test_zero_volume_counts_in_a_fold(self) -> None:  # WP23-P1-67
        bars = {"A": mk("A", range(100), zero=frozenset({3, 30, 60}))}
        with pytest.raises(BarAlignmentError):
            fold_check(bars, ["A"], H, [(ts_of(0), ts_of(100))])

    def test_off_grid_input_raises_off_grid(self) -> None:  # WP23-P1-68
        bars = {"A": [Bar("A", ts_of(0)), Bar("A", ts_of(1) + timedelta(minutes=5))]}
        with pytest.raises(BarAlignmentError) as ei:
            fold_check(bars, ["A"], H, [(ts_of(0), ts_of(2))])
        assert ei.value.code == "bar_alignment_off_grid"


# ---------------------------------------------------------------------------
# WP23-P1-69  performance guard (T-04: 6 x 20k with seeded 0.2% drops; measured ~0.3 s, bound 5 s)
# ---------------------------------------------------------------------------


def test_performance_guard_6x20k() -> None:  # WP23-P1-69
    rnd = random.Random(7)  # noqa: S311
    syms = [f"S{i}" for i in range(6)]
    n = 20_000
    bars: dict[str, list[Bar]] = {}
    for s in syms:
        drop = {k for k in range(1, n - 1) if rnd.random() < 0.002}
        bars[s] = mk(s, rng_without(n, drop))
    t = time.perf_counter()
    out = align_bars(bars, syms, H)
    elapsed = time.perf_counter() - t
    assert elapsed < 5.0, elapsed
    assert out.report.steps == n and out.report.expected == n  # union: nothing is dropped
    assert all(len(out.bars_by_symbol[s]) == len(bars[s]) for s in syms)
    t = time.perf_counter()
    build_timeline(out.bars_by_symbol, syms, H)
    assert time.perf_counter() - t < 5.0


# ---------------------------------------------------------------------------
# v2 additions: QA-04/05 zero-volume dead runs, CR-02..05
# ---------------------------------------------------------------------------

D0 = datetime(2024, 1, 1, tzinfo=UTC)


def _daily(n: int, absent: set[int], zero: set[int]) -> ba.AlignedBars[Bar]:
    lst = mk("X", rng_without(n, absent), D, D0, frozenset(zero))
    return align_bars({"X": lst}, ["X"], D)


class TestZeroVolumeDeadRuns:
    """Quant QA-04: max_gap_bars runs over slots that are absent OR zero-volume."""

    def test_nine_consecutive_zero_volume_days_abort_gap_too_long(self) -> None:  # WP23-P1-71
        out = _daily(936, set(), set(range(300, 309)))
        s = out.report.per_symbol[0]
        assert (s.zero_volume, s.missing, s.missing_ppm, s.max_gap_bars) == (9, 9, 9615, 9)
        assert s.gap_runs == ()  # structural: absent slots only
        assert [v.code for v in out.report.violations] == ["bar_alignment_gap_too_long"]
        assert (out.report.violations[0].observed, out.report.violations[0].limit) == (9, 4)
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_gap_too_long"
        assert len(out.bars_by_symbol["X"]) == 936  # zero-volume bars still reach the engine

    def test_three_absent_plus_three_zero_adjacent_merge_into_one_run(self) -> None:  # P1-72
        out = _daily(936, {300, 301, 302}, {303, 304, 305})
        s = out.report.per_symbol[0]
        assert (s.missing, s.zero_volume, s.max_gap_bars) == (6, 3, 6)
        assert slot_spans(s.gap_runs, "interior", D, D0) == [(300, 3)]  # absent only
        assert sum(g.bars for g in s.gap_runs) + s.zero_volume == s.missing
        assert [v.code for v in out.report.violations] == ["bar_alignment_gap_too_long"]

    @pytest.mark.parametrize(
        ("absent", "zero", "run"),
        [
            (set(), set(range(300, 304)), 4),  # four zero-volume: boundary passes
            ({300}, {301, 302, 303}, 4),  # absent then zeros
            ({301, 302}, {300, 303}, 4),  # zeros on both sides of an absent run
            ({300, 301}, {303, 304}, 2),  # an ok bar at 302 splits two runs of 2
        ],
    )
    def test_dead_run_boundary_four_passes(
        self, absent: set[int], zero: set[int], run: int
    ) -> None:  # WP23-P1-73
        out = _daily(936, absent, zero)
        assert out.report.per_symbol[0].max_gap_bars == run
        assert out.report.violations == ()
        assert [w.code for w in out.report.warnings].count("warn_gap_bars") == 1

    def test_five_dead_slots_fail_wherever_the_split_between_kinds_falls(self) -> None:  # P1-74
        for n_zero in range(6):
            out = _daily(936, set(range(300, 305 - n_zero)), set(range(305 - n_zero, 305)))
            assert out.report.per_symbol[0].max_gap_bars == 5, n_zero
            assert [v.code for v in out.report.violations] == ["bar_alignment_gap_too_long"]

    def test_isolated_zero_volume_bars_stay_below_the_warning(self) -> None:  # WP23-P1-75
        out = _daily(936, set(), {100, 200, 300})
        assert out.report.per_symbol[0].max_gap_bars == 1
        assert out.report.warnings == () or [w.code for w in out.report.warnings] == [
            "warn_missing_bars"
        ]

    def test_dead_runs_in_the_multi_symbol_path(self) -> None:  # WP23-P1-76
        bars = {
            "A": mk("A", range(50), zero=frozenset(range(10, 16))),
            "B": mk("B", range(50)),
        }
        out = align_bars(bars, ["A", "B"], H, tolerance=LENIENT)
        assert [s.max_gap_bars for s in out.report.per_symbol] == [6, 0]
        assert [(v.code, v.symbol) for v in align_bars(bars, ["A", "B"], H).report.violations][
            -1
        ] == ("bar_alignment_gap_too_long", "A")

    def test_fold_dead_runs(self) -> None:  # WP23-P1-77
        fold = [(ts_of(0), ts_of(1000))]  # allowed missing 20: only the run rule can bite

        def bars(zero: set[int], absent: set[int] = frozenset()) -> dict[str, list[Bar]]:  # type: ignore[assignment]
            return {"A": mk("A", rng_without(1000, set(absent)), zero=frozenset(zero))}

        fold_check(bars(set(range(5, 9))), ["A"], H, fold)  # 4 zero-volume passes
        with pytest.raises(BarAlignmentError) as ei:
            fold_check(bars(set(range(5, 10))), ["A"], H, fold)  # 5 fail
        assert ei.value.code == "bar_alignment_fold_tolerance"
        with pytest.raises(BarAlignmentError):  # 3 absent + 3 zero adjacent
            fold_check(bars({503, 504, 505}, {500, 501, 502}), ["A"], H, fold)

    def test_fold_edges_are_ok_sentinels(self) -> None:  # WP23-P1-78
        fold = [(ts_of(0), ts_of(1000))]
        edge4 = {"A": mk("A", range(1000), zero=frozenset(range(4)))}
        fold_check(edge4, ["A"], H, fold)  # dead run of 4 touching the edge passes
        edge5 = {"A": mk("A", range(1000), zero=frozenset(range(5)))}
        with pytest.raises(BarAlignmentError):
            fold_check(edge5, ["A"], H, fold)
        tail5 = {"A": mk("A", range(1000), zero=frozenset(range(995, 1000)))}
        with pytest.raises(BarAlignmentError):
            fold_check(tail5, ["A"], H, fold)
        tail_absent4 = {"A": mk("A", range(996))}
        fold_check(tail_absent4, ["A"], H, fold)  # edge absent run of 4 still passes
        with pytest.raises(BarAlignmentError):
            fold_check({"A": mk("A", range(995))}, ["A"], H, fold)


# ---------------------------------------------------------------------------
# CR-02: single symbol is the identity in BOTH entry points, bad data included
# ---------------------------------------------------------------------------


class TestSingleSymbolBadData:
    @pytest.mark.parametrize("kind", ["off_grid", "duplicate", "decreasing"])
    def test_identity_timeline_same_list_finding_and_raise(self, kind: str) -> None:  # P1-79
        ks = {"off_grid": [0, 1, 2, 3], "duplicate": [0, 1, 1, 2], "decreasing": [0, 2, 1, 3]}[kind]
        lst = mk("A", ks)
        if kind == "off_grid":
            lst[2] = Bar("A", ts_of(2) + timedelta(minutes=7))
        out = align_bars({"A": lst}, ["A"], H)
        assert out.bars_by_symbol["A"] is lst
        assert out.timeline == build_timeline({"A": lst}, ["A"], H)
        assert [st.present for st in out.timeline] == [(("A", i),) for i in range(4)]
        assert out.report.steps == 4 and out.report.per_symbol[0].off_grid == 1
        assert [v.code for v in out.report.violations] == ["bar_alignment_off_grid"]
        assert out.report.index_gap_runs == () and out.report.exec_start_ms is None
        with pytest.raises(BarAlignmentError) as ei:
            check_alignment(out.report)
        assert ei.value.code == "bar_alignment_off_grid"

    def test_multi_symbol_bad_data_keeps_the_empty_timeline(self) -> None:  # WP23-P1-80
        bad = mk("B", [0, 1, 1, 2])
        out = align_bars({"A": mk("A", range(3)), "B": bad}, ["A", "B"], H)
        assert out.timeline == () and out.bars_by_symbol["B"] is bad

    @pytest.mark.parametrize("seed", range(60))
    def test_property_single_symbol_align_equals_build_timeline(self, seed: int) -> None:  # P1-81
        rnd = random.Random(seed)  # noqa: S311
        ks = sorted(rnd.sample(range(80), rnd.randint(0, 40)))
        lst = mk("A", ks)
        roll = rnd.random()
        if lst and roll < 0.25:
            rnd.shuffle(lst)
        elif lst and roll < 0.5:
            lst.insert(rnd.randrange(len(lst)), lst[rnd.randrange(len(lst))])
        elif lst and roll < 0.75:
            i = rnd.randrange(len(lst))
            lst[i] = Bar("A", lst[i].timestamp + timedelta(seconds=rnd.randint(1, 3599)))
        before = list(lst)
        out = align_bars({"A": lst}, ["A"], H, tolerance=LENIENT)
        assert lst == before
        assert out.timeline == build_timeline({"A": lst}, ["A"], H)
        assert out.bars_by_symbol["A"] is lst
        windowed = align_bars(
            {"A": lst}, ["A"], H, window=(ts_of(10), ts_of(50)), tolerance=LENIENT
        )
        assert windowed.timeline == build_timeline(windowed.bars_by_symbol, ["A"], H)


# ---------------------------------------------------------------------------
# CR-03 / CR-04 / CR-05
# ---------------------------------------------------------------------------


class TestFoldFloorAndDegenerate:
    def test_floor_not_ceil_at_n125(self) -> None:  # WP23-P1-82 (CR-03; mutant ceil)
        fold = [(ts_of(0), ts_of(125))]  # 125 * 2% = 2.5 -> floor 2
        two = {"A": mk("A", rng_without(125, {10, 60}))}
        fold_check(two, ["A"], H, fold)
        with pytest.raises(BarAlignmentError) as ei:
            fold_check({"A": mk("A", rng_without(125, {10, 60, 100}))}, ["A"], H, fold)
        assert [(v.observed, v.limit) for v in ei.value.violations] == [(3, 2)]

    @pytest.mark.parametrize("delta", [0, -1, -24])
    def test_start_not_before_end_is_a_plain_valueerror(self, delta: int) -> None:  # P1-83 (CR-04)
        bars = {"A": mk("A", range(50))}
        good = (ts_of(0), ts_of(25))
        bad = (ts_of(30), ts_of(30 + delta))
        with pytest.raises(ValueError, match=r"^fold must have start < end$") as ei:
            fold_check(bars, ["A"], H, [good, bad])
        assert not isinstance(ei.value, BarAlignmentError)
        sub = (ts_of(1), ts_of(1) + timedelta(milliseconds=-1))  # sub-slot reversed
        with pytest.raises(ValueError, match="fold must have start < end"):
            fold_check(bars, ["A"], H, [sub])

    def test_closed_code_set_unchanged(self) -> None:  # WP23-P1-84
        assert len(BAR_ALIGNMENT_CODES) == 8


class TestIsoHelper:
    def test_iso_from_ms_and_gap_run_property(self) -> None:  # WP23-P1-85 (CR-05, P3 hand-off)
        assert "iso_from_ms" in ba.__all__
        assert iso_from_ms(ms_of(T0)) == "2026-03-02T00:00:00+00:00"
        assert iso_from_ms(ms_of(T0) + 1500) == "2026-03-02T00:00:01.500000+00:00"
        out = align_bars({"A": mk("A", rng_without(20, {5, 6}))}, ["A"], H)
        g = out.report.per_symbol[0].gap_runs[0]
        assert g.start_iso == iso_from_ms(g.start_ms) == "2026-03-02T05:00:00+00:00"
        meta = out.report.to_metadata()["symbols"][0]["gap_spans"][0]  # type: ignore[index]
        assert meta["start"] == g.start_iso
