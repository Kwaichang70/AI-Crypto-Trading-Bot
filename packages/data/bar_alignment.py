"""Multi-symbol bar alignment (WP2.3, SY-23): a pure, stdlib-only leaf.

Policy ``union_present_mask`` (SY-23-01): the engine walks the sorted union of the
observed bar timestamps; at each step only the symbols that have a bar are
"present". No synthetic bar is ever created (I-4) and no bar is dropped from a
complete symbol.

Model (SY-23-02):

* ``R``  reference window: the requested ``[start, end)`` of bar open times, or
  ``[min first, max last + tf)`` over all symbols when none is given.
* ``G``  the calendar grid: every tf slot of ``R`` (epoch-anchored for tf | 86400;
  otherwise the phase of the first bar of the first non-empty symbol).
* Per symbol, ``head`` / ``tail`` are the grid slots before the first / after the
  last observed bar; ``interior`` runs are the absent slots strictly between them.
  ``gap_runs`` are structural (absent slots only). ``max_gap_bars`` is different: it is
  the longest run of slots that are absent OR zero-volume (adjacent ones merge), because
  a venue-filled zero-volume bar cannot trigger a stop and so leaves it unchecked just
  like an absent one (quant ruling QA-04). Zero-volume bars still reach the engine.
* ``E``  execution window ``[first full row, last full row]``; bars outside ``E``
  are trimmed. The timeline is the observed union inside ``E``.

Purity: standard library only; no logging, environment, wall clock, I/O or
metrics. Timestamps are integer milliseconds since the epoch; everything that is
hashed or persisted is an ``int``, a ``str`` or a ``bool`` (never a float).
Bars are duck-typed through ``.timestamp`` and ``.volume`` and the timeframe is
passed in as integer seconds, so this module imports nothing from the project.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import Final, Generic, Protocol, TypeVar

__all__ = [
    "BAR_ALIGNMENT_CODES",
    "BAR_ALIGNMENT_POLICY",
    "BAR_ALIGNMENT_VERSION",
    "CODE_FOLD_TOLERANCE",
    "CODE_GAP_TOO_LONG",
    "CODE_HEAD_COVERAGE",
    "CODE_MISSING_BARS",
    "CODE_NOT_TRIMMED",
    "CODE_NO_COMMON_ROW",
    "CODE_OFF_GRID",
    "CODE_TAIL_COVERAGE",
    "DEFAULT_ALIGNMENT_TOLERANCE",
    "AlignedBars",
    "AlignmentFinding",
    "AlignmentReport",
    "AlignmentTolerance",
    "BarAlignmentError",
    "BarLike",
    "GapRun",
    "Step",
    "SymbolAlignment",
    "align_bars",
    "build_timeline",
    "check_alignment",
    "fold_check",
    "iso_from_ms",
]

BAR_ALIGNMENT_VERSION: Final = "wp23-1"
BAR_ALIGNMENT_POLICY: Final = "union_present_mask"

# ---------------------------------------------------------------------------
# TOLERANCE DEFAULTS  (SY-23-03, user decision U-01 option (a) = QD values)
#
# This is the ONLY place the numbers live. Switching to option (b) AD or (c) DP
# means editing exactly this block (plus the expected values in the tests).
# Code constants, never environment, never an API field. Integers only.
# ---------------------------------------------------------------------------
_TOL_MAX_MISSING_PPM: Final = 10_000  # abort if missing/|G| > 1%   (zero-volume = missing)
_TOL_WARN_MISSING_PPM: Final = 1_000  # warn  if missing/|G| > 0.1%
_TOL_MAX_GAP_BARS: Final = 4  # abort if longest interior run > 4 bars
_TOL_WARN_GAP_BARS: Final = 2  # warn  if longest interior run >= 2 bars
_TOL_MAX_HEAD_BARS: Final = 0  # abort if head gap > 0 (late listing)
_TOL_MAX_TAIL_BARS: Final = 1  # abort if tail gap > 1 (covers a dropped forming bar)
_TOL_FOLD_MISSING_FLOOR: Final = 1  # per test fold: missing > max(1, ...)
_TOL_FOLD_MISSING_PPM: Final = 20_000  # ... floor(2% * n_fold); max gap reuses _TOL_MAX_GAP_BARS
# ---------------------------------------------------------------------------

# Closed set of error codes, in evaluation (precedence) order (SY-23-07).
CODE_OFF_GRID: Final = "bar_alignment_off_grid"
CODE_NO_COMMON_ROW: Final = "bar_alignment_no_common_row"
CODE_HEAD_COVERAGE: Final = "bar_alignment_head_coverage"
CODE_TAIL_COVERAGE: Final = "bar_alignment_tail_coverage"
CODE_MISSING_BARS: Final = "bar_alignment_missing_bars"
CODE_GAP_TOO_LONG: Final = "bar_alignment_gap_too_long"
CODE_FOLD_TOLERANCE: Final = "bar_alignment_fold_tolerance"
CODE_NOT_TRIMMED: Final = "bar_alignment_not_trimmed"
BAR_ALIGNMENT_CODES: Final[tuple[str, ...]] = (
    CODE_OFF_GRID,
    CODE_NO_COMMON_ROW,
    CODE_HEAD_COVERAGE,
    CODE_TAIL_COVERAGE,
    CODE_MISSING_BARS,
    CODE_GAP_TOO_LONG,
    CODE_FOLD_TOLERANCE,
    CODE_NOT_TRIMMED,
)

_WARN_MISSING: Final = "warn_missing_bars"
_WARN_GAP: Final = "warn_gap_bars"

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_US: Final = timedelta(microseconds=1)
_MS: Final = timedelta(milliseconds=1)
_SECONDS_PER_DAY: Final = 86_400
_MAX_SPANS_IN_METADATA: Final = 20


class BarLike(Protocol):
    """Duck type of ``common.models.OHLCVBar`` as far as this leaf needs it."""

    @property
    def timestamp(self) -> datetime: ...

    @property
    def volume(self) -> Decimal | int: ...


B = TypeVar("B", bound=BarLike)


@dataclass(frozen=True, slots=True)
class AlignmentTolerance:
    """Alignment thresholds. Integers only (ppm = parts per million)."""

    max_missing_ppm: int = _TOL_MAX_MISSING_PPM
    warn_missing_ppm: int = _TOL_WARN_MISSING_PPM
    max_gap_bars: int = _TOL_MAX_GAP_BARS
    warn_gap_bars: int = _TOL_WARN_GAP_BARS
    max_head_gap_bars: int = _TOL_MAX_HEAD_BARS
    max_tail_gap_bars: int = _TOL_MAX_TAIL_BARS
    fold_max_missing_floor: int = _TOL_FOLD_MISSING_FLOOR
    fold_max_missing_ppm: int = _TOL_FOLD_MISSING_PPM

    def __post_init__(self) -> None:
        for value in self.as_dict().values():
            if type(value) is not int or value < 0:
                raise ValueError("alignment tolerance values must be non-negative integers")

    def as_dict(self) -> dict[str, int]:
        return {
            "max_missing_ppm": self.max_missing_ppm,
            "warn_missing_ppm": self.warn_missing_ppm,
            "max_gap_bars": self.max_gap_bars,
            "warn_gap_bars": self.warn_gap_bars,
            "max_head_gap_bars": self.max_head_gap_bars,
            "max_tail_gap_bars": self.max_tail_gap_bars,
            "fold_max_missing_floor": self.fold_max_missing_floor,
            "fold_max_missing_ppm": self.fold_max_missing_ppm,
        }


DEFAULT_ALIGNMENT_TOLERANCE: Final = AlignmentTolerance()


@dataclass(frozen=True, slots=True)
class AlignmentFinding:
    """One violation (``code`` in :data:`BAR_ALIGNMENT_CODES`) or warning (``warn_*``)."""

    code: str
    symbol: str
    observed: int
    limit: int
    fold: int = -1


class BarAlignmentError(ValueError):
    """Alignment rejected. The message is fixed text; never carries data (CF-36)."""

    def __init__(self, code: str, violations: tuple[AlignmentFinding, ...] = ()) -> None:
        if code not in BAR_ALIGNMENT_CODES:
            raise ValueError("unknown bar alignment code")
        super().__init__(f"bar alignment check failed ({code})")
        self.code: str = code
        self.violations: tuple[AlignmentFinding, ...] = violations

    def __reduce__(
        self,
    ) -> tuple[type[BarAlignmentError], tuple[str, tuple[AlignmentFinding, ...]]]:
        return (type(self), (self.code, self.violations))


@dataclass(frozen=True, slots=True)
class GapRun:
    """One maximal run of absent grid slots. ``kind``: head | interior | tail | index."""

    start_ms: int
    bars: int
    kind: str

    @property
    def start_iso(self) -> str:
        return iso_from_ms(self.start_ms)


@dataclass(frozen=True, slots=True)
class Step:
    """One timeline step: bar-open time and the ``(symbol, index)`` of each present bar."""

    ts: datetime
    ts_ms: int
    present: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class SymbolAlignment:
    symbol: str
    bars_in: int
    bars_out: int
    expected: int
    missing: int
    missing_ppm: int
    zero_volume: int
    max_gap_bars: int
    head: int
    tail: int
    off_grid: int
    trimmed_outside: int
    trimmed_leading: int
    trimmed_trailing: int
    holes_in_warmup: int
    holes_in_trading: int
    first_ms: int | None
    last_ms: int | None
    gap_runs: tuple[GapRun, ...]


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    version: str
    policy: str
    timeframe_s: int
    anchor_phase_ms: int
    window_start_ms: int
    window_end_ms: int
    window_requested: bool
    expected: int
    exec_start_ms: int | None
    exec_end_ms: int | None
    steps: int
    all_present_steps: int
    all_absent_slots: int
    symbols: tuple[str, ...]
    per_symbol: tuple[SymbolAlignment, ...]
    index_gap_runs: tuple[GapRun, ...]
    presence_digest: str
    tolerance: AlignmentTolerance
    violations: tuple[AlignmentFinding, ...]
    warnings: tuple[AlignmentFinding, ...]

    def to_metadata(self) -> dict[str, object]:
        """Persistable form: ints, strs, bools, lists and dicts only (never floats)."""
        core = self._core_metadata()
        core["fingerprint"] = _sha256_hex(_canonical_json(core))
        return core

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_metadata())

    @property
    def fingerprint(self) -> str:
        return _sha256_hex(_canonical_json(self._core_metadata()))

    def _core_metadata(self) -> dict[str, object]:
        index_spans = self.index_gap_runs
        return {
            "version": self.version,
            "policy": self.policy,
            "timeframe_s": self.timeframe_s,
            "anchor_phase_ms": self.anchor_phase_ms,
            "tolerance": self.tolerance.as_dict(),
            "window": {
                "start": iso_from_ms(self.window_start_ms),
                "end": iso_from_ms(self.window_end_ms),
                "requested": self.window_requested,
                "slots": self.expected,
            },
            "exec_window": (
                None
                if self.exec_start_ms is None or self.exec_end_ms is None
                else {
                    "start": iso_from_ms(self.exec_start_ms),
                    "end": iso_from_ms(self.exec_end_ms),
                }
            ),
            "steps": self.steps,
            "all_present_steps": self.all_present_steps,
            "all_absent_slots": self.all_absent_slots,
            "index_gap_spans": [_span_meta(g) for g in index_spans[:_MAX_SPANS_IN_METADATA]],
            "index_gap_spans_truncated": len(index_spans) > _MAX_SPANS_IN_METADATA,
            "symbols": [_symbol_meta(s) for s in self.per_symbol],
            "presence_digest": self.presence_digest,
            "violations": [_finding_meta(f) for f in self.violations],
            "warnings": [_finding_meta(f) for f in self.warnings],
        }


@dataclass(frozen=True, slots=True)
class AlignedBars(Generic[B]):
    """``bars_by_symbol`` holds the SAME list objects when nothing was trimmed."""

    bars_by_symbol: dict[str, list[B]]
    timeline: tuple[Step, ...]
    report: AlignmentReport


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _us(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (ts - _EPOCH) // _US


def _dt(ms: int) -> datetime:
    return _EPOCH + ms * _MS


def iso_from_ms(ms: int) -> str:
    """ISO-8601 UTC string for integer epoch milliseconds (P3 logging hand-off, CF-36)."""
    return _dt(ms).isoformat()


def _canonical_json(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _span_meta(g: GapRun) -> dict[str, object]:
    return {"start": iso_from_ms(g.start_ms), "bars": g.bars, "kind": g.kind}


def _finding_meta(f: AlignmentFinding) -> dict[str, object]:
    return {"code": f.code, "symbol": f.symbol, "observed": f.observed, "limit": f.limit}


def _symbol_meta(s: SymbolAlignment) -> dict[str, object]:
    return {
        "symbol": s.symbol,
        "bars_in": s.bars_in,
        "bars_out": s.bars_out,
        "expected": s.expected,
        "missing": s.missing,
        "missing_ppm": s.missing_ppm,
        "zero_volume": s.zero_volume,
        "max_gap_bars": s.max_gap_bars,
        "head": s.head,
        "tail": s.tail,
        "off_grid": s.off_grid,
        "trimmed_outside": s.trimmed_outside,
        "trimmed_leading": s.trimmed_leading,
        "trimmed_trailing": s.trimmed_trailing,
        "holes_in_warmup": s.holes_in_warmup,
        "holes_in_trading": s.holes_in_trading,
        "first": None if s.first_ms is None else iso_from_ms(s.first_ms),
        "last": None if s.last_ms is None else iso_from_ms(s.last_ms),
        "gap_spans": [_span_meta(g) for g in s.gap_runs[:_MAX_SPANS_IN_METADATA]],
        "gap_spans_truncated": len(s.gap_runs) > _MAX_SPANS_IN_METADATA,
    }


def _check_timeframe(timeframe_s: int) -> int:
    if type(timeframe_s) is not int or timeframe_s <= 0:
        raise ValueError("timeframe_s must be a positive integer number of seconds")
    return timeframe_s


def _unique(symbols: Sequence[str]) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(symbols))
    if not out:
        raise ValueError("at least one symbol is required")
    return out


def _ppm(part: int, whole: int) -> int:
    """Round-half-up ``part / whole`` in parts per million, integers only."""
    if whole <= 0:
        return 0
    return (2 * part * 1_000_000 + whole) // (2 * whole)


def _phase_us(lists: Mapping[str, Sequence[BarLike]], symbols: Sequence[str], tf_s: int) -> int:
    """Grid anchor: epoch for tf | 1 day; else the phase of symbols[0]'s first bar."""
    if _SECONDS_PER_DAY % tf_s == 0:
        return 0
    tf_us = tf_s * 1_000_000
    for sym in symbols:
        bars = lists[sym]
        if len(bars) > 0:
            return _us(bars[0].timestamp) % tf_us
    return 0


@dataclass(slots=True)
class _Series:
    ms: list[int]  # on-grid, strictly increasing, de-duplicated open times
    ok: list[bool]  # parallel: volume > 0
    bad: int  # bars that were off-grid or not strictly increasing


def _scan(bars: Sequence[BarLike], tf_us: int, phase_us: int) -> _Series:
    ms: list[int] = []
    ok: list[bool] = []
    bad = 0
    top: int | None = None
    for bar in bars:
        u = _us(bar.timestamp)
        on_grid = (u - phase_us) % tf_us == 0
        if not on_grid or (top is not None and u <= top):
            bad += 1
        if on_grid:
            ms.append(u // 1000)
            ok.append(bool(bar.volume > 0))
        if top is None or u > top:
            top = u
    if bad:
        first: dict[int, bool] = {}
        for m, k in zip(ms, ok, strict=True):
            first.setdefault(m, k)
        ordered = sorted(first)
        return _Series(ordered, [first[m] for m in ordered], bad)
    return _Series(ms, ok, 0)


def _grid(start_ms: int, end_ms: int, phase_ms: int, tf_ms: int) -> tuple[int, int]:
    """First grid slot >= start and the number of slots < end."""
    first = start_ms + ((phase_ms - start_ms) % tf_ms)
    if first >= end_ms:
        return first, 0
    return first, (end_ms - first + tf_ms - 1) // tf_ms


def _identity_steps(symbol: str, bars: Sequence[BarLike]) -> tuple[Step, ...]:
    """Single-symbol timeline: step i <-> bar i, no validation, no copy of the bars."""
    return tuple(
        Step(_dt(m), m, ((symbol, i),))
        for i, m in enumerate(_us(b.timestamp) // 1000 for b in bars)
    )


def _longest_dead_run(ms: Sequence[int], ok: Sequence[bool], tf_ms: int) -> int:
    """Longest run of consecutive grid slots that are absent or zero-volume (``ok`` False).

    ``ms`` is strictly increasing on-grid; slots between two entries are absent. Pass
    ``ok=True`` sentinels one slot outside each end to count edge runs (fold_check).
    """
    longest = run = 0
    prev: int | None = None
    for m, good in zip(ms, ok, strict=True):
        if prev is not None:
            run += (m - prev) // tf_ms - 1
        if good:
            longest = max(longest, run)
            run = 0
        else:
            run += 1
        prev = m
    return max(longest, run)


def _make_steps(ms_lists: Sequence[Sequence[int]], symbols: Sequence[str]) -> tuple[Step, ...]:
    slots: dict[int, list[tuple[str, int]]] = {}
    for sym, series in zip(symbols, ms_lists, strict=True):
        for i, m in enumerate(series):
            slots.setdefault(m, []).append((sym, i))
    return tuple(Step(_dt(m), m, tuple(slots[m])) for m in sorted(slots))


def _runs_between(ms: Sequence[int], tf_ms: int) -> list[GapRun]:
    runs: list[GapRun] = []
    for prev, cur in pairwise(ms):
        d = cur - prev
        if d > tf_ms:
            runs.append(GapRun(prev + tf_ms, d // tf_ms - 1, "interior"))
    return runs


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_timeline(
    bars_by_symbol: Mapping[str, Sequence[BarLike]],
    symbols: Sequence[str],
    timeframe_s: int,
) -> tuple[Step, ...]:
    """Union timeline over (already trimmed) bar lists, with a present mask per step.

    Steps carry indices into the input lists; no bar list is copied or returned.
    One symbol: identity mapping ``step i <-> bar i`` with NO validation (the engine
    fast path; the runner validates order). Several symbols: raises
    ``bar_alignment_off_grid`` (off-grid, duplicate or non-increasing bars, phase
    mismatch) or ``bar_alignment_not_trimmed`` (first/last step not a full row).
    """
    tf_s = _check_timeframe(timeframe_s)
    syms = _unique(symbols)
    lists = {s: bars_by_symbol.get(s, ()) for s in syms}
    if len(syms) == 1:
        return _identity_steps(syms[0], lists[syms[0]])
    tf_us = tf_s * 1_000_000
    phase = _phase_us(lists, syms, tf_s)
    ms_lists: list[list[int]] = []
    findings: list[AlignmentFinding] = []
    for sym in syms:
        scanned = _scan(lists[sym], tf_us, phase)
        if scanned.bad:
            findings.append(AlignmentFinding(CODE_OFF_GRID, sym, scanned.bad, 0))
        ms_lists.append(scanned.ms)
    if findings:
        raise BarAlignmentError(CODE_OFF_GRID, tuple(findings))
    steps = _make_steps(ms_lists, syms)
    n = len(syms)
    if steps and (len(steps[0].present) != n or len(steps[-1].present) != n):
        edge = steps[0] if len(steps[0].present) != n else steps[-1]
        have = {s for s, _ in edge.present}
        absent = next(s for s in syms if s not in have)
        raise BarAlignmentError(
            CODE_NOT_TRIMMED, (AlignmentFinding(CODE_NOT_TRIMMED, absent, len(edge.present), n),)
        )
    return steps


def align_bars(
    bars_by_symbol: Mapping[str, list[B]],
    symbols: Sequence[str],
    timeframe_s: int,
    *,
    window: tuple[datetime, datetime] | None = None,
    tolerance: AlignmentTolerance = DEFAULT_ALIGNMENT_TOLERANCE,
    warmup_bars: int = 0,
) -> AlignedBars[B]:
    """Measure, trim to the execution window and build the timeline. Never mutates input.

    ``window`` is the requested ``[start, end)`` of bar open times (None: union span).
    The report carries the verdict against ``tolerance`` but nothing is raised here;
    call :func:`check_alignment`. When any bar is off-grid/non-increasing the bars are
    passed through untrimmed with an empty timeline (the run must be rejected).
    ``warmup_bars`` only splits the ``holes_in_*`` diagnostics.
    """
    tf_s = _check_timeframe(timeframe_s)
    syms = _unique(symbols)
    lists: dict[str, list[B]] = {s: bars_by_symbol.get(s, []) for s in syms}
    tf_ms = tf_s * 1000
    tf_us = tf_s * 1_000_000
    phase_us = _phase_us(lists, syms, tf_s)
    phase_ms = phase_us // 1000
    series = {s: _scan(lists[s], tf_us, phase_us) for s in syms}

    if window is not None:
        start_ms = -(-_us(window[0]) // 1000)
        end_ms = -(-_us(window[1]) // 1000)
    else:
        firsts = [sr.ms[0] for sr in series.values() if sr.ms]
        lasts = [sr.ms[-1] for sr in series.values() if sr.ms]
        start_ms = min(firsts) if firsts else 0
        end_ms = (max(lasts) + tf_ms) if lasts else 0
    g_first, n_g = _grid(start_ms, end_ms, phase_ms, tf_ms)
    g_last = g_first + (n_g - 1) * tf_ms

    bounds = {
        s: (bisect_left(series[s].ms, start_ms), bisect_left(series[s].ms, end_ms)) for s in syms
    }
    inw = {s: series[s].ms[bounds[s][0] : bounds[s][1]] for s in syms}
    any_bad = any(series[s].bad for s in syms)

    # execution window E = [first full row, last full row]
    full: set[int] = set(inw[syms[0]])
    for s in syms[1:]:
        full.intersection_update(inw[s])
    e_start = min(full) if full and not any_bad else None
    e_end = max(full) if full and not any_bad else None

    out_lists: dict[str, list[B]] = {}
    out_ms: list[list[int]] = []
    per_symbol: list[SymbolAlignment] = []
    for s in syms:
        sr = series[s]
        lo, hi = bounds[s]
        win = inw[s]
        zero_volume = len(win) - sum(sr.ok[lo:hi])
        present = len(win) - zero_volume
        missing = n_g - present
        if win:
            head = (win[0] - g_first) // tf_ms
            tail = (g_last - win[-1]) // tf_ms
            runs = _runs_between(win, tf_ms)
        else:
            head, tail, runs = n_g, 0, []
        gaps: list[GapRun] = []
        if head > 0:
            gaps.append(GapRun(g_first, head, "head"))
        gaps.extend(runs)
        if tail > 0:
            gaps.append(GapRun(win[-1] + tf_ms, tail, "tail"))
        max_gap = _longest_dead_run(win, sr.ok[lo:hi], tf_ms)

        bars = lists[s]
        if any_bad:
            kept: list[B] = bars
            kept_ms = sr.ms
            lead = trail = 0
        elif e_start is None or e_end is None:
            kept = [] if bars else bars  # an already-empty list stays the same object
            kept_ms = []
            lead, trail = len(win), 0
        else:
            lo_e = bisect_left(sr.ms, e_start)
            hi_e = bisect_right(sr.ms, e_end)
            kept = bars if (lo_e == 0 and hi_e == len(bars)) else bars[lo_e:hi_e]
            kept_ms = sr.ms[lo_e:hi_e]
            lead, trail = lo_e - lo, hi - hi_e
        out_lists[s] = kept
        out_ms.append(kept_ms)
        outside = 0 if any_bad else len(sr.ms) - len(win)
        per_symbol.append(
            SymbolAlignment(
                symbol=s,
                bars_in=len(bars),
                bars_out=len(kept),
                expected=n_g,
                missing=missing,
                missing_ppm=_ppm(missing, n_g),
                zero_volume=zero_volume,
                max_gap_bars=max_gap,
                head=head,
                tail=tail,
                off_grid=sr.bad,
                trimmed_outside=outside,
                trimmed_leading=lead,
                trimmed_trailing=trail,
                holes_in_warmup=0,
                holes_in_trading=0,
                first_ms=win[0] if win else None,
                last_ms=win[-1] if win else None,
                gap_runs=tuple(gaps),
            )
        )

    timeline: tuple[Step, ...]
    if not any_bad:
        timeline = _make_steps(out_ms, syms)
    elif len(syms) == 1:  # single symbol: always the identity timeline (I-1), finding kept
        timeline = _identity_steps(syms[0], lists[syms[0]])
    else:
        timeline = ()
    steps_ms = [st.ts_ms for st in timeline]
    index_runs = [] if any_bad else _runs_between(steps_ms, tf_ms)
    index_gaps = tuple(GapRun(g.start_ms, g.bars, "index") for g in index_runs)

    # holes_in_warmup / holes_in_trading: the symbol's absent slots inside E, split at
    # the open time of timeline step `warmup_bars` (the first W steps are warmup).
    if timeline and e_end is not None:
        if warmup_bars <= 0:
            boundary = steps_ms[0]
        elif warmup_bars < len(steps_ms):
            boundary = steps_ms[warmup_bars]
        else:
            boundary = e_end + tf_ms
        split: list[SymbolAlignment] = []
        for sa, kept_ms in zip(per_symbol, out_ms, strict=True):
            in_warm = in_trade = 0
            for run in _runs_between(kept_ms, tf_ms):
                w = min(max((boundary - run.start_ms + tf_ms - 1) // tf_ms, 0), run.bars)
                in_warm += w
                in_trade += run.bars - w
            split.append(replace(sa, holes_in_warmup=in_warm, holes_in_trading=in_trade))
        per_symbol = split

    n_sym = len(syms)
    digest = hashlib.sha256()
    for s in sorted(syms):
        kept_ms = out_ms[syms.index(s)]
        digest.update(("".join(f"{s}|{m}\n" for m in kept_ms)).encode("utf-8"))
    violations, warnings = _evaluate(
        per_symbol, steps=len(timeline), any_bad=any_bad, tolerance=tolerance
    )
    report = AlignmentReport(
        version=BAR_ALIGNMENT_VERSION,
        policy=BAR_ALIGNMENT_POLICY,
        timeframe_s=tf_s,
        anchor_phase_ms=phase_ms,
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        window_requested=window is not None,
        expected=n_g,
        exec_start_ms=e_start,
        exec_end_ms=e_end,
        steps=len(timeline),
        all_present_steps=sum(1 for st in timeline if len(st.present) == n_sym),
        all_absent_slots=sum(g.bars for g in index_gaps),
        symbols=syms,
        per_symbol=tuple(per_symbol),
        index_gap_runs=index_gaps,
        presence_digest=digest.hexdigest(),
        tolerance=tolerance,
        violations=violations,
        warnings=warnings,
    )
    return AlignedBars(bars_by_symbol=out_lists, timeline=timeline, report=report)


def _evaluate(
    per_symbol: Sequence[SymbolAlignment],
    *,
    steps: int,
    any_bad: bool,
    tolerance: AlignmentTolerance,
) -> tuple[tuple[AlignmentFinding, ...], tuple[AlignmentFinding, ...]]:
    """Violations in code-precedence order (then symbol order), plus warnings."""
    tol = tolerance
    v: list[AlignmentFinding] = []
    w: list[AlignmentFinding] = []
    for s in per_symbol:
        if s.off_grid > 0:  # never tolerable: no threshold
            v.append(AlignmentFinding(CODE_OFF_GRID, s.symbol, s.off_grid, 0))
    if any_bad:  # other measurements are unreliable on off-grid data: report only this
        return tuple(v), ()
    if steps == 0:
        v.append(AlignmentFinding(CODE_NO_COMMON_ROW, "", 0, 1))
    for s in per_symbol:
        if s.head > tol.max_head_gap_bars:
            v.append(AlignmentFinding(CODE_HEAD_COVERAGE, s.symbol, s.head, tol.max_head_gap_bars))
    for s in per_symbol:
        if s.tail > tol.max_tail_gap_bars:
            v.append(AlignmentFinding(CODE_TAIL_COVERAGE, s.symbol, s.tail, tol.max_tail_gap_bars))
    for s in per_symbol:
        if s.missing * 1_000_000 > tol.max_missing_ppm * s.expected:
            v.append(
                AlignmentFinding(CODE_MISSING_BARS, s.symbol, s.missing_ppm, tol.max_missing_ppm)
            )
        elif s.missing * 1_000_000 > tol.warn_missing_ppm * s.expected:
            w.append(AlignmentFinding(_WARN_MISSING, s.symbol, s.missing_ppm, tol.warn_missing_ppm))
    for s in per_symbol:
        if s.max_gap_bars > tol.max_gap_bars:
            v.append(
                AlignmentFinding(CODE_GAP_TOO_LONG, s.symbol, s.max_gap_bars, tol.max_gap_bars)
            )
        elif s.max_gap_bars >= tol.warn_gap_bars and s.max_gap_bars > 0:
            w.append(AlignmentFinding(_WARN_GAP, s.symbol, s.max_gap_bars, tol.warn_gap_bars))
    return tuple(v), tuple(w)


def check_alignment(
    report: AlignmentReport, tolerance: AlignmentTolerance | None = None
) -> tuple[AlignmentFinding, ...]:
    """Raise ``BarAlignmentError`` for the first violation; else return the warnings.

    With ``tolerance`` the verdict is recomputed from the report's measurements; otherwise
    the one stored in the report is used. The error's ``violations`` lists all of them.
    """
    if tolerance is None:
        violations, warnings = report.violations, report.warnings
    else:
        violations, warnings = _evaluate(
            report.per_symbol,
            steps=report.steps,
            any_bad=any(s.off_grid > 0 for s in report.per_symbol),
            tolerance=tolerance,
        )
    if violations:
        raise BarAlignmentError(violations[0].code, violations)
    return warnings


def fold_check(
    bars_by_symbol: Mapping[str, Sequence[BarLike]],
    symbols: Sequence[str],
    timeframe_s: int,
    folds: Sequence[tuple[datetime, datetime]],
    *,
    tolerance: AlignmentTolerance = DEFAULT_ALIGNMENT_TOLERANCE,
) -> None:
    """Per test fold ``[start, end)``: missing and longest run (SY-23-03, SY-23-17).

    A fold violates when ``missing > max(floor, n_fold * fold_ppm // 1e6)`` or its
    longest dead run exceeds ``max_gap_bars``. Zero-volume bars are missing, and a
    dead run is a run of slots that are absent OR zero-volume (adjacent ones merge);
    the fold edges count as present, so a gap touching an edge is included. Raises
    ``bar_alignment_fold_tolerance`` (or ``off_grid``); folds in order, symbols in
    ``symbols`` order, missing before gap. A fold with ``start >= end`` is a caller bug
    and raises a plain ``ValueError`` (the closed code set is not extended).
    """
    tf_s = _check_timeframe(timeframe_s)
    syms = _unique(symbols)
    lists = {s: bars_by_symbol.get(s, ()) for s in syms}
    tf_ms = tf_s * 1000
    tf_us = tf_s * 1_000_000
    phase_us = _phase_us(lists, syms, tf_s)
    series = {s: _scan(lists[s], tf_us, phase_us) for s in syms}
    bad = tuple(AlignmentFinding(CODE_OFF_GRID, s, series[s].bad, 0) for s in syms if series[s].bad)
    if bad:
        raise BarAlignmentError(CODE_OFF_GRID, bad)
    bounds_ms = [(-(-_us(a) // 1000), -(-_us(b) // 1000)) for a, b in folds]
    if any(a >= b for a, b in bounds_ms):
        raise ValueError("fold must have start < end")
    found: list[AlignmentFinding] = []
    for k, (start_ms, end_ms) in enumerate(bounds_ms):
        g_first, n_fold = _grid(start_ms, end_ms, phase_us // 1000, tf_ms)
        g_last = g_first + (n_fold - 1) * tf_ms
        allowed = max(
            tolerance.fold_max_missing_floor, n_fold * tolerance.fold_max_missing_ppm // 1_000_000
        )
        for s in syms:
            sr = series[s]
            lo, hi = bisect_left(sr.ms, start_ms), bisect_left(sr.ms, end_ms)
            win = sr.ms[lo:hi]
            missing = n_fold - sum(sr.ok[lo:hi])
            longest = _longest_dead_run(
                [g_first - tf_ms, *win, g_last + tf_ms],
                [True, *sr.ok[lo:hi], True],
                tf_ms,
            )
            if missing > allowed:
                found.append(AlignmentFinding(CODE_FOLD_TOLERANCE, s, missing, allowed, k))
            if longest > tolerance.max_gap_bars:
                found.append(
                    AlignmentFinding(CODE_FOLD_TOLERANCE, s, longest, tolerance.max_gap_bars, k)
                )
    if found:
        raise BarAlignmentError(CODE_FOLD_TOLERANCE, tuple(found))
