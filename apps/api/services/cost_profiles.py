"""
apps/api/services/cost_profiles.py
-----------------------------------
WP2.1 glue between the pure ``trading.costs`` registry and the API layer.

* :func:`new_run_cost_profile_or_422` -- resolve the profile of a NEW run from a
  request id, mapping :class:`~trading.costs.CostProfileError` onto a stable
  422 ``{"code": ...}`` (the request value is never echoed -- CF-36).
* :func:`engine_semantics` -- the ``config["engine_semantics"]`` block
  persisted next to ``config["cost_profile"]`` (closes CF-13b-16).
* :func:`stamp_cost_config` -- write both keys into a NEW run's config
  (create / promote).  Resume, boot recovery and salvage never call it.
* :func:`cost_response_fields` -- the five server-computed response fields
  (runs AND, since WP2.1b, optimization runs).

WP2.3 adds the API half of the bar-alignment contract (SY-23-08/14/16, P0 SEC-02):
the ``engine_semantics.bar_alignment`` marker, the ``legacy_bar_alignment`` response
flag, the 422 mapping of ``BarAlignmentError``, the requested-window clamp and the
fixed request bounds.  Nothing here reads a clock: ``now`` is passed in by the router.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from fastapi import HTTPException, status

from api.schemas import CostProfileSummary, build_cost_profile_summary
from common.types import TimeFrame
from data.bar_alignment import BAR_ALIGNMENT_CODES, BAR_ALIGNMENT_VERSION, BarAlignmentError
from trading.clock import TIMESTAMP_CLOCK_VERSION
from trading.costs import (
    CostProfile,
    CostProfileError,
    FeeSource,
    is_leaderboard_cost_eligible,
    is_legacy_cost_config,
    new_run_cost_profile,
    resolve_run_cost_profile,
)
from trading.exit_config import EXIT_SEMANTICS_VERSION
from trading.safety import BACKTEST_CIRCUIT_BREAKER_DEFAULT
from trading.strategy_engine import _TIMEFRAME_SECONDS

__all__ = [
    "COST_MODEL_VERSION",
    "KNOWN_BAR_ALIGNMENTS",
    "KNOWN_TIMESTAMP_CLOCKS",
    "MAX_BACKTEST_BARS",
    "MAX_BACKTEST_SYMBOLS",
    "MIN_BACKTEST_START",
    "bar_alignment_http_error",
    "bar_alignment_response_fields",
    "check_backtest_request_bounds",
    "cost_profile_http_error",
    "cost_response_fields",
    "engine_semantics",
    "new_run_cost_profile_or_422",
    "requested_bar_window",
    "stamp_cost_config",
    "timestamp_response_fields",
]

#: Version tag of the cost model wiring (``trading.costs``), persisted in
#: ``engine_semantics.cost_model``.
COST_MODEL_VERSION = "wp21-1"
_DAILY_RISK_VERSION = "wp15-1"

#: ``engine_semantics.timestamp_clock`` values that mean "persisted rows carry bar
#: time" (WP2.2, SY-22-08).  A backtest whose stored value is outside this set is
#: legacy (wall-clock stamps).  WP2.3 does NOT bump this version (SY-23-16): the
#: bar-alignment semantics have their own marker, ``engine_semantics.bar_alignment``
#: (:data:`KNOWN_BAR_ALIGNMENTS`), so ``legacy_timestamps`` keeps its WP2.2 meaning.
KNOWN_TIMESTAMP_CLOCKS: frozenset[str] = frozenset({TIMESTAMP_CLOCK_VERSION})

#: ``engine_semantics.bar_alignment`` values that mean "the run paired its symbols'
#: bars by TIMESTAMP" (WP2.3, SY-23-16).  A multi-symbol backtest whose stored value is
#: outside this set paired bars by list index and was silently truncated to the
#: shortest series: legacy.
KNOWN_BAR_ALIGNMENTS: frozenset[str] = frozenset({BAR_ALIGNMENT_VERSION})

#: WP2.3 P0 SEC-02: fixed request bounds for the two endpoints that fetch history
#: (backtest create, optimize).  Code constants, never env or request fields.
#:
#: * ``MAX_BACKTEST_SYMBOLS`` -- SEC-02's cap; far above the 6-pair WP2.8 universe and the
#:   largest basket any test or script in the repo runs.
#: * ``MAX_BACKTEST_BARS`` -- ``ceil((end - start) / tf) * len(symbols)``.  500k bars is
#:   ~57 years of 1h bars for one symbol, ~11 years of 1h for 5 symbols, ~11 months of
#:   5m for 5 symbols or ~347 days of 1m for one: every realistic research window, while
#:   the pager's page cap (``2 * ceil(range / (300 * tf)) + 2`` per symbol) stays at a
#:   few thousand requests instead of millions (SEC-02: 7.1M pages per symbol at 1m
#:   from year 1).  The engine itself is O(n^2) in history slicing (C11), so a run
#:   near this bound is already impractical.
#: * ``MIN_BACKTEST_START`` -- Bitcoin's genesis block; no listed pair has older data,
#:   and it rules out the negative ``since`` the pager would otherwise page from.
MAX_BACKTEST_SYMBOLS: Final = 20
MAX_BACKTEST_BARS: Final = 500_000
MIN_BACKTEST_START: Final = datetime(2009, 1, 3, tzinfo=UTC)

_BAR_ALIGNMENT_ERROR_TYPE: Final = "BarAlignmentError"
_UNKNOWN_BAR_ALIGNMENT_CODE: Final = "bar_alignment_failed"


def cost_profile_http_error(exc: CostProfileError) -> HTTPException:
    """422 with a stable code and NO echo of the requested value."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": exc.code or "invalid_cost_profile"},
    )


def new_run_cost_profile_or_422(requested_id: str | None, *, mode: str) -> CostProfile:
    """Profile for a NEW run (U-2) or an HTTP 422 with the stable code."""
    try:
        return new_run_cost_profile(requested_id, mode=mode)
    except CostProfileError as exc:
        raise cost_profile_http_error(exc) from exc


def engine_semantics(
    mode: str, *, backtest_circuit_breaker: bool = BACKTEST_CIRCUIT_BREAKER_DEFAULT
) -> dict[str, object]:
    """The semantics versions a run starts under.

    ``timestamp_clock`` (WP2.2, SY-22-08) is written on the same two modes and names
    the clock the engine ran with (bar time); it is absent on paper / live, which
    stamp wall time.  Unlike ``backtest_circuit_breaker`` it IS read, by
    :func:`timestamp_response_fields`.

    ``bar_alignment`` (WP2.3, SY-23-16) is written on the same two modes and names the
    bar-pairing semantics (by timestamp, absent symbols skipped); it is read by
    :func:`bar_alignment_response_fields`.  ``timestamp_clock`` is deliberately NOT bumped.

    ``backtest_circuit_breaker`` exists ONLY on backtest runs and optimization
    ``run_metadata`` (the breaker is always on in paper / live, SY-21-11): ``False``
    in WP2.1a / 2.1b, ``True`` from WP2.1c.  The value is the one the engine will
    run with (``BACKTEST_CIRCUIT_BREAKER_DEFAULT``, the same constant behind the
    ``BacktestRunner`` / ``ParameterOptimizer`` defaults); a run row that already
    persisted ``False`` keeps it -- nothing rewrites a stored config, and no reader
    (legacy detection, leaderboard eligibility) looks at this key.
    """
    semantics: dict[str, object] = {
        "schema": 1,
        "cost_model": COST_MODEL_VERSION,
        "exit_semantics": EXIT_SEMANTICS_VERSION,
        "daily_risk": _DAILY_RISK_VERSION,
    }
    if mode in ("backtest", "optimize"):
        semantics["backtest_circuit_breaker"] = backtest_circuit_breaker
        semantics["timestamp_clock"] = TIMESTAMP_CLOCK_VERSION
        # WP2.3 (SY-23-16): bars are paired by timestamp.  Backtest + optimize only;
        # paper / live are out of scope (SY-23-19) and never carry it.
        semantics["bar_alignment"] = BAR_ALIGNMENT_VERSION
    return semantics


def stamp_cost_config(
    config: dict[str, Any],
    profile: CostProfile,
    *,
    mode: str,
    fee_source: FeeSource = "model",
    backtest_circuit_breaker: bool = BACKTEST_CIRCUIT_BREAKER_DEFAULT,
) -> None:
    """Write ``cost_profile`` + ``engine_semantics`` into a NEW run's config.

    OVERWRITES any inherited value (promote must never inherit the paper
    run's profile).  ``backtest_circuit_breaker`` is only written for the
    ``backtest`` / ``optimize`` modes.
    """
    config["cost_profile"] = profile.to_metadata(fee_source=fee_source)
    config["engine_semantics"] = engine_semantics(
        mode, backtest_circuit_breaker=backtest_circuit_breaker
    )


def cost_response_fields(config: Mapping[str, object] | None, run_mode: str) -> dict[str, Any]:
    """``costProfile`` / ``legacyCosts`` / ``costProfileInvalid`` /
    ``legacyExitSemantics`` / ``liveEquivalentCosts`` for one run.

    ``config`` is ``RunORM.config`` for a run and ``OptimizationRunORM.run_metadata``
    for an optimization run (WP2.1b): both carry the same ``cost_profile`` /
    ``engine_semantics`` keys.  ``None``, a non-mapping or a malformed value is
    legacy.  Never raises, never logs a value.

    ``live_equivalent_costs`` is the COST half of ``leaderboard_eligible`` (a valid
    persisted live-equivalent REGISTRY profile that is not legacy), computed by the
    same ``is_leaderboard_cost_eligible`` so the UI's "live-equivalent costs" filter
    can never drift from the server rule.
    """
    mode = run_mode if run_mode in ("backtest", "paper", "live") else "backtest"
    resolved = resolve_run_cost_profile(config, mode=mode, log_invalid=False)
    summary: CostProfileSummary = build_cost_profile_summary(
        resolved.profile, fee_source=resolved.fee_source
    )
    has_semantics = isinstance(config, Mapping) and isinstance(
        config.get("engine_semantics"), Mapping
    )
    return {
        "cost_profile": summary,
        # WP2.1-H (SEC-01): judged in the run's own mode, in step with ``cost_profile_invalid``
        # and the profile shown above (a tampered paper / live row reads as invalid legacy).
        "legacy_costs": is_legacy_cost_config(config, mode=mode),
        "cost_profile_invalid": resolved.invalid,
        "legacy_exit_semantics": not has_semantics,
        "live_equivalent_costs": is_leaderboard_cost_eligible(config),
    }


def timestamp_response_fields(config: Mapping[str, object] | None, run_mode: str) -> dict[str, Any]:
    """``legacyTimestamps`` for one RUN (SY-22-08 / SY-22-09).

    True iff the run is a backtest whose ``engine_semantics.timestamp_clock`` is not a
    known version (missing, unknown, non-string, ``None`` or a malformed
    ``engine_semantics``): its persisted rows carry wall-clock stamps.  Paper and live
    runs always stamp wall time by design, so they are never flagged.  Never raises,
    never logs a value.  Deliberately separate from :func:`cost_response_fields`,
    which also feeds the optimization responses where the flag has no meaning.
    """
    legacy = False
    if run_mode == "backtest":
        semantics = config.get("engine_semantics") if isinstance(config, Mapping) else None
        clock = semantics.get("timestamp_clock") if isinstance(semantics, Mapping) else None
        legacy = not (isinstance(clock, str) and clock in KNOWN_TIMESTAMP_CLOCKS)
    return {"legacy_timestamps": legacy}


def bar_alignment_response_fields(
    config: Mapping[str, object] | None, run_mode: str
) -> dict[str, Any]:
    """``legacy_bar_alignment`` for one RUN (WP2.3, SY-23-16; the SY-22-09 pattern).

    True iff ALL of:

    * the run is a backtest (paper / live are out of scope and never flagged);
    * its symbol count is not exactly one (a missing, non-list, empty, malformed or
      multi-symbol ``config["symbols"]`` fails closed -- a single-symbol legacy run
      has identical engine semantics and is NOT flagged);
    * ``engine_semantics.bar_alignment`` is not a known version (missing, unknown,
      non-string, ``None`` or a malformed ``engine_semantics``).

    Computed server-side from the stored config and never taken from a request.
    Never raises, never logs a value.
    """
    legacy = False
    if run_mode == "backtest":
        symbols = config.get("symbols") if isinstance(config, Mapping) else None
        single = isinstance(symbols, list) and len(symbols) == 1 and isinstance(symbols[0], str)
        if not single:
            semantics = config.get("engine_semantics") if isinstance(config, Mapping) else None
            marker = semantics.get("bar_alignment") if isinstance(semantics, Mapping) else None
            legacy = not (isinstance(marker, str) and marker in KNOWN_BAR_ALIGNMENTS)
    return {"legacy_bar_alignment": legacy}


def bar_alignment_http_error(exc: BarAlignmentError) -> HTTPException:
    """422 ``{"code": <closed code>, "error_type": "BarAlignmentError"}`` (SY-23-08).

    The body is built from the closed ``exc.code`` and a constant only: never
    ``str(exc)``, never a symbol, count or timestamp (CF-36 / I-8).
    """
    code = exc.code if exc.code in BAR_ALIGNMENT_CODES else _UNKNOWN_BAR_ALIGNMENT_CODE
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": code, "error_type": _BAR_ALIGNMENT_ERROR_TYPE},
    )


def _as_utc(ts: datetime) -> datetime:
    """A naive datetime is read as UTC, exactly like ``data.bar_alignment``."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _timeframe_seconds_or_422(timeframe: TimeFrame) -> int:
    try:
        return int(_TIMEFRAME_SECONDS[timeframe])
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "unsupported_timeframe"},
        ) from None


def _bounds_violation(start: datetime, end: datetime, tf_s: int, n_symbols: int) -> str | None:
    """The first violated bound's fixed code, or ``None``.

    A UTC conversion that overflows (``9999-12-31T23:00-12:00``,
    ``0001-01-01T00:00+14:00``) is out of range by definition: ``start_too_early`` for the
    start, ``range_too_large`` for the end (SEC-05), never an unhandled 500.
    """
    if n_symbols > MAX_BACKTEST_SYMBOLS:
        return "too_many_symbols"
    try:
        start_u = _as_utc(start)
    except OverflowError:
        return "start_too_early"
    if start_u < MIN_BACKTEST_START:
        return "start_too_early"
    try:
        end_u = _as_utc(end)
    except OverflowError:
        return "range_too_large"
    span_us = (end_u - start_u) // timedelta(microseconds=1)
    slots = -(-span_us // (tf_s * 1_000_000))
    return None if slots * n_symbols <= MAX_BACKTEST_BARS else "range_too_large"


def check_backtest_request_bounds(
    *, start: datetime, end: datetime, timeframe: TimeFrame, n_symbols: int
) -> None:
    """WP2.3 P0 SEC-02: 422 with a fixed code when a history request is out of bounds.

    Checked in this order, before any bar fetch (a rejection costs nothing):
    ``too_many_symbols`` (> :data:`MAX_BACKTEST_SYMBOLS`), ``start_too_early``
    (< :data:`MIN_BACKTEST_START`), ``range_too_large``
    (``ceil((end - start) / tf) * n_symbols`` > :data:`MAX_BACKTEST_BARS`).  Integer
    arithmetic only; the request value is never echoed.
    """
    tf_s = _timeframe_seconds_or_422(timeframe)
    code = _bounds_violation(start, end, tf_s, n_symbols)
    if code is None:
        return
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": code},
    )


def requested_bar_window(
    start: datetime, end: datetime, timeframe: TimeFrame, now: datetime
) -> tuple[datetime, datetime]:
    """``(start, min(end, floor_tf(now)))`` for ``BacktestRunner.run`` / ``ParameterOptimizer.run``.

    SY-23-14: a request whose ``end`` lies in the future would otherwise fail every
    tail check, and the still-forming bar (open time ``>= floor_tf(now)``) must not be
    simulated.  ``now`` is read by the ROUTER and passed in (WP22-T-06); this function
    is pure.  ``floor_tf`` is the epoch-anchored floor, the same expression the pager
    uses for its own stop (``ccxt_market_data.fetch_ohlcv_range``).
    """
    tf_s = _timeframe_seconds_or_422(timeframe)
    floor_s = (int(_as_utc(now).timestamp()) // tf_s) * tf_s
    floor_dt = datetime.fromtimestamp(floor_s, tz=UTC)
    return _as_utc(start), min(_as_utc(end), floor_dt)
