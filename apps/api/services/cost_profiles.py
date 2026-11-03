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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, status

from api.schemas import CostProfileSummary, build_cost_profile_summary
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

__all__ = [
    "COST_MODEL_VERSION",
    "KNOWN_TIMESTAMP_CLOCKS",
    "cost_profile_http_error",
    "cost_response_fields",
    "engine_semantics",
    "new_run_cost_profile_or_422",
    "stamp_cost_config",
    "timestamp_response_fields",
]

#: Version tag of the cost model wiring (``trading.costs``), persisted in
#: ``engine_semantics.cost_model``.
COST_MODEL_VERSION = "wp21-1"
_DAILY_RISK_VERSION = "wp15-1"

#: ``engine_semantics.timestamp_clock`` values that mean "persisted rows carry bar
#: time" (WP2.2, SY-22-08).  A backtest whose stored value is outside this set is
#: legacy (wall-clock stamps).  WP2.3 bumps the version and extends the set.
KNOWN_TIMESTAMP_CLOCKS: frozenset[str] = frozenset({TIMESTAMP_CLOCK_VERSION})


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
