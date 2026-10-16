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

__all__ = [
    "COST_MODEL_VERSION",
    "cost_profile_http_error",
    "cost_response_fields",
    "engine_semantics",
    "new_run_cost_profile_or_422",
    "stamp_cost_config",
]

#: Version tag of the cost model wiring (``trading.costs``), persisted in
#: ``engine_semantics.cost_model``.
COST_MODEL_VERSION = "wp21-1"
_DAILY_RISK_VERSION = "wp15-1"


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


def engine_semantics(mode: str) -> dict[str, object]:
    """The semantics versions a run starts under.

    ``backtest_circuit_breaker`` exists ONLY on backtest runs (the breaker is
    always on in paper / live): ``False`` in WP2.1a, ``True`` from WP2.1c.
    """
    semantics: dict[str, object] = {
        "schema": 1,
        "cost_model": COST_MODEL_VERSION,
        "exit_semantics": EXIT_SEMANTICS_VERSION,
        "daily_risk": _DAILY_RISK_VERSION,
    }
    if mode in ("backtest", "optimize"):
        semantics["backtest_circuit_breaker"] = False
    return semantics


def stamp_cost_config(
    config: dict[str, Any],
    profile: CostProfile,
    *,
    mode: str,
    fee_source: FeeSource = "model",
) -> None:
    """Write ``cost_profile`` + ``engine_semantics`` into a NEW run's config.

    OVERWRITES any inherited value (promote must never inherit the paper
    run's profile).
    """
    config["cost_profile"] = profile.to_metadata(fee_source=fee_source)
    config["engine_semantics"] = engine_semantics(mode)


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
        "legacy_costs": is_legacy_cost_config(config),
        "cost_profile_invalid": resolved.invalid,
        "legacy_exit_semantics": not has_semantics,
        "live_equivalent_costs": is_leaderboard_cost_eligible(config),
    }
