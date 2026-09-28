"""
apps/api/routers/strategies.py
-------------------------------
Strategy discovery endpoints for the AI Crypto Trading Bot API.

Endpoints
---------
GET /api/v1/strategies                -- List all available strategies
GET /api/v1/strategies/{name}/schema  -- Get parameter schema for a strategy

Design notes
------------
- Strategy listing is static — the registry is built at module import time
  from the concrete strategy classes in ``packages/trading/strategies/``.
- No database access is needed for these endpoints; they reflect code-level
  metadata.
- Adding a new strategy only requires registering it in ``_STRATEGY_REGISTRY``
  below (and implementing the class in the strategies package).
- The ``parameter_schema`` endpoint is the authoritative source for the
  frontend's dynamic configuration form builder.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from api.schemas import (
    ErrorResponse,
    StrategyInfoResponse,
    StrategyListResponse,
)
from trading.strategy_availability import StrategyStatus, get_availability

__all__ = ["router"]

router = APIRouter(prefix="/strategies", tags=["strategies"])

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, dict[str, Any]]:
    """
    Build the static strategy registry from concrete strategy classes.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of strategy identifier to its metadata dictionary.
        Keys match what ``RunCreateRequest.strategy_name`` accepts.

    Notes
    -----
    Importing the strategy classes here (at function call time rather than
    module import time) prevents import-time failures when the trading
    package is not installed in the test environment.
    """
    from trading.strategies import (
        BreakoutStrategy,
        DCARSIHybridStrategy,
        GridTradingStrategy,
        MACrossoverStrategy,
        ModelStrategy,
        MomentumBreakoutStrategy,
        RSIMeanReversionStrategy,
        SLTPReversionStrategy,
        SmokeRoundtripStrategy,
    )

    strategies: dict[str, dict[str, Any]] = {}

    entries = [
        ("ma_crossover", MACrossoverStrategy),
        ("rsi_mean_reversion", RSIMeanReversionStrategy),
        ("breakout", BreakoutStrategy),
        ("model_strategy", ModelStrategy),
        ("dca_rsi_hybrid", DCARSIHybridStrategy),
        ("grid_trading", GridTradingStrategy),
        ("sl_tp_reversion", SLTPReversionStrategy),
        ("momentum_breakout", MomentumBreakoutStrategy),
        # WP-SMOKE (reports/vp2-smoke/synthesis-spec.md section 8): a
        # diagnostic mechanics-test instrument, hidden from the default
        # list view (C-5) -- see list_strategies' include_diagnostic filter.
        ("smoke_roundtrip", SmokeRoundtripStrategy),
    ]

    for name, cls in entries:
        meta = cls.metadata
        av = get_availability(name)
        strategies[name] = {
            "name": name,
            "display_name": meta.name,
            "version": meta.version,
            "description": meta.description,
            "tags": list(meta.tags),
            "parameter_schema": cls.parameter_schema(),
            # Sprint 51 Cycle 2: availability metadata
            "allowed_modes": sorted(m.value for m in av.allowed_modes),
            "status": av.status.value,
            "demotion_reason": av.demotion_reason or None,
            "promotion_requirements": list(av.promotion_requirements),
        }

    return strategies


# Lazy-loaded module-level singleton
_REGISTRY: dict[str, dict[str, Any]] | None = None


def _get_registry() -> dict[str, dict[str, Any]]:
    """
    Return the lazy-loaded strategy registry.

    Thread-safety note: This is a module-level singleton. In an async
    FastAPI context there is only one thread per worker, so no locking
    is required.

    Returns
    -------
    dict[str, dict[str, Any]]
        The strategy registry.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def _to_strategy_info(entry: dict[str, Any]) -> StrategyInfoResponse:
    """
    Convert a registry entry dict to a ``StrategyInfoResponse``.

    Parameters
    ----------
    entry:
        Registry entry dictionary.

    Returns
    -------
    StrategyInfoResponse
        API response model.
    """
    return StrategyInfoResponse(
        name=entry["name"],
        display_name=entry["display_name"],
        version=entry["version"],
        description=entry["description"],
        tags=entry["tags"],
        parameter_schema=entry["parameter_schema"],
        allowed_modes=entry["allowed_modes"],
        status=entry["status"],
        demotion_reason=entry["demotion_reason"],
        promotion_requirements=entry["promotion_requirements"],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/strategies — list all available strategies
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=StrategyListResponse,
    summary="List available strategies",
    description=(
        "Returns all strategy implementations available for use in a trading run. "
        "Each entry includes metadata and the JSON Schema for its parameters. "
        "DIAGNOSTIC-status strategies (mechanics-test instruments, no trading "
        "edge) are omitted by default -- pass ?include_diagnostic=true to see them."
    ),
)
async def list_strategies(
    include_diagnostic: bool = Query(
        False,
        description=(
            "Include DIAGNOSTIC-status strategies (e.g. smoke_roundtrip) in the "
            "list. Defaults to False so the UI run-creation form never lists "
            "them by accident (WP-SMOKE, synthesis-spec.md C-5)."
        ),
    ),
) -> StrategyListResponse:
    """
    List all available trading strategies.

    Parameters
    ----------
    include_diagnostic:
        When ``False`` (the default), strategies whose availability status
        is :attr:`StrategyStatus.DIAGNOSTIC` are omitted from the list.

    Returns
    -------
    StrategyListResponse
        All available strategies with their metadata and parameter schemas.
    """
    log = logger.bind(endpoint="list_strategies", include_diagnostic=include_diagnostic)
    log.info("strategies.list_requested")

    registry = _get_registry()
    entries: list[dict[str, Any]] = list(registry.values())
    if not include_diagnostic:
        entries = [
            entry
            for entry in entries
            if entry["status"] != StrategyStatus.DIAGNOSTIC.value
        ]
    strategies = [_to_strategy_info(entry) for entry in entries]

    log.info("strategies.listed", total=len(strategies))

    return StrategyListResponse(
        strategies=strategies,
        total=len(strategies),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/strategies/{name}/schema — get parameter schema
# ---------------------------------------------------------------------------

@router.get(
    "/{name}/schema",
    response_model=StrategyInfoResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Strategy not found"},
    },
    summary="Get parameter schema for a strategy",
    description=(
        "Returns the full metadata and JSON Schema for the named strategy. "
        "Use this to build dynamic parameter configuration forms. "
        "The ``parameterSchema`` field is a valid JSON Schema object."
    ),
)
async def get_strategy_schema(
    name: str,
) -> StrategyInfoResponse:
    """
    Get parameter schema for a named strategy.

    Parameters
    ----------
    name:
        Strategy identifier, e.g. "ma_crossover", "rsi_mean_reversion",
        "breakout".

    Returns
    -------
    StrategyInfoResponse
        Strategy metadata and parameter JSON Schema.

    Raises
    ------
    HTTPException 404:
        When no strategy with the given name is registered.
    """
    log = logger.bind(endpoint="get_strategy_schema", name=name)
    log.info("strategies.schema_requested")

    registry = _get_registry()
    normalised_name = name.lower().replace("-", "_")
    entry = registry.get(normalised_name)

    if entry is None:
        log.warning("strategies.not_found", name=name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Strategy {name!r} not found. "
                f"Available: {sorted(registry.keys())}"
            ),
        )

    log.info("strategies.schema_returned", name=normalised_name)
    return _to_strategy_info(entry)
