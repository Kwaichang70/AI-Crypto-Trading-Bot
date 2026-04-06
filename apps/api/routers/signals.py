"""
apps/api/routers/signals.py
-----------------------------
GET /api/v1/signals/current — current cached values for all market signal sources.

This endpoint is intentionally unauthenticated (no API key required) because
it returns read-only, publicly-sourced market data that was already fetched by
the background startup tasks.  It never touches the database and never
triggers a new outbound HTTP request — it only reads in-process cache values.

Signal sources
--------------
- Fear & Greed Index       — data.sentiment.FearGreedClient
- CoinGecko global market  — data.market_signals.CoinGeckoClient
- FRED macro data          — data.macro_data.FREDClient
- Whale Alert on-chain     — data.whale_tracker.WhaleAlertClient

All four clients are started at API boot time (see apps/api/main.py lifespan).
Each field in the response is ``null`` when the corresponding client is not
configured, when no cache has been populated yet, or when the upstream API
was unreachable on the last refresh attempt.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter

__all__ = ["router"]

router = APIRouter(prefix="/signals", tags=["signals"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# GET /api/v1/signals/current
# ---------------------------------------------------------------------------


@router.get(
    "/current",
    summary="Current market signal values",
    description=(
        "Returns the most recent cached value from every market signal source "
        "(Fear & Greed, CoinGecko, FRED, Whale Alert).  Fields are null when "
        "the source is not configured or has not yet populated its cache."
    ),
)
async def get_current_signals() -> dict[str, Any]:
    """Return current cached values for all market signal sources."""
    result: dict[str, Any] = {
        "fearGreedIndex": None,
        "fearGreedClassification": None,
        "btcDominance": None,
        "marketCapChange24h": None,
        "totalVolumeChange24h": None,
        "fedFundsRate": None,
        "yieldCurveSpread": None,
        "whaleNetFlow": None,
        "whaleTxCount": None,
    }

    # ------------------------------------------------------------------
    # Fear & Greed Index (data.sentiment.FearGreedClient)
    # ------------------------------------------------------------------
    try:
        from data.sentiment import get_global_client as _get_fgi

        fgi_client = _get_fgi()
        if fgi_client is not None and fgi_client.cached_value is not None:
            result["fearGreedIndex"] = fgi_client.cached_value
            # _latest_cache is a (FearGreedSnapshot, float) tuple; index 0 is
            # the snapshot which carries the human-readable classification.
            if fgi_client._latest_cache is not None:
                result["fearGreedClassification"] = fgi_client._latest_cache[0].classification
    except Exception:
        logger.debug("signals.fgi_read_failed", exc_info=True)

    # ------------------------------------------------------------------
    # CoinGecko global market (data.market_signals.CoinGeckoClient)
    # ------------------------------------------------------------------
    try:
        from data.market_signals import get_global_client as _get_cg

        cg_client = _get_cg()
        if cg_client is not None:
            cg_snap = cg_client.cached_value
            if cg_snap is not None:
                result["btcDominance"] = round(cg_snap.btc_dominance, 2)
                result["marketCapChange24h"] = round(cg_snap.market_cap_change_24h, 2)
                result["totalVolumeChange24h"] = round(cg_snap.total_volume_change_24h, 2)
    except Exception:
        logger.debug("signals.coingecko_read_failed", exc_info=True)

    # ------------------------------------------------------------------
    # FRED macro data (data.macro_data.FREDClient)
    # ------------------------------------------------------------------
    try:
        from data.macro_data import get_global_client as _get_fred

        fred_client = _get_fred()
        if fred_client is not None:
            fred_snap = fred_client.cached_value
            if fred_snap is not None:
                result["fedFundsRate"] = fred_snap.fed_funds_rate
                result["yieldCurveSpread"] = fred_snap.yield_curve_spread
    except Exception:
        logger.debug("signals.fred_read_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Whale Alert on-chain flow (data.whale_tracker.WhaleAlertClient)
    # ------------------------------------------------------------------
    try:
        from data.whale_tracker import get_global_client as _get_whale

        whale_client = _get_whale()
        if whale_client is not None:
            whale_snap = whale_client.cached_value
            if whale_snap is not None:
                result["whaleNetFlow"] = round(whale_snap.net_flow, 0)
                result["whaleTxCount"] = whale_snap.large_tx_count
    except Exception:
        logger.debug("signals.whale_read_failed", exc_info=True)

    logger.debug(
        "signals.current_fetched",
        fgi=result["fearGreedIndex"],
        btc_dominance=result["btcDominance"],
        fed_funds_rate=result["fedFundsRate"],
        whale_net_flow=result["whaleNetFlow"],
    )
    return result
