"""
tests/integration/test_wp15b_resume_flush_watermark.py
--------------------------------------------------------
WP1.5b T-46a (WP15B-CR-01) -- after a resume, the equity point that
``PortfolioAccounting.from_fills`` seeds (positions valued at their OLD fill
prices) is never persisted.

If it were, the WP1.8a incremental flush would write it into
``equity_snapshots`` with the resume's wall-clock timestamp, and a second
restart on the same UTC day would read it back as ``day_first_equity`` /
``day_min_equity`` (WP1.5b hints) -- a false worst-case that blocks BUYs until
00:00 UTC although the real day PnL is zero.

The test drives the REAL ``run_paper_engine`` / ``run_live_engine`` (normal and
protective) through the REAL ``from_fills`` (no spy on it) with a resume
snapshot holding one fill (BUY 25 @ 100, initial cash 10 000, peak hint
11 000), and ends the run right after the resume block (the strategy
constructor raises).  The ``finally`` final flush is intercepted to capture the
``_IncrementalFlushState`` and the portfolio the orchestrator built.  Then a
real mark is applied and the REAL ``flush_incremental`` is run against a
recording session:

* nothing is pending at the resume instant (the seed point is already counted);
* the first persisted equity row is the first real mark (NAV 11 000) with
  ``bar_index == max_bar_index + 1`` -- for ``max_bar_index`` 7 and for the
  empty run (``-1`` -> first row ``0``);
* a second mark continues the sequence at ``+2``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.db.models import EquitySnapshotORM
from common.types import TimeFrame
from tests.integration.test_wp15b_resume_seed_wiring import _session_factory, _settings
from tests.unit._wp15_support import make_fill
from trading.costs import DEFAULT_COST_PROFILE
from trading.recovery import ResumeSnapshot

SYM = "BTC/USD"


class _Stop(Exception):
    """Raised by the strategy constructor to end the coroutine after the resume block."""


def _snapshot(max_bar_index: int) -> ResumeSnapshot:
    return ResumeSnapshot(
        run_id=str(uuid.uuid4()),
        initial_cash=Decimal("10000"),
        fills=[
            make_fill(
                quantity="25",
                price="100",
                symbol=SYM,
                executed_at=datetime.now(UTC) - timedelta(days=5),
            )
        ],
        orders=[],
        peak_equity_hint=Decimal("11000"),
        max_bar_index=max_bar_index,
    )


async def _run_until_final_flush(
    mode: str, snapshot: ResumeSnapshot, *, protective: bool
) -> dict[str, Any]:
    """Run the real coroutine; return the flush ``state`` and ``portfolio`` it built."""
    from api.services import run_orchestrator

    captured: dict[str, Any] = {}

    async def _capture_flush(**kwargs: Any) -> None:
        captured.setdefault("state", kwargs["state"])
        captured.setdefault("portfolio", kwargs["portfolio"])

    common: dict[str, Any] = {
        "run_id_str": snapshot.run_id,
        "strategy_cls": MagicMock(side_effect=_Stop),
        "strategy_name": "grid_trading",
        "strategy_params": {},
        "symbols": [SYM],
        "timeframe": TimeFrame.ONE_HOUR,
        "initial_capital": "10000",
        "cost_profile": DEFAULT_COST_PROFILE,  # WP2.1: required keyword
        "resume": snapshot,
    }
    patches: list[Any] = [
        patch("api.db.session.get_session_factory", _session_factory()),
        patch("api.config.get_settings", return_value=_settings()),
        patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        patch.object(run_orchestrator, "_flush_incremental", _capture_flush),
    ]
    if mode == "paper":
        patches.append(patch.object(run_orchestrator, "_auto_retry_paper_run", AsyncMock()))
        runner = run_orchestrator.run_paper_engine
    else:
        patches.append(patch.object(run_orchestrator, "build_live_ccxt_exchange", MagicMock()))
        patches.append(patch("trading.engines.live.LiveExecutionEngine", MagicMock()))
        runner = run_orchestrator.run_live_engine
        common["protective_mode"] = protective

    for p in patches:
        p.start()
    try:
        await runner(**common)
    finally:
        for p in reversed(patches):
            p.stop()
    assert "state" in captured, f"{mode}: the final flush was never reached"
    return captured


async def _real_flush(
    snapshot: ResumeSnapshot, portfolio: Any, state: Any
) -> list[tuple[int, Decimal]]:
    """Run the REAL ``flush_incremental``; return the persisted ``(bar_index, equity)`` rows."""
    from api.services import run_orchestrator

    added: list[Any] = []
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.merge = AsyncMock()
    db.flush = AsyncMock()
    db.add_all = MagicMock(side_effect=lambda rows: added.extend(rows))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=db)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch(
        "api.db.session.get_session_factory",
        MagicMock(return_value=MagicMock(return_value=ctx)),
    ):
        await run_orchestrator.flush_incremental(
            run_id_str=snapshot.run_id,
            portfolio=portfolio,
            execution_engine=None,
            state=state,
            log=MagicMock(),
        )
    return [(r.bar_index, r.equity) for r in added if isinstance(r, EquitySnapshotORM)]


@pytest.mark.parametrize(
    ("mode", "protective"),
    [("paper", False), ("live", False), ("live", True)],
    ids=["paper", "live", "live-protective"],
)
@pytest.mark.parametrize("max_bar_index", [7, -1], ids=["bar7", "empty-run"])
async def test_wp15_t46a_resume_seed_point_is_never_persisted(
    mode: str, protective: bool, max_bar_index: int
) -> None:
    snapshot = _snapshot(max_bar_index)
    captured = await _run_until_final_flush(mode, snapshot, protective=protective)
    portfolio, state = captured["portfolio"], captured["state"]

    # 1. at the resume instant nothing is pending: the from_fills point (cash +
    #    positions at the OLD fill price = 10 000) is already behind the watermark
    assert portfolio.get_equity_curve()[state.flushed_equity_count :] == []

    # 2. the first real mark (NAV 7 500 + 25 x 140 = 11 000) is the first persisted row
    portfolio.update_market_prices({SYM: Decimal("140")})
    assert await _real_flush(snapshot, portfolio, state) == [(max_bar_index + 1, Decimal("11000"))]

    # 3. the sequence continues without a gap or a collision
    portfolio.update_market_prices({SYM: Decimal("150")})
    assert await _real_flush(snapshot, portfolio, state) == [(max_bar_index + 2, Decimal("11250"))]
