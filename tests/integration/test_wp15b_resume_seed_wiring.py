"""
tests/integration/test_wp15b_resume_seed_wiring.py
----------------------------------------------------
WP1.5b T-33 -- both orchestrator resume paths (``run_paper_engine`` and
``run_live_engine``) hand ``resume.daily_seed`` to
``PortfolioAccounting.from_fills``.

The test drives the REAL coroutines up to the ``from_fills`` call with the
exchange / market-data / session-factory seams mocked, and replaces
``PortfolioAccounting.from_fills`` with a spy that records its keyword
arguments and then aborts the run (the coroutines end in ``error``, which is
irrelevant here).  A legacy snapshot (hints never read) must pass
``daily_seed=None`` -- the 1.5a ``replay_estimate`` path -- and a snapshot that
DID read hints must never pass ``None``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.types import TimeFrame
from trading.portfolio import PortfolioAccounting
from trading.recovery import ResumeDailySeed, ResumeSnapshot

_TODAY = datetime(2026, 3, 10, tzinfo=UTC).date()
_STARTED = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


class _Stop(Exception):
    """Raised by the spy to end the coroutine right after ``from_fills``."""


def _session_factory() -> MagicMock:
    run = MagicMock()
    run.status = "running"
    result = MagicMock()
    result.scalar_one_or_none.return_value = run
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=db)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=MagicMock(return_value=ctx))


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        exchange_api_key=None,
        exchange_api_secret=None,
        exchange_api_passphrase=None,
        exchange_id="coinbase",
    )


def _snapshot(*, with_hints: bool) -> ResumeSnapshot:
    kwargs: dict[str, Any] = {}
    if with_hints:
        kwargs = {
            "day_start_equity_hint": Decimal("10000"),
            "day_min_equity_hint": Decimal("9150"),
            "day_first_equity_hint": Decimal("10000"),
            "daily_hints_day": _TODAY,
            "run_started_at": _STARTED,
        }
    return ResumeSnapshot(
        run_id=str(uuid.uuid4()),
        initial_cash=Decimal("10000"),
        fills=[],
        orders=[],
        peak_equity_hint=Decimal("10100"),
        max_bar_index=7,
        **kwargs,
    )


async def _run_until_from_fills(
    mode: str, snapshot: ResumeSnapshot, *, protective: bool = False
) -> dict[str, Any]:
    from api.services import run_orchestrator

    recorded: dict[str, Any] = {}

    def _spy(cls: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.update(kwargs)
        raise _Stop

    common: dict[str, Any] = {
        "run_id_str": snapshot.run_id,
        "strategy_cls": MagicMock(),
        "strategy_name": "grid_trading",
        "strategy_params": {},
        "symbols": ["BTC/USD"],
        "timeframe": TimeFrame.ONE_HOUR,
        "initial_capital": "10000",
        "resume": snapshot,
    }
    patches = [
        patch("api.db.session.get_session_factory", _session_factory()),
        patch("api.config.get_settings", return_value=_settings()),
        patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        patch.object(PortfolioAccounting, "from_fills", classmethod(_spy)),
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
    assert recorded, f"{mode}: from_fills was never reached"
    return recorded


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_wp15_t33_resume_path_passes_the_daily_seed_to_from_fills(mode: str) -> None:
    snapshot = _snapshot(with_hints=True)
    kwargs = await _run_until_from_fills(mode, snapshot)

    seed = kwargs["daily_seed"]
    assert isinstance(seed, ResumeDailySeed)
    assert seed == snapshot.daily_seed
    assert seed.as_of_day == _TODAY
    assert seed.day_min_equity == Decimal("9150")
    assert seed.day_first_equity == Decimal("10000")
    assert seed.run_started_at == _STARTED
    # the pre-existing arguments are unchanged
    assert kwargs["peak_equity_hint"] == Decimal("10100")
    assert kwargs["initial_cash"] == Decimal("10000")


async def test_wp15_t33_protective_live_resume_passes_the_seed_too() -> None:
    snapshot = _snapshot(with_hints=True)
    kwargs = await _run_until_from_fills("live", snapshot, protective=True)
    assert kwargs["daily_seed"] == snapshot.daily_seed
    assert kwargs["daily_seed"] is not None


@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_wp15_t33_legacy_snapshot_without_hints_passes_none(mode: str) -> None:
    """Hints never read -> ``None`` -> the 1.5a ``replay_estimate`` path."""
    kwargs = await _run_until_from_fills(mode, _snapshot(with_hints=False))
    assert "daily_seed" in kwargs and kwargs["daily_seed"] is None
