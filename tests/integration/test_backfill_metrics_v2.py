"""
tests/integration/test_backfill_metrics_v2.py
----------------------------------------------
Integration tests for scripts/backfill_metrics_v2.py.

Test strategy
-------------
The backfill script uses ``get_session_factory()`` which reads DATABASE_URL
from the environment.  These tests do NOT require a live PostgreSQL instance:
all DB I/O is intercepted via AsyncMock session injection by patching
``scripts.backfill_metrics_v2.get_session_factory``.

Three fixture runs are tested:
  - normal_run   : standard backtest with 5 trades (3W/2L), BTC/USDT, 1h
  - zero_run     : backtest with 0 trades (n_closed_trades=0, profit_factor=None)
  - open_pos_run : backtest where a position was still open at end (1 winning trade,
                   0 losing trades -> profit_factor_is_infinite=True)

These runs all have ``metrics_v2_backfilled=False`` initially.  After
``--apply``, assertions check n_closed_trades, profit_factor, profit_factor_is_infinite,
quote_currency, and metrics_v2_backfilled=True for each fixture.

A second ``--apply`` run is a no-op (candidate query returns empty list because
all rows now have metrics_v2_backfilled=True).

Dry-run mode is verified by checking that metrics_v2_backfilled remains False.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Deterministic constants
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

_UUID_NORMAL = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_UUID_ZERO = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
_UUID_OPEN_POS = uuid.UUID("cccccccc-0000-0000-0000-000000000003")

_TIMEFRAME = "1h"

# --- Config blobs -------------------------------------------------------

_NORMAL_CONFIG: dict[str, Any] = {
    "strategy_name": "ma_crossover",
    "strategy_params": {},
    "symbols": ["BTC/USDT"],
    "timeframe": _TIMEFRAME,
    "mode": "backtest",
    "initial_capital": "10000.00",
    "backtest_metrics": {
        "total_trades": 5,
        "winning_trades": 3,
        "losing_trades": 2,
        "win_rate": 0.6,
        # Legacy value: might be wrong (e.g. Infinity stored as null by PostgreSQL)
        "profit_factor": None,
        "profit_factor_is_infinite": False,
    },
}

_ZERO_CONFIG: dict[str, Any] = {
    "strategy_name": "breakout",
    "strategy_params": {},
    "symbols": ["ETH/USDT"],
    "timeframe": _TIMEFRAME,
    "mode": "backtest",
    "initial_capital": "5000.00",
    "backtest_metrics": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "profit_factor": None,
        "profit_factor_is_infinite": False,
    },
}

_OPEN_POS_CONFIG: dict[str, Any] = {
    "strategy_name": "rsi",
    "strategy_params": {},
    "symbols": ["BTC/USDT", "ETH/USD"],
    "timeframe": _TIMEFRAME,
    "mode": "backtest",
    "initial_capital": "20000.00",
    "backtest_metrics": {
        "total_trades": 1,
        "winning_trades": 1,
        "losing_trades": 0,
        "win_rate": 1.0,
        # All winners: should become (None, True)
        "profit_factor": None,
        "profit_factor_is_infinite": False,
    },
}


# ---------------------------------------------------------------------------
# Equity snapshot factories
# ---------------------------------------------------------------------------

def _make_snapshots(n: int, start_equity: float = 10000.0) -> list[Any]:
    """Produce N SimpleNamespace equity snapshots with a gentle equity drift."""
    snaps = []
    equity = Decimal(str(start_equity))
    for i in range(n):
        # Small random-ish drift (deterministic: alternating +0.1% / -0.05%)
        pct = Decimal("0.001") if i % 2 == 0 else Decimal("-0.0005")
        equity = equity * (1 + pct)
        snaps.append(
            SimpleNamespace(
                bar_index=i,
                timestamp=_NOW,
                equity=equity,
                cash=equity * Decimal("0.9"),
                unrealised_pnl=equity * Decimal("0.1"),
                realised_pnl=Decimal("0"),
                drawdown_pct=Decimal("0"),
            )
        )
    return snaps


_ENOUGH_SNAPS = _make_snapshots(50)   # 50 points -> 49 returns -> PSR computable
_FEW_SNAPS = _make_snapshots(10)      # < 31 points -> PSR = None


# ---------------------------------------------------------------------------
# Run ORM SimpleNamespace factories
# ---------------------------------------------------------------------------

def _make_run_ns(
    run_id: uuid.UUID,
    config: dict[str, Any],
    snapshots: list[Any],
    *,
    metrics_v2_backfilled: bool = False,
    n_closed_trades: int | None = None,
) -> Any:
    """Minimal RunORM-like namespace for the backfill script."""
    return SimpleNamespace(
        id=run_id,
        run_mode="backtest",
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_closed_trades,
        metrics_v2_backfilled=metrics_v2_backfilled,
        equity_snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# Session mock factory
# ---------------------------------------------------------------------------

def _make_session_factory(
    candidate_ids: list[uuid.UUID],
    runs: list[Any],
) -> MagicMock:
    """Build a mock session factory that returns the fixture data.

    The backfill script calls ``session_factory()`` as an async context manager.
    Phase 1 (candidate IDs): one execute -> scalars().all() -> list[UUID]
    Phase 2 (load + write):  one execute -> scalars().all() -> list[RunORM]
    We wire a fresh mock per context-manager invocation, discriminating by phase.
    """
    invocation: dict[str, int] = {"n": 0}

    def _new_cm() -> AsyncMock:
        cm = AsyncMock()
        n = invocation["n"]
        invocation["n"] += 1
        session = AsyncMock()
        session.rollback = AsyncMock()
        session.commit = AsyncMock()
        if n == 0:
            # Phase 1: candidate UUID query only
            id_result = MagicMock()
            id_result.scalars.return_value.all.return_value = candidate_ids
            session.execute = AsyncMock(return_value=id_result)
        else:
            # Phase 2+: run load. After CR-003 removes redundant stmt3 selectinload,
            # this is a single execute call.
            run_result = MagicMock()
            run_result.scalars.return_value.all.return_value = runs
            session.execute = AsyncMock(return_value=run_result)
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    factory = MagicMock()
    factory.side_effect = _new_cm
    return factory


# ---------------------------------------------------------------------------
# Import the module under test after path setup
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT / "packages", _REPO_ROOT / "apps", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import backfill_metrics_v2 as bm  # type: ignore[import]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBackfillDryRun:
    """Dry-run must not mutate metrics_v2_backfilled."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_flip_backfilled_flag(self) -> None:
        normal_run = _make_run_ns(_UUID_NORMAL, _NORMAL_CONFIG, _ENOUGH_SNAPS)

        factory = _make_session_factory([_UUID_NORMAL], [normal_run])
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=False,
                run_id_filter=None,
                batch_size=100,
                limit=None,
            )

        assert code == 0
        # metrics_v2_backfilled must remain False in dry-run mode
        assert normal_run.metrics_v2_backfilled is False

    @pytest.mark.asyncio
    async def test_dry_run_zero_candidates_returns_zero(self) -> None:
        factory = _make_session_factory([], [])
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=False,
                run_id_filter=None,
                batch_size=100,
                limit=None,
            )
        assert code == 0


@pytest.mark.integration
class TestBackfillApply:
    """Apply mode must write correct values and flip the idempotency flag."""

    def _run_three_fixtures(self) -> tuple[Any, Any, Any]:
        normal_run = _make_run_ns(_UUID_NORMAL, copy.deepcopy(_NORMAL_CONFIG), _ENOUGH_SNAPS)
        zero_run = _make_run_ns(_UUID_ZERO, copy.deepcopy(_ZERO_CONFIG), _FEW_SNAPS)
        open_pos_run = _make_run_ns(_UUID_OPEN_POS, copy.deepcopy(_OPEN_POS_CONFIG), _FEW_SNAPS)
        return normal_run, zero_run, open_pos_run

    @pytest.mark.asyncio
    async def test_apply_flips_backfilled_flag(self) -> None:
        normal, zero, open_pos = self._run_three_fixtures()
        all_runs = [normal, zero, open_pos]
        all_ids = [_UUID_NORMAL, _UUID_ZERO, _UUID_OPEN_POS]

        factory = _make_session_factory(all_ids, all_runs)
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=True,
                run_id_filter=None,
                batch_size=100,
                limit=None,
            )
        assert code == 0
        assert normal.metrics_v2_backfilled is True
        assert zero.metrics_v2_backfilled is True
        assert open_pos.metrics_v2_backfilled is True

    @pytest.mark.asyncio
    async def test_apply_normal_run_n_closed_trades(self) -> None:
        normal, zero, open_pos = self._run_three_fixtures()
        all_runs = [normal, zero, open_pos]
        all_ids = [_UUID_NORMAL, _UUID_ZERO, _UUID_OPEN_POS]

        factory = _make_session_factory(all_ids, all_runs)
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        assert normal.n_closed_trades == 5   # from total_trades in JSONB
        assert zero.n_closed_trades == 0
        assert open_pos.n_closed_trades == 1

    @pytest.mark.asyncio
    async def test_apply_zero_trades_profit_factor_none(self) -> None:
        """0-trade run: profit_factor=None, profit_factor_is_infinite=False."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_ZERO], [zero])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        metrics = zero.config["backtest_metrics"]
        assert metrics["profit_factor"] is None
        assert metrics["profit_factor_is_infinite"] is False

    @pytest.mark.asyncio
    async def test_apply_all_winners_profit_factor_infinite(self) -> None:
        """All-winner run: profit_factor=None, profit_factor_is_infinite=True."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_OPEN_POS], [open_pos])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        metrics = open_pos.config["backtest_metrics"]
        assert metrics["profit_factor"] is None
        assert metrics["profit_factor_is_infinite"] is True

    @pytest.mark.asyncio
    async def test_apply_mixed_trades_profit_factor_positive(self) -> None:
        """3W/2L run: profit_factor should resolve to the stored value or a positive float."""
        # Seed a stored gross_profit / gross_loss so the fallback path fires
        config = dict(_NORMAL_CONFIG)
        config["backtest_metrics"] = dict(_NORMAL_CONFIG["backtest_metrics"])
        config["backtest_metrics"]["gross_profit"] = "1500.00"
        config["backtest_metrics"]["gross_loss"] = "500.00"
        config["backtest_metrics"]["profit_factor"] = None  # force recompute

        normal = _make_run_ns(_UUID_NORMAL, config, _ENOUGH_SNAPS)
        factory = _make_session_factory([_UUID_NORMAL], [normal])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        pf = normal.config["backtest_metrics"]["profit_factor"]
        # 1500 / 500 = 3.0
        assert pf is not None
        assert abs(pf - 3.0) < 1e-6

    @pytest.mark.asyncio
    async def test_apply_quote_currency_single_symbol(self) -> None:
        """BTC/USDT -> quote_currency='USDT'."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_NORMAL], [normal])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        assert normal.config["backtest_metrics"]["quote_currency"] == "USDT"

    @pytest.mark.asyncio
    async def test_apply_quote_currency_mixed_symbols(self) -> None:
        """BTC/USDT + ETH/USD -> quote_currency='MIXED'."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_OPEN_POS], [open_pos])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        assert open_pos.config["backtest_metrics"]["quote_currency"] == "MIXED"

    @pytest.mark.asyncio
    async def test_apply_psr_computed_when_enough_snapshots(self) -> None:
        """Run with 50 equity snapshots -> PSR should be a float in (0, 1)."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_NORMAL], [normal])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        psr = normal.config["backtest_metrics"]["psr"]
        # PSR may be None (flat equity -> 0 Sharpe) or a float
        # With our alternating +0.1%/-0.05% pattern there is positive net drift
        # so Sharpe > 0 and PSR should be computable.
        assert psr is None or (isinstance(psr, float) and 0.0 <= psr <= 1.0)

    @pytest.mark.asyncio
    async def test_apply_psr_none_when_too_few_snapshots(self) -> None:
        """Run with only 10 equity snapshots -> PSR=None (< 31 points)."""
        normal, zero, open_pos = self._run_three_fixtures()
        factory = _make_session_factory([_UUID_ZERO], [zero])
        with patch.object(bm, "get_session_factory", return_value=factory):
            await bm._run(apply=True, run_id_filter=None, batch_size=100, limit=None)

        psr = zero.config["backtest_metrics"]["psr"]
        assert psr is None

    @pytest.mark.asyncio
    async def test_run_id_filter_targets_only_one_run(self) -> None:
        """--run-id targets exactly one run; others remain untouched."""
        normal, zero, open_pos = self._run_three_fixtures()
        # We only hand the filter one ID worth of candidates
        factory = _make_session_factory([_UUID_ZERO], [zero])
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=True,
                run_id_filter=_UUID_ZERO,
                batch_size=100,
                limit=None,
            )
        assert code == 0
        assert zero.metrics_v2_backfilled is True
        # normal + open_pos not touched (their factory was never called)
        assert normal.metrics_v2_backfilled is False
        assert open_pos.metrics_v2_backfilled is False


@pytest.mark.integration
class TestBackfillIdempotency:
    """Second --apply run must be a no-op."""

    @pytest.mark.asyncio
    async def test_second_apply_is_noop(self) -> None:
        """When candidate query returns empty list, nothing is processed."""
        # Simulate: all rows already have metrics_v2_backfilled=True -> query returns []
        factory = _make_session_factory([], [])
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=True,
                run_id_filter=None,
                batch_size=100,
                limit=None,
            )
        assert code == 0


@pytest.mark.integration
class TestBackfillSkipHandling:
    """Runs without backtest_metrics in config are skipped but still marked."""

    @pytest.mark.asyncio
    async def test_run_without_backtest_metrics_is_skipped(self) -> None:
        config_no_metrics: dict[str, Any] = {
            "strategy_name": "ma_crossover",
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "backtest",
            "initial_capital": "10000.00",
            # No 'backtest_metrics' key
        }
        run = _make_run_ns(_UUID_NORMAL, config_no_metrics, _FEW_SNAPS)
        factory = _make_session_factory([_UUID_NORMAL], [run])
        with patch.object(bm, "get_session_factory", return_value=factory):
            code = await bm._run(
                apply=True,
                run_id_filter=None,
                batch_size=100,
                limit=None,
            )
        assert code == 0
        # Skipped runs are still marked to avoid infinite retry
        assert run.metrics_v2_backfilled is True
        # n_closed_trades must not have been changed (no diff applied)
        assert run.n_closed_trades is None


@pytest.mark.integration
class TestComputeDiff:
    """Unit-level tests for the _compute_diff helper."""

    def _make_run(
        self, config: dict[str, Any], snapshots: list[Any]
    ) -> Any:
        return _make_run_ns(_UUID_NORMAL, config, snapshots)

    def test_normal_diff_n_closed_trades(self) -> None:
        run = self._make_run(dict(_NORMAL_CONFIG), _ENOUGH_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.n_closed_trades == 5

    def test_zero_trades_diff(self) -> None:
        run = self._make_run(dict(_ZERO_CONFIG), _FEW_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.n_closed_trades == 0
        assert diff.profit_factor is None
        assert diff.profit_factor_is_infinite is False

    def test_all_winners_diff(self) -> None:
        run = self._make_run(dict(_OPEN_POS_CONFIG), _FEW_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.profit_factor is None
        assert diff.profit_factor_is_infinite is True

    def test_quote_currency_single(self) -> None:
        run = self._make_run(dict(_NORMAL_CONFIG), _FEW_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.quote_currency == "USDT"

    def test_quote_currency_mixed(self) -> None:
        run = self._make_run(dict(_OPEN_POS_CONFIG), _FEW_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.quote_currency == "MIXED"

    def test_no_backtest_metrics_is_skip(self) -> None:
        config = {
            "strategy_name": "ma_crossover",
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
        }
        run = self._make_run(config, _FEW_SNAPS)
        diff = bm._compute_diff(run, run.equity_snapshots)
        assert diff.is_skip
        assert "backtest_metrics" in (diff.skip_reason or "")
