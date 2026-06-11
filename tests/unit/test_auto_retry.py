"""
tests/unit/test_auto_retry.py
------------------------------
Unit tests for the crashed-paper-run auto-retry (Sprint 48 D2 design,
implemented in apps/api/services/run_orchestrator.py).

Covers: attempt-exhaustion guard, exponential backoff values, new-run
creation linked via recovered_from_run_id with an incremented attempt
counter, source-run-missing guard, and the fire-and-forget error guard.

Async note: pyproject sets asyncio_mode = "auto"; no marks needed.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.run_orchestrator import (
    _AUTO_RETRY_BASE_DELAY_SECONDS,
    _AUTO_RETRY_MAX_ATTEMPTS,
    _RUN_TASKS,
    _auto_retry_paper_run,
)
from common.types import TimeFrame


@pytest.fixture(autouse=True)
def _run_tasks_isolation() -> Iterator[None]:
    """Restore the module-global _RUN_TASKS registry after every test.

    The production code registers fire-and-forget engine tasks in this dict;
    without unconditional teardown a failing test would leak entries into
    later tests (CR-007 from the auto-retry review).
    """
    before = dict(_RUN_TASKS)
    try:
        yield
    finally:
        _RUN_TASKS.clear()
        _RUN_TASKS.update(before)


def _retry_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "crashed_run_id": str(uuid.uuid4()),
        "auto_retry_attempt": 0,
        "strategy_cls": MagicMock,
        "strategy_name": "grid_trading",
        "strategy_params": {"position_size": 100},
        "symbols": ["BTC/EUR"],
        "timeframe": TimeFrame.ONE_DAY,
        "initial_capital": "10000",
        "trailing_stop_pct": None,
        "bracket_config": {"bracket_mode": "atr", "bracket_atr_sl_multiplier": 2.0},
        "enable_adaptive_learning": False,
        "auto_apply_learning": False,
    }
    base.update(overrides)
    return base


def _mock_session_factory(crashed_run: Any) -> tuple[MagicMock, AsyncMock]:
    """Session factory whose SELECT returns ``crashed_run``."""
    select_result = MagicMock()
    select_result.scalar_one_or_none = MagicMock(return_value=crashed_run)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=select_result)
    db.add = MagicMock()
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=db)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=session_ctx)
    return factory, db


class TestAutoRetryGuards:
    async def test_exhausted_attempts_do_not_retry(self) -> None:
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await _auto_retry_paper_run(
                **_retry_kwargs(auto_retry_attempt=_AUTO_RETRY_MAX_ATTEMPTS)
            )
        # Exhausted before the backoff: no sleep, no run created.
        mock_sleep.assert_not_awaited()

    async def test_source_run_missing_aborts(self) -> None:
        factory, db = _mock_session_factory(crashed_run=None)
        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch("api.db.session.get_session_factory", return_value=factory),
        ):
            await _auto_retry_paper_run(**_retry_kwargs())
        db.add.assert_not_called()

    async def test_superseded_run_not_retried(self) -> None:
        # CR-001/CR-008: operator restarted the run during the backoff —
        # the crashed row is no longer status='error', so the retry must
        # abort instead of double-starting the strategy.
        crashed = MagicMock()
        crashed.id = uuid.uuid4()
        crashed.status = "running"  # operator already restarted it
        crashed.config = {}
        factory, db = _mock_session_factory(crashed)
        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch("api.db.session.get_session_factory", return_value=factory),
        ):
            await _auto_retry_paper_run(
                **_retry_kwargs(crashed_run_id=str(crashed.id))
            )
        db.add.assert_not_called()

    async def test_db_error_is_swallowed(self) -> None:
        # A failing retry must never propagate (fire-and-forget guard).
        factory = MagicMock(side_effect=RuntimeError("db down"))
        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch("api.db.session.get_session_factory", return_value=factory),
        ):
            await _auto_retry_paper_run(**_retry_kwargs())  # must not raise


class TestAutoRetryHappyPath:
    async def test_creates_linked_run_with_incremented_attempt(self) -> None:
        crashed_id = uuid.uuid4()
        crashed = MagicMock()
        crashed.id = crashed_id
        crashed.status = "error"
        crashed.config = {
            "strategy_name": "grid_trading",
            "symbols": ["BTC/EUR"],
            "timeframe": "1d",
        }
        factory, db = _mock_session_factory(crashed)

        captured: dict[str, Any] = {}

        async def _fake_engine(**kwargs: Any) -> None:
            captured.update(kwargs)

        with (
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch("api.db.session.get_session_factory", return_value=factory),
            patch(
                "api.services.run_orchestrator.run_paper_engine",
                new=_fake_engine,
            ),
        ):
            await _auto_retry_paper_run(
                **_retry_kwargs(crashed_run_id=str(crashed_id), auto_retry_attempt=1)
            )

        # The relaunched engine runs as a fire-and-forget task; give the
        # event loop a tick (outside the patched-sleep context) to execute it.
        import asyncio as _aio

        await _aio.sleep(0)

        # Backoff for attempt 1 -> base * 2^1.
        mock_sleep.assert_awaited_once_with(_AUTO_RETRY_BASE_DELAY_SECONDS * 2)

        # New RunORM row: paper, running, linked, attempt incremented.
        db.add.assert_called_once()
        new_run = db.add.call_args.args[0]
        assert new_run.run_mode == "paper"
        assert new_run.status == "running"
        assert new_run.recovered_from_run_id == crashed_id
        assert new_run.config["auto_retry_attempt"] == 2
        assert new_run.config["auto_retried_from"] == str(crashed_id)
        # Original config keys preserved.
        assert new_run.config["strategy_name"] == "grid_trading"
        db.commit.assert_awaited()

        # Engine relaunched with the incremented attempt + registered task.
        assert captured["auto_retry_attempt"] == 2
        assert captured["strategy_name"] == "grid_trading"
        new_id = captured["run_id_str"]
        assert new_id in _RUN_TASKS
        # Cleanup the registered (already-finished) task entry.
        _RUN_TASKS.pop(new_id, None)

    async def test_first_retry_backoff_is_base_delay(self) -> None:
        crashed = MagicMock()
        crashed.id = uuid.uuid4()
        crashed.status = "error"
        crashed.config = {}
        factory, _db = _mock_session_factory(crashed)

        async def _fake_engine(**kwargs: Any) -> None:
            _RUN_TASKS.pop(kwargs["run_id_str"], None)

        with (
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch("api.db.session.get_session_factory", return_value=factory),
            patch(
                "api.services.run_orchestrator.run_paper_engine",
                new=_fake_engine,
            ),
        ):
            await _auto_retry_paper_run(
                **_retry_kwargs(crashed_run_id=str(crashed.id), auto_retry_attempt=0)
            )
        import asyncio as _aio

        await _aio.sleep(0)  # let the fire-and-forget engine task tick + clean up
        mock_sleep.assert_awaited_once_with(_AUTO_RETRY_BASE_DELAY_SECONDS)
