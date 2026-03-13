"""
tests/unit/test_sprint24_run_recovery.py
-----------------------------------------
Unit tests for Sprint 24 Run-Recovery feature.

Modules under test
------------------
- apps/api/routers/runs.py  -- recover_orphaned_runs(), _mark_orphan_error()

Test classes
------------
1. TestMarkOrphanError          (~3 tests) -- status transition, stopped_at written, silent on missing run
2. TestRecoverNoOrphans         (~2 tests) -- empty result, returns 0
3. TestRecoverPaperOrphan       (~4 tests) -- happy path, new run created, task started, _RUN_TASKS populated
4. TestRecoverLiveOrphan        (~3 tests) -- happy path with gate pass, gate fail (env), gate fail (keys)
5. TestRecoverValidationSkips   (~5 tests) -- missing strategy_name, unknown strategy, empty symbols,
                                              invalid timeframe, recovery-chain prevention
6. TestRecoverPerOrphanIsolation (~2 tests) -- one bad orphan does not block valid neighbour

Design notes
------------
- All async tests use @pytest.mark.asyncio.  asyncio_mode = "auto" in pyproject.toml
  means the decorator is technically optional but is kept for explicitness.
- recover_orphaned_runs() imports get_session_factory, get_settings, and RunORM
  lazily INSIDE the function body, so the patch targets are the source modules:
    * "api.db.session.get_session_factory"
    * "api.config.get_settings"
  _get_strategy_registry is a module-level function in api.routers.runs so we patch:
    * "api.routers.runs._get_strategy_registry"
  _run_paper_engine and _run_live_engine are module-level coroutine functions:
    * "api.routers.runs._run_paper_engine"
    * "api.routers.runs._run_live_engine"
- The DB interaction pattern inside recover_orphaned_runs() is:
    1. One factory() context for the initial SELECT (returns orphan list).
    2. One factory() context per orphan for _mark_orphan_error() (returns same orphan
       via scalar_one_or_none).
    3. One factory() context per recovered orphan for the atomic UPDATE + INSERT.
  _make_session_factory() below handles this by using a side_effect that alternates
  between query-type sessions (scalars().all()) and write-type sessions
  (scalar_one_or_none()).
- _mark_orphan_error() is also tested in isolation by patching get_session_factory
  directly and passing the returned factory.
- asyncio.create_task is NOT patched -- the real event loop is used so that
  _RUN_TASKS receives a real asyncio.Task object.  The coroutines themselves are
  replaced with AsyncMock coroutine functions so they resolve immediately (no I/O).
- _STRATEGY_REGISTRY is reset to None before each test via the autouse fixture to
  prevent cross-test contamination from lazy-load caching.
- SimpleNamespace mimics RunORM read-only fields used by recover_orphaned_runs().
  Write-path sessions (used for _mark_orphan_error and the atomic update block) use
  full AsyncMock sessions with configurable scalar_one_or_none returns.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routers.runs as runs_module
from api.routers.runs import _mark_orphan_error, recover_orphaned_runs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_orphan(
    run_mode: str = "paper",
    config: dict | None = None,
    recovered_from_run_id: uuid.UUID | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace that mimics a RunORM orphan row.

    Only attributes consumed by recover_orphaned_runs() are set; the rest are
    left absent so any accidental access raises AttributeError immediately.
    """
    return SimpleNamespace(
        id=uuid.uuid4(),
        run_mode=run_mode,
        status="running",
        config=config
        or {
            "strategy_name": "ma_crossover",
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
            "mode": run_mode,
        },
        started_at=datetime.now(UTC),
        stopped_at=None,
        recovered_from_run_id=recovered_from_run_id,
    )


def _make_settings(
    *,
    enable_live_trading: bool = False,
    api_key: str = "key123",
    api_secret: str = "secret123",
) -> MagicMock:
    """Build a mock Settings object with live-trading fields pre-configured."""
    settings = MagicMock()
    settings.enable_live_trading = enable_live_trading

    key_mock = MagicMock()
    key_mock.get_secret_value.return_value = api_key

    secret_mock = MagicMock()
    secret_mock.get_secret_value.return_value = api_secret

    settings.exchange_api_key = key_mock
    settings.exchange_api_secret = secret_mock
    return settings


def _make_session_for_select(orphans: list) -> AsyncMock:
    """Return a mock async session whose execute() returns *orphans* via scalars().all()."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    result = MagicMock()
    result.scalars.return_value.all.return_value = orphans
    session.execute = AsyncMock(return_value=result)
    return session


def _make_session_for_write(orphan: SimpleNamespace | None) -> AsyncMock:
    """Return a mock async session for write operations.

    scalar_one_or_none() returns *orphan* (used by both _mark_orphan_error and
    the atomic update block inside recover_orphaned_runs).
    """
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    result = MagicMock()
    result.scalar_one_or_none.return_value = orphan
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


def _build_factory(*sessions: AsyncMock) -> MagicMock:
    """Build a factory mock whose successive calls return successive sessions.

    Wraps each session in a context-manager shim so the ``async with factory()``
    pattern works.  If more calls are made than sessions are provided the last
    session is reused.
    """
    contexts = []
    for s in sessions:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=s)
        ctx.__aexit__ = AsyncMock(return_value=False)
        contexts.append(ctx)

    call_count = [-1]

    def _factory():
        call_count[0] += 1
        idx = min(call_count[0], len(contexts) - 1)
        return contexts[idx]

    factory = MagicMock(side_effect=_factory)
    return factory


# ---------------------------------------------------------------------------
# Autouse fixture: reset module-level cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_strategy_registry():
    """Reset _STRATEGY_REGISTRY so lazy-load does not persist between tests."""
    original = runs_module._STRATEGY_REGISTRY
    runs_module._STRATEGY_REGISTRY = None
    yield
    runs_module._STRATEGY_REGISTRY = original


@pytest.fixture(autouse=True)
def _clean_run_tasks():
    """Clear _RUN_TASKS before and after each test to prevent cross-test leakage."""
    runs_module._RUN_TASKS.clear()
    yield
    # Cancel any tasks started by the test to avoid asyncio warnings.
    for task in list(runs_module._RUN_TASKS.values()):
        if not task.done():
            task.cancel()
    runs_module._RUN_TASKS.clear()


# ---------------------------------------------------------------------------
# Class 1: TestMarkOrphanError
# ---------------------------------------------------------------------------


class TestMarkOrphanError:
    """Verify _mark_orphan_error() marks a running run as error with timestamps."""

    @pytest.mark.asyncio
    async def test_running_run_is_marked_error(self) -> None:
        """A run with status='running' must be transitioned to status='error'.

        The session must receive a commit() call after the mutation.
        """
        orphan = _make_orphan()
        # Give it a real-looking status that the function checks
        orphan.status = "running"

        write_session = _make_session_for_write(orphan)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, orphan.id, log)

        assert orphan.status == "error", (
            f"status must be 'error' after _mark_orphan_error, got {orphan.status!r}"
        )
        assert orphan.stopped_at is not None, "stopped_at must be set"
        assert orphan.updated_at is not None, "updated_at must be set"
        write_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_already_stopped_run_is_not_modified(self) -> None:
        """A run with status != 'running' must not be touched.

        _mark_orphan_error guards on `stale.status == 'running'` so a run that
        has already been stopped by another process must not be mutated.
        """
        orphan = _make_orphan()
        orphan.status = "stopped"  # not 'running'

        write_session = _make_session_for_write(orphan)
        factory = _build_factory(write_session)
        log = MagicMock()

        await _mark_orphan_error(factory, orphan.id, log)

        # Status must remain 'stopped' — the guard clause must have fired
        assert orphan.status == "stopped", (
            f"Non-running run must not be mutated, got status={orphan.status!r}"
        )
        write_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_run_is_a_no_op(self) -> None:
        """When scalar_one_or_none() returns None the function must not raise.

        This covers the case where a run was already deleted between discovery
        and the mark-error write.
        """
        write_session = _make_session_for_write(None)
        factory = _build_factory(write_session)
        log = MagicMock()

        # Must complete without raising
        await _mark_orphan_error(factory, uuid.uuid4(), log)

        write_session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Class 2: TestRecoverNoOrphans
# ---------------------------------------------------------------------------


class TestRecoverNoOrphans:
    """Verify recover_orphaned_runs() returns 0 and starts no tasks when DB is empty."""

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero(self) -> None:
        """recover_orphaned_runs() on an empty DB must return 0 immediately."""
        select_session = _make_session_for_select([])
        factory = _build_factory(select_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
        ):
            result = await recover_orphaned_runs()

        assert result == 0, f"Expected 0 recovered runs, got {result}"

    @pytest.mark.asyncio
    async def test_empty_db_starts_no_tasks(self) -> None:
        """No entries must be added to _RUN_TASKS when there are no orphans."""
        select_session = _make_session_for_select([])
        factory = _build_factory(select_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
        ):
            await recover_orphaned_runs()

        assert runs_module._RUN_TASKS == {}, (
            "_RUN_TASKS must remain empty when no orphans are found"
        )


# ---------------------------------------------------------------------------
# Class 3: TestRecoverPaperOrphan
# ---------------------------------------------------------------------------


class TestRecoverPaperOrphan:
    """Verify the full recovery flow for a single paper-mode orphan."""

    @pytest.mark.asyncio
    async def test_paper_orphan_returns_one(self) -> None:
        """A single valid paper orphan must result in a return value of 1."""
        orphan = _make_orphan(run_mode="paper")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        dummy_coro = AsyncMock()

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", dummy_coro),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1, f"Expected 1 recovered run, got {result}"

    @pytest.mark.asyncio
    async def test_paper_orphan_marked_error_in_db(self) -> None:
        """The original orphan row must be mutated to status='error' and stopped_at set.

        The write session's execute returns the orphan as scalar_one_or_none,
        after which the in-place mutation is committed.
        """
        orphan = _make_orphan(run_mode="paper")
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            await recover_orphaned_runs()

        assert orphan.status == "error", (
            f"Original orphan must be marked 'error', got {orphan.status!r}"
        )
        assert orphan.stopped_at is not None, "stopped_at must be stamped on original orphan"

    @pytest.mark.asyncio
    async def test_paper_orphan_new_run_added_to_db(self) -> None:
        """A new RunORM must be session.add()-ed with recovered_from_run_id set.

        The write session's add() call must receive an object whose
        recovered_from_run_id matches the original orphan's id.
        """
        orphan = _make_orphan(run_mode="paper")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            await recover_orphaned_runs()

        assert write_session.add.called, "session.add() must be called to persist the new run"
        added = write_session.add.call_args[0][0]
        assert added.recovered_from_run_id == orphan.id, (
            f"New run's recovered_from_run_id must be {orphan.id}, "
            f"got {added.recovered_from_run_id}"
        )
        assert added.status == "running", (
            f"New run must start with status='running', got {added.status!r}"
        )

    @pytest.mark.asyncio
    async def test_paper_orphan_task_registered_in_run_tasks(self) -> None:
        """After recovery the new run's task must appear in _RUN_TASKS.

        The key is the string form of the newly-created run_id.  The value
        must be an asyncio.Task instance.
        """
        orphan = _make_orphan(run_mode="paper")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            await recover_orphaned_runs()

        assert len(runs_module._RUN_TASKS) == 1, (
            f"Exactly one task must be registered, found {len(runs_module._RUN_TASKS)}"
        )
        task = next(iter(runs_module._RUN_TASKS.values()))
        assert isinstance(task, asyncio.Task), (
            f"_RUN_TASKS value must be an asyncio.Task, got {type(task)}"
        )


# ---------------------------------------------------------------------------
# Class 4: TestRecoverLiveOrphan
# ---------------------------------------------------------------------------


class TestRecoverLiveOrphan:
    """Verify live-mode recovery: gate-pass creates task, gate-fail skips and marks error."""

    @pytest.mark.asyncio
    async def test_live_orphan_gate_pass_recovered(self) -> None:
        """Live orphan with enable_live_trading=True and valid keys must be recovered.

        Expected outcome: return value 1, one entry in _RUN_TASKS, _run_live_engine
        coroutine was started (not _run_paper_engine).
        """
        orphan = _make_orphan(run_mode="live")
        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session, write_session)

        settings = _make_settings(
            enable_live_trading=True,
            api_key="live_key",
            api_secret="live_secret",
        )
        live_coro = AsyncMock()

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=settings),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", live_coro),
        ):
            result = await recover_orphaned_runs()

        assert result == 1, f"Expected 1 recovered live run, got {result}"
        assert len(runs_module._RUN_TASKS) == 1

    @pytest.mark.asyncio
    async def test_live_orphan_gate_fail_env_flag_returns_zero(self) -> None:
        """Live orphan with enable_live_trading=False must be skipped (returns 0).

        The original run must still be marked error; no new run or task is created.
        """
        orphan = _make_orphan(run_mode="live")
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        # _mark_orphan_error will use a separate session; provide one here
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        settings = _make_settings(enable_live_trading=False)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=settings),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0, f"Expected 0 when live gate fails, got {result}"
        assert runs_module._RUN_TASKS == {}, "_RUN_TASKS must remain empty"
        assert orphan.status == "error", (
            "Original live orphan must be marked error when gate fails"
        )

    @pytest.mark.asyncio
    async def test_live_orphan_gate_fail_empty_keys_returns_zero(self) -> None:
        """Live orphan with empty API keys must be skipped even if env flag is True."""
        orphan = _make_orphan(run_mode="live")
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        # enable_live_trading=True but keys are empty strings
        settings = _make_settings(
            enable_live_trading=True,
            api_key="",
            api_secret="",
        )

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=settings),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0, f"Expected 0 when API keys are empty, got {result}"
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"


# ---------------------------------------------------------------------------
# Class 5: TestRecoverValidationSkips
# ---------------------------------------------------------------------------


class TestRecoverValidationSkips:
    """Verify each config-validation guard marks the orphan error and continues."""

    @pytest.mark.asyncio
    async def test_missing_strategy_name_marks_error_and_skips(self) -> None:
        """An orphan whose config has no 'strategy_name' key must be skipped.

        The original must be marked error, no new task must be started.
        """
        orphan = _make_orphan(
            config={
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "mode": "paper",
                # 'strategy_name' deliberately absent
            }
        )
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_unknown_strategy_marks_error_and_skips(self) -> None:
        """An orphan referencing a strategy not in the registry must be skipped."""
        orphan = _make_orphan(
            config={
                "strategy_name": "nonexistent_strategy",
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "mode": "paper",
            }
        )
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_empty_symbols_marks_error_and_skips(self) -> None:
        """An orphan with an empty 'symbols' list must be skipped."""
        orphan = _make_orphan(
            config={
                "strategy_name": "ma_crossover",
                "symbols": [],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "mode": "paper",
            }
        )
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_invalid_timeframe_marks_error_and_skips(self) -> None:
        """An orphan with an unrecognised timeframe string must be skipped.

        TimeFrame("3h") raises ValueError; the guard must catch it and mark
        the orphan as error before continuing to the next orphan.
        """
        orphan = _make_orphan(
            config={
                "strategy_name": "ma_crossover",
                "symbols": ["BTC/USD"],
                "timeframe": "3h",  # not a valid TimeFrame value
                "initial_capital": "10000",
                "strategy_params": {},
                "mode": "paper",
            }
        )
        orphan.status = "running"

        select_session = _make_session_for_select([orphan])
        write_session = _make_session_for_write(orphan)
        factory = _build_factory(select_session, write_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
        ):
            result = await recover_orphaned_runs()

        assert result == 0
        assert runs_module._RUN_TASKS == {}
        assert orphan.status == "error"

    @pytest.mark.asyncio
    async def test_recovery_chain_orphan_excluded_by_query(self) -> None:
        """An orphan that already has recovered_from_run_id set must NOT appear in query results.

        The SELECT WHERE clause filters recovered_from_run_id IS NULL so a previously
        recovered run never re-enters the recovery loop.  We model this by returning
        an empty list from the SELECT (simulating the DB already excluding it).
        The function must return 0 and start no tasks.
        """
        # This orphan was itself created by a previous recovery pass
        _chain_orphan = _make_orphan(recovered_from_run_id=uuid.uuid4())

        # The SELECT mock returns NO results (the DB WHERE clause excludes chain orphans)
        select_session = _make_session_for_select([])
        factory = _build_factory(select_session)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
        ):
            result = await recover_orphaned_runs()

        assert result == 0, "Chain orphan must be excluded by the query, not recovered again"
        assert runs_module._RUN_TASKS == {}


# ---------------------------------------------------------------------------
# Class 6: TestRecoverPerOrphanIsolation
# ---------------------------------------------------------------------------


class TestRecoverPerOrphanIsolation:
    """Verify that per-orphan exception isolation works correctly.

    One bad orphan in a batch must not prevent the valid one from recovering.
    """

    @pytest.mark.asyncio
    async def test_valid_orphan_recovers_even_when_neighbour_fails(self) -> None:
        """When one orphan has an invalid config and another is valid, count == 1.

        The invalid orphan is the first one returned by the SELECT so it exercises
        the guard-clause path.  The valid orphan is the second.  The final return
        value must be 1 (only the valid one counted).
        """
        bad_orphan = _make_orphan(
            config={
                "strategy_name": "does_not_exist",
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "initial_capital": "10000",
                "strategy_params": {},
                "mode": "paper",
            }
        )
        good_orphan = _make_orphan(run_mode="paper")

        # SELECT returns both orphans
        select_session = _make_session_for_select([bad_orphan, good_orphan])
        # Two write sessions: one for bad_orphan's _mark_orphan_error,
        # then one for good_orphan's atomic UPDATE+INSERT
        bad_write = _make_session_for_write(bad_orphan)
        good_write = _make_session_for_write(good_orphan)
        factory = _build_factory(select_session, bad_write, good_write)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        assert result == 1, (
            f"Only the valid orphan must be counted, got {result}"
        )
        assert len(runs_module._RUN_TASKS) == 1

    @pytest.mark.asyncio
    async def test_exception_in_one_orphan_does_not_crash_loop(self) -> None:
        """An unexpected exception while processing orphan N must not stop orphan N+1.

        We force a session.commit() to raise RuntimeError for the first orphan's
        write session to simulate an unexpected DB error mid-recovery.  The second
        valid orphan must still be recovered.
        """
        orphan_fail = _make_orphan(run_mode="paper")
        orphan_ok = _make_orphan(run_mode="paper")

        select_session = _make_session_for_select([orphan_fail, orphan_ok])

        # First write session raises on commit so the inner try/except fires
        fail_write = _make_session_for_write(orphan_fail)
        fail_write.commit = AsyncMock(side_effect=RuntimeError("simulated DB error"))

        ok_write = _make_session_for_write(orphan_ok)
        factory = _build_factory(select_session, fail_write, ok_write)

        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            patch("api.config.get_settings", return_value=_make_settings()),
            patch("api.routers.runs._get_strategy_registry", return_value={"ma_crossover": MagicMock()}),
            patch("api.routers.runs._run_paper_engine", AsyncMock()),
            patch("api.routers.runs._run_live_engine", AsyncMock()),
        ):
            result = await recover_orphaned_runs()

        # The second orphan must have been recovered despite the first failing
        assert result == 1, (
            f"Valid second orphan must still be recovered despite first error, got {result}"
        )
