"""
tests/unit/test_run_registry.py
--------------------------------
Unit tests for the RunRegistry class.

Modules under test
------------------
    apps/api/run_registry.py  --  RunRegistry

Coverage groups
---------------
1. TestRunRegistryInit             -- fresh registry is empty (2 tests)
2. TestRunRegistryRegisterGet      -- register + get roundtrip (5 tests)
3. TestRunRegistryUnregister       -- unregister behaviour (4 tests)
4. TestRunRegistryMembership       -- __contains__, active_run_ids, absent-key gets (6 tests)
5. TestRunRegistryCancelAll        -- cancel_all paths (5 tests)
6. TestRunRegistryConcurrency      -- concurrent mutations under lock (1 test)

Design notes
------------
- asyncio_mode = "auto" in pyproject.toml — no @pytest.mark.asyncio needed.
- Dummy coroutines use asyncio.sleep(999) so tasks stay PENDING during tests.
- cancel_all tests cancel/await tasks after assertions to avoid ResourceWarning.
"""

from __future__ import annotations

import asyncio

import pytest

from api.run_registry import RunRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _long_sleep() -> None:
    """A coroutine that sleeps forever — keeps tasks PENDING."""
    await asyncio.sleep(999)


async def _noop() -> None:
    """A coroutine that returns immediately."""
    return


def _make_task() -> asyncio.Task[None]:
    """Create a long-lived pending asyncio.Task."""
    return asyncio.get_running_loop().create_task(_long_sleep())


def _make_done_task() -> asyncio.Task[None]:
    """Create a task that completes immediately."""
    return asyncio.get_running_loop().create_task(_noop())


class _FakeEngine:
    def __init__(self, name: str = "engine") -> None:
        self.name = name


class _FakeLearning:
    def __init__(self, name: str = "learning") -> None:
        self.name = name


# ---------------------------------------------------------------------------
# 1. TestRunRegistryInit
# ---------------------------------------------------------------------------


class TestRunRegistryInit:
    """Fresh registry state."""

    def test_starts_empty_active_run_ids(self) -> None:
        registry = RunRegistry()
        assert registry.active_run_ids() == []

    def test_starts_with_zero_length(self) -> None:
        registry = RunRegistry()
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# 2. TestRunRegistryRegisterGet
# ---------------------------------------------------------------------------


class TestRunRegistryRegisterGet:
    """register() + get_*() roundtrip."""

    async def test_get_task_after_register(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        await registry.register("run-1", task=task, engine=engine)
        assert registry.get_task("run-1") is task
        task.cancel()

    async def test_get_engine_after_register(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        await registry.register("run-1", task=task, engine=engine)
        assert registry.get_engine("run-1") is engine
        task.cancel()

    async def test_get_learning_after_register_with_learning(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        learning = _FakeLearning()
        await registry.register("run-1", task=task, engine=engine, learning=learning)
        assert registry.get_learning("run-1") is learning
        task.cancel()

    async def test_learning_absent_when_not_provided(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        await registry.register("run-1", task=task, engine=engine)
        assert registry.get_learning("run-1") is None
        task.cancel()

    async def test_register_same_run_id_twice_overwrites(self) -> None:
        registry = RunRegistry()
        task1 = _make_task()
        task2 = _make_task()
        engine1 = _FakeEngine("e1")
        engine2 = _FakeEngine("e2")
        await registry.register("run-1", task=task1, engine=engine1)
        await registry.register("run-1", task=task2, engine=engine2)
        assert registry.get_task("run-1") is task2
        assert registry.get_engine("run-1") is engine2
        assert len(registry) == 1
        task1.cancel()
        task2.cancel()


# ---------------------------------------------------------------------------
# 3. TestRunRegistryUnregister
# ---------------------------------------------------------------------------


class TestRunRegistryUnregister:
    """unregister() behaviour."""

    async def test_unregister_removes_all_slots(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        learning = _FakeLearning()
        await registry.register("run-1", task=task, engine=engine, learning=learning)
        task.cancel()
        await registry.unregister("run-1")
        assert registry.get_task("run-1") is None
        assert registry.get_engine("run-1") is None
        assert registry.get_learning("run-1") is None
        assert len(registry) == 0

    async def test_unregister_unknown_run_id_is_noop(self) -> None:
        registry = RunRegistry()
        await registry.unregister("nonexistent-run")  # must not raise

    async def test_double_unregister_is_idempotent(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        await registry.register("run-1", task=task, engine=engine)
        task.cancel()
        await registry.unregister("run-1")
        await registry.unregister("run-1")  # second call — must not raise

    async def test_unregister_clears_learning_after_task_cancelled(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        engine = _FakeEngine()
        learning = _FakeLearning()
        await registry.register("run-1", task=task, engine=engine, learning=learning)
        task.cancel()
        await asyncio.sleep(0)  # let cancellation propagate
        await registry.unregister("run-1")
        assert registry.get_learning("run-1") is None


# ---------------------------------------------------------------------------
# 4. TestRunRegistryMembership
# ---------------------------------------------------------------------------


class TestRunRegistryMembership:
    """__contains__, active_run_ids, and get_* on absent keys."""

    async def test_contains_returns_true_for_registered_run(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        await registry.register("run-1", task=task, engine=_FakeEngine())
        assert "run-1" in registry
        task.cancel()

    async def test_contains_returns_false_after_unregister(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        await registry.register("run-1", task=task, engine=_FakeEngine())
        task.cancel()
        await registry.unregister("run-1")
        assert "run-1" not in registry

    async def test_active_run_ids_snapshot_is_independent(self) -> None:
        """Mutating the returned list must not affect the registry."""
        registry = RunRegistry()
        task = _make_task()
        await registry.register("run-1", task=task, engine=_FakeEngine())
        snapshot = registry.active_run_ids()
        snapshot.clear()
        assert len(registry) == 1
        task.cancel()

    def test_get_task_absent_returns_none(self) -> None:
        registry = RunRegistry()
        assert registry.get_task("missing") is None

    def test_get_engine_absent_returns_none(self) -> None:
        registry = RunRegistry()
        assert registry.get_engine("missing") is None

    def test_get_learning_absent_returns_none(self) -> None:
        registry = RunRegistry()
        assert registry.get_learning("missing") is None


# ---------------------------------------------------------------------------
# 5. TestRunRegistryCancelAll
# ---------------------------------------------------------------------------


class TestRunRegistryCancelAll:
    """cancel_all() behaviour."""

    async def test_cancel_all_cancels_pending_tasks(self) -> None:
        registry = RunRegistry()
        task = _make_task()
        await registry.register("run-1", task=task, engine=_FakeEngine())
        await registry.cancel_all(timeout=2.0)
        assert task.cancelled() or task.done()

    async def test_cancel_all_on_empty_registry_is_noop(self) -> None:
        registry = RunRegistry()
        await registry.cancel_all(timeout=2.0)  # must not raise

    async def test_cancel_all_tolerates_already_finished_tasks(self) -> None:
        registry = RunRegistry()
        done_task = _make_done_task()
        await asyncio.sleep(0)  # let the task complete
        await registry.register("run-done", task=done_task, engine=_FakeEngine())
        await registry.cancel_all(timeout=2.0)  # must not raise

    async def test_cancel_all_multiple_tasks(self) -> None:
        registry = RunRegistry()
        tasks = []
        for i in range(5):
            t = _make_task()
            tasks.append(t)
            await registry.register(f"run-{i}", task=t, engine=_FakeEngine(f"e{i}"))
        await registry.cancel_all(timeout=2.0)
        for t in tasks:
            assert t.cancelled() or t.done()

    async def test_cancel_all_respects_timeout(self) -> None:
        """cancel_all must return within ~timeout even if a task ignores cancellation."""

        async def _stubborn() -> None:
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                # Ignores first cancellation and blocks again
                await asyncio.sleep(999)

        registry = RunRegistry()
        task = asyncio.get_running_loop().create_task(_stubborn())
        await registry.register("stubborn-run", task=task, engine=_FakeEngine())
        # Short timeout — method must return without blocking forever.
        await registry.cancel_all(timeout=0.1)
        # Force-cancel for event-loop cleanup.
        task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task


# ---------------------------------------------------------------------------
# 6. TestRunRegistryConcurrency
# ---------------------------------------------------------------------------


class TestRunRegistryConcurrency:
    """Concurrent mutations under asyncio.Lock."""

    async def test_concurrent_register_no_lost_writes(self) -> None:
        """10 coroutines each register a unique run — all 10 must be present."""
        registry = RunRegistry()
        tasks_to_cancel: list[asyncio.Task[None]] = []

        async def _register_one(run_id: str) -> None:
            t = asyncio.get_running_loop().create_task(_long_sleep())
            tasks_to_cancel.append(t)
            await registry.register(run_id, task=t, engine=_FakeEngine(run_id))

        await asyncio.gather(*[_register_one(f"concurrent-run-{i}") for i in range(10)])

        assert len(registry) == 10
        for i in range(10):
            assert f"concurrent-run-{i}" in registry

        for t in tasks_to_cancel:
            t.cancel()
