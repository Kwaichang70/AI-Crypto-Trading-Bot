"""
tests/unit/test_wp17a_kill_switch_service.py
-----------------------------------------------
WP1.7a: unit tests for apps/api/services/kill_switch.py.

Mandatory per reports/vp2-wp1.7/synthesis-spec.md §6 (1.7a unit tests):
  - load() failure -> latched (I4)
  - order of activate() and clear() (I3)
  - apply_latch() has no await gap (I5)

Round 2 (security round 1, WP17a-S-01..S-04) adds:
  - the bare-import fail-closed default (not merely a load() failure)
  - load()/mark_unknown() catching more than SQLAlchemyError
  - the activate/clear module lock (S-03)
  - apply_latch() never leaking a free-text operator reason (S-04)

Round 3 (security round 2, WP17a-S-R2-01/S-R2-03/S-R2-06) adds:
  - activate() flips the mirror as the FIRST statement inside the lock
    (no separate post-persist re-assert step any more -- see
    test_wp17a_round3_security.py for the engines= parameter and the
    engine-mutation-under-lock coverage)
  - activate()'s persist step runs inside db.begin_nested() (S-R2-03)
  - load() reports since=None for an inactive row (S-R2-06)
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services import kill_switch


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Each test starts from an explicit LOADED, un-latched mirror --
    the fail-closed/unloaded default (I4/S-02) is exercised explicitly
    by the load()/mark_unknown() tests below via their own assertions,
    not via fixture state."""
    kill_switch.reset_state_for_tests()
    yield
    kill_switch.reset_state_for_tests()


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.risk_manager = MagicMock()
    engine.risk_manager.trigger_kill_switch = MagicMock()
    return engine


def _clear_ready_db(rowcount: int = 1, row: object | None = "present") -> AsyncMock:
    """An AsyncMock db wired so clear()'s row-lock SELECT and its UPDATE
    both succeed by default (row present, exactly one row updated)."""
    db = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = (MagicMock() if row == "present" else row)
    update_result = MagicMock()
    update_result.rowcount = rowcount
    db.execute = AsyncMock(side_effect=[select_result, update_result])
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


class TestBareDefault:
    """S-02: the bare import-time / post-reset-to-unloaded default is
    fail-closed, not un-latched."""

    def test_unloaded_mirror_reports_active(self) -> None:
        kill_switch.mark_unknown()
        assert kill_switch.is_loaded() is False
        assert kill_switch.is_active() is True
        state = kill_switch.current_state()
        assert state.active is True
        assert state.reason == kill_switch.UNKNOWN_STATE_REASON

    def test_mark_unknown_overrides_a_previously_clear_mirror(self) -> None:
        """mark_unknown() must fail the mirror closed even if it was
        previously loaded and un-latched (main.py's outer except-branch
        calls this regardless of what load() may or may not have set)."""
        kill_switch.reset_state_for_tests()
        assert kill_switch.is_active() is False

        kill_switch.mark_unknown()

        assert kill_switch.is_active() is True
        assert kill_switch.current_state().reason == kill_switch.UNKNOWN_STATE_REASON


class TestLoad:
    async def test_load_success_sets_mirror_from_row(self) -> None:
        row = MagicMock(active=False, reason=None, activated_at=None)
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        db.execute = AsyncMock(return_value=result)

        state = await kill_switch.load(db)

        assert state.active is False
        assert kill_switch.is_active() is False
        assert kill_switch.is_loaded() is True

    async def test_load_failure_leaves_process_latched(self) -> None:
        """I4: a read failure must leave the mirror latched with
        reason='latch_state_unknown', not un-latched."""
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("boom"))

        state = await kill_switch.load(db)

        assert state.active is True
        assert state.reason == kill_switch.UNKNOWN_STATE_REASON
        assert kill_switch.is_active() is True
        assert kill_switch.is_loaded() is False

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionRefusedError("connection refused"),
            OSError("bad DB name"),
            ValueError("password authentication failed"),
        ],
    )
    async def test_load_catches_more_than_sqlalchemy_error(self, exc: Exception) -> None:
        """S-02/WP17a-C-01: load() must catch every exception a real
        asyncpg/session-factory failure can raise -- ConnectionRefusedError,
        InvalidCatalogNameError-style OSError subclasses, auth failures --
        not just SQLAlchemyError."""
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=exc)

        state = await kill_switch.load(db)

        assert state.active is True
        assert state.reason == kill_switch.UNKNOWN_STATE_REASON
        assert kill_switch.is_loaded() is False

    async def test_load_missing_row_leaves_process_latched(self) -> None:
        """I4: a missing singleton row (should never happen post-018, but
        treated identically out of paranoia) also leaves the mirror
        latched."""
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result)

        state = await kill_switch.load(db)

        assert state.active is True
        assert state.reason == kill_switch.UNKNOWN_STATE_REASON
        assert kill_switch.is_loaded() is False


class TestActivateClearOrder:
    async def test_activate_flips_memory_before_persisting(self) -> None:
        """I3: the in-memory mirror must already read 'active' the
        instant activate() calls its FIRST await (db.execute) -- verified
        by making db.execute itself assert on the mirror's state."""
        observed: list[bool] = []

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            observed.append(kill_switch.is_active())
            result = MagicMock()
            result.rowcount = 1
            return result

        db = AsyncMock()
        db.execute = _execute
        db.flush = AsyncMock()
        # WP17a-S-R2-03 (round 3): activate()'s persist step now runs
        # inside db.begin_nested() (a SAVEPOINT) -- give it a working
        # async-context-manager double, matching real AsyncSession
        # behaviour (begin_nested() itself is a plain sync method).
        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)
        db.begin_nested = MagicMock(return_value=nested_cm)

        assert kill_switch.is_active() is False
        persisted = await kill_switch.activate(db, reason="test", actor="unit-test")

        assert persisted is True
        assert observed == [True], "memory must flip before the first await"
        assert kill_switch.is_active() is True

    async def test_activate_persist_failure_still_latches_in_memory(self) -> None:
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))

        persisted = await kill_switch.activate(db, reason="test", actor="unit-test")

        assert persisted is False
        assert kill_switch.is_active() is True, "fail-closed: still latched in memory"

    async def test_clear_persists_before_flipping_memory(self) -> None:
        """I3: clear() must NOT flip the mirror until its own DB write
        (and commit) succeed."""
        kill_switch.latch_in_memory("test")

        db = _clear_ready_db()

        await kill_switch.clear(db, reason="resolved", actor="unit-test")

        assert kill_switch.is_active() is False
        db.commit.assert_awaited_once()

    async def test_clear_persist_failure_raises_and_stays_latched(self) -> None:
        kill_switch.latch_in_memory("test")

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        db.rollback = AsyncMock()

        with pytest.raises(kill_switch.KillSwitchClearError):
            await kill_switch.clear(db, reason="resolved", actor="unit-test")

        assert kill_switch.is_active() is True, "a failed clear must never un-latch"
        db.rollback.assert_awaited_once()

    async def test_clear_requires_rowcount_one(self) -> None:
        """S-03: clear() must treat anything other than exactly one row
        updated as a persistence failure, not a silent success."""
        kill_switch.latch_in_memory("test")
        db = _clear_ready_db(rowcount=0)

        with pytest.raises(kill_switch.KillSwitchClearError):
            await kill_switch.clear(db, reason="resolved", actor="unit-test")

        assert kill_switch.is_active() is True
        db.rollback.assert_awaited_once()

    async def test_clear_missing_row_is_a_persistence_failure(self) -> None:
        kill_switch.latch_in_memory("test")
        db = _clear_ready_db(row=None)

        with pytest.raises(kill_switch.KillSwitchClearError):
            await kill_switch.clear(db, reason="resolved", actor="unit-test")

        assert kill_switch.is_active() is True

    async def test_clear_runs_before_commit_hook_inside_the_transaction(self) -> None:
        """S-10/S-12: clear()'s before_commit hook (the caller's
        non-swallowing audit write) must run BEFORE clear()'s own
        commit, in the same transaction."""
        kill_switch.latch_in_memory("test")
        db = _clear_ready_db()
        call_order: list[str] = []

        async def _before_commit() -> None:
            call_order.append("audit")

        real_commit = db.commit

        async def _commit() -> None:
            call_order.append("commit")
            await real_commit()

        db.commit = _commit

        await kill_switch.clear(
            db, reason="resolved", actor="unit-test", before_commit=_before_commit
        )

        assert call_order == ["audit", "commit"]

    async def test_clear_before_commit_failure_rolls_back_and_stays_latched(self) -> None:
        kill_switch.latch_in_memory("test")
        db = _clear_ready_db()

        async def _failing_before_commit() -> None:
            raise RuntimeError("audit write failed")

        with pytest.raises(kill_switch.KillSwitchClearError):
            await kill_switch.clear(
                db, reason="resolved", actor="unit-test", before_commit=_failing_before_commit
            )

        assert kill_switch.is_active() is True
        db.rollback.assert_awaited_once()
        db.commit.assert_not_awaited()

    async def test_activate_and_clear_race_reasserts_active(self) -> None:
        """S-03: activate() must win a race against a concurrent clear()
        that completes in the gap between activate()'s synchronous flip
        and its own lock acquisition -- the module lock forces the two
        to run one at a time, and activate() re-asserts active=True
        after it gets the lock regardless of ordering."""
        clear_db = _clear_ready_db()
        activate_db = AsyncMock()
        activate_result = MagicMock()
        activate_result.rowcount = 1
        activate_db.execute = AsyncMock(return_value=activate_result)
        activate_db.flush = AsyncMock()

        kill_switch.latch_in_memory("pre-existing")

        await asyncio.gather(
            kill_switch.clear(clear_db, reason="resolved", actor="clearer"),
            kill_switch.activate(activate_db, reason="new-press", actor="presser"),
        )

        # Whichever order the two coroutines' lock acquisitions actually
        # resolved in, activate() re-asserts unconditionally after its
        # own lock section -- the mirror must end up active (never
        # silently un-latched by a clear that raced it).
        assert kill_switch.is_active() is True


class TestApplyLatch:
    def test_apply_latch_is_synchronous_no_await_gap(self) -> None:
        """I5: apply_latch must be a plain (non-async) function -- there
        is no ``await`` opportunity between ``_RUN_ENGINES[id] = engine``
        and this call at any of its call sites."""
        assert not inspect.iscoroutinefunction(kill_switch.apply_latch)

    def test_apply_latch_global_active_triggers_engine(self) -> None:
        kill_switch.latch_in_memory(kill_switch.GLOBAL_REASON)
        engine = _make_engine()

        kill_switch.apply_latch(engine)

        engine.risk_manager.trigger_kill_switch.assert_called_once_with("global_kill_switch")

    def test_apply_latch_never_leaks_free_text_operator_reason(self) -> None:
        """S-04: apply_latch() must add ONLY GLOBAL_REASON, never the
        operator's free-text X-Emergency-Reason header, even though
        that text is exactly what is stored in _STATE.reason /
        persisted to the DB 'reason' column."""
        kill_switch.latch_in_memory("flash crash on BTC, operator paged")
        engine = _make_engine()

        kill_switch.apply_latch(engine)

        engine.risk_manager.trigger_kill_switch.assert_called_once_with(
            kill_switch.GLOBAL_REASON
        )

    def test_apply_latch_unloaded_uses_unknown_reason_not_free_text(self) -> None:
        kill_switch.mark_unknown()
        engine = _make_engine()

        kill_switch.apply_latch(engine)

        engine.risk_manager.trigger_kill_switch.assert_called_once_with(
            kill_switch.UNKNOWN_STATE_REASON
        )

    def test_apply_latch_inactive_does_not_trigger(self) -> None:
        engine = _make_engine()

        kill_switch.apply_latch(engine)

        engine.risk_manager.trigger_kill_switch.assert_not_called()

    def test_apply_latch_per_run_reason_applied_independently(self) -> None:
        """SY-01: the per-run entries_latch_reason is applied even when
        the global latch is inactive, and vice versa -- global state is
        never copied into the per-run reason and back."""
        engine = _make_engine()

        kill_switch.apply_latch(engine, entries_latch_reason="flatten_incomplete")

        engine.risk_manager.trigger_kill_switch.assert_called_once_with("flatten_incomplete")

    def test_apply_latch_both_reasons_applied(self) -> None:
        kill_switch.latch_in_memory("operator free text is irrelevant here")
        engine = _make_engine()

        kill_switch.apply_latch(engine, entries_latch_reason="flatten_incomplete")

        assert engine.risk_manager.trigger_kill_switch.call_count == 2
        called_reasons = {
            call.args[0] for call in engine.risk_manager.trigger_kill_switch.call_args_list
        }
        assert called_reasons == {"global_kill_switch", "flatten_incomplete"}


class TestLatchInMemory:
    def test_latch_in_memory_is_synchronous(self) -> None:
        assert not inspect.iscoroutinefunction(kill_switch.latch_in_memory)

    def test_latch_in_memory_sets_active_and_loaded(self) -> None:
        kill_switch.latch_in_memory("some reason")
        assert kill_switch.is_active() is True
        assert kill_switch.is_loaded() is True
        assert kill_switch.current_state().reason == "some reason"
