"""
tests/unit/test_wp17a_round3_security.py
--------------------------------------------
WP1.7a round 3 (security round 2 + critic round 2) regression tests,
replicating the findings in
reports/vp2-wp1.7/security-report-1.7a-r2.md (WP17a-S-R2-01..S-R2-06)
and reports/vp2-wp1.7/critic-report-1.7a-r2.md (WP17a-C-04).

The real-Postgres A7/R1 interleaving regression required by S-R2-01
lives in ``tests/migrations/test_wp17a_round3_engine_race.py`` -- it
needs a genuine Postgres row lock to reproduce the race that a plain
asyncio.Lock (with no real DB contention) cannot. This file covers the
unit-level behavioural surface:

- ``kill_switch.activate()``/``clear()``'s new ``engines`` parameter and
  the fact that engine mutation now happens ATOMICALLY with the mirror
  flip, under the module lock (S-R2-01).
- the router no longer running its own post-``clear()`` engine loop,
  and the kill-switch-flatten-then-clear scenario ending with the
  incomplete run STILL latched (S-R2-02).
- S-R2-03's ``begin_nested()`` wrapping of ``activate()``'s UPDATE.
- S-R2-06's ``since=None``-when-inactive fix in ``load()``.

S-R2-04 (the ``timeout_open`` synthetic-result outcome fix) and C-04
(the redundant quoted forward-ref) are covered by updates made directly
to ``test_wp17a_round2_security.py::TestBuildSyntheticFlattenResult``
and ``apps/api/schemas.py`` respectively -- no new test file needed for
either.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services import kill_switch
from trading.strategy_engine import FlattenResult, FlattenSymbolResult


def _make_request() -> MagicMock:
    from types import SimpleNamespace

    req = MagicMock()
    req.headers = {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


def _make_settings(admin_key: str = "a" * 32 + "b1c2d3e4f5g6") -> object:
    from pydantic import SecretStr

    from api.config import Settings

    s = Settings.model_construct()
    object.__setattr__(s, "admin_api_key", SecretStr(admin_key))
    return s


class _FakeRiskManager:
    """A real (non-Mock) reason-set double matching BaseRiskManager's
    public kill-switch surface -- used instead of MagicMock so
    activate()/clear()'s ``engines`` callback can be exercised against
    genuine mutable state rather than call-count-only assertions."""

    def __init__(self) -> None:
        self._reasons: set[str] = set()

    def trigger_kill_switch(self, reason: str) -> None:
        self._reasons.add(reason)

    def reset_kill_switch(self, reason: str) -> None:
        self._reasons.discard(reason)

    @property
    def kill_switch_active(self) -> bool:
        return bool(self._reasons)

    @property
    def kill_switch_reasons(self) -> frozenset[str]:
        return frozenset(self._reasons)


class _FakeEngine:
    def __init__(self, run_id: str | None = "run-1") -> None:
        self.run_id = run_id
        self.risk_manager = _FakeRiskManager()
        self.symbols: list[str] = ["BTC/EUR"]

    async def flatten(self, *, reason: str, timeout_s: float) -> FlattenResult:
        """Always returns an INCOMPLETE result (S-R2-02 fixture)."""
        return FlattenResult(
            run_id=self.run_id or "",
            outcome="partial",
            complete=False,
            symbols=[
                FlattenSymbolResult(
                    symbol="BTC/EUR",
                    status="in_flight",
                    cause="ledger_doubt",
                    held_before=Decimal("0.01"),
                    sold_qty=Decimal("0"),
                    remaining_qty=Decimal("0.01"),
                )
            ],
        )


@pytest.fixture(autouse=True)
def _reset() -> None:
    kill_switch.reset_state_for_tests()
    yield
    kill_switch.reset_state_for_tests()


class TestSR201ActivateLatchesEnginesUnderLock:
    """S-R2-01: activate()'s engine-latch must happen atomically with
    the mirror flip, before its own persistence attempt, and there is no
    longer a separate post-persist re-assert step."""

    async def test_activate_latches_provided_engines_before_persisting(self) -> None:
        engine = _FakeEngine("run-a")
        observed: list[bool] = []

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            observed.append(engine.risk_manager.kill_switch_active)
            result = MagicMock()
            result.rowcount = 1
            return result

        db = AsyncMock()
        db.execute = _execute
        db.flush = AsyncMock()
        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)
        db.begin_nested = MagicMock(return_value=nested_cm)

        assert engine.risk_manager.kill_switch_active is False

        persisted = await kill_switch.activate(
            db, reason="press", actor="tester", engines=lambda: [engine]
        )

        assert persisted is True
        assert observed == [True], "engine must be latched before the first persist await"
        assert engine.risk_manager.kill_switch_active is True
        assert kill_switch.GLOBAL_REASON in engine.risk_manager.kill_switch_reasons

    async def test_activate_latches_engines_even_when_persist_fails(self) -> None:
        engine = _FakeEngine("run-b")
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))

        persisted = await kill_switch.activate(
            db, reason="press", actor="tester", engines=lambda: [engine]
        )

        assert persisted is False
        assert engine.risk_manager.kill_switch_active is True
        assert kill_switch.is_active() is True

    async def test_activate_without_engines_param_is_still_valid(self) -> None:
        """Backward-compat: engines=None (the default) skips engine
        mutation entirely -- callers that only care about the mirror
        (e.g. existing unit tests) are unaffected."""
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(rowcount=1))
        db.flush = AsyncMock()
        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)
        db.begin_nested = MagicMock(return_value=nested_cm)

        persisted = await kill_switch.activate(db, reason="press", actor="tester")

        assert persisted is True
        assert kill_switch.is_active() is True


class TestSR201ClearUnlatchesEnginesOnlyAfterCommit:
    """S-R2-01: clear()'s engine-unlatch pass must run AFTER the commit
    succeeds, still inside the same lock section -- and must never touch
    an engine at all if persistence fails."""

    def _clear_ready_db(self) -> AsyncMock:
        db = AsyncMock()
        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = MagicMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        db.execute = AsyncMock(side_effect=[select_result, update_result])
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        return db

    async def test_clear_unlatches_engine_only_after_commit(self) -> None:
        engine = _FakeEngine("run-c")
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        kill_switch.latch_in_memory("pre-existing")
        db = self._clear_ready_db()

        commit_seen_engine_state: list[bool] = []
        real_commit = db.commit

        async def _commit() -> None:
            # The engine must STILL be latched at the moment commit()
            # is awaited -- the unlatch only happens after commit
            # returns successfully.
            commit_seen_engine_state.append(engine.risk_manager.kill_switch_active)
            await real_commit()

        db.commit = _commit

        outcome = await kill_switch.clear(
            db, reason="resolved", actor="tester", engines=lambda: [engine]
        )

        assert commit_seen_engine_state == [True]
        assert engine.risk_manager.kill_switch_active is False
        assert outcome.runs_unlatched == ["run-c"]
        assert outcome.runs_kept_latched == []

    async def test_clear_failure_never_touches_the_engine(self) -> None:
        engine = _FakeEngine("run-d")
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        kill_switch.latch_in_memory("pre-existing")

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        db.rollback = AsyncMock()

        with pytest.raises(kill_switch.KillSwitchClearError):
            await kill_switch.clear(
                db, reason="resolved", actor="tester", engines=lambda: [engine]
            )

        assert engine.risk_manager.kill_switch_active is True, (
            "a failed clear must never unlatch an engine"
        )
        assert kill_switch.is_active() is True

    async def test_clear_reports_kept_latched_reasons_for_engines_with_other_latches(
        self,
    ) -> None:
        engine = _FakeEngine("run-e")
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        engine.risk_manager.trigger_kill_switch("flatten_incomplete")
        kill_switch.latch_in_memory("pre-existing")
        db = self._clear_ready_db()

        outcome = await kill_switch.clear(
            db, reason="resolved", actor="tester", engines=lambda: [engine]
        )

        assert outcome.runs_unlatched == []
        assert outcome.runs_kept_latched == [("run-e", ["flatten_incomplete"])]
        assert engine.risk_manager.kill_switch_active is True
        assert kill_switch.GLOBAL_REASON not in engine.risk_manager.kill_switch_reasons

    async def test_clear_engine_with_no_run_id_falls_back_to_unknown_label(self) -> None:
        engine = _FakeEngine(run_id=None)
        engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
        kill_switch.latch_in_memory("pre-existing")
        db = self._clear_ready_db()

        outcome = await kill_switch.clear(
            db, reason="resolved", actor="tester", engines=lambda: [engine]
        )

        assert outcome.runs_unlatched == ["<unknown>"]


class TestSR202FlattenIncompleteAlwaysLatchesInMemory:
    """S-R2-02: the kill-switch flatten pass must latch
    'flatten_incomplete' on the engine unconditionally when the flatten
    result is incomplete -- even if _persist_run_flatten itself fails --
    so a later global clear can never re-enable BUYs on that run."""

    async def test_incomplete_flatten_then_global_clear_keeps_run_latched(self) -> None:
        from api.routers.emergency import (
            KillSwitchClearRequest,
            KillSwitchRequest,
            kill_switch_clear,
        )
        from api.routers.emergency import kill_switch as kill_switch_endpoint

        engine = _FakeEngine("run-f")
        run_id = engine.run_id
        assert run_id is not None

        press_db = AsyncMock()
        press_result = MagicMock()
        press_result.scalars.return_value.all.return_value = []
        press_db.execute = AsyncMock(return_value=press_result)
        press_db.commit = AsyncMock()
        press_db.flush = AsyncMock()
        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)
        press_db.begin_nested = MagicMock(return_value=nested_cm)

        with (
            patch("api.routers.emergency._RUN_ENGINES", {run_id: engine}),
            # S-R2-02: force _persist_run_flatten to fail -- the
            # in-memory latch must be applied regardless.
            patch(
                "api.routers.emergency.get_session_factory",
                side_effect=RuntimeError("db down for the flatten persist pass"),
            ),
        ):
            press_response = await kill_switch_endpoint(
                request=_make_request(),
                db=press_db,
                reason=None,
                settings=_make_settings(),
                body=KillSwitchRequest(flatten=True),
            )

        assert press_response.flatten_results[run_id].latch_persisted is False
        assert engine.risk_manager.kill_switch_active is True
        assert "flatten_incomplete" in engine.risk_manager.kill_switch_reasons

        clear_db = AsyncMock()
        clear_select_result = MagicMock()
        clear_select_result.scalar_one_or_none.return_value = MagicMock()
        clear_update_result = MagicMock()
        clear_update_result.rowcount = 1
        clear_db.execute = AsyncMock(side_effect=[clear_select_result, clear_update_result])
        clear_db.commit = AsyncMock()
        clear_db.rollback = AsyncMock()

        with (
            patch("api.routers.emergency._RUN_ENGINES", {run_id: engine}),
            patch("api.routers.emergency.get_settings", return_value=_make_settings()),
            patch("api.routers.emergency.record_audit_event_strict", new=AsyncMock()),
        ):
            clear_response = await kill_switch_clear(
                request=_make_request(),
                db=clear_db,
                body=KillSwitchClearRequest(reason="operator cleared the global latch"),
            )

        assert clear_response.runs_unlatched == []
        assert len(clear_response.runs_kept_latched) == 1
        kept = clear_response.runs_kept_latched[0]
        assert kept.run_id == run_id
        assert kept.reasons == ["flatten_incomplete"]
        assert engine.risk_manager.kill_switch_active is True, (
            "the run must STILL be latched after the global clear -- only "
            "an explicit per-run entries-latch/clear can remove "
            "flatten_incomplete"
        )
        assert kill_switch.is_active() is False, "the GLOBAL mirror is cleared"


class TestSR203ActivatePersistUsesSavepoint:
    async def test_activate_wraps_persist_in_begin_nested(self) -> None:
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(rowcount=1))
        db.flush = AsyncMock()
        nested_cm = MagicMock()
        nested_cm.__aenter__ = AsyncMock(return_value=None)
        nested_cm.__aexit__ = AsyncMock(return_value=False)
        db.begin_nested = MagicMock(return_value=nested_cm)

        persisted = await kill_switch.activate(db, reason="press", actor="tester")

        assert persisted is True
        db.begin_nested.assert_called_once()
        nested_cm.__aenter__.assert_awaited_once()
        nested_cm.__aexit__.assert_awaited_once()


class TestSR206SinceNoneWhenInactive:
    async def test_load_of_a_cleared_row_reports_since_none(self) -> None:
        row = MagicMock(active=False, reason=None, activated_at=datetime.now(tz=UTC))
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        db.execute = AsyncMock(return_value=result)

        state = await kill_switch.load(db)

        assert state.active is False
        assert state.since is None, (
            "a cleared row's activated_at must not leak into 'since' -- "
            "'since' means 'since it was latched', not 'the last time it "
            "was ever activated'"
        )

    async def test_load_of_an_active_row_reports_since(self) -> None:
        activated_at = datetime.now(tz=UTC)
        row = MagicMock(active=True, reason="test", activated_at=activated_at)
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        db.execute = AsyncMock(return_value=result)

        state = await kill_switch.load(db)

        assert state.active is True
        assert state.since == activated_at
