"""
tests/unit/test_wp22_prune.py
-----------------------------
WP2.2 P0 (SY-22-10, WP22-T-12) -- unit coverage for
``api.services.retention.prune_equity_snapshots`` against a lightweight
fake session factory (same style as ``test_wp70_idempotency_prune.py``),
plus Postgres-dialect compile checks of the emitted predicates and an AST
guard that ``main.py``'s prune loop keeps no inline DELETE / SQL.

The row-level behaviour (backtest created now with 2-year-old stamps ->
0 rows deleted, paper/live run created long ago with recent stamps -> kept,
etc.) needs a real database and is proven by
``tests/migrations/test_wp22_equity_prune_pg.py``.
"""

from __future__ import annotations

import ast
import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import TextClause
from structlog.testing import capture_logs

from api.services.retention import (
    build_backtest_prune_stmt,
    build_timestamp_prune_stmt,
    prune_equity_snapshots,
)
from common.redaction import is_sensitive_key, redact_event_dict

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
CUTOFF = NOW - timedelta(days=90)
PIN_SQL = "SET LOCAL enable_bitmapscan = off"


class _FakeResult:
    def __init__(self, rowcount: int | None) -> None:
        self.rowcount = rowcount


class _FakeSession:
    def __init__(self, owner: _FakeFactory) -> None:
        self._owner = owner
        self.commit = AsyncMock()

    async def execute(self, stmt: Any, *_a: Any, **_k: Any) -> _FakeResult:
        if isinstance(stmt, TextClause):  # the transaction-scoped planner pin
            self._owner.events.append(("pin", str(stmt)))
            return _FakeResult(0)
        self._owner.events.append(("delete", stmt))
        self._owner.statements.append(stmt)
        nxt = self._owner.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return _FakeResult(nxt)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeFactory:
    """``script`` entries are scripted DELETE rowcounts (or exceptions to raise),
    consumed in call order: paper batches, live batches, backtest batches."""

    def __init__(self, script: list[int | None | BaseException]) -> None:
        self.script = list(script)
        self.statements: list[Any] = []
        self.events: list[tuple[str, Any]] = []

    def __call__(self) -> _FakeSession:
        return _FakeSession(self)


def _pg(stmt: Any) -> Any:
    return stmt.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]


async def _prune(f: _FakeFactory, **kw: Any) -> int:
    return await prune_equity_snapshots(f, retention_days=90, now=NOW, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Return value / batching
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_return_value_is_sum_of_all_modes() -> None:
    f = _FakeFactory([7, 5, 4])  # paper, live, backtest
    assert await _prune(f) == 16
    assert len(f.statements) == 3


@pytest.mark.asyncio
async def test_zero_rows_returns_zero_and_logs_nothing() -> None:
    f = _FakeFactory([0, 0, 0])
    with capture_logs() as logs:
        assert await _prune(f) == 0
    assert logs == []


@pytest.mark.asyncio
async def test_full_batches_loop_until_short_batch() -> None:
    # paper 3,3,1 ; live 3,0 ; backtest 3,2
    f = _FakeFactory([3, 3, 1, 3, 0, 3, 2])
    assert await _prune(f, batch=3) == 15
    assert len(f.statements) == 7


@pytest.mark.asyncio
async def test_rejects_unsafe_arguments() -> None:
    f = _FakeFactory([])
    with pytest.raises(ValueError):
        await prune_equity_snapshots(f, retention_days=0, now=NOW)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await prune_equity_snapshots(f, retention_days=90, now=datetime(2026, 1, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await prune_equity_snapshots(f, retention_days=90, now=NOW, batch=0)  # type: ignore[arg-type]
    assert f.statements == []
    assert f.events == []


@pytest.mark.asyncio
async def test_cutoff_is_now_minus_retention_days() -> None:
    f = _FakeFactory([0, 0, 0])
    await _prune(f)
    for stmt in f.statements:
        assert CUTOFF in _pg(stmt).params.values()


@pytest.mark.asyncio
async def test_every_delete_is_preceded_by_the_planner_pin_in_its_session() -> None:
    f = _FakeFactory([2, 1, 2, 0, 1])  # batch=2: paper 2,1 ; live 2,0 ; backtest 1
    await _prune(f, batch=2)
    kinds = [k for k, _ in f.events]
    assert kinds == ["pin", "delete"] * 5
    assert all(v == PIN_SQL for k, v in f.events if k == "pin")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-1, None])
async def test_unknown_or_negative_rowcount_is_clamped_to_zero(bad: int | None) -> None:
    """WP22-P0V2-05 (SEC-03): DBAPI rowcount -1 / None means 'unknown', never negative."""
    f = _FakeFactory([bad, 4, bad])
    with capture_logs() as logs:
        total = await _prune(f)
    assert total == 4
    done = [e for e in logs if e["event"] == "equity_prune.completed"]
    assert len(done) == 1
    by_mode = (done[0]["deleted_paper"], done[0]["deleted_live"], done[0]["deleted_backtest"])
    assert by_mode == (0, 4, 0)
    assert all(v >= 0 for k, v in done[0].items() if k.startswith("deleted"))


# ---------------------------------------------------------------------------
# Logging: one event, per-mode counts (CR-05 / SEC-02)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_completed_event_with_per_mode_counts() -> None:
    # batch=3: paper 3 (full) + 2 (short) = 5 ; live 1 (short) ; backtest 0 (short)
    f = _FakeFactory([3, 2, 1, 0])
    with capture_logs() as logs:
        total = await _prune(f, batch=3)
    assert total == 6
    assert [e["event"] for e in logs] == ["equity_prune.completed"]
    ev = logs[0]
    assert ev["deleted"] == 6
    assert ev["deleted_paper"] == 5
    assert ev["deleted_live"] == 1
    assert ev["deleted_backtest"] == 0
    assert ev["cutoff"] == str(CUTOFF)
    assert ev["retention_days"] == 90
    assert ev["log_level"] == "info"


@pytest.mark.asyncio
async def test_partial_event_has_totals_so_far_then_reraises_in_live() -> None:
    boom = RuntimeError("db exploded")
    # paper 3 (full) + 1 (short) = 4 ; live: first batch raises
    f = _FakeFactory([3, 1, boom])
    with capture_logs() as logs, pytest.raises(RuntimeError) as ei:
        await _prune(f, batch=3)
    assert ei.value is boom
    assert [e["event"] for e in logs] == ["equity_prune.partial"]
    ev = logs[0]
    counts = (ev["deleted"], ev["deleted_paper"], ev["deleted_live"], ev["deleted_backtest"])
    assert counts == (4, 4, 0, 0)
    assert ev["failed_in"] == "live"
    assert ev["cutoff"] == str(CUTOFF)
    assert ev["retention_days"] == 90
    assert ev["log_level"] == "warning"


@pytest.mark.asyncio
async def test_partial_event_counts_batches_committed_before_the_failure() -> None:
    """Batch 1 of the SAME pass committed; batch 2 raises -> batch 1 is reported."""
    f = _FakeFactory([3, ValueError("x")])
    with capture_logs() as logs, pytest.raises(ValueError):
        await _prune(f, batch=3)
    ev = logs[0]
    assert ev["event"] == "equity_prune.partial"
    assert (ev["deleted"], ev["deleted_paper"], ev["failed_in"]) == (3, 3, "paper")


@pytest.mark.asyncio
async def test_partial_event_in_backtest_pass_keeps_earlier_modes() -> None:
    # batch=3: paper 2 (short) ; live 1 (short) ; backtest 3 (full) then raises
    f = _FakeFactory([2, 1, 3, RuntimeError("late")])
    with capture_logs() as logs, pytest.raises(RuntimeError):
        await _prune(f, batch=3)
    ev = logs[0]
    assert ev["event"] == "equity_prune.partial"
    assert (ev["deleted_paper"], ev["deleted_live"], ev["deleted_backtest"]) == (2, 1, 3)
    assert ev["deleted"] == 6
    assert ev["failed_in"] == "backtest"


@pytest.mark.asyncio
async def test_partial_event_cancellation_is_reported_and_propagates() -> None:
    f = _FakeFactory([3, asyncio.CancelledError()])
    with capture_logs() as logs, pytest.raises(asyncio.CancelledError):
        await _prune(f, batch=3)
    assert logs[0]["event"] == "equity_prune.partial"
    assert logs[0]["deleted"] == 3


@pytest.mark.asyncio
async def test_logs_are_cf36_clean_even_when_the_exception_text_is_sensitive() -> None:
    secret = "postgresql://trader:SUPERSECRET@db.internal:5432/prod"  # noqa: S105 - fake sentinel
    f = _FakeFactory([1, RuntimeError(f"could not connect to {secret}")])
    with capture_logs() as logs, pytest.raises(RuntimeError):
        await _prune(f, batch=1)
    ok = _FakeFactory([2, 0, 1])
    with capture_logs() as logs2:
        await _prune(ok, batch=3)
    for ev in (*logs, *logs2):
        assert "SUPERSECRET" not in repr(ev)
        assert "exc_info" not in ev and "exception" not in ev
        for k, v in ev.items():
            assert not is_sensitive_key(k), k
            assert isinstance(v, (int, str)), (k, v)
        payload = {k: v for k, v in ev.items() if k != "log_level"}
        assert redact_event_dict(None, "info", payload) == payload  # nothing to redact


# ---------------------------------------------------------------------------
# Postgres-dialect compile checks of the predicates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["paper", "live"])
def test_timestamp_stmt_pg_predicate(mode: str) -> None:
    c = _pg(build_timestamp_prune_stmt(CUTOFF, 100, mode))
    sql = str(c)
    # outer side is a pkey probe by array (planner cannot flip it to a seq scan)
    assert sql.startswith("DELETE FROM equity_snapshots WHERE equity_snapshots.id = ANY (array((")
    assert "JOIN LATERAL" in sql  # per-run, index-driven victim selection
    assert "equity_snapshots.run_id = runs.id" in sql
    assert "runs.run_mode =" in sql
    assert "equity_snapshots.timestamp <" in sql
    assert "ORDER BY equity_snapshots.timestamp" in sql
    assert "runs.created_at" not in sql
    assert sql.count(" LIMIT ") == 2  # per-run and overall
    assert CUTOFF in c.params.values()
    assert mode in c.params.values()


def test_backtest_stmt_pg_predicate() -> None:
    c = _pg(build_backtest_prune_stmt(CUTOFF, 100))
    sql = str(c)
    assert sql.startswith("DELETE FROM equity_snapshots WHERE equity_snapshots.id = ANY (array((")
    assert "JOIN LATERAL" in sql
    assert "equity_snapshots.run_id = runs.id" in sql
    assert "runs.created_at <" in sql
    assert "runs.run_mode =" in sql
    assert "ORDER BY equity_snapshots.bar_index" in sql
    # The simulated-time stamp must play no part in the backtest rule.
    assert "equity_snapshots.timestamp" not in sql
    assert sql.count(" LIMIT ") == 2
    assert CUTOFF in c.params.values()
    assert "backtest" in c.params.values()


@pytest.mark.parametrize("mode", ["backtest", "shadow", "", "PAPER"])
def test_timestamp_builder_rejects_modes_outside_the_allow_list(mode: str) -> None:
    with pytest.raises(ValueError):
        build_timestamp_prune_stmt(CUTOFF, 1, mode)


def test_only_allow_listed_modes_are_ever_targeted() -> None:
    """Fail-safe: unknown modes (e.g. 'shadow') match no statement."""
    modes: set[str] = set()
    for stmt in (
        build_timestamp_prune_stmt(CUTOFF, 1, "paper"),
        build_timestamp_prune_stmt(CUTOFF, 1, "live"),
        build_backtest_prune_stmt(CUTOFF, 1),
    ):
        for v in _pg(stmt).params.values():
            if isinstance(v, str):
                modes.add(v)
    assert modes == {"paper", "live", "backtest"}


# ---------------------------------------------------------------------------
# main.py has no inline DELETE / SQL (AST guard, WP22-P0V2-01 / CR-01)
# ---------------------------------------------------------------------------
_FORBIDDEN_CONST = re.compile(r"(?i)delete\s+from|equity_snapshots")
# WP22-P0-R2-CR-01: this guard is a TRIPWIRE against someone re-inlining the policy
# in main.py, not a security boundary (a determined author can always defeat an AST
# check). Each rule below has at least one synthetic mutant in ``_MUTANTS``.
_FORBIDDEN_CALL_ATTRS = frozenset(
    {
        "execute",
        "commit",
        "run_sync",
        "scalars",
        "scalar",
        "stream",
        "exec_driver_sql",
        "begin",
    }
)
#: Allow-list of what the loop may import from ``api.db``: the session-factory getter.
_ALLOWED_API_DB_IMPORTS = {"api.db.session": frozenset({"get_session_factory"})}
_FORBIDDEN_NAMES = frozenset({"EquitySnapshotORM", "_sa_delete", "delete", "text"})


def _find_loop(tree: ast.AST) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_equity_prune_loop":
            return node
    raise AssertionError("_equity_prune_loop not found")


def _violations(loop: ast.AsyncFunctionDef) -> list[str]:
    """Everything in the loop body that would let a prune policy live in main.py."""
    out: list[str] = []
    for node in ast.walk(loop):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _FORBIDDEN_CONST.search(node.value):
                out.append(f"string constant {node.value!r}")
            if "sqlalchemy" in node.value.lower():
                out.append(f"string constant {node.value!r}")
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0].startswith(
            "sqlalchemy"
        ):
            out.append(f"import from {node.module}")
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "api.db" or mod.startswith("api.db."):
                allowed = _ALLOWED_API_DB_IMPORTS.get(mod, frozenset())
                bad = [a.name for a in node.names if a.name not in allowed]
                if bad:
                    out.append(f"import from {mod}: {bad}")
        if isinstance(node, ast.Import):
            out.extend(
                f"import {a.name}"
                for a in node.names
                if a.name.split(".")[0].startswith("sqlalchemy")
            )
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_CALL_ATTRS:
            out.append(f".{node.attr}")
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                if isinstance(operand, ast.Attribute) and operand.attr == "timestamp":
                    out.append("inline .timestamp compare")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            out.append(f"name {node.id}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "_sf":
                out.append("session factory called inline")
            if isinstance(fn, ast.Name) and fn.id == "getattr":
                out.append("getattr call (dynamic attribute access)")
    return out


def _calls(loop: ast.AsyncFunctionDef) -> set[str]:
    return {
        n.func.id
        for n in ast.walk(loop)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_main_loop_has_no_inline_delete_or_sql() -> None:
    src = (Path(__file__).resolve().parents[2] / "apps" / "api" / "main.py").read_text()
    loop = _find_loop(ast.parse(src))
    assert _violations(loop) == []
    assert "prune_equity_snapshots" in _calls(loop)


_CLEAN = '''
async def _equity_prune_loop():
    from api.db.session import get_session_factory as _get_sf
    from api.services.retention import prune_equity_snapshots
    from common.logging import exc_log_fields
    _sf = _get_sf()
    while True:
        try:
            await prune_equity_snapshots(_sf, retention_days=1)
        except Exception as exc:
            log.warning("equity_prune.error", **exc_log_fields(exc))
            log.debug("equity_prune.error.trace", exc_info=exc)
'''

_MUTANTS = {
    # critic mutants A1-A3 (WP22-P0-CR-01) and neighbours
    "A1_text_delete_via_session_execute": '''
async def _equity_prune_loop():
    import sqlalchemy
    async with _sf() as session:
        await session.execute(sqlalchemy.text("DELETE FROM equity_snapshots WHERE 1=1"))
''',
    "A1b_execute_call": '''
async def _equity_prune_loop():
    await conn.execute(q)
''',
    "A2_import_from_sqlalchemy_sql": '''
async def _equity_prune_loop():
    from sqlalchemy.sql import delete as d
''',
    "A2b_import_sqlalchemy": '''
async def _equity_prune_loop():
    import sqlalchemy as sa
''',
    "A3_delete_string_constant": '''
async def _equity_prune_loop():
    q = "delete   FROM runs"
''',
    "A3b_equity_snapshots_constant": '''
async def _equity_prune_loop():
    t = "equity_snapshots"
''',
    "A3c_fstring_constant": '''
async def _equity_prune_loop():
    t = f"DELETE FROM {tbl}"
''',
    "A4_inline_timestamp_compare": '''
async def _equity_prune_loop():
    pred = Model.timestamp < cutoff
''',
    "A5_commit_call": '''
async def _equity_prune_loop():
    await something.commit()
''',
    "A6_dynamic_import": '''
async def _equity_prune_loop():
    m = __import__("SQLAlchemy")
''',
    "A7_session_factory_called": '''
async def _equity_prune_loop():
    s = _sf()
''',
    # WP22-P0-R2-CR-01: one mutant per new rule / forbidden attribute
    "R2a_run_sync": '''
async def _equity_prune_loop():
    await session.run_sync(Model.__table__.delete)
''',
    "R2b_scalars": '''
async def _equity_prune_loop():
    rows = await session.scalars(q)
''',
    "R2c_scalar": '''
async def _equity_prune_loop():
    n = await session.scalar(q)
''',
    "R2d_stream": '''
async def _equity_prune_loop():
    r = await session.stream(q)
''',
    "R2e_exec_driver_sql": '''
async def _equity_prune_loop():
    await conn.exec_driver_sql("select 1")
''',
    "R2f_begin": '''
async def _equity_prune_loop():
    async with engine.begin() as conn:
        pass
''',
    "R2g_getattr_concatenated_name": '''
async def _equity_prune_loop():
    await getattr(s, "exe" + "cute")(q)
''',
    "R2h_import_from_api_db_models_as_alias": '''
async def _equity_prune_loop():
    from api.db.models import EquitySnapshotORM as E
''',
    "R2i_import_api_db_package_member": '''
async def _equity_prune_loop():
    from api.db import models
''',
    "R2j_import_other_name_from_api_db_session": '''
async def _equity_prune_loop():
    from api.db.session import get_engine as ge
''',
}


def test_ast_guard_accepts_the_clean_shape() -> None:
    assert _violations(_find_loop(ast.parse(_CLEAN))) == []


@pytest.mark.parametrize("name", sorted(_MUTANTS))
def test_ast_guard_kills_each_mutant(name: str) -> None:
    assert _violations(_find_loop(ast.parse(_MUTANTS[name]))), name


def test_ast_guard_mutant_a5_removing_the_service_call_is_detected() -> None:
    no_call = _CLEAN.replace("await prune_equity_snapshots(_sf, retention_days=1)", "pass")
    assert "prune_equity_snapshots" not in _calls(_find_loop(ast.parse(no_call)))


def test_main_loop_error_path_follows_cf36_wp22_p0_r2_sec_03() -> None:
    """The loop logs type + location at WARNING and the traceback only at DEBUG."""
    src = (Path(__file__).resolve().parents[2] / "apps" / "api" / "main.py").read_text()
    loop = _find_loop(ast.parse(src))
    handler = next(
        h
        for n in ast.walk(loop)
        if isinstance(n, ast.Try)
        for h in n.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "Exception"
    )
    assert handler.name == "exc"
    calls = [n for n in ast.walk(handler) if isinstance(n, ast.Call)]
    by_event = {
        c.args[0].value: c
        for c in calls
        if c.args and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, str)
    }
    warn = by_event["equity_prune.error"]
    assert isinstance(warn.func, ast.Attribute) and warn.func.attr == "warning"
    assert any(k.arg is None for k in warn.keywords)  # **exc_log_fields(exc)
    assert {k.arg for k in warn.keywords} == {None}, "no exc_info / str(exc) at WARNING"
    trace = by_event["equity_prune.error.trace"]
    assert isinstance(trace.func, ast.Attribute) and trace.func.attr == "debug"
    assert [k.arg for k in trace.keywords] == ["exc_info"]
    assert "exc_log_fields" in {n.id for n in ast.walk(warn) if isinstance(n, ast.Name)}
