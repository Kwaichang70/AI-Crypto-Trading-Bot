"""WP2.4 P5: migration 021 and the ORM (T-34, T-35).

* T-34: static migration test. Chain 020 -> 021, table and constraint names, the emitted
  DDL, ``SET LOCAL lock_timeout``, and the downgrade guard on ``recorded`` rows
  (``-x allow_drop_context=1``). No database is started.
* T-35: the ORM metadata equals what the migration creates, and ``env.py`` sees the model.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

_REPO = Path(__file__).resolve().parents[2]
_ALEMBIC = _REPO / "infra" / "alembic"
_TABLE = "context_observations"
_LOCK = "LOCK TABLE context_observations IN ACCESS EXCLUSIVE MODE"


def _script() -> Any:
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(_ALEMBIC))


def _module() -> ModuleType:
    mod = _script().get_revision("021").module
    assert isinstance(mod, ModuleType)
    return mod


class _FakeOp:
    """Records what the migration asks of ``op`` and builds the table it creates."""

    def __init__(self, recorded_rows: bool = False) -> None:
        self.recorded_rows = recorded_rows
        self.executed: list[str] = []
        self.dropped: list[str] = []
        self.tables: dict[str, sa.Table] = {}
        self.metadata = sa.MetaData()
        self.queries: list[str] = []
        self.events: list[str] = []  # ordered: executes, the EXISTS query, the drop

    def execute(self, sql: str) -> None:
        self.executed.append(str(sql))
        self.events.append(str(sql))

    def create_table(self, name: str, *items: Any, **kw: Any) -> sa.Table:
        table = sa.Table(name, self.metadata, *items, **kw)
        self.tables[name] = table
        return table

    def drop_table(self, name: str) -> None:
        self.dropped.append(name)
        self.events.append(f"DROP {name}")

    def get_bind(self) -> Any:
        outer = self

        class _Result:
            def scalar(self) -> bool:
                return outer.recorded_rows

        class _Bind:
            def execute(self, stmt: Any) -> _Result:
                outer.queries.append(str(stmt))
                outer.events.append("EXISTS-QUERY")
                return _Result()

        return _Bind()


class _FakeContext:
    def __init__(self, x_args: dict[str, str], offline: bool = False) -> None:
        self._x = x_args
        self._offline = offline

    def get_x_argument(self, as_dictionary: bool = False) -> dict[str, str]:
        assert as_dictionary is True
        return dict(self._x)

    def is_offline_mode(self) -> bool:
        return self._offline


def _patched(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recorded_rows: bool = False,
    x_args: dict[str, str] | None = None,
    offline: bool = False,
) -> tuple[ModuleType, _FakeOp]:
    mod = _module()
    fake = _FakeOp(recorded_rows)
    monkeypatch.setattr(mod, "op", fake)
    monkeypatch.setattr(mod, "context", _FakeContext(x_args or {}, offline))
    return mod, fake


# ---------------------------------------------------------------------------
# T-34 chain and files
# ---------------------------------------------------------------------------


class TestT34Chain:
    def test_chain_020_to_021(self) -> None:
        script = _script()
        rev = script.get_revision("021")
        assert rev.revision == "021"
        assert rev.down_revision == "020"
        assert rev.branch_labels in (None, set(), ())
        assert rev.dependencies in (None, (), [])
        assert script.get_revision("020").down_revision == "019"

    def test_single_head_and_021_is_in_its_ancestry(self) -> None:
        script = _script()
        heads = script.get_heads()
        assert len(heads) == 1
        ancestry = {r.revision for r in script.walk_revisions(base="base", head=heads[0])}
        assert "021" in ancestry

    def test_file_name_and_location(self) -> None:
        assert (_ALEMBIC / "versions" / "021_add_context_observations.py").is_file()
        assert not (_REPO / "infra" / "migrations").exists()

    def test_infra_claude_md_names_the_real_path(self) -> None:
        text = (_REPO / "infra" / "CLAUDE.md").read_text()
        assert "alembic/versions/" in text
        assert not re.search(r"^- `migrations/`", text, flags=re.MULTILINE)


class TestT34Upgrade:
    def test_lock_timeout_is_the_first_statement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch)
        mod.upgrade()
        assert fake.executed == ["SET LOCAL lock_timeout = '5s'"]
        assert list(fake.tables) == [_TABLE]

    def test_table_columns_constraints_and_no_foreign_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod, fake = _patched(monkeypatch)
        mod.upgrade()
        table = fake.tables[_TABLE]
        assert [c.name for c in table.columns] == [
            "series_id",
            "observed_at",
            "available_at",
            "value",
            "source",
            "available_at_basis",
            "ingested_at",
        ]
        assert [c.name for c in table.primary_key.columns] == [
            "series_id",
            "observed_at",
            "available_at",
        ]
        assert table.primary_key.name == "pk_context_observations"
        assert not table.foreign_keys
        assert not table.indexes  # the PK index serves the as-of lookup (WP24-DB-04)
        assert all(not c.nullable for c in table.columns)
        checks = {
            c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)
        }
        assert checks == {
            "ck_context_obs_avail_after_obs": "available_at >= observed_at",
            "ck_context_obs_value_finite": "value <> 'NaN'",
            "ck_context_obs_basis": (
                "available_at_basis IN ('recorded', 'estimated_lag', 'vintage')"
            ),
            "ck_context_obs_series_id": "series_id <> ''",
        }

    def test_emitted_ddl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch)
        mod.upgrade()
        ddl = " ".join(
            str(CreateTable(fake.tables[_TABLE]).compile(dialect=postgresql.dialect())).split()
        )
        assert "series_id VARCHAR(64) NOT NULL" in ddl
        assert "observed_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl
        assert "available_at TIMESTAMP WITH TIME ZONE NOT NULL" in ddl
        assert "value NUMERIC(24, 8) NOT NULL" in ddl
        assert "source VARCHAR(32) NOT NULL" in ddl
        assert "available_at_basis VARCHAR(16) NOT NULL" in ddl
        assert "ingested_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in ddl
        assert (
            "CONSTRAINT pk_context_observations PRIMARY KEY (series_id, observed_at, available_at)"
        ) in ddl
        assert "FOREIGN KEY" not in ddl
        assert " UPDATE " not in ddl and "DELETE" not in ddl

    def test_upgrade_in_real_offline_mode(self) -> None:
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext

        buf = io.StringIO()
        ctx = MigrationContext.configure(
            dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buf}
        )
        with Operations.context(ctx):
            _module().upgrade()  # type: ignore[attr-defined]
        sql = " ".join(buf.getvalue().split())
        assert sql.index("SET LOCAL lock_timeout = '5s'") < sql.index(
            "CREATE TABLE context_observations"
        )
        assert "PRIMARY KEY (series_id, observed_at, available_at)" in sql


class TestT34DowngradeGuard:
    def test_refuses_while_recorded_rows_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, recorded_rows=True)
        with pytest.raises(RuntimeError, match="allow_drop_context=1"):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == []
        assert len(fake.queries) == 1
        query = " ".join(fake.queries[0].split())
        assert "available_at_basis = 'recorded'" in query
        assert query.startswith("SELECT EXISTS")

    def test_drops_when_no_recorded_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, recorded_rows=False)
        mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == [_TABLE]
        assert fake.executed == ["SET LOCAL lock_timeout = '5s'", _LOCK]

    def test_the_table_is_locked_before_the_recorded_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DB-R-06: no recorder may insert between the EXISTS check and the DROP."""
        mod, fake = _patched(monkeypatch, recorded_rows=False)
        mod.downgrade()  # type: ignore[attr-defined]
        assert fake.events == [
            "SET LOCAL lock_timeout = '5s'",
            _LOCK,
            "EXISTS-QUERY",
            f"DROP {_TABLE}",
        ]

    def test_the_refusal_also_happens_after_the_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, recorded_rows=True)
        with pytest.raises(RuntimeError):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.events == ["SET LOCAL lock_timeout = '5s'", _LOCK, "EXISTS-QUERY"]

    def test_override_flag_skips_the_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, recorded_rows=True, x_args={"allow_drop_context": "1"})
        mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == [_TABLE]
        assert fake.queries == []
        assert _LOCK not in fake.executed  # DROP TABLE takes the same lock itself

    @pytest.mark.parametrize("value", ["0", "true", "", "yes", "2"])
    def test_only_the_exact_flag_value_overrides(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        mod, fake = _patched(monkeypatch, recorded_rows=True, x_args={"allow_drop_context": value})
        with pytest.raises(RuntimeError):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == []

    def test_offline_downgrade_needs_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, offline=True)
        with pytest.raises(RuntimeError, match="offline"):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == [] and fake.queries == []
        mod2, fake2 = _patched(monkeypatch, offline=True, x_args={"allow_drop_context": "1"})
        mod2.downgrade()  # type: ignore[attr-defined]
        assert fake2.dropped == [_TABLE]
        assert fake2.executed == ["SET LOCAL lock_timeout = '5s'"]  # offline: nothing to lock

    def test_error_message_is_fixed_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, _ = _patched(monkeypatch, recorded_rows=True)
        with pytest.raises(RuntimeError) as info:
            mod.downgrade()  # type: ignore[attr-defined]
        assert "recorded" in str(info.value)
        assert "postgres" not in str(info.value).lower()


# ---------------------------------------------------------------------------
# T-35 ORM parity and env.py
# ---------------------------------------------------------------------------


def _column_shape(table: sa.Table) -> dict[str, tuple[str, bool, str | None]]:
    out: dict[str, tuple[str, bool, str | None]] = {}
    for col in table.columns:
        default = col.server_default
        default_text = None
        if default is not None:
            arg = getattr(default, "arg", None)
            default_text = (
                str(arg.compile(dialect=postgresql.dialect())) if arg is not None else None
            )
        out[col.name] = (
            col.type.compile(dialect=postgresql.dialect()),
            bool(col.nullable),
            default_text,
        )
    return out


class TestT35OrmParity:
    def test_orm_matches_the_migration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from api.db.models import ContextObservationORM

        mod, fake = _patched(monkeypatch)
        mod.upgrade()
        migrated = fake.tables[_TABLE]
        orm = ContextObservationORM.__table__
        assert isinstance(orm, sa.Table)
        assert orm.name == migrated.name == _TABLE
        assert _column_shape(orm) == _column_shape(migrated)
        assert [c.name for c in orm.columns] == [c.name for c in migrated.columns]
        assert [c.name for c in orm.primary_key.columns] == [
            c.name for c in migrated.primary_key.columns
        ]
        assert orm.primary_key.name == migrated.primary_key.name

        def checks(table: sa.Table) -> dict[str, str]:
            return {
                str(c.name): str(c.sqltext)
                for c in table.constraints
                if isinstance(c, sa.CheckConstraint)
            }

        assert checks(orm) == checks(migrated)
        # DB-R-07(b): the migration carries the same column comments as the ORM, so
        # autogenerate reports no modify_comment drift for this table.
        assert {c.name: c.comment for c in orm.columns} == {
            c.name: c.comment for c in migrated.columns
        }
        assert not orm.foreign_keys
        assert not orm.indexes

    def test_numeric_is_24_8_decimal_and_timestamps_are_tz_aware(self) -> None:
        from api.db.models import ContextObservationORM

        table = ContextObservationORM.__table__
        value = table.c.value.type
        assert isinstance(value, sa.Numeric)
        assert (value.precision, value.scale, value.asdecimal) == (24, 8, True)
        for name in ("observed_at", "available_at", "ingested_at"):
            col_type = table.c[name].type
            assert isinstance(col_type, sa.DateTime) and col_type.timezone is True

    def test_registered_on_base_metadata_through_the_env_import(self) -> None:
        # env.py does `from api.db import Base` and uses `Base.metadata`.
        from api.db import Base
        from api.db.models import ContextObservationORM

        assert Base.metadata.tables[_TABLE] is ContextObservationORM.__table__
        assert "ContextObservationORM" in __import__("api.db.models", fromlist=["__all__"]).__all__

    def test_env_py_reads_the_shared_metadata(self) -> None:
        text = (_ALEMBIC / "env.py").read_text()
        assert "from api.db import Base" in text
        assert "target_metadata = Base.metadata" in text

    def test_orm_class_is_append_only_by_declaration(self) -> None:
        from api.db.models import ContextObservationORM

        assert not hasattr(ContextObservationORM, "updated_at")
        assert not hasattr(ContextObservationORM, "revision")
