"""WP2.6 P3: static tests for migration 022 (OHLCV cache tables) and the ORM.

No database is started. Chain 021 -> 022 (ancestry-robust), table and constraint names,
rendered DDL, offline SQL, the downgrade guard (SY-26-24), ORM-vs-migration parity, the
timeframe CASE values against ``strategy_engine._TIMEFRAME_SECONDS``, the whitelist against
``TimeFrame`` minus ``1w`` and the shared identifier vectors (SY-26-23). The leaf-vs-DDL
identifier parity test belongs to P4.
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
_BARS = "ohlcv_bars"
_COV = "ohlcv_coverage"
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"
_LOCK = "LOCK TABLE ohlcv_bars, ohlcv_coverage IN ACCESS EXCLUSIVE MODE"
_BAR_COLS = "exchange symbol timeframe open_time open high low close volume".split()
_BAR_COLS += ["fetched_at", "ingest_version"]
_BAR_CHECKS = {
    "ck_ohlcv_bars_ident",
    "ck_ohlcv_bars_timeframe",
    "ck_ohlcv_bars_grid",
    "ck_ohlcv_bars_closed",
    "ck_ohlcv_bars_ohlc",
    "ck_ohlcv_bars_volume",
    "ck_ohlcv_bars_finite",
}
_COV_CHECKS = {
    "ck_ohlcv_cov_ident",
    "ck_ohlcv_cov_span",
    "ck_ohlcv_cov_timeframe",
    "ck_ohlcv_cov_grid",
}
_SYMBOL_ACCEPT = ["BTC/EUR", "A/B", "1INCH/EUR", "BTC/USD:USD", "ETH-PERP/USD"]
_SYMBOL_REJECT = [
    "",
    "/",
    "/AB",
    "AB/",
    "AB",
    "A//B",
    "A/B/C",
    "BTC|X/EUR",
    "BTC /EUR",
    "BTC/EUR\n",
    "BTC/EÜR",
]
_EXCHANGE_ACCEPT = ["coinbase", "binance", "a_1"]
_EXCHANGE_REJECT = ["", "Coinbase", "coin-base", "coin base"]


def _script() -> Any:
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(_ALEMBIC))


def _module() -> ModuleType:
    mod = _script().get_revision("022").module
    assert isinstance(mod, ModuleType)
    return mod


class _FakeOp:
    """Records what the migration asks of ``op`` and builds the tables it creates."""

    def __init__(self, rows: bool = False) -> None:
        self.rows = rows
        self.executed: list[str] = []
        self.dropped: list[str] = []
        self.tables: dict[str, sa.Table] = {}
        self.metadata = sa.MetaData()
        self.queries: list[str] = []
        self.events: list[str] = []

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
                return outer.rows

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
    rows: bool = False,
    x_args: dict[str, str] | None = None,
    offline: bool = False,
) -> tuple[ModuleType, _FakeOp]:
    mod = _module()
    fake = _FakeOp(rows)
    monkeypatch.setattr(mod, "op", fake)
    monkeypatch.setattr(mod, "context", _FakeContext(x_args or {}, offline))
    return mod, fake


def _checks(table: sa.Table) -> dict[str, str]:
    return {
        str(c.name): str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)
    }


def _upgraded(monkeypatch: pytest.MonkeyPatch) -> dict[str, sa.Table]:
    mod, fake = _patched(monkeypatch)
    mod.upgrade()  # type: ignore[attr-defined]
    return fake.tables


# ---------------------------------------------------------------------------
# Chain and files
# ---------------------------------------------------------------------------


class TestChain:
    def test_chain_021_to_022(self) -> None:
        script = _script()
        rev = script.get_revision("022")
        assert rev.revision == "022"
        assert rev.down_revision == "021"
        assert rev.branch_labels in (None, set(), ())
        assert rev.dependencies in (None, (), [])
        assert script.get_revision("021").down_revision == "020"

    def test_single_head_and_022_is_in_its_ancestry(self) -> None:
        script = _script()
        heads = script.get_heads()
        assert len(heads) == 1
        ancestry = {r.revision for r in script.walk_revisions(base="base", head=heads[0])}
        assert "022" in ancestry

    def test_file_name_and_infra_note(self) -> None:
        assert (_ALEMBIC / "versions" / "022_add_ohlcv_cache.py").is_file()
        assert "022_add_ohlcv_cache.py" in (_REPO / "infra" / "CLAUDE.md").read_text()


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------


class TestUpgrade:
    def test_lock_timeout_first_and_two_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch)
        mod.upgrade()  # type: ignore[attr-defined]
        assert fake.executed == [_LOCK_TIMEOUT]
        assert list(fake.tables) == [_BARS, _COV]

    def test_bars_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _upgraded(monkeypatch)[_BARS]
        assert [c.name for c in table.columns] == _BAR_COLS
        assert [
            c.name for c in table.primary_key.columns
        ] == "exchange symbol timeframe open_time".split()
        assert table.primary_key.name == "pk_ohlcv_bars"
        assert set(_checks(table)) == _BAR_CHECKS

    def test_coverage_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _upgraded(monkeypatch)[_COV]
        assert [
            c.name for c in table.columns
        ] == "exchange symbol timeframe covered_from covered_to verified_at".split()
        assert [c.name for c in table.primary_key.columns] == ["exchange", "symbol", "timeframe"]
        assert table.primary_key.name == "pk_ohlcv_coverage"
        assert set(_checks(table)) == _COV_CHECKS

    def test_no_fk_no_index_all_not_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for table in _upgraded(monkeypatch).values():
            assert not table.foreign_keys
            assert not table.indexes
            assert all(not c.nullable for c in table.columns)

    def test_emitted_ddl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tables = _upgraded(monkeypatch)
        bars = " ".join(
            str(CreateTable(tables[_BARS]).compile(dialect=postgresql.dialect())).split()
        )
        cov = " ".join(str(CreateTable(tables[_COV]).compile(dialect=postgresql.dialect())).split())
        for col in ("open", "high", "low", "close"):
            assert f"{col} NUMERIC(28, 12) NOT NULL" in bars
        assert "volume NUMERIC(30, 12) NOT NULL" in bars
        assert "fetched_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in bars
        assert "ingest_version SMALLINT DEFAULT 1 NOT NULL" in bars
        assert "exchange VARCHAR(32) NOT NULL" in bars and "TIMESTAMP WITH TIME ZONE" in cov
        for sql in (bars, cov):
            assert "FOREIGN KEY" not in sql and "UPDATE" not in sql and "DELETE" not in sql

    def test_checks_use_cast_numeric_and_no_time_zone_clause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tables = _upgraded(monkeypatch)
        bars, cov = _checks(tables[_BARS]), _checks(tables[_COV])
        assert bars["ck_ohlcv_bars_grid"].startswith(
            "mod(CAST(extract(epoch FROM open_time) AS numeric), CASE timeframe"
        )
        assert bars["ck_ohlcv_bars_closed"].startswith(
            "CAST(extract(epoch FROM fetched_at) AS numeric) >= "
            "CAST(extract(epoch FROM open_time) AS numeric) + CASE timeframe"
        )
        assert cov["ck_ohlcv_cov_grid"].count("CAST(extract(epoch FROM ") == 2
        for text in [*bars.values(), *cov.values()]:
            assert "AT TIME ZONE" not in text
            assert text.count("extract(epoch") == text.count("CAST(extract(epoch")

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
        assert sql.index(_LOCK_TIMEOUT) < sql.index("CREATE TABLE ohlcv_bars")
        assert sql.index("CREATE TABLE ohlcv_bars") < sql.index("CREATE TABLE ohlcv_coverage")
        assert (
            "CONSTRAINT pk_ohlcv_bars PRIMARY KEY (exchange, symbol, timeframe, open_time)" in sql
        )
        assert "CONSTRAINT pk_ohlcv_coverage PRIMARY KEY (exchange, symbol, timeframe)" in sql
        for name in _BAR_CHECKS | _COV_CHECKS:
            assert f"CONSTRAINT {name} CHECK" in sql


# ---------------------------------------------------------------------------
# Downgrade guard (SY-26-24)
# ---------------------------------------------------------------------------


class TestDowngradeGuard:
    def test_refuses_while_any_row_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, rows=True)
        with pytest.raises(RuntimeError, match="allow_drop_ohlcv=1"):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == []
        query = " ".join(fake.queries[0].split())
        assert query == (
            "SELECT EXISTS (SELECT 1 FROM ohlcv_bars) OR EXISTS (SELECT 1 FROM ohlcv_coverage)"
        )

    def test_both_tables_locked_before_the_check_then_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod, fake = _patched(monkeypatch)
        mod.downgrade()  # type: ignore[attr-defined]
        assert fake.events == [
            _LOCK_TIMEOUT,
            _LOCK,
            "EXISTS-QUERY",
            f"DROP {_COV}",
            f"DROP {_BARS}",
        ]

    def test_refusal_happens_after_the_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, rows=True)
        with pytest.raises(RuntimeError):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.events == [_LOCK_TIMEOUT, _LOCK, "EXISTS-QUERY"]

    def test_override_flag_skips_inspection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, rows=True, x_args={"allow_drop_ohlcv": "1"})
        mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == [_COV, _BARS]
        assert fake.queries == [] and fake.executed == [_LOCK_TIMEOUT]

    @pytest.mark.parametrize("value", ["0", "true", "", "yes", "2"])
    def test_only_the_exact_flag_value_overrides(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        mod, fake = _patched(monkeypatch, rows=True, x_args={"allow_drop_ohlcv": value})
        with pytest.raises(RuntimeError):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == []

    def test_offline_downgrade_needs_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod, fake = _patched(monkeypatch, offline=True)
        with pytest.raises(RuntimeError, match="offline"):
            mod.downgrade()  # type: ignore[attr-defined]
        assert fake.dropped == [] and fake.queries == []
        assert fake.executed == [_LOCK_TIMEOUT]
        mod2, fake2 = _patched(monkeypatch, offline=True, x_args={"allow_drop_ohlcv": "1"})
        mod2.downgrade()  # type: ignore[attr-defined]
        assert fake2.dropped == [_COV, _BARS]

    def test_real_offline_render_with_and_without_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext

        mod = _module()

        def render() -> str:
            buf = io.StringIO()
            ctx = MigrationContext.configure(
                dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buf}
            )
            with Operations.context(ctx):
                mod.downgrade()  # type: ignore[attr-defined]
            return " ".join(buf.getvalue().split())

        monkeypatch.setattr(mod, "context", _FakeContext({}, offline=True))
        with pytest.raises(RuntimeError, match="offline"):
            render()
        monkeypatch.setattr(mod, "context", _FakeContext({"allow_drop_ohlcv": "1"}, offline=True))
        sql = render()
        assert sql.index("DROP TABLE ohlcv_coverage") < sql.index("DROP TABLE ohlcv_bars")


# ---------------------------------------------------------------------------
# Timeframe values, whitelist and identifier vectors
# ---------------------------------------------------------------------------


class TestPinnedValues:
    def _all_checks(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        tables = _upgraded(monkeypatch)
        return {**_checks(tables[_BARS]), **_checks(tables[_COV])}

    def test_case_values_equal_strategy_engine_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from common.types import TimeFrame
        from trading.strategy_engine import _TIMEFRAME_SECONDS

        expected = {
            tf.value: s for tf, s in _TIMEFRAME_SECONDS.items() if tf is not TimeFrame.ONE_WEEK
        }
        checks = self._all_checks(monkeypatch)
        n_case = 0
        for name in ("ck_ohlcv_bars_grid", "ck_ohlcv_bars_closed", "ck_ohlcv_cov_grid"):
            for case in re.findall(r"CASE timeframe (.*?) END", checks[name]):
                pairs = re.findall(r"WHEN '(\w+)' THEN (\d+)", case)
                assert {k: int(v) for k, v in pairs} == expected
                assert len(pairs) == len(expected)
                n_case += 1
        assert n_case == 4  # bars grid, bars closed, two in coverage grid

    def test_whitelist_is_timeframe_minus_1w(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from common.types import TimeFrame

        expected = {tf.value for tf in TimeFrame} - {"1w"}
        checks = self._all_checks(monkeypatch)
        for name in ("ck_ohlcv_bars_timeframe", "ck_ohlcv_cov_timeframe"):
            assert set(re.findall(r"'(\w+)'", checks[name])) == expected
        assert "1w" not in " ".join(checks.values())

    def test_ident_vectors_through_the_ddl_patterns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checks = self._all_checks(monkeypatch)
        assert checks["ck_ohlcv_bars_ident"] == checks["ck_ohlcv_cov_ident"]
        exch, sym = re.findall(r"~ '\^(.+?)\$'", checks["ck_ohlcv_bars_ident"])
        assert (exch, sym) == ("[a-z0-9_]+", "[!-.0-{}~]+/[!-.0-{}~]+")
        for ok in _SYMBOL_ACCEPT:
            assert re.fullmatch(sym, ok), ok
        for bad in _SYMBOL_REJECT:
            assert not re.fullmatch(sym, bad), bad
        for ok in _EXCHANGE_ACCEPT:
            assert re.fullmatch(exch, ok), ok
        for bad in _EXCHANGE_REJECT:
            assert not re.fullmatch(exch, bad), bad


# ---------------------------------------------------------------------------
# ORM parity
# ---------------------------------------------------------------------------


def _column_shape(table: sa.Table) -> dict[str, tuple[str, bool, str | None, str | None]]:
    out: dict[str, tuple[str, bool, str | None, str | None]] = {}
    for col in table.columns:
        default = col.server_default
        arg = getattr(default, "arg", None)
        default_text = str(arg.compile(dialect=postgresql.dialect())) if arg is not None else None
        out[col.name] = (
            col.type.compile(dialect=postgresql.dialect()),
            bool(col.nullable),
            default_text,
            col.comment,
        )
    return out


class TestOrmParity:
    @pytest.mark.parametrize(
        ("orm_name", "table_name"), [("OhlcvBarORM", _BARS), ("OhlcvCoverageORM", _COV)]
    )
    def test_orm_matches_the_migration(
        self, monkeypatch: pytest.MonkeyPatch, orm_name: str, table_name: str
    ) -> None:
        from api.db import models

        migrated = _upgraded(monkeypatch)[table_name]
        orm = getattr(models, orm_name).__table__
        assert orm.name == migrated.name == table_name
        assert list(_column_shape(orm).items()) == list(_column_shape(migrated).items())
        assert [c.name for c in orm.primary_key.columns] == [
            c.name for c in migrated.primary_key.columns
        ]
        assert orm.primary_key.name == migrated.primary_key.name
        assert _checks(orm) == _checks(migrated)
        assert not orm.foreign_keys and not orm.indexes

    def test_numeric_precision_and_registration(self) -> None:
        from api.db import Base, models

        table = models.OhlcvBarORM.__table__
        for col in ("open", "high", "low", "close"):
            t = table.c[col].type
            assert isinstance(t, sa.Numeric) and (t.precision, t.scale) == (28, 12)
        vol = table.c.volume.type
        assert isinstance(vol, sa.Numeric) and (vol.precision, vol.scale) == (30, 12)
        for name, orm in ((_BARS, models.OhlcvBarORM), (_COV, models.OhlcvCoverageORM)):
            assert Base.metadata.tables[name] is orm.__table__
            assert orm.__name__ in models.__all__
