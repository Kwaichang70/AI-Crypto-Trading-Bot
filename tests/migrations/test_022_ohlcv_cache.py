"""WP2.6 P3: migration 022 against a real Postgres (``MIGRATION_TEST_DATABASE_URL``).

Skipped without the DSN (same convention as ``test_020_*``). Proves that PostgreSQL accepts
every CHECK expression, that each CHECK rejects its violation, that the results do not depend
on the session ``TimeZone``, that the identifier regex behaves like the shared Python vectors
(SY-26-23), the downgrade guard (both tables) and a round trip to the explicit revision 021.
Every test removes its own rows. DSN never printed.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")
_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "infra" / "alembic"
requires_postgres = pytest.mark.skipif(
    not _URL, reason="MIGRATION_TEST_DATABASE_URL not set -- needs a real Postgres"
)
pytestmark = requires_postgres

_T0 = datetime(2024, 1, 1, tzinfo=UTC)  # on every timeframe grid
_BAR_SQL = (
    "INSERT INTO ohlcv_bars (exchange, symbol, timeframe, open_time, open, high, low, close,"
    " volume, fetched_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
)
_COV_SQL = (
    "INSERT INTO ohlcv_coverage (exchange, symbol, timeframe, covered_from, covered_to)"
    " VALUES ($1, $2, $3, $4, $5)"
)
_TF_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}
_TF_SECONDS |= {"1d": 86400}
_ZONES = ["UTC", "America/Los_Angeles", "Asia/Kolkata"]
_SYMBOL_ACCEPT = ["BTC/EUR", "A/B", "1INCH/EUR", "BTC/USD:USD", "ETH-PERP/USD"]
_SYMBOL_REJECT = ["", "/", "/AB", "AB/", "AB", "A//B", "A/B/C", "BTC|X/EUR", "BTC /EUR"]
_SYMBOL_REJECT += ["BTC/EUR\n", "BTC/EÜR"]
_EXCH_ACCEPT = ["coinbase", "binance", "a_1"]
_EXCH_REJECT = ["", "Coinbase", "coin-base", "coin base"]


def _cfg(x_args: list[str] | None = None) -> Any:
    from alembic.config import Config

    assert _URL is not None
    os.environ["DATABASE_URL"] = _URL
    from api.config import get_settings

    get_settings.cache_clear()
    cfg = Config(str(_ALEMBIC_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    cfg.cmd_opts = SimpleNamespace(x=x_args or [])
    return cfg


def _dsn() -> str:
    assert _URL is not None
    return _URL.replace("postgresql+asyncpg://", "postgresql://")


async def _run(sql: str, *args: Any, tz: str | None = None) -> tuple[str | None, str | None]:
    """Execute one statement; return (sqlstate, constraint) of the failure, or (None, None)."""
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        if tz is not None:
            await conn.execute(f"SET TIME ZONE '{tz}'")
        try:
            await conn.execute(sql, *args)
        except asyncpg.PostgresError as exc:
            return exc.sqlstate, getattr(exc, "constraint_name", None)
        return None, None
    finally:
        await conn.close()


async def _val(sql: str, *args: Any) -> Any:
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


def _bar(**over: Any) -> list[Any]:
    row: dict[str, Any] = {
        "exchange": "coinbase", "symbol": "BTC/EUR", "timeframe": "1h", "open_time": _T0,
        "open": Decimal(100), "high": Decimal(110), "low": Decimal(90), "close": Decimal(105),
        "volume": Decimal(5),
    }  # fmt: skip
    row.update(over)
    row.setdefault(
        "fetched_at", row["open_time"] + timedelta(seconds=_TF_SECONDS.get(row["timeframe"], 3600))
    )
    return list(row.values())


def _insert_bar(tz: str | None = None, **over: Any) -> tuple[str | None, str | None]:
    return asyncio.run(_run(_BAR_SQL, *_bar(**over), tz=tz))


def _cov(*args: Any, tz: str | None = None) -> tuple[str | None, str | None]:
    return asyncio.run(_run(_COV_SQL, *args, tz=tz))


def _clean() -> None:
    asyncio.run(_run("DELETE FROM ohlcv_bars"))
    asyncio.run(_run("DELETE FROM ohlcv_coverage"))


@pytest.fixture
def at_head() -> Any:
    from alembic import command

    command.upgrade(_cfg(), "head")
    _clean()
    yield
    command.upgrade(_cfg(), "head")
    _clean()


# ---------------------------------------------------------------------------
# Types and constraint inventory
# ---------------------------------------------------------------------------
def test_types_defaults_and_constraint_names(at_head: None) -> None:
    rows = asyncio.run(
        _val(
            "SELECT json_object_agg(table_name || '.' || column_name, json_build_array("
            "data_type, numeric_precision, numeric_scale, is_nullable, column_default)) "
            "FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name IN ('ohlcv_bars', 'ohlcv_coverage')"
        )
    )
    import json

    cols = json.loads(rows)
    for c in ("open", "high", "low", "close"):
        assert cols[f"ohlcv_bars.{c}"][:4] == ["numeric", 28, 12, "NO"]
    assert cols["ohlcv_bars.volume"][:4] == ["numeric", 30, 12, "NO"]
    assert cols["ohlcv_bars.open_time"][0] == "timestamp with time zone"
    assert cols["ohlcv_bars.ingest_version"][0] == "smallint"
    assert cols["ohlcv_bars.ingest_version"][4] == "1"
    assert cols["ohlcv_bars.fetched_at"][4] == "now()"
    assert cols["ohlcv_coverage.covered_to"][0] == "timestamp with time zone"
    names = asyncio.run(
        _val(
            "SELECT array_agg(conname ORDER BY conname) FROM pg_constraint WHERE contype = 'c' "
            "AND conrelid IN ('ohlcv_bars'::regclass, 'ohlcv_coverage'::regclass)"
        )
    )
    assert len(names) == 11 and all(
        n.startswith(("ck_ohlcv_bars_", "ck_ohlcv_cov_")) for n in names
    )


# ---------------------------------------------------------------------------
# Every CHECK accepts good rows and rejects its violation
# ---------------------------------------------------------------------------
_M = timedelta(minutes=1)
_B = "ck_ohlcv_bars_"
# fmt: off
_VIOLATIONS: list[tuple[str, dict[str, Any], str]] = [
    ("exchange_upper", {"exchange": "Coinbase"}, _B + "ident"),
    ("tf_unknown", {"timeframe": "2m"}, _B + "timeframe"),
    ("tf_1w", {"timeframe": "1w"}, _B + "timeframe"),
    ("grid_5m_at_03", {"timeframe": "5m", "open_time": _T0 + 3 * _M}, _B + "grid"),
    ("grid_subsecond", {"timeframe": "1m", "open_time": _T0 + _M / 60000}, _B + "grid"),
    ("closed_1s_early", {"fetched_at": _T0 + timedelta(seconds=3599)}, _B + "closed"),
    ("low_gt_high", {"low": Decimal(120)}, _B + "ohlc"),
    ("low_zero", {"low": Decimal(0)}, _B + "ohlc"),
    ("low_gt_open", {"low": Decimal(101)}, _B + "ohlc"),
    ("close_gt_high", {"close": Decimal(111)}, _B + "ohlc"),
    ("open_gt_high", {"high": Decimal(99)}, _B + "ohlc"),
    ("volume_negative", {"volume": Decimal(-1)}, _B + "volume"),
    *[(f"nan_{c}", {c: Decimal("NaN")}, _B + "finite")
      for c in ("open", "high", "low", "close", "volume")],
]
# fmt: on


@pytest.mark.parametrize(
    ("over", "constraint"), [(v[1], v[2]) for v in _VIOLATIONS], ids=[v[0] for v in _VIOLATIONS]
)
def test_each_bar_check_rejects_its_violation(
    at_head: None, over: dict[str, Any], constraint: str
) -> None:
    assert _insert_bar(**over) == ("23514", constraint)


def test_valid_rows_duplicate_pk_and_range_errors(at_head: None) -> None:
    assert _insert_bar() == (None, None)
    assert _insert_bar() == ("23505", "pk_ohlcv_bars")
    assert _insert_bar(timeframe="1m") == (None, None)  # same bar time, other series key
    assert _insert_bar(symbol="A" * 33 + "/EUR")[0] == "22001"  # VARCHAR(32), not a CHECK
    assert _insert_bar(timeframe="5m", high=Decimal(10) ** 16)[0] == "22003"
    assert _insert_bar(timeframe="5m", volume=Decimal(10) ** 18)[0] == "22003"
    assert _insert_bar(timeframe="15m", fetched_at=_T0 + 15 * _M) == (None, None)  # closed exactly


def test_boundary_rows_are_accepted(at_head: None) -> None:
    """Equality edges of ck_ohlcv_bars_volume and ck_ohlcv_bars_ohlc are accepted."""
    flat = {"open": Decimal(100), "high": Decimal(100), "low": Decimal(100), "close": Decimal(100)}
    assert _insert_bar(volume=Decimal(0), **flat) == (None, None)  # zero-volume flat bar
    _clean()
    # low == open and close == high
    assert _insert_bar(low=Decimal(100), close=Decimal(110)) == (None, None)


_GRID: list[tuple[str, int, bool]] = [
    (tf, minute, minute % (s // 60) == 0 and s <= 3600)
    for tf, s in _TF_SECONDS.items()
    if s <= 3600
    for minute in (0, 1, 3, 5, 15, 30, 45)
] + [("4h", 60, False), ("4h", 240, True), ("1d", 240, False), ("1d", 1440, True)]


@pytest.mark.parametrize("tz", _ZONES)
def test_grid_and_closed_are_independent_of_session_time_zone(at_head: None, tz: str) -> None:
    for tf, minute, on_grid in _GRID:
        got = _insert_bar(tz=tz, timeframe=tf, open_time=_T0 + timedelta(minutes=minute))
        assert got == ((None, None) if on_grid else ("23514", "ck_ohlcv_bars_grid")), (tf, minute)
        _clean()
    # A naive literal is read in the session zone; the CHECK still judges the instant.
    for wall in ("2024-01-01 05:30:00", "2024-01-01 08:00:00", "2024-01-01 00:00:00"):
        sql = (
            "INSERT INTO ohlcv_bars (exchange, symbol, timeframe, open_time, open, high, low,"  # noqa: S608
            f" close, volume, fetched_at) VALUES ('coinbase', 'BTC/EUR', '1d',"
            f" '{wall}'::timestamptz, 100, 110, 90, 105, 5, '2024-01-10 00:00:00+00')"
        )
        got = asyncio.run(_run(sql, tz=tz))
        want_ok = _wall_is_utc_midnight(wall, tz)
        assert got == ((None, None) if want_ok else ("23514", "ck_ohlcv_bars_grid")), (wall, tz)
        _clean()


def _wall_is_utc_midnight(wall: str, tz: str) -> bool:
    from zoneinfo import ZoneInfo

    local = datetime.fromisoformat(wall).replace(tzinfo=ZoneInfo(tz))
    return local.astimezone(UTC).timestamp() % 86400 == 0


# ---------------------------------------------------------------------------
# Identifier regex behaves like the shared Python vectors (SY-26-23)
# ---------------------------------------------------------------------------
def test_ident_check_matches_the_python_vectors(at_head: None) -> None:
    from api.db.models import OhlcvBarORM

    ident = {str(c.name): str(c.sqltext) for c in OhlcvBarORM.__table__.constraints
             if c.__class__.__name__ == "CheckConstraint"}["ck_ohlcv_bars_ident"]  # fmt: skip
    exch_re, sym_re = re.findall(r"~ '\^(.+?)\$'", ident)
    for sym in _SYMBOL_ACCEPT + _SYMBOL_REJECT:
        py_ok = re.fullmatch(sym_re, sym) is not None
        got = _insert_bar(symbol=sym)
        assert got == ((None, None) if py_ok else ("23514", "ck_ohlcv_bars_ident")), repr(sym)
        assert py_ok == (sym in _SYMBOL_ACCEPT), repr(sym)
        _clean()
    for ex in _EXCH_ACCEPT + _EXCH_REJECT:
        py_ok = re.fullmatch(exch_re, ex) is not None
        got = _insert_bar(exchange=ex)
        assert got == ((None, None) if py_ok else ("23514", "ck_ohlcv_bars_ident")), repr(ex)
        assert py_ok == (ex in _EXCH_ACCEPT), repr(ex)
        _clean()
    assert _insert_bar(exchange="x" * 33)[0] == "22001"


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------
_H = timedelta(hours=1)


@pytest.mark.parametrize(
    ("args", "want"),
    [
        (("coinbase", "BTC/EUR", "1h", _T0, _T0 + _H), (None, None)),
        (("coinbase", "BTC/EUR", "1h", _T0, _T0), (None, None)),
        (("coinbase", "BTC/EUR", "1h", _T0 + _H, _T0), ("23514", "ck_ohlcv_cov_span")),
        (("coinbase", "BTC/EUR", "1h", _T0 + _H // 2, _T0 + _H), ("23514", "ck_ohlcv_cov_grid")),
        (("coinbase", "BTC/EUR", "1h", _T0, _T0 + _H // 2), ("23514", "ck_ohlcv_cov_grid")),
        (("coinbase", "BTC/EUR", "1w", _T0, _T0 + _H), ("23514", "ck_ohlcv_cov_timeframe")),
        (("coinbase", "/EUR", "1h", _T0, _T0 + _H), ("23514", "ck_ohlcv_cov_ident")),
        (("Coinbase", "BTC/EUR", "1h", _T0, _T0 + _H), ("23514", "ck_ohlcv_cov_ident")),
    ],
)
@pytest.mark.parametrize("tz", ["UTC", "Asia/Kolkata"])
def test_coverage_checks(
    at_head: None, args: tuple[Any, ...], want: tuple[Any, Any], tz: str
) -> None:
    assert _cov(*args, tz=tz) == want


def test_coverage_duplicate_pk(at_head: None) -> None:
    row = ("coinbase", "BTC/EUR", "1h", _T0, _T0 + _H)
    assert _cov(*row) == (None, None)
    assert _cov(*row) == ("23505", "pk_ohlcv_coverage")


# ---------------------------------------------------------------------------
# Downgrade guard and round trip to the explicit revision 021
# ---------------------------------------------------------------------------
def _version() -> str:
    return str(asyncio.run(_val("SELECT version_num FROM alembic_version")))


def _tables_exist() -> bool:
    return bool(asyncio.run(_val("SELECT to_regclass('ohlcv_bars') IS NOT NULL")))


@pytest.mark.parametrize("seed", ["bar", "coverage"])
def test_downgrade_refuses_with_rows_and_flag_overrides(at_head: None, seed: str) -> None:
    from alembic import command

    if seed == "bar":
        assert _insert_bar() == (None, None)
    else:
        assert _cov("coinbase", "BTC/EUR", "1h", _T0, _T0 + _H) == (None, None)
    with pytest.raises(RuntimeError, match="allow_drop_ohlcv=1"):
        command.downgrade(_cfg(), "021")
    assert _version() == "022" and _tables_exist()
    count = "SELECT (SELECT count(*) FROM ohlcv_bars) + (SELECT count(*) FROM ohlcv_coverage)"
    assert asyncio.run(_val(count)) == 1
    command.downgrade(_cfg(["allow_drop_ohlcv=1"]), "021")
    assert _version() == "021" and not _tables_exist()


def test_downgrade_succeeds_when_empty_and_round_trip_to_021(at_head: None) -> None:
    from alembic import command

    command.downgrade(_cfg(), "021")
    assert _version() == "021"
    assert not _tables_exist()
    assert not asyncio.run(_val("SELECT to_regclass('ohlcv_coverage') IS NOT NULL"))
    command.upgrade(_cfg(), "head")
    assert _tables_exist() and _insert_bar() == (None, None)
