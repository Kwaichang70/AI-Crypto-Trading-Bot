"""OHLCV cache: ``ohlcv_bars`` and ``ohlcv_coverage`` (WP2.6 P3)

Revision ID: 022
Revises: 021
Create Date: 2026-11-13 00:00:00.000000 UTC

Description
-----------
WP2.6 (reports/vp2-wp26/spec.md, SY-26-13/14, addendum SY-26-22..25) adds the two tables
behind the Postgres OHLCV read-through cache.

* ``ohlcv_bars`` -- one closed bar per ``(exchange, symbol, timeframe, open_time)``.
  Prices are ``NUMERIC(28,12)``, volume ``NUMERIC(30,12)``. The first write wins
  (``INSERT ... ON CONFLICT DO NOTHING`` in the store); application code never issues
  UPDATE or DELETE on this table. ``fetched_at`` and ``ingest_version`` are audit data.
* ``ohlcv_coverage`` -- one contiguous verified interval ``[covered_from, covered_to)``
  per series. Gaps inside it are permanent by contract.

CHECK constraints (the same text lives in ``OhlcvBarORM`` / ``OhlcvCoverageORM``)
---------------------------------------------------------------------------------
* ``ident``: exchange ``^[a-z0-9_]+$``; symbol has a non-empty base and quote around ONE
  ``/`` and every other character is in 0x21-0x7E except ``|`` (``[!-.0-{}~]``). The upper
  bound of 32 is the ``VARCHAR(32)`` length (SQLSTATE 22001, not 23514).
* ``timeframe``: ``TimeFrame`` minus ``1w`` (the weekly anchor is undecided, CF-26-02).
  A CASE on an unknown timeframe is NULL and a NULL CHECK passes, so the list is its own
  constraint. The CASE seconds equal ``strategy_engine._TIMEFRAME_SECONDS`` (pinned by a test).
* ``grid``: ``mod(CAST(extract(epoch FROM t) AS numeric), tf_seconds) = 0``. The epoch of a
  ``timestamptz`` is independent of the session ``TimeZone``. The CAST makes ``mod`` valid
  on any PostgreSQL version; the target is PostgreSQL 16 (infra/docker-compose.yml) where
  ``extract`` already returns an exact numeric, so sub-second values are rejected exactly.
* ``closed``: ``fetched_at >= open_time + tf`` (row-internal, no ``now()``, restore-safe).
* ``ohlc`` / ``volume`` / ``finite``: mirror ``OHLCVBar``; NaN needs its own CHECK.

No foreign keys and no secondary index (the PK serves the series + time range scan).

Lock/scan characteristics
--------------------------
Pure ``CREATE TABLE``: metadata only, instant, no lock on any existing table (checked with
ACCESS EXCLUSIVE held on ``runs`` and ``context_observations``). ``SET LOCAL lock_timeout =
'5s'`` follows the 019/021 convention. The api entrypoint runs ``alembic upgrade head`` on
every start, so the tables exist before the new image serves; the store still fails soft on a
missing table.

Downgrade path
---------------
Rows are re-fetchable in principle, but a re-fetch is not guaranteed to reproduce them
(revised bars, exchange history limits), and stored ``data_hash`` values would become
unverifiable. The downgrade therefore REFUSES while either table holds a row, unless the
operator passes ``-x allow_drop_ohlcv=1``. The check runs under ``LOCK TABLE ... IN ACCESS
EXCLUSIVE MODE`` so no writer can add a row between the check and the drop. In offline
(``--sql``) mode the tables cannot be inspected, so the flag is required there too.
Rollback: stop api -> ``pg_dump -t ohlcv_bars -t ohlcv_coverage`` (when either holds a row)
-> run ``alembic -x allow_drop_ohlcv=1 downgrade 021`` from the NEW image (an older image
cannot locate revision 022) -> deploy the old image (it cannot start on a DB at 022 either).
Do not restart the new image before the old one is deployed: its entrypoint upgrades again.

Revision chain: ``down_revision = "021"`` (single-head rule).

References
----------
SY-26-13, SY-26-14, SY-26-22..25, reports/vp2-wp26/design-db.md (DB-26-06..15, DB-26-20).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BARS = "ohlcv_bars"
_COVERAGE = "ohlcv_coverage"
_ALLOW_DROP_FLAG = "allow_drop_ohlcv"

_TF_LIST = "'1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'"
_TF_CASE = (
    "CASE timeframe WHEN '1m' THEN 60 WHEN '3m' THEN 180 WHEN '5m' THEN 300 "
    "WHEN '15m' THEN 900 WHEN '30m' THEN 1800 WHEN '1h' THEN 3600 "
    "WHEN '4h' THEN 14400 WHEN '1d' THEN 86400 END"
)
_IDENT = "exchange ~ '^[a-z0-9_]+$' AND symbol ~ '^[!-.0-{}~]+/[!-.0-{}~]+$'"


def _epoch(column: str) -> str:
    return f"CAST(extract(epoch FROM {column}) AS numeric)"


def _series_columns() -> list[sa.Column[str]]:
    return [
        sa.Column("exchange", sa.String(length=32), nullable=False, comment="ccxt exchange id."),
        sa.Column("symbol", sa.String(length=32), nullable=False, comment="ccxt unified symbol."),
        sa.Column("timeframe", sa.String(length=4), nullable=False, comment="1m .. 1d, no 1w."),
    ]


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    price = sa.Numeric(precision=28, scale=12)
    op.create_table(
        _BARS,
        *_series_columns(),
        sa.Column(
            "open_time",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Bar OPEN instant, UTC, on the timeframe grid.",
        ),
        sa.Column("open", price, nullable=False),
        sa.Column("high", price, nullable=False),
        sa.Column("low", price, nullable=False),
        sa.Column("close", price, nullable=False),
        sa.Column(
            "volume",
            sa.Numeric(precision=30, scale=12),
            nullable=False,
            comment="Base-asset volume.",
        ),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="Audit only; never part of a hash.",
        ),
        sa.Column(
            "ingest_version",
            sa.SmallInteger(),
            nullable=False,
            server_default=sa.text("1"),
            comment="Writer semantics version (pager/quantiser).",
        ),
        sa.PrimaryKeyConstraint(
            "exchange", "symbol", "timeframe", "open_time", name="pk_ohlcv_bars"
        ),
        sa.CheckConstraint(_IDENT, name="ck_ohlcv_bars_ident"),
        sa.CheckConstraint(f"timeframe IN ({_TF_LIST})", name="ck_ohlcv_bars_timeframe"),
        sa.CheckConstraint(
            f"mod({_epoch('open_time')}, {_TF_CASE}) = 0", name="ck_ohlcv_bars_grid"
        ),
        sa.CheckConstraint(
            f"{_epoch('fetched_at')} >= {_epoch('open_time')} + {_TF_CASE}",
            name="ck_ohlcv_bars_closed",
        ),
        sa.CheckConstraint(
            "low > 0 AND low <= open AND low <= close AND open <= high AND close <= high",
            name="ck_ohlcv_bars_ohlc",
        ),
        sa.CheckConstraint("volume >= 0", name="ck_ohlcv_bars_volume"),
        sa.CheckConstraint(
            "open <> 'NaN' AND high <> 'NaN' AND low <> 'NaN' "
            "AND close <> 'NaN' AND volume <> 'NaN'",
            name="ck_ohlcv_bars_finite",
        ),
    )
    op.create_table(
        _COVERAGE,
        *_series_columns(),
        sa.Column(
            "covered_from",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Inclusive start of the verified interval, on the grid.",
        ),
        sa.Column(
            "covered_to",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Exclusive end of the verified interval, on the grid.",
        ),
        sa.Column(
            "verified_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="Audit only.",
        ),
        sa.PrimaryKeyConstraint("exchange", "symbol", "timeframe", name="pk_ohlcv_coverage"),
        sa.CheckConstraint(_IDENT, name="ck_ohlcv_cov_ident"),
        sa.CheckConstraint("covered_from <= covered_to", name="ck_ohlcv_cov_span"),
        sa.CheckConstraint(f"timeframe IN ({_TF_LIST})", name="ck_ohlcv_cov_timeframe"),
        sa.CheckConstraint(
            f"mod({_epoch('covered_from')}, {_TF_CASE}) = 0 "
            f"AND mod({_epoch('covered_to')}, {_TF_CASE}) = 0",
            name="ck_ohlcv_cov_grid",
        ),
    )


def _drop_allowed() -> bool:
    args = context.get_x_argument(as_dictionary=True)
    return args.get(_ALLOW_DROP_FLAG) == "1"


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    if not _drop_allowed():
        if context.is_offline_mode():
            raise RuntimeError(
                "refusing to drop ohlcv_bars/ohlcv_coverage in offline mode: the tables "
                "cannot be inspected; pass -x allow_drop_ohlcv=1 to override"
            )
        # Close the check-then-drop window: no writer can add a row between the
        # EXISTS check and the DROP (lock_timeout bounds the wait).
        op.execute(f"LOCK TABLE {_BARS}, {_COVERAGE} IN ACCESS EXCLUSIVE MODE")
        has_rows = op.get_bind().execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM ohlcv_bars) OR EXISTS (SELECT 1 FROM ohlcv_coverage)"
            )
        )
        if has_rows.scalar():
            raise RuntimeError(
                "refusing to drop ohlcv_bars/ohlcv_coverage: they hold rows that a re-fetch "
                "may not reproduce; pg_dump both tables first, then rerun with "
                "-x allow_drop_ohlcv=1"
            )
    op.drop_table(_COVERAGE)
    op.drop_table(_BARS)
