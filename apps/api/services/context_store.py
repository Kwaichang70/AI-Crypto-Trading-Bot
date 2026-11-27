"""
apps/api/services/context_store.py
----------------------------------
WP2.4 P5 (reports/vp2-wp2.4/synthesis-spec.md SY-24-09, SY-24-12, SY-24-17): the
point-in-time context store.

* :class:`ContextStore` -- the Protocol. ``write_observations`` appends rows,
  ``load_window`` returns validated :class:`data.point_in_time.PitSeries`.
* :class:`SqlContextStore` -- Postgres (table ``context_observations``, migration 021).
* :class:`InMemoryContextStore` -- a fake with the SAME semantics (unit tests, offline use).
* :func:`load_window_safe` -- never raises; a store failure becomes a window whose
  ``reason`` is ``store_unavailable`` (the P7 caller maps it to an unavailable source).

Design rules
------------
* **Append-only.** There is no UPDATE and no DELETE anywhere in application code (pinned
  by tests). A revision is a new row with a later ``available_at``.
* **One idempotent writer.** ``INSERT ... SELECT ... WHERE NOT EXISTS (same series,
  observed_at, value) ... ON CONFLICT (series_id, observed_at, available_at) DO NOTHING``.
  The first sighting of an ``(observed_at, value)`` pair wins, so a recorder re-polling an
  unchanged value, a re-run backfill, and a backfill after recording all insert nothing; a
  changed value inserts exactly one new vintage. Known limit (accepted, db-design
  WP24-DB-03): a value that reverts to an earlier value for the same ``observed_at`` is not
  recorded.
* **One validator (CF-SYN-01).** Every row passes through the leaf's own
  ``parse_jsonl_lines``: the series allowlist, the basis allowlist, the per-series lag
  floor, the 8-decimal / ``|v| < 1e16`` value rule and ``available >= observed``. The store
  floor therefore equals the JSONL lag floor BY CONSTRUCTION; there is no second copy of
  the numbers. Rows read back from the database are validated the same way (a row that
  bypassed the writer fails the load closed).
* **Values.** Callers quantize with :func:`quantize_store_value` (8 decimals, HALF_UP,
  never ``Decimal(float)``, QR-11). Timestamps are aware and millisecond aligned.
* **Loading.** ``load_window`` takes ``start``/``end``; the caller passes
  ``start = first decision - back_ms - max_age`` (QR-12, CF-P2-11(a)). Only rows with
  ``available_at <= end`` are loaded, so later appends cannot change a digest. For each
  series the rows are those with ``observed_at >= S``, where ``S`` is the newest
  ``observed_at`` already available at ``start`` (the seed), so a late revision of the seed
  observation is kept; a series with no row available at ``start`` loads every row
  available by ``end``. For every decision time in ``[start, end]`` the result equals the
  as-of answer computed from ALL rows available by that time.
* **No wall clock.** Nothing here reads the clock; ``ingested_at`` is the database's.
* **Logs.** Only ``exc_log_fields`` (type, status, code location). Never a DSN, URL, SQL
  text, parameter or value.

The module deliberately imports nothing from ``trading``: the builder that turns a window
into ``PitSignals`` is P7 (SY-24-12, CF-P3-09).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Final, Protocol, cast

import structlog
from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from common.logging import exc_log_fields
from data.point_in_time import (
    SERIES_RULES,
    Basis,
    Observation,
    PitError,
    PitSeries,
    parse_jsonl_lines,
    to_ms,
)

__all__ = [
    "REASON_STORE_UNAVAILABLE",
    "ContextRow",
    "ContextStore",
    "ContextWindow",
    "InMemoryContextStore",
    "SqlContextStore",
    "load_window_safe",
    "quantize_store_value",
    "series_from_rows",
    "source_for",
]

logger = structlog.get_logger(__name__)

#: The closed reason a failed load reports (CF-P3-08: no exception text, ever).
REASON_STORE_UNAVAILABLE: Final = "store_unavailable"

_CODE_BAD_VALUE: Final = "pit_bad_value"
_MS_MICROS: Final = 1000
_MAX_TEXT_SCALE: Final = 64
_VALUE_QUANTUM: Final = Decimal("1e-8")
_VALUE_LIMIT: Final = Decimal("1e16")
#: A private context: quantize never depends on the ambient ``decimal`` context (CF-SYN-03).
_QUANTIZE_CONTEXT: Final = Context(prec=60, rounding=ROUND_HALF_UP)

_WRITE_TIMEOUT: Final = "5s"
_READ_TIMEOUT: Final = "15s"

#: Provider label per series (derived, never caller-supplied).
_SOURCE_BY_SERIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "alternative_me:fgi": "alternative_me",
        "fred:T10Y2Y": "fred",
        "coingecko:btc_dominance": "coingecko",
        "whale:net_flow_1h": "whale_alert",
    }
)

# One statement for live and backfill (SY-24-09). Bound parameters only; the casts give
# asyncpg a type for every parameter of the SELECT list.
_INSERT_SQL: Final = text(
    """
    INSERT INTO context_observations
        (series_id, observed_at, available_at, value, source, available_at_basis)
    SELECT CAST(:series_id AS VARCHAR(64)),
           CAST(:observed_at AS TIMESTAMPTZ),
           CAST(:available_at AS TIMESTAMPTZ),
           CAST(:value AS NUMERIC(24, 8)),
           CAST(:source AS VARCHAR(32)),
           CAST(:basis AS VARCHAR(16))
    WHERE NOT EXISTS (
        SELECT 1 FROM context_observations
        WHERE series_id = :series_id
          AND observed_at = :observed_at
          AND value = CAST(:value AS NUMERIC(24, 8))
    )
    ON CONFLICT (series_id, observed_at, available_at) DO NOTHING
    """
)

# ONE query: for each requested series the rows available by ``end`` whose observed_at is
# at or after the seed observation (newest observed_at already available at ``start``).
_LOAD_SQL: Final = text(
    """
    WITH ids AS (
        SELECT unnest(CAST(:ids AS VARCHAR(64)[])) AS series_id
    ), seed AS (
        SELECT i.series_id,
               COALESCE(
                   (SELECT max(c.observed_at) FROM context_observations c
                    WHERE c.series_id = i.series_id
                      AND c.available_at <= CAST(:start AS TIMESTAMPTZ)),
                   CAST('-infinity' AS TIMESTAMPTZ)
               ) AS floor_observed
        FROM ids i
    )
    SELECT c.series_id, c.observed_at, c.available_at, c.value,
           c.available_at_basis, c.ingested_at
    FROM context_observations c
    JOIN seed s ON s.series_id = c.series_id
    WHERE c.available_at <= CAST(:end AS TIMESTAMPTZ)
      AND c.observed_at >= s.floor_observed
    ORDER BY c.series_id, c.available_at, c.observed_at
    """
).bindparams(bindparam("ids", type_=ARRAY(String(64))))


# ---------------------------------------------------------------------------
# values and rows
# ---------------------------------------------------------------------------


def quantize_store_value(raw: Decimal | float | int | str) -> Decimal:
    """Round ``raw`` to the store's 8 decimals, ``ROUND_HALF_UP`` (QR-11).

    Floats go through ``str`` (shortest repr), never ``Decimal(float)``. NaN, infinity,
    bools and ``|v| >= 1e16`` raise ``pit_bad_value`` (fixed text, no data). PostgreSQL
    NUMERIC rounds ties away from zero, which is ``ROUND_HALF_UP`` and not the default
    half-even.
    """
    if isinstance(raw, bool):
        raise PitError(_CODE_BAD_VALUE)
    try:
        if isinstance(raw, Decimal):
            exact = raw
        elif isinstance(raw, float):
            exact = Decimal(repr(raw))  # shortest repr text, never Decimal(float)
        elif isinstance(raw, int | str):
            exact = Decimal(raw)
        else:
            raise PitError(_CODE_BAD_VALUE)
        if not exact.is_finite() or exact.copy_abs() >= _VALUE_LIMIT:
            raise PitError(_CODE_BAD_VALUE)
        rounded = exact.quantize(_VALUE_QUANTUM, context=_QUANTIZE_CONTEXT)
    except (InvalidOperation, ValueError, ArithmeticError):
        raise PitError(_CODE_BAD_VALUE) from None
    if rounded.copy_abs() >= _VALUE_LIMIT:
        raise PitError(_CODE_BAD_VALUE)
    if rounded == 0:
        return Decimal(0).quantize(_VALUE_QUANTUM)  # -0 -> 0
    return rounded


def source_for(series_id: str) -> str:
    """The provider label stored for ``series_id`` (unknown ids raise ``pit_unknown_series``)."""
    try:
        return _SOURCE_BY_SERIES[series_id]
    except KeyError:
        raise PitError("pit_unknown_series") from None


def _value_text(value: Decimal) -> str:
    """Plain text of a bounded finite value; refuses shapes whose text could be huge."""
    exponent = value.as_tuple().exponent
    if (
        not value.is_finite()
        or not isinstance(exponent, int)
        or exponent < -_MAX_TEXT_SCALE
        or value.copy_abs() >= _VALUE_LIMIT
    ):
        raise PitError(_CODE_BAD_VALUE)
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class ContextRow:
    """One validated observation row. Construction IS the validation (fail closed).

    The row is checked by the leaf's ``parse_jsonl_lines``: known series, allowed basis for
    that series, ``available_at`` at or after the lag floor of its basis, an exact finite
    8-decimal value below 1e16. Timestamps must be timezone-aware and millisecond aligned
    (the digest and the join work on integer milliseconds). Errors are :class:`PitError`
    with a fixed message; the row content is never echoed.
    """

    series_id: str
    observed_at: datetime
    available_at: datetime
    value: Decimal
    basis: str

    def __post_init__(self) -> None:
        if type(self.series_id) is not str or type(self.basis) is not str:
            raise PitError(_CODE_BAD_VALUE)
        if type(self.value) is not Decimal:
            raise PitError(_CODE_BAD_VALUE)
        stamps: list[datetime] = []
        for stamp in (self.observed_at, self.available_at):
            if not isinstance(stamp, datetime):
                raise PitError(_CODE_BAD_VALUE)
            try:
                to_ms(stamp)  # naive -> pit_naive_time
                if stamp.microsecond % _MS_MICROS != 0:
                    raise PitError(_CODE_BAD_VALUE)
                stamps.append(stamp.astimezone(UTC))
            except OverflowError:
                # datetime.min / datetime.max with a non-UTC offset cannot be normalised
                raise PitError(_CODE_BAD_VALUE) from None
        object.__setattr__(self, "observed_at", stamps[0])
        object.__setattr__(self, "available_at", stamps[1])
        line = json.dumps(
            {
                "series_id": self.series_id,
                "observed_at": stamps[0].isoformat(),
                "available_at": stamps[1].isoformat(),
                "value": _value_text(self.value),
                "basis": self.basis,
            }
        )
        parse_jsonl_lines([line])  # the one validator (CF-SYN-01); raises PitError

    @property
    def source(self) -> str:
        return source_for(self.series_id)

    def pk(self) -> tuple[str, datetime, datetime]:
        return (self.series_id, self.observed_at, self.available_at)


def series_from_rows(rows: Iterable[ContextRow]) -> dict[str, PitSeries]:
    """Build :class:`PitSeries` per series from validated rows (pure, no I/O)."""
    grouped: dict[str, list[Observation]] = {}
    for row in rows:
        grouped.setdefault(row.series_id, []).append(
            Observation(
                to_ms(row.observed_at),
                to_ms(row.available_at),
                row.value,
                cast(Basis, row.basis),
            )
        )
    return {sid: PitSeries(sid, tuple(obs)) for sid, obs in grouped.items()}


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """The result of a load: validated series plus audit facts.

    ``reason`` is None on success and :data:`REASON_STORE_UNAVAILABLE` after a failure;
    it never carries exception text. ``ingest_cutoff`` is ``max(ingested_at)`` of the
    loaded rows (SY-24-17); None for the in-memory fake and for an empty load.
    """

    series: Mapping[str, PitSeries] = field(default_factory=lambda: MappingProxyType({}))
    row_count: int = 0
    ingest_cutoff: datetime | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None

    @classmethod
    def unavailable(cls) -> ContextWindow:
        return cls(reason=REASON_STORE_UNAVAILABLE)


class ContextStore(Protocol):
    """Append-only point-in-time store (SY-24-09)."""

    async def write_observations(self, rows: Sequence[ContextRow]) -> int:
        """Append ``rows`` idempotently; return how many NEW rows were inserted."""
        ...

    async def load_window(
        self, series_ids: Collection[str], start: datetime, end: datetime
    ) -> ContextWindow:
        """Load validated series for decisions in ``[start, end]``. May raise."""
        ...


def _check_window(series_ids: Collection[str], start: datetime, end: datetime) -> list[str]:
    ids = sorted(set(series_ids))
    if not ids or any(type(i) is not str or i not in SERIES_RULES for i in ids):
        raise PitError("pit_unknown_series")
    if to_ms(start) > to_ms(end):
        raise PitError(_CODE_BAD_VALUE)
    return ids


# ---------------------------------------------------------------------------
# SQL implementation
# ---------------------------------------------------------------------------


class SqlContextStore:
    """Postgres store. Own short transactions on sessions from ``session_factory``.

    Never pass the trading / persistence session: a recorder or loader failure must not
    touch an engine transaction.

    Error text (CR-06): ``write_observations`` and ``load_window`` propagate raw SQLAlchemy
    and driver errors, whose text can carry statement parameters. Callers MUST log them
    with ``common.logging.exc_log_fields`` only (never the exception text), and the engine behind
    ``session_factory`` MUST be created with ``hide_parameters=True`` (as
    ``api.db.session`` does). ``load_window_safe`` already follows this rule.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def write_observations(self, rows: Sequence[ContextRow]) -> int:
        """Append ``rows`` in one transaction; return the number of NEW rows.

        Raises the raw database error on failure: log it with ``exc_log_fields`` only and
        use an engine with ``hide_parameters=True`` (see the class docstring, CR-06).
        """
        batch = sorted(rows, key=ContextRow.pk)
        if not batch:
            return 0
        inserted = 0
        async with self._session_factory() as session, session.begin():
            await session.execute(text(f"SET LOCAL statement_timeout = '{_WRITE_TIMEOUT}'"))
            for row in batch:
                result = await session.execute(
                    _INSERT_SQL,
                    {
                        "series_id": row.series_id,
                        "observed_at": row.observed_at,
                        "available_at": row.available_at,
                        "value": row.value,
                        "source": row.source,
                        "basis": row.basis,
                    },
                )
                inserted += int(result.rowcount or 0)  # type: ignore[attr-defined]
        return inserted

    async def load_window(
        self, series_ids: Collection[str], start: datetime, end: datetime
    ) -> ContextWindow:
        ids = _check_window(series_ids, start, end)
        async with self._session_factory() as session, session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(text(f"SET LOCAL statement_timeout = '{_READ_TIMEOUT}'"))
            result = await session.execute(_LOAD_SQL, {"ids": ids, "start": start, "end": end})
            fetched = [tuple(r) for r in result.all()]
        rows: list[ContextRow] = []
        cutoff: datetime | None = None
        for sid, observed, available, value, basis, ingested in _typed(fetched):
            rows.append(ContextRow(sid, observed, available, value, basis))
            if cutoff is None or ingested > cutoff:
                cutoff = ingested
        return ContextWindow(
            series=MappingProxyType(series_from_rows(rows)),
            row_count=len(rows),
            ingest_cutoff=cutoff,
        )


def _typed(
    fetched: Iterable[tuple[Any, ...]],
) -> Iterable[tuple[str, datetime, datetime, Decimal, str, datetime]]:
    for r in fetched:
        if len(r) != 6 or not isinstance(r[5], datetime):
            raise PitError(_CODE_BAD_VALUE)
        yield (r[0], r[1], r[2], r[3], r[4], r[5])


# ---------------------------------------------------------------------------
# in-memory fake (same semantics)
# ---------------------------------------------------------------------------


class InMemoryContextStore:
    """Fake store with the semantics of :class:`SqlContextStore` (no database, no clock)."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, datetime, datetime], ContextRow] = {}
        self.fail_with: BaseException | None = None  # test hook: raise on the next call

    def __len__(self) -> int:
        return len(self._rows)

    def rows(self) -> list[ContextRow]:
        return sorted(self._rows.values(), key=ContextRow.pk)

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    async def write_observations(self, rows: Sequence[ContextRow]) -> int:
        self._maybe_fail()
        inserted = 0
        for row in sorted(rows, key=ContextRow.pk):
            seen = any(
                r.series_id == row.series_id
                and r.observed_at == row.observed_at
                and r.value == row.value
                for r in self._rows.values()
            )
            if seen or row.pk() in self._rows:
                continue
            self._rows[row.pk()] = row
            inserted += 1
        return inserted

    async def load_window(
        self, series_ids: Collection[str], start: datetime, end: datetime
    ) -> ContextWindow:
        self._maybe_fail()
        ids = _check_window(series_ids, start, end)
        picked: list[ContextRow] = []
        for sid in ids:
            mine = [r for r in self._rows.values() if r.series_id == sid]
            seeded = [r.observed_at for r in mine if r.available_at <= start]
            floor = max(seeded) if seeded else None
            picked.extend(
                r
                for r in mine
                if r.available_at <= end and (floor is None or r.observed_at >= floor)
            )
        return ContextWindow(
            series=MappingProxyType(series_from_rows(picked)),
            row_count=len(picked),
            ingest_cutoff=None,
        )


# ---------------------------------------------------------------------------
# fail-soft loader
# ---------------------------------------------------------------------------


async def load_window_safe(
    store: ContextStore, series_ids: Collection[str], start: datetime, end: datetime
) -> ContextWindow:
    """``store.load_window`` that never raises into the caller (SY-24-12).

    Any ``Exception`` (connection error, missing table, invalid stored row, bad window)
    is logged with ``exc_log_fields`` only and returned as
    ``ContextWindow(reason="store_unavailable")``. Cancellation still propagates.
    """
    try:
        return await store.load_window(series_ids, start, end)
    except Exception as exc:
        logger.warning("context_store_load_failed", **exc_log_fields(exc))
        return ContextWindow.unavailable()
