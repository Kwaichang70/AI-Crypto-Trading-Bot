"""
packages/trading/clock.py
-------------------------
WP2.2 -- ONE clock seam for the trading package.

A backtest must stamp its fills, orders, equity points and trades with *bar*
time, never with the wall clock; paper and live keep the wall clock.  This
module provides the vocabulary for that and nothing else:

* :class:`Clock` -- structural protocol, ``now() -> tz-aware UTC datetime``.
* :func:`_wall_now` -- the single wall-clock read of this module (and the
  monkeypatch target for tests).
* :class:`WallClock` / :data:`WALL_CLOCK` -- the production clock.
* :class:`BarClock` -- simulated time owned by exactly one backtest.
* :func:`utc_now` / :func:`use_clock` -- an *ambient* clock held in a
  :class:`~contextvars.ContextVar`, used ONLY as the fallback for model
  ``default_factory`` fields.  Persisted fields are always stamped
  explicitly (SY-22-05).

Design rules (binding spec ``reports/vp2-wp2.2/synthesis-spec.md``)
-------------------------------------------------------------------
* **Leaf module.**  Imports stdlib and ``structlog`` only; no other
  ``trading.*`` module.  It reads no environment.
* **Wall time is resolved at call time.**  ``WallClock.now`` looks up
  ``_wall_now`` in the module globals on every call, so a test can
  monkeypatch it (WP22-T-06 tripwire).
* **Mode guards live elsewhere.**  Which engine may hold a ``BarClock`` is
  enforced by ``StrategyEngine`` / ``LiveExecutionEngine`` (SY-22-04, P2b);
  this module only supplies the types.  ``BarClock`` and ``WallClock`` are
  final: ``@final`` is a static (type-checker) marker only, so finality is
  also enforced at runtime by ``__init_subclass__`` (``TypeError``).  Guards
  must still use ``isinstance`` only, never ``type(x) is``; ``isinstance``
  also matches subclasses, which finality merely rules out.
* **Task isolation.**  ``_AMBIENT`` is a ``ContextVar``: a concurrent
  asyncio task or thread never observes another task's ``use_clock``.
  :func:`ambient_clock` is the public read accessor for guards/tripwires.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Final, Protocol, final, runtime_checkable

import structlog

__all__ = [
    "TIMESTAMP_CLOCK_VERSION",
    "WALL_CLOCK",
    "BarClock",
    "Clock",
    "WallClock",
    "ambient_clock",
    "use_clock",
    "utc_now",
]

#: Version tag of the timestamp semantics written to
#: ``engine_semantics.timestamp_clock`` (SY-22-08).  Append-only: a change of
#: semantics gets a new tag, never an edit of this one.
TIMESTAMP_CLOCK_VERSION: Final = "wp22-1"

_log = structlog.get_logger(__name__)


@runtime_checkable
class Clock(Protocol):
    """Source of "now".  ``now()`` returns a timezone-aware UTC datetime."""

    def now(self) -> datetime: ...


def _wall_now() -> datetime:
    """The ONE wall-clock read of ``trading.clock`` (monkeypatch target)."""
    return datetime.now(tz=UTC)


@final
class WallClock:
    """Real time.  Stateless; use the shared :data:`WALL_CLOCK` instance.

    Final (``TypeError`` on subclassing) and uses ``__slots__``: tests cannot
    ``monkeypatch.setattr`` an instance attribute; patch ``_wall_now`` instead.
    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("WallClock is final and cannot be subclassed")

    def now(self) -> datetime:
        # Resolved through the module globals at call time (not bound at
        # class creation) so ``monkeypatch.setattr(clock, "_wall_now", ...)``
        # takes effect.
        return _wall_now()

    def __repr__(self) -> str:
        return "WallClock()"


WALL_CLOCK: Final = WallClock()


def _require_aware_utc(value: datetime, what: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} requires a timezone-aware datetime")
    return value.astimezone(UTC)


@final
class BarClock:
    """Simulated time owned by ONE backtest.

    Aware datetimes only (a naive value is a programming error, not data to
    normalise), stored as UTC.  Forward-only: ``advance`` with an equal or
    earlier time is a no-op plus a debug log -- the same rule as
    ``PortfolioAccounting.advance_clock`` (multi-symbol bar misalignment).

    Not thread-safe by design: a ``BarClock`` belongs to a single backtest
    loop.  Never share one between runs.

    Final (``TypeError`` on subclassing).  Uses ``__slots__``, so an instance
    attribute cannot be monkeypatched (``monkeypatch.setattr(bar_clock,
    "now", ...)`` raises ``AttributeError``): in tests wrap the instance in a
    small delegating clock, or patch at class level
    (``patch.object(BarClock, "now")``).
    """

    __slots__ = ("_now",)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("BarClock is final and cannot be subclassed")

    def __init__(self, start: datetime) -> None:
        self._now: datetime = _require_aware_utc(start, "BarClock start")

    def now(self) -> datetime:
        return self._now

    def advance(self, at: datetime) -> None:
        """Move simulated time to ``at`` if it is strictly later."""
        at = _require_aware_utc(at, "BarClock.advance")
        if at <= self._now:
            _log.debug(
                "bar_clock.advance_ignored",
                at=at.isoformat(),
                now=self._now.isoformat(),
            )
            return
        self._now = at

    def __repr__(self) -> str:
        return f"BarClock(now={self._now.isoformat()})"


_AMBIENT: ContextVar[Clock] = ContextVar("trading_clock_ambient", default=WALL_CLOCK)


def utc_now() -> datetime:
    """Ambient "now" (wall time unless inside :func:`use_clock`).

    Intended ONLY as a pydantic ``default_factory``; persisted fields must be
    stamped explicitly from the owning clock (SY-22-05).
    """
    return _AMBIENT.get().now()


def ambient_clock() -> Clock:
    """The clock currently ambient in this context (public read accessor).

    For guards and fail-closed tripwires (e.g. "no ``BarClock`` may be ambient
    in a paper/live process"); callers must not reach into ``_AMBIENT``.
    """
    return _AMBIENT.get()


@contextmanager
def use_clock(clock: Clock) -> Iterator[None]:
    """Make ``clock`` the ambient clock for the current context.

    The previous clock is restored in ``finally`` -- on normal exit and on
    exception.  Per SY-22-05 it is entered only in
    ``StrategyEngine.run_backtest``.

    WARNING: use it ONLY as a ``with`` statement in a plain function or
    coroutine.  NEVER wrap a ``yield`` / ``async yield`` (generator or async
    generator) in it: the generator has no context of its own, so the clock
    would leak into the consumer's context while suspended, and a close or
    garbage-collection from a different ``Context`` makes the reset raise
    ``ValueError`` and leaves the clock set permanently.

    Raises ``TypeError`` on entry if ``clock`` is not a :class:`Clock`.
    """
    if isinstance(clock, type) or not isinstance(clock, Clock):  # class, not instance
        raise TypeError(f"use_clock requires a Clock, got {type(clock).__name__}")
    token = _AMBIENT.set(clock)
    try:
        yield
    finally:
        _AMBIENT.reset(token)
