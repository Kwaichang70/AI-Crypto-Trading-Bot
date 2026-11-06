"""Per-bar equity sampling (WP2.5, SY-25-02/04): a pure, stdlib-only leaf.

The equity curve of a run carries more points than bars: a seed point, one mark per
warmup step, one per processed bar and one extra point per fill (booked at the bar
close). Return metrics must be computed on ONE value per bar, from the first trading
bar on. This module derives that sample from the curve, after the run, without
touching the engine or the portfolio (SY-25-01).

Rule (SY-25-04):

1. Collapse the curve to the last point per stamp (end-of-bar mark). A decreasing
   stamp raises ``equity_sampling_unordered``.
2. Keep stamps ``>= start_at``. Nothing left (the run stopped in warmup) gives an
   empty sample.
3. The first kept stamp must equal ``start_at`` (the base point), otherwise
   ``equity_sampling_no_base_point`` is raised: a broken contract fails closed.
4. Every kept stamp on ``start_at + k * tf``: mode ``calendar_asof``. Slots
   ``k = 0..K`` each take the equity of the last stamp ``<= slot`` and
   ``carried_slots`` counts the slots ``k >= 1`` without an own point.
5. Otherwise (or when ``present_steps`` is requested) mode ``present_steps``: the
   base plus the distinct later stamps, ``carried_slots = 0``.

Grid stamps are bar CLOSE times (``close_i = open_i + tf``). A benchmark series keyed by
bar OPEN time must be shifted by one timeframe before it is joined to a sample (QA-06,
SY-25-24): benchmark joins key on ``close_i``, so an open-labelled vendor series
shifts one timeframe forward.

Purity: standard library only; no logging, environment, wall clock or I/O, and no
``trading.*`` / ``common.*`` / ``data.*`` import. Points are duck-typed through
``.timestamp`` and ``.equity`` and the timeframe is passed in as integer seconds.
Stamp arithmetic is integer ``timedelta`` arithmetic; the only float conversion is
the return formula in :func:`simple_returns`.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import ClassVar, Final, Literal, Protocol, TypeVar

__all__ = [
    "EQUITY_SAMPLING_CODES",
    "BarSample",
    "EquityPointLike",
    "EquitySamplingError",
    "SamplingMode",
    "last_per_stamp",
    "sample_equity_per_bar",
    "simple_returns",
]

SamplingMode = Literal["calendar_asof", "present_steps"]

_CODE_UNORDERED: Final = "equity_sampling_unordered"
_CODE_NO_BASE: Final = "equity_sampling_no_base_point"
EQUITY_SAMPLING_CODES: Final = (_CODE_UNORDERED, _CODE_NO_BASE)


class EquitySamplingError(ValueError):
    """Closed-code error; the message is exactly ``equity sampling failed (<code>)``."""

    CODES: ClassVar[tuple[str, ...]] = EQUITY_SAMPLING_CODES

    def __init__(self, code: str) -> None:
        if code not in EQUITY_SAMPLING_CODES:
            raise ValueError("unknown equity sampling code")
        super().__init__(f"equity sampling failed ({code})")
        self.code = code


class EquityPointLike(Protocol):
    """Duck type of an equity curve point (``EquityCurvePoint`` satisfies it)."""

    @property
    def timestamp(self) -> datetime: ...

    @property
    def equity(self) -> Decimal: ...


_P = TypeVar("_P", bound=EquityPointLike)


def simple_returns(equity: Sequence[Decimal]) -> list[float]:
    """Per-step simple returns, ``float((c - p) / p)``; 0 where ``p <= 0``.

    Byte-identical to the HEAD ``compute_returns_from_equity`` formula. Length is
    ``len(equity) - 1`` (empty for fewer than two values).
    """
    out: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        out.append(float((equity[i] - prev) / prev) if prev > Decimal(0) else float(0))
    return out


def last_per_stamp(points: Sequence[_P]) -> list[_P]:
    """Keep the last point of every run of equal stamps; the input is not mutated."""
    out: list[_P] = []
    for p in points:
        if out:
            prev_ts = out[-1].timestamp
            if p.timestamp < prev_ts:
                raise EquitySamplingError(_CODE_UNORDERED)
            if p.timestamp == prev_ts:
                out[-1] = p
                continue
        out.append(p)
    return out


@dataclass(frozen=True, slots=True)
class BarSample:
    """One equity value per bar; ``len(equity) == n_returns + 1`` (or 0 when empty).

    ``stamps[0] == start_at`` is the base point. ``carried_stamps`` lists the slots
    that carried the previous value (``carried_slots == len(carried_stamps)``).
    """

    start_at: datetime
    end_at: datetime | None
    timeframe_s: int
    stamps: tuple[datetime, ...]
    equity: tuple[Decimal, ...]
    mode: SamplingMode
    carried_slots: int
    carried_stamps: tuple[datetime, ...] = ()

    @property
    def n_returns(self) -> int:
        return max(0, len(self.equity) - 1)

    def returns(self) -> list[float]:
        return simple_returns(self.equity)

    def from_stamp(self, at: datetime) -> BarSample:
        """Sub-sample from ``at`` on; the base value is the as-of value at ``at``.

        In ``calendar_asof`` mode ``at`` must be a grid stamp (``start_at + k * tf``);
        callers derive it from ``_step_close``.
        An empty sample, or ``at`` after the last stamp, gives an empty sample.
        ``at`` before the first stamp has no as-of value and fails closed.
        """
        if not self.stamps:
            return self
        if at < self.stamps[0]:
            raise EquitySamplingError(_CODE_NO_BASE)
        if self.end_at is not None and at > self.end_at:
            return _empty(at, self.timeframe_s, self.mode)
        idx = bisect_right(self.stamps, at) - 1
        after = idx + 1
        stamps = (at, *self.stamps[after:])
        carried = tuple(c for c in self.carried_stamps if c > at)
        return BarSample(
            start_at=at,
            end_at=stamps[-1],
            timeframe_s=self.timeframe_s,
            stamps=stamps,
            equity=(self.equity[idx], *self.equity[after:]),
            mode=self.mode,
            carried_slots=len(carried),
            carried_stamps=carried,
        )


def _empty(start_at: datetime, timeframe_s: int, mode: SamplingMode) -> BarSample:
    return BarSample(
        start_at=start_at,
        end_at=None,
        timeframe_s=timeframe_s,
        stamps=(),
        equity=(),
        mode=mode,
        carried_slots=0,
    )


def sample_equity_per_bar(
    points: Sequence[_P],
    *,
    start_at: datetime,
    timeframe_s: int,
    mode: SamplingMode = "calendar_asof",
) -> BarSample:
    """Sample ``points`` once per bar from ``start_at`` (see the module docstring).

    ``mode="present_steps"`` forces the present-stamps sampling; ``"calendar_asof"``
    falls back to it when a kept stamp is off the ``start_at + k * tf`` grid.
    """
    if timeframe_s <= 0:
        raise ValueError("timeframe_s must be positive")
    kept = [p for p in last_per_stamp(points) if p.timestamp >= start_at]
    if not kept:
        return _empty(start_at, timeframe_s, mode)
    if kept[0].timestamp != start_at:
        raise EquitySamplingError(_CODE_NO_BASE)

    step = timedelta(seconds=timeframe_s)
    kept_ts = [p.timestamp for p in kept]
    on_grid = all((ts - start_at) % step == timedelta(0) for ts in kept_ts)
    if mode == "present_steps" or not on_grid:
        return BarSample(
            start_at=start_at,
            end_at=kept_ts[-1],
            timeframe_s=timeframe_s,
            stamps=tuple(kept_ts),
            equity=tuple(p.equity for p in kept),
            mode="present_steps",
            carried_slots=0,
        )

    slots = (kept_ts[-1] - start_at) // step
    present = set(kept_ts)
    stamps: list[datetime] = []
    equity: list[Decimal] = []
    carried: list[datetime] = []
    for k in range(slots + 1):
        slot = start_at + k * step
        j = bisect_right(kept_ts, slot) - 1
        stamps.append(slot)
        equity.append(kept[j].equity)
        if k >= 1 and slot not in present:
            carried.append(slot)
    return BarSample(
        start_at=start_at,
        end_at=stamps[-1],
        timeframe_s=timeframe_s,
        stamps=tuple(stamps),
        equity=tuple(equity),
        mode="calendar_asof",
        carried_slots=len(carried),
        carried_stamps=tuple(carried),
    )
