"""
packages/trading/portfolio.py
-------------------------------
Portfolio accounting: equity tracking, PnL calculation, and trade history.

This is a concrete class (not abstract) that serves as the single source
of truth for portfolio state within a trading run. It is updated on every
fill event and provides equity, drawdown, and PnL metrics to the
RiskManager and UI layer.

Design notes
------------
- All monetary values are in quote currency (e.g. USDT).
- Drawdown is computed from peak equity, which is updated on every
  equity recalculation.
- Daily PnL (realised) rolls at the UTC day boundary: ``get_daily_pnl()``
  is realised-only (FIX-07) and reads as 0 on a new UTC day even before the
  first fill of that day (WP1.5 / C5).
- WP1.5 (D-WP15-A): the daily-loss *gate* is driven by the mark-to-market
  NAV change since the UTC day start, not by the realised figure.  This
  class is the single owner of that state (``nav_day_start``, the running
  worst NAV change = the latch, the peak).  Engines and the risk manager
  are stateless readers of :meth:`PortfolioAccounting.risk_snapshot`.
- Risk-day clock: wall clock (UTC) in paper/live; in a backtest
  ``StrategyEngine.run_backtest`` pushes the bar *close* time through
  :meth:`PortfolioAccounting.advance_clock`.  The wall clock is never read
  on the risk path (``_now`` is the single seam).
- The equity curve is stored as a list of (timestamp, equity) tuples
  for charting and analysis.
- TradeResult records are created when positions are fully closed and
  represent completed round-trip trades.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Literal, TypedDict

import structlog

from common.types import OrderSide
from trading.models import Fill, Position, TradeResult
from trading.recovery import ResumeDailySeed, replay_sort_key

__all__ = [
    "DayStartSource",
    "PortfolioAccounting",
    "PortfolioSummary",
    "RiskSnapshot",
    "blend_entry_price",
]

logger = structlog.get_logger(__name__)

_PRECISION = Decimal("0.00000001")


def blend_entry_price(prev_qty: Decimal, prev_avg: Decimal, fill: Fill) -> Decimal:
    """All-in average entry price after applying a BUY ``fill`` (WP1.3b).

    Extracted verbatim from :meth:`PortfolioAccounting._apply_fill` (same
    ``_PRECISION`` / ``ROUND_HALF_UP``) so the bracket anchor's entry basis is
    byte-identical to ``Position.average_entry_price`` for every lot without
    adds.  ``prev_qty <= 0`` means a new position: the basis is the fill price
    plus the entry fee spread over the fill quantity.  Otherwise the previous
    cost (``prev_qty * prev_avg``) is blended with the fill cost
    (``quantity * price + fee``).
    """
    if prev_qty <= Decimal("0"):
        return ((fill.price * fill.quantity + fill.fee) / fill.quantity).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
    old_cost = prev_qty * prev_avg
    new_cost = fill.quantity * fill.price + fill.fee
    total_qty = prev_qty + fill.quantity
    return ((old_cost + new_cost) / total_qty).quantize(_PRECISION, rounding=ROUND_HALF_UP)

#: Where the current UTC day's ``nav_day_start`` came from (WP1.5 §5.1).
#:
#: ``run_start``       the run started today; day start is ``initial_cash``.
#: ``roll``            captured at a UTC day roll (the NAV at the previous marks).
#: ``snapshot``        resume: seeded from the persisted equity-snapshot hint.
#: ``replay_estimate`` resume without a seed: ``rebuilt_nav - realised_today``.
#: ``unknown``         resume could not establish the day start; BUYs stay
#:                     blocked until the next UTC roll (fail closed).
DayStartSource = Literal["run_start", "roll", "snapshot", "replay_estimate", "unknown"]


def _utc_now() -> datetime:
    """Default risk-day clock (wall clock, UTC).

    Module-level on purpose: it is the one monkeypatch seam the determinism
    tests (WP15-T-23) use to prove a backtest never reads the wall clock.
    """
    return datetime.now(tz=UTC)


def _utc_day(at: datetime) -> date:
    """UTC calendar day of ``at``.

    Aware datetimes are converted to UTC; a naive datetime (replayed or
    persisted data) is *treated as* UTC rather than rejected (WP1.5 C-12).
    """
    if at.tzinfo is None:
        return at.replace(tzinfo=UTC).date()
    return at.astimezone(UTC).date()


def _invalid_seed_field(seed: object) -> tuple[str, str] | None:
    """Strict type check of a resume seed (WP1.5b R-1 / SEC-05 / SEC-208).

    Returns ``(field_name, type_name)`` of the first invalid field, or ``None``
    if the seed is valid (or absent).  **No coercion**: a ``float``, ``int``,
    ``str`` or ``bool`` hint, a NaN / infinite / negative ``Decimal``, a
    ``datetime`` ``as_of_day`` and a naive ``run_started_at`` are all
    rejected.  The persisted column is ``Numeric(20, 8)`` and yields
    ``Decimal``, so a non-``Decimal`` hint is an upstream defect that must not
    be silently repaired.  Only field and type *names* are returned, never
    values.
    """
    if seed is None:
        return None
    if not isinstance(seed, ResumeDailySeed):
        return ("daily_seed", type(seed).__name__)
    as_of = seed.as_of_day
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        return ("as_of_day", type(as_of).__name__)
    for name in ("day_start_equity", "day_min_equity", "day_first_equity"):
        value = getattr(seed, name)
        if value is None:
            continue
        # ``is_finite()`` first: ordering a NaN raises.  ``>= 0`` mirrors the
        # DB check ``ck_equity_snapshots_equity_non_negative``.
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            return (name, type(value).__name__)
    started = seed.run_started_at
    if started is not None and (not isinstance(started, datetime) or started.utcoffset() is None):
        return ("run_started_at", type(started).__name__)
    return None


class PortfolioSummary(TypedDict):
    """Typed return of :meth:`PortfolioAccounting.get_summary` (WP1.6, C3).

    The keys equal the runtime dict keys exactly (pinned by a contract test),
    so a typo such as ``summary["realised_pnl"]`` is a mypy error instead of a
    silent default.  ``get_summary`` is display / API data only: risk
    decisions read :class:`RiskSnapshot`, never these keys.
    """

    run_id: str
    initial_cash: str
    current_cash: str
    current_equity: str
    peak_equity: str
    total_return_pct: float
    total_realised_pnl: str
    total_fees_paid: str
    daily_pnl: str
    drawdown_pct: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    open_positions: int
    equity_curve_length: int


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """Read-only risk state at one instant (WP1.5 §6).

    Produced by :meth:`PortfolioAccounting.risk_snapshot`, which is pure:
    taking a snapshot never changes the peak, the latch or the day.  This is
    the single typed surface the engines (and, from WP1.6, the circuit
    breaker) read; never ``get_summary()`` string keys.

    Attributes
    ----------
    as_of:
        The risk clock reading used for the snapshot.
    day:
        The UTC risk day the figures belong to.
    nav:
        ``current_equity`` at the portfolio's marks.
    peak_equity:
        ``max(running peak, nav)``.
    drawdown_pct:
        ``(peak - nav) / peak``, ``>= 0``; ``0`` when ``peak <= 0``.
    nav_day_start:
        ``max(NAV at the last pre-roll observation, NAV at the first
        observation of the day)``.  Provisional until that first observation
        (it equals the pre-roll NAV until then) and can only rise, once
        (SYN-01 / Option B).
    daily_loss_basis:
        ``max(0, min(nav_day_start, initial_cash))`` -- the denominator of
        the daily-loss threshold (D5 / D-WP15-B), fixed from the first NAV
        observation of the UTC day.
    nav_pnl:
        ``nav - nav_day_start``.
    worst_nav_pnl:
        Running minimum of the NAV change since day start (``<= 0``); the
        latch.  Once at or below ``-threshold`` BUYs stay blocked for the
        rest of the UTC day even if NAV recovers.
    realised_pnl:
        Realised PnL of the risk day (the FIX-07 figure the API shows).
    day_start_source:
        Provenance of ``nav_day_start`` (:data:`DayStartSource`).
    open_position_count:
        Number of non-flat positions.
    """

    as_of: datetime
    day: date
    nav: Decimal
    peak_equity: Decimal
    drawdown_pct: Decimal
    nav_day_start: Decimal
    daily_loss_basis: Decimal
    nav_pnl: Decimal
    worst_nav_pnl: Decimal
    realised_pnl: Decimal
    day_start_source: DayStartSource
    open_position_count: int

    @property
    def state_unknown(self) -> bool:
        """True when a resume could not establish the day start (fail closed)."""
        return self.day_start_source == "unknown"

    def gate_daily_pnl(self, nav: Decimal) -> Decimal:
        """The ``daily_pnl`` value the risk gate must be fed (§5.3). Pure.

        - ``state_unknown`` -> ``-daily_loss_basis`` (forces a block).
        - otherwise ``min(worst_nav_pnl, nav - nav_day_start)``, where ``nav``
          is the caller's own NAV (live passes the ticker-adjusted one).
        """
        if self.state_unknown:
            return -self.daily_loss_basis
        return min(self.worst_nav_pnl, nav - self.nav_day_start)


class PortfolioAccounting:
    """
    Tracks equity curve, realised/unrealised PnL, fees, and drawdown
    for a single trading run.

    Parameters
    ----------
    run_id:
        Unique identifier for the trading run.
    initial_cash:
        Starting cash balance in quote currency.
    clock:
        Risk-day clock (UTC).  Defaults to the wall clock (:func:`_utc_now`);
        injectable for paper/live tests.  A backtest overrides it by pushing
        bar time through :meth:`advance_clock`.
    """

    def __init__(
        self,
        run_id: str,
        initial_cash: Decimal = Decimal("10000"),
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._run_id = run_id
        self._initial_cash = initial_cash
        self._cash = initial_cash

        # Peak equity tracking (starts at initial cash)
        self._peak_equity = initial_cash

        # Risk-day clock (WP1.5 C-06).  ``_clock`` None -> module ``_utc_now``,
        # looked up at call time so tests can monkeypatch the seam.
        self._clock: Callable[[], datetime] | None = clock
        self._sim_now: datetime | None = None

        # Daily PnL accumulator (realised only; rolls at the UTC day boundary).
        # ``_daily_pnl_date`` stays None until the first event: the constructor
        # must NOT initialise the day from the wall clock, or every backtest
        # bar (in the past) would look like a clock regression (V5).
        self._daily_pnl = Decimal("0")
        self._daily_pnl_date: date | None = None

        # WP1.5 mark-to-market day state (the daily-loss gate input).
        self._nav_day_start: Decimal = initial_cash
        self._worst_nav_pnl: Decimal = Decimal("0")
        self._day_start_source: DayStartSource = "run_start"
        # One-shot re-base flag (SYN-01 / Option B): set by an in-run roll (and
        # by some resume seeds), consumed by the first NAV observation of the
        # day in :meth:`_record_nav`.
        self._day_start_pending: bool = False
        self._regression_logged_day: date | None = None

        # Cumulative metrics
        self._total_realised_pnl = Decimal("0")
        self._total_fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0

        # Equity curve: list of (datetime, equity_value) tuples
        self._equity_curve: list[tuple[datetime, Decimal]] = [
            (datetime.now(tz=UTC), initial_cash),
        ]

        # Completed trade history
        self._trade_history: list[TradeResult] = []

        # Position snapshots: symbol -> last known Position (for unrealised PnL)
        self._position_snapshots: dict[str, Position] = {}

        # Optional callback fired after each trade is recorded (for adaptive learning)
        self.on_trade_recorded: Callable[[TradeResult], None] | None = None

        self._log = structlog.get_logger(__name__).bind(
            run_id=run_id,
            component="portfolio_accounting",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def cash(self) -> Decimal:
        """Current cash balance in quote currency."""
        return self._cash

    @property
    def initial_cash(self) -> Decimal:
        """Starting cash balance."""
        return self._initial_cash

    @property
    def total_realised_pnl(self) -> Decimal:
        """Cumulative realised PnL across all closed trades."""
        return self._total_realised_pnl

    @property
    def total_fees_paid(self) -> Decimal:
        """Cumulative fees paid across all fills."""
        return self._total_fees_paid

    @property
    def total_trades(self) -> int:
        """Total number of completed round-trip trades."""
        return self._total_trades

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades. Returns 0.0 if no trades."""
        if self._total_trades == 0:
            return 0.0
        return self._winning_trades / self._total_trades

    @property
    def current_equity(self) -> Decimal:
        """
        Current total portfolio equity (cash + open position value).

        This is the public accessor for the equity calculation that
        consolidates all position snapshots.  Prefer this over calling
        ``get_equity(list(self._position_snapshots.values()))`` from
        outside the class.

        Returns
        -------
        Decimal:
            Total portfolio value in quote currency.
        """
        return self.get_equity(list(self._position_snapshots.values()))

    def get_open_positions(self) -> list[Position]:
        """
        Return positions with non-zero quantity (open positions only).

        Mirrors the ``get_position`` / ``get_trade_history`` accessor style.
        Returns a new list; mutations do not propagate to internal state.

        Returns
        -------
        list[Position]:
            All positions where ``is_flat`` is False (quantity > 0).
        """
        return [pos for pos in self._position_snapshots.values() if not pos.is_flat]

    def get_position(self, symbol: str) -> Position | None:
        """
        Return the current position snapshot for a symbol, or None if
        no position exists for that symbol.

        Parameters
        ----------
        symbol:
            Trading pair identifier (e.g. "BTC/USDT").

        Returns
        -------
        Position | None:
            The current position snapshot, or None.
        """
        return self._position_snapshots.get(symbol)

    # ------------------------------------------------------------------
    # Core update methods
    # ------------------------------------------------------------------

    def update_position(
        self,
        fill: Fill,
        current_price: Decimal,
    ) -> None:
        """
        Update portfolio state after a fill event.

        This method:
        1. Auto-resets daily PnL if day boundary has been crossed.
        2. Updates the cash balance based on the fill.
        3. Tracks fees.
        4. Updates the position snapshot for the symbol.
        5. Recalculates unrealised PnL for the position.
        6. Records a point on the equity curve.
        7. Updates peak equity and daily PnL.

        Parameters
        ----------
        fill:
            The fill event to process.
        current_price:
            Current market price of the asset (for unrealised PnL).
        """
        # ``at`` (position/curve stamps) stays wall time until WP2.2; the risk
        # day is resolved through the single ``_now()`` seam (WP1.5 C-06).
        self._apply_fill(
            fill, current_price, at=datetime.now(tz=UTC), day_at=self._now()
        )

    def _apply_fill(
        self,
        fill: Fill,
        current_price: Decimal,
        *,
        at: datetime,
        day_at: datetime | None = None,
        record_curve_point: bool = True,
    ) -> None:
        """
        Shared body for :meth:`update_position` and :meth:`from_fills` (WP1.8
        S6/A-06).

        Identical accounting to the original ``update_position`` body, with
        two parameterisations needed for a resume replay:

        - ``at`` replaces the implicit ``datetime.now(tz=UTC)`` so a replayed
          fill's ``opened_at``/``updated_at``/daily-PnL date come from the
          fill's own ``executed_at`` (O2), not wall-clock "now".
        - ``record_curve_point`` suppresses the equity-curve append (O6: a
          replay has no side effects on the equity curve -- ``from_fills``
          seeds a single accurate point *after* the full replay instead).

        Parameters
        ----------
        fill:
            The fill event to process.
        current_price:
            Market price used for unrealised PnL (the live/paper caller
            passes the latest bar close; the replay passes the fill's own
            price -- there is no other price available for a past fill).
        at:
            Timestamp to stamp on new/updated positions (and, when ``day_at``
            is not given, to resolve the risk-day boundary).
        day_at:
            WP1.5: the risk-clock reading that decides which UTC day this
            fill belongs to.  ``update_position`` passes ``_now()`` (bar time
            in a backtest); a resume replay leaves it ``None`` so the fill's
            own ``executed_at`` decides.
        record_curve_point:
            When ``True`` (the default, used by ``update_position``), append
            ``(at, equity)`` to the equity curve.  ``from_fills`` passes
            ``False`` for every replayed fill.
        """
        # Roll the risk day if a UTC day boundary has been crossed.  This runs
        # BEFORE the fill changes anything, so the day-start NAV captured by a
        # roll is the NAV at the previous marks (WP1.5 §5.1).
        self._ensure_daily_pnl_date(day_at if day_at is not None else at)

        # Track fees
        self._total_fees_paid += fill.fee

        # Update cash based on fill side
        if fill.side == OrderSide.BUY:
            # Buying: cash decreases by (quantity * price + fee)
            cost = fill.quantity * fill.price + fill.fee
            self._cash -= cost
        elif fill.side == OrderSide.SELL:
            # Selling: cash increases by (quantity * price - fee)
            proceeds = fill.quantity * fill.price - fill.fee
            self._cash += proceeds

        # Update position snapshot
        position = self._position_snapshots.get(fill.symbol)
        now = at

        if fill.side == OrderSide.BUY:
            if position is None or position.is_flat:
                # New position -- use all-in cost basis (includes entry fee)
                all_in_entry_price = blend_entry_price(
                    Decimal("0"), Decimal("0"), fill
                )
                self._position_snapshots[fill.symbol] = Position(
                    symbol=fill.symbol,
                    run_id=self._run_id,
                    quantity=fill.quantity,
                    average_entry_price=all_in_entry_price,
                    current_price=current_price,
                    realised_pnl=Decimal("0"),
                    unrealised_pnl=(current_price - all_in_entry_price) * fill.quantity,
                    total_fees_paid=fill.fee,
                    opened_at=now,
                    updated_at=now,
                )
            else:
                # Add to existing position
                total_qty = position.quantity + fill.quantity
                new_avg = blend_entry_price(
                    position.quantity, position.average_entry_price, fill
                )

                unrealised = (current_price - new_avg) * total_qty

                self._position_snapshots[fill.symbol] = position.model_copy(
                    update={
                        "quantity": total_qty,
                        "average_entry_price": new_avg,
                        "current_price": current_price,
                        "unrealised_pnl": unrealised,
                        "total_fees_paid": position.total_fees_paid + fill.fee,
                        "updated_at": now,
                    }
                )

        elif fill.side == OrderSide.SELL:
            if position is not None and not position.is_flat:
                sell_qty = min(fill.quantity, position.quantity)
                pnl = (fill.price - position.average_entry_price) * sell_qty - fill.fee
                remaining_qty = position.quantity - sell_qty

                # Accumulate realised PnL
                self._total_realised_pnl += pnl

                # Update daily PnL (date already ensured at top of method)
                self._daily_pnl += pnl

                if remaining_qty <= Decimal("0"):
                    # Position fully closed
                    self._position_snapshots[fill.symbol] = position.model_copy(
                        update={
                            "quantity": Decimal("0"),
                            "current_price": current_price,
                            "realised_pnl": position.realised_pnl + pnl,
                            "unrealised_pnl": Decimal("0"),
                            "total_fees_paid": position.total_fees_paid + fill.fee,
                            "updated_at": now,
                        }
                    )
                else:
                    unrealised = (
                        (current_price - position.average_entry_price) * remaining_qty
                    )
                    self._position_snapshots[fill.symbol] = position.model_copy(
                        update={
                            "quantity": remaining_qty,
                            "current_price": current_price,
                            "realised_pnl": position.realised_pnl + pnl,
                            "unrealised_pnl": unrealised,
                            "total_fees_paid": position.total_fees_paid + fill.fee,
                            "updated_at": now,
                        }
                    )

        # Record equity curve point
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        if record_curve_point:
            self._equity_curve.append((now, current_equity))

        # Update peak equity and the daily-loss latch (one of the only two
        # writers of peak / worst -- the other is update_market_prices).
        self._record_nav(current_equity)

        self._log.debug(
            "portfolio.position_updated",
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=str(fill.quantity),
            price=str(fill.price),
            fee=str(fill.fee),
            cash=str(self._cash),
            equity=str(current_equity),
        )

    @classmethod
    def from_fills(
        cls,
        run_id: str,
        initial_cash: Decimal,
        fills: Sequence[Fill],
        *,
        peak_equity_hint: Decimal | None = None,
        now: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
        daily_seed: ResumeDailySeed | None = None,
    ) -> PortfolioAccounting:
        """
        Rebuild a :class:`PortfolioAccounting` from persisted fill history
        (WP1.8a S6/A-06 -- the orphan-resume replay).

        Replays ``fills`` in ``(executed_at, fill_id)`` order through the
        same accounting path as :meth:`update_position` (``_apply_fill``),
        so the rebuilt cash/position/realised-PnL state is *identical* to
        what incremental application would have produced -- input order
        never changes the result (sorting is stable and total).

        Differences from live incremental application (O6):
        - No equity-curve point is recorded per fill; a single accurate
          point, timestamped ``now``, is seeded once after the full replay
          instead of the constructor's ``initial_cash``-only placeholder.
        - No ``TradeResult`` is recorded and ``on_trade_recorded`` never
          fires -- ``_apply_fill`` never calls ``record_trade`` itself, so
          this holds automatically.
        - ``peak_equity_hint`` (typically the highest persisted equity
          snapshot) raises the peak but never lowers it (drawdown must
          never be erased by a resume).
        - Daily PnL from fills whose day differs from ``now`` is excluded:
          the final :meth:`_ensure_daily_pnl_date` call resets the
          accumulator if the last replayed day has rolled over relative to
          ``now`` (yesterday's realised losses/gains never count against
          today's daily-loss gate after a resume).
        - WP1.5 (§5.7): the mark-to-market day state (``nav_day_start``,
          ``worst``, source) is *overwritten* from ``daily_seed`` after the
          replay.  The rebuilt NAV values positions at old fill prices, so it
          is never fed into ``worst`` (it would fabricate a loss for a
          position bought last week at 100 and now worth 80).

        Parameters
        ----------
        run_id:
            The run being rebuilt (same id -- a resume never changes it).
        initial_cash:
            The run's original starting capital.
        fills:
            Every persisted fill for this run, in any order.
        peak_equity_hint:
            The highest known equity value before the replay (e.g. the max
            persisted equity snapshot).  Never lowers the computed peak.
        now:
            Injectable clock for deterministic tests; defaults to the risk
            clock (wall clock UTC).
        clock:
            Risk-day clock for the rebuilt portfolio (see ``__init__``).
        daily_seed:
            Persisted day-start hints (WP1.5b: ``ResumeSnapshot.daily_seed``,
            read from the equity-snapshot table).  Every field is
            type-validated strictly (no coercion); an invalid seed degrades to
            ``unknown`` and never aborts the resume.  ``None`` (legacy / test
            callers, or a snapshot that never read hints) uses
            ``replay_estimate``: the gate *value* is never weaker than HEAD's
            realised-today value (``worst = min(0, realised_today)``), but the
            threshold uses the fixed day-start basis (C-02) and mark drift
            since the original fills counts toward the resume day until
            00:00 UTC (it can be stricter *or*, at the threshold, looser than
            HEAD).

        Returns
        -------
        PortfolioAccounting
            A new instance whose cash/position/realised-PnL/peak state
            matches sequential ``update_position`` application of the same
            fills.
        """
        portfolio = cls(run_id=run_id, initial_cash=initial_cash, clock=clock)
        resolved_now = now if now is not None else portfolio._now()

        ordered_fills = sorted(fills, key=replay_sort_key)
        for fill in ordered_fills:
            portfolio._apply_fill(fill, fill.price, at=fill.executed_at, record_curve_point=False)

        if peak_equity_hint is not None and peak_equity_hint > portfolio._peak_equity:
            portfolio._peak_equity = peak_equity_hint

        # Roll the daily-PnL accumulator forward to "now" -- if the last
        # replayed fill is from a prior UTC day, this resets it to 0 so
        # yesterday's realised PnL never counts as today's.
        portfolio._ensure_daily_pnl_date(resolved_now)

        rebuilt_equity = portfolio.get_equity(list(portfolio._position_snapshots.values()))
        portfolio._equity_curve = [(resolved_now, rebuilt_equity)]
        if rebuilt_equity > portfolio._peak_equity:
            portfolio._peak_equity = rebuilt_equity

        portfolio._seed_daily_state(
            daily_seed=daily_seed,
            resolved_now=resolved_now,
            rebuilt_nav=rebuilt_equity,
        )

        portfolio._log.info(
            "portfolio.rebuilt_from_fills",
            fill_count=len(ordered_fills),
            cash=str(portfolio._cash),
            equity=str(rebuilt_equity),
            peak_equity=str(portfolio._peak_equity),
        )
        return portfolio

    def update_market_prices(
        self,
        prices: dict[str, Decimal],
    ) -> None:
        """
        Update unrealised PnL for all positions with current market prices.

        Call this on every bar to keep the equity curve and unrealised
        PnL accurate between fills.

        WP1.5: this is (with fills) one of the only two writers of the peak
        and of the daily-loss latch.  It first rolls the risk day (using the
        NAV at the *previous* marks as a provisional new day start), then
        re-marks; the first NAV observation of the new day may raise that
        start once (:meth:`_record_nav`, SYN-01 / Option B).

        Parameters
        ----------
        prices:
            Mapping of symbol -> current market price.
        """
        self._ensure_daily_pnl_date(self._now())
        now = datetime.now(tz=UTC)  # curve / position stamps stay wall time (WP2.2)
        for symbol, price in prices.items():
            position = self._position_snapshots.get(symbol)
            if position is not None and not position.is_flat:
                unrealised = (
                    (price - position.average_entry_price) * position.quantity
                )
                self._position_snapshots[symbol] = position.model_copy(
                    update={
                        "current_price": price,
                        "unrealised_pnl": unrealised,
                        "updated_at": now,
                    }
                )

        # Record equity curve point
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        self._equity_curve.append((now, current_equity))

        # Update peak equity and the daily-loss latch
        self._record_nav(current_equity)

    # ------------------------------------------------------------------
    # Equity calculations
    # ------------------------------------------------------------------

    def get_equity(self, positions: list[Position]) -> Decimal:
        """
        Calculate total portfolio equity.

        Equity = cash + sum(position.notional_value for open positions)

        Parameters
        ----------
        positions:
            List of current positions (typically from the execution engine
            or from internal snapshots).

        Returns
        -------
        Decimal:
            Total portfolio value in quote currency.
        """
        position_value = Decimal("0")
        for pos in positions:
            if not pos.is_flat:
                position_value += pos.notional_value
        return self._cash + position_value

    def get_peak_equity(self) -> Decimal:
        """
        Return the highest equity value recorded since run start.

        Returns
        -------
        Decimal:
            Peak equity in quote currency.
        """
        return self._peak_equity

    def get_daily_pnl(self) -> Decimal:
        """
        Return the realised net PnL for the current UTC trading day.

        FIX-07: realised-only -- unrealised mark-to-market moves never change
        this figure (the API / UI keep showing it).  WP1.5 (C5): the read is
        *rolled* -- on a new UTC day it reports 0 even before the first fill
        or mark of that day, without mutating anything (pure read).

        The daily-loss *gate* does not use this number; it uses the
        mark-to-market :meth:`risk_snapshot` (D-WP15-A).

        Returns
        -------
        Decimal:
            Realised daily PnL in quote currency.
        """
        return self._realised_today(self._now())

    def get_drawdown(self) -> float:
        """
        Calculate the current drawdown as a percentage of peak equity.

        Drawdown = (peak_equity - current_equity) / peak_equity

        Returns
        -------
        float:
            Drawdown as a decimal fraction (e.g. 0.05 = 5%).
            Returns 0.0 if peak equity is zero or negative.
        """
        if self._peak_equity <= Decimal("0"):
            return 0.0

        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return max(0.0, float(drawdown))

    def get_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown from the equity curve.

        Iterates through the entire equity curve to find the worst
        peak-to-trough decline.

        Returns
        -------
        float:
            Maximum drawdown as a decimal fraction.
        """
        if len(self._equity_curve) < 2:
            return 0.0

        peak = Decimal("0")
        max_dd = Decimal("0")

        for _, equity in self._equity_curve:
            if equity > peak:
                peak = equity
            if peak > Decimal("0"):
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        return float(max_dd)

    def get_total_return(self) -> float:
        """
        Calculate the total return as a percentage of initial cash.

        Returns
        -------
        float:
            Total return as a decimal fraction (e.g. 0.10 = 10%).
        """
        if self._initial_cash <= Decimal("0"):
            return 0.0

        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return float(
            (current_equity - self._initial_cash) / self._initial_cash
        )

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeResult) -> None:
        """
        Record a completed round-trip trade.

        Updates win/loss counters and stores the trade in history.

        Parameters
        ----------
        trade:
            A fully populated TradeResult from a closed position.
        """
        self._trade_history.append(trade)
        self._total_trades += 1

        if trade.realised_pnl > Decimal("0"):
            self._winning_trades += 1
        elif trade.realised_pnl < Decimal("0"):
            self._losing_trades += 1
        # break-even (== 0): neither counter incremented

        self._log.info(
            "portfolio.trade_recorded",
            trade_id=str(trade.trade_id),
            symbol=trade.symbol,
            side=trade.side.value,
            entry_price=str(trade.entry_price),
            exit_price=str(trade.exit_price),
            quantity=str(trade.quantity),
            realised_pnl=str(trade.realised_pnl),
            return_pct=f"{trade.return_pct:.4%}",
        )

        # Fire adaptive learning callback (CR-001: log errors, never corrupt portfolio)
        if self.on_trade_recorded is not None:
            try:
                self.on_trade_recorded(trade)
            except Exception:
                self._log.warning("portfolio.trade_callback_error", exc_info=True)

    def get_trade_history(self) -> list[TradeResult]:
        """
        Return the complete trade history for this run.

        Returns
        -------
        list[TradeResult]:
            All completed trades, ordered by recording time.
        """
        return list(self._trade_history)

    # ------------------------------------------------------------------
    # Daily PnL management
    # ------------------------------------------------------------------

    def reset_daily_pnl(self) -> None:
        """
        Reset the realised daily PnL accumulator.

        Manual hook (no production caller): the UTC day roll in
        :meth:`_ensure_daily_pnl_date` is automatic.  Uses the risk clock
        (:meth:`_now`), never the wall clock directly.  It rolls the risk day
        first (MTM-aware) so the hook can never pre-empt the roll and strand
        the previous day's latch, and it deliberately does NOT touch the
        mark-to-market day state otherwise (``nav_day_start`` / latch): a
        manual call must not be usable to lift a same-day daily-loss latch.
        """
        self._ensure_daily_pnl_date(self._now())
        if self._daily_pnl != Decimal("0"):
            self._log.info(
                "portfolio.daily_pnl_reset",
                previous_daily_pnl=str(self._daily_pnl),
                date=str(self._daily_pnl_date),
            )
        self._daily_pnl = Decimal("0")

    # ------------------------------------------------------------------
    # Risk clock and day roll (WP1.5)
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        """The risk clock: simulated time if one was pushed, else the clock.

        The single seam every risk-day decision goes through (WP2.2 reuses it
        for curve / trade stamps).
        """
        if self._sim_now is not None:
            return self._sim_now
        clock = self._clock if self._clock is not None else _utc_now
        return clock()

    def advance_clock(self, at: datetime) -> None:
        """Push simulated (bar) time and roll the risk day if it advanced.

        Used by ``StrategyEngine.run_backtest`` once per bar, warmup
        included, so a backtest never reads the wall clock on the risk path.
        Forward-only: an equal or earlier time is a no-op (multi-symbol bar
        misalignment, pre-WP2.3).  The first push sets the day if no event
        has set it yet (the constructor leaves it ``None``).  If a mutator
        already stamped the risk day from the wall clock, an earlier pushed
        time is a regression (warning, no re-base).

        Raises
        ------
        ValueError
            If ``at`` is naive.  Bars are always timezone-aware, so a naive
            value is a programming error, not data to be normalised (C-12).
        """
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("advance_clock requires a timezone-aware datetime")
        at = at.astimezone(UTC)
        if self._sim_now is not None and at <= self._sim_now:
            if at < self._sim_now:
                self._log.debug(
                    "portfolio.advance_clock_ignored",
                    at=at.isoformat(),
                    sim_now=self._sim_now.isoformat(),
                )
            return
        self._sim_now = at
        self._ensure_daily_pnl_date(at, from_clock=True)

    def _ensure_daily_pnl_date(
        self,
        at: datetime | None = None,
        *,
        from_clock: bool = False,
    ) -> None:
        """
        Roll the risk day forward if ``at`` is on a later UTC day (mutating).

        Forward-only.  On the first call the day is simply set and nothing
        else is reset (the run starts with ``nav_day_start = initial_cash``).
        On a roll: the new ``nav_day_start`` is provisionally the NAV at the
        previous marks (``current_equity``) and a one-shot re-base is armed
        (``_day_start_pending``; consumed by the first NAV observation of the
        day in :meth:`_record_nav`, which may raise the start to that NAV --
        a depressed pre-roll mark must not become the day's cushion), the
        realised accumulator and the latch reset, and the source becomes
        ``roll``.  A time on an earlier day never
        rolls back (NTP step, mis-ordered replay): it only logs
        ``portfolio.clock_regression`` once per risk day.

        Parameters
        ----------
        at:
            The reading to resolve "today" from; defaults to :meth:`_now`.
            ``from_clock`` marks a call from :meth:`advance_clock` (the roll
            is logged at debug level -- a backtest rolls once per bar-day).
        """
        day = _utc_day(at if at is not None else self._now())
        current = self._daily_pnl_date
        if current is None:
            self._daily_pnl_date = day
            return
        if day > current:
            nav = self.current_equity
            prev_realised = self._daily_pnl
            prev_worst = self._worst_nav_pnl
            self._nav_day_start = nav
            self._day_start_pending = True
            self._daily_pnl = Decimal("0")
            self._worst_nav_pnl = Decimal("0")
            self._day_start_source = "roll"
            self._daily_pnl_date = day
            log = self._log.debug if from_clock else self._log.info
            log(
                "portfolio.daily_roll",
                prev_day=str(current),
                day=str(day),
                prev_realised=str(prev_realised),
                prev_worst=str(prev_worst),
                nav_day_start=str(nav),
            )
        elif day < current and self._regression_logged_day != current:
            self._regression_logged_day = current
            self._log.warning(
                "portfolio.clock_regression",
                risk_day=str(current),
                seen_day=str(day),
            )

    def _record_nav(self, nav: Decimal) -> None:
        """Fold a fresh NAV into the peak and the latch (R-09).

        Called only from :meth:`_apply_fill` and :meth:`update_market_prices`
        -- reads (``risk_snapshot``, engine NAV snapshots) never get here.
        """
        if nav > self._peak_equity:
            self._peak_equity = nav
        if self._day_start_pending:
            # One-shot re-base (SYN-01): the first NAV observation of the day
            # (mark or fill) raises the provisional start if it is higher, so a
            # depressed last pre-roll mark never becomes the day's cushion.
            # Done before folding into ``worst``; ``worst`` is not re-expressed.
            self._day_start_pending = False
            if nav > self._nav_day_start:
                self._log.debug(
                    "portfolio.day_start_rebased",
                    prev_nav_day_start=str(self._nav_day_start),
                    nav_day_start=str(nav),
                )
                self._nav_day_start = nav
        self._worst_nav_pnl = min(self._worst_nav_pnl, nav - self._nav_day_start)

    def _realised_today(self, now: datetime) -> Decimal:
        """Realised daily PnL as of ``now`` (virtual roll, pure)."""
        if self._daily_pnl_date is not None and _utc_day(now) > self._daily_pnl_date:
            return Decimal("0")
        return self._daily_pnl

    def risk_snapshot(self) -> RiskSnapshot:
        """Pure read of the risk state (WP1.5 C-05).

        Never mutates anything.  When the risk clock is already on a later
        UTC day than the stored day but no mutator has rolled it yet, the
        snapshot reports what a roll *would* produce (``nav_day_start = nav``,
        ``worst = 0``, ``realised = 0``, source ``roll``): the NAV at the
        existing marks equals the NAV a mutating roll would have captured.
        While a re-base is pending (:meth:`_record_nav`) the snapshot reports
        the stored provisional start and never consumes the flag.
        """
        now = self._now()
        nav = self.current_equity
        peak = max(self._peak_equity, nav)
        today = _utc_day(now)
        stored_day = self._daily_pnl_date
        if stored_day is not None and today > stored_day:
            day = today
            nav_day_start = nav
            worst = Decimal("0")
            realised = Decimal("0")
            source: DayStartSource = "roll"
        else:
            day = stored_day if stored_day is not None else today
            nav_day_start = self._nav_day_start
            worst = min(self._worst_nav_pnl, nav - nav_day_start)
            realised = self._daily_pnl
            source = self._day_start_source
        drawdown = (peak - nav) / peak if peak > Decimal("0") else Decimal("0")
        return RiskSnapshot(
            as_of=now,
            day=day,
            nav=nav,
            peak_equity=peak,
            drawdown_pct=max(Decimal("0"), drawdown),
            nav_day_start=nav_day_start,
            daily_loss_basis=max(Decimal("0"), min(nav_day_start, self._initial_cash)),
            nav_pnl=nav - nav_day_start,
            worst_nav_pnl=worst,
            realised_pnl=realised,
            day_start_source=source,
            open_position_count=len(self.get_open_positions()),
        )

    def _seed_daily_state(
        self,
        *,
        daily_seed: ResumeDailySeed | None,
        resolved_now: datetime,
        rebuilt_nav: Decimal,
    ) -> None:
        """Overwrite the MTM day state after a resume replay (§5.7).

        The rebuilt NAV never feeds ``worst`` (C-07a); ``worst`` comes only
        from persisted equity-snapshot hints.  Evaluated in table order.

        WP1.5b (R-1 / C-3): every seed field is type-validated first, before
        any comparison, arithmetic or assignment
        (:func:`_invalid_seed_field`), and the whole computation is guarded.
        An invalid seed or any exception degrades to
        ``day_start_source="unknown"`` (``resume.daily_state_unknown`` with
        ``reason="seed_invalid"``; BUYs blocked until 00:00 UTC, exits
        unaffected) -- seeding never aborts a resume, a protective resume
        included.

        ``_day_start_pending`` (the one-shot re-base, SYN-01) is assigned
        unconditionally here -- the replay's own value is overwritten.  It is
        armed only where a higher first observation could not lift a seeded
        latch (``worst == 0``) and the start is not already hardened by a
        persisted first-of-day equity.
        """
        today = _utc_day(resolved_now)
        realised_today = self._daily_pnl
        self._daily_pnl_date = today
        self._regression_logged_day = None

        nav_day_start: Decimal
        worst: Decimal
        source: DayStartSource
        unknown_reason = "no_day_start_hint"
        pending: bool
        failure: tuple[str, str] | None = None
        try:
            failure = _invalid_seed_field(daily_seed)
            if failure is None:
                if daily_seed is None:
                    # Legacy / test callers: the gate value is never weaker than HEAD's
                    # realised-today value (worst = min(0, realised_today)); the
                    # threshold uses the fixed day-start basis and mark drift since the
                    # fills counts (see ``from_fills``).
                    nav_day_start = rebuilt_nav - realised_today
                    worst = min(Decimal("0"), realised_today)
                    source = "replay_estimate"
                elif daily_seed.as_of_day < today:
                    # Midnight race: the hints describe yesterday -> roll semantics.
                    nav_day_start = rebuilt_nav
                    worst = Decimal("0")
                    source = "roll"
                elif daily_seed.as_of_day > today:
                    nav_day_start = rebuilt_nav
                    worst = Decimal("0")
                    source = "unknown"
                    unknown_reason = "seed_from_the_future"
                else:
                    day_min = daily_seed.day_min_equity
                    if daily_seed.day_start_equity is not None:
                        nav_day_start = daily_seed.day_start_equity
                        source = "snapshot"
                    elif (
                        daily_seed.run_started_at is not None
                        and _utc_day(daily_seed.run_started_at) == today
                    ):
                        nav_day_start = self._initial_cash
                        source = "run_start"
                    else:
                        nav_day_start = rebuilt_nav
                        day_min = None
                        source = "unknown"
                    if (
                        source == "snapshot"
                        and daily_seed.day_first_equity is not None
                        and daily_seed.day_first_equity > nav_day_start
                    ):
                        # Robust day start: a depressed pre-midnight snapshot never
                        # becomes the day's cushion (SYN-01 / Option B).
                        nav_day_start = daily_seed.day_first_equity
                    if day_min is not None:
                        worst = min(Decimal("0"), day_min - nav_day_start)
                    else:
                        worst = Decimal("0")

                # One-shot re-base flag per the 1.5a synthesis seed table.  Never armed
                # when the seed carries a loss (a latch can never be relaxed).
                pending = False
                if worst == Decimal("0"):
                    if source in ("roll", "replay_estimate"):
                        pending = True
                    elif source == "snapshot" and (
                        daily_seed is not None
                        and daily_seed.day_first_equity is None
                        and daily_seed.day_min_equity is None
                    ):
                        pending = True
        except Exception as exc:  # seeding must never abort a resume
            failure = ("seeding", type(exc).__name__)
        if failure is not None:
            nav_day_start = rebuilt_nav
            worst = Decimal("0")
            source = "unknown"
            unknown_reason = "seed_invalid"
            pending = False

        self._nav_day_start = nav_day_start
        self._worst_nav_pnl = worst
        self._day_start_source = source
        self._day_start_pending = pending
        if source == "unknown":
            self._log.error(
                "resume.daily_state_unknown",
                reason=unknown_reason,
                # field / type names only, never the offending value
                **({"field": failure[0], "value_type": failure[1]} if failure else {}),
                nav_day_start=str(nav_day_start),
                peak_equity=str(self._peak_equity),
            )
        else:
            self._log.info(
                "resume.risk_state_seeded",
                source=source,
                nav_day_start=str(nav_day_start),
                worst=str(worst),
                peak_equity=str(self._peak_equity),
            )

    # ------------------------------------------------------------------
    # Equity curve access
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """
        Return the full equity curve as (timestamp, equity) tuples.

        Returns
        -------
        list[tuple[datetime, Decimal]]:
            Equity curve ordered by timestamp ascending.
        """
        return list(self._equity_curve)

    # ------------------------------------------------------------------
    # Summary / snapshot
    # ------------------------------------------------------------------

    def get_summary(self) -> PortfolioSummary:
        """
        Return a comprehensive portfolio summary dict.

        Suitable for API serialization, logging, and dashboard display.

        Returns
        -------
        PortfolioSummary:
            Portfolio summary with all key metrics (a ``TypedDict``; WP1.6).
        """
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return {
            "run_id": self._run_id,
            "initial_cash": str(self._initial_cash),
            "current_cash": str(self._cash),
            "current_equity": str(current_equity),
            "peak_equity": str(self._peak_equity),
            "total_return_pct": self.get_total_return(),
            "total_realised_pnl": str(self._total_realised_pnl),
            "total_fees_paid": str(self._total_fees_paid),
            "daily_pnl": str(self.get_daily_pnl()),
            "drawdown_pct": self.get_drawdown(),
            "max_drawdown_pct": self.get_max_drawdown(),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": self.win_rate,
            "open_positions": len(
                [p for p in self._position_snapshots.values() if not p.is_flat]
            ),
            "equity_curve_length": len(self._equity_curve),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        current_equity = self.get_equity(
            list(self._position_snapshots.values())
        )
        return (
            f"PortfolioAccounting("
            f"run_id={self._run_id!r}, "
            f"equity={current_equity}, "
            f"trades={self._total_trades}, "
            f"drawdown={self.get_drawdown():.2%})"
        )
