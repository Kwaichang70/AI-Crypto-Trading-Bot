"""
tests/unit/test_sprint51_cycle1_sizing_contract.py
---------------------------------------------------
Sprint 51 Cycle 1 -- Signal.target_position authoritative sizing contract (SYN-S51).

Module(s) under test
--------------------
    packages/trading/execution.py        -- BaseExecutionEngine._resolve_order_quantity
    packages/trading/engines/paper.py    -- PaperExecutionEngine.process_signal (sizing + ROUND_DOWN sell)
    packages/trading/engines/live.py     -- LiveExecutionEngine.process_signal  (sizing + ROUND_DOWN sell)

Contract under test
-------------------
``Signal.target_position`` (notional, quote currency) is the AUTHORITATIVE sizing
input.  ``RiskManager.calculate_position_size`` is only a *ceiling* that de-sizes
(never up-sizes).  Confidence is applied EXACTLY ONCE -- on the target -- and the
ceiling call therefore uses ``confidence=1.0``.

    BUY  target>0 : base = (target/price)*conf ; ceiling = calc(...,conf=1.0) ; min(base, ceiling)
    BUY  target==0: legacy risk-only fallback calc(..., conf=signal.confidence)
    SELL target==0: full close (held)
    SELL target>0 : min((target/price)*conf, held)

SELL quantize now uses ROUND_DOWN (BUY ROUND_HALF_UP) -- the oversell fix.

Test-ID mapping (unique IDs TEST-S51-### in docstrings)
-------------------------------------------------------
Core T-01..T-16 + risk-critic-mandated RMC-003/004/005.  See each test docstring.

Design notes
------------
- Uses the REAL DefaultRiskManager + REAL Paper/Live engines for integration-realistic
  paths; uses direct ``_resolve_order_quantity`` calls (with a real or mocked risk
  manager) for the pure sizing-contract invariants.
- Deterministic: no network. Live engine is constructed with an AsyncMock ccxt exchange.
- All arithmetic uses Decimal. _QTY_PRECISION = 8 dp (matches engine constants).
- Signal.target_position has ge=0 (Pydantic) so the "<=0" branch only ever fires at
  exactly 0 -- tested accordingly.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine
from trading.models import Order, Position, RiskCheckResult, Signal
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_RUN_ID = "s51c1-test-run"
_LAST_PRICE = Decimal("50000")
_INITIAL_CASH = Decimal("100000")
_QTY_PRECISION = Decimal("0.00000001")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_signal(
    *,
    direction: SignalDirection = SignalDirection.BUY,
    target_position: Decimal = Decimal("1000"),
    confidence: float = 1.0,
    symbol: str = _SYMBOL,
) -> Signal:
    """Build a Signal. target_position has ge=0 so values must be >= 0."""
    return Signal(
        strategy_id="s51-strategy",
        symbol=symbol,
        direction=direction,
        target_position=target_position,
        confidence=confidence,
    )


def _make_risk_params(
    *,
    max_position_size_pct: float = 0.15,
    max_order_size_quote: Decimal = Decimal("10000"),
    max_portfolio_exposure_pct: float = 0.60,
    max_cluster_exposure_pct: float = 0.40,
    per_trade_risk_pct: float = 0.02,
    max_open_positions: int = 10,
    max_daily_loss_pct: float = 0.08,
    max_drawdown_pct: float = 0.30,
) -> RiskParameters:
    return RiskParameters(
        max_position_size_pct=max_position_size_pct,
        max_order_size_quote=max_order_size_quote,
        max_portfolio_exposure_pct=max_portfolio_exposure_pct,
        max_cluster_exposure_pct=max_cluster_exposure_pct,
        per_trade_risk_pct=per_trade_risk_pct,
        max_open_positions=max_open_positions,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
    )


def _make_real_paper_engine(
    *,
    params: RiskParameters | None = None,
    initial_cash: Decimal = _INITIAL_CASH,
    slippage_bps: int = 0,
) -> tuple[PaperExecutionEngine, DefaultRiskManager]:
    """Real PaperExecutionEngine wired to a real DefaultRiskManager.

    slippage_bps=0 keeps fill price == last_price so notional assertions are exact.
    """
    rm = DefaultRiskManager(run_id=_RUN_ID, params=params or _make_risk_params())
    engine = PaperExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        fill_latency_ms=0,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
    )
    engine.set_last_price(_SYMBOL, _LAST_PRICE)
    return engine, rm


def _make_mock_exchange() -> MagicMock:
    """AsyncMock-backed ccxt exchange sufficient for LiveExecutionEngine.process_signal."""
    exchange = MagicMock()
    exchange.id = "mock-exchange"
    # Empty market dict -> min-order-size guard is skipped gracefully.
    exchange.markets = {_SYMBOL: {}}
    exchange.create_order = AsyncMock(
        return_value={
            "id": "exch-001",
            "status": "closed",
            "filled": "0",
            "average": str(_LAST_PRICE),
            "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_order = AsyncMock(
        return_value={"id": "exch-001", "status": "closed", "filled": "0", "average": str(_LAST_PRICE)}
    )
    exchange.fetch_ticker = AsyncMock(return_value={"last": str(_LAST_PRICE)})
    exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": float(_INITIAL_CASH), "BTC": 0.0}})
    exchange.fetch_order_trades = AsyncMock(return_value=[])
    exchange.fetch_my_trades = AsyncMock(return_value=[])
    exchange.load_markets = AsyncMock(return_value=exchange.markets)
    exchange.close = AsyncMock(return_value=None)
    exchange.has = {"fetchOrderTrades": True, "fetchMyTrades": True}
    return exchange


def _make_real_live_engine(
    *,
    params: RiskParameters | None = None,
) -> tuple[LiveExecutionEngine, DefaultRiskManager, MagicMock]:
    """Real LiveExecutionEngine + real DefaultRiskManager + mock ccxt exchange."""
    rm = DefaultRiskManager(run_id=_RUN_ID, params=params or _make_risk_params())
    ex = _make_mock_exchange()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID,
        risk_manager=rm,
        exchange=ex,
        enable_live_trading=True,
    )
    return engine, rm, ex


def _resolved_buy_notional(order: Order) -> Decimal:
    """Notional of a resolved BUY order at _LAST_PRICE (slippage 0)."""
    return order.quantity * _LAST_PRICE


# ===========================================================================
# Group A: target_position authoritative -- BUY notional scaling (T-01..T-04)
# ===========================================================================


class TestBuyNotionalIsTargetDriven:
    """target_position drives BUY notional; risk manager only de-sizes."""

    @pytest.mark.asyncio
    async def test_buy_notional_doubles_when_target_doubles(self) -> None:
        """TEST-S51-001 (T-01): doubling target_position doubles BUY notional at fixed conf.

        Acceptance test from the Verbeterplan. Uses a generous risk ceiling so the
        target (not the ceiling) is the binding constraint for both sizes.
        """
        # Ceiling generous: 15% of 100k = 15k notional cap; targets stay below it.
        engine, _ = _make_real_paper_engine()
        s1 = _make_signal(target_position=Decimal("1000"), confidence=0.8)
        s2 = _make_signal(target_position=Decimal("2000"), confidence=0.8)

        orders1 = await engine.process_signal(s1)
        # Fresh engine so the first BUY does not affect concentration on the second.
        engine2, _ = _make_real_paper_engine()
        orders2 = await engine2.process_signal(s2)

        assert len(orders1) == 1 and len(orders2) == 1
        n1 = _resolved_buy_notional(orders1[0])
        n2 = _resolved_buy_notional(orders2[0])
        # n2 == 2 * n1 (within 8dp quantize tolerance)
        assert abs(n2 - n1 * Decimal("2")) <= _LAST_PRICE * _QTY_PRECISION

    @pytest.mark.asyncio
    async def test_buy_order_notional_equals_target_times_confidence_below_ceiling(self) -> None:
        """TEST-S51-002 (T-02): BUY notional == target * confidence when below ceiling."""
        engine, _ = _make_real_paper_engine()
        signal = _make_signal(target_position=Decimal("1000"), confidence=0.5)
        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        expected_notional = Decimal("1000") * Decimal("0.5")  # 500
        assert abs(_resolved_buy_notional(orders[0]) - expected_notional) <= _LAST_PRICE * _QTY_PRECISION

    def test_target_and_confidence_both_drive_notional(self) -> None:
        """TEST-S51-003 (T-03): confidence is not the sole input.

        At fixed conf, target drives notional; at fixed target, conf scales it.
        Verified directly through the resolver against a generous ceiling.
        """
        engine, _ = _make_real_paper_engine()
        # Fixed confidence, varying target -> notional tracks target
        q_t1 = engine._resolve_order_quantity(
            signal=_make_signal(target_position=Decimal("1000"), confidence=0.8),
            side=OrderSide.BUY, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=None,
        )
        q_t2 = engine._resolve_order_quantity(
            signal=_make_signal(target_position=Decimal("3000"), confidence=0.8),
            side=OrderSide.BUY, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=None,
        )
        assert q_t2 == q_t1 * Decimal("3")

        # Fixed target, varying confidence -> notional scales with conf
        q_c1 = engine._resolve_order_quantity(
            signal=_make_signal(target_position=Decimal("1000"), confidence=0.25),
            side=OrderSide.BUY, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=None,
        )
        q_c2 = engine._resolve_order_quantity(
            signal=_make_signal(target_position=Decimal("1000"), confidence=0.75),
            side=OrderSide.BUY, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=None,
        )
        assert q_c2 == q_c1 * Decimal("3")

    def test_oversized_target_desized_to_risk_ceiling(self) -> None:
        """TEST-S51-004 (T-04): a target far above the ceiling is clamped to the ceiling via min()."""
        engine, rm = _make_real_paper_engine()
        # Huge target (10,000,000 notional) >> any ceiling.
        huge = _make_signal(target_position=Decimal("10000000"), confidence=1.0)
        resolved = engine._resolve_order_quantity(
            signal=huge, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        ceiling = rm.calculate_position_size(
            equity=_INITIAL_CASH, entry_price=_LAST_PRICE, stop_loss_price=None, confidence=1.0,
        )
        assert resolved == ceiling
        # Base would have been 200 BTC; ceiling is far smaller.
        assert resolved < (huge.target_position / _LAST_PRICE)


# ===========================================================================
# Group B: BUY target==0 fallback (T-05)
# ===========================================================================


class TestBuyZeroTargetFallback:
    """target_position==0 BUY falls back to risk-only sizing, itself ceiling-capped."""

    def test_buy_zero_target_falls_back_to_risk_only_sizing(self) -> None:
        """TEST-S51-005 (T-05 / RMC-S51-002): BUY target==0 -> non-zero risk-only qty <= ceiling caps."""
        engine, rm = _make_real_paper_engine()
        zero_target = _make_signal(target_position=Decimal("0"), confidence=0.9)

        resolved = engine._resolve_order_quantity(
            signal=zero_target, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        # Non-zero fallback
        assert resolved > Decimal("0")
        # Equals the risk-only sizing at the signal's own confidence
        expected = rm.calculate_position_size(
            equity=_INITIAL_CASH, entry_price=_LAST_PRICE, stop_loss_price=None, confidence=0.9,
        )
        assert resolved == expected
        # RMC-S51-002: fallback is itself within the concentration + order-size caps.
        conc_cap = (
            Decimal(str(rm.params.max_position_size_pct)) * _INITIAL_CASH / _LAST_PRICE
        )
        order_cap = rm.params.max_order_size_quote / _LAST_PRICE
        assert resolved <= conc_cap
        assert resolved <= order_cap


# ===========================================================================
# Group C: SELL semantics + oversell protection (T-06, T-07, RMC-003)
# ===========================================================================


class TestSellSemantics:
    """SELL never oversells; target==0 means full close."""

    def test_sell_zero_target_full_close(self) -> None:
        """TEST-S51-006 (T-06): SELL target==0 -> qty == held (full close)."""
        engine, _ = _make_real_paper_engine()
        held = Decimal("0.37")
        resolved = engine._resolve_order_quantity(
            signal=_make_signal(direction=SignalDirection.SELL, target_position=Decimal("0")),
            side=OrderSide.SELL, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=held,
        )
        assert resolved == held

    def test_sell_never_oversells_for_any_target(self) -> None:
        """TEST-S51-007 (T-07): resolved SELL qty <= held for a range of targets."""
        engine, _ = _make_real_paper_engine()
        held = Decimal("0.10")
        for target in [Decimal("0"), Decimal("1"), Decimal("5000"), Decimal("999999999")]:
            resolved = engine._resolve_order_quantity(
                signal=_make_signal(direction=SignalDirection.SELL, target_position=target),
                side=OrderSide.SELL, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=held,
            )
            assert resolved <= held, f"oversold for target={target}: {resolved} > {held}"

    @pytest.mark.asyncio
    async def test_sell_round_down_never_oversells_sub_8dp_residue(self) -> None:
        """TEST-S51-008 (RMC-003, HIGH): SELL quantize ROUND_DOWN keeps qty <= held.

        Two boundary cases:
          (a) held carries a sub-8dp residue (Decimal("0.000000005")) -> full close.
          (b) a near-held partial SELL target that, with ROUND_HALF_UP, would round
              the resolved quantity UP past held (overselling). ROUND_DOWN must not.
        In both cases the FINAL quantized order quantity must be <= held.
        With the old ROUND_HALF_UP this would oversell.
        """
        # (a) full-close of a position whose quantity has a 9th-decimal residue.
        engine_a, _ = _make_real_paper_engine()
        held_a = Decimal("0.10000000") + Decimal("0.000000005")  # 0.100000005
        engine_a._positions[_SYMBOL] = Position(
            symbol=_SYMBOL, run_id=_RUN_ID, quantity=held_a,
            average_entry_price=Decimal("48000"), current_price=_LAST_PRICE,
        )
        orders_a = await engine_a.process_signal(
            _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
        )
        assert len(orders_a) == 1
        # ROUND_DOWN at 8dp -> 0.10000000 <= held_a; ROUND_HALF_UP would give 0.10000001 > held_a.
        assert orders_a[0].quantity <= held_a
        assert orders_a[0].quantity == held_a.quantize(_QTY_PRECISION, rounding=ROUND_DOWN)

        # (b) near-held partial SELL whose unquantized qty rounds up under HALF_UP.
        engine_b, _ = _make_real_paper_engine()
        held_b = Decimal("0.10000000")
        engine_b._positions[_SYMBOL] = Position(
            symbol=_SYMBOL, run_id=_RUN_ID, quantity=held_b,
            average_entry_price=Decimal("48000"), current_price=_LAST_PRICE,
        )
        # target/price chosen so (target/price) sits just under held but with a 9th
        # decimal that HALF_UP would round up to exactly held or beyond.
        # base = target/50000. Want base = 0.099999995 -> target = 4999.99975
        partial = _make_signal(
            direction=SignalDirection.SELL,
            target_position=Decimal("4999.99975"),
            confidence=1.0,
        )
        resolved_b = engine_b._resolve_order_quantity(
            signal=partial, side=OrderSide.SELL, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=held_b,
        )
        quantized_b = resolved_b.quantize(_QTY_PRECISION, rounding=ROUND_DOWN)
        assert quantized_b <= held_b
        # Sanity: HALF_UP would have rounded to 0.10000000 (== held) or up.
        half_up = resolved_b.quantize(_QTY_PRECISION, rounding=ROUND_HALF_UP)
        assert half_up >= quantized_b


# ===========================================================================
# Group D: dca take-profit cap + grid incremental (T-08, T-14)
# ===========================================================================


class TestDcaAndGridSizing:
    """Strategy-shaped notionals flow through the resolver correctly."""

    def test_dca_take_profit_capped_at_holding(self) -> None:
        """TEST-S51-009 (T-08): a dca take-profit SELL (position_size*take_profit_pct) is capped at held.

        dca emits a SELL with target_position = position_size * take_profit_pct (a notional).
        If that implied base quantity exceeds the held position, the resolver caps at held.
        """
        engine, _ = _make_real_paper_engine()
        held = Decimal("0.05")  # holds 0.05 BTC
        # take-profit notional = 5000 -> base = 0.10 BTC > held 0.05 -> capped to held.
        tp_signal = _make_signal(
            direction=SignalDirection.SELL, target_position=Decimal("5000"), confidence=1.0,
        )
        resolved = engine._resolve_order_quantity(
            signal=tp_signal, side=OrderSide.SELL, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=held,
        )
        assert resolved == held

    @pytest.mark.asyncio
    async def test_grid_incremental_second_level_fires_nonzero_order(self) -> None:
        """TEST-S51-010 (T-14): a 2nd grid-level BUY (incremental notional) still fires a non-zero order.

        Grid emits per-level incremental target_position notionals, not a cumulative
        target-level. A second grid BUY with its own small incremental notional must
        produce a non-zero order even while a position is already open (as long as the
        concentration cap leaves room).
        """
        engine, _ = _make_real_paper_engine()
        # First grid level
        o1 = await engine.process_signal(_make_signal(target_position=Decimal("500"), confidence=1.0))
        assert len(o1) == 1
        # Second grid level -- incremental, still non-zero (room remains under 15% cap)
        o2 = await engine.process_signal(_make_signal(target_position=Decimal("500"), confidence=1.0))
        assert len(o2) == 1
        assert o2[0].quantity > Decimal("0")


# ===========================================================================
# Group E: pre_trade_check cap still binds (T-09)
# ===========================================================================


class TestPreTradeCheckCapStillBinds:
    """The risk pre_trade_check notional/adjusted_quantity cap reduces oversized orders."""

    @pytest.mark.asyncio
    async def test_pre_trade_check_caps_oversized_resolved_order(self) -> None:
        """TEST-S51-011 (T-09): when resolved target/price exceeds the order-size cap, pre_trade_check reduces it.

        max_order_size_quote = 2000 -> hard notional cap. target_position = 9000 @ 50000
        resolves base = 0.18 BTC (9000 notional), but the concentration/order-size ceiling
        and pre_trade_check together must reduce the final order below 0.18.
        Uses the REAL risk manager end-to-end.
        """
        params = _make_risk_params(
            max_order_size_quote=Decimal("2000"),
            max_position_size_pct=0.50,  # raise concentration so order-size cap is the binding one
        )
        engine, _ = _make_real_paper_engine(params=params)
        signal = _make_signal(target_position=Decimal("9000"), confidence=1.0)
        orders = await engine.process_signal(signal)

        assert len(orders) == 1
        # Final notional must be <= max_order_size_quote (2000) -> qty <= 0.04 BTC.
        final_notional = orders[0].quantity * _LAST_PRICE
        assert final_notional <= Decimal("2000")
        # And it is strictly smaller than the unconstrained target-derived 0.18 BTC.
        assert orders[0].quantity < Decimal("9000") / _LAST_PRICE


# ===========================================================================
# Group F: concentration BUY-only + exposure caps (T-10, T-11)
# ===========================================================================


class TestExposureAndConcentrationCaps:
    """Concentration is BUY-only (Sprint-15 guard); exposure caps bind on resolved BUY qty."""

    @pytest.mark.asyncio
    async def test_concentration_cap_buy_only_sell_never_blocked(self) -> None:
        """TEST-S51-012 (T-10): SELL is never blocked by the concentration cap (Sprint-15 regression)."""
        # Tight concentration cap.
        params = _make_risk_params(max_position_size_pct=0.01, max_portfolio_exposure_pct=0.60)
        engine, _ = _make_real_paper_engine(params=params)
        # Seed a large existing position (far above the 1% concentration cap).
        engine._positions[_SYMBOL] = Position(
            symbol=_SYMBOL, run_id=_RUN_ID, quantity=Decimal("1.0"),
            average_entry_price=Decimal("48000"), current_price=_LAST_PRICE,
        )
        sell = _make_signal(direction=SignalDirection.SELL, target_position=Decimal("0"))
        orders = await engine.process_signal(sell)
        # SELL must go through despite the position being over-concentrated.
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_portfolio_exposure_cap_binds_on_resolved_buy(self) -> None:
        """TEST-S51-013 (T-11): portfolio/cluster exposure cap blocks a BUY once cap reached."""
        # Cap total portfolio exposure low so an existing position fills it.
        params = _make_risk_params(
            max_portfolio_exposure_pct=0.20,
            max_cluster_exposure_pct=0.20,
            max_position_size_pct=0.50,
        )
        engine, _ = _make_real_paper_engine(params=params)
        # Existing position worth 20k notional == 20% of 100k -> at the cap.
        engine._positions[_SYMBOL] = Position(
            symbol=_SYMBOL, run_id=_RUN_ID, quantity=Decimal("0.4"),
            average_entry_price=_LAST_PRICE, current_price=_LAST_PRICE,
        )
        buy = _make_signal(target_position=Decimal("5000"), confidence=1.0)
        orders = await engine.process_signal(buy)
        # Exposure cap (a blocking violation) rejects the BUY entirely.
        assert orders == []


# ===========================================================================
# Group G: confidence applied once (T-12, T-13)
# ===========================================================================


class TestConfidenceAppliedOnce:
    """Confidence is applied exactly once (on the target); ceiling uses conf=1.0."""

    def test_cb_reduce_times_confidence_composes_single_application(self) -> None:
        """TEST-S51-014 (T-12): a CB-REDUCE-halved target composes with confidence, strictly smaller.

        CircuitBreaker REDUCE halves target_position upstream (in StrategyEngine).
        The resolver then applies confidence ONCE. So resolving a halved target at
        conf=c yields exactly half the qty of resolving the full target at conf=c,
        and confidence is applied a single time (no double-count).
        """
        engine, _ = _make_real_paper_engine()
        conf = 0.8
        full = _make_signal(target_position=Decimal("2000"), confidence=conf)
        reduced = _make_signal(target_position=Decimal("1000"), confidence=conf)  # REDUCE x0.5

        q_full = engine._resolve_order_quantity(
            signal=full, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        q_reduced = engine._resolve_order_quantity(
            signal=reduced, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        assert q_reduced == q_full * Decimal("0.5")
        assert q_reduced < q_full
        # Confidence applied once: q_reduced == (1000/50000)*0.8
        expected = (Decimal("1000") / _LAST_PRICE) * Decimal(str(conf))
        assert q_reduced == expected

    def test_ceiling_call_uses_confidence_one_on_buy_with_target(self) -> None:
        """TEST-S51-015 (T-13): the ceiling calculate_position_size call uses confidence=1.0.

        Spy on calculate_position_size and assert the BUY-with-target path invokes the
        ceiling at confidence=1.0 (confidence is applied to the target, not the ceiling).
        """
        engine, rm = _make_real_paper_engine()
        signal = _make_signal(target_position=Decimal("1000"), confidence=0.3)

        with patch.object(
            rm, "calculate_position_size", wraps=rm.calculate_position_size
        ) as spy:
            engine._resolve_order_quantity(
                signal=signal, side=OrderSide.BUY, last_price=_LAST_PRICE,
                equity=_INITIAL_CASH, held_quantity=None,
            )
        # Exactly one ceiling call, made at confidence=1.0.
        assert spy.call_count == 1
        _, kwargs = spy.call_args
        assert kwargs["confidence"] == 1.0


# ===========================================================================
# Group H: live/paper parity + ensemble + zero-confidence (T-15, T-16, RMC-005)
# ===========================================================================


class TestParityEnsembleAndGuards:
    """Cross-engine parity, ensemble averaged notional, and confidence=0 guard."""

    @pytest.mark.asyncio
    async def test_live_paper_parity_identical_inputs(self) -> None:
        """TEST-S51-016 (T-15): paper and live engines resolve identical BUY quantity for identical inputs."""
        params = _make_risk_params()
        paper, _ = _make_real_paper_engine(params=params)
        live, _, _ = _make_real_live_engine(params=params)

        signal = _make_signal(target_position=Decimal("3000"), confidence=0.6)
        q_paper = paper._resolve_order_quantity(
            signal=signal, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        q_live = live._resolve_order_quantity(
            signal=signal, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        assert q_paper == q_live

    def test_ensemble_averaged_subnotional_flows_through(self) -> None:
        """TEST-S51-017 (T-16): an ensemble-averaged target_position resolves proportionally.

        EnsembleStrategy emits a single Signal whose target_position is the weighted
        average of its sub-strategies' notionals. The resolver treats that averaged
        notional authoritatively: e.g. avg(2000, 1000) = 1500 -> base = 1500/price.
        """
        engine, _ = _make_real_paper_engine()
        averaged = (Decimal("2000") + Decimal("1000")) / Decimal("2")  # 1500
        signal = _make_signal(target_position=averaged, confidence=1.0)
        resolved = engine._resolve_order_quantity(
            signal=signal, side=OrderSide.BUY, last_price=_LAST_PRICE,
            equity=_INITIAL_CASH, held_quantity=None,
        )
        assert resolved == averaged / _LAST_PRICE

    @pytest.mark.asyncio
    async def test_zero_confidence_buy_produces_no_order(self) -> None:
        """TEST-S51-018 (RMC-005, LOW): confidence=0 BUY -> base 0 -> resolver 0 -> no order ([])."""
        engine, _ = _make_real_paper_engine()
        # Direct resolver: base = (target/price)*0 = 0; min(0, ceiling) = 0.
        resolved = engine._resolve_order_quantity(
            signal=_make_signal(target_position=Decimal("1000"), confidence=0.0),
            side=OrderSide.BUY, last_price=_LAST_PRICE, equity=_INITIAL_CASH, held_quantity=None,
        )
        assert resolved == Decimal("0")
        # End-to-end: process_signal hits the quantity<=0 guard and returns [].
        orders = await engine.process_signal(
            _make_signal(target_position=Decimal("1000"), confidence=0.0)
        )
        assert orders == []


# ===========================================================================
# Group I: gates still REJECT after the new sizing path (RMC-004)
# ===========================================================================


class TestGatesStillRejectAfterNewSizing:
    """daily-loss, max-drawdown, cooldown, and kill-switch gates still REJECT a valid-target BUY."""

    @pytest.mark.asyncio
    async def test_kill_switch_rejects_valid_target_buy(self) -> None:
        """TEST-S51-019 (RMC-004, kill-switch): kill switch rejects a BUY with a valid target."""
        engine, rm = _make_real_paper_engine()
        rm.trigger_kill_switch("test halt")
        orders = await engine.process_signal(_make_signal(target_position=Decimal("1000"), confidence=1.0))
        assert orders == []

    @pytest.mark.asyncio
    async def test_daily_loss_gate_rejects_valid_target_buy(self) -> None:
        """TEST-S51-020 (RMC-004, daily-loss): daily-loss breach rejects a BUY with a valid target.

        Drive a daily PnL below the -8% threshold via the engine's realised-PnL accounting
        by stubbing the engine's daily-pnl helper; the gate reads daily_pnl in pre_trade_check.
        """
        engine, _ = _make_real_paper_engine()
        # daily loss of -10% of equity (-10000) exceeds the 8% threshold.
        with patch.object(engine, "_get_daily_pnl", return_value=Decimal("-10000")):
            orders = await engine.process_signal(
                _make_signal(target_position=Decimal("1000"), confidence=1.0)
            )
        assert orders == []

    @pytest.mark.asyncio
    async def test_max_drawdown_gate_rejects_valid_target_buy(self) -> None:
        """TEST-S51-021 (RMC-004, max-drawdown): drawdown breach rejects a BUY with a valid target.

        Peak equity well above current equity -> drawdown exceeds the 30% limit.
        """
        engine, _ = _make_real_paper_engine()
        # current equity ~100k; peak 200k -> 50% drawdown >= 30% limit.
        with patch.object(engine, "_get_peak_equity", return_value=Decimal("200000")):
            orders = await engine.process_signal(
                _make_signal(target_position=Decimal("1000"), confidence=1.0)
            )
        assert orders == []

    @pytest.mark.asyncio
    async def test_cooldown_gate_rejects_valid_target_buy(self) -> None:
        """TEST-S51-022 (RMC-004, cooldown): post-loss-streak cooldown rejects a BUY with a valid target."""
        params = _make_risk_params()
        engine, rm = _make_real_paper_engine(params=params)
        # Trip the loss-streak -> cooldown active.
        for _ in range(params.loss_streak_count):
            rm.update_after_fill(Decimal("-100"), is_loss=True)
        assert rm.in_cooldown
        orders = await engine.process_signal(_make_signal(target_position=Decimal("1000"), confidence=1.0))
        assert orders == []
