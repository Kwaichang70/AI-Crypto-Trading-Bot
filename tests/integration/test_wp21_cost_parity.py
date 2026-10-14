"""
tests/integration/test_wp21_cost_parity.py
--------------------------------------------
WP2.1a parity (reports/vp2-wp2.1/synthesis-spec.md SY-21-09).

    WP21-T-14  fill-for-fill parity, for ``coinbase_advanced_1`` AND ``coinbase_advanced_d6``:
                 (a) backtest == paper EXACTLY, per fill (side, qty, price, fee);
                 (b) paper == FakeCCXT-live per fill within 1e-8 (symmetric slippage,
                     ``markets[*]["taker"/"maker"]`` = the profile, tie-free fixture);
                 all three stacks hold the same profile fingerprint and an equal exit
                 cost model; no drift warning is logged.
    WP21-T-16  live drift logs are LOG-ONLY: ``cost_profile.tier_mismatch`` at start and
               exactly one ``cost_profile.fill_fee_drift`` warning per (symbol, liquidity);
               fills, orders, sizing and exits are identical with and without the profile;
               synthesised fills and fee <= 0 are skipped; the logs carry numbers/ids only.

The live stack is the REAL ``LiveExecutionEngine`` + ``StrategyEngine`` over the in-memory
``FakeCCXTExchange`` (see ``live_harness``); "paper" is the real ``PaperExecutionEngine`` stack.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import ccxt
import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    build_live_stack,
    build_paper_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.costs import CostProfile, get_cost_profile
from trading.exit_config import ExitCostModel
from trading.models import Fill

SYMBOL = "BTC/EUR"
TF = "1h"
INITIAL = Decimal("10000")
#: BUY on the first post-warmup bar (close 100), SELL on the fifth (close 110).  Target 500 =>
#: quantity 5.00000000 in every mode (well below the 15% sizing ceiling), and the fills have no
#: rounding ties: 100 * 1.001 = 100.1, 110 * 0.999 = 109.89, fees 6.006 / 6.5934 (adv1).
CLOSES = [100, 101, 103, 106, 110, 110]
SCRIPT = {0: "buy", 4: "sell"}
TARGET = "500"
PROFILES = ["coinbase_advanced_1", "coinbase_advanced_d6"]

_FillKey = tuple[str, Decimal, Decimal, Decimal]


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


def _strategy() -> ScriptedSequenceStrategy:
    return ScriptedSequenceStrategy("wp21-parity", {"script": SCRIPT, "target_notional": TARGET})


def _keys(fills: list[Fill]) -> list[_FillKey]:
    return [(f.side.value, f.quantity, f.price, f.fee) for f in fills]


def _fake(profile: CostProfile | None, *, taker: Decimal | None = None) -> FakeCCXTExchange:
    """Fake whose fees / slippage / advertised market rates equal ``profile`` (or ``taker``)."""
    p = profile or get_cost_profile("coinbase_advanced_1")
    ex = FakeCCXTExchange(
        taker_fee_pct=taker if taker is not None else p.fee_rate("taker"),
        maker_fee_pct=p.fee_rate("maker"),
        advertise_market_fees=True,
    )
    ex.register_market(SYMBOL, base="BTC", quote="EUR")
    ex.set_balance("EUR", Decimal("1000000"))
    ex.seed_flat_bars(SYMBOL, count=100, price=Decimal(CLOSES[0]), timeframe=TF)
    ex.set_fill_slippage_pct(SYMBOL, p.slippage_rate, symmetric=True)
    return ex


async def _backtest(profile: CostProfile) -> tuple[list[_FillKey], Any]:
    runner = BacktestRunner(
        strategies=[_strategy()],
        symbols=[SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        seed=11,
        cost_profile=profile,
    )
    bars = scripted_path(CLOSES, first_open=datetime(2026, 3, 10, 10, 0, tzinfo=UTC), symbol=SYMBOL)
    await runner.run({SYMBOL: bars})
    assert runner.last_execution_engine is not None
    # WP21A-CR-07: the engine that actually ran carries the very profile object we passed
    assert runner.last_execution_engine.cost_profile is profile
    engine, *_ = runner._build_engine("parity-bt")
    return _keys(runner.last_execution_engine.get_all_fills()), engine


async def _paper(
    monkeypatch: pytest.MonkeyPatch, profile: CostProfile
) -> tuple[list[_FillKey], Any]:
    ex = _fake(profile)
    patch_exchange_factory(monkeypatch, ex)
    stack = await build_paper_stack(
        exchange=ex,
        strategy=_strategy(),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id="parity-paper",
        cost_profile=profile,
    )
    await start_and_warmup(stack, "parity-paper")
    for close in CLOSES[:5]:
        await step_bar(stack, SYMBOL, Decimal(close), timeframe=TF)
    await stack.engine.stop()
    assert stack.execution.cost_profile is profile
    return _keys(stack.execution.get_all_fills()), stack


async def _live(
    monkeypatch: pytest.MonkeyPatch,
    profile: CostProfile | None,
    *,
    exchange_profile: CostProfile | None = None,
    exchange_taker: Decimal | None = None,
) -> tuple[list[_FillKey], Any, list[dict[str, Any]]]:
    ex = _fake(exchange_profile or profile, taker=exchange_taker)
    patch_exchange_factory(monkeypatch, ex)
    stack = await build_live_stack(
        exchange=ex,
        strategy=_strategy(),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id="parity-live",
        cost_profile=profile,
    )
    with capture_logs() as logs:
        await start_and_warmup(stack, "parity-live")
        for close in CLOSES[:5]:
            await step_bar(stack, SYMBOL, Decimal(close), timeframe=TF)
    await stack.engine.stop()
    return _keys(stack.execution.get_all_fills()), stack, logs


def _drift_events(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in logs if str(e["event"]).startswith("cost_profile.")]


# ---------------------------------------------------------------------------
# WP21-T-14 -- parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_id", PROFILES)
async def test_t14_backtest_equals_paper_equals_live_fill_for_fill(
    profile_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = get_cost_profile(profile_id)
    bt_fills, bt_engine = await _backtest(profile)
    with monkeypatch.context() as mp:
        paper_fills, paper_stack = await _paper(mp, profile)
    with monkeypatch.context() as mp:
        live_fills, live_stack, logs = await _live(mp, profile)

    assert [f[0] for f in bt_fills] == ["buy", "sell"]
    # analytic anchor for the first leg: 5 units at 100 * (1 + slippage), taker fee on it
    slip = profile.slippage_rate
    buy_price = Decimal(100) * (Decimal(1) + slip)
    assert bt_fills[0][1] == Decimal(5)
    assert bt_fills[0][2] == buy_price
    assert bt_fills[0][3] == (Decimal(5) * buy_price * profile.fee_rate("taker"))

    # (a) backtest == paper EXACTLY
    assert bt_fills == paper_fills
    # (b) paper == FakeCCXT-live within 1e-8, per fill
    assert len(live_fills) == len(paper_fills) == 2
    tol = Decimal("0.00000001")
    for p, lv in zip(paper_fills, live_fills, strict=True):
        assert p[0] == lv[0]
        for a, b in zip(p[1:], lv[1:], strict=True):
            assert abs(a - b) <= tol, (p, lv)

    # one profile object, one fingerprint, one exit cost model across all three stacks
    fp = profile.fingerprint()
    assert bt_engine.cost_profile.fingerprint() == fp
    assert paper_stack.engine.cost_profile.fingerprint() == fp
    assert live_stack.engine.cost_profile.fingerprint() == fp
    assert live_stack.execution.cost_profile is profile
    expected_model = ExitCostModel.from_cost_profile(profile)
    assert bt_engine._exit_cost_model == paper_stack.engine._exit_cost_model == expected_model
    assert live_stack.engine._exit_cost_model == expected_model

    # live fees equal the profile's, so no drift warning exists
    assert _drift_events(logs) == []


# ---------------------------------------------------------------------------
# WP21-T-16 -- the live drift logs
# ---------------------------------------------------------------------------


async def test_t16_matching_exchange_logs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fills, _stack, logs = await _live(monkeypatch, get_cost_profile("coinbase_advanced_1"))
    assert _drift_events(logs) == []


async def test_t16_a_cheaper_tier_warns_at_start_and_once_per_symbol_and_liquidity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = get_cost_profile("coinbase_advanced_1")
    # the account is really on the 0.6% tier, the run's estimate says 1.2%
    fills, _stack, logs = await _live(monkeypatch, default, exchange_taker=Decimal("0.006"))
    assert len(fills) == 2
    events = _drift_events(logs)

    mismatches = [e for e in events if e["event"] == "cost_profile.tier_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["log_level"] == "warning"
    assert mismatches[0]["symbol"] == SYMBOL
    assert Decimal(mismatches[0]["market_taker_bps"]) == Decimal(60)
    assert Decimal(mismatches[0]["profile_taker_bps"]) == Decimal(120)
    assert mismatches[0]["profile_id"] == "coinbase_advanced_1"

    drift = [e for e in events if e["event"] == "cost_profile.fill_fee_drift"]
    # BUY and SELL are both taker fills of the same symbol: ONE warning, then DEBUG
    assert [e["log_level"] for e in drift] == ["warning", "debug"]
    assert {e["liquidity"] for e in drift} == {"taker"}
    assert Decimal(drift[0]["fee_bps"]) == Decimal("60.00")
    assert Decimal(drift[0]["profile_bps"]) == Decimal(120)

    # numbers / ids only (CF-36): no exception text, no raw dict
    own_fields = {
        "symbol",
        "liquidity",
        "profile_id",
        "fee_bps",
        "profile_bps",
        "market_taker_bps",
        "profile_taker_bps",
    }
    bound_context = {"event", "log_level", "engine", "exchange", "live_trading_enabled", "run_id"}
    for e in events:
        assert set(e) <= own_fields | bound_context, set(e) - own_fields - bound_context
        assert all(isinstance(v, str | int | float | bool) for v in e.values())


async def test_t16_the_logs_never_change_fills_orders_sizing_or_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = get_cost_profile("coinbase_advanced_1")
    with monkeypatch.context() as mp:
        with_profile = await _live(mp, default, exchange_taker=Decimal("0.006"))
    with monkeypatch.context() as mp:
        without_profile = await _live(
            mp, None, exchange_profile=default, exchange_taker=Decimal("0.006")
        )
    fills_a, stack_a, logs_a = with_profile
    fills_b, stack_b, logs_b = without_profile

    assert _drift_events(logs_a)  # the profile run did warn ...
    assert _drift_events(logs_b) == []  # ... the profile-less run cannot
    assert fills_a == fills_b

    def _orders(stack: Any) -> list[tuple[str, Decimal, Decimal]]:
        return [(o["side"], o["amount"], o["price"]) for o in stack.exchange.order_log]

    assert _orders(stack_a) == _orders(stack_b)
    assert stack_a.portfolio.current_equity == stack_b.portfolio.current_equity
    assert stack_a.portfolio.total_fees_paid == stack_b.portfolio.total_fees_paid
    assert stack_a.engine.bar_count == stack_b.engine.bar_count


class _RaisingMarkets(dict[str, Any]):
    """``exchange.markets`` whose lookup fails with a secret-looking message."""

    def get(self, key: Any, default: Any = None) -> Any:
        raise RuntimeError("SENTINEL-secret")


class _FeeRaisingFill:
    """Proxy of a real fill whose ``fee`` read fails; everything else is delegated."""

    def __init__(self, fill: Fill) -> None:
        self._fill = fill

    @property
    def fee(self) -> Decimal:
        raise RuntimeError("SENTINEL-secret")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fill, name)


@pytest.mark.parametrize("check", ["tier", "fill"])
async def test_t16_a_failing_check_traces_the_type_only_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    """WP21A-CR-08: the swallowed error leaves ONE debug trace (type only, CF-36)."""
    from trading.engines.live import LiveExecutionEngine

    default = get_cost_profile("coinbase_advanced_1")
    with monkeypatch.context() as mp:
        base_fills, base_stack, base_logs = await _live(mp, default)

    original_tier = LiveExecutionEngine._check_tier
    original_fill = LiveExecutionEngine._observe_fill_cost

    def _failing_tier(self: Any) -> None:
        real = self._exchange.markets
        self._exchange.markets = _RaisingMarkets(real)
        try:
            original_tier(self)
        finally:
            self._exchange.markets = real

    def _failing_fill(self: Any, fill: Fill) -> None:
        original_fill(self, _FeeRaisingFill(fill))  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        if check == "tier":
            mp.setattr(LiveExecutionEngine, "_check_tier", _failing_tier)
        else:
            mp.setattr(LiveExecutionEngine, "_observe_fill_cost", _failing_fill)
        fills, stack, logs = await _live(mp, default)

    assert [e for e in base_logs if e["event"] == "cost_profile.check_failed"] == []
    failed = [e for e in logs if e["event"] == "cost_profile.check_failed"]
    assert len(failed) == (1 if check == "tier" else len(fills))
    assert len(fills) == 2
    for event in failed:
        assert event["check"] == check
        assert event["error_type"] == "RuntimeError"
        assert event["log_level"] == "debug"
        assert "SENTINEL" not in str(event)
        assert "exc_info" not in event
    assert "SENTINEL" not in str(logs)

    # the engine carried on: fills, orders, equity and fees are exactly the baseline's
    assert fills == base_fills

    def _orders(stack_: Any) -> list[tuple[str, Decimal, Decimal]]:
        return [(o["side"], o["amount"], o["price"]) for o in stack_.exchange.order_log]

    assert _orders(stack) == _orders(base_stack)
    assert stack.portfolio.current_equity == base_stack.portfolio.current_equity
    assert stack.portfolio.total_fees_paid == base_stack.portfolio.total_fees_paid


def _fill(order_id: Any, *, fee: Decimal, maker: bool = False) -> Fill:
    return Fill(
        order_id=order_id,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=Decimal(1),
        price=Decimal(100),
        fee=fee,
        fee_currency="EUR",
        is_maker=maker,
    )


def _bare_engine(profile: CostProfile | None) -> Any:
    from trading.engines.live import LiveExecutionEngine

    exchange = MagicMock()
    exchange.id = "coinbase"
    exchange.markets = {}
    return LiveExecutionEngine(
        run_id="t16-bare",
        risk_manager=MagicMock(),
        exchange=exchange,
        enable_live_trading=True,
        cost_profile=profile,
    )


def test_t16_fee_zero_and_missing_profile_are_skipped() -> None:
    from uuid import uuid4

    engine = _bare_engine(get_cost_profile("coinbase_advanced_1"))
    with capture_logs() as logs:
        engine._observe_fill_cost(_fill(uuid4(), fee=Decimal(0)))  # fee <= 0: skipped
        engine._observe_fill_cost(_fill(uuid4(), fee=Decimal("1.2")))  # 120 bps: on profile
    assert _drift_events(logs) == []
    with capture_logs() as logs:
        _bare_engine(None)._observe_fill_cost(_fill(uuid4(), fee=Decimal(5)))  # no profile
    assert logs == []
    with capture_logs() as logs:
        engine._observe_fill_cost(_fill(uuid4(), fee=Decimal(5), maker=True))  # 500 bps vs 60
        engine._observe_fill_cost(_fill(uuid4(), fee=Decimal(5), maker=True))
        engine._observe_fill_cost(_fill(uuid4(), fee=Decimal(5)))  # taker: its own warning
    levels = [(e["liquidity"], e["log_level"]) for e in _drift_events(logs)]
    assert levels == [("maker", "warning"), ("maker", "debug"), ("taker", "warning")]


async def test_t16_synthesised_fills_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    default = get_cost_profile("coinbase_advanced_1")
    ex = _fake(default)
    patch_exchange_factory(monkeypatch, ex)
    stack = await build_live_stack(
        exchange=ex,
        strategy=_strategy(),
        symbol=SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=INITIAL,
        run_id="t16-synth",
        cost_profile=default,
    )
    await start_and_warmup(stack, "t16-synth")

    def _synth(order: Any) -> list[Fill]:
        # a synthesised fill with a drifting fee: 100 bps against the profile's 120 -> would warn
        return [
            Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=Decimal(100),
                fee=Decimal("5"),
                fee_currency="EUR",
            )
        ]

    stack.execution._synthesize_unrouted_fill = _synth  # type: ignore[method-assign]
    ex.queue_trades_error(SYMBOL, ccxt.NetworkError("trades unavailable"))
    with capture_logs() as logs:
        await step_bar(stack, SYMBOL, Decimal(CLOSES[0]), timeframe=TF)
    await stack.engine.stop()
    assert [o["side"] for o in ex.order_log] == ["buy"]
    assert _drift_events(logs) == []
