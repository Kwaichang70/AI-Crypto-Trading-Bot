"""
tests/integration/test_live_nav_sizing_harness.py
-----------------------------------------------------
WP1.4 (Verbeterplan v2 §4 row 1.4, D5) -- live equity = NAV; sizing basis
harness scenarios (`reports/vp2-wp1.4/synthesis-spec.md`, H1-H6).

Drives the same *real* production stack as
``tests/integration/test_live_protective_paths.py`` (real
``StrategyEngine``, real ``LiveExecutionEngine``, real
``PortfolioAccounting``, real ``DefaultRiskManager``) against an in-memory
``FakeCCXTExchange``, but with a deliberately large EXTERNAL EUR balance
(100,000) against a small ``initial_capital`` (1,000) -- the exact
configuration D5 exists for. Every scenario below would size or cap
against the wrong number under the pre-WP1.4 account-balance-equity
behaviour; each one demonstrates the fixed (run-NAV-based) behaviour
instead.

    H1  BUY target far above run capital -> notional capped by run capital,
        never the 100x-larger account balance.
    H2  Two BUYs at a relaxed concentration cap both fill (NAV-based
        sizing/drawdown stays sane; the pre-fix account-balance equity
        made the second look like a false drawdown breach).
    H3  BUY 900 twice on a 1,000 run -> the second is capped to the
        remaining RUN cash, not the 100,000 account balance.
    H4  A resumed run whose persisted peak hint is far above current NAV
        trips the drawdown gate on the next BUY.
    H5  A balance-fetch fault blocks a BUY outright, but a protective
        stop-loss exit still fires (SELLs never consult this cap).
    H6  A EUR 100 run on a EUR 10,000 account still caps notional at the
        run's own capital, buffered by the taker fee.

Runtime target: < 15s for the whole file (no real sleeps, no real network
I/O -- identical ``_fast_sleep`` pattern to test_live_protective_paths.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from uuid import uuid4

import ccxt.async_support as ccxt_async
import pytest
from structlog.testing import capture_logs

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import OrderSide, SignalDirection, TimeFrame
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_live_stack,
    build_resumed_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from tests.integration.fakes.scripted_strategy import ScriptedSignalStrategy
from trading.models import Fill, Signal
from trading.risk import RiskParameters
from trading.strategy import BaseStrategy, StrategyMetadata

# ---------------------------------------------------------------------------
# Fixed scenario constants -- deterministic, no randomness, no wall clock.
# ---------------------------------------------------------------------------

SYMBOL = "BTC/EUR"
BASE = "BTC"
QUOTE = "EUR"
TIMEFRAME = TimeFrame.ONE_HOUR
TIMEFRAME_STR = "1h"
PRICE = Decimal("100")

# The whole point of this file: a run's OWN capital is two orders of
# magnitude smaller than the account it trades out of.
INITIAL_CAPITAL = Decimal("1000")
EXTERNAL_EUR_BALANCE = Decimal("100000")


class _RepeatedBuyStrategy(BaseStrategy):
    """WP1.4 harness-only: one scripted BUY per entry in ``targets``
    (0-based ``on_bar`` call index -> target notional), HOLD otherwise.

    Distinct from ``ScriptedSignalStrategy`` (WP1.0), which fires exactly
    once per instance -- H2/H3 need two BUY signals on the same symbol on
    consecutive bars.
    """

    metadata = StrategyMetadata(
        name="wp14_repeated_buy_test_strategy",
        description="WP1.4 harness-only: a scripted sequence of BUY signals.",
        tags=["test-only"],
    )

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._targets: dict[int, Decimal] = {
            int(idx): Decimal(str(notional))
            for idx, notional in (self._params.get("targets") or {}).items()
        }
        self._call_count = -1

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        self._call_count += 1
        if not bars or self._call_count not in self._targets:
            return []
        symbol = bars[-1].symbol
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=SignalDirection.BUY,
                target_position=self._targets[self._call_count],
                confidence=1.0,
            )
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every real sleep in the live path (see
    test_live_protective_paths.py's identical fixture for the full
    rationale -- same call sites, same reason)."""

    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch) -> FakeCCXTExchange:
    """A FakeCCXTExchange whose EUR balance (100,000) vastly exceeds the
    run's own ``initial_capital`` (1,000) -- the D5 scenario itself."""
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE)
    ex.set_balance(QUOTE, EXTERNAL_EUR_BALANCE)
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TIMEFRAME_STR)
    patch_exchange_factory(monkeypatch, ex)
    return ex


async def _build_and_warm(
    exchange: FakeCCXTExchange,
    strategy: BaseStrategy,
    *,
    initial_capital: Decimal = INITIAL_CAPITAL,
    engine_config: dict[str, object] | None = None,
    run_id: str = "wp14-nav-sizing-test-run",
    risk_params: RiskParameters | None = None,
) -> LiveStack:
    stack = await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=initial_capital,
        run_id=run_id,
        engine_config=engine_config,
        risk_params=risk_params,
    )
    await start_and_warmup(stack, run_id)
    return stack


# ---------------------------------------------------------------------------
# H1: a BUY target far above run capital is capped by the run's OWN
# capital (15% concentration cap of 1,000 = 150), never by the 100,000
# account balance HEAD would have sized against.
# ---------------------------------------------------------------------------


async def test_h1_buy_target_capped_by_run_capital_not_account_balance(
    exchange: FakeCCXTExchange,
) -> None:
    strategy = ScriptedSignalStrategy(
        "h1-oversized-target",
        {"direction": "buy", "call_index": 0, "target_notional": "10000"},
    )
    stack = await _build_and_warm(exchange, strategy)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1, f"expected exactly one BUY order; got {exchange.order_log!r}"
    notional = buy_orders[0]["amount"] * buy_orders[0]["price"]
    # 15% of the RUN's 1,000 capital (default max_position_size_pct), not
    # 15% of the 100,000 account balance HEAD would have sized against.
    assert notional <= Decimal("150")

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# H2: two BUYs at a relaxed concentration cap (max_position_size_pct=0.6)
# both fill -- NAV/peak stay sane (run-scoped), so the second is not
# rejected on a false drawdown the way HEAD's account-balance equity would
# have caused.
# ---------------------------------------------------------------------------


async def test_h2_two_30pct_buys_both_fill_at_relaxed_cap(exchange: FakeCCXTExchange) -> None:
    strategy = _RepeatedBuyStrategy(
        "h2-two-30pct-buys", {"targets": {0: "300", 1: "300"}}
    )
    risk_params = RiskParameters(
        max_position_size_pct=0.6,
        max_portfolio_exposure_pct=0.7,
        max_cluster_exposure_pct=0.7,
    )
    stack = await _build_and_warm(exchange, strategy, risk_params=risk_params)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 2, f"expected both 30% BUYs to fill; got {exchange.order_log!r}"

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# H3: BUY 900 twice on a 1,000 run -- the second is capped to the
# remaining RUN cash (tens of EUR), never the 100,000 account balance.
# ---------------------------------------------------------------------------


async def test_h3_second_900_buy_capped_to_remaining_run_cash(exchange: FakeCCXTExchange) -> None:
    strategy = _RepeatedBuyStrategy("h3-buy-900-twice", {"targets": {0: "900", 1: "900"}})
    # Disable every OTHER sizing/exposure cap so the only thing that can
    # ever reduce the second BUY below its 900 target is the D5
    # affordability cap (run cash / free quote), not the risk manager's
    # own (unrelated) concentration or exposure gates.
    risk_params = RiskParameters(
        max_position_size_pct=1.0,
        max_portfolio_exposure_pct=1.0,
        max_cluster_exposure_pct=1.0,
        per_trade_risk_pct=0.05,
        max_order_size_quote=Decimal("100000"),
    )
    stack = await _build_and_warm(exchange, strategy, risk_params=risk_params)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    cash_after_first_buy = stack.portfolio.cash
    assert cash_after_first_buy < Decimal("150"), (
        "sanity check: the first 900 BUY should leave well under 150 EUR "
        f"of run cash; got {cash_after_first_buy}"
    )

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 2, f"expected both BUY attempts to submit; got {exchange.order_log!r}"
    second_notional = buy_orders[1]["amount"] * buy_orders[1]["price"]

    # Capped to (approximately) the remaining run cash, buffered by the
    # 1% default taker margin -- and nowhere near the scripted 900 target
    # or the 100,000 EUR sitting in the account.
    assert second_notional < Decimal("150")
    buf = stack.execution._taker_buffer(SYMBOL) + stack.execution._buy_cap_slippage_pct
    assert second_notional * (Decimal("1") + buf) <= cash_after_first_buy + Decimal("0.01")

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# H4: a resumed run whose persisted peak hint sits far above current NAV
# trips the drawdown gate on the very next BUY.
# ---------------------------------------------------------------------------


async def test_h4_resumed_peak_hint_trips_drawdown_gate(exchange: FakeCCXTExchange) -> None:
    run_id = "wp14-h4-resume-run"
    # A single historical BUY fill: 8 BTC @ 100 EUR, no fee -- cash 200,
    # position value 800, NAV exactly back at the 1,000 starting capital.
    # peak_equity_hint (1,500) is a persisted high from BEFORE this fill
    # history (WP1.8) -- (1500-1000)/1500 = 33.3% > the 30% default cap.
    fill = Fill(
        order_id=uuid4(),
        symbol=SYMBOL,
        side=OrderSide.BUY,
        quantity=Decimal("8"),
        price=PRICE,
        fee=Decimal("0"),
        fee_currency=QUOTE,
    )
    # The exchange must actually hold the 8 BTC the replayed fill credits
    # the run with -- otherwise sync_positions' own-vs-exchange mismatch
    # guard (a different, pre-existing WP1.1 check) blocks the BUY first
    # and this scenario would prove nothing about the drawdown gate.
    exchange.set_balance(BASE, Decimal("8"))
    strategy = ScriptedSignalStrategy(
        "h4-post-resume-buy",
        {"direction": "buy", "call_index": 0, "target_notional": "50"},
    )
    stack = await build_resumed_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL_CAPITAL,
        run_id=run_id,
        fills=[fill],
        peak_equity_hint=Decimal("1500"),
    )
    assert stack.portfolio.current_equity == Decimal("1000")
    assert stack.portfolio.get_peak_equity() == Decimal("1500")

    await start_and_warmup(stack, run_id)

    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert buy_orders == [], f"expected the drawdown gate to reject the BUY; got {buy_orders!r}"
    assert any(e.get("event") == "live.signal_rejected" for e in cap)

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# H5: a balance-fetch fault blocks a BUY outright; a protective stop-loss
# exit still fires despite the very same fault (SELLs never consult the
# D5 affordability cap).
# ---------------------------------------------------------------------------


async def test_h5_balance_fault_blocks_buy_with_no_position_yet(
    exchange: FakeCCXTExchange,
) -> None:
    strategy = ScriptedSignalStrategy("h5-buy-blocked", {
        "direction": "buy", "call_index": 0, "target_notional": "100",
    })
    stack = await _build_and_warm(exchange, strategy)
    # A single fault is retried transparently by ccxt_retry -- ``times=``
    # keeps it failing across every retry attempt
    # (_fetch_balance_cached: max_retries=2, i.e. 3 total calls) so the
    # fault survives and genuinely reaches the engine.
    exchange.queue_balance_error(ccxt_async.NetworkError("exchange unavailable"), times=5)

    with capture_logs() as cap:
        await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    assert exchange.order_log == [], f"expected no BUY; got {exchange.order_log!r}"
    assert any(e.get("event") == "live.buy_blocked_balance_unavailable" for e in cap)

    await stack.engine.stop()


async def test_h5_balance_fault_does_not_block_stop_loss_exit(
    exchange: FakeCCXTExchange,
) -> None:
    strategy = ScriptedSignalStrategy("h5-buy-then-stop-loss", {
        "direction": "buy", "call_index": 0, "target_notional": "100",
    })
    stack = await _build_and_warm(
        exchange, strategy, engine_config={"bracket_stop_loss_pct": 0.05}
    )

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat

    # The balance fetch fails on the very bar the stop-loss trips -- the
    # exit must still reach the exchange (I-5: no new guard touches SELLs).
    # ``times=`` survives every ccxt_retry attempt (see the comment in the
    # "no position yet" scenario above).
    exchange.queue_balance_error(ccxt_async.NetworkError("exchange unavailable"), times=5)
    breach_price = (PRICE * Decimal("0.80")).quantize(Decimal("0.01"))
    await step_bar(stack, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

    sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
    assert len(sell_orders) == 1, f"expected the stop-loss SELL to fire; got {exchange.order_log!r}"

    final_position = stack.portfolio.get_position(SYMBOL)
    assert final_position is None or final_position.is_flat

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# H6: a EUR 100 run on a EUR 10,000 account still caps notional at the
# run's own capital, buffered by the taker fee -- not the account.
# ---------------------------------------------------------------------------


async def test_h6_small_run_on_larger_account_caps_at_run_capital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = FakeCCXTExchange()
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE)
    ex.set_balance(QUOTE, Decimal("10000"))
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TIMEFRAME_STR)
    patch_exchange_factory(monkeypatch, ex)

    small_capital = Decimal("100")
    strategy = ScriptedSignalStrategy("h6-small-run", {
        "direction": "buy", "call_index": 0, "target_notional": "1000",
    })
    # Disable the concentration/exposure caps so the D5 affordability cap
    # (run cash, buffered by the taker fee) is the only thing left that
    # can bind -- proving it caps at the RUN's 100 EUR, not the account's
    # 10,000 EUR.
    risk_params = RiskParameters(
        max_position_size_pct=1.0,
        max_portfolio_exposure_pct=1.0,
        max_cluster_exposure_pct=1.0,
        per_trade_risk_pct=0.05,
        max_order_size_quote=Decimal("100000"),
    )
    stack = await _build_and_warm(
        ex, strategy, initial_capital=small_capital, risk_params=risk_params,
    )

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in ex.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1, f"expected the BUY to fill (capped); got {ex.order_log!r}"
    notional = buy_orders[0]["amount"] * buy_orders[0]["price"]
    # notional * (1 + buf) <= 100 (small tolerance for the amount-precision floor).
    buf = stack.execution._taker_buffer(SYMBOL) + stack.execution._buy_cap_slippage_pct
    assert Decimal(str(notional)) * (Decimal("1") + buf) <= small_capital + Decimal("0.5")

    await stack.engine.stop()


# ---------------------------------------------------------------------------
# Security round 2 (reports/vp2-wp1.4/security-report.md, WP14-S-02, probe
# PA): a market BUY that fills above the ticker's `last` (slippage) must
# never push run cash negative -- the affordability cap's slippage margin
# (buy_cap_slippage_pct, default 0.5%) exists exactly for this.
# ---------------------------------------------------------------------------


async def test_s02_slippage_within_buffer_keeps_run_cash_non_negative(
    exchange: FakeCCXTExchange,
) -> None:
    # 0.5% slippage: the actual fill price the fake exchange applies is
    # 0.5% above the ticker `last` the engine's affordability cap saw.
    exchange.set_fill_slippage_pct(SYMBOL, Decimal("0.005"))
    strategy = ScriptedSignalStrategy(
        "s02-slippage", {"direction": "buy", "call_index": 0, "target_notional": "1000"},
    )
    # Relax every OTHER cap so the D5 affordability cap (run cash, now
    # buffered by taker + slippage) is the thing actually being sized
    # against -- otherwise the default 15% concentration cap would size
    # far below the point where slippage could ever matter.
    risk_params = RiskParameters(
        max_position_size_pct=1.0,
        max_portfolio_exposure_pct=1.0,
        max_cluster_exposure_pct=1.0,
        per_trade_risk_pct=0.05,
        max_order_size_quote=Decimal("100000"),
    )
    stack = await _build_and_warm(exchange, strategy, risk_params=risk_params)

    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)

    buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
    assert len(buy_orders) == 1, f"expected the BUY to fill; got {exchange.order_log!r}"
    assert stack.portfolio.cash >= Decimal("0"), (
        f"run cash went negative despite the slippage margin: {stack.portfolio.cash}"
    )

    await stack.engine.stop()
