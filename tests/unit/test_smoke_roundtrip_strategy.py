"""
tests/unit/test_smoke_roundtrip_strategy.py
----------------------------------------------
WP-SMOKE (reports/vp2-smoke/synthesis-spec.md section 9) -- unit + backtest
coverage for ``packages/trading/strategies/smoke_roundtrip.py``.

SMK-T-01 .. SMK-T-05: pure strategy-level coverage (no engine).
SMK-T-09 .. SMK-T-10: full backtest determinism / round-trip-identity
coverage, driven through the real ``BacktestRunner`` (no shortcuts).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from common.models import OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.backtest import BacktestRunner
from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

SYMBOL = "XRP/EUR"


def _bar(close: float, ts: datetime, symbol: str = SYMBOL) -> OHLCVBar:
    c = Decimal(str(close))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.FIVE_MINUTES,
        timestamp=ts,
        open=c,
        high=c,
        low=c,
        close=c,
        volume=Decimal("100"),
    )


def _series(count: int, price: float = 100.0, symbol: str = SYMBOL) -> list[OHLCVBar]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [_bar(price, base + timedelta(minutes=5 * i), symbol) for i in range(count)]


def _strategy(**params: object) -> SmokeRoundtripStrategy:
    return SmokeRoundtripStrategy(strategy_id="smoke-test", params=params)


# ===========================================================================
# SMK-T-01: exactly one BUY ever, on the very first on_bar call.
# ===========================================================================


class TestEntryLatch:
    def test_first_call_emits_exactly_one_buy(self) -> None:
        strat = _strategy(notional_quote=9.0)
        signals = strat.on_bar([_bar(100.0, datetime(2026, 1, 1, tzinfo=UTC))])
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == SignalDirection.BUY
        assert sig.target_position == Decimal("9.0")
        assert sig.confidence == 1.0
        assert sig.metadata == {"trigger": "smoke_roundtrip", "phase": "entry"}

    def test_no_second_buy_over_next_200_bars(self) -> None:
        strat = _strategy(notional_quote=9.0, hold_bars=1, exit_retry_bars=4)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        buys = 0
        for i in range(201):
            signals = strat.on_bar([_bar(100.0, base + timedelta(minutes=i))])
            buys += sum(1 for s in signals if s.direction == SignalDirection.BUY)
        assert buys == 1


# ===========================================================================
# SMK-T-02: bounded SELL window, then silence + one WARNING.
# ===========================================================================


class TestExitWindow:
    @pytest.mark.parametrize("hold_bars", [1, 6, 12])
    @pytest.mark.parametrize("exit_retry_bars", [1, 4])
    def test_sell_window_then_silence(self, hold_bars: int, exit_retry_bars: int) -> None:
        strat = _strategy(
            notional_quote=9.0, hold_bars=hold_bars, exit_retry_bars=exit_retry_bars
        )
        base = datetime(2026, 1, 1, tzinfo=UTC)
        total_bars = hold_bars + exit_retry_bars + 20

        with capture_logs() as cap:
            all_signals: list[list[SignalDirection]] = []
            for i in range(total_bars):
                signals = strat.on_bar([_bar(100.0, base + timedelta(minutes=i))])
                all_signals.append([s.direction for s in signals])

        # Bar 0: BUY. Bars [1 .. hold_bars-1]: nothing (HOLD).
        assert all_signals[0] == [SignalDirection.BUY]
        for i in range(1, hold_bars):
            assert all_signals[i] == []

        # Bars [hold_bars .. hold_bars+exit_retry_bars-1]: SELL target 0.
        sell_bars = range(hold_bars, hold_bars + exit_retry_bars)
        for i in sell_bars:
            assert all_signals[i] == [SignalDirection.SELL]

        # After the window: silence forever.
        for i in range(hold_bars + exit_retry_bars, total_bars):
            assert all_signals[i] == []

        closed_events = [e for e in cap if e.get("event") == "smoke_roundtrip.exit_window_closed"]
        assert len(closed_events) == 1, f"expected exactly one WARNING; got {cap!r}"
        assert closed_events[0]["log_level"] == "warning"

    def test_sell_signals_carry_signal_exit_metadata(self) -> None:
        strat = _strategy(notional_quote=9.0, hold_bars=1, exit_retry_bars=4)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        strat.on_bar([_bar(100.0, base)])  # entry
        sell = strat.on_bar([_bar(100.0, base + timedelta(minutes=1))])
        assert len(sell) == 1
        assert sell[0].direction == SignalDirection.SELL
        assert sell[0].target_position == Decimal("0")
        assert sell[0].metadata["exit_reason"] == "signal_exit"
        assert sell[0].metadata["attempt"] == 1


# ===========================================================================
# SMK-T-03: schema + Pydantic bounds, additionalProperties: false, no
# exclusiveMinimum.
# ===========================================================================


class TestParamsSchema:
    def test_schema_forbids_additional_properties_and_exclusive_bounds(self) -> None:
        schema = SmokeRoundtripStrategy.parameter_schema()
        assert schema.get("additionalProperties") is False
        for prop_schema in schema.get("properties", {}).values():
            assert "exclusiveMinimum" not in prop_schema
            assert "exclusiveMaximum" not in prop_schema

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("notional_quote", 4.99),
            ("notional_quote", 9.51),
            ("notional_quote", "nine"),
            ("hold_bars", 0),
            ("hold_bars", 13),
            ("exit_retry_bars", 0),
            ("exit_retry_bars", 6),
            ("not_a_real_param", 1),
        ],
    )
    def test_invalid_params_rejected(self, key: str, value: object) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SmokeRoundtripStrategy(strategy_id="s", params={key: value})

    def test_defaults_are_valid(self) -> None:
        strat = SmokeRoundtripStrategy(strategy_id="s", params={})
        assert strat.params["notional_quote"] == 9.00
        assert strat.params["hold_bars"] == 1
        assert strat.params["exit_retry_bars"] == 4


# ===========================================================================
# SMK-T-04: declarations.
# ===========================================================================


class TestDeclarations:
    def test_requires_exit_manager_and_no_pyramiding(self) -> None:
        assert "requires_exit_manager" in SmokeRoundtripStrategy.__dict__
        assert "default_allow_pyramiding" in SmokeRoundtripStrategy.__dict__
        assert SmokeRoundtripStrategy.__dict__["requires_exit_manager"] is True
        assert SmokeRoundtripStrategy.__dict__["default_allow_pyramiding"] is False

    def test_min_bars_required_is_one(self) -> None:
        strat = _strategy()
        assert strat.min_bars_required == 1


# ===========================================================================
# SMK-T-05: determinism across instances; a fresh instance restarts IDLE.
# ===========================================================================


class TestDeterminism:
    def test_two_instances_identical_bars_emit_identical_signals(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        bars = [_bar(100.0 + (i % 3), base + timedelta(minutes=i)) for i in range(20)]

        strat_a = _strategy(notional_quote=9.0, hold_bars=2, exit_retry_bars=3)
        strat_b = _strategy(notional_quote=9.0, hold_bars=2, exit_retry_bars=3)

        def _run(strat: SmokeRoundtripStrategy) -> list[tuple[object, ...]]:
            out: list[tuple[object, ...]] = []
            for bar in bars:
                for sig in strat.on_bar([bar]):
                    out.append(
                        (sig.direction, sig.symbol, sig.target_position, sig.metadata)
                    )
            return out

        assert _run(strat_a) == _run(strat_b)

    def test_fresh_instance_restarts_at_idle(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        strat = _strategy(notional_quote=9.0, hold_bars=1, exit_retry_bars=1)
        # Run it to DONE.
        for i in range(5):
            strat.on_bar([_bar(100.0, base + timedelta(minutes=i))])

        # A brand-new instance (e.g. after a protective resume rebuild)
        # replays IDLE -> its first call is a BUY again.
        fresh = _strategy(notional_quote=9.0, hold_bars=1, exit_retry_bars=1)
        signals = fresh.on_bar([_bar(100.0, base)])
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY


# ===========================================================================
# SMK-T-09 / SMK-T-10: full backtest via the real BacktestRunner.
# ===========================================================================


def _synthetic_bracket() -> dict[str, object]:
    # D-SMK-3: mandatory fixed SL, default 0.05 -- flat prices below never
    # breach it, so every SELL in these two tests is the strategy's own
    # signal_exit, never the bracket stop_loss.
    return {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}


async def _run_backtest(
    *, hold_bars: int, exit_retry_bars: int, num_bars: int, seed: int = 42
) -> BacktestRunner:
    strategy = SmokeRoundtripStrategy(
        strategy_id="smoke-bt",
        params={"notional_quote": 9.0, "hold_bars": hold_bars, "exit_retry_bars": exit_retry_bars},
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[SYMBOL],
        timeframe=TimeFrame.FIVE_MINUTES,
        initial_capital=Decimal("65"),
        seed=seed,
        bracket_config=_synthetic_bracket(),
    )
    bars = _series(num_bars)
    await runner.run({SYMBOL: bars})
    return runner


class TestBacktestRoundtrip:
    async def test_smk_t_09_deterministic_single_roundtrip_with_exact_identity(self) -> None:
        runner_a = await _run_backtest(hold_bars=1, exit_retry_bars=4, num_bars=200)
        runner_b = await _run_backtest(hold_bars=1, exit_retry_bars=4, num_bars=200)

        trades_a = runner_a.last_portfolio.get_trade_history()
        trades_b = runner_b.last_portfolio.get_trade_history()

        assert len(trades_a) == 1, f"expected exactly one closed trade; got {trades_a!r}"
        assert len(trades_b) == 1

        trade_a, trade_b = trades_a[0], trades_b[0]
        # Determinism: byte-identical trade record across two independent
        # runs of the identical seed/config/bars.
        trade_a_key = (
            trade_a.entry_price, trade_a.exit_price, trade_a.quantity, trade_a.realised_pnl,
        )
        assert trade_a_key == (
            trade_b.entry_price,
            trade_b.exit_price,
            trade_b.quantity,
            trade_b.realised_pnl,
        )

        assert trade_a.exit_reason == "signal_exit"
        assert trade_a.strategy_id == "smoke-bt"

        # Exactly flat at the end.
        assert runner_a.last_portfolio.get_open_positions() == []

        # Identity: R = q*(p_s - p_b) - (f_b + f_s), exact to the cent.
        gross = trade_a.quantity * (trade_a.exit_price - trade_a.entry_price)
        expected_pnl = gross - trade_a.total_fees
        assert abs(expected_pnl - trade_a.realised_pnl) <= Decimal("0.01")

    async def test_smk_t_10_history_ends_before_hold_bars_stays_held(self) -> None:
        # hold_bars=6 but the backtest only has warmup(50) + 3 bars after
        # entry -- history ends mid-HOLD, well before the first SELL
        # attempt is due.
        runner = await _run_backtest(hold_bars=6, exit_retry_bars=4, num_bars=53)

        trades = runner.last_portfolio.get_trade_history()
        assert trades == [], f"expected no closed trade yet; got {trades!r}"

        open_positions = runner.last_portfolio.get_open_positions()
        assert len(open_positions) == 1, (
            f"expected exactly one open position; got {open_positions!r}"
        )
        position = open_positions[0]
        assert position.symbol == SYMBOL

        # No second BUY: the held position's quantity is consistent with a
        # single ~EUR 9 entry at ~EUR 100/unit, never a doubled-up size (the
        # strategy's own entry latch AND the engine's default no-pyramiding
        # held gate are both independent layers against a second BUY --
        # SMK-T-01 pins the strategy-level latch directly).
        assert Decimal("0") < position.quantity < Decimal("0.15")
