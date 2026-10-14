"""
tests/integration/fakes/cost_scenarios.py
------------------------------------------
WP2.1 (2.1a-0, CF-34): deterministic backtest scenarios A-D plus SMK-T-09 for the
cost-model goldens.

The module deliberately imports NOTHING cost-related.  ``run_scenario`` takes the
cost wiring as ``**cost_kwargs`` so the very same scenario runs

* on pristine HEAD with ``maker_fee_bps=10, taker_fee_bps=15, slippage_bps=5``
  (how ``LEGACY_DIGESTS`` were recorded), and
* after WP2.1 with ``cost_profile=<CostProfile>``.

No scenario passes custom ``risk_params`` (CF-34: a custom ``RiskParameters`` is
what hid the C25 rebuild in the old T-32 golden).

Scenarios
---------
A      strategy BUY + strategy SELL, no engine exit
B      fixed bracket (SL 2% / TP 4%), two round trips (== the WP1.3b T-30 reference)
C      trailing stop 3%
D      risk-only fallback sizing (BUY with ``target_position == 0``) + strategy SELL
SMK    SMK-T-09: the smoke round trip, flat price, EUR 65 capital, fixed SL 5%
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from common.models import OHLCVBar
from common.types import TimeFrame
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

__all__ = [
    "SCENARIO_NAMES",
    "digest",
    "run_scenario",
    "scenario_closes",
    "scenario_initial_capital",
]

BT_SYMBOL = "BTC/USDT"
_FIRST_OPEN = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _Scenario:
    closes: list[int]
    script: dict[int, str]
    bracket: dict[str, Any] | None = None
    trailing: float | None = None
    targets: dict[int, str] = field(default_factory=dict)


_SCENARIOS: dict[str, _Scenario] = {
    "A": _Scenario(
        closes=[100, 101, 102, 104, 105, 100, 99, 98, 97, 96, 100, 101],
        script={0: "buy", 6: "sell"},
    ),
    "B": _Scenario(
        closes=[100, 101, 102, 104, 105, 100, 99, 98, 97, 96, 100, 101],
        script={0: "buy", 5: "buy"},
        bracket={
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.02,
            "bracket_take_profit_pct": 0.04,
        },
    ),
    "C": _Scenario(
        closes=[100, 102, 105, 108, 110, 106, 104, 103, 102, 101, 100, 100],
        script={0: "buy"},
        trailing=0.03,
    ),
    "D": _Scenario(
        closes=[100, 101, 99, 100, 102, 103, 101, 100, 99, 100],
        script={0: "buy", 5: "sell"},
        targets={0: "0"},
    ),
}

SCENARIO_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "SMK")


def scenario_closes(name: str) -> list[Decimal]:
    """The post-warmup close path of a scenario (SMK: flat 100)."""
    if name == "SMK":
        return [Decimal("100.0")] * 150
    return [Decimal(c) for c in _SCENARIOS[name].closes]


def scenario_initial_capital(name: str) -> Decimal:
    return Decimal("65") if name == "SMK" else Decimal("10000")


def _payload(runner: BacktestRunner, result: Any) -> dict[str, Any]:
    assert runner.last_execution_engine is not None
    fills = runner.last_execution_engine.get_all_fills()
    return {
        "fills": [(f.side.value, str(f.quantity), str(f.price)) for f in fills],
        "fees": [str(f.fee) for f in fills],
        "expected": [str(f.expected_price) for f in fills],
        "curve": [str(p.equity) for p in result.equity_curve],
        "final_equity": str(result.final_equity),
    }


def digest(payload: dict[str, Any]) -> str:
    """sha256 over fills / curve / final equity (same shape as the T-30 digest)."""
    stable = {k: payload[k] for k in ("fills", "curve", "final_equity")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


async def _run_smoke(**cost_kwargs: Any) -> dict[str, Any]:
    strategy = SmokeRoundtripStrategy(
        strategy_id="smoke-bt",
        params={"notional_quote": 9.0, "hold_bars": 1, "exit_retry_bars": 4},
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=["XRP/EUR"],
        timeframe=TimeFrame.FIVE_MINUTES,
        initial_capital=Decimal("65"),
        seed=42,
        bracket_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
        **cost_kwargs,
    )
    base = datetime(2026, 1, 1, tzinfo=UTC)
    price = Decimal("100.0")
    bars = [
        OHLCVBar(
            symbol="XRP/EUR",
            timeframe=TimeFrame.FIVE_MINUTES,
            timestamp=base + timedelta(minutes=5 * i),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("100"),
        )
        for i in range(200)
    ]
    result = await runner.run({"XRP/EUR": bars})
    return _payload(runner, result)


async def run_scenario(name: str, **cost_kwargs: Any) -> dict[str, Any]:
    """Run one scenario; ``cost_kwargs`` are forwarded to ``BacktestRunner``."""
    if name == "SMK":
        return await _run_smoke(**cost_kwargs)
    sc = _SCENARIOS[name]
    strategy = ScriptedSequenceStrategy(
        f"wp21-{name.lower()}",
        {"script": sc.script, "target_notional": "3000", "targets": sc.targets},
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[BT_SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=sc.bracket,
        trailing_stop_pct=sc.trailing,
        allow_pyramiding=False,
        **cost_kwargs,
    )
    bars = scripted_path(
        sc.closes, first_open=_FIRST_OPEN, symbol=BT_SYMBOL, timeframe=TimeFrame.ONE_HOUR
    )
    result = await runner.run({BT_SYMBOL: bars})
    return _payload(runner, result)
