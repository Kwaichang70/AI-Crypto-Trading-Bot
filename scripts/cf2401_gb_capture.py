"""CF-24-01 gate G-B capture: the momentum_breakout paper and live decision path (T-24, SY-01-22a).

Drives the REAL paper and live stacks (``StrategyEngine`` + ``PaperExecutionEngine`` /
``LiveExecutionEngine`` + portfolio + risk manager + circuit breaker + ``CCXTMarketDataService``)
over ``FakeCCXTExchange`` through ``_poll_and_process``, one scripted bar at a time. Per
(mode x scenario) it digests the SY-01-02 decision path: strategy signals (metadata and
confidence included), the exchange order log, fills, open positions, the equity curve, the risk
snapshot and breaker state after every bar, skip records and trade rows. ``regime_at_entry`` is
NOT digested (UD-3 (a): it is ``None`` on new trades); it is reported next to the digests.

Deterministic: the wall clock (``trading.clock._wall_now``), ``asyncio.sleep`` and the risk-day
clock are pinned, every UUID is replaced by its first-seen ordinal, sets are sorted, nothing reads
the network. The canonicaliser fails closed: an unknown type raises ``TypeError``.
Python 3.11+, stdlib only besides the repo. It imports nothing that does not exist at bd93c27, so
the same file runs on the baseline (a scratch ``git worktree``, never a checkout of the main tree)
and on the new code:

    python scripts/cf2401_gb_capture.py digests.json    # structlog owns stdout

``capture(mode, scenario, before=..., per_bar=...)`` lets the identity test put the live-context
book into a state (empty, fresh, stale, raising, poller on, clock regressed) without the script
knowing about the book.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import hashlib
import json
import platform
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "packages", _ROOT / "apps", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange  # noqa: E402
from tests.integration.fakes.live_harness import (  # noqa: E402
    ManualClock,
    build_live_stack,
    build_paper_stack,
    patch_exchange_factory,
    start_and_warmup,
)

import trading.clock as clock_mod  # noqa: E402
from common.models import MultiTimeframeContext, OHLCVBar  # noqa: E402
from common.types import TimeFrame  # noqa: E402
from trading.models import Signal  # noqa: E402
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy  # noqa: E402

SYMBOL = "BTC/EUR"
START_PRICE = Decimal("50000")
BASE = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
MODES = ("paper", "live")
#: scenario -> (bar closes pushed one per step, engine bracket config)
SCENARIOS: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
    "two_breakouts": (  # BUY, held-skip, stop-loss, re-entry
        ("51000", "50500", "52000", "48000", "53000"),
        {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05, "bracket_take_profit_pct": 0.5},
    ),
    "take_profit": (  # BUY, then a take-profit exit
        ("51000", "52000", "54000", "56000"),
        {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05, "bracket_take_profit_pct": 0.05},
    ),
    "no_signal": (  # no new high: the engine runs, nothing trades
        ("50000", "49500", "49000", "49200"),
        {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05, "bracket_take_profit_pct": 0.5},
    ),
}
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DIGESTED_NOT = {"regime_at_entry"}  # UD-3 (a)


class WallClock:
    """The pinned wall clock; ``at`` advances one hour per bar."""

    def __init__(self) -> None:
        self.at = BASE


def _plain(obj: object) -> object:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return _plain(obj.model_dump(mode="python"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _plain(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items() if str(k) not in _DIGESTED_NOT}
    if isinstance(obj, set | frozenset):
        return sorted((_plain(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, float | int | str | bool) or obj is None:
        return obj
    raise TypeError(f"G-B: undigestable type {type(obj).__name__}")  # fail closed


def _canonical(frames: dict[str, Any]) -> str:
    text = json.dumps(_plain(frames), sort_keys=True, separators=(",", ":"))
    seen: dict[str, int] = {}
    return _UUID.sub(lambda m: f"<uuid:{seen.setdefault(m.group(0), len(seen))}>", text)


async def capture(
    mode: str,
    scenario: str,
    *,
    before: Callable[[WallClock], None] | None = None,
    per_bar: Callable[[WallClock], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """Run one (mode, scenario); return ``digest`` plus the ``regime_at_entry`` values and meta."""
    closes, config = SCENARIOS[scenario]
    wall = WallClock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clock_mod, "_wall_now", lambda: wall.at)

        async def _instant(delay: float = 0, result: object = None) -> object:
            return result

        mp.setattr(asyncio, "sleep", _instant)
        exchange = FakeCCXTExchange()
        exchange.register_market(SYMBOL, base="BTC", quote="EUR")
        exchange.set_balance("EUR", Decimal("100000"))
        exchange.seed_flat_bars(SYMBOL, count=100, price=START_PRICE, timeframe="1h")
        patch_exchange_factory(mp, exchange)
        strategy = MomentumBreakoutStrategy(
            "gb-momentum", {"lookback": 3, "trend_sma_period": None, "position_size": 100}
        )
        calls: list[dict[str, Any]] = []
        real_on_bar = strategy.on_bar

        def _on_bar(
            bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None
        ) -> list[Signal]:
            signals = real_on_bar(bars, mtf_context=mtf_context)
            calls.append(
                {
                    "n": len(bars),
                    "last": bars[-1].timestamp,
                    "close": bars[-1].close,
                    "ctx_is_none": mtf_context is None,
                    "signals": signals,
                }
            )
            return signals

        strategy.on_bar = _on_bar  # type: ignore[method-assign]
        build = build_paper_stack if mode == "paper" else build_live_stack
        stack = await build(
            exchange=exchange,
            strategy=strategy,
            symbol=SYMBOL,
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("1000"),
            run_id=f"gb-{mode}-{scenario}",
            engine_config=config,
            clock=ManualClock(BASE),
        )
        if before is not None:
            before(wall)
        await start_and_warmup(stack, f"gb-{mode}-{scenario}")
        per_bar_frames: list[dict[str, Any]] = []
        for close in closes:
            wall.at += timedelta(hours=1)
            if per_bar is not None:
                outcome = per_bar(wall)
                if outcome is not None:
                    await outcome
            exchange.push_bar(SYMBOL, Decimal(close), timeframe="1h")
            await stack.engine._poll_and_process()
            per_bar_frames.append(
                {
                    "risk": stack.portfolio.risk_snapshot(),
                    "equity": stack.portfolio.current_equity,
                    "breaker": stack.circuit_breaker.state if stack.circuit_breaker else None,
                    "positions": stack.portfolio.get_open_positions(),
                }
            )
        trades = stack.portfolio.get_trade_history()
        frames = {
            "calls": calls,
            "per_bar": per_bar_frames,
            "orders": exchange.order_log,
            "fills": stack.execution.get_all_fills(),
            "equity_curve": stack.portfolio.get_equity_curve(),
            "skips": stack.engine._skip_logger.get_all_skipped(),
            "trades": trades,
        }
        regimes = [getattr(t, "regime_at_entry", None) for t in trades]
        await stack.engine.stop()
    return {
        "digest": hashlib.sha256(_canonical(frames).encode()).hexdigest(),
        "regime_at_entry": regimes,
        "ctx_none": all(c["ctx_is_none"] for c in calls),
        "orders": len(exchange.order_log),
    }


def main() -> None:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "digests": {},
    }
    for mode in MODES:
        for scenario in SCENARIOS:
            result = asyncio.run(capture(mode, scenario))
            out["digests"][f"{mode}:{scenario}"] = result
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
