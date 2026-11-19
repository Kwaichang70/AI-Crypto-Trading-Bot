"""CF-97-02 (T-9702-02): the rows persisted for a backtest are identical to HEAD's, chunked or not.

``DIGESTS`` were recorded at b9b83de with HEAD's unchanged ``persist_backtest_results`` (the script
and its log are in ``reports/vp2-cf2702/``: ``t02-record.py``, ``t02-record.log``).  Every case runs
at the real chunk size and with ``_PERSIST_CHUNK_ROWS`` patched to 7 and to 1, so every table
crosses chunk boundaries and also ends on an exact multiple.  A row is the tuple of every mapped
column attribute; ids are normalised to first-seen
ordinals (one ``Canon`` for the whole run, so the fill-to-order links stay checked).  Never record
these from patched code.
"""

# ruff: noqa: E501  (the sha256 literals exceed the line length)

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import pytest
import structlog
from sqlalchemy import inspect as sa_inspect

from api.db.models import RunORM
from api.services import run_persistence as rp
from common.types import TimeFrame
from tests._cf2697_fingerprint import _STRATS, Canon, make_bars, matrix
from trading.backtest import BacktestRunner
from trading.risk import RiskParameters

DIGESTS: dict[str, str] = {  # recorded at b9b83de, see the module docstring
    "breakout-1s-2000-s7": "6704d04e612e3c1d1595018189f73e94d88b55e258a792270d2018b93f021b61",
    "dca_rsi-1s-2000-s7": "cd31f433b327e99596eddb519edd2cb90bba8cc239468396a732ef5b1530e338",
    "dca_rsi-2s-2000-s7": "15e750bf97bcbfde382c2edc58b3556763c5b6b38706b1e43e5516ab311e2be0",
    "grid-2s-2000-s7": "465bb8b4f54877519f1149550e2b18bbf613a4ae02914b7d1303d0e3e3927983",
    "ma_crossover-1s-2000-atr-trailing": "0acc35761de74b3e8a26518445dbc53f1f06b0c5f978679806ccbcaec9a994e0",
    "ma_crossover-1s-2000-s7": "1a82e9b8235478da3d62359379572561ea58aac9727a1160bbc055ef16ed8b4c",
    "ma_crossover-1s-2000-up2000": "a4c62e15fe7f14e12364840bf98c93b4b77d44d8416d2552e005c83dbbe7cdab",
    "ma_crossover-2s-2000-breaker": "f53e0fe7a4ce1e64d9a653b345bbec50516bb08c092172e3ddc303bab1c7bd7b",
    "ma_crossover-2s-2000-s7": "a363cdbfdccd4e2d89760eb7cf6f50b3c2ffd4ba1ddd64a9e1209ad1ee341497",
    "rsi-1s-300-s7": "8ba208c1caf6c45c81589e2f02355fbbbf1b62ab1a2278d405e3cd6d75915442",
    "sl_tp-1s-2000-s7": "2395b7cd7ee403f7894225df358a85ae2f1276ba23943d1833ef861d0cb20c1c",
}


class _Rows:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def add_all(self, items: Any) -> None:
        self.rows.extend(items)

    async def flush(self) -> None:
        return None


async def run_case(name: str) -> tuple[Any, Any, Any]:
    cfg = matrix()[name]
    symbols = ["BTC/USDT", "ETH/USDT"][: cfg["nsym"]]
    seed0, drift = cfg.get("bar_seed", 7), cfg.get("drift", 0)
    bars = {s: make_bars(s, cfg["n"], seed0 + 101 * i, drift) for i, s in enumerate(symbols)}
    cls, params = _STRATS[cfg["strat"]]
    runner = BacktestRunner(
        strategies=[cls(strategy_id=f"fp-{cfg['strat']}", params=dict(params))],
        symbols=symbols, timeframe=TimeFrame.ONE_MINUTE, seed=cfg["seed"],
        bracket_config=cfg.get("bracket"), trailing_stop_pct=cfg.get("trailing"),
        risk_params=RiskParameters(**cfg["risk"]) if "risk" in cfg else None,
    )  # fmt: skip
    result = await runner.run(bars, oos_start_index=cfg.get("oos"))
    return result, runner.last_execution_engine, runner.last_portfolio


async def digest(ran: tuple[Any, Any, Any], chunk: int | None = None) -> str:
    result, ex, pf = ran
    sink, run_id, canon = _Rows(), uuid.UUID(int=1), Canon()
    run_orm = RunORM(id=run_id, config={})
    saved = getattr(rp, "_PERSIST_CHUNK_ROWS", None)
    if chunk is not None:
        rp._PERSIST_CHUNK_ROWS = chunk  # type: ignore[attr-defined]
    try:
        await rp.persist_backtest_results(
            sink, run_id, run_orm, result, structlog.get_logger("t"), ex, pf
        )  # type: ignore[arg-type]
    finally:
        if chunk is not None:
            rp._PERSIST_CHUNK_ROWS = saved  # type: ignore[attr-defined]
    tables: dict[str, list[Any]] = {}
    for row in sink.rows:
        cols = sa_inspect(row).mapper.column_attrs
        tables.setdefault(type(row).__name__, []).append(
            [[a.key, canon(getattr(row, a.key))] for a in cols]
        )
    payload = {
        "tables": tables,
        "config": canon(run_orm.config),
        "n_closed_trades": run_orm.n_closed_trades,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


CASES = [
    "ma_crossover-1s-2000-s7", "ma_crossover-2s-2000-s7", "ma_crossover-1s-2000-up2000",
    "ma_crossover-1s-2000-atr-trailing", "ma_crossover-2s-2000-breaker", "breakout-1s-2000-s7",
    "grid-2s-2000-s7", "sl_tp-1s-2000-s7", "dca_rsi-1s-2000-s7", "dca_rsi-2s-2000-s7",
    "rsi-1s-300-s7",  # zero trades (QA-9702-06)
]  # fmt: skip
_RAN: dict[str, tuple[Any, Any, Any]] = {}


@pytest.mark.parametrize("chunk", [None, 7, 1])
@pytest.mark.parametrize("name", CASES)
async def test_persisted_rows_equal_heads(name: str, chunk: int | None) -> None:
    if name not in _RAN:
        _RAN[name] = await run_case(name)
    assert await digest(_RAN[name], chunk) == DIGESTS[name]
