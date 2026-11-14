"""CF-26-97 P1: results stay bit-identical while backtests yield and other tasks run.

T-97P1-04 (global RNG state after a run) and T-97P1-06 (A-97-01): run-id-filterable backtest
entries run together under ``asyncio.gather``; the resting-LIMIT entry and the four optimizer
entries (not filterable by run id) run alone.  Every run shares the loop with a hostile task
that, at every tick, reseeds and draws from the global RNGs, sets gauges for other labels,
does Decimal work at precision 5, checks that no ambient clock leaked into its own context,
and latches a registered fake paper engine through the kill-switch fan-out the emergency
route uses.  Every digest must equal ``BASE_DIGESTS`` and the fake engine must be latched.
No timing assertion anywhere.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from decimal import Decimal, localcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from api.services import kill_switch
from api.services.run_orchestrator import _RUN_ENGINES
from common.metrics import metrics
from tests._cf2697_fingerprint import capture, matrix, run_case
from tests.integration.test_cf2697_identity import BASE_DIGESTS
from trading.clock import WALL_CLOCK, ambient_clock

_GATHERED = ["ma_crossover-2s-2000-breaker", "ma_crossover-2s-300-s7", "sl_tp-1s-2000-s7"]
_ALONE = [
    "paper-resting-limit-orders",
    "optimizer-in-sample",
    "optimizer-walk-forward",
    "optimizer-tied-in-sample",
    "optimizer-tied-walk-forward",
]
_SHORT = [n for n, c in matrix().items() if "strat" in c and (c["n"] <= 300 or "risk" in c)]


class _FakeRisk:
    def __init__(self) -> None:
        self.latched = 0

    def trigger_kill_switch(self, reason: str) -> None:
        self.latched += 1


@pytest.fixture
async def hostile(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SimpleNamespace]:
    fake = SimpleNamespace(risk_manager=_FakeRisk())
    monkeypatch.setitem(_RUN_ENGINES, "cf2697-fake", fake)
    state = SimpleNamespace(errors=[], ticks=0, max_tasks=0, fake=fake)
    stop = asyncio.Event()

    async def loop() -> None:
        while not stop.is_set():
            try:
                random.seed(1)
                np.random.seed(1)
                draw = random.random() + np.random.rand()  # noqa: S311
                metrics.gauge("cf2697_hostile", draw, {"run_id": "other"})
                with localcontext() as ctx:
                    ctx.prec = 5
                    assert Decimal(1) / Decimal(7) == Decimal("0.14286")
                assert ambient_clock() is WALL_CLOCK
                kill_switch.latch_in_memory("cf2697")
                for engine in list(_RUN_ENGINES.values()):
                    engine.risk_manager.trigger_kill_switch(kill_switch.GLOBAL_REASON)
                state.max_tasks = max(state.max_tasks, len(asyncio.all_tasks()))
            except Exception as exc:  # recorded, asserted empty by the test
                state.errors.append(repr(exc))
            state.ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(loop())
    await asyncio.sleep(0)
    yield state
    stop.set()
    await task


async def test_gathered_backtests_under_a_hostile_task_equal_base(hostile: SimpleNamespace) -> None:
    before = len(asyncio.all_tasks())
    with capture() as cap:
        digests = await asyncio.gather(*(run_case(n, cap) for n in _GATHERED))
    assert dict(zip(_GATHERED, digests, strict=True)) == {n: BASE_DIGESTS[n] for n in _GATHERED}
    assert hostile.errors == [] and hostile.fake.risk_manager.latched > 0
    assert hostile.ticks >= 300  # the hostile task really interleaved with the bar loops
    assert hostile.max_tasks <= before + len(_GATHERED)  # the engines spawn no task
    assert len(asyncio.all_tasks()) == before


@pytest.mark.parametrize("name", _ALONE)
async def test_unfilterable_entries_alone_under_a_hostile_task_equal_base(
    name: str, hostile: SimpleNamespace
) -> None:
    with capture() as cap:
        assert await run_case(name, cap) == BASE_DIGESTS[name]
    assert hostile.errors == [] and hostile.fake.risk_manager.latched > 0


def _np_equal(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return bool(a[0] == b[0] and (a[1] == b[1]).all() and a[2:] == b[2:])


@pytest.mark.parametrize("name", _SHORT)
async def test_global_rng_state_after_a_run_is_the_freshly_seeded_state(name: str) -> None:
    saved = random.getstate(), np.random.get_state()
    try:
        with capture() as cap:
            await run_case(name, cap)
        got = random.getstate(), np.random.get_state()
        random.seed(matrix()[name]["seed"])
        np.random.seed(matrix()[name]["seed"])
        assert got[0] == random.getstate() and _np_equal(got[1], np.random.get_state())
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
