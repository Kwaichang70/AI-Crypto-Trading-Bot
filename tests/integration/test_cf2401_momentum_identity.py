"""CF-24-01 gate G-B (T-24, SY-01-22a): the momentum paper and live decision path is byte-identical.

The digests below were captured ONCE by ``scripts/cf2401_gb_capture.py`` at the baseline commit in
a scratch ``git worktree`` (never a checkout of the main tree) and are committed, so G-B can be
rerun at P4b, P5 and P9 without that worktree:

* commit ``bd93c27`` (no live context code; ``53b945c``, the P3 parent of this commit, gives the
  same six digests, because P3 is unwired);
* Python ``3.11.17``; script sha256
  ``adc9a16e91d2db95e104ff74b9d58ff0a937aab80ed69de49d82d86ffad020ae`` (the fail-closed
  canonicaliser: an unknown type raises, UUIDs are numbered by first sight, P4AS-E06);
* scenarios: ``two_breakouts`` (BUY, held-skip, stop-loss, re-entry), ``take_profit`` and
  ``no_signal``, each in ``paper`` and ``live`` mode (``FakeCCXTExchange``, no network).

On the new code every book state (empty, fresh then ageing, stale, raising, a poller refreshing
the book every bar, a regressed clock) must reproduce the baseline digest of its scenario, the
momentum strategy must receive no context at all (it consumes no field), and the only trade-row
difference allowed is ``regime_at_entry``, which is ``None`` (UD-3 (a)).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from scripts.cf2401_gb_capture import MODES, SCENARIOS, WallClock, _canonical, _plain, capture

import trading.mtf_signals as mtf
from data.live_context import LIVE_BOOK, LiveContextBook
from data.point_in_time import to_ms
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit.test_cf2401_live_context_engine import DAY, fresh_book
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cf2401_gb_capture.py"
SCRIPT_SHA256 = "adc9a16e91d2db95e104ff74b9d58ff0a937aab80ed69de49d82d86ffad020ae"
BASELINE = {  # "<mode>:<scenario>" -> sha256 at bd93c27
    "live:no_signal": "5c48a44348bc236d01e5b58430beea88a54380a80b4275e24c5024e5ca115394",
    "live:take_profit": "d1bbe5f3ee49bec51cc0bdd56e32f41dcc3f147b1b6998f300e59ae5e37b97fd",
    "live:two_breakouts": "9ed186db0a3eec8d722e2e51b3f44f47e297272d38ca863ddee3c1f004765825",
    "paper:no_signal": "d526fc376add9f686b273f4275a1aae726a4b00dd2578d08cbc71d098c16994c",
    "paper:take_profit": "100fc7a1b3a9d32503c690055de5d565fbbe000aba0067d87b01374dffe70ae6",
    "paper:two_breakouts": "93663a2fefd815a885fc310b6fdedf36b3967d8be8744233d2a242d4b33b56cf",
}


def _fresh(wall: WallClock) -> None:
    fresh_book(to_ms(wall.at))


def _stale(wall: WallClock) -> None:
    # 30 d: past every cap; the FRED row ages from its estimated availability (10 d still served)
    fresh_book(to_ms(wall.at) - 30 * DAY)
    assert mtf.LIVE_SIGNALS.values_at() == mtf.SignalValues()  # all stale (QR-P4A-02)


def _regressed(wall: WallClock) -> None:
    LIVE_BOOK.heartbeat(to_ms(wall.at) + 30 * DAY)  # a mark in the far future: every read neutral


def _poller(wall: WallClock) -> None:  # a fake poller: heartbeat and a fresh publish every bar
    LIVE_BOOK.heartbeat(to_ms(wall.at))
    fresh_book(to_ms(wall.at))


@contextmanager
def _raising() -> Iterator[MagicMock]:
    with patch.object(LiveContextBook, "view", side_effect=RuntimeError("boom")) as view:
        yield view


_STATES: dict[
    str, tuple[Callable[[WallClock], None] | None, Callable[[WallClock], None] | None, bool]
] = {
    "empty": (None, None, False),
    "fresh_then_ageing": (_fresh, None, False),
    "stale": (_stale, None, False),
    "poller_on": (None, _poller, False),
    "clock_regressed": (_regressed, None, False),
    "raising_book": (None, None, True),
}


def test_the_committed_baseline_is_complete_and_bound_to_the_script() -> None:
    assert sorted(BASELINE) == sorted(f"{m}:{s}" for m in MODES for s in SCENARIOS)
    assert all(len(v) == 64 for v in BASELINE.values())
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == SCRIPT_SHA256


def test_the_canonicaliser_fails_closed_and_numbers_uuids_by_first_sight() -> None:
    u1, u2 = UUID(int=1), UUID(int=2)
    assert _plain(date(2026, 3, 10)) == "2026-03-10"
    assert _plain(frozenset({"b", "a"})) == ["a", "b"]
    assert _plain(frozenset({8, 1})) == [1, 8]  # int sets iterate 8, 1: the sort is what orders
    text = _canonical({"a": u1, "b": u1, "c": u2})
    assert text.count("<uuid:0>") == 2 and text.count("<uuid:1>") == 1
    with pytest.raises(TypeError):  # never digested by its type name (QR-P4A-03)
        _plain(object())


@pytest.mark.parametrize("state", sorted(_STATES))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("mode", MODES)
async def test_momentum_reproduces_the_baseline_in_every_book_state(
    mode: str, scenario: str, state: str
) -> None:
    before, per_bar, raising = _STATES[state]
    bars = len(SCENARIOS[scenario][0])
    with _raising() if raising else nullcontext() as spy:
        result = await capture(mode, scenario, before=before, per_bar=per_bar)
    # the engine really read the book on every bar (momentum just ignored what it saw)
    assert spy is None or spy.call_count == bars
    assert LIVE_BOOK.regressed_reads == (bars if state == "clock_regressed" else 0)
    assert result["digest"] == BASELINE[f"{mode}:{scenario}"]
    assert result["ctx_none"] is True  # a non-consumer reads no field: it receives no context
    assert all(r is None for r in result["regime_at_entry"])  # UD-3 (a)


async def test_the_digest_is_sensitive_to_the_context_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a momentum that consumed every field, at stage ``fresh``, WOULD change the run."""
    monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")
    monkeypatch.setattr(MomentumBreakoutStrategy, "consumes_mtf_signals", mtf.ALL_MTF_SIGNAL_FIELDS)
    result = await capture("paper", "two_breakouts", before=_fresh)
    assert result["ctx_none"] is False
    assert result["digest"] != BASELINE["paper:two_breakouts"]
