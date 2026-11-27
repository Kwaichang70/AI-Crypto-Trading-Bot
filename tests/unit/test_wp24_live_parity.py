"""WP24-T-15 (adapted for CF-24-01 P4a, SY-01-21d) -- the live source ignores ``decision_at``.

The frozen HEAD ``bbe7142`` oracle, the moved-block hash and the byte-for-byte parity with the
pre-WP2.4 provider reads were retired on purpose: they pinned the defect CF-24-01 fixes
(``reports/vp2-cf2401/synthesis-spec.md`` SY-01-11, SY-01-21d). What stays here: the decision
time of the live path is the wall clock, and a ``BaseException`` is never swallowed. The new
body and its pins live in ``test_cf2401_live_context_engine.py`` and ``test_wp24_ast_pins.py``.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import trading.mtf_signals as mtf
from common.types import RunMode
from data.live_context import LIVE_BOOK, LiveContextBook
from data.point_in_time import to_ms
from tests.unit.test_cf2401_live_context_engine import make_engine, random_book
from tests.unit.test_cf2401_live_context_engine import wall as wall  # the pinned-clock fixture

TS = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


@pytest.mark.usefixtures("wall")
@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
class TestT15LiveParity:
    def test_decision_at_is_ignored_on_the_live_path(
        self, mode: RunMode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mtf, "LIVE_CONTEXT_STAGE", "fresh")  # values, not the neutral stage
        engine = make_engine(mode)
        rng = random.Random(15)  # noqa: S311  # seed control, not security
        for _ in range(25):
            random_book(rng, to_ms(TS))
            without = engine._build_mtf_context(TS)
            with_stamp = engine._build_mtf_context(TS, decision_at=datetime(2001, 1, 1, tzinfo=UTC))
            assert with_stamp == without
        assert LIVE_BOOK.regressed_reads == 0


class TestT15SwallowAllIsPreserved:
    def test_a_base_exception_is_not_swallowed_by_either_implementation(self) -> None:
        engine = make_engine(RunMode.PAPER)
        with patch.object(LiveContextBook, "view", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                engine._build_mtf_context(TS)
