"""
tests/unit/test_wp21h_fingerprint_log.py
------------------------------------------
WP2.1-H, WP21H-SEC-03 (CF-36): ``persist_backtest_results`` compares the fingerprint stored on
the run row with the one the result carries and logs an error on a mismatch.  The STORED value
came from the database (it may have been edited), so it is logged only when it is a well-formed
``[0-9a-f]{16}`` digest -- anything else is logged as the fixed ``"<malformed>"``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from structlog.testing import capture_logs

from api.services.run_persistence import _loggable_fingerprint, persist_backtest_results
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from trading.costs import DEFAULT_COST_PROFILE

_SENTINEL = "SENTINEL-" + "q" * 14  # 23 characters: inside the old ``[:32]`` slice
_RESULT_FP = "f" * 16
_EVENT = "runs.cost_profile_fingerprint_mismatch"


def _db() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


async def _persist(stored_fingerprint: object) -> list[dict[str, Any]]:
    """Run ``persist_backtest_results`` for a row whose stored fingerprint is
    ``stored_fingerprint``; return the captured log events."""
    profile = DEFAULT_COST_PROFILE.to_metadata()
    profile["fingerprint"] = stored_fingerprint
    run_orm = SimpleNamespace(config={"cost_profile": profile})
    result = MagicMock()
    result.trades = []
    result.equity_curve = []
    result.total_trades = 0
    result.total_return_pct = 0.0
    result.sharpe_ratio = 0.0
    result.cost_profile_fingerprint = _RESULT_FP
    with (
        patch("api.services.run_persistence.build_backtest_metrics") as metrics,
        patch("sqlalchemy.orm.attributes.flag_modified"),
        capture_logs() as logs,
    ):
        metrics.return_value.model_dump.return_value = {}
        await persist_backtest_results(
            _db(), uuid.uuid4(), run_orm, result, structlog.get_logger("wp21h-test")
        )
    return list(logs)


_MALFORMED = [
    pytest.param(_SENTINEL, id="sentinel"),
    pytest.param(_SENTINEL * 3, id="sentinel_long"),
    pytest.param("0123456789abcdef" + _SENTINEL, id="digest_plus_sentinel"),
    pytest.param("0123456789abcde", id="15_hex"),
    pytest.param("0123456789abcdef0", id="17_hex"),
    pytest.param("0123456789ABCDEF", id="uppercase"),
    pytest.param("0123456789abcdef\n", id="trailing_newline"),
    pytest.param("0123456789abcdef\r\nINJECTED " + _SENTINEL, id="newline_injection"),
    pytest.param("\x1b[31m" + _SENTINEL, id="ansi_escape"),
    pytest.param("", id="empty"),
    pytest.param(None, id="none"),
    pytest.param(1234567890123456, id="int"),
    pytest.param([_SENTINEL], id="list"),
    pytest.param({"k": _SENTINEL}, id="dict"),
]


class TestSec03TheStoredFingerprintIsOnlyLoggedWhenWellFormed:
    @pytest.mark.parametrize("stored", _MALFORMED)
    async def test_a_tampered_fingerprint_never_appears_in_the_log(self, stored: object) -> None:
        events = await _persist(stored)
        errors = [e for e in events if e["event"] == _EVENT]
        assert len(errors) == 1  # logged, did not raise
        assert errors[0]["config_fingerprint"] == "<malformed>"
        assert errors[0]["result_fingerprint"] == _RESULT_FP
        assert _SENTINEL not in repr(events)

    async def test_a_well_formed_fingerprint_is_logged_as_is(self) -> None:
        events = await _persist("0123456789abcdef")
        errors = [e for e in events if e["event"] == _EVENT]
        assert len(errors) == 1
        assert errors[0]["config_fingerprint"] == "0123456789abcdef"

    async def test_the_legitimate_default_fingerprint_is_logged_as_is(self) -> None:
        legit = DEFAULT_COST_PROFILE.fingerprint()
        events = await _persist(legit)
        assert [e["config_fingerprint"] for e in events if e["event"] == _EVENT] == [legit]

    async def test_a_matching_fingerprint_logs_nothing(self) -> None:
        events = await _persist(_RESULT_FP)
        assert [e for e in events if e["event"] == _EVENT] == []


class TestLoggableFingerprint:
    @pytest.mark.parametrize(
        "value", ["0" * 16, "abcdef0123456789", DEFAULT_COST_PROFILE.fingerprint()]
    )
    def test_well_formed_digests_pass_through(self, value: str) -> None:
        assert _loggable_fingerprint(value) == value

    @pytest.mark.parametrize("value", [v.values[0] for v in _MALFORMED])
    def test_everything_else_is_malformed(self, value: object) -> None:
        assert _loggable_fingerprint(value) == "<malformed>"
