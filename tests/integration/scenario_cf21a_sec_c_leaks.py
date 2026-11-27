"""
tests/integration/scenario_cf21a_sec_c_leaks.py
-----------------------------------------------
CHILD-SESSION SCENARIO for ``test_cf21a_sec_c_logging_isolation.py`` (CF-21a-SEC-C /
WP2.1-H).  This module is deliberately NOT named ``test_*``: the regular collection never
picks it up.  The regression file runs it in a FRESH ``python -m pytest`` process, so the
"leaky lifespan, then check" sequence is reproduced from a clean interpreter wherever (and
however, e.g. under xdist) the outer test is scheduled.

The tests below are ordered (a class runs in definition order) and share ``_STATE``:

1. ``test_1`` takes a baseline BEFORE the shared ``client_dev`` lifespan starts, runs the
   lifespan (``configure_logging``) and asserts it really installed the global state it is
   expected to leave behind (non-vacuity), then warms a module-level logger.
2. ``test_2`` asserts every piece of that state is back to the baseline.
3. ``test_3`` asserts ``structlog.testing.capture_logs`` still reaches module-level
   loggers (the ST20 mechanism).

``_structlog_unfiltered`` is imported because 15 real test modules use it, and its
config restore is what swaps the processor list under a cached logger.

Each check fails loudly (never passes vacuously) if its predecessor did not run first.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from structlog.testing import capture_logs

import common.logging as clog
from api.routers import runs as runs_module
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401 -- reproduces the list swap

_STATE: dict[str, Any] = {}

_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _snapshot() -> dict[str, Any]:
    root = logging.getLogger()
    return {
        "configured": clog._LOGGING_CONFIGURED,
        "root_level": root.level,
        # pytest swaps its own handlers per test phase; they are not part of the contract.
        "root_handlers": [
            h for h in root.handlers if not type(h).__module__.startswith("_pytest.")
        ],
        "uvicorn": {
            n: (
                list(logging.getLogger(n).handlers),
                logging.getLogger(n).propagate,
                logging.getLogger(n).level,
                list(logging.getLogger(n).filters),
            )
            for n in _UVICORN
        },
    }


@pytest.fixture()
def _baseline() -> Iterator[None]:
    """Requested BEFORE ``client_dev`` so it is recorded before the lifespan runs."""
    _STATE.clear()
    _STATE["before"] = _snapshot()
    yield


class TestLeakyLifespanThenCheck:
    def test_1_lifespan_installs_the_global_state(self, _baseline: None, client_dev: Any) -> None:
        access = logging.getLogger("uvicorn.access")
        error = logging.getLogger("uvicorn.error")
        assert any(isinstance(f, clog._AccessQueryFilter) for f in access.filters)
        assert any(isinstance(f, clog._WsQueryFilter) for f in error.filters)
        assert clog._LOGGING_CONFIGURED is True
        # First use of a module-level logger under the lifespan configuration: with
        # ``cache_logger_on_first_use=True`` it would cache against the lifespan's list.
        runs_module.logger.info("cf21a_sec_c.warm_cache")
        _STATE["leaked"] = True

    def test_2_global_state_is_back_to_the_baseline(self) -> None:
        assert _STATE.get("leaked") is True, "test_1 did not run first: refusing to pass vacuously"
        after = _snapshot()
        assert after == _STATE["before"]
        access = logging.getLogger("uvicorn.access")
        error = logging.getLogger("uvicorn.error")
        assert not any(isinstance(f, clog._AccessQueryFilter) for f in access.filters)
        assert not any(isinstance(f, clog._WsQueryFilter) for f in error.filters)
        assert clog._LOGGING_CONFIGURED is False

    def test_3_capture_logs_reaches_module_level_loggers(self) -> None:
        assert _STATE.get("leaked") is True, "test_1 did not run first: refusing to pass vacuously"
        with capture_logs() as logs:
            runs_module.logger.info("cf21a_sec_c.probe", key_prefix="abcd1234")
        probe = [e for e in logs if e["event"] == "cf21a_sec_c.probe"]
        assert len(probe) == 1 and probe[0]["key_prefix"] == "abcd1234"
