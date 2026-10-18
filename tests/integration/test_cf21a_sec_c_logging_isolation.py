"""
tests/integration/test_cf21a_sec_c_logging_isolation.py
-------------------------------------------------------
CF-21a-SEC-C / WP2.1-H regression guard: a ``TestClient`` lifespan must not leave
global logging state behind for the next test.

The API lifespan runs ``common.logging.configure_logging`` and never undoes it.  Left
alone it poisons every later test in the process:

* ``_AccessQueryFilter`` stays on ``uvicorn.access`` (fails the CF-36 "no-access-log
  state is untouched" tests, ``test_t12a`` / ``test_t33b``);
* module-level structlog loggers cache a bound logger holding the processor list of
  that moment, while fixtures such as ``_structlog_unfiltered`` later restore a
  *different* list.  ``structlog.testing.capture_logs`` only mutates the configured
  list, so it then captures nothing, and absence assertions such as
  ``TestST20LoggingHygiene`` pass (or fail) vacuously.

The autouse ``_global_logging_state_isolated`` fixture in ``tests/conftest.py`` fixes
this for every test.

Why this file does not simply have a "dirty it" test followed by a "check it" test:
the original pair was order-dependent.  Under ``pytest-xdist`` (or when only the second
test is selected) the check passed vacuously, and in a combined unit + integration run
the named CF-36 / ST20 tests stayed green even with the fixture removed because of where
they happened to be scheduled.  Instead, the leaky-lifespan-then-check sequence runs in a
FRESH interpreter (``scenario_cf21a_sec_c_leaks.py``, a child ``pytest`` session) that
inherits this repository's ``tests/conftest.py``.  The outcome therefore depends only on
whether the fixture works, never on where or in which process this file is scheduled:

* without the fixture (or with either half of it removed) the child's checks fail, and
  so do the tests below, in isolation and in any combined order;
* a child that did not run (collection error, missing scenario file, a missing test)
  fails ``test_child_session_ran_every_scenario_test`` instead of passing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pytest
import structlog

import common.logging as clog
from tests._cf36_logging import isolated_logging_state

_REPO = Path(__file__).resolve().parents[2]
_SCENARIO = Path(__file__).with_name("scenario_cf21a_sec_c_leaks.py")

_EXPECTED = (
    "test_1_lifespan_installs_the_global_state",
    "test_2_global_state_is_back_to_the_baseline",
    "test_3_capture_logs_reaches_module_level_loggers",
)


@dataclass(frozen=True)
class ChildRun:
    returncode: int
    outcomes: dict[str, str]
    tail: str


def _outcome(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


@pytest.fixture(scope="module")
def child_run(tmp_path_factory: pytest.TempPathFactory) -> ChildRun:
    """Run the leaky-lifespan scenario once, in a fresh interpreter."""
    report = tmp_path_factory.mktemp("cf21a_sec_c") / "child.xml"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PYTEST_XDIST", "COV_CORE", "COVERAGE")) and k != "PYTEST_ADDOPTS"
    }
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own interpreter
        [
            sys.executable,
            "-m",
            "pytest",
            str(_SCENARIO),
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            f"--junitxml={report}",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    outcomes: dict[str, str] = {}
    if report.exists():
        for case in ET.parse(report).getroot().iter("testcase"):  # noqa: S314 -- our own file
            outcomes[str(case.get("name"))] = _outcome(case)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
    return ChildRun(proc.returncode, outcomes, tail)


class TestFixtureIsolatesLifespanLeaks:
    def test_child_session_ran_every_scenario_test(self, child_run: ChildRun) -> None:
        """Non-vacuity: the scenario really executed all of its steps, in a child process."""
        assert set(child_run.outcomes) == set(_EXPECTED), child_run.tail
        assert child_run.outcomes[_EXPECTED[0]] == "passed", child_run.tail

    def test_lifespan_leaks_are_removed_for_the_next_test(self, child_run: ChildRun) -> None:
        """uvicorn filters / handlers / levels, root logger and ``_LOGGING_CONFIGURED`` restored."""
        assert child_run.outcomes.get(_EXPECTED[1]) == "passed", child_run.tail

    def test_capture_logs_still_sees_module_level_loggers_after_a_lifespan(
        self, child_run: ChildRun
    ) -> None:
        """The ST20 mechanism: ``capture_logs`` must reach ``runs.logger`` after a lifespan."""
        assert child_run.outcomes.get(_EXPECTED[2]) == "passed", child_run.tail

    def test_child_session_exits_clean(self, child_run: ChildRun) -> None:
        assert child_run.returncode == 0, child_run.tail

    def test_lifespan_structlog_config_is_applied_uncached_in_the_test_process(self) -> None:
        """In-process half: ``configure_logging`` must not enable logger caching under test.

        Production keeps ``cache_logger_on_first_use=True``; this pins only the harness rule
        (RK-12) that the autouse fixture applies to the test process.
        """
        with isolated_logging_state():
            clog.configure_structlog(log_level="INFO", json_logs=True)
            assert structlog.get_config()["cache_logger_on_first_use"] is False
