"""CF-26-97 opt-in self-check plugin (SY-97-11): every ``get_max_drawdown`` call is compared with
the frozen base body.  Not active in a normal run.  Load it with::

    PYTHONPATH=. .venv/bin/python -m pytest -p tests._cf2697_selfcheck <test files>

Each call of ``PortfolioAccounting.get_max_drawdown`` during the session is re-computed by
``tests._cf2697_reference`` on the same object; the float (hex) or the exception type and args
must be equal.  The session fails on a mismatch or on zero checks, and prints its counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._cf2697_reference import reference_get_max_drawdown

_STATE: dict[str, Any] = {"checks": 0, "mismatches": []}


def _outcome(fn: Any, obj: Any) -> tuple[Any, ...]:
    try:
        value = fn(obj)
    except Exception as exc:
        return ("exc", type(exc), exc.args, exc)
    return ("ok", value.hex(), None, None)


def pytest_sessionstart(session: pytest.Session) -> None:
    from trading import portfolio

    root = Path(__file__).resolve().parents[1]
    if not Path(portfolio.__file__).resolve().is_relative_to(root):
        raise pytest.UsageError(f"selfcheck: trading imported from outside {root}")
    original = portfolio.PortfolioAccounting.get_max_drawdown

    def checked(self: Any) -> float:
        got, want = _outcome(original, self), _outcome(reference_get_max_drawdown, self)
        _STATE["checks"] += 1
        if got[:3] != want[:3]:
            _STATE["mismatches"].append((got[:3], want[:3]))
        if got[0] == "exc":
            raise got[3]
        return float.fromhex(got[1])

    portfolio.PortfolioAccounting.get_max_drawdown = checked  # type: ignore[method-assign]


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.write_line(
        f"cf2697 selfcheck: {_STATE['checks']} checks, {len(_STATE['mismatches'])} mismatches"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _STATE["checks"] == 0 or _STATE["mismatches"]:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
