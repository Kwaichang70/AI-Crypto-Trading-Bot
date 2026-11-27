"""CF-26-97 P3 opt-in self-check plugin: every ``PaperExecutionEngine.get_open_orders`` call is
compared with the frozen base filter.  Not active in a normal run.  Load it with::

    PYTHONPATH=. .venv/bin/python -m pytest -p tests._cf2697_selfcheck_orders <test files>

Each call is re-computed by ``reference_get_open_orders`` on the same engine; the result must hold
the same objects in the same order (identity), or raise the same exception type with the same
args.  The session fails on a mismatch or on zero checks, and prints its counts.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import pytest

_STATE: dict[str, Any] = {"checks": 0, "mismatches": []}


def _outcome(fn: Any, engine: Any) -> tuple[Any, ...]:
    try:
        result = fn(engine)
    except Exception as exc:
        return ("exc", type(exc), exc.args, exc)
    return ("ok", [id(o) for o in result], None, result)


def pytest_sessionstart(session: pytest.Session) -> None:
    from tests._cf2697_reference_orders import reference_get_open_orders
    from trading.engines import paper

    root = Path(__file__).resolve().parents[1]
    if not Path(paper.__file__).resolve().is_relative_to(root):
        raise pytest.UsageError(f"selfcheck: trading imported from outside {root}")
    original = paper.PaperExecutionEngine.get_open_orders

    @wraps(original)
    def checked(self: Any) -> Any:
        got, want = _outcome(original, self), _outcome(reference_get_open_orders, self)
        _STATE["checks"] += 1
        if got[:3] != want[:3]:
            _STATE["mismatches"].append((got[:3], want[:3]))
        if got[0] == "exc":
            raise got[3]
        return got[3]

    paper.PaperExecutionEngine.get_open_orders = checked  # type: ignore[method-assign]


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.write_line(
        f"cf2697 orders selfcheck: {_STATE['checks']} checks, "
        f"{len(_STATE['mismatches'])} mismatches"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _STATE["checks"] == 0 or _STATE["mismatches"]:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
