"""CF-97-59: the kill-switch tests must not leak process-global state into test_cf36_dsn.py.

Run in a child pytest, so the failing order (kill switch first) is exercised whatever order the
parent session uses. The child sees no ``DATABASE_URL`` and no extra pytest options.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["tests/unit/test_sprint50_cycle3_killswitch.py", "tests/unit/test_cf36_dsn.py"]


def _child_env() -> dict[str, str]:
    keep = {k: v for k, v in os.environ.items() if k not in {"DATABASE_URL", "PYTEST_ADDOPTS"}}
    return {**keep, "PYTHONHASHSEED": "0"}


def test_cf9759_the_kill_switch_file_can_run_before_the_dsn_file() -> None:
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-q",
        *FILES,
    ]
    done = subprocess.run(  # noqa: S603
        argv, cwd=ROOT, env=_child_env(), capture_output=True, text=True, check=False, timeout=120
    )
    assert done.returncode == 0, done.stdout[-1500:]


def test_cf9759_no_unit_test_writes_database_url_outside_monkeypatch() -> None:
    text = (ROOT / FILES[0]).read_text(encoding="utf-8")
    assert 'os.environ["DATABASE_URL"]' not in text
    assert "monkeypatch.setenv" in text
