"""
tests/unit/test_cf36_hygiene_check.py
-------------------------------------
T-33: the deploy-time self-check ``python -m api.log_hygiene_check`` (AC-14).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import structlog

import api
import common
import common.logging as clog
from api import log_hygiene_check as check
from common import redaction as R
from tests._cf36_logging import isolated_logging_state

S = "CF36SENTINEL"
ENV = {
    "DATABASE_URL": f"postgresql+asyncpg://trading:{S}-DbPw-0123456789@db:5432/x",
    "ADMIN_API_KEY": f"{S}adminkey0123456789abcdef0123456789",
    "FRED_API_KEY": f"{S}-fred-key-0123456789",
    "TELEGRAM_BOT_TOKEN": f"123456789:AA{S}TelegramSecret_0123456789-xyz",
    "REDIS_URL": f"redis://:{S}-RedisPw-0123456@redis:6379/0",
    "EXCHANGE_API_SECRET": (
        f"-----BEGIN EC PRIVATE KEY-----\\nMHcCAQEEI{S}Line1Line1Line1AAAA\\nBBBBBBbcdefghij{S}Line2Line2Line2\\n"  # noqa: E501
        "-----END EC PRIVATE KEY-----"
    ),
    "WHALE_ALERT_API_KEY": "short",
}
_SETTINGS_ENV = (
    "DATABASE_URL",
    "ADMIN_API_KEY",
    "FRED_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "REDIS_URL",
    "EXCHANGE_API_SECRET",
    "WHALE_ALERT_API_KEY",
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_PASSPHRASE",
    "POSTGRES_PASSWORD",
    "LIVE_TRADING_CONFIRM_TOKEN",
    "API_KEY_HASH",
    "API_KEY_HASH_SECONDARY",
    "ENABLE_LIVE_TRADING",
    "LOG_LEVEL",
    "DEBUG",
    "REQUIRE_API_AUTH",
)


@pytest.fixture
def check_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def _run(capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    with isolated_logging_state():
        code = check.main()
    out, err = capsys.readouterr()
    return code, out, err


def test_t33_pass_with_fake_secrets(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(capsys)
    assert code == 0, out
    assert out.strip().startswith("CF36 log hygiene: PASS")
    assert len(out.strip().splitlines()) == 1 and err == ""
    assert "secrets_registered=6" in out and "warn=whale_alert_api_key" in out
    assert S not in out + err


def test_t33_the_check_leaves_global_state_as_it_found_it(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys as _sys

    before_stdout, before_stderr = _sys.stdout, _sys.stderr
    before_cfg = structlog.get_config()
    before_handlers = list(logging.getLogger().handlers)
    before_count = R.registered_secret_count()
    _run(capsys)
    assert _sys.stdout is before_stdout and _sys.stderr is before_stderr
    assert structlog.get_config()["processors"] == before_cfg["processors"]
    assert logging.getLogger().handlers == before_handlers
    assert R.registered_secret_count() == before_count


def test_t33_fail_when_the_redactor_is_identity(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check_env.setattr(R, "redact_event_dict", lambda _l, _m, ed: ed)
    code, out, err = _run(capsys)
    assert code == 1
    assert out.strip().startswith("CF36 log hygiene: FAIL leak_")
    assert S not in out + err


def test_t33_fail_when_a_probe_path_is_silent_non_vacuity(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check_env.setattr(check, "_emit_value_probes", lambda _v, _k, _c: None)
    check_env.setattr(check, "_emit_synthetic_probes", lambda _c: None)
    check_env.setattr(check, "_emit_short_probes", lambda _v, _c: None)
    # Nothing emitted -> nothing to leak AND nothing to count: expected counts are 0,
    # so make the counter claim emissions that never happened.
    real = check._Counter

    class Lying(real):  # type: ignore[valid-type, misc]
        def __init__(self) -> None:
            super().__init__()
            self.native = 5
            self.std = 5

    check_env.setattr(check, "_Counter", Lying)
    code, out, _err = _run(capsys)
    assert code == 1 and out.strip() == "CF36 log hygiene: FAIL non_vacuity"


def test_t33_fail_when_ccxt_debug_is_not_pinned(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import common.logging as clog

    original = clog.configure_logging

    def unpinned(*a: object, **k: object) -> None:
        original(*a, **k)  # type: ignore[arg-type]
        logging.getLogger("ccxt").setLevel(logging.DEBUG)

    check_env.setattr(clog, "configure_logging", unpinned)
    code, out, _err = _run(capsys)
    assert code == 1 and out.strip() == "CF36 log hygiene: FAIL ccxt_debug"


def test_t33_fail_when_uvicorn_handlers_are_not_absorbed(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import common.logging as clog

    original = clog.configure_logging

    def keep_uvicorn(*a: object, **k: object) -> None:
        original(*a, **k)  # type: ignore[arg-type]
        logging.getLogger("uvicorn").addHandler(logging.NullHandler())

    check_env.setattr(clog, "configure_logging", keep_uvicorn)
    code, out, _err = _run(capsys)
    assert code == 1 and out.strip() == "CF36 log hygiene: FAIL uvicorn_absorption"


def test_t33_fail_on_live_trading_with_debug_logging(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check_env.setenv("ENABLE_LIVE_TRADING", "true")
    check_env.setenv("LOG_LEVEL", "DEBUG")
    code, out, err = _run(capsys)
    assert code == 1 and out.strip() == "CF36 log hygiene: FAIL log_level_debug"
    assert S not in out + err
    # DEBUG alone (no live trading) is not a failure of this check
    check_env.setenv("ENABLE_LIVE_TRADING", "false")
    code, out, _err = _run(capsys)
    assert code == 0 and out.startswith("CF36 log hygiene: PASS")


def test_t33_invalid_settings_exit_2_without_echoing_input(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check_env.setenv("ADMIN_API_KEY", f"{S}short")
    code, out, err = _run(capsys)
    assert code == 2 and out.strip() == "CF36 log hygiene: FAIL settings_invalid"
    assert S not in out + err


def test_t33_console_mode_passes_too(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check_env.setenv("DEBUG", "true")
    code, out, _err = _run(capsys)
    assert code == 0 and out.startswith("CF36 log hygiene: PASS")


def test_t33_python_dash_m_entry_point_exits_zero_and_prints_one_line(tmp_path: Path) -> None:
    pythonpath = os.pathsep.join(
        [
            str(Path(api.__file__).resolve().parent.parent),
            str(Path(common.__file__).resolve().parent.parent),
        ]
    )
    env = {k: v for k, v in os.environ.items() if k not in _SETTINGS_ENV}
    env.update(ENV)
    env["PYTHONPATH"] = pythonpath
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "api.log_hygiene_check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().startswith("CF36 log hygiene: PASS")
    assert len(proc.stdout.strip().splitlines()) == 1
    assert S not in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# T-33 (cf36a-fix): live-trading FAIL semantics (CF36A-SEC-04 / CR-02)
# ---------------------------------------------------------------------------

LIVE_KEY = "FAKEEXCHKEY0123456789"
LIVE_SECRET = "FAKEEXCHSECRET0123456789"  # noqa: S105
LIVE_CONFIRM = "FAKECONFIRMTOKEN0123456"
_LIVE_VALUES = (LIVE_KEY, LIVE_SECRET, LIVE_CONFIRM, "ab12cd", "ab1cd", "yes!!")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """No Settings variable at all, and an empty cwd (no stray ``.env``)."""
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def _assert_one_line_without_values(out: str, err: str) -> str:
    assert err == "", err
    lines = out.strip().splitlines()
    assert len(lines) == 1, out
    for value in _LIVE_VALUES:
        assert value not in out + err
        # neither a value nor its LENGTH may be printed
        assert not re.search(rf"(?<![0-9A-Za-z]){len(value)}(?![0-9A-Za-z])", out), len(value)
    return lines[0]


_FAIL = "CF36 log hygiene: FAIL live_secret_unregistered fields="


@pytest.mark.parametrize(
    ("env", "code", "line"),
    [
        pytest.param({}, 0, "CF36 log hygiene: PASS secrets_registered=0 warn=none", id="paper"),
        pytest.param(
            {"ENABLE_LIVE_TRADING": "true"},
            1,
            _FAIL + "exchange_api_key,exchange_api_secret",
            id="live-nothing",
        ),
        pytest.param(
            {
                "ENABLE_LIVE_TRADING": "true",
                "EXCHANGE_API_KEY": "ab12cd",
                "EXCHANGE_API_SECRET": LIVE_SECRET,
            },
            1,
            _FAIL + "exchange_api_key",
            id="live-6-char-key",
        ),
        pytest.param(
            {
                "ENABLE_LIVE_TRADING": "true",
                "EXCHANGE_API_KEY": LIVE_KEY,
                "EXCHANGE_API_SECRET": LIVE_SECRET,
                "LIVE_TRADING_CONFIRM_TOKEN": "ab1cd",
            },
            1,
            _FAIL + "live_trading_confirm_token",
            id="live-5-char-confirm-token",
        ),
        pytest.param(
            {
                "ENABLE_LIVE_TRADING": "true",
                "EXCHANGE_API_KEY": LIVE_KEY,
                "EXCHANGE_API_SECRET": LIVE_SECRET,
                "LIVE_TRADING_CONFIRM_TOKEN": "yes!!",
            },
            1,
            _FAIL + "live_trading_confirm_token",
            id="live-short-confirm-token-with-encodable-chars",
        ),
        pytest.param(
            {
                "ENABLE_LIVE_TRADING": "true",
                "EXCHANGE_API_KEY": LIVE_KEY,
                "EXCHANGE_API_SECRET": LIVE_SECRET,
                "LIVE_TRADING_CONFIRM_TOKEN": LIVE_CONFIRM,
            },
            0,
            "CF36 log hygiene: PASS secrets_registered=3 warn=none",
            id="live-full",
        ),
        pytest.param(
            {
                "ENABLE_LIVE_TRADING": "true",
                "EXCHANGE_API_KEY": LIVE_KEY,
                "EXCHANGE_API_SECRET": LIVE_SECRET,
            },
            0,
            "CF36 log hygiene: PASS secrets_registered=2 warn=none",
            id="live-key-and-secret-only",
        ),
    ],
)
def test_t33_live_trading_fail_semantics(
    env: dict[str, str],
    code: int,
    line: str,
    clean_env: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, value in env.items():
        clean_env.setenv(name, value)
    got_code, out, err = _run(capsys)
    assert got_code == code, out
    assert _assert_one_line_without_values(out, err) == line


def test_t33_live_with_nothing_configured_fails_in_the_real_python_dash_m_process(
    tmp_path: Path,
) -> None:
    pythonpath = os.pathsep.join(
        [
            str(Path(api.__file__).resolve().parent.parent),
            str(Path(common.__file__).resolve().parent.parent),
        ]
    )
    env = {k: v for k, v in os.environ.items() if k not in _SETTINGS_ENV}
    env.update({"PYTHONPATH": pythonpath, "ENABLE_LIVE_TRADING": "true"})
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "api.log_hygiene_check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout.strip() == _FAIL + "exchange_api_key,exchange_api_secret"
    assert proc.stderr == ""


# ---------------------------------------------------------------------------
# T-33a / T-33b: the uvicorn.access invariant (CF36A-SEC-01)
# ---------------------------------------------------------------------------


def test_t33a_fail_when_the_access_query_filter_is_missing(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = clog.configure_logging

    def without_filter(*a: object, **k: object) -> None:
        original(*a, **k)  # type: ignore[arg-type]
        access = logging.getLogger("uvicorn.access")
        access.filters[:] = [
            f for f in access.filters if not isinstance(f, clog._AccessQueryFilter)
        ]

    check_env.setattr(clog, "configure_logging", without_filter)
    code, out, _err = _run(capsys)
    assert code == 1 and out.strip() == "CF36 log hygiene: FAIL uvicorn_access_query"


def test_t33b_the_no_access_log_state_passes(
    check_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = clog.configure_logging

    def no_access_log(*a: object, **k: object) -> None:
        access = logging.getLogger("uvicorn.access")
        access.handlers.clear()  # what uvicorn --no-access-log leaves behind
        access.propagate = False
        original(*a, **k)  # type: ignore[arg-type]

    check_env.setattr(clog, "configure_logging", no_access_log)
    code, out, _err = _run(capsys)
    assert code == 0 and out.startswith("CF36 log hygiene: PASS"), out
    access = logging.getLogger("uvicorn.access")
    assert not any(isinstance(f, clog._AccessQueryFilter) for f in access.filters)
