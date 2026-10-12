"""
apps/api/log_hygiene_check.py
-----------------------------
Deploy-time self-check for CF-36 log hygiene (G-LIVE L-19, layer 1).

Run inside the deployed container, before the live restart::

    docker compose -f infra/docker-compose.yml exec api python -m api.log_hygiene_check

Expected: exit code 0 and a single line starting ``CF36 log hygiene: PASS``.

With ``ENABLE_LIVE_TRADING=true`` the check also FAILS (exit 1,
``FAIL live_secret_unregistered fields=<names>``) when ``exchange_api_key`` or
``exchange_api_secret`` is not exact-match registered (missing, too short or
trivial), or when a set ``exchange_api_passphrase`` / ``live_trading_confirm_token``
is not registered.  Field names only; never a value or a length.

What it does
------------
Builds the real :class:`api.config.Settings`, configures logging exactly as the
API lifespan does (after applying uvicorn's own ``LOGGING_CONFIG``), registers
the real secret set, and then pushes every configured secret (and fixed
synthetic canaries) through the PRODUCTION logging paths -- structlog
(``error=`` field, plain field, chained ``.exception``), the stdlib bridge,
a ``uvicorn.error`` record and the Telegram URL form -- entirely in memory.
It then asserts that no secret (nor its URL-quoted form) reached either
stream, that every line was actually emitted (non-vacuity), that every
bridge line is valid JSON (JSON mode), and that the processor-chain
invariants hold.

What it prints
--------------
Exactly one line: ``CF36 log hygiene: PASS secrets_registered=<n> warn=<names|none>``
or ``CF36 log hygiene: FAIL <check-id>``.  It never prints a secret value, a
length, a buffer, or an exception message.

Scope: this verifies layer 1 (the central redactor) in the image.  The site
fixes (layer 2) are verified by the CF-36 pytest suite (L-19 step 1).

Exit codes: 0 PASS, 1 FAIL (leak, chain invariant incl. ``uvicorn_access_query``,
``live_secret_unregistered``, DEBUG under live trading), 2 Settings could not be built.
"""

from __future__ import annotations

import io
import json
import logging
import logging.config
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlsplit

if TYPE_CHECKING:
    from api.config import Settings

OK_PREFIX = "CF36 log hygiene: PASS"
FAIL_PREFIX = "CF36 log hygiene: FAIL"

#: Fixed, obviously fake, unregistered canaries: they exercise the pattern
#: rules (R-02/03/05/06/07/08) independently of the exact-match registry.
_CANARY_PW = "cf36-selfcheck-canary-pw-0123456789"
_CANARY_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjZjM2LWNhbmFyeSJ9.Y2YzNi1zaWduYXR1cmUtY2FuYXJ5"
_CANARY_TG = "987654321:AAcf36SelfCheckCanary_0123456789abcdefgh"
_CANARY_TG_SECRET = _CANARY_TG.split(":", 1)[1]

_LOGGER_NAMES = ("uvicorn", "uvicorn.error", "uvicorn.access", "ccxt")

#: Under ``ENABLE_LIVE_TRADING`` these Settings fields MUST be exact-match
#: registered (an unregistered live credential is only pattern-protected).
_LIVE_REQUIRED = ("exchange_api_key", "exchange_api_secret")
#: ... and these too when they are set (non-empty).
_LIVE_IF_SET = ("exchange_api_passphrase", "live_trading_confirm_token")


class _Fail(Exception):
    """Carries a check id only (never a value)."""

    def __init__(self, check_id: str) -> None:
        super().__init__(check_id)
        self.check_id = check_id


# ---------------------------------------------------------------------------
# State isolation
# ---------------------------------------------------------------------------


@contextmanager
def _isolated_logging(out: io.StringIO, err: io.StringIO) -> Iterator[None]:
    """Redirect the streams and restore every piece of global logging state."""
    import structlog

    import common.logging as clog
    from common import redaction

    root = logging.getLogger()
    saved: dict[str, Any] = {
        "stdout": sys.stdout,
        "stderr": sys.stderr,
        "root_handlers": list(root.handlers),
        "root_level": root.level,
        "loggers": {
            n: (
                list(logging.getLogger(n).handlers),
                logging.getLogger(n).propagate,
                logging.getLogger(n).level,
                list(logging.getLogger(n).filters),
            )
            for n in _LOGGER_NAMES
        },
        "structlog": structlog.get_config(),
        "ctx": dict(structlog.contextvars.get_contextvars()),
        "configured": clog._LOGGING_CONFIGURED,
        "registry": redaction._get_registry(),
    }
    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved["stdout"], saved["stderr"]
        root.handlers[:] = saved["root_handlers"]
        root.setLevel(saved["root_level"])
        for n, (handlers, propagate, level, filters) in saved["loggers"].items():
            lg = logging.getLogger(n)
            lg.handlers[:] = handlers
            lg.propagate = propagate
            lg.setLevel(level)
            lg.filters[:] = filters
        structlog.configure(**saved["structlog"])
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**saved["ctx"])
        clog._LOGGING_CONFIGURED = saved["configured"]
        redaction._set_registry(saved["registry"])


# ---------------------------------------------------------------------------
# Probe emission
# ---------------------------------------------------------------------------


def _principal(value: str, kind: str) -> str | None:
    """The text that identifies the secret inside a probe line."""
    from common.redaction import pem_body

    if kind == "url":
        try:
            return urlsplit(value).password or None
        except ValueError:
            return None
    if kind == "pem":
        return pem_body(value)
    if kind == "telegram":
        return value
    return value


class _Counter:
    def __init__(self) -> None:
        self.native = 0
        self.std = 0


def _emit_value_probes(value: str, kind: str, c: _Counter) -> None:
    """Push one registered secret through every production path."""
    import structlog

    log = structlog.get_logger("cf36.selfcheck")
    std = logging.getLogger("cf36.selfcheck")
    uv = logging.getLogger("uvicorn.error")
    url = f"https://h.example/p?api_key={value}&signature={value}"

    log.error("cf36.selfcheck.url", error=f"GET {url}")
    log.error("cf36.selfcheck.field", detail=value)
    c.native += 2
    try:
        try:
            raise RuntimeError(f"inner {url}")
        except RuntimeError as inner:
            raise ValueError(f"outer {value}") from inner
    except ValueError:
        log.exception("cf36.selfcheck.exc")
        c.native += 1
    try:
        raise RuntimeError(url)
    except RuntimeError:
        std.error("cf36.selfcheck.std %s", url, exc_info=True)
        uv.error("cf36.selfcheck.uvicorn Exception in ASGI application %s", url, exc_info=True)
        c.std += 2
    if kind == "telegram":
        tg_url = f"https://api.telegram.org/bot{value}/sendMessage"
        log.error("cf36.selfcheck.telegram", error=f"POST {tg_url}")
        std.error("cf36.selfcheck.std %s", tg_url)
        c.native += 1
        c.std += 1


def _emit_short_probes(value: str, c: _Counter) -> None:
    """Short / unregistered secrets: only the structured positions are protected."""
    import structlog

    log = structlog.get_logger("cf36.selfcheck")
    std = logging.getLogger("cf36.selfcheck")
    log.error("cf36.selfcheck.short", detail=f"password={value}")
    std.error("cf36.selfcheck.std postgresql+asyncpg://u:%s@h/db", value)
    c.native += 1
    c.std += 1


def _emit_synthetic_probes(c: _Counter) -> None:
    import structlog

    log = structlog.get_logger("cf36.selfcheck")
    std = logging.getLogger("cf36.selfcheck")
    forms = {
        "url_params": f"GET https://h.example/p?api_key={_CANARY_PW}&signature={_CANARY_PW}",
        "dsn": f"postgresql+asyncpg://u:{_CANARY_PW}@h:5432/db",
        "redis": f"redis://:{_CANARY_PW}@redis:6379/0",
        "header_repr": (
            f"{{'X-MBX-APIKEY': '{_CANARY_PW}', 'Authorization': 'Bearer {_CANARY_JWT}'}}"
        ),
        "bearer": f"Authorization: Bearer {_CANARY_PW}",
        "jwt": f"token {_CANARY_JWT}",
        "telegram_path": f"https://api.telegram.org/bot{_CANARY_TG}/sendMessage",
        "telegram_bare": f"payload {_CANARY_TG}",
    }
    for key, text in forms.items():
        log.error(f"cf36.selfcheck.synthetic.{key}", error=text)
        std.error("cf36.selfcheck.synthetic.%s %s", key, text)
        c.native += 1
        c.std += 1


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _forbidden_strings(
    secrets: list[tuple[str, str, str]],
) -> tuple[set[str], set[str]]:
    """Return (always_forbidden, composite_forbidden) strings; never logged."""
    from common.redaction import secret_variants

    always: set[str] = {
        _CANARY_PW,
        quote(_CANARY_PW, safe=""),
        _CANARY_JWT,
        _CANARY_TG,
        _CANARY_TG_SECRET,
    }
    composite: set[str] = set()
    for _name, value, kind in secrets:
        variants = secret_variants(value, kind)
        if variants:
            always |= set(variants)
            principal = _principal(value, kind)
            if principal:
                always.add(quote(principal, safe=""))
                if kind == "url":
                    always.add(unquote(principal))
        else:
            # Unregistered (short / trivial): only structured positions are covered.
            principal = _principal(value, kind)
            if principal:
                composite.add(f"password={principal}")
                composite.add(f"u:{principal}@")
    return {a for a in always if a}, composite


def _category(line: str) -> str:
    if "cf36.selfcheck.synthetic" in line:
        return "leak_synthetic"
    if "cf36.selfcheck.uvicorn" in line:
        return "leak_uvicorn"
    if "cf36.selfcheck.std" in line:
        return "leak_stdlib"
    if "cf36.selfcheck" in line:
        return "leak_native"
    return "leak_other"


def _check_chain_invariants(json_mode: bool) -> None:
    """T-17 invariants on the live configuration (native chain + bridge)."""
    import structlog

    from common import redaction

    def tail_ok(processors: list[Any]) -> bool:
        if len(processors) < 3:
            return False
        last, second = processors[-1], processors[-2]
        if not isinstance(last, redaction.RedactingRenderer):
            return False
        if second is not redaction.redact_event_dict:
            return False
        names = [getattr(p, "__name__", type(p).__name__) for p in processors[:-2]]
        return any(n in ("format_exc_info", "ExceptionRenderer") for n in names)

    native = list(structlog.get_config()["processors"])
    if not tail_ok(native):
        raise _Fail("chain_native")

    handler = logging.getLogger().handlers[0] if logging.getLogger().handlers else None
    formatter = getattr(handler, "formatter", None)
    if formatter is None or not tail_ok(list(getattr(formatter, "processors", []))):
        raise _Fail("chain_bridge")

    import common.logging as clog

    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        if lg.handlers or not lg.propagate:
            raise _Fail("uvicorn_absorption")
    access = logging.getLogger("uvicorn.access")
    if access.handlers:
        raise _Fail("uvicorn_absorption")
    # ``--no-access-log`` state (no handlers, propagate=False) is accepted; with the
    # access log on, the query-strip filter must be installed (CF36A-SEC-01).
    if access.propagate and not any(isinstance(f, clog._AccessQueryFilter) for f in access.filters):
        raise _Fail("uvicorn_access_query")
    if logging.getLogger("ccxt").level != logging.WARNING:
        raise _Fail("ccxt_level")

    if not json_mode:
        import structlog.dev

        inner = native[-1].inner
        fmt = getattr(inner, "_exception_formatter", structlog.dev.plain_traceback)
        if fmt is not structlog.dev.plain_traceback:
            raise _Fail("chain_traceback_formatter")


def _run(
    settings: Settings, secrets: list[tuple[str, str, str]], out: io.StringIO, err: io.StringIO
) -> tuple[list[str], list[str]]:
    """Configure, emit and assert.  Returns (registered_field_names, warn_names)."""
    import structlog
    import uvicorn.config

    import common.logging as clog
    from api.config import register_settings_secrets
    from common import redaction

    json_mode = not settings.debug

    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
    structlog.reset_defaults()
    redaction._reset_registry_for_tests()
    clog._LOGGING_CONFIGURED = False
    clog.configure_logging(
        log_level="INFO",
        json_output=json_mode,
        service_name="log-hygiene-check",
        environment="selfcheck",
    )
    structlog.configure(cache_logger_on_first_use=False)
    register_settings_secrets(settings)

    registered: list[str] = []
    warns: list[str] = []
    counter = _Counter()
    for name, value, kind in secrets:
        if redaction.secret_variants(value, kind):
            registered.append(name)
            principal = _principal(value, kind)
            if principal:
                _emit_value_probes(principal, kind, counter)
        elif redaction.secret_skip_reason(value, kind) is not None:
            warns.append(name)
            principal = _principal(value, kind)
            if principal:
                _emit_short_probes(principal, counter)
    _emit_synthetic_probes(counter)

    # ccxt DEBUG must never reach a sink, even when the root logger is DEBUG.
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:
        ccxt_marker = "cf36.selfcheck.ccxt"
        logging.getLogger("ccxt.base.exchange").debug("%s Request: %s", ccxt_marker, _CANARY_PW)
    finally:
        root.setLevel(previous)

    out_text, err_text = out.getvalue(), err.getvalue()
    always, composite = _forbidden_strings(secrets)

    if "cf36.selfcheck.ccxt" in out_text or "cf36.selfcheck.ccxt" in err_text:
        raise _Fail("ccxt_debug")

    for line in (out_text + "\n" + err_text).splitlines():
        if any(f in line for f in always) or any(c in line for c in composite):
            raise _Fail(_category(line))

    native_lines = [ln for ln in out_text.splitlines() if "cf36.selfcheck" in ln]
    std_lines = [ln for ln in err_text.splitlines() if "cf36.selfcheck" in ln]
    if len(native_lines) < counter.native or len(std_lines) < counter.std:
        raise _Fail("non_vacuity")

    if json_mode:
        for ln in err_text.splitlines():
            if not ln.strip():
                continue
            try:
                json.loads(ln)
            except ValueError as exc:
                raise _Fail("bridge_not_json") from exc
        for ln in native_lines:
            try:
                json.loads(ln)
            except ValueError as exc:
                raise _Fail("native_not_json") from exc

    _check_chain_invariants(json_mode)
    return sorted(registered), sorted(warns)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the self-check; print one line; return the process exit code."""
    real_out = sys.stdout

    try:
        from api.config import Settings, iter_settings_secrets

        settings = Settings()
    except Exception:
        print(f"{FAIL_PREFIX} settings_invalid", file=real_out)
        return 2

    if settings.enable_live_trading and settings.log_level.upper() == "DEBUG":
        print(f"{FAIL_PREFIX} log_level_debug", file=real_out)
        return 1

    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        secrets = iter_settings_secrets(settings)
        with _isolated_logging(out_buf, err_buf):
            registered, warns = _run(settings, secrets, out_buf, err_buf)
    except _Fail as fail:
        print(f"{FAIL_PREFIX} {fail.check_id}", file=real_out)
        return 1
    except Exception:
        print(f"{FAIL_PREFIX} internal_error", file=real_out)
        return 1

    if settings.enable_live_trading:
        present = {name for name, _, _ in secrets}
        missing = [n for n in _LIVE_REQUIRED if n not in registered] + [
            n for n in _LIVE_IF_SET if n in present and n not in registered
        ]
        if missing:
            print(
                f"{FAIL_PREFIX} live_secret_unregistered fields={','.join(missing)}", file=real_out
            )
            return 1
    warn_text = ",".join(warns) if warns else "none"
    print(f"{OK_PREFIX} secrets_registered={len(registered)} warn={warn_text}", file=real_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
