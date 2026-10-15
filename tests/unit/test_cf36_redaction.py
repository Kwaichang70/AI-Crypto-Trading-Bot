"""
tests/unit/test_cf36_redaction.py
---------------------------------
CF-36 layer 1 (cf36a): the redaction module in isolation.

    T-01  rule table R-01..R-08 (+ bounded R-02)
    T-02  Telegram forms          T-03  DSN forms
    T-04  headers / JWT           T-05  PEM
    T-06  registry variants, URL-kind password only, PEM == normalizer, threads
    T-15  golden byte-identity (old chain vs new chain)
    T-16  AST key-name guard (R-10)
    T-18  performance + ReDoS near-miss corpus (+ T-18p pinned worst cases, T-18u uvicorn
          probe, T-18g generated sweep)
    T-34  registry non-disclosure
    T-36  prefilter completeness and gate soundness (+ T-36t precompiled templates)
    cf36a-fix: R-07 linearity, R-08/R-11 quoted values, T-01q, size-cap straddle,
          registry URL variants, bytes final pass, processor-key carve-out
    T-37  rule order (R-09 first) and JSON validity
    T-38  short and empty secrets

All secrets are fake sentinels.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import enum
import json
import os
import pickle
import random
import re
import string
import subprocess
import sys
import threading
import time
import timeit
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

import aiohttp
import pytest
import structlog
import yarl
from multidict import CIMultiDict, CIMultiDictProxy

from common import redaction as R
from common.config import build_renderer
from tests._cf36_logging import PARITY_NAMES, U, fake_jwt

S = "CF36SENTINEL"  # distinctive prefix of every fake secret
# Fake credential shapes are assembled at runtime so that secret scanners do not
# match the source text (the runtime values are what the tests exercise).
_DASHES = "-" * 5
_PK = "PRIVATE KEY"
_TG_ID = "1234" + "56789"
_V10 = "VAL" + "0123456789"
_V8 = "VAL" + "01234567"
TG_TOKEN = f"{_TG_ID}:AA{S}TelegramSecret_0123456789-xyz"
TG_SECRET = TG_TOKEN.split(":", 1)[1]
DSN_PW = f"{S}-DbPw-0123456789"
PEM_BODY_LINE_1 = f"MHcCAQEEI{S}Line1Line1Line1Line1Line1AAAAAA"
PEM_BODY_LINE_2 = f"BBBBBBbcdefghij{S}Line2Line2Line2Line2Line2"
PEM = (
    f"{_DASHES}BEGIN EC {_PK}{_DASHES}\n"
    f"{PEM_BODY_LINE_1}\n{PEM_BODY_LINE_2}\n"
    f"{_DASHES}END EC {_PK}{_DASHES}"
)
JWT = fake_jwt('{"alg":"ES256"}', '{"sub":"cf36-sentinel"}', "signature-sentinel")


def _no_sentinel(text: str) -> None:
    assert S not in text, text


# ---------------------------------------------------------------------------
# T-01  rule table
# ---------------------------------------------------------------------------

_R08_NAMES = [
    "api_key",
    "apiKey",
    "API_KEY",
    "apikey",
    "api-key",
    "access_key",
    "AccessKeyId",
    "secret",
    "secret_key",
    "client_secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "confirm_token",
    "signature",
    "Signature",
    "sign",
    "sig",
    "passphrase",
    "password",
    "passwd",
    "pwd",
    "private_key",
    "key",
]

_RULE_CASES: list[tuple[str, str]] = [
    # R-01
    (f"x {PEM} y", "x [REDACTED_PRIVATE_KEY] y"),
    (f"{_DASHES}BEGIN {_PK}{_DASHES}\nabc", "[REDACTED_PRIVATE_KEY]"),
    # R-02
    (
        "postgresql+asyncpg://trading:pw@db:5432/x",
        "postgresql+asyncpg://trading:[REDACTED]@db:5432/x",
    ),
    ("redis://:pw@redis:6379/0", "redis://:[REDACTED]@redis:6379/0"),
    ("postgres://u:p%40ss@h/db", "postgres://u:[REDACTED]@h/db"),
    ("https://user:pass@example.com/x", "https://user:[REDACTED]@example.com/x"),
    ("postgres://u@h/db", "postgres://u@h/db"),
    ("https://api.binance.com/api/v3/time", "https://api.binance.com/api/v3/time"),
    ("http://127.0.0.1:8000/api/v1/runs/42", "http://127.0.0.1:8000/api/v1/runs/42"),
    # R-03
    (
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        "https://api.telegram.org/bot123456789:[REDACTED]/sendMessage",
    ),
    # R-04
    (f"payload {TG_TOKEN} end", "payload [REDACTED_TG_TOKEN] end"),
    # R-05
    (
        "{'X-MBX-APIKEY': '" + "MBXKEY" + "0123456789', 'Content-Type': 'x'}",
        "{'X-MBX-APIKEY': '[REDACTED]', 'Content-Type': 'x'}",
    ),
    ("Authorization: Bearer abc.def-123", "Authorization: [REDACTED]"),
    ("cb-access-sign=ZZZZ0123456789", "cb-access-sign=[REDACTED]"),
    # R-06
    ("failed with bearer " + "abcdefgh" + "12345 now", "failed with Bearer [REDACTED] now"),
    # R-07
    (f"tok {JWT} z", "tok [REDACTED_JWT] z"),
    # R-08 spot checks (the full matrix is parametrised separately)
    ("https://h/p?a=1&signature=SIG0123456789&b=2", "https://h/p?a=1&signature=[REDACTED]&b=2"),
    (
        "host=db user=u password=" + "hunter" + "22 dbname=x",
        "host=db user=u password=[REDACTED] dbname=x",
    ),
    # negatives: ordinary text is untouched
    ("BTC/USDT", "BTC/USDT"),
    ("sign of trend", "sign of trend"),
    ("127.0.0.1:8000/api/v1/runs/42", "127.0.0.1:8000/api/v1/runs/42"),
    (
        "key_prefix=ab12 monkey=1 tokens=5 signal=buy design=x",
        "key_prefix=ab12 monkey=1 tokens=5 signal=buy design=x",
    ),
    ("api_key=[REDACTED]", "api_key=[REDACTED]"),
]


@pytest.mark.parametrize(("raw", "expected"), _RULE_CASES)
def test_t01_rule_table(raw: str, expected: str) -> None:
    assert R.redact_text(raw) == expected


@pytest.mark.parametrize("name", _R08_NAMES)
@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("https://h/p?{n}=VAL0123456789&x=1", "https://h/p?{n}=[REDACTED]&x=1"),
        ("https://h/p?x=1&{n}=VAL0123456789", "https://h/p?x=1&{n}=[REDACTED]"),
        ("{n}=VAL0123456789&x=1", "{n}=[REDACTED]&x=1"),
        ("https://h/p?{n}%3DVAL0123456789&x=1", "https://h/p?{n}%3D[REDACTED]&x=1"),
        ("a=1&{n}=VAL0123456789", "a=1&{n}=[REDACTED]"),
    ],
)
def test_t01_r08_matrix(name: str, template: str, expected: str) -> None:
    out = R.redact_text(template.format(n=name))
    if name == "api-key":
        # "api-key" is also a header name (R-05, applied first); its value class
        # is not '&'-terminated, so trailing query parameters are redacted too.
        # That is over-redaction in the safe direction; the secret must be gone.
        assert "VAL0123456789" not in out and "[REDACTED]" in out
        return
    assert out == expected.format(n=name)


def test_t01_r08_is_case_insensitive_and_covers_htx_and_form_bodies() -> None:
    out = R.redact_text(
        "AccessKeyId=AKIA0123456789&SignatureMethod=HmacSHA256&Signature=ABCDEF0123"
    )
    assert out == "AccessKeyId=[REDACTED]&SignatureMethod=HmacSHA256&Signature=[REDACTED]"
    assert R.redact_text("a=1&secret=x") == "a=1&secret=[REDACTED]"


def test_t01_r02_is_bounded_not_superlinear() -> None:
    """ARB-CF36-03: the unbounded design pattern took ~13 s on 68 KB of 'a.b.' runs."""
    for n in (2_048, 16_384, 68_000):
        s = "a." * (n // 2)
        t0 = time.perf_counter()
        R._RULES[1].rx.search(s)
        assert time.perf_counter() - t0 < 0.25, n
        t0 = time.perf_counter()
        out = R.redact_text(s)
        assert out == s or n > R.LIMIT  # dotted runs hold no credential (only the cap cuts)
        assert time.perf_counter() - t0 < 0.25, n


def test_t01_rules_are_ordered_and_complete() -> None:
    assert [r.id for r in R._RULES] == [*(f"R-0{i}" for i in range(1, 9)), "R-11"]


@pytest.mark.parametrize(
    "context",
    ["", "x ", "k=", "/", "'", '"', ":", ".", "(", "Bearer ", "token=", "\n", "[", "{"],
    ids=repr,
)
def test_t01_r07_still_matches_after_every_boundary(context: str) -> None:
    """CF36A-SEC-02: the linear R-07 keeps matching a JWT that is not glued to an identifier."""
    out = R.redact_text(f"{context}{JWT} z")
    assert JWT not in out and "[REDACTED" in out, out


def test_t01_r07_documented_residual_rk19_jwt_glued_to_an_identifier() -> None:
    """RK-19 (accepted): ``x-eyJ...`` / ``_eyJ...`` is not matched by the linear R-07."""
    for glued in (f"x-{JWT}", f"_{JWT}", f"a{JWT}"):
        assert R.redact_text(glued) == glued


_R08_R11_CASES: list[tuple[str, str]] = [
    # R-08 quoted values (kwarg / dataclass / BasicAuth repr)
    (f"api_key='{_V10}'", "api_key='[REDACTED]'"),
    ('password="abc def"', 'password="[REDACTED]"'),
    (
        f"Creds(api_key='{_V10}', secret='{_V10}')",
        "Creds(api_key='[REDACTED]', secret='[REDACTED]')",
    ),
    (f"password=b'{_V10}'", "password=b'[REDACTED]'"),
    ("password=''", "password=''"),
    # R-08 snake_case-prefixed names (the old lookbehind skipped every one of them)
    ("db_password=" + "hunter" + "22", "db_password=[REDACTED]"),
    (f"exchange_api_secret={_V10}", "exchange_api_secret=[REDACTED]"),
    (f"telegram_bot_token={_V10}", "telegram_bot_token=[REDACTED]"),
    (f"x_api_key={_V10}", "x_api_key=[REDACTED]"),
    (f"apiSecret={_V10}&x=1", "apiSecret=[REDACTED]&x=1"),
    (f"CB-ACCESS-SIGN={_V10}", "CB-ACCESS-SIGN=[REDACTED]"),
    # R-11 ``name: value`` and quoted keys
    (f'{{"apiKey": "{_V10}"}}', '{"apiKey": "[REDACTED]"}'),
    (f"{{'Token': '{_V10}'}}", "{'Token': '[REDACTED]'}"),
    (f"api_key: {_V10}", "api_key: [REDACTED]"),
    ('{"sign": -1}', '{"sign": [REDACTED]}'),
    ("{'password': b'VAL0123456789'}", "{'password': b'[REDACTED]'}"),
    ('{"token": null}', '{"token": null}'),
    ("{'token': None}", "{'token': None}"),
    ('{"auth": true, "n": 1}', '{"auth": true, "n": 1}'),
    ("{'api_key': '[REDACTED]'}", "{'api_key': '[REDACTED]'}"),
    # still untouched
    ("idempotency_key=abc123 cache_key: abc123", "idempotency_key=abc123 cache_key: abc123"),
    ("[Authorization: [REDACTED]]", "[Authorization: [REDACTED]]"),
]


@pytest.mark.parametrize(("raw", "expected"), _R08_R11_CASES)
def test_t01_r08_r11_extended_cases(raw: str, expected: str) -> None:
    assert R.redact_text(raw) == expected


@pytest.mark.parametrize("name", ["cookie", "dsn", "credential", "credentials", "Set-Cookie"])
def test_t01_r10_covers_cookie_dsn_and_credential_keys(name: str) -> None:
    """CF36A-CR-06(b): dropping one of these from the grammar must fail a test."""
    assert R.is_sensitive_key(name)
    out = R.redact_event_dict(None, "info", {"event": "e", name: U, "nested": {name: U}})
    assert out[name] == "[REDACTED]" and out["nested"] == {name: "[REDACTED]"}


def test_t01_r10_is_camel_case_aware_and_suffix_anchored() -> None:
    for yes in ("apiSecret", "XApiKey", "APIKey", "botToken", "AccessKeyId", "X-CMC_PRO_API_KEY"):
        assert R.is_sensitive_key(yes), yes
    for no in ("idempotency_key", "public_key", "tokens", "design", "assign", "WWW-Authenticate"):
        assert not R.is_sensitive_key(no), no
    # suffix-anchored: a long key keeps its LAST 128 characters
    assert R.is_sensitive_key("x" * 500 + "_password")


# T-01q: quoted / repr rows (CF36A-SEC-03)


def test_t01q_dataclass_namedtuple_and_basicauth_reprs() -> None:
    creds = dataclasses.make_dataclass("Creds", [("api_key", str), ("secret", str)])
    nt = collections.namedtuple("NT", "apiSecret")
    rows = [
        repr(creds(U, U)),
        repr(nt(U)),
        repr(aiohttp.BasicAuth("u", U)),
        repr({"password": U.encode()}),
    ]
    for text in rows:
        assert U in text, "premise: the row carries the value"
        out = R.redact_text(text)
        assert U not in out and "[REDACTED" in out, out
    # the same rows through the walker (native path)
    ev = R.redact_event_dict(None, "info", {"event": "e", "obj": creds(U, U), "nt": nt(U)})
    assert U not in repr(ev) and "[REDACTED" in repr(ev)


def test_t01q_aiohttp_client_response_error_repr_with_credential_headers() -> None:
    headers = CIMultiDictProxy(
        CIMultiDict(
            {"apikey": U, "X-CMC_PRO_API_KEY": U, "X-Api-Secret": U, "Token": U, "Accept": "x"}
        )
    )
    info = aiohttp.RequestInfo(
        url=yarl.URL("https://h/p"),
        method="GET",
        headers=headers,
        real_url=yarl.URL("https://h/p"),
    )
    exc = aiohttp.ClientResponseError(info, (), status=401, message="Unauthorized")
    assert U in repr(exc), "premise"
    out = R.redact_text(repr(exc))
    assert U not in out and out.count("[REDACTED]") == 4
    assert "'Accept': 'x'" in out


def test_t01q_empty_and_null_values_are_unchanged() -> None:
    for text in ("password=''", '{"token": null}', "{'token': None}", "{'password': ''}"):
        assert R.redact_text(text) == text


@pytest.mark.parametrize("depth7", [True, False], ids=["depth7", "items300"])
def test_t01q_over_limit_containers_still_lose_no_password(depth7: bool) -> None:
    """CF36A-CR-05 (closed by R-11) / CR-06(c): the repr path redacts a ``password`` key."""
    payload: dict[str, Any]
    if depth7:
        payload = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"password": U}}}}}}}}
    else:
        payload = {f"k{i}": i for i in range(300)}
        payload["password"] = U
    out = R.redact_event_dict(None, "info", {"event": "e", "payload": payload})
    text = repr(out)
    assert U not in text and "[REDACTED]" in text


# ---------------------------------------------------------------------------
# T-02 .. T-05  per-secret-family samples
# ---------------------------------------------------------------------------


def test_t02_unregistered_real_length_telegram_secret_in_a_bot_url() -> None:
    """CF36A-CR-06(a): the real secret half is 35 characters (R-03 ``{20,}``)."""
    secret = "AAH" + "k3" * 16
    assert len(secret) == 35
    out = R.redact_text(f"https://api.telegram.org/bot123456789:{secret}/sendMessage")
    assert out == "https://api.telegram.org/bot123456789:[REDACTED]/sendMessage"


def test_t02_telegram_forms_rules_only() -> None:
    forms = [
        f"InvalidURL: https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        f'{{"ok":false,"description":"Not Found: POST /bot{TG_TOKEN}/sendMessage"}}',
        f'{{"token_value":"{TG_TOKEN}"}}',
        f"ClientConnectorError(https://api.telegram.org/bot{TG_TOKEN}/getMe)",
    ]
    for form in forms:
        out = R.redact_text(form)
        assert TG_TOKEN not in out and TG_SECRET not in out, out
        assert "[REDACTED" in out


def test_t02_telegram_registered_token_and_half() -> None:
    assert R.register_secret(TG_TOKEN, name="telegram_bot_token", kind="telegram") >= 2
    for form in (f"x {TG_TOKEN} y", f"x {TG_SECRET} y", f"bot{TG_TOKEN}"):
        out = R.redact_text(form)
        assert TG_SECRET not in out and TG_TOKEN not in out, out


def test_t03_dsn_forms() -> None:
    forms = {
        f"postgresql+asyncpg://trading:{DSN_PW}@db:5432/x": "postgresql+asyncpg://trading:[REDACTED]@db:5432/x",
        f"postgres://u:p%40ss{S}@h/db": "postgres://u:[REDACTED]@h/db",
        f"redis://:{DSN_PW}@redis:6379/0": "redis://:[REDACTED]@redis:6379/0",
        "postgres://user-only@h/db": "postgres://user-only@h/db",
    }
    for raw, expected in forms.items():
        assert R.redact_text(raw) == expected
    sa = f"Could not parse SQLAlchemy URL from string 'postgresql+asyncpg://u:{DSN_PW}@db/x'"
    out = R.redact_text(sa)
    _no_sentinel(out)
    assert "[REDACTED]" in out


def test_t04_headers_and_jwt() -> None:
    hdr = f"{{'X-MBX-APIKEY': '{S}KEY0123456789', 'Authorization': 'Bearer {JWT}'}}"
    out = R.redact_text(hdr)
    _no_sentinel(out)
    assert JWT not in out and "[REDACTED" in out
    assert "[REDACTED_JWT]" in R.redact_text(f"payload {JWT}")
    assert R.redact_text(f"CB-ACCESS-SIGN: {S}SIGN0123456789") == "CB-ACCESS-SIGN: [REDACTED]"
    assert R.redact_text(f"X-Admin-Key: {S}ADMIN0123456789abcdef") == "X-Admin-Key: [REDACTED]"
    assert (
        R.redact_text(f"x-live-confirm-token={S}TOK0123456789") == "x-live-confirm-token=[REDACTED]"
    )
    assert R.redact_text(f"KC-API-PASSPHRASE: {S}PASS0123456789") == "KC-API-PASSPHRASE: [REDACTED]"
    assert R.redact_text(f"Bearer {S}TOKEN0123456789") == "Bearer [REDACTED]"


def test_t05_pem_forms() -> None:
    escaped = PEM.replace("\n", "\\n")
    for form in (PEM, escaped, json.dumps({"k": PEM})):
        out = R.redact_text(form)
        _no_sentinel(out)
        assert "[REDACTED_PRIVATE_KEY]" in out
    # unterminated key (cut at a log boundary) is still removed to the end
    assert S not in R.redact_text(PEM[: PEM.index("-----END")])
    # registered: the body survives neither raw, escaped nor JSON-rendered
    R.register_secret(PEM, name="exchange_api_secret", kind="pem")
    body = PEM_BODY_LINE_1 + PEM_BODY_LINE_2
    for form in (body, PEM_BODY_LINE_1, f"line {PEM_BODY_LINE_2} x", escaped):
        _no_sentinel(R.redact_text(form))


# ---------------------------------------------------------------------------
# T-06  registry
# ---------------------------------------------------------------------------


def test_t06_variants_cover_every_rendering() -> None:
    secret = f'{S}/a b+c"d\\e\nf\u00e9'
    added = R.register_secret(secret, name="x")
    assert added >= 5
    from urllib.parse import quote, quote_plus

    renderings = {
        secret,
        quote(secret, safe=""),
        quote_plus(secret),
        json.dumps(secret)[1:-1],
        secret.encode("unicode_escape").decode(),
    }
    for form in renderings:
        out = R.redact_text(f"before {form} after")
        _no_sentinel(out)
        assert "[REDACTED]" in out


def test_t06_url_kind_registers_only_the_password() -> None:
    dsn = f"postgresql+asyncpg://trading:{DSN_PW}@db.internal:5432/tradingdb"
    assert R.register_secret(dsn, name="database_url", kind="url") >= 1
    # host / db name stay readable in legitimate logs
    assert R.redact_text("connecting to db.internal:5432/tradingdb as trading") == (
        "connecting to db.internal:5432/tradingdb as trading"
    )
    out = R.redact_text(f"password is {DSN_PW} ok")
    assert DSN_PW not in out and "[REDACTED]" in out
    # a URL without a password registers nothing
    assert R.register_secret("postgresql+asyncpg://trading@db/x", name="x", kind="url") == 0
    assert R.register_secret("redis://localhost:6379/0", name="x", kind="url") == 0
    # percent-encoded password: both the encoded and the decoded rendering are covered
    R._reset_registry_for_tests()
    R.register_secret(f"postgres://u:p%40ss-{S}-0123@h/db", name="x", kind="url")
    assert S not in R.redact_text(f"p@ss-{S}-0123")
    assert S not in R.redact_text(f"p%40ss-{S}-0123")


def test_t06_pem_kind_body_equals_the_orchestrator_normalizer() -> None:
    from api.services.run_orchestrator import _normalize_exchange_secret

    for value in (PEM, PEM.replace("\n", "\\n"), "plainbase64bodywithoutheaders0123456789=="):
        assert R.pem_body(value) == _normalize_exchange_secret(value)
    assert R.pem_body(PEM) in R.secret_variants(PEM, "pem")


def test_t06_telegram_kind_adds_secret_half() -> None:
    variants = R.secret_variants(TG_TOKEN, "telegram")
    assert TG_TOKEN in variants and TG_SECRET in variants


def test_t06_registration_never_logs_or_raises() -> None:
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        R.register_secret(f"{S}-registration-check-0123", name="x")
        R.register_secrets_from_env({"X_API_KEY": f"{S}-env-registration-0123"})
    assert logs == []
    assert R.register_secret(None, name="x") == 0
    assert R.register_secret(12345, name="x") == 0  # type: ignore[arg-type]


def test_t06_concurrent_registration_and_redaction() -> None:
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(i: int) -> None:
        try:
            for j in range(40):
                R.register_secret(f"{S}-thread-{i}-{j}-0123456789", name="t")
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                R.redact_event_dict(
                    None, "info", {"event": "x", "detail": f"{S}-thread-0-0-0123456789"}
                )
        except BaseException as exc:
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()
    assert errors == []
    assert R.registered_secret_count() == 8 * 40
    assert S not in R.redact_text(f"{S}-thread-7-39-0123456789")


# ---------------------------------------------------------------------------
# T-13 (processor part)  fail-safe
# ---------------------------------------------------------------------------


class _BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError(f"{S} repr exploded")


def _failures() -> int:
    from common.metrics import metrics

    return int(metrics.get_all()["counters"].get("log_redaction_failures_total", 0))


def test_t13_value_with_raising_repr_is_replaced_not_propagated() -> None:
    before = _failures()
    out = R.redact_event_dict(
        None, "info", {"event": "e", "bad": _BadRepr(), "ok": "fine", "level": "info"}
    )
    assert out["bad"] == R.REDACTION_FAILED
    assert out["ok"] == "fine" and out["level"] == "info"
    assert S not in repr(out)
    assert _failures() > before


def test_t13_rule_failure_yields_marker_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def subn(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError(S)

    monkeypatch.setattr(R._RULES[7], "rx", Boom())
    before = _failures()
    assert R.redact_text(f"a=b&api_key={_V8}") == R.REDACTION_FAILED
    assert _failures() > before


def test_t13_whole_processor_failure_keeps_only_safe_fields() -> None:
    class Evil:
        """A mapping whose ``dict(...)`` conversion fails (outside the per-key guard)."""

        def __init__(self, **data: Any) -> None:
            self._data = data

        def keys(self) -> Any:
            raise RuntimeError(S)

        def get(self, key: str, default: Any = None) -> Any:
            return self._data.get(key, default)

    evil = Evil(
        event="secret event api_key=VAL",
        level="warning",
        timestamp="2026-01-01T00:00:00Z",
        logger="a.b.c",
        other="x",
    )
    out = R.redact_event_dict(None, "warning", evil)
    assert out == {
        "event": R.REDACTION_FAILED,
        "redaction_error": True,
        "level": "warning",
        "timestamp": "2026-01-01T00:00:00Z",
        "logger": "a.b.c",
    }
    out2 = R.redact_event_dict(None, "x", Evil(level="nonsense", logger="bad logger!"))
    assert out2 == {"event": R.REDACTION_FAILED, "redaction_error": True}


def test_t13_cyclic_and_deep_structures_do_not_raise() -> None:
    cyc: dict[str, Any] = {}
    cyc["self"] = cyc
    deep: Any = "leaf"
    for _ in range(50):
        deep = [deep]
    out = R.redact_event_dict(None, "info", {"event": "e", "cyc": cyc, "deep": deep})
    assert "cyc" in out and "deep" in out


def test_t13_recursion_error_in_structlog_hook_is_contained() -> None:
    class Rec:
        def __structlog__(self) -> str:
            raise RecursionError("maximum recursion depth exceeded")

    out = R.redact_event_dict(None, "info", {"event": "e", "r": Rec()})
    assert out["r"] == R.REDACTION_FAILED


def test_t13_base_exceptions_are_not_swallowed() -> None:
    class Interrupt:
        def __repr__(self) -> str:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        R.redact_event_dict(None, "info", {"event": "e", "x": Interrupt()})


def test_t13_oversized_string_is_cut_after_redaction() -> None:
    big = "z" * 10_000_000 + f" api_key={S}TAILKEY0123"
    out = R.redact_text(big)
    assert len(out) < R.LIMIT + 100
    assert "[TRUNCATED len=" in out
    assert S not in out
    # a secret that starts inside the kept region but ends in the margin is wholly redacted
    boundary = "q" * (R.LIMIT - 10) + f" api_key={S}BOUNDARY0123456789 " + "q" * 50_000
    red = R.redact_text(boundary)
    assert S not in red and "BOUNDARY" not in red


def test_t13_traceback_keeps_head_and_tail() -> None:
    head = "Traceback (most recent call last):\n"
    tail = f"RuntimeError: final line api_key={S}FINAL0123456789\n"
    exc = head + ('  File "x.py", line 1, in f\n' * 5000) + tail
    assert len(exc) > R.LIMIT
    out = R.redact_event_dict(None, "error", {"event": "e", "exception": exc})["exception"]
    assert out.startswith(head)
    assert "RuntimeError: final line api_key=[REDACTED]" in out
    assert S not in out and "[TRUNCATED len=" in out


# ---------------------------------------------------------------------------
# T-15  golden byte identity: the old chain vs the new chain
# ---------------------------------------------------------------------------


class _Color(enum.Enum):
    RED = "red"


class _StrColor(str, enum.Enum):  # noqa: UP042
    BLUE = "blue"


class _WithStructlog:
    def __structlog__(self) -> str:
        return "custom-structlog-repr"


_TS = "2026-01-15T12:00:00.000000Z"
_BASE = {
    "timestamp": _TS,
    "level": "info",
    "logger": "trading.engine",
    "service": "trading-bot",
    "env": "production",
}


def _corpus() -> list[dict[str, Any]]:
    uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    when = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    events: list[dict[str, Any]] = [
        {
            "event": "engine.intrabar_exit_signal",
            "symbol": "BTC/USDT",
            "price": Decimal("65000.12"),
            "side": "sell",
        },
        {
            "event": "bracket_exit.anchor_finalized",
            "run_id": "run_20260227T143022Z_a3f7c1b2",
            "stop": 0.0123,
        },
        {"event": "circuit_breaker.halt_entered", "reason": "drawdown", "dd_pct": Decimal("12.5")},
        {
            "event": "http.request",
            "method": "GET",
            "path": "/api/v1/runs/42",
            "status": 200,
            "ms": 12.3,
        },
        {"event": "runs.intrabar_exit_checks", "checks": 3, "ok": True, "none": None},
        {"event": "idempotency.claimed", "key_prefix": "ab12cd34", "scope": "POST /runs"},
        {"event": "decimal.uuid.datetime", "d": Decimal("1.50"), "u": uid, "t": when},
        {"event": "enum.values", "e": _Color.RED, "se": _StrColor.BLUE, "obj": _WithStructlog()},
        {
            "event": "collections",
            "s": {1, 2},
            "fs": frozenset({3}),
            "b": b"ab",
            "t": (1, "two", 3.0),
        },
        {"event": "nested", "d": [1, 2, {"b": Decimal("2")}]},
        {"event": "int.keys", "d": {1: "one", 2: "two"}, "mix": {"a": 1, 2: "b"}},
        {
            "event": "text.symbols",
            "pair": "BTC/USDT",
            "note": "sign of trend",
            "addr": "127.0.0.1:8000/api/v1/runs/42",
        },
        {
            "event": "user.message",
            "user_message": "Exchange unavailable — the exchange may be experiencing issues.",
        },
        {"event": "empty.collections", "l": [], "d": {}, "t": (), "s": set()},
        {"event": "unicode", "text": "caf\u00e9 \u20ac \u2713", "emoji": "ok"},
        {"event": "floats", "nan": float("nan"), "inf": float("inf"), "neg": -0.0, "big": 10**20},
        {"event": "long.plain", "text": "word " * 2000},
        {"event": "equals.but.clean", "detail": "rate=0.5 side=buy qty=1.25 key_prefix=ab12"},
        {"event": "colon.but.clean", "detail": "ratio 3:1 at 10:30:00 (UTC) host 127.0.0.1:8000"},
        {
            "event": "url.but.clean",
            "url": "https://api.binance.com/api/v3/time",
            "ws": "wss://stream.binance.com:9443/ws",
        },
        {
            "event": "exception.text",
            "exception": 'Traceback (most recent call last):\n  File "x.py", line 1, in f\nValueError: bad',  # noqa: E501
        },
        {
            "event": "stack.text",
            "stack": 'Stack (most recent call last):\n  File "x.py", line 2, in g',
        },
        {"event": "none.level", "level": "warning", "timestamp": _TS},
        {"event": "warning.with.list", "items": ["a", "b", "c"], "n": 3},
        {"event": "tuple.in.dict", "d": {"k": (1, 2)}},
        {"event": "bool.int", "flag": False, "count": 0},
        {"event": "dotted.event.name_with-dash:colon", "x": 1},
        {"event": "Mixed Case Event With Spaces", "x": 1},
        {"event": "large.int.list", "xs": list(range(150))},
        {"event": "over.limit.list", "xs": list(range(500))},
        {"event": "deep", "d": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}},
        {
            "event": "ccxt.user_message",
            "user_message": "Rate limited — too many requests. The bot will retry automatically.",
        },
    ]
    return [{**_BASE, **e} for e in events]


def _old_chain(json_logs: bool) -> list[Any]:
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    return [structlog.processors.format_exc_info, renderer]


def _new_chain(json_logs: bool) -> list[Any]:
    return [
        structlog.processors.format_exc_info,
        R.redact_event_dict,
        R.RedactingRenderer(build_renderer(json_logs)),
    ]


def _run(chain: list[Any], event: dict[str, Any]) -> Any:
    ed: Any = dict(event)
    for proc in chain:
        ed = proc(None, "info", ed)
    return ed


@pytest.mark.parametrize("json_logs", [True, False], ids=["json", "console"])
def test_t15_normal_logs_are_byte_identical(json_logs: bool) -> None:
    corpus = _corpus()
    assert len(corpus) >= 30
    # secrets registered, so the exact-match path is live while output must not change
    R.register_secret(f"{S}-registered-but-never-logged-0123", name="x")
    old, new = _old_chain(json_logs), _new_chain(json_logs)
    for event in corpus:
        a, b = _run(old, event), _run(new, event)
        assert a == b, f"output changed for {event['event']!r}"
    assert len({_run(new, e) for e in corpus}) >= 30


@pytest.mark.parametrize("json_logs", [True, False], ids=["json", "console"])
def test_t15_exception_events_are_identical_when_clean(json_logs: bool) -> None:
    try:
        raise ValueError("plain failure")
    except ValueError as exc:
        event = {**_BASE, "event": "x.failed", "exc_info": exc}
    assert _run(_old_chain(json_logs), event) == _run(_new_chain(json_logs), event)


def test_t15_clean_event_returns_the_same_value_objects() -> None:
    event = _corpus()[6]
    out = R.redact_event_dict(None, "info", event)
    assert out is not event
    for key, value in event.items():
        assert out[key] is value


def test_t15_changed_values_are_replaced_only_where_changed() -> None:
    event = {
        **_BASE,
        "event": "e",
        "clean": Decimal("1"),
        "dirty": f"https://h/p?api_key={S}KEY0123456789",
    }
    out = R.redact_event_dict(None, "info", event)
    assert out["clean"] is event["clean"]
    assert out["dirty"] == "https://h/p?api_key=[REDACTED]"


# ---------------------------------------------------------------------------
# T-16  AST key-name guard (R-10)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}
#: Keyword names that match R-10 and are reviewed + allowed.  Starts EMPTY (V-11).
_R10_ALLOWLIST: frozenset[str] = frozenset()


def _log_call_keywords() -> tuple[int, set[str]]:
    calls = 0
    names: set[str] = set()
    for root in ("apps", "packages"):
        for path in (_REPO / root).rglob("*.py"):
            if "tests" in path.parts or "alembic" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _LOG_METHODS
                    and node.args
                    and isinstance(node.args[0], (ast.Constant, ast.JoinedStr))
                ):
                    calls += 1
                    names.update(k.arg for k in node.keywords if k.arg)
    return calls, names


def test_t16_no_production_log_keyword_matches_the_sensitive_key_rule() -> None:
    calls, names = _log_call_keywords()
    assert calls >= 500, f"scan found only {calls} log calls (vacuous)"
    assert len(names) >= 150
    matching = {n for n in names if R.is_sensitive_key(n)}
    assert matching == set(_R10_ALLOWLIST), (
        f"log keyword(s) {sorted(matching - _R10_ALLOWLIST)} match R-10 and would be redacted "
        "whole; review them and extend _R10_ALLOWLIST consciously"
    )
    for safe in (
        "key_prefix",
        "token_count",
        "error_type",
        "exc_type",
        "api_auth_enabled",
        "auth_method",
    ):
        assert not R.is_sensitive_key(safe)
    for sensitive in (
        "api_key",
        "token",
        "password",
        "x-api-key",
        "client_secret",
        "authorization",
        "dsn",
    ):
        assert R.is_sensitive_key(sensitive)


def test_t16_r10_redacts_whole_values_at_any_depth() -> None:
    out = R.redact_event_dict(
        None,
        "info",
        {
            "event": "e",
            "token": "abc",
            "nested": {
                "password": "p",
                "keep": 1,
                "api_key": None,
                "secret": True,
                "deeper": [{"authorization": "Bearer x"}],
            },
        },
    )
    assert out["token"] == "[REDACTED]"  # noqa: S105
    assert out["nested"] == {
        "password": "[REDACTED]",
        "keep": 1,
        "api_key": None,
        "secret": True,
        "deeper": [{"authorization": "[REDACTED]"}],
    }
    # None / bool values are not redacted at the top level either
    assert R.redact_event_dict(None, "i", {"event": "e", "token": None, "password": False}) == {
        "event": "e",
        "token": None,
        "password": False,
    }


# ---------------------------------------------------------------------------
# T-18  performance and ReDoS corpus
# ---------------------------------------------------------------------------

_FRAGMENTS = [
    "a.",
    "a-",
    "a+",
    "1:",
    "x=",
    "a://",
    "bot1",
    "eyJa.",
    "-----BEGIN ",
    "api-key:",
    "authorization: ",
    "a.b.",
    "://",
    "a:",
    ":",
    "@",
    "%3D",
    "bearer ",
    "Bearer a",
    "123456:",
    "123456789012:",
    "key=",
    "api_key",
    "token",
    "sign",
    "a://:",
    "a://a:",
    "http://u:",
    f"{_DASHES}BEGIN {_PK}{_DASHES}",
    "X-MBX-APIKEY ",
    "authorization=  ",
    "api-key   ",
    "a.a.a.",
    "1234567:aaaa",
    "bot123456:",
    "eyJaaaaa.aaaaa.",
    "secret=",
    "-eyJ",
    "eyJ-",
    "_-eyJ",
    ": key=",
    "key=a&",
    ": x-api-key=",
    "Token=: ",
    "apiSecret: ",
    "'password': 'x', ",
    "password=abc",
    "x-admin-key: ",
    "kc-api-passphrase=",
    "a=",
    "&",
    "%",
    "[",
    "]",
]


def _best(fn: Any, number: int, repeat: int = 5) -> float:
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def _clean_event() -> dict[str, Any]:
    return {
        "event": "engine.bar_processed",
        "timestamp": _TS,
        "level": "info",
        "logger": "trading.engine",
        "service": "trading-bot",
        "env": "production",
        "run_id": "run_20260227T143022Z_a3f7c1b2",
        "symbol": "BTC/USDT",
        "price": Decimal("65000.12"),
        "qty": 0.5,
        "count": 12,
        "side": "buy",
        "ts": datetime(2026, 1, 1, tzinfo=UTC),
        "ok": True,
    }


_BENCH = r"""
import json, timeit, traceback
from datetime import UTC, datetime
from decimal import Decimal
from common import redaction as R
from common.config import build_renderer

def best(fn, number, repeat=5):
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number

for i in range(5):
    R.register_secret("CF36SENTINEL-perf-secret-%d-0123456789abcdef" % i, name="p")
event = {
    "event": "engine.bar_processed", "timestamp": "2026-01-15T12:00:00.000000Z", "level": "info",
    "logger": "trading.engine", "service": "trading-bot", "env": "production",
    "run_id": "run_20260227T143022Z_a3f7c1b2", "symbol": "BTC/USDT", "price": Decimal("65000.12"),
    "qty": 0.5, "count": 12, "side": "buy", "ts": datetime(2026, 1, 1, tzinfo=UTC), "ok": True,
}
line = build_renderer(True)(None, "info", dict(event))
walker = best(lambda: R.redact_event_dict(None, "info", event), 5000)
final = best(lambda: R._MATCHER.sub_final(line), 20000)

def deep(n):
    if n == 0:
        raise RuntimeError(
            "GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&"
            "signature=CF36SENTINELSIG0123456789 503 " + "x" * 300
        )
    deep(n - 1)

try:
    deep(25)
except RuntimeError as exc:
    tb = "".join(traceback.format_exception(exc)) * 3
tb = tb[:5120] if len(tb) >= 5120 else tb + "x" * (5120 - len(tb))
ev_tb = dict(event, exception=tb)
tbt = best(lambda: R.redact_event_dict(None, "error", ev_tb), 100)
print(json.dumps({"walker": walker, "final": final, "traceback_5kb": tbt, "tb_len": len(tb)}))
"""


def _bench() -> dict[str, float]:
    """Time the hot path in a clean subprocess.

    The project's pytest ``addopts`` enable ``--cov``; coverage's tracer inflates
    pure-Python code ~8x, which would make a microsecond budget meaningless.  The
    subprocess environment is scrubbed of ``COV_CORE_*`` / ``COVERAGE_*`` so it
    runs untraced.
    """
    import os
    import subprocess
    import sys

    import common as _common

    pythonpath = str(Path(_common.__file__).resolve().parent.parent)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE", "COVERAGE"))}
    env["PYTHONPATH"] = pythonpath
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _BENCH],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return {k: float(v) for k, v in json.loads(proc.stdout.strip().splitlines()[-1]).items()}


def test_t18_clean_event_overhead_is_within_budget() -> None:
    timings = _bench()
    walker, final = timings["walker"], timings["final"]
    assert walker + final <= 50e-6, f"walker {walker * 1e6:.1f}us + final pass {final * 1e6:.1f}us"
    # identity: a clean event comes back with the very same value objects
    for i in range(5):
        R.register_secret(f"{S}-perf-secret-{i}-0123456789abcdef", name="p")
    event = _clean_event()
    out = R.redact_event_dict(None, "info", event)
    assert all(out[k] is event[k] for k in event), (
        "a clean event must return the same value objects"
    )
    line = build_renderer(True)(None, "info", dict(event))
    assert R.RedactingRenderer(build_renderer(True))(None, "info", dict(event)) == line


def test_t18_traceback_budget() -> None:
    timings = _bench()
    assert timings["tb_len"] == 5120
    assert timings["traceback_5kb"] <= 5e-3, f"{timings['traceback_5kb'] * 1e3:.2f} ms"


def test_t18_near_miss_corpus_stays_fast_after_the_cap() -> None:
    assert len(_FRAGMENTS) >= 30
    worst = 0.0
    for frag in _FRAGMENTS:
        s = frag * (1_048_576 // len(frag))
        t0 = time.perf_counter()
        R.redact_text(s)
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        assert dt <= 0.15, f"{frag!r}: {dt * 1e3:.0f} ms"
        t0 = time.perf_counter()
        R.redact_text(s, head_tail=True)
        assert time.perf_counter() - t0 <= 0.30, f"{frag!r} head_tail"
    assert worst > 0


def test_t18_mixed_fragment_soup_stays_fast() -> None:
    rnd = random.Random(1337)  # noqa: S311
    s = "".join(rnd.choice(_FRAGMENTS) for _ in range(60_000))[: R.LIMIT + R.MARGIN]
    t0 = time.perf_counter()
    R.redact_text(s)
    assert time.perf_counter() - t0 <= 0.15


# ---------------------------------------------------------------------------
# T-34  non-disclosure of the registry
# ---------------------------------------------------------------------------


def test_t34_registry_is_opaque() -> None:
    canaries = [f"{S}/canary+one-0123456789", f"{S}.canary(two)[x]-0123456789"]
    for c in canaries:
        R.register_secret(c, name="c")
    matcher = R._MATCHER
    shown = [repr(matcher), str(matcher), repr(vars(R)), str(vars(R)), repr(structlog.get_config())]
    for text in shown:
        for c in canaries:
            assert c not in text and re.escape(c) not in text, "registry leaked a canary"
        assert S not in text
    with pytest.raises(TypeError):
        pickle.dumps(matcher)
    with pytest.raises(TypeError):
        matcher.__reduce__()
    with pytest.raises(TypeError):
        matcher.__getstate__()
    assert not hasattr(matcher, "__dict__")
    assert repr(matcher) == "<SecretMatcher>"
    assert R.registered_secret_count() >= 2
    assert not any(
        name for name in dir(R) if name.lower() in {"secrets", "get_secrets", "registry"}
    )


# ---------------------------------------------------------------------------
# T-36  prefilter completeness + gate soundness
# ---------------------------------------------------------------------------


def _core_ungated(s: str) -> str:
    """R-09 then EVERY rule unconditionally (no prefilter, no gates)."""
    s, _ = R._MATCHER.subn_all(s)
    for rule in R._RULES:
        s = rule.rx.sub(rule.repl, s)
    return s


def _positive_samples() -> list[str]:
    samples = [raw for raw, expected in _RULE_CASES if raw != expected]
    samples += [f"{n}=VAL0123456789" for n in _R08_NAMES]
    samples += [f"{n}%3DVAL0123456789" for n in _R08_NAMES]
    samples += [
        f"X-Admin-Key: {S}0123456789",
        f"x-live-confirm-token: {S}0123456789",
        f"KC-API-PASSPHRASE: {S}0123456789",
        f"{{'kc-api-key': '{S}0123456789'}}",
        f"{{'CB-ACCESS-SIGN': '{S}0123456789'}}",
        f"OK-ACCESS-PASSPHRASE: {S}0123456789",
        f"x-bapi-api-key: {S}0123456789",
        f"X-MBX-APIKEY: {S}0123456789",
        f"api-sign={S}0123456789",
        f"proxy-authorization: Basic {S}0123456789",
        f"Authorization: {S}0123456789",
        f"x-api-key: {S}0123456789",
    ]
    # cf36a-fix: R-11 positives and the T-01p text forms (``=``, quoted, JSON, repr, ``:``)
    samples += [raw for raw, expected in _R08_R11_CASES if raw != expected]
    for n in PARITY_NAMES:
        samples += [
            f"{n}={U}",
            f"{n}='{U}'",
            json.dumps({n: U}),
            repr({n: U}),
            f"{n}: {U}",
        ]
    return samples


def test_t36_every_positive_sample_hits_the_prefilter() -> None:
    samples = _positive_samples()
    assert len(samples) >= 70
    for s in samples:
        assert R._prefilter_hit(s.lower()), s
        assert R._PREFILTER.search(s), f"spec prefilter must hit too: {s}"
        assert R.redact_text(s) != s, s


#: CF36A-SEC-202 (C-1): a None/null/boolean is kept only when it is the WHOLE value.
_C1_NEGATIVE = [
    "require_api_auth=True but api_key_hash is empty",
    "Settings(require_api_auth=False, allow_credentials=True)",
    "api_key=null",
    "token=TRUE&x=1",
    "auth=None",
    "token: null",  # CR-FIX2-01: pins the ``\Z`` branch of the R-11 skip
]
_C1_POSITIVE = [f"token: true-{U}", f"token: null.{U}", f"api_key=false-{U}", f"api_key=true{U}"]


def _equivalence_corpus() -> list[str]:
    R.register_secret(f"{S}-exact-0123456789", name="x")
    corpus = _positive_samples() + [raw for raw, _ in _RULE_CASES] + _FRAGMENTS
    rnd = random.Random(20260924)  # noqa: S311
    pieces = [
        *_FRAGMENTS,
        "apiSecret",
        ": ",
        "Token",
        "b'",
        "%",
        "'",
        "api_key=",
        "VAL0123456789",
        " ",
        "&",
        "?",
        "https://",
        "u:pw@h",
        "Bearer ",
        JWT,
        TG_TOKEN,
        "authorization: ",
        "'",
        '"',
        "{",
        "}",
        "%3D",
        f"{S}-exact-0123456789",
        f"{_DASHES}BEGIN EC {_PK}{_DASHES}\nAB\n{_DASHES}END EC {_PK}{_DASHES}",
    ]
    corpus += _C1_POSITIVE + _C1_NEGATIVE
    for _ in range(4000):
        corpus.append("".join(rnd.choice(pieces) for _ in range(rnd.randint(1, 12))))
    assert len(corpus) >= 4000
    return corpus


def test_t36_gates_and_prefilter_never_change_the_output() -> None:
    for s in _equivalence_corpus():
        assert R._core(s) == _core_ungated(s), repr(s)


#: The original template string of every rule (``_Rule`` keeps only the compiled callable).
_TEMPLATES = {
    "R-01": "[REDACTED_PRIVATE_KEY]",
    "R-02": r"\1\2:[REDACTED]@",
    "R-03": r"bot\1:[REDACTED]",
    "R-04": "[REDACTED_TG_TOKEN]",
    "R-05": r"\1\2[REDACTED]",
    "R-06": "Bearer [REDACTED]",
    "R-07": "[REDACTED_JWT]",
    "R-08": r"\1\2\3[REDACTED]",
    "R-11": r"\1\2\3[REDACTED]",
}


def test_t36t_precompiled_templates_equal_re_expansion() -> None:
    """D-04: ``_compile_template`` must behave exactly like ``re``'s own template expansion."""
    assert set(_TEMPLATES) == {r.id for r in R._RULES}
    corpus = _equivalence_corpus()
    hits = 0
    for rule in R._RULES:
        for s in corpus:
            expected = rule.rx.sub(_TEMPLATES[rule.id], s)
            assert rule.rx.sub(rule.repl, s) == expected, (rule.id, s)
            hits += expected != s
    assert hits > 100, "vacuous: the corpus must exercise the replacements"
    # unmatched groups render as "" exactly like ``re``
    rx = re.compile(r"(a)|(b)")
    repl = R._compile_template(r"<\1|\2>")
    assert rx.sub(repl, "ab") == rx.sub(r"<\1|\2>", "ab") == "<a|><|b>"


def test_t36_header_gate_covers_every_header_name() -> None:
    names = [
        "authorization",
        "proxy-authorization",
        "x-mbx-apikey",
        "cb-access-key",
        "cb-access-sign",
        "cb-access-passphrase",
        "ok-access-key",
        "ok-access-sign",
        "ok-access-passphrase",
        "x-bapi-api-key",
        "x-bapi-sign",
        "kc-api-key",
        "kc-api-sign",
        "kc-api-passphrase",
        "api-key",
        "api-sign",
        "x-api-key",
        "x-admin-key",
        "x-live-confirm-token",
    ]
    rule = R._RULES[4]
    for name in names:
        text = f"{name}: VALUE0123456789"
        assert rule.applies(text.lower()), name
        assert rule.rx.search(text), name


# ---------------------------------------------------------------------------
# T-37  order and JSON validity
# ---------------------------------------------------------------------------


def test_t37_exact_match_runs_before_the_pattern_rules() -> None:
    secret = f"abc/key={S}TAIL0123456789"
    R.register_secret(secret, name="x")
    out = R.redact_text(f"error: {secret} (end)")
    assert out == "error: [REDACTED] (end)"
    assert S not in out and "abc/" not in out


def test_t37_existing_markers_are_not_rematched() -> None:
    assert R.redact_text("api_key=[REDACTED]") == "api_key=[REDACTED]"
    assert R.redact_text("Authorization: [REDACTED]") == "Authorization: [REDACTED]"
    once = R.redact_text(f"?api_key={S}KEY0123456789&x=1")
    assert R.redact_text(once) == once
    assert "]]" not in once


@pytest.mark.parametrize(
    "secret",
    [
        f'{S}-with"quote-0123456789',
        f"{S}-with\\backslash-0123456789",
        f"{S}-with,comma-0123456789",
        f"{S}-with:colon-0123456789",
        f'{S}-all"\\,:-0123456789',
    ],
)
def test_t37_rendered_json_lines_stay_valid(secret: str) -> None:
    R.register_secret(secret, name="x")
    renderer = R.RedactingRenderer(build_renderer(True))
    for field in (secret, f"prefix {secret} suffix", f"url?x={secret}&y=1"):
        ed = R.redact_event_dict(
            None, "info", {**_BASE, "event": "e", "detail": field, "nested": {"k": field}}
        )
        line = renderer(None, "info", ed)
        parsed = json.loads(line)
        assert S not in line and secret not in line
        assert "[REDACTED" in parsed["detail"]
    # the final-pass matcher never carries a quote / backslash variant, so a
    # final-pass substitution can never cut a JSON string boundary
    if '"' in secret or "\\" in secret:
        raw_line = json.dumps({"detail": secret})
        assert R.RedactingRenderer(lambda *_: raw_line)(None, "i", {}) == raw_line


# ---------------------------------------------------------------------------
# T-38  short and empty secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "a",
        "ab",
        "abc",
        "abcdefg",
        "12345678901",
        "changeme",
        "true",
        "None",
        "replace_me",
        "Password",
        "\t\n",
    ],
)
def test_t38_short_empty_and_trivial_values_are_never_registered(value: str) -> None:
    assert R.register_secret(value, name="x") == 0
    assert R.registered_secret_count() == 0
    text = f"the value {value} and abc ab a abcdefg changeme appears here"
    assert R.redact_text(text) == text
    event = {**_BASE, "event": "e", "detail": text}
    assert _run(_old_chain(True), event) == _run(_new_chain(True), event)


def test_t38_boundary_lengths() -> None:
    assert R.register_secret("abcdefgh", name="x") == 1  # 8 chars is the minimum
    R._reset_registry_for_tests()
    assert R.register_secret("123456789012", name="x") == 1  # 12 digits
    R._reset_registry_for_tests()
    assert R.register_secret("1234567", name="x") == 0


def test_t38_structured_positions_still_redact_short_secrets() -> None:
    assert R.redact_text("password=abc") == "password=[REDACTED]"
    assert (
        R.redact_text("postgresql+asyncpg://u:abc@h/db") == "postgresql+asyncpg://u:[REDACTED]@h/db"
    )
    assert R.redact_text("api_key=a") == "api_key=[REDACTED]"
    assert R.redact_text("X-API-Key: ab") == "X-API-Key: [REDACTED]"


def test_t38_skip_reasons_never_reveal_the_value() -> None:
    assert R.secret_skip_reason("abc") == "too_short"
    assert R.secret_skip_reason("changeme") == "trivial"
    assert R.secret_skip_reason("12345678901") == "digits_only"
    assert R.secret_skip_reason("") is None
    assert R.secret_skip_reason(None) is None
    assert R.secret_skip_reason("postgresql+asyncpg://u@h/db", "url") is None
    assert R.secret_skip_reason(f"{S}-registrable-0123") is None


def test_t38_env_fallback_registers_only_suffix_anchored_credentials() -> None:
    env = {
        "FOO_API_KEY": "x" * 20 + "1",
        "LOG_LEVEL": "DEBUG",
        "SOME_TOKEN": "false",
        "CACHE_KEY_TTL": "3600000000000",
        "DATABASE_URL": f"postgresql+asyncpg://u:pw-{S}-123456789@h/db",
        "KEYBOARD_LAYOUT": "qwertyuiop",
        "APP_PASSWORD_HINT": "a-long-hint-value-0123",
        "TELEGRAM_BOT_TOKEN": TG_TOKEN,
    }
    added = R.register_secrets_from_env(env)
    assert added >= 4
    assert S not in R.redact_text(f"pw-{S}-123456789")
    assert S not in R.redact_text(TG_SECRET)
    assert R.redact_text("LOG_LEVEL DEBUG qwertyuiop a-long-hint-value-0123 3600000000000") == (
        "LOG_LEVEL DEBUG qwertyuiop a-long-hint-value-0123 3600000000000"
    )
    assert "x" * 20 + "1" not in R.redact_text("see " + "x" * 20 + "1")


# ---------------------------------------------------------------------------
# T-18p / T-18u / T-18g  ReDoS regression (CF36A-SEC-02, ARB-CF36A-08/09)
# ---------------------------------------------------------------------------

_PINNED_WORST = [
    "-eyJ",
    "eyJ-",
    "_-eyJ",
    "eyJ_",
    "a-eyJ",
    "eyJ1-",
    "-eyJ_",
    ": key=",
    "=: key",
    ": sign=",
    ": auth=",
    "Token=: ",
    "key=a&",
    ": x-api-key=",
    "key%3D: ",
    "key=: ",
    "'password': 'x', ",
    "apiSecret: x ",
    # cf36a-fix2: unbounded possessive _QVAL (CR-201) and the C-1 whole-value skip
    "key='",
    "api_key=b'",
    "'token': '",
    "key='[REDACTED]' ",
    "token: true-",
    "api_key=null&",
]


def _min_ms(fn: Any, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


@pytest.mark.parametrize("frag", _PINNED_WORST, ids=repr)
def test_t18p_pinned_worst_cases_stay_linear(frag: str) -> None:
    """Pre-fix: ``-eyJ`` 1.9 s; ``: key=`` 105-153 ms (Match.expand per match)."""
    s = (frag * (R.LIMIT // len(frag) + 2))[: R.LIMIT + R.MARGIN]
    assert _min_ms(lambda: R.redact_text(s)) <= 150, frag


def test_t18p_head_tail_on_a_160kb_jwt_lookalike_run() -> None:
    s = "-eyJ" * 40_000
    assert len(s) == 160_000
    assert _min_ms(lambda: R.redact_text(s, head_tail=True)) <= 300


def test_t18u_the_uvicorn_probe_message_is_fast() -> None:
    """The exact access message of the real-server probe (pre-fix 1314 ms)."""
    probe = '127.0.0.1:54321 - "GET /' + "-eyJ" * 15000 + ' HTTP/1.1" 200'
    assert _min_ms(lambda: R.redact_text(probe)) <= 150


_SWEEP = r"""
import itertools, json, random, time
from common import redaction as R

TOK = ["eyJ", "-", ".", "a", "1", ":", "=", "/", "@", "bot", "key", "://", " ", "'", "%3D",
       "bearer ", "[", "_", "123456:", "x-api-key", "-----BEGIN ", "PRIVATE KEY-----", "\n",
       '"', "\t",
       "api_key", "Token", "password", "sign", "aB", "b'", '": "', "secret", "%", "auth"]
assert len(TOK) == 35

def timed(frag, size):
    s = (frag * (size // len(frag) + 2))[:size]
    t0 = time.perf_counter()
    R.redact_text(s)
    return time.perf_counter() - t0

frags = {a + b for a in TOK for b in TOK} | set(TOK)
assert len(frags) >= 1200
rnd = random.Random(20261012)
frags |= {rnd.choice(TOK) + rnd.choice(TOK) + rnd.choice(TOK) for _ in range(3000)}
screen = sorted(((timed(f, 8192), f) for f in frags), reverse=True)
worst = []
for _, f in screen[:25]:
    worst.append((min(timed(f, R.LIMIT + R.MARGIN) for _ in range(3)), f))
worst.sort(reverse=True)
print(json.dumps({"n": len(frags), "worst": [[round(t * 1e3, 1), f] for t, f in worst]}))
"""


def test_t18g_generated_sweep_has_no_slow_fragment() -> None:
    """All 1-2-token products of 35 tokens + 3,000 seeded 3-token products (8 KB screen),
    then the 25 slowest re-timed at the cap window.  Runs untraced (coverage inflates ~8x)."""
    import common as _common

    pythonpath = str(Path(_common.__file__).resolve().parent.parent)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE", "COVERAGE"))}
    env["PYTHONPATH"] = pythonpath
    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SWEEP],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["n"] >= 1260
    assert len(result["worst"]) == 25
    for ms, frag in result["worst"]:
        assert ms <= 150, f"{frag!r}: {ms} ms"
    assert time.perf_counter() - t0 <= 90


# ---------------------------------------------------------------------------
# CR-04  processor-key carve-out
# ---------------------------------------------------------------------------


def test_cr04_caller_supplied_values_under_processor_keys_get_the_full_rules() -> None:
    dsn = "https://u:" + "pw1234567" + "@h/?api_key=" + "ABCDEF1234"
    out = R.redact_event_dict(None, "info", {"event": "x", "env": dsn, "service": dsn})
    for key in ("env", "service"):
        assert "pw1234567" not in out[key] and "ABCDEF1234" not in out[key], key
        assert "[REDACTED" in out[key]
    # what the standard processors emit keeps object identity (byte-identity, T-15)
    ts = "2026-10-12T11:33:00.123456Z"
    event = {"event": "x", "timestamp": ts, "level": "info", "logger": "a.b", "service": "svc-1"}
    redacted = R.redact_event_dict(None, "info", event)
    assert all(redacted[k] is event[k] for k in event)


# ---------------------------------------------------------------------------
# SEC-05 / CR-06(d)  straddle-safe size cap
# ---------------------------------------------------------------------------


def _register_many(n: int, tag: str) -> list[str]:
    values = [f"{S}{tag}{i:03d}" + "Z" * (204 - len(S) - len(tag) - 3) for i in range(n)]
    assert {len(v) for v in values} == {204}
    for v in values:
        assert R.register_secret(v, name="big") > 0
    return values


def test_sec05_a_secret_crossing_the_cut_is_wholly_redacted() -> None:
    sec = f"{S}-straddle-" + "Q" * 80
    R.register_secret(sec, name="s")
    h = R.LIMIT // 2
    # (i) crossing index LIMIT (plain cap) and LIMIT // 2 (head)
    for start in (R.LIMIT - 40, h - 40):
        s = "x" * start + sec + "y" * (R.LIMIT + 500)
        out = R.redact_text(s, head_tail=start < R.LIMIT // 2 + 1)
        assert sec[:12] not in out and sec[-12:] not in out, start
    # ... and the tail window's start, from the other side
    n = R.LIMIT + 5_000
    s = "x" * (n - h - 40 - len(sec)) + sec + "y" * (h + 40)
    out = R.redact_text(s, head_tail=True)
    assert sec[:12] not in out and sec[-12:] not in out
    assert "[REDACTED]" in out or "TRUNCATED" in out


def test_sec05_shrinking_replacements_do_not_pull_the_window_edge_into_the_output() -> None:
    """(ii) The reproducer: 30 x 204-char registered secrets, then one straddling LIMIT+MARGIN."""
    big = _register_many(30, "A")
    straddle = "STRADDLESECRET" + "Q" * 186
    R.register_secret(straddle, name="s")
    pre = "".join(big) + " "
    fill = "x" * (R.LIMIT + R.MARGIN - len(pre) - 20)
    s = pre + fill + straddle + " tail" + "y" * 100
    assert len(s) > R.LIMIT + R.MARGIN
    out = R.redact_text(s)
    assert straddle[:12] not in out, "a 12+ char prefix of a registered secret survived"
    assert S not in out
    # (v) the plain-cap output is bounded
    marker = f"...[TRUNCATED len={len(s)}]"
    assert len(out) <= R.LIMIT + len(marker) and out.endswith(marker)


def test_sec05_mirror_for_the_tail_window() -> None:
    """(iii) A registered secret whose LAST 20 characters open the tail window."""
    big = _register_many(30, "B")
    straddle = "STRADDLESECRET" + "Q" * 186
    R.register_secret(straddle, name="s")
    h = R.LIMIT // 2
    joined = "".join(big)
    after = "x" * (h + R.MARGIN - 20 - len(joined)) + joined
    s = "H" * R.LIMIT + straddle + after
    out = R.redact_text(s, head_tail=True)
    assert straddle[-12:] not in out, "a 12+ char suffix of a registered secret survived"
    assert S not in out


def test_sec05_a_tail_window_shrinking_by_more_than_half_the_limit_yields_an_empty_tail() -> None:
    """(iv) ``tr[-0:]`` would be the WHOLE string; the tail must be empty instead."""
    big = _register_many(180, "C")
    s = "H" * R.LIMIT + "".join(big)
    out = R.redact_text(s, head_tail=True)
    assert out.endswith(f"\n...[TRUNCATED len={len(s)}]...\n")
    assert S not in out


def test_sec05_a_tail_window_shrinking_by_exactly_half_the_limit_yields_an_empty_tail() -> None:
    """Boundary of (iv): ``keep == 0`` must not become ``tr[-0:]`` (the whole window)."""
    big = _register_many(168, "D")  # 168 x 194 = 32592 characters of shrinkage
    last = f"{S}E000" + "Z" * (186 - len(S) - 4)  # 186 -> 10: another 176 = exactly LIMIT // 2
    assert len(last) == 186
    R.register_secret(last, name="last")
    s = "H" * R.LIMIT + "".join(big) + last
    out = R.redact_text(s, head_tail=True)
    assert out.endswith(f"\n...[TRUNCATED len={len(s)}]...\n")
    assert S not in out


# ---------------------------------------------------------------------------
# SEC-06  registry variants for URL contexts
# ---------------------------------------------------------------------------


def test_sec06_yarl_quoting_forms_of_a_registered_secret_are_redacted() -> None:
    v = "Ab/c+d=e f?g:h@i!j0123"
    assert R.register_secret(v, name="y") > 0
    forms = {
        "query": str(yarl.URL("https://h/p").with_query({"q": v})),
        "path": str(yarl.URL("https://h/p") / v),
        "lowercase-hex": "x "
        + re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), quote(v, safe="")),
        "default-quote": "x " + quote(v),
        "quote_plus": "x " + quote_plus(v),
    }
    for name, text in forms.items():
        assert "%" in text or "+" in text, f"premise ({name})"
        out = R.redact_text(text)
        assert "[REDACTED]" in out, name
        assert "Ab%2Fc" not in out and "j0123" not in out and "Ab/c" not in out, (name, out)


def test_sec06_the_variant_quoting_forms_equal_yarls_on_random_values() -> None:
    """Pin: a yarl upgrade that changes its quoting must fail loudly (verified on 1.22.0)."""
    rnd = random.Random(20261012)  # noqa: S311
    alphabet = [c for c in string.printable if c not in "\t\n\r\x0b\x0c"]
    base = yarl.URL("https://h/p")
    for _ in range(200):
        v = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(8, 24))).strip() or "Aaaaaaaa1"
        v = "A" + v  # yarl refuses a path segment that starts with "/"
        query = str(base.with_query({"q": v}))
        assert query == "https://h/p?q=" + quote_plus(v, safe="!$'()*,/:?@"), repr(v)
        path = str(base / v)
        assert path == "https://h/p/" + quote(v, safe="/:@!$'()*,;=+&"), repr(v)


# ---------------------------------------------------------------------------
# SEC-11 / CR-10  final pass on bytes output
# ---------------------------------------------------------------------------


def test_sec11_bytes_output_gets_the_final_exact_pass() -> None:
    secret = f"{S}-bytes-secret-0123456789"
    R.register_secret(secret, name="b")
    renderer = R.RedactingRenderer(lambda *_a: f'{{"event": "{secret}"}}'.encode())
    out = renderer(None, "info", {})
    assert isinstance(out, bytes)
    assert secret.encode() not in out and b"[REDACTED]" in out
    out_ba = R.RedactingRenderer(lambda *_a: bytearray(f"x {secret}".encode()))(None, "i", {})
    assert isinstance(out_ba, bytes) and secret.encode() not in out_ba


def test_sec11_invalid_utf8_fails_closed_and_is_counted() -> None:
    from common.metrics import metrics

    key = "log_redaction_failures_total"
    before = int(metrics.get_all()["counters"][key])
    out = R.RedactingRenderer(lambda *_a: b"\xff\xfe not utf-8")(None, "info", {})
    assert out == '{"event": "[REDACTION_FAILED]", "level": "error", "redaction_error": true}'
    assert int(metrics.get_all()["counters"][key]) == before + 1


# ---------------------------------------------------------------------------
# CR-06(e)  the defensive ``exc_info`` branch of the processor
# ---------------------------------------------------------------------------


def test_cr06e_exc_info_tuple_is_formatted_redacted_and_removed() -> None:
    secret = f"{S}-exc-info-secret-0123456789"
    R.register_secret(secret, name="e")
    try:
        raise RuntimeError(f"boom {secret}")
    except RuntimeError as exc:
        out = R.redact_event_dict(
            None, "error", {"event": "x", "exc_info": (type(exc), exc, exc.__traceback__)}
        )
    assert "exc_info" not in out
    assert secret not in out["exception"] and "[REDACTED]" in out["exception"]


# ---------------------------------------------------------------------------
# _CRED_NAMES: one representative per alternative (mutation guard for the shared grammar)
# ---------------------------------------------------------------------------


def _top_level_alternatives(pattern: str) -> list[str]:
    alts: list[str] = []
    cur = ""
    depth = 0
    in_class = False
    for ch in pattern:
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif not in_class and ch == "(":
            depth += 1
        elif not in_class and ch == ")":
            depth -= 1
        if ch == "|" and depth == 0 and not in_class:
            alts.append(cur)
            cur = ""
        else:
            cur += ch
    alts.append(cur)
    return alts


_KEY_PREFIXES = (
    "secret",
    "access",
    "private",
    "signing",
    "admin",
    "master",
    "encryption",
    "hmac",
    "session",
)
_TOKEN_PREFIXES = (
    "access",
    "refresh",
    "auth",
    "bot",
    "bearer",
    "session",
    "id",
    "confirm",
    "csrf",
)
_CRED_REPRESENTATIVES: dict[str, list[str]] = {
    "api[_-]?key": ["api_key", "api-key", "apikey"],
    "apikey": ["apikey", "APIKEY"],
    "api[_-]?secret": ["api_secret", "apisecret"],
    "api[_-]?sign": ["api_sign", "api-sign"],
    "api[_-]?passphrase": ["api_passphrase", "apipassphrase"],
    "(?:secret|access|private|signing|admin|master|encryption|hmac|session)[_-]?key": [
        f"{p}{sep}key" for p in _KEY_PREFIXES for sep in ("_", "-", "")
    ],
    "access[_-]?key[_-]?id": ["access_key_id", "accesskeyid", "AccessKeyId"],
    "access[_-]?secret": ["access_secret", "accesssecret"],
    "client[_-]?secret": ["client_secret", "clientsecret"],
    "(?:access|refresh|auth|bot|bearer|session|id|confirm|csrf)[_-]?token": [
        f"{p}{sep}token" for p in _TOKEN_PREFIXES for sep in ("_", "")
    ],
    "secret": ["secret"],
    "token": ["token"],
    "pass[_-]?word": ["password", "pass_word", "passWord"],
    "passwd": ["passwd"],
    "pwd": ["pwd"],
    "pass[_-]?phrase": ["passphrase", "pass_phrase", "passPhrase"],
    "signature": ["signature"],
    "sign": ["sign"],
    "sig": ["sig"],
    "authorization": ["authorization"],
    "auth": ["auth"],
    "cookie": ["cookie"],
    "dsn": ["dsn"],
    "credentials?": ["credential", "credentials"],
}


def test_cred_names_every_alternative_is_exercised_by_name() -> None:
    """Adding, renaming or dropping a ``_CRED_NAMES`` alternative must touch this test."""
    assert set(_top_level_alternatives(R._CRED_NAMES)) == set(_CRED_REPRESENTATIVES)
    for samples in _CRED_REPRESENTATIVES.values():
        for name in samples:
            assert R.is_sensitive_key(name), name
            assert R.redact_text(f"{name}=VAL0123456789") == f"{name}=[REDACTED]", name
            assert R.redact_text(f'{{"{name}": "VAL0123456789"}}') == (
                f'{{"{name}": "[REDACTED]"}}'
            ), name


def test_r10_key_match_is_anchored_with_backslash_z_not_dollar() -> None:
    assert R.is_sensitive_key("password")
    assert not R.is_sensitive_key("password\n")
    assert not R.is_sensitive_key("passwords")


def test_r08_r11_quoted_value_bound_covers_a_long_ordinary_secret() -> None:
    long_value = "A1b2" * 250  # 1000 characters
    assert R.redact_text(f"api_key='{long_value}' x") == "api_key='[REDACTED]' x"
    assert R.redact_text(f'{{"token": "{long_value}"}}') == '{"token": "[REDACTED]"}'


@pytest.mark.parametrize("n", [1025, 5000])
def test_r08_r11_quoted_value_longer_than_1024_is_wholly_redacted(n: int) -> None:
    """CF36A-CR-201 (O-1): no plaintext tail survives a long quoted value."""
    value = "A1b2" * (n // 4 + 1)
    assert R.redact_text(f"api_key='{value}' x") == "api_key='[REDACTED]' x"
    assert R.redact_text(f'{{"token": "{value}"}}') == '{"token": "[REDACTED]"}'


@pytest.mark.parametrize("text", _C1_NEGATIVE)
def test_c1_whole_value_none_null_boolean_is_kept(text: str) -> None:
    """CF36A-SEC-202: auth / CORS posture flags stay readable."""
    assert R.redact_text(text) == text


@pytest.mark.parametrize("text", _C1_POSITIVE)
def test_c1_a_value_that_merely_starts_with_a_literal_is_redacted(text: str) -> None:
    """CF36A-SEC-202: the old ``\\b`` skip let ``true-<secret>`` through R-11."""
    out = R.redact_text(text)
    assert U not in out
    assert "[REDACTED" in out


def test_c3_a_short_value_does_not_register_through_its_encoded_variant() -> None:
    """CF36A-SEC-201: ``yes!!`` must not register as ``yes%21%21``."""
    assert R.register_secret("yes!!", name="t") == 0
    assert R.secret_variants("yes!!") == frozenset()
    assert R.secret_skip_reason("yes!!") == "too_short"
    assert R.register_secret("  sp  ", name="t") == 0
    assert R.secret_variants("  sp  ") == frozenset()
