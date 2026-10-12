"""
tests/unit/test_cf36_dsn.py
----------------------------
CF-36 cf36b, CF36A-SEC-07: a DSN password with a raw ``/``, ``#``, ``?``, ``@``,
``:`` or ``%``.

* ``Settings.assemble_database_url`` percent-encodes the password, so the DSN stays
  well-formed and SQLAlchemy / asyncpg connect with the ORIGINAL password;
* ``register_settings_secrets`` registers the password exactly as SQLAlchemy parses
  it, so a DSN supplied directly (``DATABASE_URL=...`` with a raw ``/``) is still
  covered by the exact-match redaction even though ``urlsplit`` disagrees.

All secrets are fake.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from api.config import Settings, register_settings_secrets
from common import redaction as R
from tests._cf36_logging import SENTINEL as S

#: every URL-significant character in one password
HARD_PW = f"{S}/p#w?x@y:z%41+&=_0123"


def _settings(**values: str) -> Settings:
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    R._reset_registry_for_tests()


def test_sec07_assembled_dsn_round_trips_the_original_password() -> None:
    settings = _settings(postgres_password=HARD_PW, postgres_host="db.internal")
    dsn = settings.database_url.get_secret_value()
    parsed = make_url(dsn)
    assert parsed.password == HARD_PW, "SQLAlchemy must decode back to the original password"
    assert (parsed.username, parsed.host, parsed.port, parsed.database) == (
        "trading",
        "db.internal",
        5432,
        "trading_bot",
    )
    assert "/p#w" not in dsn and "?x@" not in dsn, "no raw URL-significant characters"
    # CF36B-CR-05: make_url tolerates a raw "/" in a password, so check the assembled text too
    userinfo = dsn.split("://", 1)[1].rsplit("@", 1)[0]
    assert not any(ch in userinfo.split(":", 1)[1] for ch in "/#?@:"), (
        "raw URL delimiter in the password"
    )


def test_sec07_engine_is_built_with_the_original_password_without_a_parse_error() -> None:
    settings = _settings(postgres_password=HARD_PW)
    engine = create_async_engine(settings.database_url.get_secret_value())
    try:
        assert engine.url.password == HARD_PW
    finally:
        # create_async_engine does not connect; nothing to dispose
        pass


@pytest.mark.parametrize("pw", ["plainpassword123", "Abc-123_def.456", ""])
def test_sec07_plain_passwords_keep_the_same_dsn(pw: str) -> None:
    settings = _settings(postgres_password=pw)
    assert settings.database_url.get_secret_value() == (
        f"postgresql+asyncpg://trading:{pw}@localhost:5432/trading_bot"
    )


def test_sec07_an_explicit_database_url_is_left_untouched() -> None:
    explicit = "postgresql+asyncpg://u:Pw/Secret1234@h:5432/db"
    settings = _settings(database_url=explicit, postgres_password="ignored-because-explicit")  # noqa: S106
    assert settings.database_url.get_secret_value() == explicit


def test_sec07_direct_dsn_with_a_raw_slash_is_registered_as_sqlalchemy_parses_it() -> None:
    """``urlsplit`` stops the netloc at the first ``/`` and sees the password ``Pw`` only
    (too short to register); SQLAlchemy accepts the raw ``/`` and uses the full password."""
    password = f"{S}Pw/Secret1234"
    dsn = f"postgresql+asyncpg://u:{password}@h:5432/db"
    assert make_url(dsn).password == password  # premise
    settings = _settings(database_url=dsn)

    assert register_settings_secrets(settings) > 0

    leaked = f"asyncpg connect failed for password {password} on h"
    out = R.redact_text(leaked)
    assert password not in out and S not in out
    assert "[REDACTED" in out


def test_sec07_registration_adds_nothing_for_a_dsn_without_a_password() -> None:
    assert register_settings_secrets(_settings(database_url="postgresql+asyncpg://u@h/db")) == 0
    assert R.registered_secret_count() == 0


def test_sec07_helper_never_raises_and_never_echoes_an_unparsable_dsn() -> None:
    from api.config import _sqlalchemy_dsn_password

    for bad in ("not a dsn at all ://", "", "postgresql+asyncpg://u:x@[::1", "://"):
        assert _sqlalchemy_dsn_password(bad) is None
    assert _sqlalchemy_dsn_password("postgresql+asyncpg://u:pw1234567@h/db") == "pw1234567"


def test_sec07_a_short_dsn_password_is_still_not_registered() -> None:
    assert register_settings_secrets(_settings(database_url="postgresql+asyncpg://u:abc@h/db")) == 0
    assert R.registered_secret_count() == 0
