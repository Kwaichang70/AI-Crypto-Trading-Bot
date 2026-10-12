"""
tests/unit/test_cf36_settings.py
--------------------------------
CF-36 cf36a: the secret registry wiring on ``api.config.Settings`` (F-16).

    T-19  register_settings_secrets covers every secret field (+ name meta-test)
    T-20  env fallback registers exactly the credential-named variables
    T-31  ValidationError / repr / model_dump never carry a secret
    T-38  the Settings WARNING names the field, never the value or its length
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError
from structlog.testing import capture_logs

from api.config import Settings, iter_settings_secrets, register_settings_secrets
from common import redaction as R

S = "CF36SENTINEL"
DB_PW = f"{S}-DbPw-0123456789"
REDIS_PW = f"{S}-RedisPw-0123456789"
TG_TOKEN = f"123456789:AA{S}TelegramSecret_0123456789-xyz"
TG_SECRET = TG_TOKEN.split(":", 1)[1]
PEM = (
    "-----BEGIN EC PRIVATE KEY-----\n"
    f"MHcCAQEEI{S}Line1Line1Line1Line1AAAA\nBBBBBBbcdefghij{S}Line2Line2Line2Line2\n"
    "-----END EC PRIVATE KEY-----"
)
HASH1 = hashlib.sha256(f"{S}-client-key-1".encode()).hexdigest()
HASH2 = hashlib.sha256(f"{S}-client-key-2".encode()).hexdigest()

_SECRET_FIELD_NAME = re.compile(
    r"(key|secret|token|password|passphrase|dsn|_url|_hash(_secondary)?)$"
)
#: Fields matching the name pattern that are consciously NOT secrets.  Starts empty.
_NAME_ALLOWLIST: frozenset[str] = frozenset()


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": f"postgresql+asyncpg://trading:{DB_PW}@db.internal:5432/tradingdb",
        "postgres_password": f"{S}-PgPw-0123456789",
        "exchange_api_key": f"{S}-exchange-key-0123456789",
        "exchange_api_secret": PEM,
        "exchange_api_passphrase": f"{S}-passphrase-0123456789",
        "live_trading_confirm_token": f"{S}-confirm-token-0123456789",
        "admin_api_key": f"{S}adminkey0123456789abcdef0123456789",
        "fred_api_key": f"{S}-fred-key-0123456789",
        "whale_alert_api_key": f"{S}-whale-key-0123456789",
        "telegram_bot_token": TG_TOKEN,
        "redis_url": f"redis://:{REDIS_PW}@redis:6379/0",
        "api_key_hash": HASH1,
        "api_key_hash_secondary": HASH2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


# ---------------------------------------------------------------------------
# T-19
# ---------------------------------------------------------------------------


def test_t19_every_secret_field_is_discovered_and_registered() -> None:
    settings = _settings()
    found = {name for name, _v, _k in iter_settings_secrets(settings)}
    expected = {
        "database_url",
        "postgres_password",
        "exchange_api_key",
        "exchange_api_secret",
        "exchange_api_passphrase",
        "live_trading_confirm_token",
        "admin_api_key",
        "fred_api_key",
        "whale_alert_api_key",
        "telegram_bot_token",
        "redis_url",
        "api_key_hash",
        "api_key_hash_secondary",
    }
    assert found == expected
    assert register_settings_secrets(settings) >= len(expected) - 1  # url kinds add 1 each

    for secret in (
        DB_PW,
        f"{S}-PgPw-0123456789",
        f"{S}-exchange-key-0123456789",
        f"{S}-passphrase-0123456789",
        f"{S}-confirm-token-0123456789",
        f"{S}adminkey0123456789abcdef0123456789",
        f"{S}-fred-key-0123456789",
        f"{S}-whale-key-0123456789",
        TG_TOKEN,
        TG_SECRET,
        REDIS_PW,
        HASH1,
        HASH2,
        f"MHcCAQEEI{S}Line1Line1Line1Line1AAAA" + f"BBBBBBbcdefghij{S}Line2Line2Line2Line2",
        f"MHcCAQEEI{S}Line1Line1Line1Line1AAAA",
        f"BBBBBBbcdefghij{S}Line2Line2Line2Line2",
    ):
        out = R.redact_text(f"leak: {secret} end")
        assert secret not in out, secret
        assert "[REDACTED" in out


def test_t19_database_url_registers_only_the_password() -> None:
    register_settings_secrets(_settings())
    text = "host db.internal:5432/tradingdb user trading"
    assert R.redact_text(text) == text


def test_t19_name_meta_test_every_secret_looking_field_is_registered_or_allowlisted() -> None:
    registered = {name for name, _v, _k in iter_settings_secrets(_settings())}
    matching = {n for n in Settings.model_fields if _SECRET_FIELD_NAME.search(n)}
    assert len(matching) >= 13, "meta-test is vacuous"
    missing = matching - registered - _NAME_ALLOWLIST
    assert not missing, (
        f"Settings field(s) {sorted(missing)} look like secrets but are not registered by "
        "register_settings_secrets; register them or add them to _NAME_ALLOWLIST"
    )


def test_t19_secretstr_typed_fields_are_all_registered() -> None:
    settings = _settings()
    registered = {name for name, _v, _k in iter_settings_secrets(settings)}
    for name in Settings.model_fields:
        if isinstance(getattr(settings, name, None), SecretStr):
            assert name in registered, name


def test_t19_logs_names_and_counts_only() -> None:
    with capture_logs() as logs:
        register_settings_secrets(_settings())
    (done,) = [e for e in logs if e["event"] == "log_redaction.secrets_registered"]
    assert done["fields"] == sorted(done["fields"]) and len(done["fields"]) == 13
    assert isinstance(done["variants"], int) and done["variants"] >= 13
    assert S not in repr(logs)
    assert not [e for e in logs if e["event"] == "log_redaction.secret_not_registered"]


def test_t19_default_settings_register_nothing_and_do_not_warn() -> None:
    with capture_logs() as logs:
        assert register_settings_secrets(Settings(_env_file=None)) == 0
    assert [e["event"] for e in logs] == ["log_redaction.secrets_registered"]
    assert logs[0]["fields"] == [] and logs[0]["variants"] == 0


# ---------------------------------------------------------------------------
# T-20
# ---------------------------------------------------------------------------


def test_t20_env_fallback_registers_exactly_the_key_and_the_dsn_password() -> None:
    foo = "x" * 20 + "1"
    env = {
        "FOO_API_KEY": foo,
        "LOG_LEVEL": "DEBUG",
        "SOME_TOKEN": "false",
        "CACHE_KEY_TTL": "3600000000000",
        "DATABASE_URL": "postgresql+asyncpg://u:pw-123456789@h/db",
    }
    assert R.register_secrets_from_env(env) == 2
    assert R.registered_secret_count() == 2
    assert R.redact_text(f"k {foo} p pw-123456789") == "k [REDACTED] p [REDACTED]"
    for untouched in ("DEBUG", "false", "3600000000000", "postgresql+asyncpg://u", "@h/db"):
        assert R.redact_text(untouched) == untouched or untouched.startswith("postgresql")


def test_t20_configure_logging_registers_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests._cf36_logging import isolated_logging_state

    monkeypatch.setenv("CF36_TEST_API_KEY", f"{S}-env-api-key-0123456789")
    import common.logging as clog

    with isolated_logging_state():
        clog._LOGGING_CONFIGURED = False
        clog.configure_logging(log_level="INFO", json_output=True)
        assert S not in R.redact_text(f"{S}-env-api-key-0123456789")
        R._reset_registry_for_tests()
        clog._LOGGING_CONFIGURED = False
        clog.configure_logging(log_level="INFO", json_output=True, register_env_secrets=False)
        assert R.registered_secret_count() == 0


# ---------------------------------------------------------------------------
# T-31
# ---------------------------------------------------------------------------


#: Short enough that pydantic does not elide the middle of ``input_value`` (it
#: truncates long inputs), so the premise test below is meaningful.
_BAD_DSN = f"postgresql://u:{S}pw@h/x"


def test_t31_validation_error_never_echoes_secret_input() -> None:
    with pytest.raises(ValidationError) as info:
        Settings(
            _env_file=None,
            database_url=_BAD_DSN,
            admin_api_key=f"{S}short",
        )
    text = str(info.value)
    assert "database_url" in text and "admin_api_key" in text, "errors must still name the fields"
    assert S not in text and "input_value" not in text
    # NOTE: ``ValidationError.errors()`` still carries ``input`` (the flag only
    # affects str()/repr()).  Nothing in the app calls it on Settings errors;
    # the boot path prints str(exc) (uvicorn), which is what this test pins.


def test_t31_boot_path_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", _BAD_DSN)
    monkeypatch.setenv("ADMIN_API_KEY", f"{S}short")
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None)
    assert S not in str(info.value)


def test_t31_the_premise_without_the_flag_the_input_is_echoed() -> None:
    class Leaky(Settings):
        model_config = {**Settings.model_config, "hide_input_in_errors": False}  # noqa: RUF012

    with pytest.raises(ValidationError) as info:
        Leaky(_env_file=None, database_url=_BAD_DSN)
    assert S in str(info.value), "premise: pydantic echoes input_value by default"
    assert Settings.model_config["hide_input_in_errors"] is True


def test_t31_repr_and_dump_hide_telegram_token_and_redis_password() -> None:
    settings = _settings()
    assert isinstance(settings.telegram_bot_token, SecretStr)
    assert isinstance(settings.redis_url, SecretStr)
    for text in (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
    ):
        assert TG_TOKEN not in text and TG_SECRET not in text and REDIS_PW not in text
        assert DB_PW not in text and S not in text
    assert settings.telegram_bot_token.get_secret_value() == TG_TOKEN


# ---------------------------------------------------------------------------
# T-38  Settings warnings
# ---------------------------------------------------------------------------


def test_t38_skipped_settings_fields_warn_with_name_and_reason_only() -> None:
    short_values = {
        "fred_api_key": "abc",
        "postgres_password": "12345678901",
        "whale_alert_api_key": "changeme",
    }
    settings = _settings(**short_values)
    with capture_logs() as logs:
        register_settings_secrets(settings)
    warns = {e["field"]: e for e in logs if e["event"] == "log_redaction.secret_not_registered"}
    assert set(warns) == {"fred_api_key", "postgres_password", "whale_alert_api_key"}
    assert warns["fred_api_key"]["reason"] == "too_short"
    assert warns["postgres_password"]["reason"] == "digits_only"
    assert warns["whale_alert_api_key"]["reason"] == "trivial"
    for e in warns.values():
        assert e["log_level"] == "warning"
        assert set(e) == {"event", "log_level", "field", "reason"}, "no value, no length"
    blob = repr(logs)
    assert (
        "abc" not in blob.replace("fred_api_key", "")
        and "12345678901" not in blob
        and "changeme" not in blob
    )
    # the short values were NOT registered (no mangling of ordinary text)
    assert R.redact_text("abc 12345678901 changeme") == "abc 12345678901 changeme"


# ---------------------------------------------------------------------------
# T-31 (cf36a-fix): every BaseSettings subclass hides the input (CF36A-CR-03)
# ---------------------------------------------------------------------------


def test_t31_every_basesettings_subclass_hides_input() -> None:
    from pydantic_settings import BaseSettings

    import api.config
    import common.config  # noqa: F401  (import every module that defines settings)

    def walk(c: type) -> Any:
        for sub in c.__subclasses__():
            yield sub
            yield from walk(sub)

    classes = [
        c
        for c in walk(BaseSettings)
        if c.__module__.split(".")[0] in {"api", "common", "trading", "data"}
    ]
    assert api.config.Settings in classes
    assert len(classes) >= 2
    for c in classes:
        assert c.model_config.get("hide_input_in_errors") is True, c.__qualname__
