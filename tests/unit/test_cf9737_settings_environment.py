"""CF-97-37 patch 37-A: ``Settings.environment``, the production validator, compose pin."""

from __future__ import annotations

import ast
import hashlib
import secrets
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from api.config import Settings

ROOT = Path(__file__).resolve().parents[2]
GOOD_HASH = hashlib.sha256(secrets.token_bytes(32)).hexdigest()  # a valid digest, built per run
ADMIN = "CF9737-dummy-admin-key-0000000000000001"


def _prod(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "production",
        "require_api_auth": True,
        "api_key_hash": GOOD_HASH,
        "database_url": "postgresql+asyncpg://x:x@localhost:5432/x",
    }
    base.update(over)
    return Settings(_env_file=None, **base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENVIRONMENT", "REQUIRE_API_AUTH", "API_KEY_HASH", "DEBUG", "RATE_LIMIT_ENABLED"):
        monkeypatch.delenv(name, raising=False)


def test_code_default_is_development_and_auth_off() -> None:
    s = Settings(_env_file=None)
    assert s.environment == "development"
    assert s.require_api_auth is False


@pytest.mark.parametrize("env", ["development", "test"])
def test_non_production_is_not_validated(env: str) -> None:
    s = Settings(_env_file=None, environment=env, debug=True, rate_limit_enabled=False)
    assert s.environment == env and s.require_api_auth is False


@pytest.mark.parametrize("bad", ["prod", "staging", "Production ", ""])
def test_unknown_environment_is_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment=bad)


def test_environment_is_read_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert Settings(_env_file=None).environment == "test"


def test_valid_production_is_accepted() -> None:
    assert _prod().environment == "production"
    assert _prod(admin_api_key=ADMIN).environment == "production"


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"require_api_auth": False}, "require_api_auth must be true"),
        ({"api_key_hash": ""}, "api_key_hash must be set"),
        ({"rate_limit_enabled": False}, "rate_limit_enabled must be true"),
        ({"debug": True}, "debug must be false"),
        (
            {"admin_api_key": ADMIN, "api_key_hash": hashlib.sha256(ADMIN.encode()).hexdigest()},
            "admin_api_key must differ",
        ),
        (
            {
                "admin_api_key": ADMIN,
                "api_key_hash_secondary": hashlib.sha256(ADMIN.encode()).hexdigest(),
            },
            "admin_api_key must differ",
        ),
    ],
)
def test_production_rules_refuse(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError) as exc:
        _prod(**override)
    assert fragment in str(exc.value)


@pytest.mark.parametrize(
    "weak",
    [
        "changeme",
        "password",
        "secret",
        "test",
        "dev",
        "my-secret-api-key",
        "my-secret-key",
        "test-integration-api-key-2026",
    ],
)
@pytest.mark.parametrize("slot", ["api_key_hash", "api_key_hash_secondary"])
def test_weak_hash_denylist(weak: str, slot: str) -> None:
    digest = hashlib.sha256(weak.encode()).hexdigest()
    over: dict[str, Any] = {slot: digest}
    if slot == "api_key_hash_secondary":
        over["api_key_hash"] = GOOD_HASH
    with pytest.raises(ValidationError) as exc:
        _prod(**over)
    assert "is weak" in str(exc.value)


def test_weak_hash_is_fine_outside_production() -> None:
    digest = hashlib.sha256(b"secret").hexdigest()
    assert Settings(_env_file=None, environment="test", api_key_hash=digest).api_key_hash == digest


def test_all_failures_are_reported_and_no_value_is_echoed() -> None:
    weak = hashlib.sha256(b"changeme").hexdigest()
    with pytest.raises(ValidationError) as exc:
        _prod(
            require_api_auth=False,
            api_key_hash=weak,
            admin_api_key=ADMIN,
            debug=True,
            rate_limit_enabled=False,
        )
    text = str(exc.value) + repr(exc.value.errors(include_input=False))
    for fragment in ("require_api_auth", "is weak", "rate_limit_enabled", "debug"):
        assert fragment in text
    assert weak not in text
    assert ADMIN not in text
    assert "changeme" not in text


def test_compose_namespaces_the_host_variable() -> None:
    lines = (ROOT / "infra" / "docker-compose.yml").read_text().splitlines()
    env_lines = [ln.strip() for ln in lines if ln.strip().startswith("ENVIRONMENT:")]
    assert len(env_lines) == 1
    assert "${TRADING_ENVIRONMENT:-development}" in env_lines[0]
    assert "${ENVIRONMENT" not in "\n".join(lines)
    # inert: this patch does not change the auth default
    assert "REQUIRE_API_AUTH:    ${REQUIRE_API_AUTH:-false}" in "\n".join(lines)


# -- CR-37A-01: main.py pins (no lifespan, no Postgres) -------------------------


def _main_tree() -> ast.Module:
    return ast.parse((ROOT / "apps" / "api" / "main.py").read_text())


def _call_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def test_main_passes_settings_environment_to_configure_logging() -> None:
    calls = [
        n
        for n in ast.walk(_main_tree())
        if isinstance(n, ast.Call) and _call_name(n) == "configure_logging"
    ]
    assert len(calls) == 1
    env = [k for k in calls[0].keywords if k.arg == "environment"]
    assert [ast.unparse(k.value) for k in env] == ["settings.environment"]


def test_main_logs_auth_disabled_while_auth_is_off() -> None:
    hits = []
    for node in ast.walk(_main_tree()):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not settings.require_api_auth":
            for stmt in node.body:
                call = stmt.value if isinstance(stmt, ast.Expr) else None
                if (
                    isinstance(call, ast.Call)
                    and _call_name(call) == "log.warning"
                    and ast.unparse(call.args[0]) == "'api.auth_disabled'"
                ):
                    hits.append(call)
    assert len(hits) == 1


# -- SE-37A-02: nothing in apps/ may read ValidationError.errors() with the input ---


def _errors_calls_with_input(source: str) -> list[int]:
    """Line numbers of ``.errors(...)`` calls that do not pass ``include_input=False``."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _call_name(node).endswith(".errors"):
            ok = any(
                k.arg == "include_input"
                and isinstance(k.value, ast.Constant)
                and k.value.value is False
                for k in node.keywords
            )
            if not ok:
                bad.append(node.lineno)
    return bad


def test_errors_pin_flags_a_planted_offender() -> None:
    assert _errors_calls_with_input("exc.errors()\n") == [1]
    assert _errors_calls_with_input("exc.errors(include_input=True)\n") == [1]
    assert _errors_calls_with_input("exc.errors(include_input=False)\n") == []


def test_no_module_in_apps_reads_validation_errors_with_input() -> None:
    files = sorted((ROOT / "apps").rglob("*.py"))
    assert len(files) >= 30  # non-vacuous (47 at a7c6b9e)
    offenders = {
        f"{f.relative_to(ROOT)}:{line}"
        for f in files
        for line in _errors_calls_with_input(f.read_text())
    }
    assert not offenders, sorted(offenders)


# -- SE-37A-08: hash format is enforced by the field validator in production too ---


@pytest.mark.parametrize("slot", ["api_key_hash", "api_key_hash_secondary"])
@pytest.mark.parametrize(
    "bad",
    [GOOD_HASH[:63], GOOD_HASH + "a", "g" + GOOD_HASH[1:], " " + GOOD_HASH[1:]],
    ids=["short", "long", "not-hex", "space"],  # fixed ids: the values change every run
)
def test_production_refuses_a_malformed_hash(slot: str, bad: str) -> None:
    over: dict[str, Any] = {slot: bad}
    if slot == "api_key_hash_secondary":
        # another valid digest, never equal to GOOD_HASH
        over["api_key_hash"] = hashlib.sha256(GOOD_HASH.encode()).hexdigest()
    with pytest.raises(ValidationError) as exc:
        _prod(**over)
    assert bad not in str(exc.value)
    assert "api_key_hash must" in str(exc.value)


def test_production_accepts_an_uppercase_hash_and_lowercases_it() -> None:
    s = _prod(api_key_hash=GOOD_HASH.upper())
    assert s.api_key_hash == GOOD_HASH
