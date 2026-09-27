"""
tests/unit/test_wp17b_migrations_teardown_guard.py
-----------------------------------------------------
WP1.7b round 2 (WP17b-S-06): pure, no-DB unit tests for the destructive
migrations-teardown guard in ``tests/migrations/conftest.py``.

Deliberately hermetic -- every case here exercises
``destructive_teardown_allowed``/``is_scratch_db_name`` directly as pure
string-decision functions. No Postgres, no asyncpg connection, no
``MIGRATION_TEST_DATABASE_URL`` needed: these tests run unconditionally,
in every environment.
"""

from __future__ import annotations

from tests.migrations.conftest import (
    destructive_teardown_allowed,
    is_scratch_db_name,
)

_REAL_APP_DSN = "postgresql+asyncpg://app:secret@db-prod.internal:5432/trading_bot"
_SCRATCH_DSN_TEST = "postgresql+asyncpg://wp17a_test:pw@localhost:5432/wp17a_migration_test"
_SCRATCH_DSN_SCRATCH = "postgresql+asyncpg://wp17b:pw@localhost:5432/wp17b_scratch_1234_5"
_SCRATCH_DSN_TMP = "postgresql+asyncpg://u:pw@localhost:5432/tmp_migrations_db"
_SCRATCH_DSN_MIXED_CASE = "postgresql+asyncpg://u:pw@localhost:5432/WP17B_SCRATCH_DB"
_NON_SCRATCH_DSN = "postgresql+asyncpg://u:pw@localhost:5432/wp17b_db_1234_5"


class TestIsScratchDbName:
    def test_name_containing_test_is_scratch(self) -> None:
        assert is_scratch_db_name(_SCRATCH_DSN_TEST) is True

    def test_name_containing_scratch_is_scratch(self) -> None:
        assert is_scratch_db_name(_SCRATCH_DSN_SCRATCH) is True

    def test_name_containing_tmp_is_scratch(self) -> None:
        assert is_scratch_db_name(_SCRATCH_DSN_TMP) is True

    def test_match_is_case_insensitive(self) -> None:
        assert is_scratch_db_name(_SCRATCH_DSN_MIXED_CASE) is True

    def test_name_without_any_marker_is_not_scratch(self) -> None:
        assert is_scratch_db_name(_NON_SCRATCH_DSN) is False

    def test_real_app_dsn_is_not_scratch(self) -> None:
        assert is_scratch_db_name(_REAL_APP_DSN) is False


class TestDestructiveTeardownAllowed:
    def test_unset_migration_url_is_refused(self) -> None:
        allowed, reason = destructive_teardown_allowed(
            None, original_database_url=None, allow_destructive_env=None
        )
        assert allowed is False
        assert "unset" in reason

    def test_empty_migration_url_is_refused(self) -> None:
        allowed, _reason = destructive_teardown_allowed(
            "", original_database_url=None, allow_destructive_env=None
        )
        assert allowed is False

    def test_equal_to_original_database_url_is_refused_even_if_scratch_named(
        self,
    ) -> None:
        """WP17b-S-06 (a): if MIGRATION_TEST_DATABASE_URL is literally the
        same DSN as the app's own DATABASE_URL, refuse -- regardless of
        whether the name happens to look scratch-y."""
        allowed, reason = destructive_teardown_allowed(
            _SCRATCH_DSN_TEST,
            original_database_url=_SCRATCH_DSN_TEST,
            allow_destructive_env=None,
        )
        assert allowed is False
        assert "DATABASE_URL" in reason

    def test_non_scratch_name_is_refused_by_default(self) -> None:
        """WP17b-S-06 (b): a DB name that doesn't match the scratch
        pattern is refused when no override is set."""
        allowed, reason = destructive_teardown_allowed(
            _NON_SCRATCH_DSN,
            original_database_url=_REAL_APP_DSN,
            allow_destructive_env=None,
        )
        assert allowed is False
        assert "scratch" in reason

    def test_non_scratch_name_allowed_with_explicit_override(self) -> None:
        allowed, reason = destructive_teardown_allowed(
            _NON_SCRATCH_DSN,
            original_database_url=_REAL_APP_DSN,
            allow_destructive_env="1",
        )
        assert allowed is True
        assert "override" in reason

    def test_override_value_must_be_exactly_one(self) -> None:
        """Only the literal string "1" overrides -- "true"/"yes"/"0" must
        NOT silently enable the destructive path."""
        for bad_value in ("true", "yes", "0", "TRUE", ""):
            allowed, _reason = destructive_teardown_allowed(
                _NON_SCRATCH_DSN,
                original_database_url=_REAL_APP_DSN,
                allow_destructive_env=bad_value,
            )
            assert allowed is False, f"allow_destructive_env={bad_value!r} must not override"

    def test_scratch_name_different_from_original_is_allowed(self) -> None:
        allowed, reason = destructive_teardown_allowed(
            _SCRATCH_DSN_SCRATCH,
            original_database_url=_REAL_APP_DSN,
            allow_destructive_env=None,
        )
        assert allowed is True
        assert "matched" in reason

    def test_scratch_name_allowed_when_original_database_url_is_none(self) -> None:
        """A process that never had DATABASE_URL set at all (no .env,
        no prior test) must not spuriously refuse -- only an EQUALITY
        match refuses, never the mere absence of an original value."""
        allowed, _reason = destructive_teardown_allowed(
            _SCRATCH_DSN_TEST, original_database_url=None, allow_destructive_env=None
        )
        assert allowed is True

    def test_equality_check_takes_priority_over_override(self) -> None:
        """Even MIGRATION_TEST_ALLOW_DESTRUCTIVE=1 must not resurrect a
        MIGRATION_TEST_DATABASE_URL == DATABASE_URL refusal -- override
        only ever relaxes the NAME-pattern check (b), never check (a)."""
        allowed, reason = destructive_teardown_allowed(
            _REAL_APP_DSN,
            original_database_url=_REAL_APP_DSN,
            allow_destructive_env="1",
        )
        assert allowed is False
        assert "DATABASE_URL" in reason
