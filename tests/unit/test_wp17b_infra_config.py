"""
tests/unit/test_wp17b_infra_config.py
--------------------------------------
WP1.7b: static (no docker/caddy binary required) checks for the C21 infra
change -- the admin proxy route ordering in infra/Caddyfile, and the
fail-fast admin key wiring in infra/docker-compose.yml.

Mandatory per reports/vp2-wp1.7/synthesis-spec.md:
  - Sub-WP split (1.7b): "A Caddy `handle /api/admin/*` -> `ui:3000` block
    placed before `/api/*`." and "Compose `ADMIN_API_KEY` and
    `INTERNAL_ADMIN_API_KEY` with `:?` fail-fast."
  - Section 6 (1.7b tests): "A static check that `/api/admin/*` comes
    before `/api/*` in the Caddyfile." and "`docker compose config` goes
    in the runbook, because Docker isn't available in the sandbox."

These tests are pure file/text parsing (Caddyfile is not YAML/JSON, so no
`caddy validate` is attempted here -- the caddy binary is not installed in
this sandbox, see infra/CADDY-RUNBOOK.md and
reports/vp2-wp1.7/devops-design.md WP17-D-04). `docker compose config` IS
available in this sandbox (only the daemon is not running, which `config`
does not require) and was run manually as part of producing the WP1.7b
patch; it is not re-run here to keep this suite hermetic and independent
of the `docker` CLI being on PATH in every environment that runs pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CADDYFILE = _REPO_ROOT / "infra" / "Caddyfile"
_COMPOSE = _REPO_ROOT / "infra" / "docker-compose.yml"
_DOCKERFILE_API = _REPO_ROOT / "infra" / "Dockerfile.api"
_RUNBOOK = _REPO_ROOT / "infra" / "CADDY-RUNBOOK.md"


# ---------------------------------------------------------------------------
# Caddyfile: /api/admin/* must be ordered before the generic /api/* block
# ---------------------------------------------------------------------------


def _primary_site_block(caddyfile_text: str) -> str:
    """Extract the primary `{{TAILSCALE_FQDN}} {` ... `}` block only.

    Excludes the `{{TAILSCALE_FQDN}}:3001 { ... }` Grafana block, which also
    contains `reverse_proxy` directives but no `handle` blocks -- isolating
    the primary block keeps this test resilient to unrelated edits below it.
    """
    lines = caddyfile_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "{{TAILSCALE_FQDN}} {":
            start = i
            break
    assert start is not None, (
        "could not find the primary '{{TAILSCALE_FQDN}} {' site block in "
        "infra/Caddyfile -- has the site block header changed?"
    )

    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{")
        depth -= lines[i].count("}")
        if depth == 0 and i > start:
            end = i
            break
    assert end is not None, "unbalanced braces in the primary Caddyfile site block"
    return "\n".join(lines[start : end + 1])


@pytest.fixture(scope="module")
def caddyfile_text() -> str:
    assert _CADDYFILE.exists(), f"missing {_CADDYFILE}"
    return _CADDYFILE.read_text()


@pytest.fixture(scope="module")
def primary_block(caddyfile_text: str) -> str:
    return _primary_site_block(caddyfile_text)


def test_caddyfile_braces_are_balanced(caddyfile_text: str) -> None:
    assert caddyfile_text.count("{") == caddyfile_text.count("}")


def test_caddyfile_has_admin_and_generic_api_handle_blocks(primary_block: str) -> None:
    assert "handle /api/admin/* {" in primary_block
    assert "handle /api/* {" in primary_block
    assert "handle /api/auth/* {" in primary_block


def test_caddyfile_admin_block_ordered_before_generic_api_block(
    primary_block: str,
) -> None:
    """WP17-D-01 / C21: Caddy v2 evaluates `handle` in source order and stops
    at the first match, so `/api/admin/*` MUST be textually before the
    catch-all `/api/*` block or every admin call would be swallowed by the
    generic FastAPI proxy instead of reaching the Next.js admin routes that
    inject X-Admin-Key server-side."""
    admin_idx = primary_block.index("handle /api/admin/* {")
    generic_idx = primary_block.index("handle /api/* {")
    assert admin_idx < generic_idx, (
        "handle /api/admin/* must appear before the generic handle /api/* "
        "block in infra/Caddyfile (Caddy matches handle blocks in source "
        "order and stops at the first match)"
    )


def test_caddyfile_auth_block_also_ordered_before_generic_api_block(
    primary_block: str,
) -> None:
    """Pre-existing invariant (not new in WP1.7b): /api/auth/* must also
    precede the generic /api/* block, for the same reason."""
    auth_idx = primary_block.index("handle /api/auth/* {")
    generic_idx = primary_block.index("handle /api/* {")
    assert auth_idx < generic_idx


def test_caddyfile_admin_block_proxies_to_ui_not_api(primary_block: str) -> None:
    """The admin block must reverse_proxy to ui:3000 (the Next.js server
    routes that inject X-Admin-Key), never directly to api:8000 -- otherwise
    the admin key injection step is skipped entirely."""
    admin_start = primary_block.index("handle /api/admin/* {")
    generic_start = primary_block.index("handle /api/* {")
    admin_block = primary_block[admin_start:generic_start]
    assert "reverse_proxy ui:3000" in admin_block
    assert "reverse_proxy api:8000" not in admin_block


def test_caddyfile_generic_api_block_still_proxies_to_api(primary_block: str) -> None:
    generic_start = primary_block.index("handle /api/* {\n")
    # Slice a small, bounded window after the generic handle block opens.
    window = primary_block[generic_start : generic_start + 200]
    assert "reverse_proxy api:8000" in window


# ---------------------------------------------------------------------------
# docker-compose.yml: ADMIN_API_KEY / INTERNAL_ADMIN_API_KEY fail-fast wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose_doc() -> dict[str, object]:
    assert _COMPOSE.exists(), f"missing {_COMPOSE}"
    text = _COMPOSE.read_text()
    # docker-compose.yml uses ${VAR:?msg} interpolation syntax throughout
    # (e.g. POSTGRES_PASSWORD, NEXTAUTH_SECRET already did before WP1.7b).
    # These are plain YAML scalar strings -- safe_load does not attempt
    # shell interpolation, so this parses cleanly without docker/env vars.
    return yaml.safe_load(text)


def test_compose_parses_as_valid_yaml(compose_doc: dict[str, object]) -> None:
    assert isinstance(compose_doc, dict)
    assert "services" in compose_doc
    for required in ("api", "ui", "postgres", "redis"):
        assert required in compose_doc["services"], f"missing service: {required}"


def test_compose_api_admin_key_is_fail_fast(compose_doc: dict[str, object]) -> None:
    env = compose_doc["services"]["api"]["environment"]
    assert "ADMIN_API_KEY" in env
    value = env["ADMIN_API_KEY"]
    assert value.startswith("${ADMIN_API_KEY:?"), (
        f"api.environment.ADMIN_API_KEY must use ${{ADMIN_API_KEY:?...}} "
        f"fail-fast interpolation, got: {value!r}"
    )


def test_compose_ui_internal_admin_key_is_fail_fast(compose_doc: dict[str, object]) -> None:
    env = compose_doc["services"]["ui"]["environment"]
    assert "INTERNAL_ADMIN_API_KEY" in env
    value = env["INTERNAL_ADMIN_API_KEY"]
    assert value.startswith("${INTERNAL_ADMIN_API_KEY:?"), (
        f"ui.environment.INTERNAL_ADMIN_API_KEY must use "
        f"${{INTERNAL_ADMIN_API_KEY:?...}} fail-fast interpolation, got: "
        f"{value!r}"
    )


def test_compose_admin_keys_are_distinct_env_vars(compose_doc: dict[str, object]) -> None:
    """Per devops-design.md WP17-D-02: these are two independently-set
    operator variables (api's ADMIN_API_KEY and ui's INTERNAL_ADMIN_API_KEY,
    which the operator must set equal in .env), not one var reused via
    compose variable substitution -- matching the existing NEXTAUTH_SECRET
    duplication pattern already used for the ui service."""
    api_value = compose_doc["services"]["api"]["environment"]["ADMIN_API_KEY"]
    ui_value = compose_doc["services"]["ui"]["environment"]["INTERNAL_ADMIN_API_KEY"]
    assert "${ADMIN_API_KEY" in api_value
    assert "${INTERNAL_ADMIN_API_KEY" in ui_value


def test_compose_api_service_keeps_single_worker(compose_doc: dict[str, object]) -> None:
    """DC-3: the in-process _RUN_ENGINES registry and the kill-switch latch
    mirror (WP1.7a) assume a single API worker process. Compose must not
    override the Dockerfile.api entrypoint/command with a multi-worker
    invocation."""
    api_service = compose_doc["services"]["api"]
    assert "command" not in api_service, (
        "docker-compose.yml api service must not override the "
        "Dockerfile.api CMD (--workers 1) -- see DC-3 in "
        "reports/vp2-wp1.7/final-synthesis-1.7a.md"
    )
    assert _DOCKERFILE_API.exists(), f"missing {_DOCKERFILE_API}"
    dockerfile_text = _DOCKERFILE_API.read_text()
    assert '"--workers", "1"' in dockerfile_text or "--workers 1" in dockerfile_text


def test_compose_no_secrets_committed(compose_doc: dict[str, object]) -> None:
    """No literal secret values -- only ${VAR...} interpolation or empty/
    documented defaults -- for the two new admin key variables."""
    api_value = compose_doc["services"]["api"]["environment"]["ADMIN_API_KEY"]
    ui_value = compose_doc["services"]["ui"]["environment"]["INTERNAL_ADMIN_API_KEY"]
    for value in (api_value, ui_value):
        assert value.startswith("${"), f"admin key must be a ${{...}} placeholder, got {value!r}"


# ---------------------------------------------------------------------------
# Runbook: DC-6 (S-R3-02) kill-switch clear race note must be present (CF-B5)
# ---------------------------------------------------------------------------


def test_runbook_has_dc6_killswitch_clear_race_note() -> None:
    assert _RUNBOOK.exists(), f"missing {_RUNBOOK}"
    text = _RUNBOOK.read_text()
    assert "kill-switch" in text.lower()
    assert "latched" in text.lower()
    assert "GET /api/v1/emergency/kill-switch" in text
    assert "latched" in text and "false" in text


def test_runbook_has_dc5_rollback_steps() -> None:
    text = _RUNBOOK.read_text()
    assert "alembic downgrade 017" in text


def test_runbook_has_dc3_single_worker_note() -> None:
    text = _RUNBOOK.read_text()
    assert "--workers 1" in text or "single worker" in text.lower()


def test_runbook_has_dc4_migration_before_app_note() -> None:
    text = _RUNBOOK.read_text()
    assert "migration 018" in text.lower() or "018_kill_switch_latch_flatten" in text


# ---------------------------------------------------------------------------
# Runbook: S-08 (docker compose config --quiet) and S-09 (admin key sha256
# comparison), added in WP1.7b round 2 per critic-report-1.7b.md /
# security-report-1.7b.md
# ---------------------------------------------------------------------------


def test_runbook_has_s08_compose_config_quiet_step() -> None:
    """S-08: the runbook must tell the operator to use `--quiet` and must
    warn that the plain form leaks secrets."""
    text = _RUNBOOK.read_text()
    assert "docker compose" in text
    assert "config --quiet" in text, (
        "runbook must document 'docker compose config --quiet' as the "
        "pre-deploy validation step (S-08)"
    )
    lowered = text.lower()
    assert "secret" in lowered and "--quiet" in text, (
        "runbook must warn that plain `docker compose config` (without "
        "--quiet) prints interpolated secrets (S-08)"
    )


def test_runbook_s08_does_not_instruct_the_leaky_plain_form() -> None:
    """The mandatory pre-deploy step itself must invoke --quiet, not the
    plain form, so a copy-pasting operator doesn't leak secrets."""
    text = _RUNBOOK.read_text()
    idx = text.index("STEP 2a")
    section = text[idx : idx + 2000]
    assert "docker compose --env-file" in section
    assert "config --quiet" in section


def test_runbook_has_s09_admin_key_match_check() -> None:
    """S-09: the runbook must give a way to confirm ADMIN_API_KEY and
    INTERNAL_ADMIN_API_KEY are identical without printing either value --
    e.g. by comparing sha256 hashes -- and must not instruct printing the
    raw values for comparison."""
    text = _RUNBOOK.read_text()
    assert "ADMIN_API_KEY" in text and "INTERNAL_ADMIN_API_KEY" in text
    assert "sha256" in text.lower(), (
        "runbook must document a hash-based (non-printing) comparison of "
        "the two admin key variables (S-09)"
    )
    idx = text.index("### S-09")
    section = text[idx : idx + 1500]
    assert "sha256sum" in section
    # The comparison snippet itself must never echo the raw variable value.
    assert "echo \"$ADMIN_API_KEY\"" not in section
    assert "echo \"$INTERNAL_ADMIN_API_KEY\"" not in section


def test_runbook_s09_keeps_compose_vars_separate() -> None:
    """Per the coordinator's round-2 instruction: don't change compose to
    derive one admin key from the other -- keep them as two independently
    set variables and only add an operator-run check."""
    compose_text = _COMPOSE.read_text()
    assert "INTERNAL_ADMIN_API_KEY: ${INTERNAL_ADMIN_API_KEY:?" in compose_text, (
        "compose must still declare INTERNAL_ADMIN_API_KEY as its own "
        "independently-set variable, not derived from ADMIN_API_KEY"
    )
