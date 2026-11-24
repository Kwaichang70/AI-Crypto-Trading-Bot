"""
tests/unit/test_cf9737_r2_wiring.py
------------------------------------
CF-97-37 patch 37-C1 (release R2, wiring): static pins, no docker binary, no caddy binary.

What R2 does: ``ui`` gets the raw ``API_KEY`` and ``api`` gets ``API_KEY_HASH`` as required
(``:?``) variables, Caddy gets a capped ``/bff/*`` block, ``api`` and ``ui`` get bounded Docker
logs, and a runbook orders the deploy so both variables are in ``.env`` before the new compose
file reaches the host (an unset required variable aborts compose for the whole project).

What R2 does NOT do, and these pins say so: ``REQUIRE_API_AUTH`` and the ``TRADING_ENVIRONMENT``
default are unchanged (that is patch 37-D; 37-D edits the two default pins below on purpose).

The Caddyfile syntax is pinned by text only: no ``caddy`` binary exists in the sandbox.

r2 (synthesis-37c1-r1.md section 5) added the runbook pins: the A/B pre-check form, the exact-echo
rule for the two key checks (each with a planted offender), the caddy-adapt count check that
replaced the HTTP 413 probe, the rollback that keeps the log rotation, and the section 3.3 rows of
the deploy runbook.
A new file on purpose: the existing infra suites are not edited by this change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _ROOT / "infra" / "docker-compose.yml"
_CADDYFILE = _ROOT / "infra" / "Caddyfile"
_DOCKERFILE_UI = _ROOT / "infra" / "Dockerfile.ui"
_ENV_EXAMPLE = _ROOT / ".env.example"
_RUNBOOK = _ROOT / "docs" / "runbooks" / "api-key-auth-switch.md"

_REQUIRED = r"^\$\{{{name}:\?[^}}\s][^}}]*\}}$"
_LOGGING = {"driver": "json-file", "options": {"max-size": "50m", "max-file": "5"}}


_DEPLOY_RUNBOOK = _ROOT / "docs" / "runbooks" / "deploy-fase1-minimumset.md"
_OK_ECHOES = {"echo OK", "echo MISMATCH"}


def _echo_offenders(text: str) -> list[str]:
    """Every ``echo ...`` in ``text`` that is not exactly ``echo OK`` or ``echo MISMATCH``."""
    found = [m.strip() for m in re.findall(r"\becho\b[^\n|&;']*", text)]
    return [e for e in found if e not in _OK_ECHOES]


def _ab_pre_check_ok(note: str) -> bool:
    """The REQUIRE_API_AUTH pre-check is the A/B form: all lines equal the ``=false`` lines."""
    has_a = re.search(
        r"^grep -c -E '\^REQUIRE_API_AUTH=' \"\$DEPLOY_DIR/\.env\"\s+# A2", note, re.M
    )
    has_b = re.search(
        r"^grep -c -E '\^REQUIRE_API_AUTH=false\$' \"\$DEPLOY_DIR/\.env\"\s+# B2", note, re.M
    )
    relaxed = [ln for ln in note.splitlines() if "^REQUIRE_API_AUTH=" in ln and "expected 0" in ln]
    return bool(has_a and has_b and not relaxed)


def _container_check_ok(text: str) -> bool:
    """The container check: hash over stdin, single-quoted script, prints only OK or MISMATCH."""
    m = re.search(r"\| \$DC exec -T ui sh -c '([^']*(?:\"[^']*)*)'", text)
    if not m or "$API_KEY" not in m.group(1) or "API_KEY_HASH" in m.group(1):
        return False
    stdin_feed = re.search(
        r"^grep '\^API_KEY_HASH=' \"\$DEPLOY_DIR/\.env\" \| cut -d= -f2- \\\n", text, re.M
    )
    return bool(stdin_feed) and not _echo_offenders(m.group(1)) and "read h;" in m.group(1)


@pytest.fixture(scope="module")
def compose_text() -> str:
    return _COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def services(compose_text: str) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(compose_text)["services"]


# ---------------------------------------------------------------------------
# compose: the two required variables, scoped to one service each
# ---------------------------------------------------------------------------


class TestComposeVariables:
    def test_ui_api_key_is_required_with_a_message(self, services: dict[str, Any]) -> None:
        value = services["ui"]["environment"]["API_KEY"]
        assert re.fullmatch(_REQUIRED.format(name="API_KEY"), value), value

    def test_api_key_hash_is_required_with_a_message(self, services: dict[str, Any]) -> None:
        value = services["api"]["environment"]["API_KEY_HASH"]
        assert re.fullmatch(_REQUIRED.format(name="API_KEY_HASH"), value), value

    def test_no_default_operator_on_either_variable(self, compose_text: str) -> None:
        # `:-`, `-`, `:=`, `=` and `:+` supply or alter a default; only `:?` may follow the name.
        found = list(re.finditer(r"\$\{(API_KEY_HASH|API_KEY)([^}]*)\}", compose_text))
        assert {m.group(1) for m in found} == {"API_KEY", "API_KEY_HASH"}
        for match in found:
            assert match.group(2).startswith(":?"), match.group(0)

    def test_the_raw_key_reaches_only_the_ui(self, services: dict[str, Any]) -> None:
        holders = [n for n, s in services.items() if "API_KEY" in (s.get("environment") or {})]
        assert holders == ["ui"]

    def test_the_hash_reaches_only_the_api(self, services: dict[str, Any]) -> None:
        holders = [n for n, s in services.items() if "API_KEY_HASH" in (s.get("environment") or {})]
        assert holders == ["api"]

    def test_the_key_is_runtime_only_not_a_build_arg(self, services: dict[str, Any]) -> None:
        args = services["ui"]["build"]["args"]
        assert not [name for name in args if "API_KEY" in name], sorted(args)

    def test_release_2_changes_no_auth_default(self, services: dict[str, Any]) -> None:
        # 37-D (the flip) edits these two assertions on purpose.
        env = services["api"]["environment"]
        assert env["REQUIRE_API_AUTH"] == "${REQUIRE_API_AUTH:-false}"
        assert env["ENVIRONMENT"] == "${TRADING_ENVIRONMENT:-development}"


class TestNoKeyInTheImageOrTheBrowser:
    def test_dockerfile_ui_never_declares_the_key(self) -> None:
        for line in _DOCKERFILE_UI.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0]
            if re.match(r"\s*(ARG|ENV)\b", code):
                assert "API_KEY" not in code, line

    def test_no_next_public_name_carries_a_key(self, compose_text: str) -> None:
        for path in (_COMPOSE, _DOCKERFILE_UI, _ENV_EXAMPLE):
            hits = re.findall(r"NEXT_PUBLIC_\w*KEY\w*", path.read_text(encoding="utf-8"))
            assert hits == [], (path.name, hits)


# ---------------------------------------------------------------------------
# compose: bounded Docker logs (SE-37A-03)
# ---------------------------------------------------------------------------


class TestLogRotation:
    @pytest.mark.parametrize("name", ["api", "ui"])
    def test_json_file_rotation_is_explicit(self, services: dict[str, Any], name: str) -> None:
        assert services[name]["logging"] == _LOGGING

    def test_other_services_are_not_touched(self, services: dict[str, Any]) -> None:
        assert [n for n, s in services.items() if "logging" in s and n not in ("api", "ui")] == []


# ---------------------------------------------------------------------------
# Caddy: the capped /bff/* block (text pins; Caddy itself was not run)
# ---------------------------------------------------------------------------

_BFF_HEADER = re.compile(r"^[ \t]*handle[ \t]+/bff/\*[ \t]*\{[ \t]*$", re.MULTILINE)


def _without_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _primary_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "{{TAILSCALE_FQDN}} {")
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start : i + 1])
    raise AssertionError("unbalanced braces in the primary Caddyfile site block")


def _body(text: str, header_start: int) -> str:
    open_idx = text.index("{", header_start)
    depth = 0
    for i in range(open_idx, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces in handle block")


@pytest.fixture(scope="module")
def primary() -> str:
    return _primary_block(_without_comments(_CADDYFILE.read_text(encoding="utf-8")))


class TestCaddyBffBlock:
    def test_block_exists_once_in_the_primary_site_block(self, primary: str) -> None:
        assert len(_BFF_HEADER.findall(primary)) == 1
        assert len(_BFF_HEADER.findall(_without_comments(_CADDYFILE.read_text()))) == 1

    def test_block_precedes_the_catch_all(self, primary: str) -> None:
        match = _BFF_HEADER.search(primary)
        assert match
        assert match.start() < primary.index("handle {")

    def test_existing_order_is_unchanged(self, primary: str) -> None:
        auth = primary.index("handle /api/auth/* {")
        admin = primary.index("handle /api/admin/* {")
        generic = primary.index("handle /api/* {")
        catch_all = primary.index("handle {")
        assert auth < admin < generic < catch_all

    def test_block_carries_the_1mb_cap_and_proxies_to_the_ui(self, primary: str) -> None:
        match = _BFF_HEADER.search(primary)
        assert match
        body = _body(primary, match.start())
        code = [ln.strip() for ln in body.splitlines() if ln.strip()]
        assert code == ["request_body {", "max_size 1MB", "}", "reverse_proxy ui:3000"], code

    def test_block_never_proxies_to_the_api(self, primary: str) -> None:
        match = _BFF_HEADER.search(primary)
        assert match
        assert "api:8000" not in _body(primary, match.start())

    def test_the_operator_path_and_the_metrics_block_are_kept(self, primary: str) -> None:
        generic = _body(primary, primary.index("handle /api/* {"))
        assert "reverse_proxy api:8000" in generic
        assert "handle /api/v1/metrics* {" in primary

    def test_the_catch_all_is_unchanged(self, primary: str) -> None:
        body = _body(primary, primary.index("handle {"))
        assert [ln.strip() for ln in body.splitlines() if ln.strip()] == ["reverse_proxy ui:3000"]


# ---------------------------------------------------------------------------
# .env.example: names without values
# ---------------------------------------------------------------------------


class TestEnvExample:
    @pytest.fixture(scope="class")
    def lines(self) -> list[str]:
        return _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    @pytest.mark.parametrize("name", ["API_KEY", "API_KEY_HASH"])
    def test_names_the_variable_once_with_an_empty_value(self, lines: list[str], name: str) -> None:
        hits = [ln for ln in lines if re.match(rf"{name}=", ln)]
        assert hits == [f"{name}="], hits

    def test_no_commented_example_value(self, lines: list[str]) -> None:
        for ln in lines:
            if ln.lstrip().startswith("#"):
                assert not re.search(r"\bAPI_KEY(_HASH)?\s*=\s*[0-9a-fA-F]{16,}", ln), ln


# ---------------------------------------------------------------------------
# runbook: order, key generation rule, checks, rollback, disclosures
# ---------------------------------------------------------------------------


class TestRunbook:
    @pytest.fixture(scope="class")
    def note(self) -> str:
        return _RUNBOOK.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def flat(self, note: str) -> str:
        return " ".join(note.split())

    def test_states_the_project_wide_abort_and_the_order(self, flat: str) -> None:
        assert "**every** `$DC` command fail" in flat
        assert "must be in `.env` before the new compose file arrives" in flat
        assert "**before any `down`, `up` or `stop`**" in flat
        assert "must print `exit=0`" in flat

    def test_steps_are_in_order(self, note: str) -> None:
        marks = [
            "### 3.1 Pre-checks",
            "### 3.2 Generate the key",
            "### 3.3 Check the pair",
            "### 3.4 Sync the files",
            '$DC config --quiet; echo "exit=$?"',
            "$DC up -d --build api ui",
            "### 3.5 Checks after the deploy",
            "## 4. Rollback",
        ]
        positions = [note.index(m) for m in marks]
        assert positions == sorted(positions), dict(zip(marks, positions, strict=True))

    def test_key_generation_rule(self, note: str, flat: str) -> None:
        assert "\nopenssl rand -hex 32 | env_set_key_pair\n" in note
        assert "printf '%s' \"$key\" | sha256sum | cut -d' ' -f1 | env_set API_KEY_HASH" in note
        assert "printf '%s\\n' \"$key\" | env_set API_KEY" in note
        assert "at least 32 random bytes (64 hex characters)" in flat
        assert "independent of `ADMIN_API_KEY`" in flat
        assert not re.search(r"API_KEY=[0-9a-fA-F]{16,}", note)

    def test_the_host_check_prints_exactly_ok_or_mismatch(self, note: str) -> None:
        block = re.search(r"^key_pair_check\(\) \{.*?^\}\n", note, re.S | re.M)
        assert block
        assert set(re.findall(r"echo (\w+)", block.group(0))) == {"OK", "MISMATCH"}
        assert _echo_offenders(block.group(0)) == []

    @pytest.mark.parametrize(
        "planted",
        [
            "x && echo OK || echo MISMATCH $k",  # security's mutant C7
            "x && echo $k || echo MISMATCH",
            'x && echo OK "$h" || echo MISMATCH',
            "x && echo OK || echo MISMATCH; echo done",
        ],
    )
    def test_the_exact_echo_rule_rejects_planted_offenders(self, planted: str) -> None:
        assert _echo_offenders(planted) != []

    def test_the_container_check_is_the_stdin_form_and_prints_exactly_ok_or_mismatch(
        self, note: str
    ) -> None:
        assert _container_check_ok(note)
        assert (
            'sh -c \'read h; [ "$(printf %s "$API_KEY" | sha256sum | cut -d" " -f1)" = "$h" ]'
            " && echo OK || echo MISMATCH'"
        ) in note

    @pytest.mark.parametrize(
        "planted",
        [
            # the script prints the hash too
            "grep '^API_KEY_HASH=' \"$DEPLOY_DIR/.env\" | cut -d= -f2- \\\n"
            '  | $DC exec -T ui sh -c \'read h; [ "$API_KEY" = "$h" ]'
            " && echo OK || echo MISMATCH $h'",
            # the hash goes into the argv instead of over stdin
            "grep '^API_KEY_HASH=' \"$DEPLOY_DIR/.env\" | cut -d= -f2- \\\n"
            "  | $DC exec -T ui sh -c 'h=$API_KEY_HASH; echo OK || echo MISMATCH'",
            # the expected hash is not fed from stdin at all
            "$DC exec -T ui sh -c 'read h; echo OK || echo MISMATCH'",
        ],
    )
    def test_the_container_check_rejects_planted_offenders(self, planted: str) -> None:
        assert not _container_check_ok(planted)

    def test_pre_checks_print_counts_and_use_the_exact_development_form(self, note: str) -> None:
        assert "grep -c -E '^(TRADING_)?ENVIRONMENT=development$'" in note
        assert "printenv ENVIRONMENT TRADING_ENVIRONMENT REQUIRE_API_AUTH | wc -l" in note
        assert 'test -e "$DEPLOY_DIR/.env.local"; echo "env_local=$?"' in note

    def test_the_auth_pre_check_is_the_a_b_form(self, note: str) -> None:
        assert _ab_pre_check_ok(note)
        # planted offender: the r1 line, which stops a deploy whose .env says REQUIRE_API_AUTH=false
        r1 = (
            "grep -c -E '^REQUIRE_API_AUTH=' \"$DEPLOY_DIR/.env\""
            "   # expected 0: R2 changes no default\n"
        )
        assert not _ab_pre_check_ok(r1)

    @pytest.mark.parametrize(
        ("env_lines", "passes"),
        [
            ("", True),  # no line: 0 and 0
            ("REQUIRE_API_AUTH=false\n", True),  # the normal host: 1 and 1
            ("REQUIRE_API_AUTH=true\n", False),
            ("REQUIRE_API_AUTH=\n", False),
            ("REQUIRE_API_AUTH=false\nREQUIRE_API_AUTH=true\n", False),
        ],
    )
    def test_the_a_b_counts_decide_as_intended(
        self, note: str, env_lines: str, passes: bool
    ) -> None:
        pats = re.findall(r"^grep -c -E '(\^REQUIRE_API_AUTH=[^']*)'", note, re.M)
        assert pats == ["^REQUIRE_API_AUTH=", "^REQUIRE_API_AUTH=false$"]
        a, b = (len(re.findall(p, env_lines, re.M)) for p in pats)
        assert (a == b) is passes

    def test_has_the_incident_escape_hatch(self, flat: str) -> None:
        assert "the plain Docker CLI still works" in flat
        assert "does not interpolate the compose file" in flat
        assert "Never use a general `docker inspect`" in flat

    def test_checks_after_the_deploy(self, note: str) -> None:
        assert "{{.HostConfig.LogConfig.Type}} {{json .HostConfig.LogConfig.Config}}" in note
        assert "api_auth_shadow_total" in note
        assert '`{result="reject"}` is 0' in note

    def test_the_413_probe_is_gone(self, note: str) -> None:
        # The UI middleware covers /bff/* and redirects before the body is read, so a status code
        # (413, 502 or 307) cannot tell whether the block is in force.
        assert "head -c 2097152" not in note
        assert "/bff/api/v1/x" not in note

    def test_the_caddy_adapt_count_check_replaces_it(self, note: str, flat: str) -> None:
        adapt = "$DC exec -T caddy caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile"
        assert note.count(adapt) == 2
        assert "grep -o '\"/bff/\\*\"' | wc -l" in note
        assert "# expected: at least 1" in note
        assert "grep -o '\"max_size\":1000000' | wc -l" in note
        assert "# expected: 4 (auth, admin, api, bff)" in note
        assert "A different count is a stop" in flat
        assert "the first host run records them under V-9737-05" in flat

    def test_the_v_9737_10_check_prints_one_field(self, note: str) -> None:
        line = "$DC config --format json | jq -r '.services.api.environment.ENVIRONMENT'"
        assert line in note
        assert "# expected: development" in note
        for ln in note.splitlines():
            if "config --format json" in ln:
                assert "| jq -r '.services.api.environment.ENVIRONMENT'" in ln, ln

    def test_never_under_set_x(self, note: str, flat: str) -> None:
        assert "Never run these steps under `set -x`" in flat
        assert not re.search(r"^\s*(set\s+-\w*x|set\s+-o\s+xtrace|bash\s+-x)", note, re.M)

    def test_warns_against_docker_run_with_the_production_env_file(self, flat: str) -> None:
        assert (
            "`docker run --env-file .env` (`Dockerfile.api:14`) is for development only, "
            "never with the production `.env`" in flat
        )

    def test_names_the_four_admin_routes_and_what_the_counter_shows(self, flat: str) -> None:
        assert "the four existing admin proxy routes" in flat
        assert "start sending it as `X-API-Key`" in flat
        assert "the kill-switch routes are admin-only and are not counted" in flat
        assert "counted as `missing`" in flat
        assert "log rotation bounds the resulting WARNING lines" in flat

    def test_states_what_the_gate_does_not_prove(self, flat: str) -> None:
        assert "does **not** prove that every UI path attaches the key" in flat
        assert "R3 gate (V-9737-12)" in flat

    def test_rollback_reverts_only_the_required_lines_and_keeps_the_rotation(
        self, flat: str
    ) -> None:
        assert "Revert only the two `:?` lines" in flat
        assert "Keep the `logging:` blocks and the `/bff/*` Caddy block" in flat
        assert "`API_KEY_HASH` may stay in `.env`" in flat
        assert "Do not revert the whole patch" in flat
        assert "Revert the R2 patch" not in flat

    def test_discloses_what_was_not_verified(self, flat: str) -> None:
        assert "was **not** run against Caddy" in flat
        assert "no `caddy` binary was available" in flat
        assert "Not verified when this was written" in flat

    def test_no_command_prints_a_secret(self, note: str) -> None:
        banned = re.compile(
            r"docker\s+(container\s+)?inspect(?!\s+--format)"
            r"|printenv\s*(\||$)"
            r"|exec\s+(-T\s+)?\w+\s+env\b"
            r"|\$DC\s+config\b(?!\s+--quiet"
            r"|\s+--format json \| jq -r '\.services\.api\.environment\.ENVIRONMENT')"
            r"|docker\s+compose\s+config\b(?!\s+--quiet)"
            r"|echo\s+[\"']?\$\{?(key|KEY|k|a|h|API_KEY)\b"
        )
        hits = [ln for ln in note.splitlines() if banned.search(ln)]
        assert all("never" in ln.lower() for ln in hits), hits


class TestDeployRunbookRows:
    @pytest.fixture(scope="class")
    def section(self) -> str:
        text = _DEPLOY_RUNBOOK.read_text(encoding="utf-8")
        start = text.index("### 3.3 Environment inventory")
        return text[start : text.index("### 3.4", start)]

    @pytest.mark.parametrize(
        ("row", "owner"),
        [("API_KEY", "(ui)"), ("API_KEY_HASH", "(api)")],
    )
    def test_the_variable_is_listed_as_a_new_requirement(
        self, section: str, row: str, owner: str
    ) -> None:
        rows = [ln for ln in section.splitlines() if ln.startswith(f"| **`{row}`** {owner}")]
        assert len(rows) == 1, rows
        assert "**New requirement (CF-97-37, release R2).**" in rows[0]
        assert "**before** the new compose file reaches the host" in rows[0]

    def test_the_key_row_names_the_generation_rule(self, section: str) -> None:
        row = next(ln for ln in section.splitlines() if ln.startswith("| **`API_KEY`** (ui)"))
        assert "at least 32 random bytes" in row
        assert "docs/runbooks/api-key-auth-switch.md" in row
        assert "must differ from `ADMIN_API_KEY`" in row

    def test_the_existing_rows_are_kept(self, section: str) -> None:
        for name in (
            "POSTGRES_PASSWORD",
            "ADMIN_API_KEY",
            "INTERNAL_ADMIN_API_KEY",
            "GRAFANA_PASSWORD",
        ):
            assert f"`{name}`" in section, name
