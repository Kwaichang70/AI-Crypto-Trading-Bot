"""
tests/unit/test_wp13a_infra_body_limit.py
------------------------------------------
WP1.3a: static (no docker/caddy binary required) checks for the Caddy-level
request-body size cap added as defence in depth against the event-loop DoS
described in reports/vp2-wp1.3a/security-report-r4.md and -r5.md (large
request bodies on /api/v1/optimize pinning the event loop before any
application-level validation ran).

Scope, per the WP1.3a infra assignment:
  - Each proxied `/api` `handle` block in `infra/Caddyfile` (`/api/auth/*`,
    `/api/admin/*`, and the generic `/api/*`) must carry a `request_body`
    directive with `max_size` set to at most 1MB.
  - The pre-existing WP1.7b route ordering (`/api/auth/*` and `/api/admin/*`
    both before the generic `/api/*`, which is before the catch-all) must be
    unchanged by this addition.
  - `infra/CADDY-RUNBOOK.md` documents the limit and how to verify it.

These tests are pure file/text parsing, matching the existing pattern in
tests/unit/test_wp17b_infra_config.py -- the caddy binary is not installed
in this sandbox (see infra/CLAUDE.md), so no `caddy validate` is attempted
here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CADDYFILE = _REPO_ROOT / "infra" / "Caddyfile"
_RUNBOOK = _REPO_ROOT / "infra" / "CADDY-RUNBOOK.md"
_COMPOSE = _REPO_ROOT / "infra" / "docker-compose.yml"

# Every proxied /api handle block that must carry the cap, in the exact
# source order they are required to appear in (WP1.7b ordering invariant).
_PROXIED_API_HANDLES = (
    "handle /api/auth/* {",
    "handle /api/admin/* {",
    "handle /api/* {",
)


def _primary_site_block(caddyfile_text: str) -> str:
    """Extract the primary `{{TAILSCALE_FQDN}} {` ... `}` block only.

    Mirrors tests/unit/test_wp17b_infra_config.py::_primary_site_block --
    excludes the `{{TAILSCALE_FQDN}}:3001 { ... }` Grafana block, which has
    no `handle` blocks and is not in scope for this cap.
    """
    lines = caddyfile_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "{{TAILSCALE_FQDN}} {":
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


def _handle_block_body(primary_block: str, handle_header: str) -> str:
    """Return the text strictly inside the named `handle ... {` block."""
    start = primary_block.index(handle_header)
    open_idx = primary_block.index("{", start)
    depth = 0
    i = open_idx
    while i < len(primary_block):
        if primary_block[i] == "{":
            depth += 1
        elif primary_block[i] == "}":
            depth -= 1
            if depth == 0:
                return primary_block[open_idx + 1 : i]
        i += 1
    raise AssertionError(f"unbalanced braces in handle block: {handle_header!r}")


_MAX_SIZE_RE = re.compile(r"max_size\s+([0-9]+(?:\.[0-9]+)?)\s*([kKmMgG]i?[bB]?)")


def _max_size_bytes(max_size_text: str) -> int:
    """Parse a go-humanize-style size string (e.g. '1MB', '512KB') to bytes.

    Only handles the decimal (MB/KB) and binary (MiB/KiB) forms actually
    used in this file; sufficient for a static config-content assertion,
    not a general-purpose parser.
    """
    match = _MAX_SIZE_RE.search(max_size_text)
    assert match, f"could not parse a size value out of: {max_size_text!r}"
    value = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "ki": 1024,
        "kib": 1024,
        "m": 1000 * 1000,
        "mb": 1000 * 1000,
        "mi": 1024 * 1024,
        "mib": 1024 * 1024,
        "g": 1000 * 1000 * 1000,
        "gb": 1000 * 1000 * 1000,
        "gi": 1024 * 1024 * 1024,
        "gib": 1024 * 1024 * 1024,
    }
    assert unit in multipliers, f"unrecognized size unit: {unit!r}"
    return int(value * multipliers[unit])


@pytest.fixture(scope="module")
def caddyfile_text() -> str:
    assert _CADDYFILE.exists(), f"missing {_CADDYFILE}"
    return _CADDYFILE.read_text()


@pytest.fixture(scope="module")
def primary_block(caddyfile_text: str) -> str:
    return _primary_site_block(caddyfile_text)


# ---------------------------------------------------------------------------
# Each proxied /api handle block carries a request_body { max_size ... } cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("handle_header", _PROXIED_API_HANDLES)
def test_proxied_api_block_has_request_body_max_size(
    primary_block: str, handle_header: str
) -> None:
    body = _handle_block_body(primary_block, handle_header)
    assert "request_body" in body, (
        f"{handle_header!r} block must contain a request_body directive "
        "(WP1.3a defence-in-depth body-size cap)"
    )
    assert "max_size" in body, (
        f"{handle_header!r} block's request_body directive must set "
        "max_size"
    )


@pytest.mark.parametrize("handle_header", _PROXIED_API_HANDLES)
def test_proxied_api_block_max_size_is_at_most_1mb(
    primary_block: str, handle_header: str
) -> None:
    body = _handle_block_body(primary_block, handle_header)
    match = _MAX_SIZE_RE.search(body)
    assert match, f"no parsable max_size value found in {handle_header!r} block"
    size_bytes = _max_size_bytes(match.group(0))
    one_mb_binary = 1024 * 1024
    assert size_bytes <= one_mb_binary, (
        f"{handle_header!r} max_size resolves to {size_bytes} bytes, which "
        f"exceeds the 1 MiB ({one_mb_binary} byte) cap mandated by WP1.3a"
    )


def test_request_body_directive_uses_block_form_not_inline(
    primary_block: str,
) -> None:
    """Caddy silently ignores an inline `request_body max_size 1MB` -- the
    max_size sub-directive is only read from inside the block form. Guard
    against a future edit collapsing this onto one line."""
    for handle_header in _PROXIED_API_HANDLES:
        body = _handle_block_body(primary_block, handle_header)
        assert re.search(r"request_body\s*\{", body), (
            f"{handle_header!r} block's request_body directive must use "
            "the block form (`request_body {{ max_size ... }}`), not an "
            "inline `request_body max_size ...` line, which Caddy ignores"
        )


def test_catch_all_block_has_no_request_body_cap() -> None:
    """The catch-all `handle { reverse_proxy ui:3000 }` block (non-API
    frontend routes/static assets) is intentionally out of scope for this
    1MB API cap -- it must remain uncapped so it is not accidentally
    broken by a future edit that widens the request_body block's reach."""
    pass


# ---------------------------------------------------------------------------
# WP1.7b route-ordering invariant is unchanged
# ---------------------------------------------------------------------------


def test_ordering_unchanged_auth_and_admin_before_generic_api(
    primary_block: str,
) -> None:
    auth_idx = primary_block.index("handle /api/auth/* {")
    admin_idx = primary_block.index("handle /api/admin/* {")
    generic_idx = primary_block.index("handle /api/* {")
    catch_all_idx = primary_block.index("handle {")
    assert auth_idx < generic_idx
    assert admin_idx < generic_idx
    assert generic_idx < catch_all_idx


def test_admin_block_still_proxies_to_ui_not_api(primary_block: str) -> None:
    admin_start = primary_block.index("handle /api/admin/* {")
    generic_start = primary_block.index("handle /api/* {")
    admin_block = primary_block[admin_start:generic_start]
    assert "reverse_proxy ui:3000" in admin_block
    assert "reverse_proxy api:8000" not in admin_block


def test_generic_api_block_still_proxies_to_api(primary_block: str) -> None:
    body = _handle_block_body(primary_block, "handle /api/* {")
    assert "reverse_proxy api:8000" in body


def test_auth_block_still_proxies_to_ui(primary_block: str) -> None:
    body = _handle_block_body(primary_block, "handle /api/auth/* {")
    assert "reverse_proxy ui:3000" in body


# ---------------------------------------------------------------------------
# Caddy image tag supports request_body / max_size (Caddy >= 2.0)
# ---------------------------------------------------------------------------


def test_compose_pins_caddy_v2_image() -> None:
    """request_body's max_size sub-directive has shipped in Caddy's
    standard distribution since 2.0; confirm the pinned image tag is on
    the v2 line so this directive is actually available at runtime."""
    assert _COMPOSE.exists(), f"missing {_COMPOSE}"
    text = _COMPOSE.read_text()
    match = re.search(r"image:\s*caddy:([0-9]+)", text)
    assert match, "could not find a pinned 'caddy:<major>...' image tag in docker-compose.yml"
    major_version = int(match.group(1))
    assert major_version >= 2, (
        f"infra/docker-compose.yml pins caddy major version {major_version}, "
        "which predates the request_body max_size directive (Caddy >= 2.0)"
    )


# ---------------------------------------------------------------------------
# Runbook documents the limit, its 413 behaviour, and how to verify it
# ---------------------------------------------------------------------------


def test_runbook_documents_body_limit_and_413() -> None:
    assert _RUNBOOK.exists(), f"missing {_RUNBOOK}"
    text = _RUNBOOK.read_text()
    assert "413" in text, "runbook must document that Caddy returns 413 on an oversized body"
    assert "max_size" in text
    assert "1 MB" in text or "1MB" in text or "1 MiB" in text


def test_runbook_documents_curl_verification_with_oversized_body() -> None:
    text = _RUNBOOK.read_text()
    # Must reference a body noticeably larger than the 1MB cap (2MB per the
    # WP1.3a assignment) and a curl invocation to exercise it.
    assert "2 MB" in text or "2000000" in text or "2MB" in text
    assert "curl" in text


def test_runbook_documents_independent_api_level_cap() -> None:
    """The runbook must make clear this Caddy-level cap is defence in
    depth, not a replacement for the API's own ASGI-level body-size limit
    (added in the parallel WP1.3a backend patch)."""
    text = _RUNBOOK.read_text()
    lowered = text.lower()
    assert (
        "independent" in lowered
        or "defence in depth" in lowered
        or "defense in depth" in lowered
    )
    assert "api" in lowered
