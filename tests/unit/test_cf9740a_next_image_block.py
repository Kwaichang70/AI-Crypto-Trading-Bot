"""
tests/unit/test_cf9740a_next_image_block.py
-------------------------------------------
CF-97-40-A (37-C2): static pins for the Caddy block that answers 404 for the Next.js
Image Optimizer (``/_next/image``) at the edge, and for the fact that makes the block safe:
the dashboard does not use ``next/image``.

Pure file/text parsing (the caddy binary is not installed in the sandbox), in the same style
as tests/unit/test_cf9734_caddy_metrics_block.py, which is not edited by this change.

What is pinned:
  - exactly one ``/_next/`` handle block exists, in the primary site block, with the matcher
    ``/_next/image*``; it answers 404, does not proxy and sits above the catch-all (a style
    rule: Caddy sorts handle blocks itself, so this is not a routing claim);
  - while that block exists, nothing under ``apps/ui/src`` or in the one ``next.config.*``
    uses ``next/image`` or sets ``images`` (if the UI starts to use the optimizer, the edge
    block would break it silently: remove the block on purpose, with the review it needs);
  - the runbook carries the live check, the mandatory spelling loop and the 400 meaning, and
    every curl line in it prints only a status code (and a byte count), never a body.

Every pin is run against planted offenders as well as the real files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CADDYFILE = _REPO_ROOT / "infra" / "Caddyfile"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "caddy-metrics-block.md"
_UI_APP = _REPO_ROOT / "apps" / "ui"
_UI_SRC = _UI_APP / "src"

_IMAGE_HEADER = re.compile(r"^[ \t]*handle[ \t]+/_next/image\*[ \t]*\{[ \t]*$", re.MULTILINE)
_ANY_NEXT_HEADER = re.compile(r"^[ \t]*handle[ \t]+(/_next\S*)[ \t]*\{[ \t]*$", re.MULTILINE)
_CATCH_ALL_HEADER = re.compile(r"^[ \t]*handle[ \t]*\{[ \t]*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Text helpers (same approach as test_cf9734_caddy_metrics_block.py)
# ---------------------------------------------------------------------------


def _without_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _primary_site_block(code: str) -> str:
    lines = code.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "{{TAILSCALE_FQDN}} {"), None)
    assert start is not None, "primary site block header not found in infra/Caddyfile"
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start : i + 1])
    raise AssertionError("unbalanced braces in the primary Caddyfile site block")


def _body_after(text: str, header_start: int) -> str:
    open_idx = text.index("{", header_start)
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces in handle block")


def _code_lines(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if ln.strip()]


_ORDER_MESSAGE = (
    "style: the /_next/image* block must sit above the catch-all `handle {` (Caddy sorts "
    "single-path handle blocks itself and runs the catch-all last, so this is not a routing "
    "fault; the rule keeps the file readable and stays correct under a `route` wrapper, "
    "which keeps source order)"
)


def _block_problems(caddyfile_text: str) -> list[str]:
    """Everything wrong with the /_next/image edge block in this Caddyfile text (empty = ok)."""
    code = _without_comments(caddyfile_text)
    primary = _primary_site_block(code)
    problems: list[str] = []

    if len(_IMAGE_HEADER.findall(primary)) != 1:
        problems.append("no single `handle /_next/image* {` block in the primary site block")
    if len(_IMAGE_HEADER.findall(code)) != 1:
        problems.append("the /_next/image* block must exist exactly once in the whole file")
    # Only that one /_next/ matcher may exist: a wider one (/_next/*) would shadow
    # /_next/static/*, and a second one would be an unreviewed route.
    matchers = _ANY_NEXT_HEADER.findall(code)
    if matchers != ["/_next/image*"]:
        problems.append(
            f"the /_next/ handle matchers must be exactly ['/_next/image*'], got {matchers}"
        )

    match = _IMAGE_HEADER.search(primary)
    catch_all = _CATCH_ALL_HEADER.search(primary)
    if catch_all is None:
        problems.append("the catch-all `handle {` block was not found")
    if match and catch_all and match.start() > catch_all.start():
        problems.append(_ORDER_MESSAGE)
    if match:
        lines = _code_lines(_body_after(primary, match.start()))
        if lines != ["respond 404"]:
            problems.append(f"the block body must be exactly ['respond 404'], got {lines}")
    return problems


# ---------------------------------------------------------------------------
# Next.js image usage scan
# ---------------------------------------------------------------------------

_USAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    # import ... from "next/image", require(`next/image`), import("next/legacy/image"), ...
    # The quote must directly precede `next/`, so the string "/_next/image" (middleware) is
    # not a usage.
    "next/image module": re.compile(r"""["'`]next/(?:legacy/|future/)?image["'`]"""),
    "<Image JSX element": re.compile(r"<Image\b"),
}
_CONFIG_PATTERNS: dict[str, re.Pattern[str]] = {
    # images: {...}, "images": {...}, 'images' : {...}
    "images: option in next.config": re.compile(r"""(?<![\w.])["']?images["']?\s*:"""),
    # shorthand property: { images }, { a: 1, images, b }
    "images shorthand in next.config": re.compile(r"[{,]\s*images\s*[,}]"),
    # nextConfig.images = {...} and { ["images"]: {...} }
    "images assignment in next.config": re.compile(r"\.images\s*=(?!=)"),
    "computed images key in next.config": re.compile(r"""\[\s*["']images["']\s*\]"""),
}
_SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mdx"}
_CONFIG_NAMES = (
    "next.config.js",
    "next.config.mjs",
    "next.config.cjs",
    "next.config.ts",
    "next.config.mts",
)


def _usage_findings(name: str, text: str, *, is_config: bool) -> list[str]:
    patterns = dict(_USAGE_PATTERNS)
    if is_config:
        patterns.update(_CONFIG_PATTERNS)
    return [f"{name}: {label}" for label, rx in patterns.items() if rx.search(text)]


def _config_files(app: Path) -> list[Path]:
    return [app / n for n in _CONFIG_NAMES if (app / n).exists()]


def _scan_ui(app: Path) -> list[str]:
    findings: list[str] = []
    src = app / "src"
    for path in sorted(src.rglob("*")):
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES and "node_modules" not in path.parts:
            findings += _usage_findings(
                str(path.relative_to(src)), path.read_text(encoding="utf-8"), is_config=False
            )
    configs = _config_files(app)
    if len(configs) != 1:
        findings.append(f"next.config: exactly one config file expected, found {len(configs)}")
    for config in configs:
        findings += _usage_findings(config.name, config.read_text(encoding="utf-8"), is_config=True)
    return findings


_COMMAND = re.compile(r"\b(curl|wget) -")
_ANY_TOOL = re.compile(r"\b(?:curl|wget)\b")
_CURL_FORBIDDEN = re.compile(
    r"\s(?:-[ivDO]\b|--(?:include|verbose|trace\S*|dump-header|output)\b|-o\s+-(?:\s|$))"
)
_WGET_FORBIDDEN = re.compile(r"-O\s*-(?:\s|$)|--output-document[= ]-")


def _command_problems(text: str) -> list[str]:
    """Every `curl -`/`wget -` line (fenced, indented or inside an inline span) must print only a
    status code: one tool per line, output to /dev/null, no header or body options."""
    problems: list[str] = []
    for ln in text.splitlines():
        match = _COMMAND.search(ln)
        if not match:
            continue
        if len(_ANY_TOOL.findall(ln)) != 1:
            problems.append(f"exactly one curl/wget per line: {ln.strip()}")
        if match.group(1) == "curl":
            required, forbidden = ("-o /dev/null", "-w '%{http_code}"), _CURL_FORBIDDEN
        else:
            required, forbidden = ("-O /dev/null",), _WGET_FORBIDDEN
        problems += [f"missing {r!r}: {ln.strip()}" for r in required if r not in ln]
        if forbidden.search(ln):
            problems.append(f"forbidden option: {ln.strip()}")
    return problems


@pytest.fixture(scope="module")
def caddyfile_text() -> str:
    return _CADDYFILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


class TestImageBlock:
    def test_the_real_caddyfile_has_no_problems(self, caddyfile_text: str) -> None:
        assert _block_problems(caddyfile_text) == []


class TestPlantedBlockOffenders:
    """The pin must fail on each of these edits of the real file."""

    _GOOD = "    handle /_next/image* {\n        respond 404\n    }\n"

    def _edit(self, text: str, replacement: str) -> str:
        assert text.count(self._GOOD) == 1
        return text.replace(self._GOOD, replacement)

    def test_removed(self, caddyfile_text: str) -> None:
        assert _block_problems(self._edit(caddyfile_text, "")) != []

    def test_respond_200(self, caddyfile_text: str) -> None:
        bad = self._edit(caddyfile_text, self._GOOD.replace("respond 404", "respond 200"))
        assert any("respond 404" in p for p in _block_problems(bad))

    def test_proxying_instead_of_404(self, caddyfile_text: str) -> None:
        bad = self._edit(caddyfile_text, self._GOOD.replace("respond 404", "reverse_proxy ui:3000"))
        assert _block_problems(bad) != []

    def test_matcher_widened_to_all_of_next(self, caddyfile_text: str) -> None:
        bad = self._edit(caddyfile_text, self._GOOD.replace("/_next/image*", "/_next/*"))
        assert _block_problems(bad) != []

    def test_matcher_without_prefix_star(self, caddyfile_text: str) -> None:
        bad = self._edit(caddyfile_text, self._GOOD.replace("/_next/image*", "/_next/image"))
        assert _block_problems(bad) != []

    def test_second_next_block_added(self, caddyfile_text: str) -> None:
        extra = "    handle /_next/static/* {\n        respond 404\n    }\n"
        assert _block_problems(self._edit(caddyfile_text, self._GOOD + extra)) != []

    def test_moved_below_the_catch_all_breaks_the_style_rule(self, caddyfile_text: str) -> None:
        without = self._edit(caddyfile_text, "")
        marker = "    # Catch-all: everything else goes to the Next.js frontend.\n    handle {\n"
        assert without.count(marker) == 1
        catch_all_start = without.index(marker)
        catch_all_end = without.index("    }\n", catch_all_start) + len("    }\n")
        moved = without[:catch_all_end] + self._GOOD + without[catch_all_end:]
        problems = _block_problems(moved)
        assert any("above the catch-all" in p for p in problems)
        assert not any("never match" in p or "first match" in p for p in problems)

    def test_only_in_a_comment(self, caddyfile_text: str) -> None:
        commented = "    # handle /_next/image* {\n    #     respond 404\n    # }\n"
        assert _block_problems(self._edit(caddyfile_text, commented)) != []

    def test_block_in_the_grafana_site_only(self, caddyfile_text: str) -> None:
        without = self._edit(caddyfile_text, "")
        grafana = "    reverse_proxy grafana:3000\n"
        assert without.count(grafana) == 1
        assert _block_problems(without.replace(grafana, self._GOOD + grafana)) != []


# ---------------------------------------------------------------------------
# The UI must not use the optimizer while the edge block exists
# ---------------------------------------------------------------------------


class TestUiDoesNotUseTheOptimizer:
    def test_the_scan_sees_the_ui_sources_and_exactly_one_config(self) -> None:
        # Guard against a scan that silently covers nothing.
        files = [p for p in _UI_SRC.rglob("*") if p.suffix in _SOURCE_SUFFIXES]
        assert len(files) > 20
        assert len(_config_files(_UI_APP)) == 1

    def test_no_next_image_usage_while_the_edge_block_exists(self, caddyfile_text: str) -> None:
        assert _IMAGE_HEADER.search(_without_comments(caddyfile_text)), "edge block is missing"
        assert _scan_ui(_UI_APP) == [], (
            "the UI uses next/image or sets `images` while infra/Caddyfile answers 404 for "
            "/_next/image*: images would break. Remove the edge block deliberately (with review) "
            "or stop using the optimizer."
        )

    def test_the_middleware_path_strings_are_not_usage(self) -> None:
        middleware = (_UI_SRC / "middleware.ts").read_text(encoding="utf-8")
        assert "_next/image" in middleware
        assert _usage_findings("middleware.ts", middleware, is_config=False) == []

    @pytest.mark.parametrize(
        "planted",
        [
            'import Image from "next/image";',
            "import Image from 'next/image'",
            'import Img from "next/legacy/image";',
            'import Img from "next/future/image";',
            'const Image = require("next/image");',
            "const Image = require(`next/image`);",
            'const m = await import("next/image");',
            "const m = await import(`next/image`);",
            'export const A = () => <Image src="/x.png" alt="" width={1} height={1} />;',
            "<Image\n  src={u}\n/>",
        ],
    )
    def test_planted_source_offenders_are_found(self, planted: str) -> None:
        assert _usage_findings("planted.tsx", planted, is_config=False) != []

    @pytest.mark.parametrize(
        "planted",
        [
            'const nextConfig = { images: { domains: ["example.com"] } };',
            "const nextConfig = {\n  images : { unoptimized: false },\n};",
            'export default { "images": { unoptimized: true } };',
            "export default { 'images': {} };",
            'const images = {}; export default { images, output: "standalone" };',
            "export default { output: 'standalone', images };",
            "nextConfig.images = { unoptimized: false };",
            'export default { ["images"]: {} };',
            'import Image from "next/image";',
        ],
    )
    def test_planted_config_offenders_are_found(self, planted: str) -> None:
        assert _usage_findings("next.config.mjs", planted, is_config=True) != []

    @pytest.mark.parametrize(
        "benign",
        [
            "// the Image Optimizer is not used",
            "const nextImages = 3; const myimages: string[] = [];",
            "const x = <ImageGallery />;",
        ],
    )
    def test_benign_text_is_not_flagged(self, benign: str) -> None:
        assert _usage_findings("ok.tsx", benign, is_config=True) == []

    @staticmethod
    def _app(tmp_path: Path, config_name: str, config_text: str) -> Path:
        (tmp_path / "src" / "components").mkdir(parents=True)
        (tmp_path / "src" / "components" / "ok.tsx").write_text("export const K = 1;\n", "utf-8")
        (tmp_path / config_name).write_text(config_text, encoding="utf-8")
        return tmp_path

    def test_a_planted_ui_file_fails_the_directory_scan(self, tmp_path: Path) -> None:
        app = self._app(tmp_path, "next.config.mjs", "export default { output: 'standalone' };\n")
        (app / "src" / "components" / "logo.tsx").write_text(
            'import Image from "next/image";\n', encoding="utf-8"
        )
        assert _scan_ui(app) == ["components/logo.tsx: next/image module"]

    def test_a_planted_config_fails_the_directory_scan(self, tmp_path: Path) -> None:
        app = self._app(
            tmp_path, "next.config.mjs", "export default { images: { domains: [] } };\n"
        )
        assert _scan_ui(app) == ["next.config.mjs: images: option in next.config"]

    def test_a_lone_typescript_config_is_scanned(self, tmp_path: Path) -> None:
        app = self._app(tmp_path, "next.config.ts", "export default { images: {} };\n")
        assert _scan_ui(app) == ["next.config.ts: images: option in next.config"]

    def test_two_config_files_fail_the_exactly_one_rule(self, tmp_path: Path) -> None:
        app = self._app(tmp_path, "next.config.mjs", "export default {};\n")
        (app / "next.config.js").write_text("module.exports = {};\n", encoding="utf-8")
        assert _scan_ui(app) == ["next.config: exactly one config file expected, found 2"]


# ---------------------------------------------------------------------------
# Runbook
# ---------------------------------------------------------------------------

_SPELLINGS = (
    "'/_next/image'",
    "'/_next/image/'",
    "'/_NEXT/IMAGE'",
    "'/_next/%69mage'",
    "'/_next%2Fimage'",
    "'//_next/image'",
    "'/_next//image'",
    "'/_next/static/../image'",
    "'/_next/image%3Fx'",
)


class TestRunbook:
    @pytest.fixture(scope="class")
    def section(self) -> str:
        note = _RUNBOOK.read_text(encoding="utf-8")
        return note[note.index("## 5. Second edge block: `/_next/image*`") :]

    def test_shows_the_block_and_the_live_check(self, section: str) -> None:
        assert "handle /_next/image* {" in section
        assert "respond 404" in section
        assert (
            "curl -s -o /dev/null -w '%{http_code}\\n' "
            "'https://<host>/_next/image?url=%2Fx&w=64&q=75'"
        ) in section
        assert "curl -s -o /dev/null -w '%{http_code}\\n' 'https://<host>/'" in section

    def test_states_the_expectations(self, section: str) -> None:
        flat = " ".join(section.split())
        assert "Expected: `404` for the first line;" in flat
        assert "`400` there means the optimizer answered" in flat
        assert "`200` or a redirect" in flat
        assert "**not** `404`" in flat

    def test_points_at_validate_and_reload(self, section: str) -> None:
        assert "validate" in section and "reload" in section
        assert "caddy version" in section

    def test_the_spelling_loop_is_complete(self, section: str) -> None:
        flat = " ".join(section.split())
        assert "--path-as-is" in section
        loop_curls = [ln for ln in section.splitlines() if "curl -s --path-as-is" in ln]
        assert len(loop_curls) == 1
        assert "%{size_download}" in loop_curls[0]
        for spelling in _SPELLINGS:
            assert spelling in section, spelling
        assert "`400` means the optimizer answered" in flat
        assert "`308` means Caddy did not match" in flat

    def test_every_command_line_prints_only_a_status_code(self, section: str) -> None:
        assert _command_problems(section) == []
        commands = [ln for ln in section.splitlines() if _COMMAND.search(ln)]
        assert sum("curl -" in ln for ln in commands) >= 3
        assert sum("wget -" in ln for ln in commands) == 1

    def test_states_the_miss_rules(self, section: str) -> None:
        flat = " ".join(section.split())
        assert "keep the block" in flat
        assert "A byte count of `0` shows that the `404` came from Caddy" in flat
        assert "`307` means Caddy did not match and the UI's sign-in redirect answered" in flat
        assert 'echo "rc=$?"' in section

    @pytest.mark.parametrize(
        "planted",
        [
            "curl -s -o /dev/null -w '%{http_code}\\n' --trace-ascii - https://h/",
            "curl -s -o - -w '%{http_code}\\n' https://h/",
            "curl -s --output - -w '%{http_code}\\n' https://h/",
            "curl -s -o /dev/null -w '%{http_code}\\n' --dump-header - https://h/",
            "curl -v -o /dev/null -w '%{http_code}\\n' https://h/",
            "    curl -s -i -o /dev/null -w '%{http_code}\\n' https://h/",
            "Run `curl -s https://h/` and read the page.",
            "curl -s -o /dev/null -w '%{http_code}\\n' https://a/; "
            "curl -s -o /dev/null -w '%{http_code}\\n' https://b/",
            "$DC exec -T caddy wget -q -O - 'http://ui/'",
            "$DC exec -T caddy wget -q 'http://ui/'",
        ],
    )
    def test_planted_command_offenders_are_found(self, planted: str) -> None:
        assert _command_problems(planted) != []

    @pytest.mark.parametrize(
        "benign",
        [
            "`--path-as-is` keeps curl from tidying the path.",
            "    curl -s -o /dev/null -w '%{http_code}\\n' 'https://<host>/'",
            "`$DC exec -T caddy wget -q -O /dev/null 'http://ui:3000/x'; echo \"rc=$?\"`",
        ],
    )
    def test_benign_command_text_is_not_flagged(self, benign: str) -> None:
        assert _command_problems(benign) == []
