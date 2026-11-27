"""CF-97-64: the allow-marker comment check is linear.

The old, quadratic-per-marker ``_in_comment`` is kept here as the oracle for the equivalence tests.
Credential-shaped values come from ``rand`` (run time), so this file holds no flaggable literal.
"""

from __future__ import annotations

import random
import re
import time

import pytest

from tests.unit.test_fihyg01_secret_literals import deadline, hits_for, rand, scan

OLD_COMMENT_OPEN = re.compile(r"#|//|/\*|<!--")


def old_in_comment(text: str, marker_start: int) -> bool:
    for opener in OLD_COMMENT_OPEN.finditer(text, 0, marker_start):
        before = text[: opener.start()]
        if all(len(re.findall(r"(?<!\\)" + q, before)) % 2 == 0 for q in ('"', "'", "`")):
            return True
    return False


def old_allow_marker(text: str) -> bool:
    return any(old_in_comment(text, m.start()) for m in scan.ALLOW_RE.finditer(text))


MARK = "secret-scan: allow (reason is long enough)"
TRICKY = [
    f"x = 1  # {MARK}",
    f"x = 1  // {MARK}",
    f"x = 1  /* {MARK} */",
    f"<!-- {MARK} -->",
    f"x = 1  {MARK}",
    f'x = "{MARK}"',
    f'x = "# {MARK}"',
    f"x = '// {MARK}'",
    f'x = "a" # {MARK}',
    f"x = 'a' // {MARK}",
    f'x = "it\'s" # {MARK}',
    f"# it's fine {MARK}",
    f'x = "unbalanced # {MARK}',
    f"x = 'unbalanced // {MARK}",
    f"x = `tpl` // {MARK}",
    f"x = `tpl // {MARK}",
    f'x = "esc \\" quote" # {MARK}',
    f'x = "esc \\" # {MARK}',
    f'x = "a\' b" // {MARK}',
    f"x = 'a\" b' // {MARK}",
    f"x = \"a\" + 'b' + `c` # {MARK}",
    f"url = http://example.org/x {MARK}",
    f'url = "http://example.org/x" {MARK}',
    f'url = "http://example.org/x" # {MARK}',
    f"# first # second {MARK}",
    f'"# a" # {MARK}',
    f'"a" "# {MARK}',
    "secret-scan: allow (inner secret-scan: allow (valid reason here))",
    "# secret-scan: allow (short)",
    f"# {MARK} // {MARK}",
    f"{MARK} # {MARK}",
    f"\\' # {MARK}",
    f"'\\' # {MARK}",
    "",
    "# nothing here",
]


@pytest.mark.parametrize("line", TRICKY)
def test_cf9764_the_new_check_equals_the_old_one_on_tricky_lines(line: str) -> None:
    assert scan._allow_marker(line) == old_allow_marker(line)


def test_cf9764_the_new_check_equals_the_old_one_on_random_lines() -> None:
    rng = random.Random(9764)  # noqa: S311  (a fixed seed, not a secret)
    parts = [
        '"',
        "'",
        "`",
        "\\",
        "#",
        "//",
        "/*",
        "<!--",
        " ",
        "x",
        MARK,
        "secret-scan: allow (",
        ")",
    ]
    for _ in range(3000):
        line = "".join(rng.choice(parts) for _ in range(rng.randint(1, 14)))
        assert scan._allow_marker(line) == old_allow_marker(line), line


def test_cf9764_a_comment_still_exempts_and_a_string_still_does_not() -> None:
    value = rand()
    assert hits_for("a.py", f'API_KEY = "{value}"  # {MARK}') == []
    assert hits_for("a.py", f'API_KEY = "{value}"; y = "# {MARK}"') == [(1, "KEY", "entropy")]


def _crowded(count: int, quote: str) -> str:
    markers = "".join(f"// secret-scan: allow (reason number {i} here) " for i in range(count))
    return f'API_KEY = "{rand()}"  x = {quote} ' + markers


def test_cf9764_many_markers_after_an_unbalanced_quote_are_fast() -> None:
    line = _crowded(2000, "'")
    with deadline(2.0):
        hits, allowed = scan.scan_line("a.py", 1, line)
    assert [h.word for h in hits] == ["KEY"] and allowed == []  # the quote is open: no comment


def test_cf9764_many_markers_after_a_balanced_quote_exempt_quickly() -> None:
    line = _crowded(2000, "'x'")
    with deadline(2.0):
        hits, allowed = scan.scan_line("a.py", 1, line)
    assert hits == [] and [h.word for h in allowed] == ["KEY"]


@pytest.mark.parametrize("quote", ["'", '"', "`"])
def test_cf9764_a_large_crowded_line_scales_linearly(quote: str) -> None:
    small, big = _crowded(2000, quote), _crowded(20000, quote)
    t0 = time.perf_counter()
    scan.scan_line("a.py", 1, small)
    t1 = time.perf_counter()
    with deadline(5.0):
        scan.scan_line("a.py", 1, big)
    assert time.perf_counter() - t1 < 5.0 and t1 - t0 < 2.0


def test_cf9764_quote_free_openers_and_escapes_are_linear() -> None:
    lines = ["# " * 200000 + MARK, "\\'" * 200000 + " // " + MARK, '"' * 200001 + "# " + MARK]
    for body in lines:
        with deadline(3.0):
            scan._allow_marker(f'API_KEY = "{rand()}" ' + body)
