#!/usr/bin/env python3
"""FI-HYG-01 / PR-HYG-01: flag fixed secret literals on the lines a commit range adds.

Usage::

    python -I scripts/check_secret_literals.py <parent>..<commit>   # the gate
    python -I scripts/check_secret_literals.py                      # staged diff (not a gate)
    python -I scripts/check_secret_literals.py <range> --allow-empty

Gate contract: the executor runs it on ``<parent>..<commit>`` in the throwaway worktree after the
local commit and before the fast-forward; the gate tester repeats it on the pushed commit. Callers
must block on any non-zero exit and record the summary line. Never use the staged default as a
gate: an intent-to-add file is not in it. Pass ``--allow-empty`` only after
``git diff --numstat <range>`` shows 0 added lines.

Only lines the diff ADDS are read (git run with ``--text``, bytes, split on the newline byte only).
Rules, per added line of every non-binary file:

* Always, never exemptible: ``aws-prefix``, ``github-prefix``, ``google-prefix``, ``jwt-prefix``,
  ``private-key-prefix``; ``secret-file`` for a new ``.env``, ``.env.*`` (not ``.env.example``),
  ``*.pem``, ``*.key``, ``*.p12``, ``*.pfx``, ``id_rsa*`` or ``id_ed25519*`` file.
* Always, exemptible: ``url-credentials`` (``scheme://user:PASSWORD@``, password of 12 or more
  characters with 3.0 bits per character, not a ``$`` reference or a word-shaped placeholder).
* Except in deny-listed files (lockfiles, ``*.map``, ``*.snap``, ``*.min.js``, ``*.min.css``,
  images, fonts, ``.pdf``): ``entropy``, a literal of 20 or more characters with 3.5 or more bits
  per character assigned to a name with a KEY, SECRET, TOKEN, PASSWORD, PASSPHRASE, HASH, DSN,
  PASSWD, PWD, CREDENTIAL or BEARER word (``NAME = "..."``, ``NAME := "..."``, ``NAME: "..."``,
  ``name="..."``, ``"X-Api-Key": "..."``, ``os.environ["NAME"] = "..."``,
  ``setenv("NAME", "...")``). Config-like files (.json .yaml .yml .toml .ini .cfg .conf .sh .env*
  Dockerfile* *.example) and .md files also get the unquoted ``NAME=value`` and ``NAME: value``
  forms; .md gets nothing else.
  Skipped: whitespace or braces, whole ``$NAME`` / ``${NAME}`` / ``$(...)`` references, paths, URLs
  without credentials, e-mail addresses, UUIDs, word-segmented placeholders. ``${NAME:-default}``
  is judged by its default.

A same-line comment ``secret-scan: allow (<reason, 8+ characters, starts with a letter>)`` exempts
the exemptible rules and is reported on stderr. Output: ``path:line: WORD [rule]`` where WORD is a
canonical name word or ``-``. No value, identifier, line text or git message is ever printed.

Exit: 0 scanned, no hit. 1 hits. 2 usage error, git failure, not a repository, or nothing scanned.

Accepted gaps (second line of defence: GitGuardian on the pull request): multi-line assignments,
implicit concatenation, values with whitespace, diceware passphrases, a value on another line than
its name, vendor prefixes this repository does not use. Standard library only.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

MIN_LENGTH = 20
MIN_ENTROPY = 3.5
MIN_URL_PASSWORD = 12
MIN_URL_ENTROPY = 3.0
MIN_REASON = 8
NAME_WORDS = tuple(
    "KEY SECRET TOKEN PASSWORD PASSPHRASE HASH DSN PASSWD PWD CREDENTIAL BEARER".split()
)
DENY_ENDINGS = tuple(
    ".lock .min.js .min.css .map .snap .ico .pdf .png .jpg .jpeg .gif .webp .bmp .svg".split()
    + ".woff .woff2 .ttf .otf .eot".split()
)
CONFIG_SUFFIXES = frozenset(".json .yaml .yml .toml .ini .cfg .conf .sh".split())
GIT_DIFF: tuple[str, ...] = ("diff", "-U0", "--text", "--no-color", "--no-ext-diff")
GIT_DIFF += ("--no-textconv",)
GIT_DIFF += ("--no-renames", "--src-prefix=a/", "--dst-prefix=b/")
REV_RE = re.compile(r"[^\s;|&\-][^\s;|&]*")

CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+")
STRING_RES = {
    '"': re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"'),
    "'": re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'"),
    "`": re.compile(r"`([^`\\]*(?:\\.[^`\\]*)*)`"),
}
NAME = r"(?<![\w.\-])(\w[\w.\-]*)"
ASSIGN_RE = re.compile(NAME + r"""["'`]?\]?\s*(?::=|=|:)\s*[A-Za-z]{0,2}(?=["'`])""")
TYPED_RE = re.compile(NAME + r"""\s*:\s*[\w.\[\], |]{1,40}=\s*[A-Za-z]{0,2}(?=["'`])""")
SETENV_RE = re.compile(
    r"""\b(?:setenv|getenv)\(\s*(["'`])([^"'`]{1,64})\1\s*,\s*[A-Za-z]{0,2}(?=["'`])"""
)
UNQUOTED_RE = re.compile(NAME + r"""\s*[:=]\s*([^\s"'`#,;]{20,})""")
DOCKER_RE = re.compile(r"^\s*(?:ENV|ARG)\s+([A-Za-z_]\w*)\s+(\S{20,})", re.IGNORECASE)
URL_CRED_RE = re.compile(r"""://([^\s/:@"'`]{0,64}):([^\s/@"'`]{1,200})@""")
ALLOW_RE = re.compile(rf"secret-scan:\s*allow\s*\(\s*([A-Za-z][^)]{{{MIN_REASON - 1},}})\)")
COMMENT_OPEN_RE = re.compile(r"#|//|/\*|<!--")
# Pieces, so that this file holds no credential-shaped text of its own.
PEM = "-" * 5 + r"BEGIN(?: [A-Z0-9]+)* PRIVATE KEY"
PREFIX_RULES = (
    ("aws-prefix", re.compile(r"A[KS]IA[0-9A-Z]{16}")),
    ("github-prefix", re.compile(r"gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("google-prefix", re.compile(r"GOCSPX-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{35}")),
    ("jwt-prefix", re.compile(r"eyJ[\w-]{5,300}\.eyJ[\w-]{5,2000}\.")),
    ("private-key-prefix", re.compile(PEM)),
)
EXEMPT_NEVER = frozenset(rule for rule, _ in PREFIX_RULES) | {"secret-file"}

VAR_REF_RE = re.compile(r"\$\{?[A-Za-z_]\w*\}?|\$\([^)]*\)")
DEFAULT_RE = re.compile(r"\$\{[A-Za-z_]\w*:?[-=+?](.*)\}")
PATH_START_RE = re.compile(r"(?:\.{1,2})?/")
PATH_END_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,4}$")
URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://")
EMAIL_RE = re.compile(r"[^@\s/]+@[^@\s/]+\.[A-Za-z]{2,}")
UUID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
HEADER_RE = re.compile(r'diff --git (?:"a/(.+)" "b/\1"|a/(.+) b/\2)')
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hit:
    """One finding. Deliberately has no field for any text of the line."""

    path: str
    line: int
    word: str
    rule: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.word} [{self.rule}]"


@dataclass
class FileDiff:
    path: str
    shown: str
    new: bool = False
    lines: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class Result:
    hits: list[Hit] = field(default_factory=list)
    allowed: list[Hit] = field(default_factory=list)
    lines: int = 0
    files: int = 0
    denied: int = 0
    binary: int = 0

    def summary(self) -> str:
        return (
            f"secret-scan: scanned {self.lines} added lines in {self.files} files "
            f"({len(self.allowed)} allowed, {self.denied} denied-type files, "
            f"{self.binary} binary files)"
        )


def shannon_entropy(text: str) -> float:
    """Shannon entropy of ``text`` in bits per character (0.0 for the empty string)."""
    if not text:
        return 0.0
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in Counter(text).values())


def secret_word(name: str) -> str | None:
    """The canonical word if a segment of ``name`` equals it (or its plural), else None.

    Segments split at any non-alphanumeric character and at camelCase humps, so ``apiKey``,
    ``API_KEYS`` and ``X-Admin-Key`` match while ``monkey``, ``monkeypatch`` and ``hashlib`` do not.
    """
    spaced = CAMEL_2.sub(r"\1_\2", CAMEL_1.sub(r"\1_\2", name))
    for segment in WORD_SPLIT.split(spaced):
        upper = segment.upper()
        for word in NAME_WORDS:
            if upper in (word, word + "S"):
                return word
    return None


def looks_like_path(value: str) -> bool:
    return bool(PATH_START_RE.match(value) or PATH_END_RE.search(value))


def looks_like_word_segments(value: str) -> bool:
    parts = re.split(r"[-_.]", value)
    return len(parts) >= 3 and all(
        len(p) <= 12 and (p.isdigit() or re.search(r"[aeiouAEIOU]", p)) for p in parts
    )


def is_candidate(
    value: str, min_length: int = MIN_LENGTH, min_entropy: float = MIN_ENTROPY
) -> bool:
    """True for a fixed high-entropy literal; False for placeholders, references, paths, etc."""
    if len(value) < min_length:
        return False
    default = DEFAULT_RE.fullmatch(value)
    if default:
        value = default.group(1)
        if len(value) < min_length:
            return False
    elif VAR_REF_RE.fullmatch(value):
        return False
    if "{" in value or "}" in value or re.search(r"\s", value):
        return False
    if looks_like_path(value) or URL_RE.match(value) or EMAIL_RE.fullmatch(value):
        return False
    if UUID_RE.fullmatch(value) or looks_like_word_segments(value):
        return False
    return shannon_entropy(value) >= min_entropy


def _in_comment(text: str, marker_start: int) -> bool:
    for opener in COMMENT_OPEN_RE.finditer(text, 0, marker_start):
        before = text[: opener.start()]
        if all(len(re.findall(r"(?<!\\)" + q, before)) % 2 == 0 for q in ('"', "'", "`")):
            return True
    return False


def _allow_marker(text: str) -> bool:
    return any(_in_comment(text, m.start()) for m in ALLOW_RE.finditer(text))


def _assigned(text: str, config: bool, md: bool, docker: bool) -> Iterator[tuple[str, str]]:
    """Yield ``(canonical word, rule)`` for each assignment form holding a candidate literal."""
    seen: set[int] = set()
    if not md:
        forms = [ASSIGN_RE, SETENV_RE] + ([TYPED_RE] if ":" in text else [])
        for form in forms:
            for m in form.finditer(text):
                word = secret_word(m.group(2) if form is SETENV_RE else m.group(1))
                pos = m.end()
                if word and pos not in seen:
                    lit = STRING_RES[text[pos]].match(text, pos)
                    if lit and is_candidate(lit.group(1)):
                        seen.add(pos)
                        yield word, "entropy"
    if config or md:
        for m in UNQUOTED_RE.finditer(text):
            word = secret_word(m.group(1))
            if word and is_candidate(m.group(2)):
                yield word, "unquoted"
    if docker and (d := DOCKER_RE.match(text)):
        word = secret_word(d.group(1))
        if word and is_candidate(d.group(2)):
            yield word, "unquoted"


def _url_password(password: str) -> bool:
    return (
        len(password) >= MIN_URL_PASSWORD
        and not password.startswith("$")
        and not looks_like_word_segments(password)
        and shannon_entropy(password) >= MIN_URL_ENTROPY
    )


def scan_line(
    shown: str,
    lineno: int,
    text: str,
    entropy: bool = True,
    config: bool = False,
    md: bool = False,
    docker: bool = False,
) -> tuple[list[Hit], list[Hit]]:
    """Scan one added line; return ``(hits, allowed)``. Neither holds any text of the line."""
    if len(text) < MIN_LENGTH:
        return [], []
    found = [
        Hit(shown, lineno, "-", rule) for rule, pattern in PREFIX_RULES if pattern.search(text)
    ]
    if any(_url_password(m.group(2)) for m in URL_CRED_RE.finditer(text)):
        found.append(Hit(shown, lineno, "-", "url-credentials"))
    if entropy:
        found += [
            Hit(shown, lineno, word, rule) for word, rule in _assigned(text, config, md, docker)
        ]
    unique = list(dict.fromkeys(found))
    if not unique or not _allow_marker(text):
        return unique, []
    return (
        [h for h in unique if h.rule in EXEMPT_NEVER],
        [h for h in unique if h.rule not in EXEMPT_NEVER],
    )


def parse_diff(diff: bytes) -> list[FileDiff]:
    """Split ``git diff -U0`` bytes into files and their added lines (newline byte only)."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    old_left = new_left = new_no = 0
    for raw in diff.split(b"\n"):
        if old_left > 0 or new_left > 0:
            tag = raw[:1]
            if tag == b"+":
                new_left -= 1
                if cur is not None:
                    cur.lines.append((new_no, raw[1:].decode("utf-8", "replace")))
                new_no += 1
            elif tag == b"-":
                old_left -= 1
            elif tag != b"\\":
                old_left -= 1
                new_left -= 1
                new_no += 1
            continue
        line = raw.decode("utf-8", "replace")
        header = HEADER_RE.match(line)
        if header:
            quoted = header.group(1) is not None
            path = header.group(1) if quoted else header.group(2)
            cur = FileDiff(path, f'"{path}"' if quoted else path)
            files.append(cur)
        elif cur is not None and line.startswith("new file mode"):
            cur.new = True
        elif cur is not None and (hunk := HUNK_RE.match(line)):
            old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new_left = int(hunk.group(3)) if hunk.group(3) is not None else 1
            new_no = int(hunk.group(2))
    return files


def is_secret_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if name == ".env.example":
        return False
    return (
        name == ".env"
        or name.startswith((".env.", "id_rsa", "id_ed25519"))
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
    )


def scan_diff(diff: bytes, binary_paths: Collection[str] = ()) -> Result:
    """Scan git diff bytes; ``binary_paths`` are numstat ``-`` paths (a NUL confirms binary)."""
    result = Result()
    for item in parse_diff(diff):
        if item.new and is_secret_file(item.path):
            result.hits.append(Hit(item.shown, 0, "-", "secret-file"))
        if item.path in binary_paths and any("\x00" in text for _no, text in item.lines):
            result.binary += 1
            continue
        name = PurePosixPath(item.path).name.lower()
        suffix = PurePosixPath(name).suffix
        denied = name == "package-lock.json" or name.endswith(DENY_ENDINGS)
        config = (
            suffix in CONFIG_SUFFIXES
            or suffix == ".example"
            or name.startswith((".env", "dockerfile"))
        )
        docker = name.startswith("dockerfile")
        result.files += 1
        result.denied += denied
        result.lines += len(item.lines)
        for number, text in item.lines:
            hits, allowed = scan_line(
                item.shown, number, text, not denied, config, suffix == ".md", docker
            )
            result.hits += hits
            result.allowed += allowed
    return result


def get_diff(rev: str | None) -> tuple[bytes, set[str]]:
    """Return ``(git diff bytes, numstat binary paths)`` for ``rev`` (the staged diff if None)."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found")
    target = [rev, "--"] if rev else ["--cached", "--"]
    outputs = []
    for args in (GIT_DIFF, ("diff", "--numstat", "--no-renames", "--no-ext-diff")):
        proc = subprocess.run(  # noqa: S603  (fixed argv, no shell, revision checked by main)
            [git, *args, *target], capture_output=True, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git diff failed (exit {proc.returncode})")
        outputs.append(proc.stdout)
    binary: set[str] = set()
    for row in outputs[1].split(b"\n"):
        added, _, rest = row.decode("utf-8", "replace").partition("\t")
        if added == "-":
            path = rest.split("\t", 1)[-1]
            binary.add(path[1:-1] if path.startswith('"') else path)
    return outputs[0], binary


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    allow_empty = "--allow-empty" in args
    revs = [a for a in args if a != "--allow-empty"]
    if len(revs) > 1 or (revs and not REV_RE.fullmatch(revs[0])):
        print(
            "secret-scan: usage: check_secret_literals.py [<A..B>] [--allow-empty]", file=sys.stderr
        )
        return 2
    try:
        diff, binary = get_diff(revs[0] if revs else None)
    except RuntimeError as exc:
        print(f"secret-scan: {exc}", file=sys.stderr)
        return 2
    result = scan_diff(diff, binary)
    for hit in result.hits:
        print(hit.render())
    for hit in result.allowed:
        print(f"{hit.path}:{hit.line}: allowed [{hit.rule}]", file=sys.stderr)
    print(result.summary(), file=sys.stderr)
    if result.hits:
        print(f"secret-scan: {len(result.hits)} finding(s); see the docstring.", file=sys.stderr)
        return 1
    if result.lines == 0 and not allow_empty:
        print("secret-scan: nothing was scanned; refusing to pass.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
