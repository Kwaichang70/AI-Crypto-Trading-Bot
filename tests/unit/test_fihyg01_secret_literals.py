"""FI-HYG-01 r2: tests for ``scripts/check_secret_literals.py``.

Test names start with the requirement they pin (``test_r01_...`` is R-01 of the r1 synthesis).
Every credential-shaped value is built at run time from ``string`` constants or a system RNG, so
this file holds no flaggable literal itself (PR-HYG-01); the last tests scan it and the script.
"No leak" means: no 8-character slice of a planted value in stdout or stderr.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import secrets
import shutil
import signal
import string
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_secret_literals.py"
ALPHABET = string.ascii_letters + string.digits  # 62 distinct characters
UPPER = string.ascii_uppercase
HIGH = ALPHABET[:32]  # 32 distinct characters: 5.0 bits per character
LOW = "changeme" * 4  # 2.75 bits per character
RNG = secrets.SystemRandom()
PEM = "-" * 5 + "BEGIN RSA PRIVATE KEY" + "-" * 5
PREFIXED = [  # (id, value, rule)
    ("akia", "AKIA" + UPPER[:16], "aws-prefix"),
    ("asia", "ASIA" + UPPER[:16], "aws-prefix"),
    ("ghp", "ghp_" + ALPHABET[:36], "github-prefix"),
    ("gho", "gho_" + ALPHABET[:36], "github-prefix"),
    ("github-pat", "github_pat_" + ALPHABET[:40], "github-prefix"),
    ("gocspx", "GOCSPX-" + ALPHABET[:28], "google-prefix"),
    ("aiza", "AIza" + ALPHABET[:35], "google-prefix"),
    ("jwt", "eyJ" + ALPHABET[:20] + ".eyJ" + ALPHABET[:30] + "." + ALPHABET[:20], "jwt-prefix"),
    ("pem", PEM, "private-key-prefix"),
]
LINE_RE = re.compile(r"^\S.*:\d+: (?:[A-Z]+|-) \[[a-z-]+\]$")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_secret_literals", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


scan = _load()


def rand(n: int = 32) -> str:
    """``n`` distinct random alphanumerics: unpredictable, exactly log2(n) bits per character."""
    return "".join(RNG.sample(ALPHABET, n))


def diff_for(
    path: str,
    added: list[str],
    start: int = 1,
    removed: tuple[str, ...] = (),
    new: bool = False,
    deleted: bool = False,
) -> bytes:
    """A ``git diff -U0`` for one file that adds ``added`` and removes ``removed``."""
    old = f"-{start},{len(removed)}" if removed else f"-{start - 1},0"
    lines = [f"diff --git a/{path} b/{path}"]
    lines += ["new file mode 100644"] if new else ["deleted file mode 100644"] if deleted else []
    lines += [f"--- a/{path}", f"+++ b/{path}", f"@@ {old} +{start},{len(added)} @@"]
    lines += [f"-{text}" for text in removed] + [f"+{text}" for text in added]
    return ("\n".join(lines) + "\n").encode()


def hits_for(path: str, *added: str) -> list[tuple[int, str, str]]:
    result = scan.scan_diff(diff_for(path, list(added)))
    return [(h.line, h.word, h.rule) for h in result.hits]


def assert_no_leak(secret: str, output: str) -> None:
    assert secret not in output
    for i in range(len(secret) - 7):
        assert secret[i : i + 8] not in output, "an 8-character slice of the value was printed"


def run_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diff: bytes, *args: str
) -> tuple[int, str, str]:
    monkeypatch.setattr(scan, "get_diff", lambda _rev: (diff, set()))
    code = scan.main(list(args) or ["A..B"])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@contextlib.contextmanager
def deadline(seconds: float) -> Iterator[None]:
    def boom(*_: object) -> None:
        raise AssertionError(f"took more than {seconds}s")

    old = signal.signal(signal.SIGALRM, boom)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# a throwaway git repository
# ---------------------------------------------------------------------------
class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(path),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        argv = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args]
        done = subprocess.run(  # noqa: S603
            argv, cwd=self.path, env=self.env, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    def commit(self, message: str = "c") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def write(self, name: str, content: str | bytes) -> None:
        data = content.encode() if isinstance(content, str) else content
        (self.path / name).write_bytes(data)

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-I", str(SCRIPT), *args],
            cwd=self.path,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def scan_files(self, files: dict[str, str | bytes]) -> subprocess.CompletedProcess[str]:
        self.write("seed.txt", "seed\n")
        base = self.commit("base")
        for name, content in files.items():
            self.write(name, content)
        return self.run(f"{base}..{self.commit('adds')}")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    return Repo(tmp_path)


# ---------------------------------------------------------------------------
# thresholds and helpers
# ---------------------------------------------------------------------------
def test_entropy_helper_and_thresholds() -> None:
    assert scan.shannon_entropy("") == 0.0
    assert scan.shannon_entropy("abab") == pytest.approx(1.0)
    assert scan.shannon_entropy(HIGH) == pytest.approx(5.0)
    assert scan.shannon_entropy(LOW) == pytest.approx(2.75)
    assert (scan.MIN_LENGTH, scan.MIN_ENTROPY) == (20, 3.5)


@pytest.mark.parametrize(
    ("literal", "flagged"),
    [
        (ALPHABET[:20], True),  # exactly 20 characters
        (ALPHABET[:19], False),
        (ALPHABET[:12] * 2, True),  # 24 characters, 12 symbols: 3.585 bits
        (ALPHABET[:11] * 2, False),  # 22 characters, 11 symbols: 3.459 bits
        (LOW, False),
        ("x" * 40, False),
    ],
)
def test_length_and_entropy_boundaries(literal: str, flagged: bool) -> None:
    assert bool(hits_for("a.py", f'API_KEY = "{literal}"')) is flagged


# ---------------------------------------------------------------------------
# R-01: never print a value
# ---------------------------------------------------------------------------
LEAK_LINES = [  # (template, expected hits); v = the value to hide, c = a real candidate
    ('note = "see apikey-{v} here"; API_KEY = "{c}"', 1),
    ('"{v}_token": "{c}"', 1),
    ('secret.{v} = "{c}"', 1),
    ('{v}key = "{c}"', 0),
    ('x = "{c}"  # {v}hash', 0),
    ("{v}_key = '{c}'", 1),
    ("process.env.{v}_PASSWORD = `{c}`", 1),
]


@pytest.mark.parametrize(("template", "expected"), LEAK_LINES)
def test_r01_the_name_channel_cannot_carry_a_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    template: str,
    expected: int,
) -> None:
    v, c = rand(30), rand(32)
    code, out, err = run_main(monkeypatch, capsys, diff_for("a.py", [template.format(v=v, c=c)]))
    assert code == (1 if expected else 0)
    assert len(out.splitlines()) == expected
    assert all(LINE_RE.match(line) for line in out.splitlines())
    for secret in (v, c):
        assert_no_leak(secret, out + err)


@pytest.mark.parametrize("name", ["a.py", "c.json", "d.example"])
def test_r01_hits_print_path_line_word_and_rule_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    c = rand()
    code, out, err = run_main(monkeypatch, capsys, diff_for(name, [f'ADMIN_TOKEN = "{c}"']))
    assert (code, out) == (1, f"{name}:1: TOKEN [entropy]\n")
    assert_no_leak(c, err)


@pytest.mark.parametrize(("ident", "secret", "rule"), PREFIXED, ids=[p[0] for p in PREFIXED])
def test_r01_prefix_hits_print_the_rule_and_no_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ident: str,
    secret: str,
    rule: str,
) -> None:
    code, out, err = run_main(monkeypatch, capsys, diff_for("a.txt", [f"see {secret} now"]))
    assert (code, out) == (1, f"a.txt:1: - [{rule}]\n")
    assert_no_leak(secret, err)


def test_r01_a_range_argument_holding_a_value_is_not_echoed(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    v = rand(30)
    done = repo.run(f"{v}..HEAD")
    assert done.returncode == 2
    assert re.fullmatch(r"secret-scan: git diff failed \(exit \d+\)\n", done.stderr)
    assert done.stdout == ""
    assert_no_leak(v, done.stdout + done.stderr)


def test_r01_paths_are_printed_as_git_quotes_them(repo: Repo) -> None:
    done = repo.scan_files(
        {"a\tb.py": f'API_KEY = "{rand()}"\n', "e\x1b[31m.py": f'API_KEY = "{rand()}"\n'}
    )
    assert done.returncode == 1
    assert sorted(done.stdout.splitlines()) == [
        '"a\\tb.py":1: KEY [entropy]',
        '"e\\033[31m.py":1: KEY [entropy]',
    ]


def test_r01_the_hit_type_has_no_field_for_text() -> None:
    assert set(scan.Hit.__dataclass_fields__) == {"path", "line", "word", "rule"}


# ---------------------------------------------------------------------------
# R-02: bytes, split on the newline byte only
# ---------------------------------------------------------------------------
SEPARATORS = [b"\x0c", b"\x0b", chr(0x2028).encode(), b"\r", b"\xc2\x85", b"\x85"]


@pytest.mark.parametrize("sep", SEPARATORS, ids=["ff", "vt", "u2028", "cr", "nel", "raw85"])
def test_r02_a_line_separator_does_not_hide_the_next_added_line(repo: Repo, sep: bytes) -> None:
    content = b"# c" + sep + b"d\n" + f'API_KEY = "{rand()}"\n'.encode()
    done = repo.scan_files({"a.py": content})
    assert (done.returncode, done.stdout) == (1, "a.py:2: KEY [entropy]\n")


def test_r02_crlf_files_keep_correct_line_numbers(repo: Repo) -> None:
    done = repo.scan_files({"a.py": f'a = 1\r\nb = 2\r\nAPI_KEY = "{rand()}"\r\n'})
    assert (done.returncode, done.stdout) == (1, "a.py:3: KEY [entropy]\n")


# ---------------------------------------------------------------------------
# R-03: bounded time
# ---------------------------------------------------------------------------
TIMED_LINES = {
    "backslashes": 'API_KEY = "' + "\\" * 5000,
    "critic-regex": "const re = '" + "\\d+\\s\\d+\\s" * 14,
    "critic-regex-long": "const re = '" + "\\d+\\s\\d+\\s" * 2000,
    "dotted": "a." * 50000,
    "dotted-assign": "a." * 50000 + " = 1",
    "escaped-quotes": 'API_KEY = "' + '\\"' * 50000,
    "alternating-quotes": "a=\"'" * 20000,
    "single-quote-backslashes": "API_KEY = '" + "\\d+\\s" * 2000,
    "backtick-backslashes": "API_KEY = `" + "\\d+\\s" * 2000,
    "named-critic-line": "const API_KEY = '" + "\\d+\\s\\d+\\s" * 14,
    "jwt-spam": "eyJ" * 100000,
    "two-megabytes": "x = " + "a" * 2_000_000,
}


@pytest.mark.parametrize("key", list(TIMED_LINES))
def test_r03_pathological_lines_finish_quickly(key: str) -> None:
    with deadline(2.0):
        scan.scan_line("f.js", 1, TIMED_LINES[key], True, True, False, False)


def test_r03_a_two_megabyte_literal_is_still_scanned() -> None:
    with deadline(5.0):
        long = 'API_KEY = "' + rand() * 30000 + '"'
        assert len(scan.scan_line("f.py", 1, long, True, False, False, False)[0]) <= 1
        tail = "a=" * 50 + f'API_KEY = "{rand()}"'
        assert [h.word for h in scan.scan_line("f.py", 1, "x" * 2_000_000 + " " + tail)[0]] == [
            "KEY"
        ]


# ---------------------------------------------------------------------------
# R-04: --text; R-05: strict revision handling
# ---------------------------------------------------------------------------
def test_r04_the_git_argv_has_text_and_dashdash(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(scan.subprocess, "run", fake)
    scan.get_diff("A..B")
    assert "--text" in seen[0] and seen[0][-2:] == ["A..B", "--"]
    assert seen[1][-2:] == ["A..B", "--"]
    scan.get_diff(None)
    assert seen[2][-2:] == ["--cached", "--"]


@pytest.mark.parametrize("attr", ["-diff", "binary"])
def test_r04_diff_attributes_cannot_hide_a_secret(repo: Repo, attr: str) -> None:
    done = repo.scan_files({".gitattributes": f"a.py {attr}\n", "a.py": f'API_KEY = "{rand()}"\n'})
    assert (done.returncode, done.stdout) == (1, "a.py:1: KEY [entropy]\n")


def test_r05_a_working_tree_path_is_not_a_range(repo: Repo) -> None:
    repo.write("a.py", f'API_KEY = "{rand()}"\n')
    repo.commit()
    assert repo.run("a.py").returncode == 2


@pytest.mark.parametrize(
    "args", [["-x"], ["--output=zz"], ["A..B", "C..D"], ["A..B;touch", "zz"], ["a b"]]
)
def test_r05_bad_arguments_exit_2(repo: Repo, args: list[str]) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    done = repo.run(*args)
    assert (done.returncode, done.stdout) == (2, "")
    assert not (repo.path / "zz").exists()


def test_r05_a_shell_metacharacter_creates_nothing(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    assert repo.run("HEAD;touch x").returncode == 2
    assert not (repo.path / "x").exists()


@pytest.mark.parametrize("form", ["{a}..{b}", "{a}...{b}", "{a}"])
def test_r05_range_forms_work(repo: Repo, form: str) -> None:
    repo.write("a.py", "x = 1\n")
    a = repo.commit()
    repo.write("a.py", f'x = 1\nAPI_KEY = "{rand()}"\n')
    b = repo.commit()
    done = repo.run(form.format(a=a, b=b))
    assert done.returncode == 1


# ---------------------------------------------------------------------------
# R-06: coverage is visible, an empty scan fails
# ---------------------------------------------------------------------------
def test_r06_an_empty_staged_diff_exits_2_with_the_summary(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    done = repo.run()
    assert done.returncode == 2
    assert (
        "scanned 0 added lines in 0 files (0 allowed, 0 denied-type files, 0 binary files)"
        in done.stderr
    )


def test_r06_an_intent_to_add_only_state_exits_2(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    repo.write("b.py", f'API_KEY = "{rand()}"\n')
    repo.git("add", "-N", "b.py")
    assert repo.run().returncode == 2


def test_r06_allow_empty_passes_an_empty_range(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    head = repo.commit()
    assert repo.run(f"{head}..{head}").returncode == 2
    assert repo.run(f"{head}..{head}", "--allow-empty").returncode == 0
    assert repo.run("--allow-empty", f"{head}..{head}").returncode == 0


def test_r06_summary_numbers_on_a_three_file_diff(repo: Repo) -> None:
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 40 + b"ghp_" + ALPHABET[:36].encode()
    done = repo.scan_files(
        {
            "a.py": f'x = 1\nAPI_KEY = "{rand()}"  # secret-scan: allow (test fixture)\n',
            "package-lock.json": f'{{"integrity": "sha512-{rand(60)}"}}\n',
            "logo.png": png,
        }
    )
    assert done.returncode == 0
    assert done.stdout == ""
    assert (
        "scanned 3 added lines in 2 files (1 allowed, 1 denied-type files, 1 binary files)"
        in done.stderr
    )
    assert "a.py:2: allowed [entropy]" in done.stderr


# ---------------------------------------------------------------------------
# R-07 / R-08: file types and secret files
# ---------------------------------------------------------------------------
def _planted() -> dict[str, str]:
    v = rand()
    return {
        "config.json": f'{{"api_key": "{v}"}}\n',
        "deploy.sh": f'export API_KEY="{v}"\n',
        "c.yaml": f"api_key: {v}\n",
        "c.toml": f'api_key = "{v}"\n',
        "conf.ini": f"api_key = {v}\n",
        "main.go": f'apiKey := "{v}"\n',
        "Dockerfile": f"ENV API_KEY={v}\n",
        "Dockerfile.dev": f"ENV API_KEY {v}\n",
        "s.js": f'const API_KEY = "{v}";\n',
        "s.jsx": f'const API_KEY = "{v}";\n',
        "s.cjs": f'module.exports = {{ API_KEY: "{v}" }};\n',
        "s.ts": f"const API_KEY = '{v}';\n",
        "README.md": f"API_KEY={v}\n",
        "x.env.example": f"API_KEY={v}\n",
    }


@pytest.mark.parametrize("name", list(_planted()))
def test_r07_a_planted_secret_is_found_in_every_text_file_type(repo: Repo, name: str) -> None:
    done = repo.scan_files({name: _planted()[name]})
    assert done.returncode == 1, done.stderr
    assert re.fullmatch(rf"{re.escape(name)}:1: KEY \[(entropy|unquoted)\]\n", done.stdout)


def test_r07_a_pem_header_in_a_pem_file_and_in_a_deny_listed_lockfile(repo: Repo) -> None:
    assert repo.scan_files({"k.pem": PEM + "\n"}).returncode == 1
    token = "ghp_" + ALPHABET[:36]
    assert hits_for("package-lock.json", f'"resolved": "{token}"') == [(1, "-", "github-prefix")]


@pytest.mark.parametrize(
    "name",
    [
        "package-lock.json",
        "poetry.lock",
        "uv.lock",
        "a.min.js",
        "a.min.css",
        "a.map",
        "a.snap",
        "i.svg",
    ],
)
def test_r07_deny_listed_files_skip_the_entropy_rules(name: str) -> None:
    v = rand()
    assert (
        hits_for(
            name, f'"integrity": "sha512-{rand(60)}"', f'var API_KEY="{v}"', f'API_KEY = "{v}"'
        )
        == []
    )
    result = scan.scan_diff(diff_for(name, ["x = 1"]))
    assert (result.denied, result.files) == (1, 1)


def test_r07_a_lockfile_integrity_line_exits_0(repo: Repo) -> None:
    done = repo.scan_files({"package-lock.json": f'{{"integrity": "sha512-{rand(60)}="}}\n'})
    assert done.returncode == 0


def test_r07_binary_files_are_skipped_and_counted() -> None:
    body = "\x00\x01 " + "ghp_" + ALPHABET[:36]
    result = scan.scan_diff(diff_for("x.bin", [body]), {"x.bin"})
    assert (result.binary, result.files, result.hits) == (1, 0, [])
    text = body.replace("\x00", "")
    assert scan.scan_diff(diff_for("x.bin", [text]), {"x.bin"}).hits  # no NUL: text
    assert scan.scan_diff(diff_for("x.bin", [body]), set()).binary == 0  # not a numstat path


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        ".env.local",
        "k.pem",
        "deploy.key",
        "c.p12",
        "c.pfx",
        "id_rsa",
        "id_ed25519.pub",
        "sub/dir/.env.staging",
    ],
)
def test_r08_adding_a_secret_file_is_flagged_by_path_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    marker = "# secret-scan: allow (it is a fixture file)"
    diff = diff_for(name, [f"placeholder {marker}"], new=True)
    code, out, _err = run_main(monkeypatch, capsys, diff)
    assert (code, out) == (1, f"{name}:0: - [secret-file]\n")


@pytest.mark.parametrize("name", [".env.example", "env.txt", "key.md", "a.py", "dir/.envrc.md"])
def test_r08_other_new_files_are_not_secret_files(name: str) -> None:
    assert scan.scan_diff(diff_for(name, ["x = 1"], new=True)).hits == []


def test_r08_a_modified_or_deleted_secret_file_is_not_an_added_path() -> None:
    assert scan.scan_diff(diff_for(".env.production", ["x = 1"])).hits == []
    assert scan.scan_diff(diff_for(".env.production", [], deleted=True)).hits == []


def test_r08_a_real_secret_file_in_a_repository(repo: Repo) -> None:
    done = repo.scan_files({"deploy.key": "not a real key\n"})
    assert (done.returncode, done.stdout) == (1, "deploy.key:0: - [secret-file]\n")


# ---------------------------------------------------------------------------
# R-09: url credentials and prefixes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "line",
    [
        'DATABASE_URL = "postgresql+asyncpg://app:{p}@db:5432/x"',
        "REDIS_URL=redis://:{p}@cache:6379/0",
        "url: amqp://guest:{p}@host",
        "see https://u:{p}@example.org/",
    ],
)
def test_r09_a_url_with_a_random_password_is_found(line: str) -> None:
    name = "a.py" if "DATABASE" in line else "a.example"
    assert hits_for(name, line.format(p=rand(24)))[0][1:] == ("-", "url-credentials")


@pytest.mark.parametrize(
    "password",
    [
        "pw",
        "test1234567",
        "changeme-admin-test",
        "scratch_database_pass_word",
        "${DB_PASSWORD}",
        "$DB_PASSWORD_VALUE",
        "x" * 20,
    ],
)
def test_r09_low_entropy_word_shaped_or_referenced_passwords_pass(password: str) -> None:
    assert (
        hits_for("a.py", f'DATABASE_URL = "postgresql://test:{password}@localhost:5432/test"') == []
    )


@pytest.mark.parametrize(("ident", "secret", "rule"), PREFIXED, ids=[p[0] for p in PREFIXED])
@pytest.mark.parametrize("path", ["notes.txt", "a.py", "s.go", "x.min.js"])
def test_r09_each_prefix_is_found_unnamed_in_any_file(
    path: str, ident: str, secret: str, rule: str
) -> None:
    assert hits_for(path, f"see {secret} now") == [(1, "-", rule)]


def test_r09_bare_prefix_words_are_not_credentials() -> None:
    text = (
        "AKIA ids, ghp_ tokens, GOCSPX- secrets, eyJ headers and " + "-" * 5 + "BEGIN CERTIFICATE"
    )
    assert hits_for("a.md", text) == []


# ---------------------------------------------------------------------------
# R-10: whole-word names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "word"),
    [
        ("apiKey", "KEY"), ("API_KEYS", "KEY"), ("db_passwd", "PASSWD"), ("X-Admin-Key", "KEY"),
        ("bearerToken", "BEARER"), ("process.env.API_KEY", "KEY"), ("GOOD_HASH", "HASH"),
        ("postgres_dsn", "DSN"), ("DB_PASSPHRASE", "PASSPHRASE"), ("client_secret", "SECRET"),
        ("db_pwd", "PWD"), ("credentials", "CREDENTIAL"), ("APIKey", "KEY"),
    ],
)  # fmt: skip
def test_r10_names_that_match(name: str, word: str) -> None:
    assert scan.secret_word(name) == word


@pytest.mark.parametrize(
    "name",
    [
        "monkey",
        "turkey",
        "hockey",
        "monkeypatch",
        "hashlib",
        "keyboard",
        "apikey",
        "tokenizer",
        "x",
        "",
    ],
)
def test_r10_names_that_do_not_match(name: str) -> None:
    assert scan.secret_word(name) is None
    assert hits_for("a.py", f'{name} = "{rand()}"') == []


# ---------------------------------------------------------------------------
# R-11: assignment forms only
# ---------------------------------------------------------------------------
FORMS = [
    'API_KEY = "{v}"', 'API_KEY := "{v}"', 'API_KEY: "{v}"', "Client(api_key='{v}')",
    '"X-Api-Key": "{v}"', 'os.environ["API_KEY"] = "{v}"', 'process.env.API_KEY = "{v}";',
    'setenv("API_KEY", "{v}")', 'monkeypatch.setenv("API_KEY", "{v}")',
    'os.getenv("API_KEY", "{v}")',
    'API_KEY: str = "{v}"', "const API_KEY = `{v}`", 'headers = {{"x-api-key": "{v}"}}',
]  # fmt: skip


@pytest.mark.parametrize("form", FORMS)
def test_r11_each_assignment_form_is_found(form: str) -> None:
    assert [h[1:] for h in hits_for("a.py", form.format(v=rand()))] == [("KEY", "entropy")]


@pytest.mark.parametrize(
    "line",
    [
        'register("{v}", API_KEY)',
        'print("{v}", api_key_header)',
        'route = "/api/v1/{v}"; API_KEY_FILE = None',
        'if API_KEY == "{v}": pass',
        'other = "{v}"  # the API_KEY is elsewhere',
        'monkeypatch.setattr(app, "{v}", API_KEY)',
    ],
)
def test_r11_a_name_elsewhere_on_the_line_is_not_an_assignment(line: str) -> None:
    assert hits_for("a.py", line.format(v=rand())) == []


# ---------------------------------------------------------------------------
# R-12: false-positive filters
# ---------------------------------------------------------------------------
def test_r12_each_filter_on_its_own() -> None:
    v = rand()
    assert (
        scan.looks_like_path("/" + v)
        and scan.looks_like_path("./" + v)
        and scan.looks_like_path("../" + v)
    )
    assert scan.looks_like_path(v + ".json") and not scan.looks_like_path(v)
    assert scan.looks_like_word_segments("test-admin-api-key-for-unit-tests")
    assert scan.looks_like_word_segments("1234-abc_def.567")
    assert not scan.looks_like_word_segments("-".join(rand(14) for _ in range(3)))
    assert not scan.looks_like_word_segments(v)
    for literal in (
        "/" + v,
        v + ".json",
        "https://" + v,
        f"{v}@example.museum",
        str(uuid.uuid4()),
        "test-admin-api-key-for-unit-tests",
    ):
        assert not scan.is_candidate(literal), literal
    assert scan.is_candidate(v) and scan.is_candidate("-".join(rand(14) for _ in range(3)))


@pytest.mark.parametrize(
    "literal",
    [
        "/etc/{v}",
        "./{v}",
        "../{v}",
        "{v}.json",
        "https://{v}.example.org/x",
        "{v}@example.org",
        "test-admin-api-key-for-unit-tests",
    ],
)
def test_r12_filtered_literals_pass_in_a_secret_assignment(literal: str) -> None:
    assert hits_for("a.py", f'API_KEY = "{literal.format(v=rand())}"') == []


def test_r12_the_idempotency_uuid_form_passes() -> None:
    assert hits_for("a.ts", f'const OPTS = {{ idempotencyKey: "{uuid.uuid4()}" }} as const;') == []
    assert hits_for("a.ts", f'const OPTS = {{ idempotencyKey: "{uuid.uuid4()}" }}') == []


def test_r12_a_bcrypt_or_argon_hash_is_found() -> None:
    bcrypt = f'PASSWORD_HASH = "$2b$12${ALPHABET[:53]}"'
    argon = f'api_key_hash = "$argon2id$v=19$m=65536,t=3,p=4${rand(40)}"'
    assert [h[1:] for h in hits_for("a.py", bcrypt)] == [("PASSWORD", "entropy")]
    assert [h[1:] for h in hits_for("a.py", argon)] == [("KEY", "entropy")]


def test_r12_dollar_references_and_defaults() -> None:
    v = rand()
    assert hits_for("a.py", f'API_KEY = "${{API_KEY:-{v}}}"') == [(1, "KEY", "entropy")]
    assert hits_for("c.yml", f"API_KEY: ${{API_KEY:-{v}}}") == [(1, "KEY", "unquoted")]
    for ref in (
        "${SOME_LONG_VARIABLE_NAME_XYZ}",
        "$SOME_LONG_VARIABLE_NAME_XYZ",
        "$(generate-secret-value-xyz)",
        "${API_KEY:?must be set in the env file}",
    ):
        assert hits_for("a.py", f'API_KEY = "{ref}"') == []
        assert hits_for("c.yml", f"API_KEY: {ref}") == []


def test_r12_whitespace_and_braces_are_skipped() -> None:
    v = rand()
    assert hits_for("a.py", f'API_KEY = "{v} {v}"') == []
    assert hits_for("a.py", f'API_KEY = f"{v}-{{suffix}}"') == []
    assert hits_for("a.ts", f"const API_KEY = `{v}${{suffix}}`;") == []


def test_r12_markdown_takes_unquoted_forms_only() -> None:
    v = rand()
    assert hits_for("README.md", f'API_KEY = "{v}"', f"password = '{v}'") == []
    assert hits_for("README.md", f"API_KEY={v}") == [(1, "KEY", "unquoted")]
    assert hits_for("README.md", f"api_key: {v}") == [(1, "KEY", "unquoted")]
    assert hits_for("a.py", f"API_KEY = {v}") == []  # an unquoted value is not a literal in code


def test_r12_unquoted_values_in_config_files() -> None:
    assert hits_for("c.yml", f"API_KEY: {LOW}", "API_KEY=", "API_KEY=${API_KEY}") == []
    assert hits_for("c.yml", f"API_KEY: {rand()},") == [(1, "KEY", "unquoted")]


# ---------------------------------------------------------------------------
# R-13: the allow marker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("opener", ["#", "//", "/*", "<!--"])
def test_r13_a_marker_in_a_comment_with_a_reason_exempts_and_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], opener: str
) -> None:
    line = f'API_KEY = "{rand()}"  {opener} secret-scan: allow (public test vector)'
    code, out, err = run_main(monkeypatch, capsys, diff_for("a.py", [line]))
    assert (code, out) == (0, "")
    assert "a.py:1: allowed [entropy]" in err and "(1 allowed," in err


@pytest.mark.parametrize(
    "tail",
    [
        "# secret-scan: allow (short)",  # 5 characters
        "# secret-scan: allow (1234567890)",  # does not start with a letter
        "# secret-scan: allow ()",
        "# secret-scan: allow",
        '; other = "secret-scan: allow (valid reason here)"',  # inside a string, no comment
        '; other = "# secret-scan: allow (valid reason here)"',  # a comment opener inside a string
    ],
)
def test_r13_other_markers_do_not_exempt(tail: str) -> None:
    assert hits_for("a.py", f'API_KEY = "{rand()}"  {tail}') == [(1, "KEY", "entropy")]


def test_r13_prefix_and_secret_file_hits_cannot_be_exempted() -> None:
    marker = "# secret-scan: allow (documented example here)"
    assert hits_for("a.py", f'x = "{"AKIA" + UPPER[:16]}"  {marker}') == [(1, "-", "aws-prefix")]
    assert hits_for("a.py", f"{PEM}  {marker}") == [(1, "-", "private-key-prefix")]
    assert [h.rule for h in scan.scan_diff(diff_for("k.pem", [f"x {marker}"], new=True)).hits] == [
        "secret-file"
    ]


def test_r13_url_credentials_are_exemptible_and_a_prefix_beside_them_is_not() -> None:
    marker = "# secret-scan: allow (local scratch database)"
    url = f'DATABASE_URL = "postgresql://u:{rand(24)}@db/x"  {marker}'
    assert hits_for("a.py", url) == []
    both = url.replace("@db/x", f"@db/x{PREFIXED[0][1]}")
    assert [h[2] for h in hits_for("a.py", both)] == ["aws-prefix"]


# ---------------------------------------------------------------------------
# R-14 / R-16: exit codes and the documented gate contract
# ---------------------------------------------------------------------------
def test_r14_exit_codes_with_a_stubbed_diff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = diff_for("a.py", [f'API_KEY = "{LOW}"'])
    assert run_main(monkeypatch, capsys, clean)[0] == 0
    assert run_main(monkeypatch, capsys, diff_for("a.py", [f'API_KEY = "{HIGH}"']))[0] == 1
    assert run_main(monkeypatch, capsys, b"")[0] == 2
    assert run_main(monkeypatch, capsys, b"", "A..B", "--allow-empty")[0] == 0


def test_r14_outside_a_repository_exits_2_with_a_fixed_message(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
    }
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout == "" and re.fullmatch(
        r"secret-scan: git diff failed \(exit \d+\)\n", done.stderr
    )


def test_r14_git_missing_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(scan.shutil, "which", lambda _name: None)
    assert scan.main(["A..B"]) == 2
    assert capsys.readouterr().err == "secret-scan: git not found\n"


def test_r16_the_docstring_states_the_gate_contract() -> None:
    doc = scan.__doc__
    assert "<parent>..<commit>" in doc and "block on any non-zero exit" in doc
    assert "never use the staged default as a" in " ".join(doc.lower().split())
    for gap in ("multi-line assignments", "implicit concatenation", "diceware", "GitGuardian"):
        assert gap in " ".join(doc.split())


# ---------------------------------------------------------------------------
# diff parsing
# ---------------------------------------------------------------------------
def test_removed_lines_are_not_scanned() -> None:
    result = scan.scan_diff(diff_for("a.py", ["ok = 1"], removed=(f'API_KEY = "{rand()}"',)))
    assert result.hits == []


def test_line_numbers_follow_the_new_file_across_hunks() -> None:
    text = diff_for("a.py", ["ok = 1", f'API_KEY = "{rand()}"'], start=10)
    text += b"@@ -40,0 +50,1 @@\n" + f'+DB_PASSWORD = "{rand()}"\n'.encode()
    assert [(h.line, h.word) for h in scan.scan_diff(text).hits] == [(11, "KEY"), (50, "PASSWORD")]


def test_an_added_line_that_looks_like_a_file_header_is_still_scanned() -> None:
    text = diff_for("a.py", [f'++ API_KEY = "{rand()}"'])
    assert b"+++ API_KEY" in text
    assert [(h.line, h.word) for h in scan.scan_diff(text).hits] == [(1, "KEY")]


def test_a_removed_header_lookalike_does_not_hide_the_next_file() -> None:
    first = diff_for("a.py", ["x = 1"], removed=("-- old comment",))
    second = diff_for("b.py", [f'API_KEY = "{rand()}"'])
    assert [(h.path, h.line) for h in scan.scan_diff(first + second).hits] == [("b.py", 1)]


def test_deleted_files_and_no_newline_markers() -> None:
    deleted = diff_for("a.py", [], removed=(f'API_KEY = "{rand()}"',), deleted=True)
    assert scan.scan_diff(deleted).hits == []
    text = diff_for("a.py", ["a = 1"]) + b"\\ No newline at end of file\n"
    text += diff_for("b.py", [f'API_KEY = "{rand()}"'], start=3)
    assert [(h.path, h.line) for h in scan.scan_diff(text).hits] == [("b.py", 3)]


def test_removed_and_unchanged_lines_in_a_real_repository(repo: Repo) -> None:
    v = f'API_KEY = "{rand()}"'
    repo.write("a.py", f"{v}\na = 1\nb = 2\nc = 3\n")
    base = repo.commit("legacy offender")
    repo.write("a.py", f"{v}\na = 1\nb = 22\nc = 3\n")
    touched = repo.commit("touches another line")
    repo.write("a.py", "a = 1\nb = 22\nc = 3\n")
    removed = repo.commit("removes the offender")
    assert repo.run(f"{base}..{touched}").returncode == 0
    assert repo.run(f"{touched}..{removed}").returncode == 2  # nothing added: nothing scanned
    assert repo.run(f"{touched}..{removed}", "--allow-empty").returncode == 0
    assert repo.run(f"{removed}..{base}").returncode == 1  # the reverse diff adds it back


def test_staged_default_scans_what_is_staged(repo: Repo) -> None:
    repo.write("a.py", "x = 1\n")
    repo.commit()
    repo.write("a.py", f'x = 1\nAPI_KEY = "{rand()}"\n')
    assert repo.run().returncode == 2  # nothing staged: nothing scanned
    repo.git("add", "a.py")
    assert (repo.run().returncode, repo.run().stdout) == (1, "a.py:2: KEY [entropy]\n")


def test_a_real_binary_file_is_skipped_and_counted(repo: Repo) -> None:
    png = b"\x89PNG\x00\x00" + AWS_BYTES + b"\n" + b"\x00" * 10
    done = repo.scan_files({"logo.png": png, "a.py": "x = 1\n"})
    assert done.returncode == 0
    assert "1 binary files" in done.stderr


AWS_BYTES = ("AKIA" + UPPER[:16]).encode()


# ---------------------------------------------------------------------------
# the scanner passes on its own files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [SCRIPT, Path(__file__)], ids=["script", "tests"])
def test_the_scanner_passes_on_its_own_files(path: Path) -> None:
    shown = str(path.relative_to(ROOT))
    lines = path.read_text(encoding="utf-8").splitlines()
    result = scan.scan_diff(diff_for(shown, lines, new=True))
    assert result.hits == [] and result.lines == len(lines)
