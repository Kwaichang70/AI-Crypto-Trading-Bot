"""CF-97-49: the whole-file mode (``--files``) of ``scripts/check_secret_literals.py``.

Files are written to temporary directories, so nothing here touches the real ``reports/`` tree.
Every credential-shaped value is built at run time (``rand``, ``string`` constants).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_fihyg01_secret_literals import (
    ALPHABET,
    PEM,
    SCRIPT,
    Repo,
    assert_no_leak,
    deadline,
    rand,
    scan,
)


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def write(root: Path, name: str, content: str | bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def hit_lines(done: subprocess.CompletedProcess[str]) -> list[str]:
    return sorted(done.stdout.splitlines())


CONTENT = {
    "a.py": lambda v: (
        f'x = 1\nAPI_KEY = "{v}"\nDB_PASSWORD = "{v}"  # secret-scan: allow (test fixture)\n'
    ),
    "c.yml": lambda v: f"name: x\napi_key: {v}\n",
    "README.md": lambda v: f"text\nAPI_KEY={v}\n",
    "notes.txt": lambda v: f"see ghp_{ALPHABET[:36]} now\nurl postgresql://u:{v[:24]}@h/d\n",
    "package-lock.json": lambda v: f'{{"integrity": "sha512-{v}"}}\n',
    "Dockerfile": lambda v: f"FROM x\nENV API_KEY={v}\n",
}


# ---------------------------------------------------------------------------
# equivalence with the diff mode on the same content
# ---------------------------------------------------------------------------
def test_cf9749_files_mode_equals_diff_mode_on_the_same_content(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = Repo(tmp_path)
    repo.write("seed.txt", "seed\n")
    base = repo.commit("base")
    v = rand()
    for name, make in CONTENT.items():
        repo.write(name, make(v))
    head = repo.commit("adds")
    diff_done = repo.run(f"{base}..{head}")
    files_done = run(tmp_path, "--files", *CONTENT)
    assert hit_lines(files_done) == hit_lines(diff_done) != []
    assert files_done.returncode == diff_done.returncode == 1
    assert files_done.stderr.count(" allowed [") == diff_done.stderr.count(" allowed [") == 1
    assert "denied-type files" in files_done.stderr


def test_cf9749_a_clean_file_set_exits_0_in_both_modes(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "x = 1\ny = 2\n")
    done = run(tmp_path, "--files", "a.py")
    assert (done.returncode, done.stdout) == (0, "")
    assert "scanned 2 lines in 1 files" in done.stderr


# ---------------------------------------------------------------------------
# .patch / .diff: the content after the diff prefix, removed lines included
# ---------------------------------------------------------------------------
PATCH = (
    "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    '-API_KEY = "{a}"\n+API_KEY = "{b}"\n context = 1\n-DB_PASSWORD = "{c}"\n'
)


@pytest.mark.parametrize("suffix", [".patch", ".diff"])
def test_cf9749_patch_files_are_read_after_the_prefix_and_removed_lines_too(
    tmp_path: Path, suffix: str
) -> None:
    write(tmp_path, "p" + suffix, PATCH.format(a=rand(), b=rand(), c=rand()))
    done = run(tmp_path, "--files", "p" + suffix)
    assert hit_lines(done) == [
        f"p{suffix}:5: KEY [entropy]",
        f"p{suffix}:6: KEY [entropy]",
        f"p{suffix}:8: PASSWORD [entropy]",
    ]


def test_cf9749_a_dash_prefix_hides_a_name_in_a_plain_file_but_not_in_a_patch(
    tmp_path: Path,
) -> None:
    line = f'-API_KEY = "{rand()}"\n'
    write(tmp_path, "plain.txt", line)
    write(tmp_path, "same.patch", line)
    assert hit_lines(run(tmp_path, "--files", "plain.txt", "same.patch")) == [
        "same.patch:1: KEY [entropy]"
    ]


def test_cf9749_a_patch_line_without_a_prefix_is_scanned_as_it_is(tmp_path: Path) -> None:
    write(tmp_path, "p.patch", f'API_KEY = "{rand()}"\n')
    assert hit_lines(run(tmp_path, "--files", "p.patch")) == ["p.patch:1: KEY [entropy]"]


def test_cf9749_a_prefix_hit_in_a_patch_is_never_exemptible(tmp_path: Path) -> None:
    secret = "ghp_" + ALPHABET[:36]
    write(
        tmp_path, "p.patch", f"-x = 1  # secret-scan: allow (documented example)\n+y = '{secret}'\n"
    )
    assert hit_lines(run(tmp_path, "--files", "p.patch")) == ["p.patch:2: - [github-prefix]"]


# ---------------------------------------------------------------------------
# directories, symlinks, .git, secret files, binary, empty sets
# ---------------------------------------------------------------------------
def test_cf9749_directories_are_walked_recursively_in_sorted_order(tmp_path: Path) -> None:
    v = rand()
    for name in ("z/b.py", "a/deep/c.py", "a/b.py", "top.py"):
        write(tmp_path, name, f'API_KEY = "{v}"\n')
    write(tmp_path, ".git/config.py", f'API_KEY = "{v}"\n')
    done = run(tmp_path, "--files", ".")
    assert done.stdout.splitlines() == [
        "top.py:1: KEY [entropy]",  # a directory's files, then its sorted subdirectories
        "a/b.py:1: KEY [entropy]",
        "a/deep/c.py:1: KEY [entropy]",
        "z/b.py:1: KEY [entropy]",
    ]


def test_cf9749_symlinks_are_not_followed_out_of_the_given_roots(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    write(outside, "leak.py", f'API_KEY = "{rand()}"\n')
    write(root, "ok.py", "x = 1\n")
    (root / "link.py").symlink_to(outside / "leak.py")
    (root / "linkdir").symlink_to(outside, target_is_directory=True)
    (tmp_path / "rootlink").symlink_to(outside, target_is_directory=True)
    done = run(tmp_path, "--files", "root", "rootlink")
    assert (done.returncode, done.stdout) == (0, "")
    assert "3 symlinks skipped" in done.stderr and "scanned 1 lines in 1 files" in done.stderr


def test_cf9749_a_secret_file_is_reported_by_name_and_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path, "deploy.key", "not read\n")
    monkeypatch.setattr(scan, "_file_lines", lambda _p: pytest.fail("a secret file was read"))
    result = scan.scan_files([str(tmp_path / "deploy.key")])
    assert [h.rule for h in result.hits] == ["secret-file"] and result.files == 0
    done = run(tmp_path, "--files", "deploy.key")
    assert (done.returncode, done.stdout) == (1, "deploy.key:0: - [secret-file]\n")


def test_cf9749_binary_files_are_skipped_and_counted(tmp_path: Path) -> None:
    write(tmp_path, "b.bin", b"\x89PNG\x00\x00" + b"ghp_" + ALPHABET[:36].encode() + b"\n")
    write(tmp_path, "t.py", "x = 1\n")
    done = run(tmp_path, "--files", ".")
    assert (done.returncode, done.stdout) == (0, "")
    assert "1 binary files" in done.stderr


def test_cf9749_lockfiles_keep_only_the_prefix_rules(tmp_path: Path) -> None:
    write(
        tmp_path,
        "package-lock.json",
        f'{{"integrity": "{rand(60)}", "t": "ghp_{ALPHABET[:36]}"}}\n',
    )
    done = run(tmp_path, "--files", "package-lock.json")
    assert done.stdout == "package-lock.json:1: - [github-prefix]\n"
    assert "1 denied-type files" in done.stderr


def test_cf9749_an_empty_set_exits_2_and_allow_empty_passes(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    write(tmp_path, "zero.txt", "")
    assert run(tmp_path, "--files", "empty").returncode == 2
    assert run(tmp_path, "--files", "zero.txt").returncode == 2
    assert run(tmp_path, "--files", "empty", "--allow-empty").returncode == 0
    assert run(tmp_path, "--allow-empty", "--files", "zero.txt").returncode == 0


def test_cf9749_a_missing_path_exits_2_without_echoing_it(tmp_path: Path) -> None:
    value = rand(30)
    done = run(tmp_path, "--files", value)
    assert done.returncode == 2 and done.stdout == ""
    assert_no_leak(value, done.stderr)


def test_cf9749_unreadable_and_oversized_files_exit_2_unless_there_are_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path, "a.py", "x = 1\n")
    write(tmp_path, "big.txt", "x\n")
    real = scan.MAX_FILE_BYTES
    monkeypatch.setattr(scan, "MAX_FILE_BYTES", 1)
    result = scan.scan_files([str(tmp_path)])
    assert result.unreadable == [
        scan.quote_path(str(tmp_path / "a.py")),
        scan.quote_path(str(tmp_path / "big.txt")),
    ]
    monkeypatch.setattr(scan, "MAX_FILE_BYTES", real)
    sparse = tmp_path / "sparse.bin"
    with sparse.open("wb") as handle:
        handle.truncate(real + 1)  # sparse: no real disk or memory use
    done = run(tmp_path, "--files", "sparse.bin", "a.py")
    assert done.returncode == 2 and "sparse.bin: unreadable or too large" in done.stderr
    write(tmp_path, "hit.py", f'API_KEY = "{rand()}"\n')
    assert run(tmp_path, "--files", "sparse.bin", "hit.py").returncode == 1


def test_cf9749_a_fifo_is_unreadable(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("no fifo")
    os.mkfifo(tmp_path / "pipe")
    done = run(tmp_path, "--files", "pipe")
    assert done.returncode == 2 and "pipe: unreadable or too large" in done.stderr


# ---------------------------------------------------------------------------
# every swallowed OSError ends in the unreadable list (CR-HYG-18)
# ---------------------------------------------------------------------------
def _deny_scandir(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    real = os.scandir

    def fake(path: object = ".") -> object:
        if Path(str(path)).name == name:
            raise PermissionError(13, "denied", str(path))
        return real(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "scandir", fake)


def _main(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = scan.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cf9749_r2_an_unlistable_directory_exits_2_and_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "ok.py", "x = 1\n")
    write(tmp_path, "locked/x.py", f'API_KEY = "{rand()}"\n')
    _deny_scandir(monkeypatch, "locked")
    code, out, err = _main(capsys, "--files", str(tmp_path))
    assert (code, out) == (2, "")
    assert f"{tmp_path / 'locked'}: unreadable or too large" in err
    assert "1 unreadable" in err and "scanned 1 lines in 1 files" in err


def test_cf9749_r2_a_finding_elsewhere_gives_exit_1_and_the_directory_is_still_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "hit.py", f'API_KEY = "{rand()}"\n')
    write(tmp_path, "locked/x.py", "x = 1\n")
    _deny_scandir(monkeypatch, "locked")
    code, out, err = _main(capsys, "--files", str(tmp_path))
    assert (code, out) == (1, f"{tmp_path / 'hit.py'}:1: KEY [entropy]\n")
    assert f"{tmp_path / 'locked'}: unreadable or too large" in err


def test_cf9749_r2_an_unlistable_root_and_a_nested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "root", "a/b/c.py", "x = 1\n")
    _deny_scandir(monkeypatch, "root")
    assert _main(capsys, "--files", str(tmp_path / "root"))[0] == 2
    monkeypatch.undo()
    _deny_scandir(monkeypatch, "b")
    code, _out, err = _main(capsys, "--files", str(tmp_path / "root"))
    assert code == 2 and f"{tmp_path / 'root' / 'a' / 'b'}: unreadable or too large" in err


def test_cf9749_r2_an_unlistable_directory_name_is_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "ok.py", "x = 1\n")
    write(tmp_path, "e\x1b[31mdir/x.py", "x = 1\n")
    _deny_scandir(monkeypatch, "e\x1b[31mdir")
    code, _out, err = _main(capsys, "--files", str(tmp_path))
    assert code == 2 and "\x1b" not in err and "e\\033[31mdir" in err


def test_cf9749_r2_a_file_that_vanishes_or_cannot_be_opened_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "a.py", "x = 1\n")
    write(tmp_path, "gone.py", "x = 1\n")
    write(tmp_path, "deny.py", "x = 1\n")
    real_stat, real_open = os.stat, open

    def stat(path: object, *a: object, **k: object) -> object:
        if Path(str(path)).name == "gone.py":
            raise FileNotFoundError(2, "vanished", str(path))
        return real_stat(path, *a, **k)  # type: ignore[call-overload]

    def opener(file: object, *a: object, **k: object) -> object:
        if Path(str(file)).name == "deny.py":
            raise PermissionError(13, "denied", str(file))
        return real_open(file, *a, **k)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr("builtins.open", opener)
    code, out, err = _main(capsys, "--files", str(tmp_path))
    monkeypatch.undo()
    assert (code, out) == (2, "")
    for name in ("gone.py", "deny.py"):
        assert f"{tmp_path / name}: unreadable or too large" in err
    assert "2 unreadable" in err and "scanned 1 lines in 1 files" in err


def test_cf9749_r2_a_read_error_is_unreadable_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "a.py", "x = 1\n")

    class Broken:
        def __enter__(self) -> Broken:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def read(self) -> bytes:
            raise OSError(5, "I/O error")

    monkeypatch.setattr("builtins.open", lambda *_a, **_k: Broken())
    code, _out, err = _main(capsys, "--files", str(tmp_path / "a.py"))
    monkeypatch.undo()
    assert code == 2 and "a.py: unreadable or too large" in err


def test_cf9749_r2_the_docstring_lists_the_new_gaps_and_the_unlistable_rule() -> None:
    flat = " ".join(scan.__doc__.split())
    for phrase in (
        "a directory that cannot be listed",
        '``-NAME = "..."`` in a plain file',
        "command-line flag forms",
        "--api-key",
        "--password",
    ):
        assert phrase in flat


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        ["--files"],
        ["--files", "--allow-empty"],
        ["A..B", "--files", "a.py"],
        ["--files", "-x"],
        ["--files", "--files", "a.py"],
        ["--files", "--", "--"],
    ],
)
def test_cf9749_bad_files_arguments_exit_2(tmp_path: Path, args: list[str]) -> None:
    write(tmp_path, "a.py", "x = 1\n")
    done = run(tmp_path, *args)
    assert (done.returncode, done.stdout) == (2, "")
    assert done.stderr.startswith("secret-scan: usage:") or "does not exist" in done.stderr


def test_cf9749_dashdash_ends_the_options(tmp_path: Path) -> None:
    write(tmp_path, "-odd.py", f'API_KEY = "{rand()}"\n')
    assert run(tmp_path, "--files", "-odd.py").returncode == 2
    assert hit_lines(run(tmp_path, "--files", "--", "-odd.py")) == ["-odd.py:1: KEY [entropy]"]


def test_cf9749_diff_mode_usage_is_unchanged(tmp_path: Path) -> None:
    for args in (["-x"], ["A..B", "C..D"], ["a b"]):
        assert run(tmp_path, *args).returncode == 2


def test_cf9749_the_docstring_documents_the_mode() -> None:
    flat = " ".join(scan.__doc__.split())
    for phrase in (
        "--files",
        "symlinks",
        "never read",
        "``.patch`` and ``.diff``",
        "64 MiB",
        "excludes a revision",
    ):
        assert phrase in flat


# ---------------------------------------------------------------------------
# output contract: no value, no identifier, quoted hostile names
# ---------------------------------------------------------------------------
def test_cf9749_no_value_reaches_the_output(tmp_path: Path) -> None:
    values = [rand(), rand(30), rand(24)]
    write(tmp_path, "a.py", f'API_KEY = "{values[0]}"\n{values[1]}_token = "{values[2]}"\n')
    write(tmp_path, "c.yml", f"password: {values[0]}\nurl: postgresql://u:{values[2]}@h/d\n")
    write(tmp_path, "p.patch", f"-SECRET = '{values[1]}'\n+x = '{PEM}{values[0]}'\n")
    done = run(tmp_path, "--files", ".")
    assert done.returncode == 1
    for value in values:
        assert_no_leak(value, done.stdout + done.stderr)
    assert all(
        re.fullmatch(r"\S.*:\d+: (?:[A-Z]+|-) \[[a-z-]+\]", line)
        for line in done.stdout.splitlines()
    )


def test_cf9749_hostile_file_names_are_quoted_on_one_line(tmp_path: Path) -> None:
    v = rand()
    write(tmp_path, "a\tb.py", f'API_KEY = "{v}"\n')
    write(tmp_path, "e\x1b[31m.py", f'API_KEY = "{v}"\n')
    write(tmp_path, 'q"uote.py', f'API_KEY = "{v}"\n')
    write(tmp_path, "café.py", f'API_KEY = "{v}"\n')
    done = run(tmp_path, "--files", ".")
    assert hit_lines(done) == sorted(
        [
            '"a\\tb.py":1: KEY [entropy]',
            '"e\\033[31m.py":1: KEY [entropy]',
            '"q\\"uote.py":1: KEY [entropy]',
            '"caf\\303\\251.py":1: KEY [entropy]',
        ]
    )
    assert "\x1b" not in done.stdout + done.stderr


def test_cf9749_the_summary_counts(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "x = 1\ny = 2\n")
    write(tmp_path, "b.md", "z\n")
    done = run(tmp_path, "--files", "a.py", "b.md")
    expected = (
        "secret-scan: scanned 3 lines in 2 files (0 allowed, 0 denied-type files, "
        "0 binary files, 0 symlinks skipped, 0 unreadable)"
    )
    assert expected in done.stderr


# ---------------------------------------------------------------------------
# size and time
# ---------------------------------------------------------------------------
def test_cf9749_a_large_file_is_scanned_in_linear_time(tmp_path: Path) -> None:
    v = rand()
    body = ("x = 1  # a short line\n" * 99 + f'API_KEY = "{v}"\n') * 2000  # 2,000 hits, 200k lines
    write(tmp_path, "big.py", body)
    with deadline(20.0):
        result = scan.scan_files([str(tmp_path / "big.py")])
    assert result.lines == 200000 and len(result.hits) == 2000


def test_cf9749_a_very_long_line_and_pathological_lines_finish(tmp_path: Path) -> None:
    write(
        tmp_path,
        "long.py",
        "x = " + "a" * 2_000_000 + "\n" + 'API_KEY = "' + "\\" * 5000 + "\n" + "a." * 50000 + "\n",
    )
    with deadline(10.0):
        assert scan.scan_files([str(tmp_path / "long.py")]).hits == []
