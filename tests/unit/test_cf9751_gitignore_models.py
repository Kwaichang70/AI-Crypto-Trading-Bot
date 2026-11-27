"""
tests/unit/test_cf9751_gitignore_models.py
------------------------------------------
CF-97-51: the ``models/`` rule of ``.gitignore`` is meant for ML model artefacts (the
repo-root ``models/`` directory that docker-compose mounts at ``/app/models``, plus
``model_artifacts/``). It also matched the UI route directory ``apps/ui/src/app/models/``, so
``git add`` of a NEW file there was skipped silently. The fix is the negation
``!apps/ui/src/app/models/`` after the rule.

The pins run ``git check-ignore --no-index`` against a throwaway repository that holds a copy
of the real ``.gitignore`` (so they do not depend on the checkout state and never touch the
real index). Planted-rule mutants prove that the pins fail for the broken variants. They are
skipped cleanly when ``git`` or the ``.gitignore`` is not available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GITIGNORE = _REPO_ROOT / ".gitignore"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not _GITIGNORE.is_file(),
    reason="git or the repository .gitignore is not available",
)

# Paths that are model artefacts or other currently ignored paths: they must stay ignored.
_STAYS_IGNORED = (
    "models/BTC_active.json",
    "models/new.txt",
    "models/x.pkl",
    "models/.gitkeep",  # tracked by force today; the rule itself still matches it
    "apps/api/models/BTC_active.json",
    "packages/data/models/m.bin",
    "model_artifacts/a.bin",
    "apps/ui/src/app/models/x.pkl",  # artefact extensions stay ignored inside the UI directory
    "apps/ui/src/models/a.ts",
)
# New source files in the UI route directory must NOT be ignored.
_MUST_BE_TRACKABLE = (
    "apps/ui/src/app/models/page.tsx",
    "apps/ui/src/app/models/new.tsx",
    "apps/ui/src/app/models/sub/a.tsx",
    "apps/ui/src/app/models/_helper.ts",
)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    git = shutil.which("git") or "git"
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, paths are test constants
        [git, *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def _make_repo(root: Path, gitignore_text: str) -> Path:
    root.mkdir()
    assert _git(["init", "-q"], root).returncode == 0
    (root / ".gitignore").write_text(gitignore_text, encoding="utf-8")
    return root


def _is_ignored(repo: Path, path: str) -> bool:
    result = _git(["check-ignore", "--no-index", "-q", "--", path], repo)
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


def _problems(repo: Path) -> list[str]:
    out = [f"must stay ignored: {p}" for p in _STAYS_IGNORED if not _is_ignored(repo, p)]
    out += [f"must not be ignored: {p}" for p in _MUST_BE_TRACKABLE if _is_ignored(repo, p)]
    return out


@pytest.fixture(scope="module")
def real_text() -> str:
    return _GITIGNORE.read_text(encoding="utf-8")


class TestRealRules:
    def test_the_real_gitignore_has_no_problems(self, tmp_path: Path, real_text: str) -> None:
        assert _problems(_make_repo(tmp_path / "r", real_text)) == []

    def test_a_new_file_in_the_checkout_is_not_ignored(self) -> None:
        # Against the real checkout (skipped when the tests run outside a git work tree).
        if _git(["rev-parse", "--is-inside-work-tree"], _REPO_ROOT).returncode != 0:
            pytest.skip("not a git checkout")
        new = "apps/ui/src/app/models/does-not-exist-yet.tsx"
        assert _git(["check-ignore", "-q", "--", new], _REPO_ROOT).returncode == 1
        assert (
            _git(["check-ignore", "-q", "--", "models/BTC_active.json"], _REPO_ROOT).returncode == 0
        )

    def test_the_negation_follows_the_models_rule(self, real_text: str) -> None:
        lines = [ln.strip() for ln in real_text.splitlines()]
        assert lines.count("models/") == 1
        assert lines.count("!apps/ui/src/app/models/") == 1
        assert lines.index("models/") < lines.index("!apps/ui/src/app/models/")


class TestPlantedRuleMutants:
    """Each broken variant of the rule must make the pin report a problem."""

    def _mutant(self, tmp_path: Path, real_text: str, old: str, new: str) -> Path:
        assert real_text.count(old) == 1, old
        return _make_repo(tmp_path / "m", real_text.replace(old, new))

    def test_original_rule_without_the_negation(self, tmp_path: Path, real_text: str) -> None:
        repo = self._mutant(tmp_path, real_text, "!apps/ui/src/app/models/\n", "")
        assert "must not be ignored: apps/ui/src/app/models/new.tsx" in _problems(repo)

    def test_negation_placed_before_the_rule_is_ineffective(
        self, tmp_path: Path, real_text: str
    ) -> None:
        text = real_text.replace("!apps/ui/src/app/models/\n", "")
        assert text.count("\nmodels/\n") == 1
        text = text.replace("\nmodels/\n", "\n!apps/ui/src/app/models/\nmodels/\n")
        repo = _make_repo(tmp_path / "m", text)
        assert "must not be ignored: apps/ui/src/app/models/new.tsx" in _problems(repo)

    def test_root_anchored_rule_loses_other_ignored_paths(
        self, tmp_path: Path, real_text: str
    ) -> None:
        # Anchoring models/ unignores apps/api/models/ and packages/data/models/.
        repo = self._mutant(tmp_path, real_text, "\nmodels/\n", "\n/models/\n")
        problems = _problems(repo)
        assert "must stay ignored: apps/api/models/BTC_active.json" in problems
        assert "must stay ignored: packages/data/models/m.bin" in problems

    def test_negation_that_also_unignores_artefacts_is_caught(
        self, tmp_path: Path, real_text: str
    ) -> None:
        repo = self._mutant(
            tmp_path,
            real_text,
            "!apps/ui/src/app/models/\n",
            "!apps/ui/src/app/models/\n!models/\n",
        )
        assert "must stay ignored: models/BTC_active.json" in _problems(repo)

    def test_models_rule_removed_entirely(self, tmp_path: Path, real_text: str) -> None:
        repo = self._mutant(tmp_path, real_text, "\nmodels/\n", "\n")
        assert "must stay ignored: models/BTC_active.json" in _problems(repo)
