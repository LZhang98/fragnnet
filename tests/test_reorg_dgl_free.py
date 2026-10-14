"""DGL-free main environment (reorganization plan Phase 1 step 4).

ICEBERG was the only DGL user and was retired in Phase 1 step 2, so nothing the repository
tracks may import, install or require DGL. The isolation gate in ``test_reorg_boundary.py``
blocks DGL at import time; these checks cover the install side.

The live environment is deliberately not checked: the editable install's metadata (and the
gitignored ``src/fragnnet.egg-info``) keep listing ``dgl`` until ``pip install --no-deps -e .``
is re-run, which the reorganization does not do.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Tracked files that install the main environment or document how to.
INSTALL_PATHSPECS = (
    "pyproject.toml",
    "uv.lock",
    "INSTALL.md",
    "env",
    "scripts/setup_job_cc.py",
    "scripts/manual_slurm",
)


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower()


def test_pyproject_requires_no_dgl() -> None:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    groups = {"dependencies": project["dependencies"]}
    groups.update(project.get("optional-dependencies", {}))
    offenders = {
        group: req
        for group, reqs in groups.items()
        for req in reqs
        if _requirement_name(req) == "dgl"
    }
    assert offenders == {}


def test_tracked_install_files_do_not_mention_dgl() -> None:
    result = subprocess.run(
        ["git", "grep", "-n", "-i", "-w", "-I", "dgl", "--", *INSTALL_PATHSPECS],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr  # 1 means no match
    assert result.stdout == ""


def _imported_modules(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_tracked_module_imports_dgl() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "src/*.py", "scripts/*.py", "preproc_scripts/*.py"],
        cwd=REPO,
        text=True,
    ).split()
    offenders = {}
    for path in tracked:
        if not (REPO / path).is_file():
            continue
        tree = ast.parse((REPO / path).read_text(), filename=path)
        hits = sorted(
            name
            for name in _imported_modules(tree)
            if name == "dgl" or name.startswith("dgl.") or name == "fragnnet.utils.dgl_compat_utils"
        )
        if hits:
            offenders[path] = hits
    assert offenders == {}
