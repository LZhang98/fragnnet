"""uv.lock must match pyproject.toml.

After a dependency edit in ``pyproject.toml``, run ``uv lock`` and commit the lock with it;
otherwise ``uv sync --frozen`` installs stale versions. Skipped when uv is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_uv_lock_matches_pyproject() -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not installed")
    result = subprocess.run(
        [uv, "lock", "--check"], cwd=REPO, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
