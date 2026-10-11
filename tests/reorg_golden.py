"""Golden-record helpers for the repository reorganization safety net (plan Phase 0).

Records live under ``tests/golden/reorg/``. Regenerate deliberately, never to make a failing
refactor pass:

    REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_<name>.py

A golden comparison is exact. JSON records are compared key by key and report what is new,
changed or gone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).parent / "golden" / "reorg"
UPDATE = os.environ.get("REORG_UPDATE_GOLDEN") == "1"


def _path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def load_json(name: str) -> dict:
    path = _path(name)
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(name: str, key: str, value: Any) -> None:
    golden = load_json(name)
    golden[key] = value
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    _path(name).write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


def ratchet_json(name: str, key: str, current: dict) -> None:
    """Assert ``current`` equals the stored record, or store it when updating.

    Args:
        name: Golden file stem under ``tests/golden/reorg``.
        key: Record key inside that file.
        current: JSON-serializable mapping to compare.
    """
    current = json.loads(json.dumps(current, sort_keys=True, default=str))
    if UPDATE:
        write_json(name, key, current)
        return
    golden = load_json(name)
    assert key in golden, f"no golden {name}:{key}; run with REORG_UPDATE_GOLDEN=1"
    expected = golden[key]
    changed = {k: v for k, v in current.items() if k not in expected or v != expected[k]}
    gone = sorted(set(expected) - set(current))
    message = []
    if changed:
        message.append(f"{name}:{key} new or changed (fix the code, or regenerate if intended):")
        for k, v in sorted(changed.items()):
            message.append(f"  {k}: now {v!r}, was {expected.get(k, '<absent>')!r}")
    if gone:
        message.append(f"{name}:{key} no longer present (lock in with REORG_UPDATE_GOLDEN=1):")
        message += [f"  {k}" for k in gone]
    assert not message, "\n".join(message)


def environment_meta() -> dict:
    """Builds and CPU dispatch that bit-exact tensor records depend on."""
    import lightning
    import numpy
    import torch

    return {
        "torch": str(torch.__version__),
        "numpy": numpy.__version__,
        "lightning": lightning.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }


def ratchet_records(name: str, key: str, record: dict) -> None:
    """Exact comparison of a tensor-digest record, pinned to the environment that made it.

    Bit-exact records hold only for the same torch/numpy build and CPU instruction set (plan,
    Phase 0 determinism). On a different environment the comparison is skipped, not loosened;
    regenerate the records deliberately there.
    """
    import pytest

    meta = environment_meta()
    if not UPDATE:
        stored = load_json(name).get(f"{key}:env")
        if stored is not None and stored != meta:
            pytest.skip(f"{name}:{key} recorded on {stored}, running on {meta}; regenerate")
    else:
        write_json(name, f"{key}:env", meta)
    ratchet_json(name, key, record)
