"""Import-boundary ratchet for the repository reorganization (plan Phase 0 item 10).

Two checks, both compared with ``tests/golden/reorg/boundary.json``:

- The AST check lists every import edge that breaks the dependency rule, using the
  module-to-layer table in ``reorg_layers``.
- The isolation gate imports core entry modules, and selects each model type through the
  registry, in fresh subprocesses with DGL blocked, and records which non-core modules load.

The golden file is a two-sided ratchet. A new violation fails. A resolved violation also fails
until the golden file is regenerated, so every phase locks in its progress:

    REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_boundary.py

The plan's gate table says which entries must be empty after Phases 1, 3 and 4.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import reorg_layers
from reorg_golden import ratchet_json
from reorg_layers import ENTRY_MODULES, MODEL_TYPES


def test_ast_boundary_violations_match_golden() -> None:
    ratchet_json("boundary", "ast_violations", reorg_layers.boundary_violations())


def test_layer_table_covers_every_module() -> None:
    """Every src module classifies to a known layer, and the shared layers are not empty."""
    files = reorg_layers._module_files()
    layers = {reorg_layers.layer_of(m) for m in files}
    assert {"core", "fragmentation", "fragnnet", "masster"} <= layers
    for module in reorg_layers.FRAGMENTATION | reorg_layers.FRAGNNET | reorg_layers.MASSTER:
        assert module in files or module in reorg_layers._compiled_modules(), module


def _probe_all(targets: tuple[str, ...]) -> dict:
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reorg_layers.non_core_loaded, targets))
    return dict(zip(targets, results))


@pytest.mark.parametrize("group", ["entry_modules", "model_selection"])
def test_isolation_gate_matches_golden(group: str) -> None:
    if group == "entry_modules":
        current = _probe_all(ENTRY_MODULES)
    else:
        current = {
            k.split(":", 1)[1]: v
            for k, v in _probe_all(tuple(f"model_type:{t}" for t in MODEL_TYPES)).items()
        }
    ratchet_json("boundary", f"isolation_{group}", current)
