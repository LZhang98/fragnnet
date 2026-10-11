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

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import reorg_layers

GOLDEN = Path(__file__).parent / "golden" / "reorg" / "boundary.json"
UPDATE = os.environ.get("REORG_UPDATE_GOLDEN") == "1"

ENTRY_MODULES = (
    "fragnnet.runner",
    "fragnnet.utils.script_utils",
    "fragnnet.pl_model.spectrum_pl",
    "fragnnet.dataset.spec_mol_dataset",
    "fragnnet.dataset.base_dataset",
    "fragnnet.utils.feat_utils",
    "fragnnet.utils.proc_utils",
    "fragnnet.dataset.data_sampler",
    "fragnnet.utils.pl_utils",
)
MODEL_TYPES = (
    "frag_gnn",
    "fragnnet",
    "masster",
    "neims",
    "chemlm",
    "massformer",
    "iceberg_gen",
    "iceberg_inten",
    "graff",
    "precursor",
    "gnn",
    "mces_pretraining",
    "clip",
    "spec_mol_clip_pretraining",
)


def _load_golden() -> dict:
    if not GOLDEN.exists():
        return {}
    return json.loads(GOLDEN.read_text())


def _write_golden(key: str, value: dict) -> None:
    golden = _load_golden()
    golden[key] = value
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


def _ratchet(key: str, current: dict) -> None:
    if UPDATE:
        _write_golden(key, current)
        return
    golden = _load_golden()
    assert key in golden, f"no golden for {key!r}; run with REORG_UPDATE_GOLDEN=1"
    expected = golden[key]
    new = {k: v for k, v in current.items() if k not in expected or v != expected[k]}
    gone = sorted(set(expected) - set(current))
    message = []
    if new:
        message.append("new or changed (fix the code, or regenerate if intended):")
        message += [f"  {k}: {v}" for k, v in sorted(new.items())]
    if gone:
        message.append("resolved (lock in with REORG_UPDATE_GOLDEN=1):")
        message += [f"  {k}" for k in gone]
    assert not message, "\n".join(message)


def test_ast_boundary_violations_match_golden() -> None:
    _ratchet("ast_violations", reorg_layers.boundary_violations())


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
    _ratchet(f"isolation_{group}", current)
