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
    """Every src module classifies to a known layer, and each layer package exists."""
    modules = set(reorg_layers._module_files()) | reorg_layers._compiled_modules()
    layers = {m: reorg_layers.layer_of(m) for m in modules}
    assert not [m for m, layer in layers.items() if layer == "unclassified"]
    assert set(layers.values()) == {
        "core",
        "fragmentation",
        "fragnnet",
        "masster",
        "neims",
        "precursor",
        "gnn",
        "baseline:graff",
        "baseline:massformer",
    }
    assert "fragnnet.fragmentation" in modules


def _probe_all(targets: tuple[str, ...]) -> dict:
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reorg_layers.non_core_loaded, targets))
    return dict(zip(targets, results))


@pytest.fixture(scope="module")
def isolation() -> dict[str, dict]:
    """Non-core modules loaded by each entry module and by each model selection."""
    selection = _probe_all(tuple(f"model_type:{t}" for t in MODEL_TYPES))
    return {
        "entry_modules": _probe_all(ENTRY_MODULES),
        "model_selection": {k.split(":", 1)[1]: v for k, v in selection.items()},
    }


@pytest.mark.parametrize("group", ["entry_modules", "model_selection"])
def test_isolation_gate_matches_golden(isolation: dict[str, dict], group: str) -> None:
    ratchet_json("boundary", f"isolation_{group}", isolation[group])


def test_phase1_isolation_gate(isolation: dict[str, dict]) -> None:
    """The plan's Phase 1 gate, enforced independently of the golden file.

    No entry module loads a baseline or DGL, and selecting a model loads no DGL and no baseline
    other than its own.
    """
    entry_offenders = {
        module: sorted(layer for layer in loaded if layer.startswith("baseline:") or layer == "dgl")
        for module, loaded in isolation["entry_modules"].items()
    }
    assert {m: v for m, v in entry_offenders.items() if v} == {}
    own = {"graff": "baseline:graff", "massformer": "baseline:massformer"}
    selection_offenders = {
        model_type: sorted(
            layer
            for layer in loaded
            if layer == "dgl" or (layer.startswith("baseline:") and layer != own.get(model_type))
        )
        for model_type, loaded in isolation["model_selection"].items()
    }
    assert {t: v for t, v in selection_offenders.items() if v} == {}


# Plan gate table, Phase 3 row. script_utils stays report-only until Phase 4.
PHASE3_ENTRY_MODULES = (
    "fragnnet.runner",
    "fragnnet.pl_model.spectrum_pl",
    "fragnnet.dataset.spec_mol_dataset",
    "fragnnet.dataset.base_dataset",
    "fragnnet.utils.feat_utils",
    "fragnnet.utils.proc_utils",
    "fragnnet.dataset.data_sampler",
    "fragnnet.utils.pl_utils",
)
# Non-core layers a model selection may load: its own, plus fragmentation/ for the two fragment
# models. Since Phase 8 every model type has a layer of its own.
PHASE3_SELECTION_LAYERS = {
    "frag_gnn": {"fragnnet", "fragmentation"},
    "fragnnet": {"fragnnet", "fragmentation"},
    "masster": {"masster", "fragmentation"},
    "neims": {"neims"},
    "precursor": {"precursor"},
    "gnn": {"gnn"},
    "graff": {"baseline:graff"},
    "massformer": {"baseline:massformer"},
}


def test_phase3_isolation_gate(isolation: dict[str, dict]) -> None:
    """The plan's Phase 3 gate, enforced independently of the golden file.

    The Phase 3 entry modules load no module outside core (stricter than the plan's list of
    forbidden modules, and true after Phase 3), and a model selection loads only its own layer,
    plus fragmentation/ for FraGNNet and MASSter.
    """
    entry_offenders = {
        module: loaded
        for module, loaded in isolation["entry_modules"].items()
        if module in PHASE3_ENTRY_MODULES and loaded
    }
    assert entry_offenders == {}
    selection_offenders = {
        model_type: sorted(set(loaded) - PHASE3_SELECTION_LAYERS.get(model_type, set()))
        for model_type, loaded in isolation["model_selection"].items()
    }
    assert {t: v for t, v in selection_offenders.items() if v} == {}


def test_phase4_isolation_gate(isolation: dict[str, dict]) -> None:
    """The plan's Phase 4 gate: core is fully separated from the models.

    Every entry module, script_utils included, loads no module outside core, and no import edge
    anywhere breaks the dependency rule. Model selections are checked by the Phase 3 gate.
    """
    assert {m: loaded for m, loaded in isolation["entry_modules"].items() if loaded} == {}
    assert reorg_layers.boundary_violations() == {}
