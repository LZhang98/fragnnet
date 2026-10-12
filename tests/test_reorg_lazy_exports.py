"""No lazy package exports (reorganization plan Phase 6).

Phase 1 step 5 kept FraGNNet's and MASSter's classes reachable from ``fragnnet.model``,
``fragnnet.pl_model`` and ``fragnnet.dataset`` through a PEP 562 ``__getattr__``, so importing a
shared submodule did not load either model. Phase 6 moved those classes to ``fragnnet.models``
and removed the shims: the shared packages export core classes only, every importer names the
defining module, and no package initializer defines ``__getattr__``.
"""

from __future__ import annotations

import ast
import importlib

import pytest
from reorg_layers import SRC_ROOT

FRAG = "fragnnet.models.fragnnet"
MASSTER = "fragnnet.models.masster"
# The names the shims used to export, and where they live now.
FORMER_LAZY_EXPORTS = {
    ("fragnnet.dataset", "CEContextDynamicBatchSampler"): f"{FRAG}.frag_data_sampler",
    ("fragnnet.dataset", "DualGroupDynamicBatchSampler"): f"{FRAG}.frag_data_sampler",
    ("fragnnet.dataset", "SpecMolFragDynamicBatchSampler"): f"{FRAG}.frag_data_sampler",
    ("fragnnet.dataset", "IterableSpecMolFragDataset"): f"{FRAG}.iterable_spec_mol_frag_dataset",
    ("fragnnet.dataset", "SpecMolFragDataset"): f"{FRAG}.spec_mol_frag_dataset",
    ("fragnnet.dataset", "MassterCEContextDynamicBatchSampler"): f"{MASSTER}.masster_data_sampler",
    ("fragnnet.model", "FraGNNetModel"): f"{FRAG}.fragnnet_model",
    ("fragnnet.model", "MassterModel"): f"{MASSTER}.masster_model",
    ("fragnnet.pl_model", "FraGNNetPL"): f"{FRAG}.fragnnet_pl",
    ("fragnnet.pl_model", "MassterPL"): f"{MASSTER}.masster_pl",
}


@pytest.mark.parametrize(("package", "name"), FORMER_LAZY_EXPORTS, ids=lambda v: str(v))
def test_shared_packages_no_longer_export_model_classes(package: str, name: str) -> None:
    pkg = importlib.import_module(package)
    assert not hasattr(pkg, name)
    with pytest.raises(ImportError):
        exec(f"from {package} import {name}", {})


@pytest.mark.parametrize(("package", "name"), FORMER_LAZY_EXPORTS, ids=lambda v: str(v))
def test_model_classes_live_in_their_model_package(package: str, name: str) -> None:
    module = FORMER_LAZY_EXPORTS[package, name]
    assert getattr(importlib.import_module(module), name).__module__ == module


def test_no_package_initializer_defines_getattr() -> None:
    with_getattr = set()
    for init in (SRC_ROOT / "fragnnet").rglob("__init__.py"):
        tree = ast.parse(init.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
            with_getattr.add(".".join(init.parent.relative_to(SRC_ROOT).parts))
    assert not with_getattr
