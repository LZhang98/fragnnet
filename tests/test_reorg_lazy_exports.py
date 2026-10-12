"""No lazy package exports (reorganization plan Phase 6).

Phase 1 step 5 kept FraGNNet's and MASSter's classes reachable from ``fragnnet.model``,
``fragnnet.pl_model`` and ``fragnnet.dataset`` through a PEP 562 ``__getattr__``, so importing a
shared submodule did not load either model. Phase 6 moved those classes to ``fragnnet.models``
and removed the shims: the shared packages export core classes only, every importer names the
defining module, and no package initializer defines ``__getattr__``. Phase 8 step 0 removed
the remaining eager re-exports from ``fragnnet.model`` and ``fragnnet.pl_model`` (the core
models and the shared base classes), so their initializers import nothing. Phase 8 then renamed
``fragnnet.model`` to ``fragnnet.layers`` (shared building blocks) and ``fragnnet.loss``, so
the tables below check the successor package, and no module binds a fragnnet package to ``nn``.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

import pytest
from reorg_layers import REPO_ROOT, SRC_ROOT

FRAG = "fragnnet.models.fragnnet"
MASSTER = "fragnnet.models.masster"
# The names the shims used to export, and where they live now. fragnnet.model is now
# fragnnet.layers.
FORMER_LAZY_EXPORTS = {
    ("fragnnet.dataset", "CEContextDynamicBatchSampler"): f"{FRAG}.fragnnet_data_sampler",
    ("fragnnet.dataset", "DualGroupDynamicBatchSampler"): f"{FRAG}.fragnnet_data_sampler",
    ("fragnnet.dataset", "SpecMolFragDynamicBatchSampler"): f"{FRAG}.fragnnet_data_sampler",
    ("fragnnet.dataset", "IterableSpecMolFragDataset"): f"{FRAG}.fragnnet_dataset",
    ("fragnnet.dataset", "SpecMolFragDataset"): f"{FRAG}.fragnnet_dataset",
    ("fragnnet.dataset", "MassterCEContextDynamicBatchSampler"): f"{MASSTER}.masster_data_sampler",
    ("fragnnet.layers", "FraGNNetModel"): f"{FRAG}.fragnnet_model",
    ("fragnnet.layers", "MassterModel"): f"{MASSTER}.masster_model",
    ("fragnnet.pl_model", "FraGNNetPL"): f"{FRAG}.fragnnet_pl",
    ("fragnnet.pl_model", "MassterPL"): f"{MASSTER}.masster_pl",
}
# The names the shared initializers re-exported eagerly until Phase 8 step 0.
FORMER_EAGER_EXPORTS = {
    ("fragnnet.layers", "CEModel"): "fragnnet.layers.metadata_embedding",
    ("fragnnet.layers", "InstModel"): "fragnnet.layers.metadata_embedding",
    ("fragnnet.layers", "PrecModel"): "fragnnet.layers.metadata_embedding",
    ("fragnnet.layers", "GNNModel"): "fragnnet.models.gnn.gnn_model",
    ("fragnnet.layers", "NeimsModel"): "fragnnet.models.neims.neims_model",
    ("fragnnet.layers", "PrecursorModel"): "fragnnet.models.precursor.precursor_model",
    ("fragnnet.pl_model", "SpectrumPL"): "fragnnet.pl_model.spectrum_pl",
    ("fragnnet.pl_model", "BinnedPL"): "fragnnet.pl_model.binned_pl",
    ("fragnnet.pl_model", "NeimsPL"): "fragnnet.models.neims.neims_pl",
    ("fragnnet.pl_model", "PrecursorPL"): "fragnnet.models.precursor.precursor_pl",
    ("fragnnet.pl_model", "GNNPL"): "fragnnet.models.gnn.gnn_pl",
}
FORMER_EXPORTS = {**FORMER_LAZY_EXPORTS, **FORMER_EAGER_EXPORTS}


@pytest.mark.parametrize(("package", "name"), FORMER_EXPORTS, ids=lambda v: str(v))
def test_shared_packages_no_longer_export_model_classes(package: str, name: str) -> None:
    pkg = importlib.import_module(package)
    assert not hasattr(pkg, name)
    with pytest.raises(ImportError):
        exec(f"from {package} import {name}", {})


@pytest.mark.parametrize(("package", "name"), FORMER_EXPORTS, ids=lambda v: str(v))
def test_model_classes_live_in_their_model_package(package: str, name: str) -> None:
    module = FORMER_EXPORTS[package, name]
    assert getattr(importlib.import_module(module), name).__module__ == module


def test_no_package_initializer_defines_getattr() -> None:
    with_getattr = set()
    for init in (SRC_ROOT / "fragnnet").rglob("__init__.py"):
        tree = ast.parse(init.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
            with_getattr.add(".".join(init.parent.relative_to(SRC_ROOT).parts))
    assert not with_getattr


@pytest.mark.parametrize("package", ["fragnnet.layers", "fragnnet.loss", "fragnnet.pl_model"])
def test_shared_model_initializers_import_nothing(package: str) -> None:
    init = SRC_ROOT.joinpath(*package.split("."), "__init__.py")
    tree = ast.parse(init.read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_model_package_is_gone() -> None:
    """``fragnnet.model`` became ``fragnnet.layers`` and ``fragnnet.loss`` (Phase 8)."""
    assert not list(SRC_ROOT.joinpath("fragnnet", "model").glob("*.py"))


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO_ROOT / f for f in out]


def test_no_fragnnet_import_binds_nn() -> None:
    """``nn`` means ``torch.nn`` everywhere, so no fragnnet package or module is bound to it."""
    offenders = []
    for path in _tracked_python_files():
        try:
            if not path.is_file():
                continue
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                bound = [
                    a.asname or a.name.split(".")[0]
                    for a in node.names
                    if a.name.startswith("fragnnet")
                ]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fragnnet"):
                bound = [a.asname or a.name for a in node.names]
            else:
                continue
            if "nn" in bound:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []
