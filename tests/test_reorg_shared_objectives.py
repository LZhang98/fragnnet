"""Shared spectrum objectives in ``SpectrumPL`` (reorganization plan Phase 2).

The spectral-leak, spectral-CE-auxiliary terms and the Hellinger weight validation
live in ``SpectrumPL`` helpers that FraGNNet and MASSter call. ``test_reorg_losses.py`` holds the
equivalence proof (its bit-exact records did not change). This file checks the structure and the
helper contracts the records do not reach, such as FraGNNet's ``oos_loss`` auxiliary branch.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as th

from fragnnet.pl_model.spectrum_pl import SpectrumPL

# validate_hellinger_auxiliary_weights was inlined into SpectrumPL in Phase 8.
SHARED_LOSS_FUNCTIONS = {
    "sparse_outside_window_intensity_fraction",
}
HELPERS = {
    "_validate_spectral_auxiliary_weights",
    "_spectral_leak_loss",
    "_spectral_ce_auxiliary",
}


@pytest.mark.parametrize(
    "module", ["fragnnet.models.fragnnet.fragnnet_pl", "fragnnet.models.masster.masster_pl"]
)
def test_models_use_the_shared_helpers_not_copies(module: str) -> None:
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not imported & SHARED_LOSS_FUNCTIONS
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }
    assert called >= HELPERS


def _stub_sparse_ce_fn(calls: list):
    ios, oos = th.tensor([0.5, 1.5]), th.tensor([0.25, 0.75])

    def sparse_ce_fn(*args, **kwargs):
        calls.append((args, kwargs))
        return ios, oos, th.zeros(2), th.zeros(2)

    return sparse_ce_fn, ios, oos


@pytest.mark.parametrize("include_oos", [False, True])
def test_spectral_ce_auxiliary_adds_oos_only_when_asked(include_oos: bool) -> None:
    calls: list = []
    sparse_ce_fn, ios, oos = _stub_sparse_ce_fn(calls)
    inputs = [th.full((1,), float(i)) for i in range(7)]
    tol, min_mz = th.tensor([0.01]), th.tensor([10.0])
    result = SpectrumPL._spectral_ce_auxiliary(
        sparse_ce_fn,
        *inputs,
        include_oos=include_oos,
        tol_per_sample=tol,
        min_mz_per_sample=min_mz,
    )
    if include_oos:
        assert th.equal(result, ios + oos)
    else:
        assert result is ios
    (args, kwargs) = calls[0]
    assert all(a is b for a, b in zip(args, inputs, strict=True))
    assert kwargs == {"tol_per_sample": tol, "min_mz_per_sample": min_mz}
