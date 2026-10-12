"""Shared spectrum objectives in ``SpectrumPL`` (reorganization plan Phase 2).

The spectral-leak, spectral-CE-auxiliary and CE-pair terms and the Hellinger weight validation
live in ``SpectrumPL`` helpers that FraGNNet and MASSter call. ``test_reorg_losses.py`` holds the
equivalence proof (its bit-exact records did not change). This file checks the structure and the
helper contracts the records do not reach, such as FraGNNet's ``oos_loss`` auxiliary branch.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th

import fragnnet.pl_model.spectrum_pl as spectrum_pl
from fragnnet.pl_model.spectrum_pl import SpectrumPL

SHARED_LOSS_FUNCTIONS = {
    "ce_switch_pair_bce",
    "sparse_outside_window_intensity_fraction",
    "validate_hellinger_auxiliary_weights",
}
HELPERS = {
    "_validate_spectral_auxiliary_weights",
    "_spectral_leak_loss",
    "_spectral_ce_auxiliary",
    "_build_ce_pair_loss_fn",
}


@pytest.mark.parametrize(
    "module", ["fragnnet.models.fragnnet.fragnnet_pl", "fragnnet.pl_model.masster_pl"]
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


def _ce_pair_owner(weight: float, inst: list[str], prec: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        hparams=SimpleNamespace(
            spec_params={"inst_types": ["QTOF", "FT", "IT"], "prec_types": ["[M+H]+", "[M+Na]+"]},
            ce_pair_loss_weight=weight,
            ce_pair_eligible_inst_types=inst,
            ce_pair_eligible_prec_types=prec,
            ce_pair_min_endpoint_confidence=0.6,
            target_rel_ints_thresh=0.05,
        )
    )


def _batch(enabled: bool) -> dict:
    # Sorted labels: inst FT=0, IT=1, QTOF=2; prec [M+H]+=0, [M+Na]+=1.
    return {
        "ce_pair_enabled": enabled,
        "spec_inst_type": th.tensor([[2], [0], [2], [1]]),
        "spec_prec_type": th.tensor([[0], [0], [1], [0]]),
        "group_id": th.tensor([[0], [0], [1], [1]]),
        **{
            f"ce_pair_{k}": th.zeros(1)
            for k in (
                "true_mzs",
                "true_ints",
                "true_batch_idxs",
                "pred_mzs",
                "pred_logprobs",
                "pred_batch_idxs",
                "ce_values",
            )
        },
    }


def test_ce_pair_eligibility_uses_sorted_label_indices(monkeypatch) -> None:
    seen = {}

    def fake_bce(**kwargs):
        seen.update(kwargs)
        return th.ones(kwargs["batch_size"]), th.full((kwargs["batch_size"],), 2.0)

    monkeypatch.setattr(spectrum_pl, "ce_switch_pair_bce", fake_bce)
    owner = _ce_pair_owner(1.0, ["QTOF"], ["[M+H]+"])
    fn = SpectrumPL._build_ce_pair_loss_fn(owner)
    tol, min_mz = th.tensor([0.01] * 4), th.tensor([10.0] * 4)
    pair_loss, endpoints = fn(_batch(True), 4, th.zeros(3, dtype=th.float64), tol, min_mz)
    assert seen["eligible_samples"].tolist() == [True, False, False, False]
    assert seen["group_ids"].tolist() == [0, 0, 1, 1]
    assert seen["batch_size"] == 4
    assert seen["min_endpoint_confidence"] == 0.6
    assert seen["min_contender_relative_intensity"] == 0.05
    assert seen["tol_per_sample"] is tol and seen["min_mz_per_sample"] is min_mz
    assert pair_loss.tolist() == [1.0] * 4 and endpoints.tolist() == [2.0] * 4


def test_ce_pair_is_zero_like_the_reference_when_disabled(monkeypatch) -> None:
    def fail(**kwargs):
        raise AssertionError("ce_switch_pair_bce must not run when the term is disabled")

    monkeypatch.setattr(spectrum_pl, "ce_switch_pair_bce", fail)
    fn = SpectrumPL._build_ce_pair_loss_fn(_ce_pair_owner(1.0, ["QTOF"], ["[M+H]+"]))
    reference = th.zeros(3, dtype=th.float64)
    pair_loss, endpoints = fn(_batch(False), 4, reference, None, None)
    for out in (pair_loss, endpoints):
        assert out.dtype == th.float64 and out.tolist() == [0.0] * 4


def test_ce_pair_unknown_type_fails_at_setup_only_when_the_term_is_on() -> None:
    with pytest.raises(ValueError):
        SpectrumPL._build_ce_pair_loss_fn(_ce_pair_owner(1.0, ["Orbitrap"], ["[M+H]+"]))
    SpectrumPL._build_ce_pair_loss_fn(_ce_pair_owner(0.0, ["Orbitrap"], ["[M+H]+"]))
