import math

import torch as th

from fragnnet.model.fragnnet_model import (
    _apply_formula_support_gate,
    _formula_support_keep_mask,
)
from fragnnet.pl_model.fragnnet_pl import _formula_support_targets


def test_formula_support_gate_preserves_logits_at_initialization() -> None:
    intensity_logits = th.tensor([[2.0, 1.0, -3.0]])
    initial_logit = math.log(99.0)
    gated_logits = _apply_formula_support_gate(
        intensity_logits,
        th.full_like(intensity_logits, initial_logit),
        th.tensor([[True, True, False]]),
        initial_logit,
    )

    assert th.allclose(
        th.softmax(gated_logits, dim=1),
        th.softmax(intensity_logits, dim=1),
        rtol=0.0,
        atol=1e-7,
    )


def test_formula_support_gate_suppresses_unsupported_candidate() -> None:
    intensity_logits = th.zeros((1, 3))
    support_logits = th.tensor([[5.0, -5.0, 0.0]])
    valid_mask = th.tensor([[True, True, False]])
    gated_logits = _apply_formula_support_gate(
        intensity_logits,
        support_logits,
        valid_mask,
        5.0,
    )
    probabilities = th.softmax(gated_logits, dim=1)

    assert probabilities[0, 0] > 100.0 * probabilities[0, 1]
    assert th.isclose(probabilities[0, 0], probabilities[0, 2])


def test_formula_support_selection_uses_probabilities_and_keeps_map() -> None:
    logits = th.tensor([10.0, 5.0, -10.0, -11.0])
    keep = _formula_support_keep_mask(
        logits,
        th.ones(4, dtype=th.bool),
        th.tensor([0, 0, 1, 1]),
        probability_threshold=0.99,
    )

    assert keep.tolist() == [True, True, True, False]


def test_formula_support_targets_reduce_isotopes_to_formula() -> None:
    targets = _formula_support_targets(
        true_mzs=th.tensor([100.0005]),
        true_batch_idxs=th.tensor([0]),
        formula_support_logits=th.zeros(3),
        formula_peak_mzs=th.tensor([100.0, 101.003, 150.0]),
        formula_peak_formula_idxs=th.tensor([1, 1, 2]),
        formula_peak_batch_idxs=th.tensor([0, 0, 0]),
        batch_size=1,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        attribution="gaussian_posterior",
    )

    assert th.equal(targets, th.tensor([0.0, 1.0, 0.0]))


def test_formula_support_targets_skips_spectrum_with_no_candidate_peaks() -> None:
    """A spectrum whose fragments collapse to only the NULL formula has no peaks."""
    targets = _formula_support_targets(
        true_mzs=th.tensor([100.0005, 50.0]),
        true_batch_idxs=th.tensor([0, 1]),
        formula_support_logits=th.zeros(3),
        formula_peak_mzs=th.tensor([100.0, 101.003, 150.0]),
        formula_peak_formula_idxs=th.tensor([1, 1, 2]),
        formula_peak_batch_idxs=th.tensor([0, 0, 0]),
        batch_size=2,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        attribution="gaussian_posterior",
    )

    assert th.equal(targets, th.tensor([0.0, 1.0, 0.0]))
