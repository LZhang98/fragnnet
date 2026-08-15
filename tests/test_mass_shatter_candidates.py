"""Tests for sparse MASShatter cut-mask/component batching."""

import pytest
import torch as th

from fragnnet.frag.mass_shatter_candidates import build_mass_shatter_candidates


def test_candidate_batch_keeps_molecules_and_masks_separate() -> None:
    logits = th.tensor([0.4, -0.7], requires_grad=True)
    candidates = build_mass_shatter_candidates(
        logits,
        chemical_bond_index=th.tensor([[0, 2], [1, 3]], dtype=th.long),
        atom_batch_indices=th.tensor([0, 0, 1, 1], dtype=th.long),
        atom_ptr=th.tensor([0, 2, 4], dtype=th.long),
        max_masks=2,
        explore_masks=0,
        exploration_temperature=1.0,
        include_single_bond_cuts=True,
    )

    assert candidates.mask_log_probabilities.shape == (4,)
    assert candidates.mask_batch_indices.tolist() == [0, 0, 1, 1]
    assert candidates.component_mask_indices.tolist() == [0, 0, 1, 2, 3, 3]
    assert candidates.component_batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert candidates.component_cut_counts.tolist() == [1, 1, 0, 0, 1, 1]

    membership_batches = th.tensor([0, 0, 1, 1])[candidates.atom_indices]
    component_batches = candidates.component_batch_indices[candidates.atom_component_indices]
    assert th.equal(membership_batches, component_batches)
    assert candidates.boundary_bond_indices.tolist() == [0, 0, 1, 1]

    candidates.mask_log_probabilities.sum().backward()
    assert logits.grad is not None
    assert th.isfinite(logits.grad).all()


def test_candidate_batch_supports_bondless_precursor() -> None:
    candidates = build_mass_shatter_candidates(
        th.empty(0, requires_grad=True),
        chemical_bond_index=th.empty((2, 0), dtype=th.long),
        atom_batch_indices=th.tensor([0], dtype=th.long),
        atom_ptr=th.tensor([0, 1], dtype=th.long),
        max_masks=100,
        explore_masks=0,
        exploration_temperature=1.0,
        include_single_bond_cuts=True,
    )

    assert candidates.mask_log_probabilities.tolist() == [0.0]
    assert candidates.component_mask_indices.tolist() == [0]
    assert candidates.atom_indices.tolist() == [0]
    assert candidates.complement_atom_indices.numel() == 0
    assert candidates.boundary_bond_indices.numel() == 0


def test_candidate_batch_forces_empty_and_single_cut_exploration() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0, 2.0]),
        chemical_bond_index=th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.long),
        atom_batch_indices=th.zeros(4, dtype=th.long),
        atom_ptr=th.tensor([0, 4], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_temperature=1.0,
        include_single_bond_cuts=True,
    )

    # MAP all-cut mask + forced empty mask + each of the three single cuts.
    assert candidates.mask_log_probabilities.shape[0] == 5
    assert sorted(candidates.component_cut_counts.unique().tolist()) == [0, 1, 3]


def test_mixed_top_k_retains_exploitation_and_is_seeded() -> None:
    kwargs = {
        "bond_logits": th.tensor([2.0, 0.7, -0.4, -1.3]),
        "chemical_bond_index": th.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=th.long
        ),
        "atom_batch_indices": th.zeros(5, dtype=th.long),
        "atom_ptr": th.tensor([0, 5], dtype=th.long),
        "max_masks": 4,
        "explore_masks": 2,
        "exploration_temperature": 1.0,
        "include_single_bond_cuts": False,
    }

    first = build_mass_shatter_candidates(
        **kwargs,
        generator=th.Generator().manual_seed(17),
    )
    second = build_mass_shatter_candidates(
        **kwargs,
        generator=th.Generator().manual_seed(17),
    )

    # Four learned masks plus the forced empty mask outside the learned budget.
    assert first.mask_log_probabilities.shape == (5,)
    assert th.equal(first.mask_log_probabilities, second.mask_log_probabilities)

    logits = kwargs["bond_logits"]
    expected_map_log_probability = (
        th.nn.functional.logsigmoid(logits[:2]).sum()
        + th.nn.functional.logsigmoid(-logits[2:]).sum()
    )
    expected_second_log_probability = (
        th.nn.functional.logsigmoid(logits[:3]).sum()
        + th.nn.functional.logsigmoid(-logits[3])
    )
    assert th.isclose(first.mask_log_probabilities[0], expected_map_log_probability)
    assert th.isclose(first.mask_log_probabilities[1], expected_second_log_probability)
    assert first.mask_log_probabilities.unique().numel() == 5


def test_mixed_top_k_validates_exploration_parameters() -> None:
    common_kwargs = {
        "bond_logits": th.tensor([0.0]),
        "chemical_bond_index": th.tensor([[0], [1]], dtype=th.long),
        "atom_batch_indices": th.zeros(2, dtype=th.long),
        "atom_ptr": th.tensor([0, 2], dtype=th.long),
        "max_masks": 2,
        "include_single_bond_cuts": False,
    }

    with pytest.raises(ValueError, match="explore_masks"):
        build_mass_shatter_candidates(
            **common_kwargs,
            explore_masks=2,
            exploration_temperature=1.0,
        )
    with pytest.raises(ValueError, match="exploration_temperature"):
        build_mass_shatter_candidates(
            **common_kwargs,
            explore_masks=1,
            exploration_temperature=0.0,
        )
