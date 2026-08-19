"""Tests for sparse MASShatter cut-mask/component batching."""

import pytest
import torch as th

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterStaticGraph,
    build_mass_shatter_candidates,
)

BOUNDED_CUT_COUNT_WEIGHTS = (0.0, 0.0, 0.35, 0.30, 0.20, 0.10, 0.05)


def test_candidate_batch_keeps_molecules_and_masks_separate() -> None:
    logits = th.tensor([0.4, -0.7], requires_grad=True)
    candidates = build_mass_shatter_candidates(
        logits,
        chemical_bond_index=th.tensor([[0, 2], [1, 3]], dtype=th.long),
        atom_batch_indices=th.tensor([0, 0, 1, 1], dtype=th.long),
        atom_ptr=th.tensor([0, 2, 4], dtype=th.long),
        max_masks=2,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
    )

    assert candidates.mask_log_probabilities.shape == (4,)
    assert candidates.mask_batch_indices.tolist() == [0, 0, 1, 1]
    assert candidates.mask_cut_counts.tolist() == [1, 0, 0, 1]
    assert candidates.component_mask_indices.tolist() == [0, 0, 1, 2, 3, 3]
    assert candidates.component_batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert candidates.component_cut_counts.tolist() == [1, 1, 0, 0, 1, 1]
    assert candidates.proposal_single_cut_fraction.tolist() == pytest.approx([0.5, 0.5])
    assert candidates.unique_component_count.tolist() == [3.0, 3.0]

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
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
    )

    assert candidates.mask_log_probabilities.tolist() == [0.0]
    assert candidates.component_mask_indices.tolist() == [0]
    assert candidates.atom_indices.tolist() == [0]
    assert candidates.boundary_bond_indices.numel() == 0
    assert candidates.timings.total_ms >= 0.0


@pytest.mark.parametrize("num_atoms", [256, 257])
def test_candidate_membership_supports_four_compiled_words_and_fallback(
    num_atoms: int,
) -> None:
    candidates = build_mass_shatter_candidates(
        th.full((num_atoms - 1,), -1.0),
        chemical_bond_index=th.tensor(
            [range(num_atoms - 1), range(1, num_atoms)],
            dtype=th.long,
        ),
        atom_batch_indices=th.zeros(num_atoms, dtype=th.long),
        atom_ptr=th.tensor([0, num_atoms], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=False,
    )

    assert candidates.atom_indices.tolist() == list(range(num_atoms))
    assert candidates.atom_component_indices.tolist() == [0] * num_atoms


def test_candidate_batch_forces_empty_and_single_cut_exploration() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0, 2.0]),
        chemical_bond_index=th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.long),
        atom_batch_indices=th.zeros(4, dtype=th.long),
        atom_ptr=th.tensor([0, 4], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
    )

    # MAP all-cut mask + forced empty mask + each of the three single cuts.
    assert candidates.mask_log_probabilities.shape[0] == 5
    assert sorted(candidates.component_cut_counts.unique().tolist()) == [0, 1, 3]


def test_candidate_batch_preserves_vectorized_boundary_order() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0, 2.0]),
        chemical_bond_index=th.tensor(
            [[0, 1, 2], [1, 2, 0]],
            dtype=th.long,
        ),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
    )

    assert candidates.mask_batch_indices.tolist() == [0, 0, 0, 0, 0]
    assert candidates.mask_cut_counts.tolist() == [3, 0, 1, 1, 1]
    assert candidates.component_mask_indices.tolist() == [0, 0, 0, 1, 2, 3, 4]
    assert candidates.component_cut_counts.tolist() == [3, 3, 3, 0, 1, 1, 1]
    assert candidates.atom_indices.tolist() == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    assert candidates.atom_component_indices.tolist() == [
        0,
        1,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
    ]
    assert candidates.boundary_bond_indices.tolist() == [0, 0, 1, 1, 2, 2]
    assert candidates.boundary_component_indices.tolist() == [0, 1, 1, 2, 2, 0]


def test_candidate_batch_applies_minimum_probability_only_to_top_k() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0]),
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=4,
        min_mask_probability=0.5,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=False,
    )

    # The MAP mask passes; the otherwise low-probability empty mask remains forced.
    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


def test_candidate_batch_rejects_threshold_above_map_probability() -> None:
    with pytest.raises(ValueError, match="excludes every learned mask"):
        build_mass_shatter_candidates(
            th.zeros(4),
            chemical_bond_index=th.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=th.long
            ),
            atom_batch_indices=th.zeros(5, dtype=th.long),
            atom_ptr=th.tensor([0, 5], dtype=th.long),
            max_masks=4,
            min_mask_probability=0.2,
            explore_masks=0,
            exploration_mode="bounded_additive",
            exploration_temperature=1.0,
            exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
            include_single_bond_cuts=False,
        )


def test_legacy_refill_respects_minimum_probability() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([100.0, 100.0]),
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=4,
        min_mask_probability=0.5,
        explore_masks=2,
        exploration_mode="bernoulli_replace",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=False,
        generator=th.Generator().manual_seed(17),
    )

    # Exploration repeatedly proposes the MAP mask, and refill must not cross
    # the threshold to restore the historical fixed budget.
    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.exploration_mask_count.tolist() == [0.0]
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


def test_legacy_mixed_top_k_retains_exploitation_and_is_seeded() -> None:
    kwargs = {
        "bond_logits": th.tensor([2.0, 0.7, -0.4, -1.3]),
        "chemical_bond_index": th.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=th.long
        ),
        "atom_batch_indices": th.zeros(5, dtype=th.long),
        "atom_ptr": th.tensor([0, 5], dtype=th.long),
        "max_masks": 4,
        "explore_masks": 2,
        "exploration_mode": "bernoulli_replace",
        "exploration_temperature": 1.0,
        "exploration_cut_count_weights": BOUNDED_CUT_COUNT_WEIGHTS,
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
    assert first.exploration_mask_count.tolist() == [2.0]


def test_bounded_exploration_is_additive_seeded_and_respects_cut_count() -> None:
    kwargs = {
        "bond_logits": th.linspace(-2.0, -3.0, 8),
        "chemical_bond_index": th.tensor(
            [range(8), range(1, 9)],
            dtype=th.long,
        ),
        "atom_batch_indices": th.zeros(9, dtype=th.long),
        "atom_ptr": th.tensor([0, 9], dtype=th.long),
        "max_masks": 4,
        "explore_masks": 3,
        "exploration_mode": "bounded_additive",
        "exploration_temperature": 1.0,
        "exploration_cut_count_weights": (0.0, 0.0, 1.0),
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
    deterministic = build_mass_shatter_candidates(
        **{**kwargs, "explore_masks": 0},
        generator=th.Generator().manual_seed(17),
    )

    assert first.mask_log_probabilities.shape == (7,)
    assert th.equal(first.mask_log_probabilities, second.mask_log_probabilities)
    assert th.equal(
        first.mask_log_probabilities[:4],
        deterministic.mask_log_probabilities,
    )
    assert first.exploration_mask_count.tolist() == [3.0]
    assert first.exploration_mean_cut_count.tolist() == [2.0]


def test_bounded_exploration_does_not_refill_unavailable_slots() -> None:
    candidates = build_mass_shatter_candidates(
        bond_logits=th.tensor([-2.0, -3.0]),
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=1,
        explore_masks=4,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=(0.0, 0.0, 1.0),
        include_single_bond_cuts=False,
        generator=th.Generator().manual_seed(17),
    )

    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.exploration_mask_count.tolist() == [1.0]
    assert candidates.exploration_mean_cut_count.tolist() == [2.0]


def test_bounded_exploration_skips_an_exhausted_proposal_space() -> None:
    generator = th.Generator().manual_seed(17)
    initial_generator_state = generator.get_state().clone()

    candidates = build_mass_shatter_candidates(
        bond_logits=th.tensor([-2.0, -3.0, -4.0]),
        chemical_bond_index=th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.long),
        atom_batch_indices=th.zeros(4, dtype=th.long),
        atom_ptr=th.tensor([0, 4], dtype=th.long),
        max_masks=256,
        explore_masks=64,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
        generator=generator,
    )

    assert candidates.exploration_mask_count.tolist() == [0.0]
    assert th.equal(generator.get_state(), initial_generator_state)


def test_exploration_validates_parameters() -> None:
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
            exploration_mode="bernoulli_replace",
            exploration_temperature=1.0,
            exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        )
    with pytest.raises(ValueError, match="exploration_temperature"):
        build_mass_shatter_candidates(
            **common_kwargs,
            explore_masks=1,
            exploration_mode="bounded_additive",
            exploration_temperature=0.0,
            exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        )
    with pytest.raises(ValueError, match="cut_count_weights"):
        build_mass_shatter_candidates(
            **common_kwargs,
            explore_masks=1,
            exploration_mode="bounded_additive",
            exploration_temperature=1.0,
            exploration_cut_count_weights=(0.0, 1.0),
        )


def test_candidate_batch_accepts_static_cpu_graph_sidecar() -> None:
    chemical_bond_index = th.tensor([[0, 1], [1, 2]], dtype=th.long)
    atom_batch_indices = th.zeros(3, dtype=th.long)
    atom_ptr = th.tensor([0, 3], dtype=th.long)
    candidates = build_mass_shatter_candidates(
        th.tensor([1.0, -1.0]),
        chemical_bond_index=chemical_bond_index,
        atom_batch_indices=atom_batch_indices,
        atom_ptr=atom_ptr,
        max_masks=2,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
        static_graph=MassShatterStaticGraph(
            chemical_bond_index=chemical_bond_index,
            atom_batch_indices=atom_batch_indices,
            atom_ptr=atom_ptr,
        ),
    )

    assert candidates.component_mask_indices.tolist() == [0, 0, 1, 2, 2]
    assert candidates.atom_indices.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
