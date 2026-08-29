"""Tests for sparse MASShatter cut-mask/component batching."""

import numpy as np
import pytest
import torch as th

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterStaticGraph,
    build_mass_shatter_candidates,
)
from fragnnet.frag.mass_shatter_candidates import (
    _top_mixture_bond_sets as _top_mixture_bond_sets_fn,
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
        include_single_bond_cuts=True,
    )

    assert candidates.mask_log_probabilities.tolist() == [0.0]
    assert candidates.component_mask_indices.tolist() == [0]
    assert candidates.atom_indices.tolist() == [0]
    assert candidates.boundary_bond_indices.numel() == 0
    assert candidates.timings.total_ms >= 0.0


def test_bernoulli_mixture_is_normalized_and_backpropagates() -> None:
    bond_logits = th.tensor(
        [[2.0, -2.0], [-2.0, 2.0]],
        requires_grad=True,
    )
    mixture_logits = th.tensor([[0.4, -0.3]], requires_grad=True)
    mixture_log_weights = th.log_softmax(mixture_logits, dim=1)
    candidates = build_mass_shatter_candidates(
        bond_logits,
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=4,
        include_single_bond_cuts=False,
        mixture_log_weights=mixture_log_weights,
    )

    assert candidates.mask_log_probabilities.shape == (4,)
    all_masks = th.tensor(
        [[False, False], [True, False], [False, True], [True, True]],
    )
    component_log_probabilities = th.where(
        all_masks[:, :, None],
        th.nn.functional.logsigmoid(bond_logits)[None],
        th.nn.functional.logsigmoid(-bond_logits)[None],
    ).sum(dim=1)
    expected_log_probabilities = th.logsumexp(
        component_log_probabilities + mixture_log_weights,
        dim=1,
    )
    assert th.allclose(
        candidates.mask_log_probabilities.sort().values,
        expected_log_probabilities.sort().values,
        atol=1e-6,
    )
    assert candidates.mask_log_probabilities.exp().sum().item() == pytest.approx(
        1.0,
        abs=1e-6,
    )
    weights = th.arange(1, 5, dtype=th.float)
    (weights * candidates.mask_log_probabilities).sum().backward()
    assert bond_logits.grad is not None
    assert mixture_logits.grad is not None
    assert th.isfinite(bond_logits.grad).all()
    assert th.isfinite(mixture_logits.grad).all()
    assert bond_logits.grad.abs().sum() > 0
    assert mixture_logits.grad.abs().sum() > 0


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
        include_single_bond_cuts=False,
    )

    assert candidates.atom_indices.tolist() == list(range(num_atoms))
    assert candidates.atom_component_indices.tolist() == [0] * num_atoms


def test_candidate_batch_forces_empty_and_single_cut_masks() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0, 2.0]),
        chemical_bond_index=th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.long),
        atom_batch_indices=th.zeros(4, dtype=th.long),
        atom_ptr=th.tensor([0, 4], dtype=th.long),
        max_masks=1,
        include_single_bond_cuts=True,
    )

    # MAP all-cut mask + forced empty mask + each of the three single cuts.
    assert candidates.mask_log_probabilities.shape[0] == 5
    assert sorted(candidates.component_cut_counts.unique().tolist()) == [0, 1, 3]


def test_unique_component_budget_continues_past_unproductive_ring_masks() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([-2.0, -2.0, -2.0]),
        chemical_bond_index=th.tensor(
            [[0, 1, 2], [1, 2, 0]],
            dtype=th.long,
        ),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=2,
        max_unique_components=3,
        max_enumerated_masks=8,
        include_single_bond_cuts=True,
    )

    # Empty + all single-ring cuts yield only the intact atom set, so selection
    # must continue to a two-cut mask to obtain the requested unique fragments.
    assert candidates.mask_cut_counts.tolist() == [0, 1, 1, 1, 2]
    assert candidates.unique_component_count.tolist() == [3.0]
    assert candidates.num_unique_components == 3
    assert candidates.component_unique_indices.max().item() == 2
    assert candidates.unique_representative_component_indices.numel() == 3
    assert candidates.mask_log_probabilities.exp().sum().item() <= 1.0 + 1e-7


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
        include_single_bond_cuts=False,
    )

    # The MAP mask passes; the otherwise low-probability empty mask remains forced.
    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


def test_candidate_batch_applies_map_relative_probability_only_to_top_k() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([2.0, 2.0]),
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=4,
        min_mask_relative_probability=0.5,
        include_single_bond_cuts=False,
    )

    # Only the MAP mask passes; the below-threshold empty mask remains forced.
    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


def test_map_relative_probability_can_stop_below_unique_fragment_target() -> None:
    candidates = build_mass_shatter_candidates(
        th.full((4,), 2.0),
        chemical_bond_index=th.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            dtype=th.long,
        ),
        atom_batch_indices=th.zeros(5, dtype=th.long),
        atom_ptr=th.tensor([0, 5], dtype=th.long),
        max_masks=8,
        min_mask_relative_probability=0.5,
        max_unique_components=8,
        max_enumerated_masks=8,
        include_single_bond_cuts=False,
    )

    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.unique_component_count.tolist() == [6.0]


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
        include_single_bond_cuts=False,
    )

    # Refill must not cross the threshold to restore the historical fixed budget.
    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


def test_legacy_refill_respects_map_relative_probability() -> None:
    candidates = build_mass_shatter_candidates(
        th.tensor([100.0, 100.0]),
        chemical_bond_index=th.tensor([[0, 1], [1, 2]], dtype=th.long),
        atom_batch_indices=th.zeros(3, dtype=th.long),
        atom_ptr=th.tensor([0, 3], dtype=th.long),
        max_masks=4,
        min_mask_relative_probability=0.5,
        include_single_bond_cuts=False,
    )

    assert candidates.mask_log_probabilities.shape == (2,)
    assert candidates.component_cut_counts.tolist() == [2, 2, 2, 0]


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
        include_single_bond_cuts=True,
        static_graph=MassShatterStaticGraph(
            chemical_bond_index=chemical_bond_index,
            atom_batch_indices=atom_batch_indices,
            atom_ptr=atom_ptr,
        ),
    )

    assert candidates.component_mask_indices.tolist() == [0, 0, 1, 2, 2]
    assert candidates.atom_indices.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_shared_candidate_groups_match_repeated_construction_and_gradients() -> None:
    chemical_bond_index = th.tensor(
        [[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]],
        dtype=th.long,
    )
    atom_batch_indices = th.tensor([0] * 4 + [1] * 4, dtype=th.long)
    atom_ptr = th.tensor([0, 4, 8], dtype=th.long)
    common_kwargs = {
        "chemical_bond_index": chemical_bond_index,
        "atom_batch_indices": atom_batch_indices,
        "atom_ptr": atom_ptr,
        "max_masks": 4,
        "max_unique_components": 6,
        "max_enumerated_masks": 8,
        "include_single_bond_cuts": True,
    }
    repeated_logits = th.tensor(
        [0.8, -0.4, 0.2, 0.8, -0.4, 0.2],
        requires_grad=True,
    )
    shared_logits = repeated_logits.detach().clone().requires_grad_(True)

    repeated = build_mass_shatter_candidates(repeated_logits, **common_kwargs)
    shared = build_mass_shatter_candidates(
        shared_logits,
        **common_kwargs,
        candidate_group_ids=th.tensor([19, 19], dtype=th.long),
    )

    for name, expected in vars(repeated).items():
        if name == "timings":
            continue
        actual = getattr(shared, name)
        if isinstance(expected, th.Tensor):
            assert th.equal(actual, expected), name
        else:
            assert actual == expected, name

    weights = th.arange(1, repeated.mask_log_probabilities.numel() + 1)
    (repeated.mask_log_probabilities * weights).sum().backward()
    (shared.mask_log_probabilities * weights).sum().backward()
    assert th.equal(shared_logits.grad, repeated_logits.grad)


def test_shared_candidate_groups_reject_different_logits() -> None:
    with pytest.raises(ValueError, match="identical proposal logits"):
        build_mass_shatter_candidates(
            th.tensor([0.8, -0.4, 0.8, -0.3]),
            chemical_bond_index=th.tensor(
                [[0, 1, 3, 4], [1, 2, 4, 5]],
                dtype=th.long,
            ),
            atom_batch_indices=th.tensor([0, 0, 0, 1, 1, 1], dtype=th.long),
            atom_ptr=th.tensor([0, 3, 6], dtype=th.long),
            max_masks=4,
            include_single_bond_cuts=True,
            candidate_group_ids=th.tensor([5, 5], dtype=th.long),
        )


def _mixture_bond_sets(ptr: np.ndarray, bonds: np.ndarray) -> list[tuple[int, ...]]:
    """Expand a CSR mixture pool into bond-set tuples."""
    return [tuple(int(b) for b in bonds[ptr[i] : ptr[i + 1]]) for i in range(ptr.size - 1)]


def test_top_mixture_bond_sets_ranks_by_exact_mixture_probability() -> None:
    # Two bonds, two components. Every one of the four bond sets is enumerable, so
    # the exact mixture probability can be written out and compared term by term.
    logits = np.array([[1.0, -2.0], [-0.5, 1.5]])
    mixture_log_weights = np.log(np.array([0.25, 0.75]))
    ptr, bonds = _top_mixture_bond_sets_fn(logits, mixture_log_weights, 4, 0.0, 0.0)
    bond_sets = _mixture_bond_sets(ptr, bonds)

    assert sorted(bond_sets) == [(), (0,), (0, 1), (1,)]

    log_sigmoid = lambda x: -np.logaddexp(0.0, -x)  # noqa: E731
    expected = {}
    for bond_set in [(), (0,), (0, 1), (1,)]:
        per_component = np.zeros(2)
        for bond in range(2):
            chosen = bond in bond_set
            per_component += log_sigmoid(logits[bond] if chosen else -logits[bond])
        expected[bond_set] = np.logaddexp.reduce(mixture_log_weights + per_component)

    ranked = sorted(expected, key=lambda s: (-expected[s], s))
    assert bond_sets == ranked
    # Probabilities are a normalized distribution over the full enumeration.
    assert sum(np.exp(v) for v in expected.values()) == pytest.approx(1.0)


def test_top_mixture_bond_sets_deduplicates_across_components() -> None:
    # Identical components must produce identical top-k pools, so the union
    # collapses to a single copy of every bond set.
    logits = np.tile(np.array([[0.9], [-0.4], [0.2]]), (1, 2))
    mixture_log_weights = np.log(np.array([0.5, 0.5]))
    ptr, bonds = _top_mixture_bond_sets_fn(logits, mixture_log_weights, 8, 0.0, 0.0)
    bond_sets = _mixture_bond_sets(ptr, bonds)

    assert len(bond_sets) == len(set(bond_sets)) == 8
    # Bonds 0 and 2 carry positive logits, so the MAP mask cuts exactly those two.
    assert bond_sets[0] == (0, 2)


def test_top_mixture_bond_sets_limit_one_keeps_the_best_map_mask() -> None:
    # A limit of one pools each component's MAP mask and keeps the more probable of
    # the two. Component 1 cuts both bonds because both of its logits are positive.
    logits = np.array([[0.3, 0.1], [-0.2, 0.4]])
    ptr, bonds = _top_mixture_bond_sets_fn(logits, np.log(np.array([0.5, 0.5])), 1, 0.0, 0.0)

    assert _mixture_bond_sets(ptr, bonds) == [(0, 1)]


def test_top_mixture_bond_sets_all_intact_pool_has_zero_width() -> None:
    # Edge case: every component's MAP mask is empty, so the padded matrix carries no
    # columns at all and the CSR pointer must still describe one zero-length bond set.
    logits = np.array([[-1.0, -2.0], [-0.5, -1.5]])
    ptr, bonds = _top_mixture_bond_sets_fn(logits, np.log(np.array([0.5, 0.5])), 1, 0.0, 0.0)

    assert ptr.tolist() == [0, 0]
    assert bonds.size == 0
    assert _mixture_bond_sets(ptr, bonds) == [()]


def test_top_mixture_bond_sets_applies_probability_floors() -> None:
    logits = np.array([[2.0, 2.0], [-3.0, -3.0]])
    mixture_log_weights = np.log(np.array([0.5, 0.5]))
    unfiltered, _ = _top_mixture_bond_sets_fn(logits, mixture_log_weights, 4, 0.0, 0.0)
    assert unfiltered.size - 1 == 4

    # A relative floor of 0.5 keeps only masks within 2x of the MAP mask.
    ptr, bonds = _top_mixture_bond_sets_fn(logits, mixture_log_weights, 4, 0.0, 0.5)
    assert _mixture_bond_sets(ptr, bonds) == [(0,)]


def test_top_mixture_bond_sets_returns_int64_csr() -> None:
    logits = np.array([[0.5, -0.5], [0.25, 0.75], [-1.0, 0.1]])
    ptr, bonds = _top_mixture_bond_sets_fn(logits, np.log(np.array([0.4, 0.6])), 6, 0.0, 0.0)

    assert ptr.dtype == np.int64 and bonds.dtype == np.int64
    assert ptr[0] == 0 and ptr[-1] == bonds.size
    assert np.all(np.diff(ptr) >= 0)


@pytest.mark.parametrize("mask_selection_threads", [2, 6])
def test_threaded_mask_selection_matches_serial(mask_selection_threads: int) -> None:
    """Threaded mixture selection must reproduce the serial candidates exactly."""
    generator = th.Generator().manual_seed(11)
    num_molecules, num_bonds_each, num_mixtures = 12, 6, 2
    bond_logits = th.randn(num_bonds_each * num_molecules, num_mixtures, generator=generator)
    mixture_log_weights = th.log_softmax(
        th.randn(num_molecules, num_mixtures, generator=generator).double(),
        dim=1,
    )
    left = [m * (num_bonds_each + 1) + b for m in range(num_molecules) for b in range(num_bonds_each)]
    kwargs = {
        "chemical_bond_index": th.tensor([left, [index + 1 for index in left]], dtype=th.long),
        "atom_batch_indices": th.arange(num_molecules).repeat_interleave(num_bonds_each + 1),
        "atom_ptr": th.arange(0, num_molecules * (num_bonds_each + 1) + 1, num_bonds_each + 1),
        "max_masks": 8,
        "include_single_bond_cuts": True,
        "mixture_log_weights": mixture_log_weights,
    }
    serial = build_mass_shatter_candidates(bond_logits, mask_selection_threads=1, **kwargs)
    threaded = build_mass_shatter_candidates(
        bond_logits, mask_selection_threads=mask_selection_threads, **kwargs
    )

    assert th.equal(threaded.mask_batch_indices, serial.mask_batch_indices)
    assert th.equal(threaded.mask_cut_counts, serial.mask_cut_counts)
    assert th.equal(threaded.component_batch_indices, serial.component_batch_indices)
    assert th.equal(threaded.atom_indices, serial.atom_indices)
    assert th.equal(threaded.atom_component_indices, serial.atom_component_indices)
    assert th.allclose(
        threaded.mask_log_probabilities, serial.mask_log_probabilities, atol=0.0, rtol=0.0
    )


def test_threaded_mask_selection_shares_collision_energy_groups() -> None:
    """Grouped collision-energy views stay shared when selection runs on threads."""
    # Two molecules with identical logits in one group: the second must reuse the
    # first's ranked masks rather than being recomputed by a worker thread.
    logits = th.tensor([[0.8, -0.2], [-0.4, 0.6], [0.8, -0.2], [-0.4, 0.6]])
    mixture_log_weights = th.log_softmax(th.tensor([[0.3, -0.3], [0.3, -0.3]]).double(), dim=1)
    kwargs = {
        "chemical_bond_index": th.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=th.long),
        "atom_batch_indices": th.tensor([0, 0, 0, 1, 1, 1], dtype=th.long),
        "atom_ptr": th.tensor([0, 3, 6], dtype=th.long),
        "max_masks": 4,
        "include_single_bond_cuts": True,
        "mixture_log_weights": mixture_log_weights,
        "candidate_group_ids": th.tensor([3, 3], dtype=th.long),
    }
    serial = build_mass_shatter_candidates(logits, mask_selection_threads=1, **kwargs)
    threaded = build_mass_shatter_candidates(logits, mask_selection_threads=4, **kwargs)

    assert th.equal(threaded.mask_batch_indices, serial.mask_batch_indices)
    assert th.equal(threaded.mask_cut_counts, serial.mask_cut_counts)
    # Both views select the same number of masks, which is what sharing guarantees.
    assert int((threaded.mask_batch_indices == 0).sum()) == int(
        (threaded.mask_batch_indices == 1).sum()
    )


def test_mask_selection_threads_must_be_positive() -> None:
    with pytest.raises(ValueError, match="mask_selection_threads must be at least 1"):
        build_mass_shatter_candidates(
            th.tensor([0.4, -0.7]),
            chemical_bond_index=th.tensor([[0, 2], [1, 3]], dtype=th.long),
            atom_batch_indices=th.tensor([0, 0, 1, 1], dtype=th.long),
            atom_ptr=th.tensor([0, 2, 4], dtype=th.long),
            max_masks=2,
            include_single_bond_cuts=True,
            mask_selection_threads=0,
        )
