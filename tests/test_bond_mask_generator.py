"""Tests for direct fragment generation from bond-break probabilities."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fragnnet.frag.bond_mask_generator import (
    aggregate_fragment_marginals,
    connected_component_atom_masks,
    generate_fragment_marginals,
    top_k_cut_masks,
    translate_cut_masks,
)


def _brute_force_masks(logits: list[float]) -> list[tuple[tuple[int, ...], float]]:
    masks: list[tuple[tuple[int, ...], float]] = []
    for bits in itertools.product((0, 1), repeat=len(logits)):
        log_probability = sum(
            -float(np.logaddexp(0.0, -logit))
            if is_broken
            else -float(np.logaddexp(0.0, logit))
            for logit, is_broken in zip(logits, bits)
        )
        broken = tuple(index for index, is_broken in enumerate(bits) if is_broken)
        masks.append((broken, log_probability))
    masks.sort(key=lambda item: -item[1])
    return masks


@pytest.mark.parametrize("num_bonds", range(0, 9))
def test_top_k_masks_match_brute_force(num_bonds: int) -> None:
    rng = np.random.default_rng(9183 + num_bonds)
    logits = (rng.normal(size=num_bonds) + np.arange(num_bonds) * 1e-4).tolist()
    expected = _brute_force_masks(logits)[: min(40, 2**num_bonds)]
    actual = top_k_cut_masks(logits, k=40)

    assert [mask.broken_bond_indices for mask in actual] == [mask for mask, _ in expected]
    assert [mask.log_probability for mask in actual] == pytest.approx(
        [log_probability for _, log_probability in expected]
    )


def test_top_k_masks_include_all_cut_counts_without_a_depth_limit() -> None:
    masks = top_k_cut_masks([0.0, 0.0, 0.0, 0.0], k=16)

    assert len({mask.broken_bond_indices for mask in masks}) == 16
    assert {mask.cut_count for mask in masks} == {0, 1, 2, 3, 4}
    assert all(
        left.log_probability >= right.log_probability
        for left, right in itertools.pairwise(masks)
    )


def test_top_k_masks_can_exclude_empty_mask() -> None:
    masks = top_k_cut_masks([-3.0, -2.0], k=3, include_empty=False)

    assert len(masks) == 3
    assert all(mask.broken_bond_indices for mask in masks)


def test_top_k_masks_are_stable_for_extreme_finite_logits() -> None:
    masks = top_k_cut_masks([-1000.0, 1000.0], k=4)

    assert len(masks) == 4
    assert all(math.isfinite(mask.log_probability) for mask in masks)
    assert masks[0].broken_bond_indices == (1,)


def test_top_k_masks_reject_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        top_k_cut_masks([0.0, math.nan], k=2)


def test_connected_components_for_chain_cuts() -> None:
    endpoints = [(0, 1), (1, 2)]

    assert connected_component_atom_masks(3, endpoints, []) == (0b111,)
    assert connected_component_atom_masks(3, endpoints, [0]) == (0b001, 0b110)
    assert connected_component_atom_masks(3, endpoints, [0, 1]) == (
        0b001,
        0b010,
        0b100,
    )


def test_single_ring_cut_does_not_create_a_fragment() -> None:
    triangle = [[0, 1, 2], [1, 2, 0]]
    masks = top_k_cut_masks([10.0, -10.0, -10.0], k=1)
    translated = translate_cut_masks(masks, num_atoms=3, bond_endpoints=triangle)
    fragments = aggregate_fragment_marginals(translated)

    assert translated[0].component_atom_masks == (0b111,)
    assert fragments == ()


def test_fragment_generation_continues_past_unproductive_ring_masks() -> None:
    triangle = [[0, 1, 2], [1, 2, 0]]
    result = generate_fragment_marginals(
        [-5.0, -5.0, -5.0],
        num_atoms=3,
        bond_endpoints=triangle,
        max_masks=1,
        max_enumerated_masks=5,
    )

    assert len(result.cut_masks) == 5
    assert result.fragments


def test_two_ring_cuts_create_components() -> None:
    triangle = [[0, 1, 2], [1, 2, 0]]

    assert connected_component_atom_masks(3, triangle, [0, 1]) == (0b010, 0b101)


def test_fragment_probability_sums_duplicate_cut_mask_paths() -> None:
    # A-B-C: fragment {A} exists whenever bond A-B breaks, regardless of B-C.
    probability_ab = 0.2
    probability_bc = 0.3
    logits = [
        math.log(probability_ab / (1.0 - probability_ab)),
        math.log(probability_bc / (1.0 - probability_bc)),
    ]
    masks = top_k_cut_masks(logits, k=4)
    translated = translate_cut_masks(masks, num_atoms=3, bond_endpoints=[(0, 1), (1, 2)])
    fragments = aggregate_fragment_marginals(translated)
    fragment_a = next(fragment for fragment in fragments if fragment.atom_mask == 0b001)

    assert math.exp(fragment_a.log_probability) == pytest.approx(probability_ab)
    assert fragment_a.contributing_mask_count == 2
    assert fragment_a.atom_indices == (0,)


def test_generate_fragment_marginals_validates_bond_count() -> None:
    with pytest.raises(ValueError, match="same number of bonds"):
        generate_fragment_marginals(
            [0.0],
            num_atoms=3,
            bond_endpoints=[(0, 1), (1, 2)],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_masks": 0}, "max_masks"),
        ({"max_unique_fragments": 0}, "max_unique_fragments"),
        ({"max_masks": 2, "max_enumerated_masks": 1}, "at least max_masks"),
        ({"num_atoms": 0}, "num_atoms"),
    ],
)
def test_generate_fragment_marginals_validates_limits(
    kwargs: dict[str, int],
    message: str,
) -> None:
    arguments = {
        "bond_logits": [0.0],
        "num_atoms": 2,
        "bond_endpoints": [[0], [1]],
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        generate_fragment_marginals(**arguments)


def test_generator_accepts_model_head_endpoint_layout() -> None:
    endpoints = [[0, 1, 2], [1, 2, 3]]

    result = generate_fragment_marginals(
        [5.0, -5.0, -5.0],
        num_atoms=4,
        bond_endpoints=endpoints,
        max_masks=1,
    )

    assert result.cut_masks[0].component_atom_masks == (0b0001, 0b1110)


@pytest.mark.parametrize(
    ("num_atoms", "endpoints", "broken", "message"),
    [
        (0, [[], []], [], "positive"),
        (2, [[0], [1], [2]], [], "shape"),
        (2, [[0], [2]], [], "outside"),
        (2, [[0], [1]], [1], "invalid bond"),
    ],
)
def test_connected_components_validates_inputs(
    num_atoms: int,
    endpoints: list[list[int]],
    broken: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        connected_component_atom_masks(num_atoms, endpoints, broken)
