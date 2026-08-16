"""Tests for one-pass bond, charge, and hydrogen-transfer heads."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    EndpointBondBreakHead,
    JointIonHydrogenHead,
    component_charge_log_probabilities,
    cut_mask_log_probabilities,
    masked_ion_hydrogen_log_probabilities,
    pool_boundary_bond_embeddings,
    pool_component_and_complement_embeddings,
    pool_component_embeddings,
)


def test_bond_break_head_is_invariant_to_endpoint_order() -> None:
    th.manual_seed(17)
    head = EndpointBondBreakHead(
        atom_embedding_dim=5,
        bond_feature_dim=3,
        context_dim=0,
        hidden_dim=8,
        dropout=0.0,
    )
    atom_embeddings = th.randn(4, 5)
    bond_endpoints = th.tensor([[0, 1, 2], [1, 2, 3]], dtype=th.long)
    bond_features = th.randn(3, 3)

    forward_logits = head(atom_embeddings, bond_endpoints, bond_features)
    reverse_logits = head(atom_embeddings, bond_endpoints.flip(0), bond_features)

    assert forward_logits.shape == (3,)
    assert th.allclose(forward_logits, reverse_logits)


def test_bond_break_head_uses_batched_context_and_backpropagates() -> None:
    head = EndpointBondBreakHead(4, 2, 3, hidden_dim=7, dropout=0.0)
    atom_embeddings = th.randn(5, 4, requires_grad=True)
    bond_features = th.randn(3, 2, requires_grad=True)
    context = th.randn(2, 3, requires_grad=True)
    logits = head(
        atom_embeddings,
        th.tensor([[0, 1, 3], [1, 2, 4]], dtype=th.long),
        bond_features,
        context=context,
        bond_batch_indices=th.tensor([0, 0, 1], dtype=th.long),
    )

    logits.sum().backward()

    assert atom_embeddings.grad is not None
    assert bond_features.grad is not None
    assert context.grad is not None


def test_bond_break_head_supports_fullgraph_compile() -> None:
    head = EndpointBondBreakHead(4, 2, 3, hidden_dim=7, dropout=0.0)
    compiled_head = th.compile(head, backend="eager", fullgraph=True)

    logits = compiled_head(
        th.randn(5, 4),
        th.tensor([[0, 1, 3], [1, 2, 4]], dtype=th.long),
        th.randn(3, 2),
        context=th.randn(2, 3),
        bond_batch_indices=th.tensor([0, 0, 1], dtype=th.long),
    )

    assert logits.shape == (3,)


def test_bond_break_head_requires_configured_context() -> None:
    head = EndpointBondBreakHead(4, 0, 2, hidden_dim=7, dropout=0.0)

    with pytest.raises(ValueError, match="required"):
        head(
            th.randn(2, 4),
            th.tensor([[0], [1]], dtype=th.long),
        )


def test_atom_charge_affinity_head_shape() -> None:
    head = AtomChargeAffinityHead(5, 2, hidden_dim=8, dropout=0.0)
    logits = head(
        th.randn(4, 5),
        context=th.randn(2, 2),
        atom_batch_indices=th.tensor([0, 0, 1, 1], dtype=th.long),
    )

    assert logits.shape == (4,)


def test_component_charge_probabilities_normalize_across_components() -> None:
    atom_logits = th.tensor([0.0, 1.0, -1.0])

    log_probabilities = component_charge_log_probabilities(
        atom_logits,
        atom_indices=th.tensor([0, 1, 2]),
        atom_component_indices=th.tensor([0, 1, 1]),
        component_group_indices=th.tensor([0, 0]),
    )

    assert th.exp(log_probabilities).sum() == pytest.approx(1.0)


def test_component_charge_probabilities_normalize_per_cut_mask() -> None:
    atom_logits = th.tensor(
        [0.0, 0.0, math.log(0.25), math.log(0.75)],
        requires_grad=True,
    )

    log_probabilities = component_charge_log_probabilities(
        atom_logits,
        atom_indices=th.arange(4),
        atom_component_indices=th.arange(4),
        component_group_indices=th.tensor([0, 0, 1, 1]),
    )

    assert th.exp(log_probabilities).tolist() == pytest.approx(
        [0.5, 0.5, 0.25, 0.75]
    )
    log_probabilities.sum().backward()
    assert atom_logits.grad is not None
    assert th.isfinite(atom_logits.grad).all()


def test_logmeanexp_charge_pooling_removes_component_size_prior() -> None:
    atom_logits = th.zeros(3)

    logsumexp_probs = th.exp(
        component_charge_log_probabilities(
            atom_logits,
            atom_indices=th.tensor([0, 1, 2]),
            atom_component_indices=th.tensor([0, 1, 1]),
            component_group_indices=th.tensor([0, 0]),
            pooling="logsumexp",
        )
    )
    logmeanexp_probs = th.exp(
        component_charge_log_probabilities(
            atom_logits,
            atom_indices=th.tensor([0, 1, 2]),
            atom_component_indices=th.tensor([0, 1, 1]),
            component_group_indices=th.tensor([0, 0]),
            pooling="logmeanexp",
        )
    )

    assert logsumexp_probs.tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert logmeanexp_probs.tolist() == pytest.approx([0.5, 0.5])


def test_component_charge_probabilities_reject_empty_component() -> None:
    with pytest.raises(ValueError, match="at least one atom"):
        component_charge_log_probabilities(
            th.zeros(2),
            atom_indices=th.tensor([0]),
            atom_component_indices=th.tensor([0]),
            component_group_indices=th.tensor([0, 0]),
        )


@pytest.mark.parametrize("pooling", ["logsumexp", "logmeanexp"])
def test_grouped_charge_probabilities_match_dense_reference(
    pooling: Literal["logsumexp", "logmeanexp"],
) -> None:
    th.manual_seed(912)
    atom_logits = th.randn(8, requires_grad=True)
    component_group_indices = th.tensor([0, 0, 1, 1, 1, 2])
    memberships = [
        [0, 1],
        [2, 3, 4],
        [0, 5],
        [1, 2, 6],
        [3, 4, 7],
        [5, 6, 7],
    ]
    atom_indices = th.tensor([atom for atoms in memberships for atom in atoms])
    atom_component_indices = th.tensor(
        [component for component, atoms in enumerate(memberships) for _ in atoms]
    )

    actual = component_charge_log_probabilities(
        atom_logits,
        atom_indices,
        atom_component_indices,
        component_group_indices,
        pooling=pooling,
    )
    component_logits = th.stack(
        [th.logsumexp(atom_logits[atoms], dim=0) for atoms in memberships]
    )
    if pooling == "logmeanexp":
        component_logits = component_logits - th.log(
            th.tensor([len(atoms) for atoms in memberships], dtype=atom_logits.dtype)
        )
    expected = th.empty_like(component_logits)
    for group_index in component_group_indices.unique():
        group_mask = component_group_indices == group_index
        expected[group_mask] = th.log_softmax(component_logits[group_mask], dim=0)

    assert th.allclose(actual, expected)
    actual.sum().backward()
    assert atom_logits.grad is not None
    assert th.isfinite(atom_logits.grad).all()


def test_component_charge_probabilities_reject_invalid_group_index() -> None:
    with pytest.raises(ValueError, match="invalid group"):
        component_charge_log_probabilities(
            th.zeros(2),
            atom_indices=th.tensor([0, 1]),
            atom_component_indices=th.tensor([0, 1]),
            component_group_indices=th.tensor([0, 2]),
        )


def test_cut_mask_log_probabilities_match_manual_values_and_backpropagate() -> None:
    logits = th.tensor([-1.2, 0.7, 2.1], requires_grad=True)
    masks = th.tensor(
        [
            [False, False, False],
            [True, False, True],
            [True, True, True],
        ]
    )
    actual = cut_mask_log_probabilities(logits, masks)
    expected = []
    for mask in masks:
        expected.append(
            sum(
                F.logsigmoid(logit) if is_cut else F.logsigmoid(-logit)
                for logit, is_cut in zip(logits, mask)
            )
        )

    assert th.allclose(actual, th.stack(expected))
    actual.sum().backward()
    assert logits.grad is not None


def test_pool_component_embeddings_mean() -> None:
    embeddings = th.tensor(
        [[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]],
        requires_grad=True,
    )

    pooled = pool_component_embeddings(
        embeddings,
        item_indices=th.tensor([0, 1, 2]),
        component_indices=th.tensor([0, 0, 1]),
        num_components=2,
    )

    assert th.allclose(pooled, th.tensor([[2.0, 3.0], [8.0, 10.0]]))
    pooled.sum().backward()
    assert embeddings.grad is not None


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_joint_component_and_complement_pooling_matches_dense_reference(
    reduction: Literal["sum", "mean"],
) -> None:
    embeddings = th.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        requires_grad=True,
    )
    atom_batch_indices = th.tensor([0, 0, 1, 1])
    component_batch_indices = th.tensor([0, 0, 1, 1])
    memberships = [[0], [1], [2, 3], [3]]
    atom_indices = th.tensor([atom for atoms in memberships for atom in atoms])
    atom_component_indices = th.tensor(
        [component for component, atoms in enumerate(memberships) for _ in atoms]
    )

    components, complements = pool_component_and_complement_embeddings(
        embeddings,
        atom_indices,
        atom_component_indices,
        component_batch_indices,
        atom_batch_indices,
        num_components=4,
        num_molecules=2,
        reduction=reduction,
    )

    expected_components = []
    expected_complements = []
    for component_index, atoms in enumerate(memberships):
        molecule_atoms = th.nonzero(
            atom_batch_indices == component_batch_indices[component_index]
        ).flatten()
        complement_atoms = molecule_atoms[~th.isin(molecule_atoms, th.tensor(atoms))]
        component_values = embeddings[atoms]
        complement_values = embeddings[complement_atoms]
        if reduction == "sum":
            expected_components.append(component_values.sum(dim=0))
            expected_complements.append(complement_values.sum(dim=0))
        else:
            expected_components.append(component_values.mean(dim=0))
            expected_complements.append(
                complement_values.mean(dim=0)
                if complement_values.shape[0] > 0
                else th.zeros_like(embeddings[0])
            )

    assert th.allclose(components, th.stack(expected_components))
    assert th.allclose(complements, th.stack(expected_complements))
    (components.sum() + complements.sum()).backward()
    assert embeddings.grad is not None


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_sparse_component_pooling_matches_dense_reference(
    reduction: Literal["sum", "mean"],
) -> None:
    th.manual_seed(431)
    embeddings = th.randn(17, 6)
    membership = th.rand(7, 17) < 0.18
    membership[th.arange(7), th.arange(7)] = True
    component_indices, item_indices = membership.nonzero(as_tuple=True)

    actual = pool_component_embeddings(
        embeddings,
        item_indices,
        component_indices,
        num_components=membership.shape[0],
        reduction=reduction,
    )
    expected = membership.to(embeddings.dtype) @ embeddings
    if reduction == "mean":
        expected = expected / membership.sum(dim=1, keepdim=True)

    assert th.allclose(actual, expected)


def test_sparse_component_pooling_rejects_empty_component() -> None:
    with pytest.raises(ValueError, match="at least one pooled item"):
        pool_component_embeddings(
            th.randn(2, 3),
            item_indices=th.tensor([0]),
            component_indices=th.tensor([0]),
            num_components=2,
            reduction="sum",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"item_embeddings": th.zeros(2)}, "item_embeddings"),
        ({"num_components": -1}, "nonnegative"),
        ({"component_indices": th.tensor([0, 1])}, "matching vectors"),
        ({"item_indices": th.tensor([0.0])}, "torch.long"),
        ({"reduction": "maximum"}, "unknown"),
        ({"item_indices": th.tensor([2])}, "missing item"),
        ({"component_indices": th.tensor([2])}, "missing component"),
    ],
)
def test_sparse_component_pooling_validates_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "item_embeddings": th.zeros(2, 3),
        "item_indices": th.tensor([0]),
        "component_indices": th.tensor([0]),
        "num_components": 2,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        pool_component_embeddings(**arguments)  # type: ignore[arg-type]


def test_pool_boundary_bond_embeddings() -> None:
    # Chain A-B-C, with components {A} and {B,C}. Only A-B crosses either boundary.
    bond_embeddings = th.tensor([[2.0, 4.0], [10.0, 20.0]])

    pooled = pool_boundary_bond_embeddings(
        bond_embeddings,
        boundary_bond_indices=th.tensor([0, 0]),
        component_indices=th.tensor([0, 1]),
        num_components=3,
    )

    assert th.allclose(
        pooled,
        th.tensor([[2.0, 4.0], [2.0, 4.0], [0.0, 0.0]]),
    )


def test_boundary_pooling_allows_no_memberships() -> None:
    pooled = pool_boundary_bond_embeddings(
        th.empty(0, 4),
        boundary_bond_indices=th.empty(0, dtype=th.long),
        component_indices=th.empty(0, dtype=th.long),
        num_components=3,
    )

    assert pooled.shape == (3, 4)
    assert th.count_nonzero(pooled) == 0


def test_sparse_component_operations_support_fullgraph_compile() -> None:
    compiled_pool = th.compile(pool_component_embeddings, backend="eager", fullgraph=True)
    compiled_charge = th.compile(
        component_charge_log_probabilities,
        backend="eager",
        fullgraph=True,
    )

    pooled = compiled_pool(
        th.randn(3, 4),
        th.tensor([0, 1, 2]),
        th.tensor([0, 0, 1]),
        num_components=2,
    )
    charge_log_probabilities = compiled_charge(
        th.randn(3),
        th.tensor([0, 1, 2]),
        th.tensor([0, 1, 1]),
        th.tensor([0, 0]),
    )

    assert pooled.shape == (2, 4)
    assert charge_log_probabilities.shape == (2,)


def test_joint_ion_hydrogen_head_shape_and_gradient() -> None:
    head = JointIonHydrogenHead(
        fragment_embedding_dim=6,
        context_dim=4,
        hidden_dim=9,
        num_ion_modes=2,
        num_h_shifts=5,
        dropout=0.0,
    )
    fragment_embeddings = th.randn(3, 6, requires_grad=True)
    logits = head(
        fragment_embeddings,
        th.randn(3, 6),
        th.randn(3, 6),
        th.tensor([1, 2, 4]),
        context=th.randn(3, 4),
    )
    logits.sum().backward()

    assert logits.shape == (3, 2, 5)
    assert fragment_embeddings.grad is not None


def test_masked_ion_hydrogen_probabilities_zero_invalid_channels() -> None:
    logits = th.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    valid = th.tensor(
        [
            [[True, True, True], [False, True, False]],
            [[False, False, True], [True, False, False]],
        ]
    )

    log_probabilities = masked_ion_hydrogen_log_probabilities(logits, valid)

    assert th.isneginf(log_probabilities[~valid]).all()
    assert th.exp(log_probabilities).flatten(start_dim=1).sum(dim=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_masked_ion_hydrogen_probabilities_require_valid_channel() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        masked_ion_hydrogen_log_probabilities(
            th.zeros(1, 2, 3),
            th.zeros(1, 2, 3, dtype=th.bool),
        )
