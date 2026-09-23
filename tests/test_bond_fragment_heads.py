"""Tests for one-pass bond, charge, and hydrogen-transfer heads."""

from __future__ import annotations

import inspect
import math
from typing import Literal

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    BondFocusedBreakHead,
    CandidateIntensityTransformerHead,
    EndpointBondBreakHead,
    IonModeProposalHead,
    component_charge_log_probabilities,
    cut_mask_log_probabilities,
    latent_pathway_charge_log_probabilities,
    masked_ion_hydrogen_log_probabilities,
    pool_boundary_bond_embeddings,
    pool_component_and_complement_embeddings,
    pool_component_embeddings,
)


def test_ion_mode_proposal_head_scores_each_fragment() -> None:
    head = IonModeProposalHead(4, 3, 8, 2, 0.0)
    fragment_embeddings = th.randn(5, 4, requires_grad=True)
    logits = head(
        fragment_embeddings,
        th.randn(5, 4),
        th.randn(5, 4),
        th.tensor([0, 1, 2, 1, 3]),
        th.tensor([3, 2, 1, 4, 2]),
        th.randn(5, 3),
    )

    logits.sum().backward()

    assert logits.shape == (5, 2)
    assert fragment_embeddings.grad is not None


def test_separate_ion_h_support_head_copies_intensity_initialization() -> None:
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=6,
        context_dim=3,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        feedforward_dim=24,
        num_inducing_points=4,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
        separate_ion_h_support_head=True,
    )

    assert head.ion_h_support_head is not None
    assert th.equal(head.ion_h_support_head.weight, head.ion_hydrogen_head.weight)
    assert th.equal(head.ion_h_support_head.bias, head.ion_hydrogen_head.bias)


def test_separate_ion_h_support_head_copies_loaded_intensity_weights() -> None:
    kwargs = {
        "fragment_embedding_dim": 6,
        "context_dim": 3,
        "model_dim": 12,
        "num_layers": 1,
        "num_heads": 3,
        "feedforward_dim": 24,
        "num_inducing_points": 4,
        "num_ion_modes": 2,
        "num_h_shifts": 3,
        "dropout": 0.0,
        "use_base_probability_feature": True,
    }
    source = CandidateIntensityTransformerHead(**kwargs)
    with th.no_grad():
        source.ion_hydrogen_head.weight.fill_(0.25)
        source.ion_hydrogen_head.bias.fill_(-0.5)
    target = CandidateIntensityTransformerHead(
        **kwargs,
        separate_ion_h_support_head=True,
    )

    target.load_state_dict(source.state_dict(), strict=True)

    assert target.ion_h_support_head is not None
    assert th.equal(target.ion_h_support_head.weight, target.ion_hydrogen_head.weight)
    assert th.equal(target.ion_h_support_head.bias, target.ion_hydrogen_head.bias)


def test_removed_atom_decoder_weights_are_rejected_by_load_state_dict() -> None:
    """A checkpoint trained with the removed atom decoder must fail loudly, not load."""
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=6,
        context_dim=3,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        feedforward_dim=24,
        num_inducing_points=4,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
    )
    state_dict = head.state_dict()
    state_dict["atom_decoder_gate"] = th.zeros(())
    state_dict["atom_decoder.layers.0.linear1.weight"] = th.zeros(24, 12)

    with pytest.raises(RuntimeError, match="fragment-to-atom decoder"):
        head.load_state_dict(state_dict, strict=False)


def test_support_and_oos_logits_have_distinct_output_slots() -> None:
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=2,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        predict_oos=True,
    )
    outputs = head(
        th.randn(2, 4),
        th.randn(2, 4),
        th.randn(2, 4),
        th.zeros(2),
        th.zeros(2),
        th.tensor([0, 0]),
        th.tensor([0, 0]),
        batch_size=1,
        num_masks=1,
        return_support_logits=True,
    )

    assert outputs[3].shape == (2, 2, 3)
    assert outputs[4].shape == (1,)


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_candidate_intensity_transformer_is_permutation_equivariant(
    attention_type: Literal["induced", "full"],
) -> None:
    th.manual_seed(41)
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=6,
        context_dim=3,
        model_dim=12,
        num_layers=2,
        num_heads=3,
        feedforward_dim=24,
        num_inducing_points=4,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
        attention_type=attention_type,
    ).eval()
    fragment_embeddings = th.randn(7, 6)
    complement_embeddings = th.randn(7, 6)
    boundary_embeddings = th.randn(7, 6)
    charge_log_probabilities = th.randn(7)
    mask_log_probabilities = th.randn(7)
    component_batch_indices = th.tensor([0, 0, 0, 1, 1, 1, 1])
    component_mask_indices = th.tensor([0, 0, 1, 2, 2, 3, 4])
    context = th.randn(7, 3)
    permutation = th.tensor([4, 1, 6, 0, 3, 2, 5])

    outputs = head(
        fragment_embeddings,
        complement_embeddings,
        boundary_embeddings,
        charge_log_probabilities,
        mask_log_probabilities,
        component_batch_indices,
        component_mask_indices,
        batch_size=2,
        num_masks=5,
        context=context,
    )
    permuted_outputs = head(
        fragment_embeddings[permutation],
        complement_embeddings[permutation],
        boundary_embeddings[permutation],
        charge_log_probabilities[permutation],
        mask_log_probabilities[permutation],
        component_batch_indices[permutation],
        component_mask_indices[permutation],
        batch_size=2,
        num_masks=5,
        context=context[permutation],
    )
    inverse_permutation = th.argsort(permutation)

    assert outputs[0].shape == (7,)
    assert outputs[1].shape == (5,)
    assert outputs[2].shape == (7, 2, 3)
    assert th.allclose(outputs[0], permuted_outputs[0][inverse_permutation], atol=1e-6)
    assert th.allclose(outputs[1], permuted_outputs[1], atol=1e-6)
    assert th.allclose(outputs[2], permuted_outputs[2][inverse_permutation], atol=1e-6)


def test_candidate_intensity_transformer_backpropagates() -> None:
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=2,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
    )
    fragment_embeddings = th.randn(5, 4, requires_grad=True)
    outputs = head(
        fragment_embeddings,
        th.randn(5, 4),
        th.randn(5, 4),
        th.randn(5),
        th.randn(5),
        th.tensor([0, 0, 1, 1, 1]),
        th.tensor([0, 0, 1, 2, 2]),
        batch_size=2,
        num_masks=3,
        context=th.randn(5, 2),
    )

    sum(output.sum() for output in outputs).backward()

    assert fragment_embeddings.grad is not None
    assert th.isfinite(fragment_embeddings.grad).all()
    assert head.inducing_tokens.grad is not None
    assert head.inducing_tokens.grad.abs().sum() > 0


def test_candidate_intensity_transformer_can_exclude_base_probability_feature() -> None:
    th.manual_seed(42)
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
    ).eval()
    common_features = (
        th.randn(4, 4),
        th.randn(4, 4),
        th.randn(4, 4),
    )
    component_batch_indices = th.tensor([0, 0, 0, 0])
    component_mask_indices = th.tensor([0, 0, 1, 2])

    baseline = head(
        *common_features,
        th.zeros(4),
        th.zeros(4),
        component_batch_indices,
        component_mask_indices,
        batch_size=1,
        num_masks=3,
    )
    perturbed = head(
        *common_features,
        th.tensor([100.0, 10.0, -10.0, -100.0]),
        th.tensor([-100.0, -10.0, 10.0, 100.0]),
        component_batch_indices,
        component_mask_indices,
        batch_size=1,
        num_masks=3,
    )

    assert head.input_projection[0].in_features == 3 * 4 + 1
    for baseline_output, perturbed_output in zip(baseline, perturbed, strict=True):
        assert th.allclose(baseline_output, perturbed_output)


def test_candidate_intensity_transformer_buckets_ragged_padding() -> None:
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
    ).eval()
    observed_shapes: list[th.Size] = []
    hook = head.layers[0].register_forward_pre_hook(
        lambda _module, args: observed_shapes.append(args[0].shape)
    )

    tokens = th.randn(25, 8)
    component_batch_indices = th.tensor([0] * 2 + [1] * 3 + [2] * 20)
    contextualized = head._contextualize_by_molecule(
        tokens,
        component_batch_indices,
        batch_size=3,
    )
    hook.remove()

    assert contextualized.shape == (25, 8)
    assert observed_shapes == [
        th.Size([1, 2, 8]),
        th.Size([1, 4, 8]),
        th.Size([1, 32, 8]),
    ]
    assert sum(shape[0] * shape[1] for shape in observed_shapes) < 3 * 20


def test_candidate_intensity_transformer_preserves_large_counts_under_bfloat16() -> None:
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
    ).eval()
    observed_shapes: list[th.Size] = []
    hook = head.layers[0].register_forward_pre_hook(
        lambda _module, args: observed_shapes.append(args[0].shape)
    )
    tokens = th.randn(261, 8)
    component_batch_indices = th.zeros(261, dtype=th.long)

    assert "component_counts" not in inspect.signature(head.forward).parameters
    with th.autocast(device_type="cpu", dtype=th.bfloat16):
        contextualized = head._contextualize_by_molecule(
            tokens,
            component_batch_indices,
            batch_size=1,
        )
    hook.remove()

    assert contextualized.shape == tokens.shape
    assert observed_shapes == [th.Size([1, 512, 8])]


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_candidate_intensity_transformer_isolates_molecules(
    attention_type: Literal["induced", "full"],
) -> None:
    th.manual_seed(43)
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
        attention_type=attention_type,
    ).eval()
    fragment_embeddings = th.randn(5, 4)
    complement_embeddings = th.randn(5, 4)
    boundary_embeddings = th.randn(5, 4)
    common_args = (
        complement_embeddings,
        boundary_embeddings,
        th.randn(5),
        th.randn(5),
        th.tensor([0, 0, 1, 1, 1]),
        th.tensor([0, 0, 1, 2, 2]),
    )

    baseline = head(
        fragment_embeddings,
        *common_args,
        batch_size=2,
        num_masks=3,
    )
    perturbed_embeddings = fragment_embeddings.clone()
    perturbed_embeddings[2:] += 100.0
    perturbed = head(
        perturbed_embeddings,
        *common_args,
        batch_size=2,
        num_masks=3,
    )

    assert th.allclose(baseline[0][:2], perturbed[0][:2])
    assert th.allclose(baseline[1][0], perturbed[1][0])
    assert th.allclose(baseline[2][:2], perturbed[2][:2])


def _build_contextualization_head(
    attention_type: Literal["induced", "full"],
    *,
    num_layers: int = 1,
) -> CandidateIntensityTransformerHead:
    return CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=8,
        num_layers=num_layers,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=3,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=True,
        attention_type=attention_type,
    ).eval()


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_candidate_contextualization_ignores_bucket_padding(
    attention_type: Literal["induced", "full"],
) -> None:
    """A ragged molecule must match an unpadded reference pass exactly."""
    th.manual_seed(51)
    head = _build_contextualization_head(attention_type)
    # Three candidates land in the power-of-two bucket of four, leaving one padded slot.
    tokens = th.randn(3, 8)

    bucketed = head._contextualize_by_molecule(
        tokens,
        th.zeros(3, dtype=th.long),
        batch_size=1,
    )
    inducing_tokens = (
        None if head.inducing_tokens is None else head.inducing_tokens.expand(1, -1, -1)
    )
    reference, _ = head.layers[0](
        tokens.unsqueeze(0),
        inducing_tokens,
        th.zeros((1, 3), dtype=th.bool),
    )

    assert th.allclose(bucketed, reference.squeeze(0), atol=1e-6)


def test_full_attention_couples_candidates_within_a_molecule() -> None:
    """Direct candidate attention must propagate one candidate into another."""
    th.manual_seed(52)
    head = _build_contextualization_head("full")
    tokens = th.randn(4, 8)
    component_batch_indices = th.tensor([0, 0, 1, 1])

    baseline = head._contextualize_by_molecule(
        tokens,
        component_batch_indices,
        batch_size=2,
    )
    perturbed_tokens = tokens.clone()
    perturbed_tokens[0] += 5.0
    perturbed = head._contextualize_by_molecule(
        perturbed_tokens,
        component_batch_indices,
        batch_size=2,
    )

    assert not th.allclose(baseline[1], perturbed[1], atol=1e-4)
    assert th.allclose(baseline[2:], perturbed[2:], atol=1e-6)


def test_full_attention_drops_the_latent_set_and_its_parameters() -> None:
    induced = _build_contextualization_head("induced", num_layers=2)
    full = _build_contextualization_head("full", num_layers=2)

    assert induced.inducing_tokens is not None
    assert full.inducing_tokens is None
    assert not any("inducing_tokens" in name for name, _ in full.named_parameters())
    assert not any("latent" in name for name, _ in full.named_parameters())
    # Full attention carries one attention block and one feed-forward block per layer
    # rather than two, so a win over induced is not bought with extra parameters.
    assert sum(p.numel() for p in full.parameters()) < sum(
        p.numel() for p in induced.parameters()
    )


def test_full_attention_backpropagates_through_every_layer_parameter() -> None:
    th.manual_seed(53)
    head = _build_contextualization_head("full", num_layers=2)
    tokens = th.randn(5, 8, requires_grad=True)

    contextualized = head._contextualize_by_molecule(
        tokens,
        th.tensor([0, 0, 0, 1, 1]),
        batch_size=2,
    )
    contextualized.square().sum().backward()

    assert tokens.grad is not None
    assert th.isfinite(tokens.grad).all()
    for name, parameter in head.layers.named_parameters():
        assert parameter.grad is not None, name
        assert th.isfinite(parameter.grad).all(), name


def test_candidate_intensity_transformer_validates_attention_type() -> None:
    with pytest.raises(ValueError, match="unknown attention_type"):
        _build_contextualization_head("inducing")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_inducing_points must be positive"):
        CandidateIntensityTransformerHead(
            fragment_embedding_dim=4,
            context_dim=0,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            num_inducing_points=0,
                num_ion_modes=1,
            num_h_shifts=3,
            dropout=0.0,
            use_base_probability_feature=True,
        )
    # Full attention never reads the latent count, so zero is acceptable there.
    assert (
        CandidateIntensityTransformerHead(
            fragment_embedding_dim=4,
            context_dim=0,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            num_inducing_points=0,
                num_ion_modes=1,
            num_h_shifts=3,
            dropout=0.0,
            use_base_probability_feature=True,
            attention_type="full",
        ).inducing_tokens
        is None
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


def test_bond_focused_head_is_endpoint_invariant_and_backpropagates() -> None:
    th.manual_seed(23)
    head = BondFocusedBreakHead(
        atom_embedding_dim=8,
        bond_feature_dim=3,
        context_dim=2,
        bond_state_dim=8,
        head_hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    atom_embeddings = th.randn(5, 8, requires_grad=True)
    bond_features = th.randn(3, 3, requires_grad=True)
    context = th.randn(2, 2, requires_grad=True)
    endpoints = th.tensor([[0, 1, 3], [1, 2, 4]])
    bond_batch_indices = th.tensor([0, 0, 1])

    forward_logits = head(
        atom_embeddings,
        endpoints,
        bond_features,
        context=context,
        bond_batch_indices=bond_batch_indices,
    )
    reverse_logits = head(
        atom_embeddings,
        endpoints.flip(0),
        bond_features,
        context=context,
        bond_batch_indices=bond_batch_indices,
    )

    assert th.allclose(forward_logits, reverse_logits)
    forward_logits.sum().backward()
    assert atom_embeddings.grad is not None
    assert bond_features.grad is not None
    assert context.grad is not None


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
def test_bond_focused_head_supports_cuda_bfloat16_autocast() -> None:
    head = BondFocusedBreakHead(
        atom_embedding_dim=8,
        bond_feature_dim=3,
        context_dim=2,
        bond_state_dim=8,
        head_hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    ).cuda()
    atom_embeddings = th.randn(5, 8, device="cuda", requires_grad=True)
    bond_features = th.randn(3, 3, device="cuda", requires_grad=True)
    context = th.randn(2, 2, device="cuda", requires_grad=True)
    endpoints = th.tensor([[0, 1, 3], [1, 2, 4]], device="cuda")
    bond_batch_indices = th.tensor([0, 0, 1], device="cuda")

    with th.autocast(device_type="cuda", dtype=th.bfloat16):
        logits = head(
            atom_embeddings,
            endpoints,
            bond_features,
            context=context,
            bond_batch_indices=bond_batch_indices,
        )
    logits.sum().backward()

    assert logits.shape == (3,)
    assert atom_embeddings.grad is not None
    assert bond_features.grad is not None
    assert context.grad is not None


def test_bond_focused_head_communicates_only_between_adjacent_bonds() -> None:
    th.manual_seed(29)
    head = BondFocusedBreakHead(
        atom_embedding_dim=6,
        bond_feature_dim=2,
        context_dim=0,
        bond_state_dim=6,
        head_hidden_dim=6,
        num_layers=2,
        dropout=0.0,
    ).eval()
    atom_embeddings = th.randn(5, 6)
    endpoints = th.tensor([[0, 1, 3], [1, 2, 4]])
    bond_features = th.randn(3, 2)
    perturbed_features = bond_features.clone()
    perturbed_features[0] += 5.0

    baseline = head(atom_embeddings, endpoints, bond_features)
    perturbed = head(atom_embeddings, endpoints, perturbed_features)

    assert not th.isclose(baseline[1], perturbed[1])
    assert th.isclose(baseline[2], perturbed[2])


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

    assert th.exp(log_probabilities).tolist() == pytest.approx([0.5, 0.5, 0.25, 0.75])
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
    component_logits = th.stack([th.logsumexp(atom_logits[atoms], dim=0) for atoms in memberships])
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


@pytest.mark.parametrize("pooling", ["logsumexp", "logmeanexp"])
def test_latent_pathway_charge_probabilities_match_expanded_components(
    pooling: Literal["logsumexp", "logmeanexp"],
) -> None:
    atom_logits = th.tensor([0.2, -0.4, 1.1], requires_grad=True)
    unique_memberships = ((0, 1), (2,), (0,))
    pathway_unique_indices = th.tensor([0, 1, 0, 2])
    pathway_mask_indices = th.tensor([0, 0, 1, 1])
    unique_atom_indices = th.tensor([atom for atoms in unique_memberships for atom in atoms])
    atom_unique_indices = th.tensor(
        [unique for unique, atoms in enumerate(unique_memberships) for _ in atoms]
    )

    actual = latent_pathway_charge_log_probabilities(
        atom_logits,
        unique_atom_indices,
        atom_unique_indices,
        pathway_unique_indices,
        pathway_mask_indices,
        num_unique_components=3,
        num_masks=2,
        pooling=pooling,
    )
    expanded_memberships = [unique_memberships[index] for index in pathway_unique_indices.tolist()]
    expected = component_charge_log_probabilities(
        atom_logits,
        th.tensor([atom for atoms in expanded_memberships for atom in atoms]),
        th.tensor([pathway for pathway, atoms in enumerate(expanded_memberships) for _ in atoms]),
        pathway_mask_indices,
        pooling=pooling,
    )

    assert th.allclose(actual, expected)
    actual.sum().backward()
    assert atom_logits.grad is not None
    assert th.isfinite(atom_logits.grad).all()


@pytest.mark.skipif(
    not th.cuda.is_available() or not th.cuda.is_bf16_supported(),
    reason="CUDA BF16 not available",
)
def test_latent_pathway_charge_probabilities_support_cuda_bfloat16_autocast() -> None:
    atom_logits = th.tensor(
        [0.2, -0.4, 1.1],
        device="cuda",
        dtype=th.bfloat16,
        requires_grad=True,
    )

    with th.autocast(device_type="cuda", dtype=th.bfloat16):
        actual = latent_pathway_charge_log_probabilities(
            atom_logits,
            unique_atom_indices=th.tensor([0, 1, 2, 0], device="cuda"),
            atom_unique_component_indices=th.tensor([0, 0, 1, 2], device="cuda"),
            pathway_unique_component_indices=th.tensor([0, 1, 0, 2], device="cuda"),
            pathway_mask_indices=th.tensor([0, 0, 1, 1], device="cuda"),
            num_unique_components=3,
            num_masks=2,
        )
        loss = -actual.mean()
    loss.backward()

    mask_indices = th.tensor([0, 0, 1, 1], device="cuda")
    mask_probabilities = actual.exp().new_zeros(2).index_add(
        0,
        mask_indices,
        actual.exp(),
    )
    assert mask_probabilities.tolist() == pytest.approx([1.0, 1.0], abs=2e-3)
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
