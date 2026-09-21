"""Tests for the 2026-09-21 attention rework and parameter-budget changes.

Covers the candidate transformer's normalization placement, QK-norm and pairwise mass
bias, the bond head's endpoint aggregation, the GNN residual-branch activation, and the
weight-decay parameter split. Every option defaults to the historical behaviour, so each
test also pins the default.
"""

from __future__ import annotations

import pytest
import torch as th

from fragnnet.model.bond_fragment_heads import (
    BondFocusedBreakHead,
    CandidateIntensityTransformerHead,
    PairwiseMassBias,
    _BiasedSelfAttention,
)
from fragnnet.model.nn_blocks import GNN

HEAD_KWARGS = {
    "fragment_embedding_dim": 4,
    "context_dim": 0,
    "model_dim": 8,
    "num_layers": 1,
    "num_heads": 2,
    "feedforward_dim": 16,
    "num_inducing_points": 2,
    "num_ion_modes": 2,
    "num_h_shifts": 3,
    "dropout": 0.0,
    "use_base_probability_feature": False,
}


def _head_inputs(num_components: int = 4) -> tuple:
    th.manual_seed(7)
    return (
        th.randn(num_components, 4),
        th.randn(num_components, 4),
        th.randn(num_components, 4),
        th.tensor([1, 2, 1, 3][:num_components]),
        th.tensor([2, 3, 4, 2][:num_components]),
        th.zeros(num_components),
        th.zeros(num_components),
        th.zeros(num_components, dtype=th.long),
        th.arange(num_components),
    )


# --------------------------------------------------------------------------- attention


def test_biased_self_attention_matches_stock_multihead_attention() -> None:
    """The refactored layer is a drop-in: same state dict, same numbers."""
    th.manual_seed(0)
    model_dim, num_heads = 32, 4
    stock = th.nn.MultiheadAttention(model_dim, num_heads, dropout=0.0, batch_first=True)
    ours = _BiasedSelfAttention(model_dim, num_heads, dropout=0.0)
    missing, unexpected = ours.load_state_dict(stock.state_dict(), strict=False)
    assert missing == [] and unexpected == []

    tokens = th.randn(3, 6, model_dim)
    padding = th.zeros(3, 6, dtype=th.bool)
    padding[0, 4:] = True
    stock.eval()
    ours.eval()
    reference, _ = stock(tokens, tokens, tokens, key_padding_mask=padding, need_weights=False)
    assert th.allclose(ours(tokens, padding), reference, atol=1e-6)


def test_pairwise_mass_bias_starts_at_zero_and_respects_padding() -> None:
    bias_module = PairwiseMassBias()
    masses = th.tensor([[100.0, 82.0, 72.0, 0.0]])
    valid = th.tensor([[True, True, True, False]])

    bias = bias_module(masses, valid)
    assert bias.shape == (1, 1, 4, 4)
    assert th.count_nonzero(bias) == 0, "zero-initialized weights must make this a no-op"

    with th.no_grad():
        water_centre = int(th.argmin((bias_module.centres - 18.0106).abs()))
        bias_module.weights[water_centre] = 1.0
    bias = bias_module(masses, valid)
    # 100 -> 82 is a water loss and should sit at the centre; 100 -> 72 should not.
    assert bias[0, 0, 0, 1] > 0.9
    assert bias[0, 0, 0, 2] < 0.2
    assert bias[0, 0, :, 3].abs().max() == 0.0, "padded candidates must never be biased"


def test_mass_pair_bias_is_a_no_op_at_initialization() -> None:
    """Enabling the bias must not perturb a warm start."""
    th.manual_seed(3)
    baseline = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full")
    th.manual_seed(3)
    biased = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type="full", mass_pair_bias=True
    )
    biased.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    biased.eval()

    inputs = _head_inputs()
    masses = th.tensor([180.0, 162.0, 152.0, 134.0])
    reference = baseline(*inputs, batch_size=1, num_masks=4)
    with_bias = biased(*inputs, batch_size=1, num_masks=4, component_neutral_masses=masses)
    for left, right in zip(reference, with_bias):
        assert th.allclose(left, right, atol=1e-6)


def test_mass_pair_bias_changes_outputs_once_weighted() -> None:
    th.manual_seed(3)
    head = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type="full", mass_pair_bias=True
    )
    head.eval()
    inputs = _head_inputs()
    masses = th.tensor([180.0, 162.0, 152.0, 134.0])
    before = head(*inputs, batch_size=1, num_masks=4, component_neutral_masses=masses)[2]
    with th.no_grad():
        head.mass_pair_bias.weights.fill_(2.0)
    after = head(*inputs, batch_size=1, num_masks=4, component_neutral_masses=masses)[2]
    assert not th.allclose(before, after)


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_norm_placement_keeps_parameter_names(attention_type: str) -> None:
    """post/pre must be A/B-comparable from one checkpoint, so names cannot change."""
    post = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type=attention_type)
    pre = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type=attention_type, norm_placement="pre"
    )
    assert set(post.state_dict()) == set(pre.state_dict())
    assert sum(p.numel() for p in post.parameters()) == sum(p.numel() for p in pre.parameters())


def test_pre_norm_changes_the_computation() -> None:
    th.manual_seed(3)
    post = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full")
    th.manual_seed(3)
    pre = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type="full", norm_placement="pre"
    )
    pre.load_state_dict(post.state_dict())
    post.eval()
    pre.eval()
    inputs = _head_inputs()
    assert not th.allclose(
        post(*inputs, batch_size=1, num_masks=4)[2],
        pre(*inputs, batch_size=1, num_masks=4)[2],
    )


def test_pair_bias_and_qk_norm_require_full_attention() -> None:
    with pytest.raises(ValueError, match="mass_pair_bias requires"):
        CandidateIntensityTransformerHead(**HEAD_KWARGS, mass_pair_bias=True)
    with pytest.raises(ValueError, match="qk_norm requires"):
        CandidateIntensityTransformerHead(**HEAD_KWARGS, qk_norm=True)


def test_qk_norm_adds_only_query_key_norm_parameters() -> None:
    plain = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full")
    normed = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full", qk_norm=True)
    added = set(normed.state_dict()) - set(plain.state_dict())
    assert added and all("q_norm" in key or "k_norm" in key for key in added)


def test_mass_pair_bias_requires_masses() -> None:
    head = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type="full", mass_pair_bias=True
    )
    with pytest.raises(ValueError, match="component_neutral_masses"):
        head(*_head_inputs(), batch_size=1, num_masks=4)


# ------------------------------------------------------------------------- bond head


def _bond_head(aggregation: str) -> BondFocusedBreakHead:
    th.manual_seed(11)
    return BondFocusedBreakHead(
        atom_embedding_dim=8,
        bond_feature_dim=3,
        context_dim=0,
        bond_state_dim=8,
        head_hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        endpoint_aggregation=aggregation,
    )


def test_endpoint_aggregation_defaults_to_mean_and_validates() -> None:
    assert _bond_head("mean").endpoint_aggregation == "mean"
    assert (
        BondFocusedBreakHead(
            atom_embedding_dim=8,
            bond_feature_dim=3,
            context_dim=0,
            bond_state_dim=8,
            head_hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        ).endpoint_aggregation
        == "mean"
    )
    with pytest.raises(ValueError, match="endpoint_aggregation"):
        _bond_head("median")


def test_endpoint_aggregation_modes_differ_on_unequal_degrees() -> None:
    """A star graph has one degree-3 atom and three degree-1 atoms."""
    th.manual_seed(5)
    atom_embeddings = th.randn(4, 8)
    bond_features = th.randn(3, 3)
    endpoints = th.tensor([[0, 0, 0], [1, 2, 3]])

    outputs = {}
    for aggregation in ("mean", "sum", "sqrt"):
        head = _bond_head(aggregation)
        head.eval()
        outputs[aggregation] = head(atom_embeddings, endpoints, bond_features)
    assert not th.allclose(outputs["mean"], outputs["sum"])
    assert not th.allclose(outputs["mean"], outputs["sqrt"])


# -------------------------------------------------------------------- GNN residuals


@pytest.mark.parametrize("gnn_type", ["GINE", "GINE_GSA", "NodeMLP"])
def test_residual_activation_defaults_to_relu(gnn_type: str) -> None:
    gnn = GNN(
        hidden_size=16,
        num_layers=2,
        node_feats_size=8,
        edge_feats_size=4 if gnn_type != "NodeMLP" else 0,
        gnn_type=gnn_type,
        dropout=0.0,
        normalization="layer",
    )
    assert gnn.gnn.residual_activation == "relu"


def test_residual_activation_is_validated() -> None:
    with pytest.raises(ValueError, match="residual_activation"):
        GNN(
            hidden_size=16,
            num_layers=1,
            node_feats_size=8,
            edge_feats_size=4,
            gnn_type="GINE",
            dropout=0.0,
            normalization="layer",
            residual_activation="swish",
        )


def test_identity_residual_allows_negative_branch_contributions() -> None:
    """With relu the branch can only add; with identity it can attenuate."""
    th.manual_seed(13)
    node_features = th.randn(6, 8)
    edge_index = th.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_attr = th.randn(5, 4)
    batch_index = th.zeros(6, dtype=th.long)

    branches = {}
    for activation in ("relu", "identity"):
        th.manual_seed(13)
        gnn = GNN(
            hidden_size=8,
            num_layers=1,
            node_feats_size=8,
            edge_feats_size=4,
            gnn_type="GINE",
            dropout=0.0,
            normalization="layer",
            residual_activation=activation,
        )
        gnn.eval()
        projected = gnn.input_project(node_features)
        output = gnn.gnn(projected, batch_index, edge_index, edge_attr)
        branches[activation] = output - projected

    assert branches["relu"].min() >= 0.0
    assert branches["identity"].min() < 0.0


# ------------------------------------------------------------------ weight-decay split


def test_weight_decay_split_exempts_norms_biases_and_free_tokens() -> None:
    """LayerNorm weights, biases and 1-D learned tokens must not be decayed."""
    from fragnnet.pl_model.spectrum_pl import SpectrumPL

    class _Stub(th.nn.Module):
        _split_weight_decay_parameters = SpectrumPL._split_weight_decay_parameters

        def __init__(self) -> None:
            super().__init__()
            self.linear = th.nn.Linear(4, 4)
            self.norm = th.nn.LayerNorm(4)
            self.batch_norm = th.nn.BatchNorm1d(4)
            self.inducing_tokens = th.nn.Parameter(th.zeros(4))

    stub = _Stub()
    decay, no_decay = stub._split_weight_decay_parameters()

    assert all(parameter.ndim >= 2 for parameter in decay)
    assert len(decay) == 1 and decay[0] is stub.linear.weight
    exempt = {id(parameter) for parameter in no_decay}
    for parameter in (
        stub.linear.bias,
        stub.norm.weight,
        stub.norm.bias,
        stub.batch_norm.weight,
        stub.batch_norm.bias,
        stub.inducing_tokens,
    ):
        assert id(parameter) in exempt
    # Every trainable parameter is covered exactly once.
    assert len(decay) + len(no_decay) == len(list(stub.parameters()))


# --------------------------------------------------------------- CE-conditioned attention

CE_DIM = 5


def _ce_head(**overrides) -> CandidateIntensityTransformerHead:
    th.manual_seed(3)
    return CandidateIntensityTransformerHead(
        **HEAD_KWARGS,
        attention_type="full",
        ce_feature_dim=CE_DIM,
        **overrides,
    )


def test_ce_pair_bias_requires_the_mass_bias_and_full_attention() -> None:
    with pytest.raises(ValueError, match="ce_pair_bias modulates"):
        _ce_head(ce_pair_bias=True)
    with pytest.raises(ValueError, match="ce_film requires"):
        CandidateIntensityTransformerHead(**HEAD_KWARGS, ce_film=True, ce_feature_dim=CE_DIM)
    with pytest.raises(ValueError, match="positive ce_feature_dim"):
        CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full", ce_film=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mass_pair_bias": True, "ce_pair_bias": True},
        {"ce_film": True},
        {"mass_pair_bias": True, "ce_pair_bias": True, "ce_film": True},
    ],
)
def test_ce_conditioning_is_a_no_op_at_initialization(overrides: dict) -> None:
    """Zero-initialized CE projections must leave step 0 identical to the unconditioned head."""
    needs_masses = overrides.get("mass_pair_bias", False)
    th.manual_seed(3)
    baseline = CandidateIntensityTransformerHead(
        **HEAD_KWARGS, attention_type="full", mass_pair_bias=needs_masses
    )
    conditioned = _ce_head(**overrides)
    conditioned.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    conditioned.eval()

    inputs = _head_inputs()
    masses = th.tensor([180.0, 162.0, 152.0, 134.0]) if needs_masses else None
    ce_features = th.randn(1, CE_DIM)
    reference = baseline(*inputs, batch_size=1, num_masks=4, component_neutral_masses=masses)
    with_ce = conditioned(
        *inputs,
        batch_size=1,
        num_masks=4,
        component_neutral_masses=masses,
        ce_features=ce_features,
    )
    for left, right in zip(reference, with_ce):
        assert th.allclose(left, right, atol=1e-6)


def test_ce_pair_bias_makes_the_bias_depend_on_collision_energy() -> None:
    bias_module = PairwiseMassBias(ce_feature_dim=CE_DIM)
    masses = th.tensor([[100.0, 82.0, 72.0]]).expand(2, 3).contiguous()
    valid = th.ones(2, 3, dtype=th.bool)
    low_then_high = th.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])

    assert th.count_nonzero(bias_module(masses, valid, ce_features=low_then_high)) == 0
    with th.no_grad():
        bias_module.ce_weights.weight[:, 0] = 1.0
        bias_module.ce_weights.weight[:, 1] = -1.0
    bias = bias_module(masses, valid, ce_features=low_then_high)
    # Identical candidate masses, different collision energies -> different competition.
    assert not th.allclose(bias[0], bias[1])


def test_ce_film_changes_attention_once_weighted() -> None:
    head = _ce_head(ce_film=True)
    head.eval()
    inputs = _head_inputs()
    ce_features = th.randn(1, CE_DIM)
    before = head(*inputs, batch_size=1, num_masks=4, ce_features=ce_features)[2]
    with th.no_grad():
        for layer in head.layers:
            th.nn.init.normal_(layer.candidate_attention.ce_film.weight, std=0.5)
    after = head(*inputs, batch_size=1, num_masks=4, ce_features=ce_features)[2]
    assert not th.allclose(before, after)


def test_ce_features_are_required_exactly_when_conditioning_is_on() -> None:
    conditioned = _ce_head(ce_film=True)
    with pytest.raises(ValueError, match="ce_features must be supplied"):
        conditioned(*_head_inputs(), batch_size=1, num_masks=4)
    plain = CandidateIntensityTransformerHead(**HEAD_KWARGS, attention_type="full")
    with pytest.raises(ValueError, match="ce_features must be supplied"):
        plain(*_head_inputs(), batch_size=1, num_masks=4, ce_features=th.randn(1, CE_DIM))
