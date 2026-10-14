"""Tests for the 2026-09-21 attention rework and parameter-budget changes.

Covers the candidate transformer's normalization placement, QK-norm and pairwise mass
bias, the bond head's endpoint aggregation, the GNN residual-branch activation, and the
weight-decay parameter split. Every option defaults to the historical behaviour, so each
test also pins the default.
"""

from __future__ import annotations

import math

import pytest
import torch as th

from fragnnet.layers.gnn import GNN
from fragnnet.models.masster.masster_candidate_transformer import (
    CandidateIntensityTransformerHead,
    PairwiseMassBias,
    _BiasedSelfAttention,
)
from fragnnet.models.masster.masster_proposal_heads import BondFocusedBreakHead

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


# ----------------------------------------------------------------- m/z positional encoding


def _mz_head(**overrides) -> CandidateIntensityTransformerHead:
    th.manual_seed(3)
    return CandidateIntensityTransformerHead(**{**HEAD_KWARGS, **overrides}).eval()


def test_mz_encoding_matches_the_transformer_formula() -> None:
    head = _mz_head(mz_positional_encoding=True)
    masses = th.tensor([0.0, 18.0106, 134.25, 1499.9], dtype=th.float64)

    encoding = head._mz_encoding(masses)

    expected = th.zeros(4, 8, dtype=th.float64)
    for row, mass in enumerate(masses.tolist()):
        for i in range(4):
            angle = mass * 10000.0 ** (-2 * i / 8)
            expected[row, 2 * i], expected[row, 2 * i + 1] = math.sin(angle), math.cos(angle)
    assert encoding.shape == (4, 8)
    assert th.allclose(encoding, expected, atol=1e-12)


def test_mz_encoding_of_a_mass_offset_is_a_fixed_rotation() -> None:
    """PE(m + dm) = R(dm) PE(m) for every m, which lets attention express a neutral loss."""
    head = _mz_head(mz_positional_encoding=True)
    masses = th.tensor([90.0, 162.0, 400.5], dtype=th.float64)
    offset = 18.0106
    base, shifted = head._mz_encoding(masses), head._mz_encoding(masses + offset)

    angles = offset * head.mz_inverse_frequencies
    sin_base, cos_base = base[:, 0::2], base[:, 1::2]
    assert th.allclose(shifted[:, 0::2], sin_base * angles.cos() + cos_base * angles.sin())
    assert th.allclose(shifted[:, 1::2], cos_base * angles.cos() - sin_base * angles.sin())


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_mz_encoding_adds_no_parameters_and_keeps_initialization(attention_type: str) -> None:
    plain = _mz_head(attention_type=attention_type).state_dict()
    encoded = _mz_head(attention_type=attention_type, mz_positional_encoding=True).state_dict()

    assert list(encoded) == list(plain)
    assert all(th.equal(encoded[key], plain[key]) for key in plain)


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_mz_encoding_makes_the_output_depend_on_candidate_mass(attention_type: str) -> None:
    plain = _mz_head(attention_type=attention_type)
    encoded = _mz_head(attention_type=attention_type, mz_positional_encoding=True)
    inputs = _head_inputs()
    masses = th.tensor([180.0, 162.0, 152.0, 134.0])

    reference = plain(*inputs, batch_size=1, num_masks=4)[2]
    with_masses = encoded(*inputs, batch_size=1, num_masks=4, component_neutral_masses=masses)[2]
    flipped = masses.flip(0)
    swapped = encoded(*inputs, batch_size=1, num_masks=4, component_neutral_masses=flipped)[2]
    assert not th.allclose(with_masses, reference)
    assert not th.allclose(with_masses, swapped)


def test_mz_encoding_requires_masses() -> None:
    head = _mz_head(mz_positional_encoding=True)
    with pytest.raises(ValueError, match="mz_positional_encoding is on"):
        head(*_head_inputs(), batch_size=1, num_masks=4)
    with pytest.raises(ValueError, match="even model_dim"):
        _mz_head(mz_positional_encoding=True, model_dim=9, num_heads=3)


# ------------------------------------------------------------- neutral-loss encoding

# Two spectra: candidates 0-1 belong to a 180 Da precursor (candidate 0 is the precursor), and
# candidates 2-3 to a 152 Da precursor.
NL_MASSES = th.tensor([180.0, 162.0, 152.0, 134.0])
NL_PRECURSORS = th.tensor([180.0, 152.0])
NL_BATCH = th.tensor([0, 0, 1, 1])


def _two_spectrum_inputs() -> tuple:
    inputs = list(_head_inputs())
    inputs[5] = NL_BATCH
    return tuple(inputs)


def _pre_attention_tokens(head: CandidateIntensityTransformerHead, **kwargs) -> th.Tensor:
    """Run ``head`` and return the tokens that enter candidate contextualization."""
    captured = []

    def capture(tokens: th.Tensor, *args, **kwargs) -> th.Tensor:
        captured.append(tokens)
        return tokens

    head._contextualize_by_molecule = capture
    head(*_two_spectrum_inputs(), batch_size=2, num_masks=4, **kwargs)
    return captured[0]


def _encoding(positions: th.Tensor, width: int) -> th.Tensor:
    expected = th.zeros(len(positions), width, dtype=th.float64)
    for row, position in enumerate(positions.tolist()):
        for i in range(width // 2):
            angle = position * 10000.0 ** (-2 * i / width)
            expected[row, 2 * i], expected[row, 2 * i + 1] = math.sin(angle), math.cos(angle)
    return expected


def test_nl_encoding_matches_the_transformer_formula_on_the_loss() -> None:
    head = _mz_head(nl_positional_encoding=True)
    losses = th.tensor([0.0, 18.0106, 43.9898, 1250.5], dtype=th.float64)

    assert head.nl_inverse_frequencies.shape == (4,)
    assert th.allclose(head._nl_encoding(losses), _encoding(losses, 8), atol=1e-12)
    # The precursor itself has no loss: sin 0 = 0 and cos 0 = 1 at every frequency.
    assert th.equal(head._nl_encoding(th.zeros(1))[0, 0::2], th.zeros(4, dtype=th.float64))
    assert th.equal(head._nl_encoding(th.zeros(1))[0, 1::2], th.ones(4, dtype=th.float64))


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_nl_encoding_adds_each_spectrums_loss_to_its_tokens(attention_type: str) -> None:
    """Every candidate's token gains PE(M - m) of its own spectrum's precursor mass M."""
    masses = {"component_neutral_masses": NL_MASSES}
    plain = _pre_attention_tokens(_mz_head(attention_type=attention_type))
    encoded = _pre_attention_tokens(
        _mz_head(attention_type=attention_type, nl_positional_encoding=True),
        **masses,
        precursor_neutral_masses=NL_PRECURSORS,
    )

    losses = NL_PRECURSORS[NL_BATCH] - NL_MASSES
    assert th.equal(losses, th.tensor([0.0, 18.0, 0.0, 18.0]))
    assert th.allclose(encoded - plain, _encoding(losses, 8).float(), atol=1e-6)


def test_mz_and_nl_encodings_split_the_token_width() -> None:
    """With both on, the mass takes the first half of the width and the loss the second."""
    head = _mz_head(mz_positional_encoding=True, nl_positional_encoding=True)
    plain = _pre_attention_tokens(_mz_head())
    encoded = _pre_attention_tokens(
        head,
        component_neutral_masses=NL_MASSES,
        precursor_neutral_masses=NL_PRECURSORS,
    )

    assert head.mz_inverse_frequencies.shape == head.nl_inverse_frequencies.shape == (2,)
    expected = th.cat(
        (_encoding(NL_MASSES, 4), _encoding(NL_PRECURSORS[NL_BATCH] - NL_MASSES, 4)), dim=1
    )
    assert th.allclose(encoded - plain, expected.float(), atol=1e-6)
    # Alone, the mass keeps the full width: the configuration mz1 ran with.
    mz_only = _pre_attention_tokens(
        _mz_head(mz_positional_encoding=True), component_neutral_masses=NL_MASSES
    )
    assert th.allclose(mz_only - plain, _encoding(NL_MASSES, 8).float(), atol=1e-6)


@pytest.mark.parametrize("attention_type", ["induced", "full", "none"])
def test_nl_encoding_adds_no_parameters_and_keeps_initialization(attention_type: str) -> None:
    plain = _mz_head(attention_type=attention_type).state_dict()
    for flags in (
        {"nl_positional_encoding": True},
        {"nl_positional_encoding": True, "mz_positional_encoding": True},
    ):
        encoded = _mz_head(attention_type=attention_type, **flags).state_dict()
        assert list(encoded) == list(plain)
        assert all(th.equal(encoded[key], plain[key]) for key in plain)


def test_nl_encoding_moves_outputs_only_with_its_own_precursor() -> None:
    head = _mz_head(attention_type="full", nl_positional_encoding=True)
    inputs = _two_spectrum_inputs()
    masses = {"component_neutral_masses": NL_MASSES}

    reference = head(
        *inputs, batch_size=2, num_masks=4, **masses, precursor_neutral_masses=NL_PRECURSORS
    )[2]
    second_moved = head(
        *inputs,
        batch_size=2,
        num_masks=4,
        **masses,
        precursor_neutral_masses=th.tensor([180.0, 170.0]),
    )[2]
    assert th.allclose(second_moved[:2], reference[:2])
    assert not th.allclose(second_moved[2:], reference[2:])


def test_nl_encoding_requires_precursor_masses() -> None:
    head = _mz_head(nl_positional_encoding=True)
    inputs = _two_spectrum_inputs()
    masses = {"component_neutral_masses": NL_MASSES}
    with pytest.raises(ValueError, match="nl_positional_encoding or mz_positional_encoding is on"):
        head(*inputs, batch_size=2, num_masks=4, precursor_neutral_masses=NL_PRECURSORS)
    with pytest.raises(ValueError, match="precursor_neutral_masses must be supplied exactly"):
        head(*inputs, batch_size=2, num_masks=4, **masses)
    with pytest.raises(ValueError, match="one mass per spectrum"):
        head(
            *inputs,
            batch_size=2,
            num_masks=4,
            **masses,
            precursor_neutral_masses=th.tensor([180.0]),
        )
    with pytest.raises(ValueError, match="precursor_neutral_masses must be supplied exactly"):
        _mz_head()(*inputs, batch_size=2, num_masks=4, precursor_neutral_masses=NL_PRECURSORS)
    with pytest.raises(ValueError, match="even model_dim"):
        _mz_head(nl_positional_encoding=True, model_dim=9, num_heads=3)
    with pytest.raises(ValueError, match="divisible by 4"):
        _mz_head(mz_positional_encoding=True, nl_positional_encoding=True, model_dim=6)


# --------------------------------------------------------- mass window and no attention

# Candidates 0 and 1 lie 10 Da apart; 2 and 3 are more than 15 Da from every other candidate.
WINDOW_MASSES = th.tensor([100.0, 110.0, 200.0, 400.0])


def _moved_outputs(head: CandidateIntensityTransformerHead, moved: int, **kwargs) -> th.Tensor:
    """Which candidates' ion/H logits change when one candidate's input embedding moves."""
    inputs = list(_head_inputs())
    reference = head(*inputs, batch_size=1, num_masks=4, **kwargs)[2]
    inputs[0] = inputs[0].clone()
    inputs[0][moved] += 5.0
    moved_output = head(*inputs, batch_size=1, num_masks=4, **kwargs)[2]
    return ~th.isclose(moved_output, reference, atol=1e-6).all(dim=(1, 2))


def test_mass_window_is_off_by_default_and_validated() -> None:
    head = _mz_head(attention_type="full")
    assert head.mass_window == 0.0
    assert not head.uses_token_masses
    assert _mz_head(attention_type="full", mass_window=15.0).uses_token_masses
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="mass_window must be finite and nonnegative"):
            _mz_head(attention_type="full", mass_window=bad)
    for attention_type in ("induced", "none"):
        with pytest.raises(ValueError, match="mass_window requires attention_type='full'"):
            _mz_head(attention_type=attention_type, mass_window=15.0)


def test_mass_window_hides_candidates_outside_the_window() -> None:
    """With one layer, a candidate reads only the candidates within the window of its mass."""
    head = _mz_head(attention_type="full", mass_window=15.0)
    masses = {"component_neutral_masses": WINDOW_MASSES}

    assert _moved_outputs(head, 1, **masses).tolist() == [True, True, False, False]
    assert _moved_outputs(head, 2, **masses).tolist() == [False, False, True, False]
    # Full attention over the same set lets every candidate read every other.
    assert _moved_outputs(_mz_head(attention_type="full"), 2).all()


def test_mass_window_mask_values() -> None:
    head = _mz_head(attention_type="full", mass_window=15.0)
    padding = th.tensor([[False, False, False, True]])
    masses = th.tensor([[100.0, 110.0, 300.0, 0.0]])

    bias = head._mass_window_bias(masses, padding)

    inf = float("-inf")
    expected = th.tensor(
        [
            [0.0, 0.0, inf, inf],
            [0.0, 0.0, inf, inf],
            [inf, inf, 0.0, inf],
            # The padded row keeps every key; the key padding mask hides padded keys later.
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert bias.shape == (1, 1, 4, 4)
    assert th.equal(bias[0, 0], expected)


def test_mass_window_wider_than_the_set_equals_full_attention() -> None:
    full = _mz_head(attention_type="full")
    windowed = _mz_head(attention_type="full", mass_window=1000.0)
    inputs = _head_inputs()

    assert list(windowed.state_dict()) == list(full.state_dict())
    assert all(th.equal(windowed.state_dict()[k], v) for k, v in full.state_dict().items())
    assert th.allclose(
        windowed(*inputs, batch_size=1, num_masks=4, component_neutral_masses=WINDOW_MASSES)[2],
        full(*inputs, batch_size=1, num_masks=4)[2],
        atol=1e-6,
    )


def test_mass_window_keeps_padded_buckets_finite_through_backward() -> None:
    """A padded slot far from every real mass must not produce an all -inf attention row."""
    head = _mz_head(attention_type="full", mass_window=15.0, num_layers=2).train()
    inputs = list(_head_inputs())
    # Molecule 0 holds three candidates (padded to four); molecule 1 holds one.
    inputs[5] = th.tensor([0, 0, 0, 1])
    masses = th.tensor([300.0, 310.0, 500.0, 250.0])

    outputs = head(*inputs, batch_size=2, num_masks=4, component_neutral_masses=masses)
    sum(output.sum() for output in outputs).backward()

    assert all(th.isfinite(output).all() for output in outputs)
    assert all(
        th.isfinite(parameter.grad).all()
        for parameter in head.parameters()
        if parameter.grad is not None
    )


def test_mass_window_combines_with_the_mass_pair_bias() -> None:
    head = _mz_head(attention_type="full", mass_window=15.0, mass_pair_bias=True)
    with th.no_grad():
        head.mass_pair_bias.weights.fill_(1.0)
    masses = {"component_neutral_masses": WINDOW_MASSES}

    assert _moved_outputs(head, 2, **masses).tolist() == [False, False, True, False]


@pytest.mark.parametrize("norm_placement", ["post", "pre"])
def test_no_attention_scores_each_candidate_from_its_own_token(norm_placement: str) -> None:
    head = _mz_head(attention_type="none", norm_placement=norm_placement, num_layers=2)

    for moved in range(4):
        expected = [index == moved for index in range(4)]
        assert _moved_outputs(head, moved).tolist() == expected


def test_no_attention_keeps_only_feed_forward_sublayers() -> None:
    head = _mz_head(attention_type="none", num_layers=2)
    full = _mz_head(attention_type="full", num_layers=2)
    keys, full_keys = set(head.state_dict()), set(full.state_dict())

    assert head.inducing_tokens is None
    assert not any("attention" in key for key in keys)
    assert keys == {key for key in full_keys if "attention" not in key}
    # The m/z encoding is per token, so it still applies.
    plain = _mz_head(attention_type="none")
    encoded = _mz_head(attention_type="none", mz_positional_encoding=True)
    inputs = _head_inputs()
    with_masses = encoded(
        *inputs, batch_size=1, num_masks=4, component_neutral_masses=WINDOW_MASSES
    )
    assert not th.allclose(with_masses[2], plain(*inputs, batch_size=1, num_masks=4)[2])


def test_no_attention_rejects_attention_options() -> None:
    with pytest.raises(ValueError, match="attention_type='none' has no attention"):
        _mz_head(attention_type="none", set_message_substitution=0.3)
    with pytest.raises(ValueError, match="attention_type='none' has no attention"):
        _mz_head(attention_type="none", set_key_dropout=0.3)
    with pytest.raises(ValueError, match="mass_pair_bias requires"):
        _mz_head(attention_type="none", mass_pair_bias=True)


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
