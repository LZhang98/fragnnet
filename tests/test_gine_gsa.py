"""Tests for the per-graph global attention used by the GINE_GSA encoder."""

from __future__ import annotations

import math

import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.layers.gine_gsa import GINE_GSA, _graph_softmax_attn


def _reference_attention(
    queries: th.Tensor,
    keys: th.Tensor,
    values: th.Tensor,
    batch_index: th.Tensor,
    num_graphs: int,
) -> th.Tensor:
    """Per-graph softmax attention written straight from the definition."""
    head_dim = queries.shape[-1]
    out = th.zeros_like(values)
    for graph in range(num_graphs):
        members = (batch_index == graph).nonzero().flatten()
        for head in range(queries.shape[1]):
            scores = (queries[members, head] @ keys[members, head].T) / math.sqrt(head_dim)
            out[members, head] = th.softmax(scores, dim=-1) @ values[members, head]
    return out


def _random_qkv(
    num_nodes: int,
    num_heads: int,
    head_dim: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    return tuple(th.randn(num_nodes, num_heads, head_dim, dtype=th.float64) for _ in range(3))


def test_graph_softmax_attn_matches_definition() -> None:
    th.manual_seed(11)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    expected = _reference_attention(queries, keys, values, batch_index, 3)

    assert attended.shape == values.shape
    assert th.allclose(attended, expected, atol=1e-12)


def test_graph_softmax_attn_isolates_graphs() -> None:
    """Padded keys must be masked, so one molecule cannot reach another."""
    th.manual_seed(12)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)

    baseline = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    perturbed_inputs = [tensor.clone() for tensor in (queries, keys, values)]
    for tensor in perturbed_inputs:
        tensor[3:] += 50.0
    perturbed = _graph_softmax_attn(*perturbed_inputs, batch_index, 3)

    assert th.allclose(baseline[:3], perturbed[:3], atol=1e-12)
    assert not th.allclose(baseline[3:], perturbed[3:], atol=1e-6)


def test_graph_softmax_attn_is_permutation_equivariant_within_a_graph() -> None:
    th.manual_seed(13)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)
    permutation = th.tensor([2, 0, 1, 3, 4, 5])

    baseline = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    permuted = _graph_softmax_attn(
        queries[permutation],
        keys[permutation],
        values[permutation],
        batch_index,
        3,
    )

    assert th.allclose(baseline[permutation], permuted, atol=1e-12)


def test_graph_softmax_attn_returns_own_value_for_a_singleton_graph() -> None:
    """A one-node graph attends only to itself, so softmax is the identity on V."""
    th.manual_seed(14)
    batch_index = th.tensor([0, 0, 1])
    queries, keys, values = _random_qkv(3, 2, 4)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 2)

    assert th.allclose(attended[2], values[2], atol=1e-12)


def test_graph_softmax_attn_ignores_dense_padding() -> None:
    """A small molecule's output must not depend on how large its batch mates are."""
    th.manual_seed(15)
    small_queries, small_keys, small_values = _random_qkv(2, 2, 4)
    large_queries, large_keys, large_values = _random_qkv(9, 2, 4)

    alone = _graph_softmax_attn(
        small_queries,
        small_keys,
        small_values,
        th.zeros(2, dtype=th.long),
        1,
    )
    padded = _graph_softmax_attn(
        th.cat([small_queries, large_queries]),
        th.cat([small_keys, large_keys]),
        th.cat([small_values, large_values]),
        th.tensor([0, 0] + [1] * 9),
        2,
    )

    assert th.allclose(alone, padded[:2], atol=1e-12)


def test_graph_softmax_attn_tolerates_an_empty_trailing_graph() -> None:
    """A graph with no nodes must not turn a fully masked softmax row into NaN."""
    th.manual_seed(18)
    batch_index = th.tensor([0, 0, 2])
    queries, keys, values = _random_qkv(3, 2, 4)

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 3)

    assert th.isfinite(attended).all()
    assert th.allclose(attended[2], values[2], atol=1e-12)


def test_graph_softmax_attn_rejects_unsorted_batch_index() -> None:
    """Dense packing assumes graph-contiguous nodes; an unsorted batch must fail loudly."""
    queries, keys, values = _random_qkv(4, 2, 4)

    with pytest.raises(ValueError, match="batch_index must be sorted"):
        _graph_softmax_attn(queries, keys, values, th.tensor([0, 1, 0, 1]), 2)


@pytest.mark.parametrize("normalization", ["layer", "graph"])
def test_gine_gsa_forward_and_backward(normalization: str) -> None:
    th.manual_seed(16)
    hidden_size, num_heads = 16, 4
    encoder = GINE_GSA(
        hidden_size=hidden_size,
        node_feats_size=hidden_size,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization=normalization,
        num_heads=num_heads,
    ).eval()
    x = th.randn(5, hidden_size, requires_grad=True)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = th.randn(edge_index.shape[1], 3)

    out = encoder(x, batch_index, edge_index, edge_attr)
    out.square().sum().backward()

    assert out.shape == (5, hidden_size)
    assert th.isfinite(out).all()
    assert x.grad is not None and th.isfinite(x.grad).all()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert th.isfinite(parameter.grad).all(), name
    assert any(
        encoder.get_parameter(name).grad.abs().sum() > 0
        for name, _ in encoder.named_parameters()
        if name.startswith("gsa_")
    )


def test_gine_gsa_keeps_molecules_independent() -> None:
    """A second molecule's features must not change the first molecule's output."""
    th.manual_seed(17)
    hidden_size = 16
    encoder = GINE_GSA(
        hidden_size=hidden_size,
        node_feats_size=hidden_size,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization="layer",
        num_heads=4,
    ).eval()
    x = th.randn(5, hidden_size)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = th.randn(edge_index.shape[1], 3)

    baseline = encoder(x, batch_index, edge_index, edge_attr)
    perturbed_x = x.clone()
    perturbed_x[3:] += 20.0
    perturbed = encoder(perturbed_x, batch_index, edge_index, edge_attr)

    assert th.allclose(baseline[:3], perturbed[:3], atol=1e-5)


# Training-only noise on the normalized global-attention message (memorization-route arms gm1 and
# gm2). Off, it must draw nothing; on, it must act in training only and add no parameters.


def _noisy_encoder(**kwargs) -> GINE_GSA:
    th.manual_seed(21)
    return GINE_GSA(
        hidden_size=16,
        node_feats_size=16,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization="batch",
        num_heads=4,
        **kwargs,
    )


def _encoder_inputs() -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    th.manual_seed(22)
    x = th.randn(5, 16)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = th.randn(edge_index.shape[1], 3)
    return x, batch_index, edge_index, edge_attr


@pytest.mark.parametrize("option", [{}, {"global_message_swap_unit": "molecule"}])
def test_global_message_noise_off_draws_no_random_numbers_in_training(option: dict) -> None:
    encoder = _noisy_encoder(**option).train()
    inputs = _encoder_inputs()
    rng_before = th.get_rng_state()

    encoder(*inputs)

    assert th.equal(th.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "option",
    [
        {"global_message_substitution": 0.5},
        {"global_message_noise_std": 0.5},
        {"global_message_swap": 0.5, "global_message_swap_unit": "atom"},
        {"global_message_swap": 0.9, "global_message_swap_unit": "molecule"},
        {"global_message_swap": 0.5, "global_message_swap_unit": "feature"},
    ],
)
def test_global_message_noise_is_training_only_and_adds_no_parameters(option: dict) -> None:
    plain, noisy = _noisy_encoder(), _noisy_encoder(**option)
    assert list(noisy.state_dict()) == list(plain.state_dict())
    noisy.load_state_dict(plain.state_dict(), strict=True)
    inputs = _encoder_inputs()

    assert th.equal(noisy.eval()(*inputs), plain.eval()(*inputs))
    th.manual_seed(5)
    assert not th.allclose(noisy.train()(*inputs), plain.train()(*inputs))


def test_global_message_substitution_swaps_whole_messages_for_the_detached_batch_mean() -> None:
    encoder = _noisy_encoder(global_message_substitution=0.5)
    th.manual_seed(6)
    message = th.randn(64, 16, requires_grad=True)
    th.manual_seed(7)

    perturbed = encoder._perturb_global_message(message, th.zeros(64, dtype=th.long), 1)
    perturbed.sum().backward()

    batch_mean = message.detach().mean(dim=0)
    replaced = (perturbed == batch_mean).all(dim=1)
    kept = (perturbed == message).all(dim=1)
    assert (replaced ^ kept).all()
    assert 16 < int(replaced.sum()) < 48
    # The mean is detached, so a replaced atom's message gets no gradient.
    assert th.equal(message.grad[replaced], th.zeros(int(replaced.sum()), 16))
    assert th.equal(message.grad[kept], th.ones(int(kept.sum()), 16))


def test_global_message_noise_scales_with_each_features_batch_std() -> None:
    encoder = _noisy_encoder(global_message_noise_std=0.5)
    th.manual_seed(8)
    message = th.randn(20000, 16) * th.linspace(0.5, 4.0, 16)
    message[:, 0] = 3.0
    th.manual_seed(9)

    noise = encoder._perturb_global_message(message, th.zeros(20000, dtype=th.long), 1) - message

    # A constant feature has zero batch spread, so it gets no noise.
    assert th.equal(noise[:, 0], th.zeros(20000))
    expected_std = 0.5 * message[:, 1:].std(dim=0, unbiased=False)
    assert th.allclose(noise[:, 1:].std(dim=0), expected_std, rtol=0.03)
    assert (noise[:, 1:].mean(dim=0).abs() < 0.03 * expected_std).all()


# Swap noise (arms gm3-gm6): replaced messages are other atoms' real, detached messages.


def _molecule_batch(num_molecules: int = 120) -> th.Tensor:
    """Sorted graph assignment with molecule sizes cycling through 1-5 atoms."""
    sizes = th.arange(num_molecules) % 5 + 1
    return th.repeat_interleave(th.arange(num_molecules), sizes)


def _source_rows(perturbed: th.Tensor, message: th.Tensor) -> th.Tensor:
    """Index of the ``message`` row equal to each ``perturbed`` row; raises if any has none."""
    matches = (perturbed[:, None, :] == message[None, :, :]).all(dim=2)
    assert (matches.sum(dim=1) == 1).all()
    return matches.int().argmax(dim=1)


def test_global_message_atom_swap_takes_whole_detached_messages_from_batch_atoms() -> None:
    encoder = _noisy_encoder(global_message_swap=0.5, global_message_swap_unit="atom")
    batch_index = _molecule_batch()
    th.manual_seed(10)
    message = th.randn(batch_index.shape[0], 16, requires_grad=True)
    th.manual_seed(11)

    perturbed = encoder._perturb_global_message(message, batch_index, 120)
    perturbed.sum().backward()

    source = _source_rows(perturbed.detach(), message.detach())
    swapped = source != th.arange(batch_index.shape[0])
    num_atoms = batch_index.shape[0]
    assert 0.4 * num_atoms < int(swapped.sum()) < 0.6 * num_atoms
    # Donors are detached, so a swapped atom's message gets no gradient.
    assert th.equal(message.grad[swapped], th.zeros(int(swapped.sum()), 16))
    assert th.equal(message.grad[~swapped], th.ones(int((~swapped).sum()), 16))


def test_global_message_molecule_swap_moves_whole_molecules_to_one_other_donor() -> None:
    encoder = _noisy_encoder(global_message_swap=0.5, global_message_swap_unit="molecule")
    batch_index = _molecule_batch()
    th.manual_seed(12)
    message = th.randn(batch_index.shape[0], 16, requires_grad=True)
    th.manual_seed(13)

    perturbed = encoder._perturb_global_message(message, batch_index, 120)
    perturbed.sum().backward()

    source_graph = batch_index[_source_rows(perturbed.detach(), message.detach())]
    swapped_molecules = 0
    for graph in range(120):
        atoms = batch_index == graph
        donors = source_graph[atoms].unique()
        # Every atom of a molecule is kept, or every atom comes from one other molecule.
        assert donors.numel() == 1
        if int(donors) == graph:
            assert th.equal(perturbed[atoms], message[atoms])
        else:
            swapped_molecules += 1
            assert th.equal(message.grad[atoms], th.zeros(int(atoms.sum()), 16))
    assert 40 < swapped_molecules < 80


def test_global_message_molecule_swap_leaves_a_single_molecule_unchanged() -> None:
    encoder = _noisy_encoder(global_message_swap=0.9, global_message_swap_unit="molecule")
    message = th.randn(4, 16)

    perturbed = encoder._perturb_global_message(message, th.zeros(4, dtype=th.long), 1)

    assert th.equal(perturbed, message)


def test_global_message_feature_swap_takes_each_entry_from_the_same_feature() -> None:
    encoder = _noisy_encoder(global_message_swap=0.5, global_message_swap_unit="feature")
    batch_index = _molecule_batch()
    th.manual_seed(14)
    message = th.randn(batch_index.shape[0], 16)
    th.manual_seed(15)

    perturbed = encoder._perturb_global_message(message, batch_index, 120)

    # Each entry is a value of its own feature column, and about half the entries moved.
    assert (perturbed[:, None, :] == message[None, :, :]).any(dim=1).all()
    changed = float((perturbed != message).float().mean())
    assert 0.45 < changed < 0.55
    # Entries are drawn independently, so a changed row mixes several donors.
    assert ((perturbed != message).sum(dim=1) > 1).float().mean() > 0.9


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"global_message_substitution": 1.0}, "must be in"),
        ({"global_message_substitution": -0.1}, "must be in"),
        ({"global_message_noise_std": -0.1}, "must be nonnegative"),
        ({"global_message_swap": 1.0}, "must be in"),
        ({"global_message_swap": -0.1}, "must be in"),
        ({"global_message_swap_unit": "graph"}, "must be one of"),
    ],
)
def test_global_message_noise_rejects_invalid_settings(option: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _noisy_encoder(**option)


# Bond dropout (arm de1): a dropped bond leaves local message passing in both directions, and
# kept messages are scaled by 1/(1 - p). Global attention still sees every atom.


def test_drop_bonds_removes_both_directions_of_a_bond_together() -> None:
    encoder = _noisy_encoder(bond_dropout=0.5)
    # A 201-atom chain: forward edges first, reverse edges after.
    forward = th.stack([th.arange(200), th.arange(1, 201)])
    edge_index = th.cat([forward, forward.flip(0)], dim=1)
    edge_attr = th.arange(400, dtype=th.float32)[:, None]
    th.manual_seed(3)

    kept_index, kept_attr = encoder._drop_bonds(edge_index, edge_attr, 201)

    kept_pairs = set(map(tuple, kept_index.T.tolist()))
    assert all((dst, src) in kept_pairs for src, dst in kept_pairs)
    # Each kept edge keeps its own features.
    assert th.equal(edge_index[:, kept_attr[:, 0].long()], kept_index)
    assert 70 < len(kept_pairs) // 2 < 130


def test_scaled_source_scales_gine_messages_and_not_the_root_term() -> None:
    """The identity the encoder relies on: GINEConv((w x, x), w e) = x + w sum relu(x_j + e)."""
    th.manual_seed(4)
    conv = pyg.nn.conv.GINEConv(nn=th.nn.Identity(), eps=0.0, edge_dim=None)
    x = th.randn(4, 8, dtype=th.float64)
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 2]])
    edge_attr = th.randn(5, 8, dtype=th.float64)
    scale = 1.0 / (1.0 - 0.3)

    scaled = conv((x * scale, x), edge_index, edge_attr * scale)

    messages = th.relu(x[edge_index[0]] + edge_attr)
    expected = x + scale * th.zeros_like(x).index_add_(0, edge_index[1], messages)
    assert th.allclose(scaled, expected, atol=1e-12)


def test_bond_dropout_is_training_only_and_adds_no_parameters() -> None:
    plain, dropped = _noisy_encoder(), _noisy_encoder(bond_dropout=0.5)
    assert list(dropped.state_dict()) == list(plain.state_dict())
    dropped.load_state_dict(plain.state_dict(), strict=True)
    inputs = _encoder_inputs()

    assert th.equal(dropped.eval()(*inputs), plain.eval()(*inputs))
    th.manual_seed(5)
    assert not th.allclose(dropped.train()(*inputs), plain.train()(*inputs))


def test_bond_dropout_draws_once_per_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """One draw per pass, so every layer sees the same molecule with the same bonds missing."""
    encoder = _noisy_encoder(bond_dropout=0.5).train()
    calls = []
    original = encoder._drop_bonds
    monkeypatch.setattr(encoder, "_drop_bonds", lambda *args: calls.append(args) or original(*args))

    encoder(*_encoder_inputs())

    # Two layers, one draw.
    assert len(encoder.layers) == 2
    assert len(calls) == 1


@pytest.mark.parametrize("bond_dropout", [1.0, -0.1])
def test_bond_dropout_rejects_invalid_settings(bond_dropout: float) -> None:
    with pytest.raises(ValueError, match="bond_dropout must be in"):
        _noisy_encoder(bond_dropout=bond_dropout)


# ------------------------------------------------------------------ context token


def _reference_attention_with_slot(
    queries: th.Tensor,
    keys: th.Tensor,
    values: th.Tensor,
    batch_index: th.Tensor,
    extra_keys: th.Tensor,
    extra_values: th.Tensor,
) -> th.Tensor:
    """Per-graph attention over the graph's nodes plus that graph's one extra slot."""
    head_dim = queries.shape[-1]
    out = th.zeros_like(values)
    for graph in range(extra_keys.shape[0]):
        members = (batch_index == graph).nonzero().flatten()
        for head in range(queries.shape[1]):
            graph_keys = th.cat((keys[members, head], extra_keys[graph, head][None]))
            graph_values = th.cat((values[members, head], extra_values[graph, head][None]))
            scores = (queries[members, head] @ graph_keys.T) / math.sqrt(head_dim)
            out[members, head] = th.softmax(scores, dim=-1) @ graph_values
    return out


def test_graph_softmax_attn_context_slot_matches_definition() -> None:
    th.manual_seed(31)
    batch_index = th.tensor([0, 0, 0, 1, 1, 2])
    queries, keys, values = _random_qkv(6, 3, 8)
    extra_keys, extra_values = (th.randn(3, 3, 8, dtype=th.float64) for _ in range(2))

    attended = _graph_softmax_attn(queries, keys, values, batch_index, 3, extra_keys, extra_values)
    expected = _reference_attention_with_slot(
        queries, keys, values, batch_index, extra_keys, extra_values
    )

    assert attended.shape == values.shape
    assert th.allclose(attended, expected, atol=1e-12)
    # A singleton graph now splits its attention between itself and its slot.
    plain = _graph_softmax_attn(queries, keys, values, batch_index, 3)
    assert not th.allclose(attended[5], plain[5])


def test_graph_softmax_attn_context_slot_rejects_bad_inputs() -> None:
    queries, keys, values = _random_qkv(4, 2, 4)
    batch_index = th.tensor([0, 0, 1, 1])
    slot = th.randn(2, 2, 4, dtype=th.float64)
    with pytest.raises(ValueError, match="given together"):
        _graph_softmax_attn(queries, keys, values, batch_index, 2, slot, None)
    with pytest.raises(ValueError, match="must have shape"):
        _graph_softmax_attn(queries, keys, values, batch_index, 2, slot[:1], slot[:1])


def _context_encoder(context_token_dim: int) -> GINE_GSA:
    th.manual_seed(32)
    return GINE_GSA(
        hidden_size=16,
        node_feats_size=16,
        edge_feats_size=3,
        num_layers=2,
        dropout=0.0,
        normalization="layer",
        num_heads=4,
        context_token_dim=context_token_dim,
    ).eval()


def _two_molecule_graph() -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    th.manual_seed(33)
    x = th.randn(5, 16)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    return x, batch_index, edge_index, th.randn(edge_index.shape[1], 3)


def test_gine_gsa_context_token_adds_only_its_projections() -> None:
    """Every existing tensor initializes as without the token; only its projections are new."""
    plain = _context_encoder(0).state_dict()
    token = _context_encoder(6).state_dict()

    new_keys = sorted(set(token) - set(plain))
    assert new_keys == sorted(
        f"gsa_context_{kind}.{layer}.{param}"
        for kind in ("k", "v")
        for layer in (0, 1)
        for param in ("weight", "bias")
    )
    assert all(th.equal(token[key], plain[key]) for key in plain)
    assert token["gsa_context_k.0.weight"].shape == (16, 6)


def test_gine_gsa_context_token_reads_each_molecules_own_context() -> None:
    encoder = _context_encoder(6)
    inputs = _two_molecule_graph()
    th.manual_seed(34)
    context = th.randn(2, 6)

    baseline = encoder(*inputs, context=context)
    second_moved = context.clone()
    second_moved[1] += 3.0
    moved = encoder(*inputs, context=second_moved)
    first_moved = context.clone()
    first_moved[0] += 3.0

    assert baseline.shape == (5, 16)
    assert th.allclose(moved[:3], baseline[:3], atol=1e-6)
    assert not th.allclose(moved[3:], baseline[3:], atol=1e-4)
    assert not th.allclose(encoder(*inputs, context=first_moved)[:3], baseline[:3], atol=1e-4)


def test_gine_gsa_context_token_trains_its_projections() -> None:
    encoder = _context_encoder(6).train()
    context = th.randn(2, 6, requires_grad=True)
    encoder(*_two_molecule_graph(), context=context).square().sum().backward()

    assert context.grad is not None and context.grad.abs().sum() > 0
    for kind in ("k", "v"):
        for layer in (0, 1):
            grad = encoder.get_parameter(f"gsa_context_{kind}.{layer}.weight").grad
            assert grad is not None and th.isfinite(grad).all() and grad.abs().sum() > 0


def test_gine_gsa_context_token_requires_matching_context() -> None:
    inputs = _two_molecule_graph()
    with pytest.raises(ValueError, match="exactly when context_token_dim"):
        _context_encoder(6)(*inputs)
    with pytest.raises(ValueError, match="exactly when context_token_dim"):
        _context_encoder(0)(*inputs, context=th.randn(2, 6))
    with pytest.raises(ValueError, match="must have shape"):
        _context_encoder(6)(*inputs, context=th.randn(3, 6))
    with pytest.raises(ValueError, match="nonnegative"):
        _context_encoder(-1)
