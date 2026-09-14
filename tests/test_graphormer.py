"""Focused tests for the native PyG Graphormer molecular encoder."""

import pytest
import torch as th

from fragnnet.model.nn_blocks import GNN, GINEGraphormer, Graphormer


def _encoder() -> GNN:
    th.manual_seed(41)
    return GNN(
        hidden_size=16,
        num_layers=2,
        node_feats_size=5,
        edge_feats_size=3,
        gnn_type="Graphormer",
        dropout=0.0,
        normalization="layer",
        num_heads=4,
    )


def _hybrid_encoder() -> GNN:
    th.manual_seed(41)
    return GNN(
        hidden_size=16,
        num_layers=2,
        node_feats_size=5,
        edge_feats_size=3,
        gnn_type="GINE_GRAPHORMER",
        dropout=0.0,
        normalization="layer",
        num_heads=4,
    )


def _two_graph_batch() -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    generator = th.Generator().manual_seed(9)
    x = th.randn(5, 5, generator=generator)
    batch_index = th.tensor([0, 0, 0, 1, 1])
    edge_index = th.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]],
        dtype=th.long,
    )
    edge_attr = th.randn(6, 3, generator=generator)
    return x, batch_index, edge_index, edge_attr


def test_graphormer_returns_one_state_per_atom_and_backpropagates() -> None:
    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    x.requires_grad_()
    edge_attr.requires_grad_()
    model = _encoder()

    output = model(x, batch_index, edge_index, edge_attr)
    output.square().sum().backward()

    assert output.shape == (5, 16)
    assert th.isfinite(output).all()
    assert x.grad is not None and th.isfinite(x.grad).all()
    assert edge_attr.grad is not None and edge_attr.grad.abs().sum() > 0


def test_graphormer_supports_cpu_bfloat16_autocast() -> None:
    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    x.requires_grad_()
    edge_attr.requires_grad_()
    model = _encoder()

    with th.autocast(device_type="cpu", dtype=th.bfloat16):
        output = model(x, batch_index, edge_index, edge_attr)
        loss = output.square().sum()
    loss.backward()

    assert output.shape == (5, 16)
    assert th.isfinite(output).all()
    assert x.grad is not None and th.isfinite(x.grad).all()
    assert edge_attr.grad is not None and th.isfinite(edge_attr.grad).all()


def test_graphormer_batched_graphs_match_independent_encoding() -> None:
    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    model = _encoder().eval()

    with th.no_grad():
        batched = model(x, batch_index, edge_index, edge_attr)
        first = model(x[:3], th.zeros(3, dtype=th.long), edge_index[:, :4], edge_attr[:4])
        second = model(
            x[3:],
            th.zeros(2, dtype=th.long),
            edge_index[:, 4:] - 3,
            edge_attr[4:],
        )

    assert th.allclose(batched[:3], first, atol=1e-6)
    assert th.allclose(batched[3:], second, atol=1e-6)


def test_graphormer_encodes_bonds_along_multi_hop_shortest_paths() -> None:
    model = Graphormer(
        hidden_size=8,
        edge_feats_size=1,
        num_layers=1,
        dropout=0.0,
        normalization="layer",
        num_heads=2,
    )
    with th.no_grad():
        model.spatial_embedding.weight.zero_()
        model.edge_bias.weight.fill_(1.0)
        model.edge_distance_mixing.copy_(
            th.eye(2).unsqueeze(0).expand(model.multi_hop_max_distance, -1, -1)
        )
    batch_index = th.zeros(3, dtype=th.long)
    edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = th.tensor([[1.0], [1.0], [3.0], [3.0]])

    bias = model._structural_attention_bias(batch_index, edge_index, edge_attr, max_nodes=3)

    assert bias[0, :, 0, 2].tolist() == pytest.approx([2.0, 2.0])


def test_gine_graphormer_runs_local_and_global_branches() -> None:
    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    x.requires_grad_()
    edge_attr.requires_grad_()
    model = _hybrid_encoder()

    output = model(x, batch_index, edge_index, edge_attr)
    output.square().sum().backward()

    assert output.shape == (5, 16)
    assert th.isfinite(output).all()
    assert isinstance(model.gnn, GINEGraphormer)
    assert model.gnn.local_edge_transform.weight.grad is not None
    assert model.gnn.local_edge_transform.weight.grad.abs().sum() > 0
    assert model.gnn.edge_bias.weight.grad is not None
    assert model.gnn.edge_bias.weight.grad.abs().sum() > 0


def test_gine_graphormer_batched_graphs_match_independent_encoding() -> None:
    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    model = _hybrid_encoder().eval()

    with th.no_grad():
        batched = model(x, batch_index, edge_index, edge_attr)
        first = model(x[:3], th.zeros(3, dtype=th.long), edge_index[:, :4], edge_attr[:4])
        second = model(
            x[3:],
            th.zeros(2, dtype=th.long),
            edge_index[:, 4:] - 3,
            edge_attr[4:],
        )

    assert th.allclose(batched[:3], first, atol=1e-6)
    assert th.allclose(batched[3:], second, atol=1e-6)


def test_graphormer_rejects_non_layer_normalization() -> None:
    with pytest.raises(ValueError, match="mol_normalization: layer"):
        Graphormer(
            hidden_size=16,
            edge_feats_size=3,
            num_layers=2,
            dropout=0.0,
            normalization="batch",
            num_heads=4,
        )

    with pytest.raises(ValueError, match="mol_normalization: layer"):
        GINEGraphormer(
            hidden_size=16,
            edge_feats_size=3,
            num_layers=2,
            dropout=0.0,
            normalization="batch",
            num_heads=4,
        )


def test_graphormer_structural_sizes_are_configurable() -> None:
    model = GNN(
        hidden_size=16,
        num_layers=1,
        node_feats_size=5,
        edge_feats_size=3,
        gnn_type="Graphormer",
        dropout=0.0,
        normalization="layer",
        num_heads=4,
        max_degree=8,
        max_spatial_distance=16,
        multi_hop_max_distance=3,
    )

    assert model.gnn.in_degree_embedding.weight.shape == (9, 16)
    assert model.gnn.out_degree_embedding.weight.shape == (9, 16)
    assert model.gnn.spatial_embedding.weight.shape == (18, 4)
    assert model.gnn.edge_distance_mixing.shape == (3, 4, 4)

    x, batch_index, edge_index, edge_attr = _two_graph_batch()
    assert model(x, batch_index, edge_index, edge_attr).shape == (5, 16)


def test_graphormer_structural_sizes_default_when_unset() -> None:
    """``None`` must fall through to the class defaults, not crash the embeddings."""
    model = GNN(
        hidden_size=16,
        num_layers=1,
        node_feats_size=5,
        edge_feats_size=3,
        gnn_type="GINE_GRAPHORMER",
        dropout=0.0,
        normalization="layer",
        num_heads=4,
        max_degree=None,
        max_spatial_distance=None,
        multi_hop_max_distance=None,
    )

    assert (model.gnn.max_degree, model.gnn.max_spatial_distance) == (64, 128)
    assert model.gnn.multi_hop_max_distance == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_degree": 0},
        {"max_spatial_distance": 0},
        {"multi_hop_max_distance": 0},
    ],
)
def test_graphormer_rejects_degenerate_structural_sizes(overrides: dict) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        Graphormer(
            hidden_size=16,
            edge_feats_size=3,
            num_layers=1,
            dropout=0.0,
            normalization="layer",
            num_heads=4,
            **overrides,
        )


def test_gine_graphormer_keeps_bond_features_on_scale_with_atom_states() -> None:
    """The local branch must keep its fan-in-aware init, not Graphormer's flat std=0.02.

    ``GINEConv`` messages are ``relu(x_j + local_edge_transform(edge_attr))``, and ``x_j``
    comes from ``GNN.input_project`` outside this module. Initializing the edge projection
    at std=0.02 -- tuned for fan_in == hidden_size, not for a handful of bond features --
    drops its output roughly 8x below the atom states it is added to.
    """
    th.manual_seed(0)
    hidden_size, edge_feats_size, node_feats_size = 384, 13, 40
    model = GINEGraphormer(
        hidden_size=hidden_size,
        edge_feats_size=edge_feats_size,
        num_layers=2,
        dropout=0.0,
        normalization="layer",
        num_heads=8,
    )
    input_project = th.nn.Linear(node_feats_size, hidden_size)

    with th.no_grad():
        atom_scale = input_project(th.randn(4096, node_feats_size)).std()
        bond_scale = model.local_edge_transform(th.randn(4096, edge_feats_size)).std()

    assert 0.5 < (bond_scale / atom_scale).item() < 2.0

    # the global attention stack does keep the Graphormer init
    assert model.layers[0].qkv.weight.std().item() == pytest.approx(0.02, abs=0.005)


def test_graphormer_edge_bias_accumulates_every_head_independently() -> None:
    """The direct-bond scatter fills all heads at once; heads must not be mixed up."""
    model = Graphormer(
        hidden_size=8,
        edge_feats_size=1,
        num_layers=1,
        dropout=0.0,
        normalization="layer",
        num_heads=2,
        multi_hop_max_distance=1,
    )
    with th.no_grad():
        model.spatial_embedding.weight.zero_()
        # head 0 reads the bond feature, head 1 reads it with the opposite sign and 10x gain
        model.edge_bias.weight.copy_(th.tensor([[1.0], [-10.0]]))
        model.edge_distance_mixing.copy_(th.eye(2).unsqueeze(0))
    batch_index = th.zeros(2, dtype=th.long)
    edge_index = th.tensor([[0, 1], [1, 0]])
    edge_attr = th.tensor([[2.0], [2.0]])

    bias = model._structural_attention_bias(batch_index, edge_index, edge_attr, max_nodes=2)

    assert bias[0, :, 0, 1].tolist() == pytest.approx([2.0, -20.0])
    assert bias[0, :, 1, 0].tolist() == pytest.approx([2.0, -20.0])


def test_graphormer_saturates_far_distances_instead_of_marking_them_unreachable() -> None:
    """A small ``max_spatial_distance`` shortens the table; it must not sever the graph.

    Reachable pairs beyond the limit belong in the last real bucket, the way ``max_degree``
    clamps. Folding them into the disconnected bucket would tell the model that distant
    atoms of one molecule are as unrelated as atoms of two different molecules -- and would
    also drop them from the multi-hop bond-path bias.
    """
    max_spatial_distance = 3
    model = Graphormer(
        hidden_size=8,
        edge_feats_size=1,
        num_layers=1,
        dropout=0.0,
        normalization="layer",
        num_heads=2,
        max_spatial_distance=max_spatial_distance,
    )
    # one 10-atom chain (true distances 0..9) beside one disconnected pair
    num_chain_atoms = 10
    batch_index = th.cat((th.zeros(num_chain_atoms, dtype=th.long), th.ones(2, dtype=th.long)))
    bonds = []
    for atom in range(num_chain_atoms - 1):
        bonds += [[atom, atom + 1], [atom + 1, atom]]
    edge_index = th.tensor(bonds).T
    edge_attr = th.ones(edge_index.shape[1], 1)

    distances, _, _ = model._shortest_path_distances(
        batch_index, edge_index, max_nodes=num_chain_atoms
    )

    # true distances survive, so the disconnected bucket stays exclusive to disconnection
    assert distances[0, 0].tolist() == list(range(num_chain_atoms))
    unreachable = model.unreachable_distance(num_chain_atoms)
    assert distances[1, 0, 1].item() == unreachable
    assert distances[0, 0].max().item() < unreachable

    # far-but-connected atoms keep a real bond-path bias
    bias = model._structural_attention_bias(
        batch_index, edge_index, edge_attr, max_nodes=num_chain_atoms
    )
    near = bias[0, :, 0, max_spatial_distance]
    far = bias[0, :, 0, num_chain_atoms - 1]
    disconnected = bias[1, :, 0, 1]
    assert th.isfinite(far).all()
    assert not th.allclose(far, disconnected)
    # both sit in the saturated spatial bucket, so they differ only via the path encoding
    assert not th.allclose(far, near)
