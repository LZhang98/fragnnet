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
