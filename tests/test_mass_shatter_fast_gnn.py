"""Connected-fragment search, probability semantics, and MASShatter integration."""

from itertools import combinations

import pytest
import torch as th
import torch.nn.functional as F
from test_mass_shatter_model import _build_batch, _build_model

from fragnnet.frag.mass_shatter_fast_gnn import build_fast_gnn_candidates


def _graph(n, edges, logits=None, device="cpu", **kwargs):
    edges = th.tensor(edges, dtype=th.long, device=device).reshape(-1, 2).T.contiguous()
    if logits is None:
        logits = th.linspace(-0.8, 0.6, edges.shape[1], device=device, requires_grad=True)
    return build_fast_gnn_candidates(
        logits,
        edges,
        th.zeros(n, dtype=th.long, device=device),
        th.tensor([0, n], device=device),
        max_fragments=kwargs.pop("max_fragments", 128),
        beam_width=kwargs.pop("beam_width", 128),
        max_boundary_cuts=kwargs.pop("max_boundary_cuts", 3),
        **kwargs,
    ), logits


def _sets(candidates):
    return [
        frozenset(candidates.atom_indices[candidates.atom_component_indices == i].tolist())
        for i in range(candidates.num_components)
    ]


def _connected(nodes, edges):
    reached = {next(iter(nodes))}
    while True:
        updated = reached | {
            v for u, v in edges + [(v, u) for u, v in edges] if u in reached and v in nodes
        }
        if updated == reached:
            return reached == set(nodes)
        reached = updated


@pytest.mark.parametrize("edges", [[(0, 1), (1, 2)], [(0, 1), (1, 2), (2, 0)]])
def test_complete_small_graph_search_and_live_boundary_score(edges):
    candidates, logits = _graph(3, edges)
    fragments = _sets(candidates)
    expected = {
        frozenset(nodes)
        for k in range(1, 4)
        for nodes in combinations(range(3), k)
        if _connected(nodes, edges)
    }
    assert set(fragments) == expected
    assert len(fragments) == len(expected)
    for i, fragment in enumerate(fragments):
        boundary = th.tensor([(u in fragment) != (v in fragment) for u, v in edges])
        expected_score = th.where(boundary, F.logsigmoid(logits), F.logsigmoid(-logits)).sum()
        th.testing.assert_close(candidates.mask_log_probabilities[i], expected_score)
        assert candidates.mask_cut_counts[i] == boundary.sum()
    # A normalized boundary distribution has the same gradient as summed boundary logits.
    actual = candidates.mask_log_probabilities.log_softmax(0)
    direct = th.stack(
        [logits[th.tensor([(u in f) != (v in f) for u, v in edges])].sum() for f in fragments]
    ).log_softmax(0)
    th.testing.assert_close(actual, direct)
    grad = th.autograd.grad(actual[0], logits, retain_graph=True)[0]
    th.testing.assert_close(grad, th.autograd.grad(direct[0], logits)[0])


@pytest.mark.parametrize("n,edges", [(1, []), (3, []), (4, [(0, 1), (2, 3)])])
def test_zero_bond_and_disconnected_precursor(n, edges):
    candidates, _ = _graph(n, edges)
    fragments = _sets(candidates)
    assert frozenset(range(n)) in fragments
    assert all(f == frozenset(range(n)) or _connected(f, edges) for f in fragments)
    assert th.isfinite(candidates.mask_log_probabilities).all()


def test_budget_and_boundary_limit():
    candidates, _ = _graph(
        5, [(0, 1), (1, 2), (2, 3), (3, 4)], max_fragments=3, beam_width=2, max_boundary_cuts=1
    )
    assert candidates.num_components <= 3
    assert frozenset(range(5)) in _sets(candidates)
    assert (candidates.mask_cut_counts <= 1).all()
    precursor, _ = _graph(3, [(0, 1), (1, 2)], max_fragments=1)
    assert _sets(precursor) == [frozenset(range(3))]
    zero, _ = _graph(3, [(0, 1), (1, 2)], max_boundary_cuts=0)
    assert _sets(zero) == [frozenset(range(3))]


def test_multiword_memberships_and_deterministic_ties():
    # Zero-bond atoms exercise bit 62, 63 and 64 without a long beam expansion.
    first, _ = _graph(65, [], max_fragments=70, beam_width=70)
    second, _ = _graph(65, [], max_fragments=70, beam_width=70)
    assert _sets(first) == _sets(second)
    assert len(_sets(first)) == 66
    assert frozenset({64}) in _sets(first)
    assert frozenset(range(65)) in _sets(first)


def test_batched_unsorted_bonds_map_back_to_original_ids():
    logits = th.tensor([0.7, -0.2, 0.1], requires_grad=True)
    edges = th.tensor([[3, 0, 1], [4, 1, 2]])
    candidates = build_fast_gnn_candidates(
        logits,
        edges,
        th.tensor([0, 0, 0, 1, 1]),
        th.tensor([0, 3, 5]),
        max_fragments=20,
        beam_width=20,
        max_boundary_cuts=3,
    )
    assert candidates.unique_component_count.tolist() == [6, 3]
    assert candidates.component_batch_indices.tolist() == [0] * 6 + [1] * 3
    for i, f in enumerate(_sets(candidates)):
        actual = candidates.boundary_bond_indices[candidates.boundary_component_indices == i]
        expected = {j for j, (u, v) in enumerate(edges.T.tolist()) if (u in f) != (v in f)}
        assert set(actual.tolist()) == expected


@pytest.mark.parametrize("selection_only", [False, True])
def test_model_forward_probability_and_proposal_gradients(selection_only):
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        max_masks=12,
        proposal_selection_only=selection_only,
        use_base_probability_feature=not selection_only,
        proposal_type="fast_gnn",
    )
    output = model(
        mol_pyg=batch, spec_prec_type=th.tensor([0, 1]), spec_nce=th.tensor([20.0, 30.0])
    )
    for i in range(2):
        probs = output["pred_logprobs"][output["pred_batch_idxs"] == i].exp()
        th.testing.assert_close(probs.sum(), th.tensor(1.0))
    assert th.isfinite(output["pred_logprobs"]).all()
    if selection_only:
        loss = -output["proposal_fragment_logprobs"][0]
    else:
        loss = -output["pred_logprobs"][0]
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.bond_break_head.parameters()
    )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_memberships_and_gradients():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    cpu, cpu_logits = _graph(4, edges)
    gpu, gpu_logits = _graph(4, edges, device="cuda")
    assert _sets(cpu) == _sets(gpu)
    th.testing.assert_close(cpu.mask_log_probabilities, gpu.mask_log_probabilities.cpu())
    cpu.mask_log_probabilities.log_softmax(0)[0].backward()
    gpu.mask_log_probabilities.log_softmax(0)[0].backward()
    th.testing.assert_close(cpu_logits.grad, gpu_logits.grad.cpu())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mass_shatter_proposal_type": "unknown"}, "unknown proposal_type"),
        ({"mass_shatter_unique_fragment_mode": False}, "fast_gnn requires"),
        ({"mass_shatter_proposal_num_mixtures": 2}, "fast_gnn requires"),
        ({"mass_shatter_smarts_templates": True}, "fast_gnn requires"),
        ({"mass_shatter_share_proposal_across_ev": True}, "fast_gnn requires"),
        ({"mass_shatter_min_mask_relative_probability": 0.1}, "fast_gnn requires"),
        ({"mass_shatter_fast_gnn_beam_width": 0}, "beam width"),
        ({"mass_shatter_fast_gnn_max_boundary_cuts": -1}, "boundary cuts"),
        ({"mass_shatter_fast_gnn_beam_width": 1.5}, "integers"),
    ],
)
def test_invalid_configuration(overrides, match):
    from test_mass_shatter_pl import _build_pl

    settings = {"mass_shatter_proposal_type": "fast_gnn", "mass_shatter_unique_fragment_mode": True}
    settings.update(overrides)
    with pytest.raises(ValueError, match=match):
        _build_pl(**settings)


def _small_pl_config():
    from pathlib import Path

    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "config/template.yml").read_text())
    config.update(
        model_type="mass_shatter",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        loss_vectorized=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        mass_shatter_context_dim=8,
        mass_shatter_metadata_embedding_dim=4,
        mass_shatter_head_hidden_dim=16,
        mass_shatter_intensity_transformer_dim=16,
        mass_shatter_intensity_transformer_num_layers=1,
        mass_shatter_intensity_transformer_feedforward_dim=32,
        mass_shatter_intensity_transformer_num_inducing_points=4,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_unique_fragment_target=12,
        mass_shatter_ranked_mask_budget=12,
        mass_shatter_max_h_transfer=1,
        mass_shatter_fast_gnn_beam_width=16,
        mass_shatter_proposal_type="fast_gnn",
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_log_base_ce=False,
    )
    return config


@pytest.mark.parametrize(
    "output_mode,loss_type", [("probability", "cross_entropy"), ("independent", "jss_distance")]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
def test_lightning_loss_and_staged_training(output_mode, loss_type, device):
    from rdkit import Chem
    from torch_geometric.data import Batch

    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.utils.feat_utils import get_mol_graph

    config = _small_pl_config()
    config.update(
        mass_shatter_intensity_output=output_mode,
        loss_type=loss_type,
        mass_shatter_proposal_only_until_epoch=0,
    )
    pl = MassShatterPL(**config).to(device)
    params = config["mol_params"]
    graphs = [
        get_mol_graph(
            Chem.MolFromSmiles(smiles),
            params["pyg_node_feats"],
            params["pyg_edge_feats"],
            params["pyg_pe_embed_k"],
            True,
        )
        for smiles in ("CCO", "CCN")
    ]
    batch = Batch.from_data_list(graphs).to(device)
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0], device=device),
        "spec_inst_type": th.tensor([0, 0], device=device),
        "spec_nce": th.tensor([20.0, 30.0], device=device),
    }
    for epoch in (0, 1):
        pl.zero_grad(set_to_none=True)
        pl._apply_staged_training(epoch)
        output = pl.model(**inputs)
        targets = th.stack(
            [output["pred_mzs"][output["pred_batch_idxs"] == i][0] for i in range(2)]
        ).detach()
        losses = pl.loss_fn(
            **output,
            true_mzs=targets.float(),
            true_logprobs=th.zeros(2, device=device),
            true_batch_idxs=th.arange(2, device=device),
            batch_size=2,
        )
        assert set(pl.loss_names) <= losses.keys()
        assert "retained_mask_probability" not in losses
        assert "fast_gnn_fragment_count" in losses
        assert th.isfinite(losses["loss"]).all()
        losses["loss"].sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pl.model.bond_break_head.parameters()
        )
        intensity_grad = sum(
            p.grad.abs().sum().item()
            for p in pl.model.intensity_head.parameters()
            if p.grad is not None
        )
        assert (intensity_grad > 0) == (epoch == 1)
        assert all(
            not p.requires_grad and p.grad is None for p in pl.model.charge_head.parameters()
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_proposal_roundtrip_and_legacy_migration(tmp_path, legacy):
    import lightning as L

    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.utils.pl_utils import get_pl_hparams

    config = _small_pl_config()
    if legacy:
        config["mass_shatter_proposal_type"] = "cut_masks"
    model = MassShatterPL(**config)
    checkpoint = {
        "hyper_parameters": dict(model.hparams),
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": L.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    keys = (
        "mass_shatter_proposal_type",
        "mass_shatter_fast_gnn_beam_width",
        "mass_shatter_fast_gnn_max_boundary_cuts",
    )
    if legacy:
        for key in keys:
            del checkpoint["hyper_parameters"][key]
    migrated = get_pl_hparams(checkpoint)
    assert migrated["mass_shatter_proposal_type"] == config["mass_shatter_proposal_type"]
    path = tmp_path / "model.ckpt"
    th.save(checkpoint, path)
    restored = MassShatterPL.load_from_checkpoint(path, map_location="cpu", strict=True)
    assert restored.model.proposal_type == config["mass_shatter_proposal_type"]
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        th.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    for key in keys:
        incomplete = dict(config)
        del incomplete[key]
        with pytest.raises(AttributeError, match=key):
            MassShatterPL(**incomplete)


def test_config_arm_resolves_and_constructs():
    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.runner import load_config

    config = load_config(
        "config/template.yml",
        "config/masshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml",
        None,
    )
    model = MassShatterPL(**config)
    assert model.model.proposal_type == "fast_gnn"
    assert model.model.fast_gnn_beam_width == 256


def test_batch_padding_does_not_change_proposals():
    short_edges = [(0, 1), (1, 2), (2, 3)]
    single, _ = _graph(
        4, short_edges, logits=th.tensor([-0.4, 0.2, 0.8]), beam_width=2, max_fragments=8
    )
    long_edges = [(i, i + 1) for i in range(4, 14)]
    edges = th.tensor(short_edges + long_edges).T
    batched = build_fast_gnn_candidates(
        th.cat([th.tensor([-0.4, 0.2, 0.8]), th.zeros(10)]),
        edges,
        th.tensor([0] * 4 + [1] * 11),
        th.tensor([0, 4, 15]),
        max_fragments=8,
        beam_width=2,
        max_boundary_cuts=3,
    )
    assert set(_sets(single)) == {
        f for f, b in zip(_sets(batched), batched.component_batch_indices) if b == 0
    }


def test_connected_multiword_complements():
    edges = [(i, i + 1) for i in range(64)]
    candidates, _ = _graph(65, edges, beam_width=2, max_fragments=12, max_boundary_cuts=1)
    fragments = _sets(candidates)
    assert any(len(f) == 64 for f in fragments)
    assert all(_connected(f, edges) for f in fragments)
    assert (candidates.mask_cut_counts <= 1).all()


@pytest.mark.parametrize(
    "edges,match",
    [
        ([(0, 1), (1, 0)], "unique chemical bonds"),
        ([(0, 0)], "distinct atoms"),
        ([(0, 4)], "out of range"),
    ],
)
def test_invalid_graph_rejected(edges, match):
    with pytest.raises(ValueError, match=match):
        _graph(3, edges)
