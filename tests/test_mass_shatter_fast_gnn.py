"""Connected-fragment search, probability semantics, and MASShatter integration."""

from itertools import combinations

import pytest
import torch as th
import torch.nn.functional as F
from test_mass_shatter_model import _build_batch, _build_model

from fragnnet.frag.mass_shatter_fast_gnn import build_fast_gnn_candidates
from fragnnet.runner import load_config


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
    references = []
    for i, fragment in enumerate(fragments):
        boundary = th.tensor([(u in fragment) != (v in fragment) for u, v in edges])
        internal = th.tensor([(u in fragment) and (v in fragment) for u, v in edges])
        expected_score = F.logsigmoid(logits[boundary]).sum() + F.logsigmoid(-logits[internal]).sum()
        references.append(expected_score)
        th.testing.assert_close(candidates.mask_log_probabilities[i], expected_score)
        assert candidates.mask_cut_counts[i] == boundary.sum()
    # Normalized scores and gradients agree with an independent edge-membership reference.
    actual = candidates.mask_log_probabilities.log_softmax(0)
    direct = th.stack(references).log_softmax(0)
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
        proposal_method="connected_fragment_beam",
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
        ({"mass_shatter_proposal_method": "unknown"}, "unknown proposal_method"),
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

    settings = {
        "mass_shatter_proposal_method": "connected_fragment_beam",
        "mass_shatter_unique_fragment_mode": True,
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=match):
        _build_pl(**settings)


def _small_pl_config():
    from pathlib import Path

    config = load_config(
        str(Path(__file__).parents[1] / "config"),
        None,
        None,
        model_type="mass_shatter",
    )
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
        mass_shatter_proposal_method="connected_fragment_beam",
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


def test_candidate_coverage_metrics_survive_the_fast_gnn_metric_rename():
    """Beam arms log candidate_wrecall/candidate_recall: the coverage ceiling under test."""
    from rdkit import Chem
    from torch_geometric.data import Batch

    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.utils.feat_utils import get_mol_graph

    config = _small_pl_config()
    config.update(
        mass_shatter_proposal_only_until_epoch=0,
        mass_shatter_candidate_coverage_metrics=True,
        mass_shatter_candidate_cut_diagnostics=True,
    )
    pl = MassShatterPL(**config)
    # The connected_fragment_beam rename drops the mask/pathway diagnostics; candidate
    # coverage is not a mask diagnostic and must pass through both filters intact.
    assert "candidate_wrecall" in pl.loss_names
    assert "retained_mask_probability" not in pl.loss_names
    assert {"candidate_recall", "candidate_wrecall_cut_1"} <= pl.metric_names
    assert "candidate_mzs_by_cut" in pl.extra_metric_args

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
    batch = Batch.from_data_list(graphs)
    pl._apply_staged_training(1)
    output = pl.model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_inst_type=th.tensor([0, 0]),
        spec_nce=th.tensor([20.0, 30.0]),
    )
    assert output["candidate_cut_counts"].shape == output["candidate_mzs_by_cut"].shape
    # Target one beam candidate per spectrum, so both recalls must be exactly 1.
    targets = th.stack(
        [
            output["candidate_mzs_by_cut"][output["candidate_batch_idxs_by_cut"] == i][0]
            for i in range(2)
        ]
    ).detach()

    losses = pl.loss_fn(
        **output,
        true_mzs=targets,
        true_ints=th.ones(2),
        true_logprobs=th.zeros(2),
        true_batch_idxs=th.arange(2),
        batch_size=2,
    )
    assert set(pl.loss_names) <= losses.keys()
    assert losses["candidate_wrecall"].tolist() == pytest.approx([1.0, 1.0])

    metrics = pl.metric_fn(
        true_mzs=targets,
        true_ints=th.ones(2),
        true_batch_idxs=th.arange(2),
        pred_mzs=output["pred_mzs"].detach(),
        pred_ints=output["pred_logprobs"].detach().exp(),
        pred_batch_idxs=output["pred_batch_idxs"],
        candidate_mzs_by_cut=output["candidate_mzs_by_cut"],
        candidate_batch_idxs_by_cut=output["candidate_batch_idxs_by_cut"],
        candidate_cut_counts=output["candidate_cut_counts"],
        true_prec_mzs=th.tensor([50.0, 50.0]),
        split="val",
    )
    assert metrics["candidate_recall"].tolist() == pytest.approx([1.0, 1.0])
    # The loss path owns candidate_wrecall; returning it here would collide on merge.
    assert "candidate_wrecall" not in metrics


@pytest.mark.parametrize(
    "config_path",
    [
        "config/massshatter/merged/unique_fragment/nist20_ma_mi_merged_fast_gnn_proposal.yml",
        "config/massshatter/merged/unique_fragment/"
        "nist20_ma_mi_merged_fast_gnn_proposal_fragments128.yml",
        "config/massshatter/merged/unique_fragment/"
        "nist20_ma_mi_merged_fast_gnn_proposal_fragments512.yml",
        "config/massshatter/ma_mi_unmerged/cmf/"
        "nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml",
        "config/massshatter/merged/warmup_duration/"
        "nist20_merged_independent_jss_warmup_e0_4_dense_subformula_fast_gnn_proposal.yml",
    ],
)
def test_prepared_beam_arms_log_candidate_coverage(config_path):
    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    config = load_config("config", config_path, None)
    assert config["mass_shatter_candidate_coverage_metrics"] is True
    model = MassShatterPL(**config)
    assert model.model.proposal_method == "connected_fragment_beam"
    assert "candidate_wrecall" in model.loss_names
    assert "candidate_recall" in model.metric_names


def test_merged_jss_beam_arm_matches_its_baseline_apart_from_the_proposal() -> None:
    """The beam arm exists to be comparable to the lead merged run, so pin the diff."""
    import yaml

    directory = "config/massshatter/merged/warmup_duration"
    stem = "nist20_merged_independent_jss_warmup_e0_4_dense_subformula"
    with open(f"{directory}/{stem}_fast.yml") as handle:
        baseline = yaml.safe_load(handle)
    with open(f"{directory}/{stem}_fast_gnn_proposal.yml") as handle:
        arm = yaml.safe_load(handle)

    absent = object()
    changed = {
        key for key in set(baseline) | set(arm) if baseline.get(key, absent) != arm.get(key, absent)
    }
    assert changed == {
        "wandb_name",
        "wandb_tags",
        "mass_shatter_proposal_method",
        "mass_shatter_fast_gnn_beam_width",
        "mass_shatter_fast_gnn_max_boundary_cuts",
        "mass_shatter_candidate_coverage_metrics",
        # cut_masks-only, and dropped rather than left to imply a device candidate path.
        "mass_shatter_candidate_backend",
    }
    # The comparison is only meaningful on the lead's split, not the MA/MI one.
    assert arm["split_dp"] == "data/split/nist20_inchikey"
    assert arm["loss_type"] == "jss_distance"
    assert arm["mass_shatter_fast_gnn_beam_width"] == arm["mass_shatter_unique_fragment_target"]
    assert "mass_shatter_candidate_backend" not in arm


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_proposal_roundtrip_and_legacy_migration(tmp_path, legacy):
    import lightning as L

    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.utils.pl_utils import get_pl_hparams

    config = _small_pl_config()
    if legacy:
        config["mass_shatter_proposal_method"] = "cut_masks"
    model = MassShatterPL(**config)
    checkpoint = {
        "hyper_parameters": dict(model.hparams),
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": L.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    keys = (
        "mass_shatter_proposal_method",
        "mass_shatter_fast_gnn_beam_width",
        "mass_shatter_fast_gnn_max_boundary_cuts",
    )
    if legacy:
        for key in keys:
            del checkpoint["hyper_parameters"][key]
    migrated = get_pl_hparams(checkpoint)
    assert migrated["mass_shatter_proposal_method"] == config["mass_shatter_proposal_method"]
    path = tmp_path / "model.ckpt"
    th.save(checkpoint, path)
    restored = MassShatterPL.load_from_checkpoint(path, map_location="cpu", strict=True)
    assert restored.model.proposal_method == config["mass_shatter_proposal_method"]
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
    config = load_config(
        "config",
        "config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml",
        None,
    )
    model = MassShatterPL(**config)
    assert model.model.proposal_method == "connected_fragment_beam"
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


@pytest.mark.parametrize("padding_atoms", [6, 11, 65])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tied_beam_is_independent_of_batch_padding(padding_atoms, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA unavailable")
    edges = [(i, (i + 1) % 5) for i in range(5)]
    single, _ = _graph(
        5, edges, logits=th.zeros(5, device=device), device=device,
        beam_width=1, max_fragments=8,
    )
    padded_edges = edges + [(i, i + 1) for i in range(5, 5 + padding_atoms - 1)]
    batched = build_fast_gnn_candidates(
        th.zeros(len(padded_edges), device=device),
        th.tensor(padded_edges, device=device).T.contiguous(),
        th.tensor([0] * 5 + [1] * padding_atoms, device=device),
        th.tensor([0, 5, 5 + padding_atoms], device=device),
        max_fragments=8, beam_width=1, max_boundary_cuts=3,
    )
    assert set(_sets(single)) == {
        f for f, b in zip(_sets(batched), batched.component_batch_indices) if b == 0
    }


@pytest.mark.parametrize("first_logit", [2.0, 20.0])
def test_complement_is_scored_before_output_budget_pruning(first_logit):
    # The weak seed {0, 1} must survive long enough to emit its strong complement.
    # 2.0 catches global pruning; 20.0 also catches pruning within a growth round.
    candidates, _ = _graph(
        6, [(i, i + 1) for i in range(5)],
        logits=th.tensor([first_logit, 6.0, -2.0, -4.0, -1.5]),
        beam_width=128, max_fragments=4,
    )
    assert set(_sets(candidates)) == {
        frozenset({0}), frozenset({1}), frozenset({2, 3, 4, 5}), frozenset(range(6)),
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_fragments", [2, 4])
@pytest.mark.parametrize("edges", [
    [(i, i + 1) for i in range(5)],
    [(i, (i + 1) % 6) for i in range(6)],
    [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)],
])
def test_budget_matches_exhaustive_reachable_fragment_scores(seed, max_fragments, edges):
    logits = 3 * th.randn(len(edges), generator=th.Generator().manual_seed(seed))
    references = []
    for size in range(1, 6):
        for nodes in combinations(range(6), size):
            fragment = frozenset(nodes)
            complement = frozenset(range(6)) - fragment
            # Large fragments are reachable only through a connected small complement.
            if not _connected(fragment, edges) or (size > 3 and not _connected(complement, edges)):
                continue
            boundary = th.tensor([(u in fragment) != (v in fragment) for u, v in edges])
            if boundary.sum() > 3:
                continue
            internal = th.tensor([(u in fragment) and (v in fragment) for u, v in edges])
            score = F.logsigmoid(logits[boundary]).sum() + F.logsigmoid(-logits[internal]).sum()
            references.append((float(score), fragment))
    ranked = sorted(references, key=lambda entry: entry[0], reverse=True)
    expected = {f for _, f in ranked[:max_fragments - 1]} | {frozenset(range(6))}
    candidates, _ = _graph(
        6, edges, logits=logits, beam_width=128, max_fragments=max_fragments,
    )
    assert set(_sets(candidates)) == expected


def test_large_growth_round_preserves_batched_memberships_and_budget():
    # A wide beam on K11 visits >256 unique seeds in a round, spanning workspace chunks.
    edges = list(combinations(range(11), 2)) + [(11, 12), (12, 13)]
    candidates = build_fast_gnn_candidates(
        -th.ones(len(edges)), th.tensor(edges).T.contiguous(),
        th.tensor([0] * 11 + [1] * 3), th.tensor([0, 11, 14]),
        max_fragments=5, beam_width=512, max_boundary_cuts=30,
    )
    fragments = _sets(candidates)
    assert {
        f for f, b in zip(fragments, candidates.component_batch_indices) if b == 0
    } == {frozenset({i}) for i in range(4)} | {frozenset(range(11))}
    chain, _ = _graph(3, [(0, 1), (1, 2)], logits=-th.ones(2), max_fragments=5, beam_width=512)
    assert {
        frozenset(a - 11 for a in f)
        for f, b in zip(fragments, candidates.component_batch_indices) if b == 1
    } == set(_sets(chain))
    assert candidates.unique_component_count.tolist() == [5, 5]


def test_connected_multiword_complements():
    edges = [(i, i + 1) for i in range(64)]
    candidates, _ = _graph(65, edges, beam_width=2, max_fragments=130, max_boundary_cuts=1)
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


@pytest.mark.parametrize(
    "old_value,expected",
    [("fast_gnn", "connected_fragment_beam"), ("cut_masks", "cut_masks")],
)
def test_original_proposal_names_migrate_in_configs_and_checkpoints(tmp_path, old_value, expected):
    import lightning as L
    import yaml

    from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
    from fragnnet.utils.pl_utils import get_pl_hparams

    old_key = "mass_shatter_proposal_type"
    new_key = "mass_shatter_proposal_method"
    config = _small_pl_config()
    del config[new_key]
    config[old_key] = old_value
    model = MassShatterPL(**config)
    assert model.model.proposal_method == expected
    assert old_key not in model.hparams
    checkpoint = {
        "hyper_parameters": config,
        "state_dict": model.state_dict(),
        "pytorch-lightning_version": L.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    migrated = get_pl_hparams(checkpoint)
    assert migrated[new_key] == expected
    assert old_key not in migrated
    assert checkpoint["hyper_parameters"][old_key] == old_value
    path = tmp_path / "old.ckpt"
    th.save(checkpoint, path)
    restored = MassShatterPL.load_from_checkpoint(path, map_location="cpu", strict=True)
    assert restored.model.proposal_method == expected
    override = MassShatterPL.load_from_checkpoint(
        path, map_location="cpu", strict=True, **{new_key: "cut_masks"}
    )
    assert override.model.proposal_method == "cut_masks"
    custom = tmp_path / "old.yml"
    custom.write_text(yaml.safe_dump({old_key: old_value}))
    assert load_config("config", str(custom), None)[new_key] == expected
    assert load_config("config", None, [f"{old_key}={old_value}"])[new_key] == expected
    assert (
        load_config("config", str(custom), [f"{new_key}=cut_masks"])[new_key]
        == "cut_masks"
    )
    assert load_config(str(custom), None, None)[new_key] == expected


def test_outside_bonds_are_unrestricted():
    logits = th.zeros(2, requires_grad=True)
    candidates, _ = _graph(3, [(0, 1), (1, 2)], logits=logits)
    fragments = _sets(candidates)
    endpoint = fragments.index(frozenset({0}))
    pair = fragments.index(frozenset({0, 1}))
    scores = candidates.mask_log_probabilities
    th.testing.assert_close(scores[endpoint].exp(), th.tensor(0.5))
    th.testing.assert_close(scores[pair].exp(), th.tensor(0.25))
    th.testing.assert_close(
        th.autograd.grad(scores[endpoint], logits)[0], th.tensor([0.5, 0.0])
    )


@pytest.mark.parametrize("beam_width", [1, 128])
def test_selection_uses_fragment_probability(beam_width):
    # Boundary-logit ranking favors the middle singleton; fragment probability favors A.
    logits = th.tensor([2.0, 1.0])
    candidates, _ = _graph(
        3, [(0, 1), (1, 2)], logits=logits,
        max_fragments=2, beam_width=beam_width,
    )
    assert set(_sets(candidates)) == {frozenset({0}), frozenset({0, 1, 2})}


@pytest.mark.parametrize("seed", range(5))
def test_capped_search_matches_exhaustive_chain_ranking(seed):
    # Wide search enumerates the chain; mirrors must use their own internal scores.
    generator = th.Generator().manual_seed(seed)
    logits = th.randn(5, generator=generator)
    edges = [(i, i + 1) for i in range(5)]
    reference = []
    for start in range(6):
        for stop in range(start + 1, 7):
            fragment = frozenset(range(start, stop))
            if len(fragment) == 6:
                continue
            boundary = th.tensor([(u in fragment) != (v in fragment) for u, v in edges])
            internal = th.tensor([(u in fragment) and (v in fragment) for u, v in edges])
            score = F.logsigmoid(logits[boundary]).sum() + F.logsigmoid(-logits[internal]).sum()
            reference.append((float(score), fragment))
    expected = {f for _, f in sorted(reference, key=lambda entry: entry[0], reverse=True)[:3]}
    candidates, _ = _graph(6, edges, logits=logits, max_fragments=4, beam_width=128)
    assert set(_sets(candidates)) == expected | {frozenset(range(6))}
