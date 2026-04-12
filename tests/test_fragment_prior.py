"""Tests for the structure-only fragment prior head in FraGNNetModel."""

import pytest
import torch as th
from torch_geometric.data import Data

from fragnnet.model.fragnnet_model import FraGNNetModel
from fragnnet.utils.frag_utils import th_mask_to_long


def _make_model(
    *,
    fragment_prior: bool = False,
    fragment_prior_weight: float = 0.0,
    fragment_prior_loss: bool = False,
    mlp_output_format: str = "formula",
) -> FraGNNetModel:
    return FraGNNetModel(
        num_depth=2,
        num_hs=1,
        num_elements=3,
        int_embedder="abs-sines",
        int_embedder_tight=False,
        mol_node_feats=[],
        mol_edge_feats=[],
        mol_pe_embed_k=0,
        mol_hidden_size=8,
        mol_num_layers=1,
        mol_gnn_type="NodeMLP",
        mol_normalization="none",
        mol_dropout=0.0,
        mol_pool_type="mean",
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "boundary_pair"],
        frag_edge_feats=[],
        frag_hidden_size=8,
        frag_num_layers=1,
        frag_gnn_type="NodeMLP",
        frag_normalization="none",
        frag_dropout=0.0,
        frag_pool_type="mean",
        frag_embed_combine="cat",
        frag_pool_combine="subtract",
        mlp_output_format=mlp_output_format,
        mlp_hidden_size=8,
        mlp_normalization="none",
        mlp_dropout=0.0,
        mlp_num_layers=1,
        mlp_use_residuals=False,
        cc_interstage_type="direct",
        cc_interstage_use_rest=False,
        nb_iso=False,
        skip_edge_loss=True,
        mask_null_formula=False,
        predict_oos=False,
        bin_output=False,
        mz_bin_res=0.01,
        mz_max=200.0,
        ce_insert_location="mol",
        ce_insert_type="embed",
        ce_insert_merge=False,
        ce_insert_size=4,
        ce_max=200.0,
        ce_mean=60.0,
        ce_std=40.0,
        prec_insert_location="mol",
        prec_insert_size=4,
        prec_types=["[M+H]+"],
        inst_insert_location="mol",
        inst_insert_size=4,
        inst_types=["FT"],
        frag_mode_insert_location="mol",
        frag_mode_insert_size=4,
        frag_modes=["HCD"],
        frag_mode_scale=False,
        ce_scale=False,
        ce_scaler_hidden_dim=8,
        fragment_prior=fragment_prior,
        fragment_prior_weight=fragment_prior_weight,
        fragment_prior_conditioning="structure_only",
        fragment_prior_loss=fragment_prior_loss,
        fragment_prior_loss_weight=0.0,
        output_formula_str=False,
    )


def _make_forward_inputs() -> dict:
    mol_pyg = Data(
        x=th.zeros((2, 0), dtype=th.float32),
        edge_index=th.tensor([[0, 1], [1, 0]], dtype=th.long),
        edge_attr=th.zeros((2, 0), dtype=th.float32),
        batch=th.zeros(2, dtype=th.long),
    )

    root_mask = th.zeros((128,), dtype=th.bool)
    root_mask[:2] = True
    frag_mask = th.zeros((128,), dtype=th.bool)
    frag_mask[0] = True
    cc = th_mask_to_long(th.stack([root_mask, frag_mask], dim=0))

    depth = th.tensor([[1, 0], [0, 1]], dtype=th.long)
    base_formula = th.tensor([[2, 0, 0], [1, 0, 0]], dtype=th.long)
    h_formulae_idx = th.tensor([[1, 0, 0], [2, 3, 0]], dtype=th.long)
    h_counts = th.zeros((2, 0), dtype=th.long)
    nb_iso = th.zeros((2, 0), dtype=th.long)
    cmf = th.zeros((2, 0), dtype=th.long)
    prepass = th.zeros((2, 0), dtype=th.long)
    frag_x = th.cat(
        [depth, cc, base_formula, h_formulae_idx, h_counts, nb_iso, cmf, prepass], dim=1
    )
    node_feat_idxs = th.tensor([[0, 2, 4, 7, 10, 10, 10, 10, 10]], dtype=th.long)
    edge_feat_idxs = th.tensor([[0, 0, 0, 0, 0]], dtype=th.long)
    frag_pyg = Data(
        x=frag_x,
        edge_index=th.zeros((2, 0), dtype=th.long),
        edge_attr=th.zeros((0, 0), dtype=th.long),
        node_feat_idxs=node_feat_idxs,
        edge_feat_idxs=edge_feat_idxs,
        batch=th.zeros(2, dtype=th.long),
    )

    return {
        "mol_pyg": mol_pyg,
        "frag_pyg": frag_pyg,
        "mol_num_nodes": th.tensor([0, 2], dtype=th.long),
        "frag_num_nodes": th.tensor([0, 2], dtype=th.long),
        "frag_formula_peak_idxs": th.tensor([1, 2, 3], dtype=th.long),
        "frag_formula_peak_mzs": th.tensor([101.0, 51.0, 50.0], dtype=th.float32),
        "frag_formula_peak_probs": th.ones(3, dtype=th.float32),
        "frag_formula_sizes": th.tensor([4], dtype=th.long),
        "frag_formula_cumsizes": th.tensor([0, 4], dtype=th.long),
        "frag_formula_peak_sizes": th.tensor([3], dtype=th.long),
        "spec_ce": th.tensor([30.0], dtype=th.float32),
        "spec_ce_batch_idxs": th.tensor([0], dtype=th.long),
        "spec_prec_type": th.tensor([0], dtype=th.long),
        "spec_inst_type": th.tensor([0], dtype=th.long),
        "spec_frag_mode": th.tensor([0], dtype=th.long),
    }


def _forward_with_prior_state(model: FraGNNetModel, enabled: bool) -> dict:
    old_prior = model.fragment_prior
    old_module = model.fragment_prior_module
    try:
        model.fragment_prior = enabled
        if not enabled:
            model.fragment_prior_module = None
        return model(**_make_forward_inputs())
    finally:
        model.fragment_prior = old_prior
        model.fragment_prior_module = old_module


def test_zero_initialized_prior_matches_disabled_model():
    model = _make_model(fragment_prior=True, fragment_prior_weight=0.1).eval()
    with th.no_grad():
        enabled = _forward_with_prior_state(model, True)
        disabled = _forward_with_prior_state(model, False)

    assert th.allclose(
        enabled["pred_joint_logprobs"], disabled["pred_joint_logprobs"], atol=1e-6
    )
    assert th.allclose(enabled["pred_logprobs"], disabled["pred_logprobs"], atol=1e-6)


def test_zero_weight_prior_matches_disabled_model_after_bias_change():
    model = _make_model(fragment_prior=True, fragment_prior_weight=0.0).eval()
    assert model.fragment_prior_module is not None
    with th.no_grad():
        model.fragment_prior_module.output_layer.bias[0] = 1.0
        enabled = _forward_with_prior_state(model, True)
        disabled = _forward_with_prior_state(model, False)

    assert th.allclose(
        enabled["pred_joint_logprobs"], disabled["pred_joint_logprobs"], atol=1e-6
    )


def test_positive_weight_prior_changes_joint_distribution_after_bias_change():
    model = _make_model(fragment_prior=True, fragment_prior_weight=0.5).eval()
    assert model.fragment_prior_module is not None
    with th.no_grad():
        model.fragment_prior_module.output_layer.bias[0] = 1.0
        enabled = _forward_with_prior_state(model, True)
        disabled = _forward_with_prior_state(model, False)

    assert not th.allclose(
        enabled["pred_joint_logprobs"], disabled["pred_joint_logprobs"], atol=1e-6
    )


def test_fragment_prior_loss_not_implemented():
    with pytest.raises(NotImplementedError, match="fragment_prior_loss"):
        _make_model(fragment_prior=True, fragment_prior_loss=True)


def test_fragment_prior_rejects_non_formula_output_format():
    with pytest.raises(NotImplementedError, match="mlp_output_format='formula'"):
        _make_model(fragment_prior=True, mlp_output_format="node_formula")

