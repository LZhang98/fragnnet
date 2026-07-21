"""Tests for cc_atom_dropout: training-only per-atom dropout on the fragment cc mask.

Each atom is independently dropped from a fragment's atom-membership set during training, so the
pooled ``cc`` embedding sees a random subset of the fragment's atoms. This regularizes against
memorizing exact atom sets (aimed at the train->val gap), and is much finer-grained than
``cc_feature_dropout``, which zeroes the whole cc group for the whole batch on one scalar draw.

Invariants under test:
- eval() is unaffected (dropout is training-only) and deterministic;
- train() with p>0 perturbs the prediction, and different draws give different predictions;
- p=0 is a no-op (train output == eval output);
- p=1.0 does not produce NaN/inf -- every fragment is restored rather than emptied.
"""

from __future__ import annotations

import pytest
import torch as th
import torch_geometric as pyg

from fragnnet.model.fragnnet_model import FraGNNetModel
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.feat_utils import get_frag_graph, get_mol_graph
from fragnnet.utils.frag_utils import compute_dags, extract_mol_info

MOL_NODE_FEATS = ["a_onehot", "a_degree", "a_ring", "a_mass"]
MOL_EDGE_FEATS = ["b_degree"]
FRAG_NODE_FEATS = ["cc", "base_formula", "depth", "h_formulae_idx"]


def _build_model(
    cc_atom_dropout: float,
    frag_node_feats: list[str] | None = None,
    candidate_ion_head: bool = False,
    ion_channel_factorized: bool = False,
) -> FraGNNetModel:
    if frag_node_feats is None:
        frag_node_feats = FRAG_NODE_FEATS
    return FraGNNetModel(
        num_depth=4,
        num_hs=2,
        num_elements=17,
        int_embedder="fourier",
        int_embedder_tight=True,
        mol_node_feats=MOL_NODE_FEATS,
        mol_edge_feats=MOL_EDGE_FEATS,
        mol_pe_embed_k=0,
        mol_hidden_size=32,
        mol_num_layers=2,
        mol_gnn_type="MPNN",
        mol_normalization="layer",
        mol_dropout=0.0,
        mol_num_heads=1,
        mol_pool_type="mean",
        frag_node_feats=frag_node_feats,
        frag_edge_feats=[],
        frag_hidden_size=32,
        frag_num_layers=2,
        frag_gnn_type="NodeMLP",
        frag_normalization="layer",
        frag_dropout=0.0,
        frag_pool_type="max",
        frag_embed_combine="cat",
        frag_pool_combine="none",
        mlp_output_format="formula",
        mlp_hidden_size=32,
        mlp_normalization="layer",
        mlp_dropout=0.0,
        mlp_num_layers=2,
        mlp_use_residuals=True,
        cc_interstage_type="add",
        cc_interstage_use_rest=False,
        nb_iso=False,
        skip_edge_loss=True,
        mask_null_formula=False,
        predict_oos=False,
        bin_output=False,
        mz_bin_res=0.01,
        mz_max=1500.0,
        ce_insert_location="mol",
        ce_insert_type="embed_clamped",
        ce_insert_merge=True,
        ce_insert_size=8,
        nce_max=200.0,
        nce_mean=30.0,
        nce_std=20.0,
        prec_insert_location="mol",
        prec_insert_size=8,
        prec_types=["[M+H]+"],
        inst_insert_location="mol",
        inst_insert_size=8,
        inst_types=["FT"],
        frag_mode_insert_location="mol",
        frag_mode_insert_size=8,
        frag_modes=["HCD"],
        frag_mode_scale=False,
        ce_scale=False,
        ce_scaler_hidden_dim=16,
        adduct_scale=False,
        stratum_scale=False,
        stratum_scaler_embed_dim=8,
        output_formula_str=False,
        use_nce=True,
        use_ace=False,
        ace_max=50.0,
        ace_mean=20.0,
        ace_std=10.0,
        cc_feature_dropout=0.0,
        cc_atom_dropout=cc_atom_dropout,
        candidate_ion_head=candidate_ion_head,
        ion_channel_factorized=ion_channel_factorized,
        ion_channel_rank=4,
    )


def _forward_kwargs(
    smiles: str = "CCOC(=O)c1ccccc1",
    frag_node_feats: list[str] | None = None,
) -> dict:
    if frag_node_feats is None:
        frag_node_feats = FRAG_NODE_FEATS
    mol = mol_from_smiles(smiles)
    mol_pyg = get_mol_graph(
        mol, atom_feats=MOL_NODE_FEATS, bond_feats=MOL_EDGE_FEATS, pe_embed_k=0, bigraph=True
    )
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=3,  # must match model num_depth=4 (depth-3 frags + root)
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=frag_node_feats,
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
    )
    num_formulae = len(dag_d["idx_to_formula"])
    nz = th.nonzero(dag_d["formula_peak_probs"])
    peak_idxs = nz[:, 0]
    return {
        "mol_pyg": pyg.data.Batch.from_data_list([mol_pyg]),
        "frag_pyg": pyg.data.Batch.from_data_list([frag_pyg]),
        "mol_num_nodes": th.tensor([0, mol_pyg.num_nodes]),
        "frag_num_nodes": th.tensor([0, frag_pyg.num_nodes]),
        "frag_formula_peak_idxs": peak_idxs,
        "frag_formula_peak_mzs": dag_d["formula_peak_mzs"][nz[:, 0], nz[:, 1]],
        "frag_formula_peak_probs": dag_d["formula_peak_probs"][nz[:, 0], nz[:, 1]],
        "frag_formula_sizes": th.tensor([num_formulae]),
        "frag_formula_cumsizes": th.tensor([0, num_formulae]),
        "frag_formula_peak_sizes": th.tensor([peak_idxs.shape[0]]),
        "spec_ce": th.tensor([30.0]),
        "spec_ce_batch_idxs": th.tensor([0]),
        "spec_prec_type": th.tensor([0]),
        "spec_inst_type": th.tensor([0]),
        "spec_frag_mode": th.tensor([0]),
    }


class TestCCAtomDropoutValidation:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="cc_atom_dropout"):
            _build_model(cc_atom_dropout=1.5)
        with pytest.raises(ValueError, match="cc_atom_dropout"):
            _build_model(cc_atom_dropout=-0.1)


class TestCCAtomDropoutBehavior:
    def test_no_op_when_zero(self) -> None:
        """p=0: train output equals eval output (dropout inactive)."""
        model = _build_model(cc_atom_dropout=0.0)
        kwargs = _forward_kwargs()
        with th.no_grad():
            model.eval()
            out_eval = model(**kwargs)
            model.train()
            out_train = model(**kwargs)
        assert th.allclose(out_eval["pred_logprobs"], out_train["pred_logprobs"], atol=1e-6)

    def test_inactive_in_eval(self) -> None:
        """Dropout is training-only: eval is deterministic across repeated forwards."""
        model = _build_model(cc_atom_dropout=0.5)
        model.eval()
        kwargs = _forward_kwargs()
        with th.no_grad():
            a = model(**kwargs)
            b = model(**kwargs)
        assert th.allclose(a["pred_logprobs"], b["pred_logprobs"], atol=1e-6)

    def test_active_in_training(self) -> None:
        """p>0 in train(): perturbs predictions, and separate draws differ from each other."""
        model = _build_model(cc_atom_dropout=0.5)
        kwargs = _forward_kwargs()
        with th.no_grad():
            model.eval()
            out_eval = model(**kwargs)
            model.train()
            th.manual_seed(0)
            out_a = model(**kwargs)
            th.manual_seed(1)
            out_b = model(**kwargs)
        # training forward is perturbed relative to eval
        assert not th.allclose(out_eval["pred_logprobs"], out_a["pred_logprobs"], atol=1e-4)
        # different dropout draws give different predictions (it is genuinely stochastic)
        assert not th.allclose(out_a["pred_logprobs"], out_b["pred_logprobs"], atol=1e-4)

    def test_extreme_dropout_stays_finite(self) -> None:
        """p=1.0 must not produce NaN/inf: every fragment is restored, never emptied."""
        model = _build_model(cc_atom_dropout=1.0)
        model.train()
        with th.no_grad():
            out = model(**_forward_kwargs())
        assert th.isfinite(out["pred_logprobs"]).all()
        assert th.isfinite(out["pred_mzs"]).all()

    def test_gradients_flow_under_dropout(self) -> None:
        """The perturbed cc path still backprops into the mol encoder."""
        model = _build_model(cc_atom_dropout=0.5)
        model.train()
        out = model(**_forward_kwargs())
        out["pred_logprobs"].exp().sum().backward()
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if "mol_embedder" in n and p.grad is not None
        ]
        assert len(grads) > 0
        assert any(th.any(g != 0) for g in grads)


class TestCandidateIonForward:
    def test_candidate_ion_head_forward_is_finite(self) -> None:
        frag_node_feats = ["cc", "base_formula", "depth", "h_formulae_idx", "h_counts"]
        model = _build_model(
            cc_atom_dropout=0.0,
            frag_node_feats=frag_node_feats,
            candidate_ion_head=True,
        )
        model.eval()
        with th.no_grad():
            out = model(**_forward_kwargs(frag_node_feats=frag_node_feats))
        assert th.isfinite(out["pred_logprobs"]).all()
        assert th.isfinite(out["pred_mzs"]).all()

    def test_candidate_ion_checkpoint_backward(self) -> None:
        frag_node_feats = ["cc", "base_formula", "depth", "h_formulae_idx", "h_counts"]
        model = _build_model(
            cc_atom_dropout=0.0,
            frag_node_feats=frag_node_feats,
            candidate_ion_head=True,
        )
        model.candidate_ion_chunk_size = 64
        model.candidate_ion_checkpoint = True
        model.train()
        out = model(**_forward_kwargs(frag_node_feats=frag_node_feats))
        loss = -out["pred_logprobs"].mean()
        loss.backward()
        grads = [p.grad for p in model.formula_module.parameters() if p.grad is not None]
        assert grads
        assert any(th.isfinite(g).all() and th.any(g != 0) for g in grads)


class TestIonChannelFactorizedForward:
    def test_ion_channel_factorized_backward(self) -> None:
        model = _build_model(cc_atom_dropout=0.0, ion_channel_factorized=True)
        model.train()
        out = model(**_forward_kwargs())
        loss = -out["pred_logprobs"].mean()
        loss.backward()
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if n.startswith("ion_channel_") and p.grad is not None
        ]
        assert grads
        assert any(th.isfinite(g).all() and th.any(g != 0) for g in grads)
