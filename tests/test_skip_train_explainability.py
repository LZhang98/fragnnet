"""Tests for skip_explainability_when_training (training-time memory fast path).

When enabled, the model skips the terminal frag_real_* / nb_iso explainability marginals
(P(n), P(f), P(f|n), P(n|f), P(h)) on TRAINING forwards -- they feed only the entropy aux losses
and eval/inference, so they are dead weight on the training forward peak otherwise.

The invariants under test:
- training + flag on  -> the explainability out_d keys are None, but the loss-critical spectrum
  path (pred_mzs, pred_logprobs) is bit-for-bit identical to the flag-off baseline;
- eval / inference    -> the marginals are always computed (so val aux metrics are unaffected),
  even with the flag on;
- flag off (default)  -> marginals always computed (backward compatible).
"""

from __future__ import annotations

import torch as th
import torch_geometric as pyg

from fragnnet.model.fragnnet_model import FraGNNetModel
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.feat_utils import get_frag_graph, get_mol_graph
from fragnnet.utils.frag_utils import compute_dags, extract_mol_info

MOL_NODE_FEATS = ["a_onehot", "a_degree", "a_ring", "a_mass"]
MOL_EDGE_FEATS = ["b_degree"]
FRAG_NODE_FEATS = [
    "cc",
    "base_formula",
    "depth",
    "h_formulae_idx",
    "out_degree",
    "in_degree",
    "num_heavy_atoms",
]

# out_d keys that come from the frag_real_* explainability block
EXPL_KEYS = [
    "pred_formula_logprobs",
    "pred_node_logprobs",
    "pred_node_formula_logprobs",
    "pred_formula_node_logprobs",
    "pred_joint_logprobs",
    "pred_h_logprobs",
]


def _build_model(skip_explainability_when_training: bool) -> FraGNNetModel:
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
        frag_node_feats=FRAG_NODE_FEATS,
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
        skip_explainability_when_training=skip_explainability_when_training,
    )


def _forward_kwargs(smiles: str = "CCO", ce: float = 30.0) -> dict:
    mol = mol_from_smiles(smiles)
    mol_pyg = get_mol_graph(
        mol, atom_feats=MOL_NODE_FEATS, bond_feats=MOL_EDGE_FEATS, pe_embed_k=0, bigraph=True
    )
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=3,
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
        frag_node_feats=FRAG_NODE_FEATS,
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
        "spec_ce": th.tensor([ce]),
        "spec_ce_batch_idxs": th.tensor([0]),
        "spec_prec_type": th.tensor([0]),
        "spec_inst_type": th.tensor([0]),
        "spec_frag_mode": th.tensor([0]),
        "boundary_pair_frag_idxs": dag_d["dag"].boundary_pair_frag_idxs,
        "boundary_pair_in_idxs": dag_d["dag"].boundary_pair_in_local,
        "boundary_pair_out_idxs": dag_d["dag"].boundary_pair_out_local,
    }


class TestSkipTrainExplainability:
    def test_training_skips_marginals(self) -> None:
        """train() + flag on -> explainability keys are None, spectrum path still finite."""
        model = _build_model(skip_explainability_when_training=True)
        model.train()
        with th.no_grad():
            out = model(**_forward_kwargs())
        for k in EXPL_KEYS:
            assert out[k] is None, f"{k} should be None on the training fast path"
        assert th.isfinite(out["pred_logprobs"]).all()
        assert th.isfinite(out["pred_mzs"]).all()

    def test_training_spectrum_matches_baseline(self) -> None:
        """The loss-critical spectrum path is bit-for-bit identical with the flag on vs off."""
        base = _build_model(skip_explainability_when_training=False)
        skip = _build_model(skip_explainability_when_training=True)
        skip.load_state_dict(base.state_dict())
        base.train()
        skip.train()
        kwargs = _forward_kwargs("CCO", ce=70.0)
        with th.no_grad():
            out_base = base(**kwargs)
            out_skip = skip(**kwargs)
        assert th.equal(out_base["pred_mzs"], out_skip["pred_mzs"])
        assert th.equal(out_base["pred_logprobs"], out_skip["pred_logprobs"])
        assert th.equal(out_base["pred_batch_idxs"], out_skip["pred_batch_idxs"])
        # baseline still populates the marginals; the fast path drops them
        assert out_base["pred_node_formula_logprobs"] is not None
        assert out_skip["pred_node_formula_logprobs"] is None

    def test_eval_computes_marginals_even_with_flag(self) -> None:
        """eval() forces the marginals even with the flag on (val aux metrics stay valid)."""
        model = _build_model(skip_explainability_when_training=True)
        model.eval()
        with th.no_grad():
            out = model(**_forward_kwargs())
        for k in EXPL_KEYS:
            assert out[k] is not None, f"{k} must be computed in eval mode"
            assert th.isfinite(out[k]).all()

    def test_explicit_request_overrides_flag_in_train(self) -> None:
        """return_explainability_tensors=True computes marginals even in train() with the flag."""
        model = _build_model(skip_explainability_when_training=True)
        model.train()
        with th.no_grad():
            out = model(**_forward_kwargs(), return_explainability_tensors=True)
        assert out["pred_node_formula_logprobs"] is not None
        assert out["pred_formula_formula_idxs"] is not None  # only set when requested

    def test_flag_off_always_computes(self) -> None:
        """Default (flag off) computes marginals in train mode -- backward compatible."""
        model = _build_model(skip_explainability_when_training=False)
        model.train()
        with th.no_grad():
            out = model(**_forward_kwargs())
        for k in EXPL_KEYS:
            assert out[k] is not None
