"""Tests for frag_q: per-fragment Gasteiger charge sum DAG node feature.

frag_q is computed on-the-fly in get_frag_graph from the stored cc bitmask and
per-atom Gasteiger partial charges. It is stored as dag_pyg.frag_q (float32,
shape [N, 1]), NOT inside dag_pyg.x.

Covers:
1. frag_q attribute absent when "frag_q" not in frag_node_feats.
2. frag_q is all-zeros (shape [N, 1], float32) when gasteiger_arr=None.
3. Root node frag_q == sum of all Gasteiger charges (root contains all atoms).
4. Every fragment's frag_q matches the manual sum over its cc atom membership.
5. frag_q shape and dtype.
6. frag_q survives PyG batching unchanged.
7. Model forward pass runs and produces finite output with frag_q enabled.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from rdkit.Chem import AllChem

from fragnnet.models.fragnnet.frag_feat_utils import get_frag_graph
from fragnnet.models.fragnnet.frag_utils import (
    compute_dags,
    extract_mol_info,
    get_node_feats,
    th_long_to_mask,
)
from fragnnet.utils.data_utils import mol_from_smiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_dag(smiles: str, max_depth: int = 3) -> dict:
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )


def _gasteiger(smiles: str) -> np.ndarray:
    """Compute Gasteiger charges; NaN/Inf replaced with 0."""
    mol = mol_from_smiles(smiles)
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array(
        [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()],
        dtype=np.float32,
    )
    return np.where(np.isfinite(charges), charges, 0.0).astype(np.float32)


def _frag_q_graph(dag_d: dict, gasteiger_arr: np.ndarray | None) -> th.Tensor:
    """Return frag_pyg.frag_q after running get_frag_graph with "frag_q"."""
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=["frag_q"],
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
        gasteiger_arr=gasteiger_arr,
    )
    return frag_pyg.frag_q


# ---------------------------------------------------------------------------
# 1. Attribute absent when not requested
# ---------------------------------------------------------------------------


def test_frag_q_absent_when_not_requested() -> None:
    dag_d = _run_dag("CCO")
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=["cc", "base_formula"],
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
    )
    assert not hasattr(frag_pyg, "frag_q"), "frag_q should not be set when not in frag_node_feats"


# ---------------------------------------------------------------------------
# 2. Zeros fallback when gasteiger_arr is None
# ---------------------------------------------------------------------------


def test_frag_q_zeros_without_gasteiger() -> None:
    dag_d = _run_dag("CCO")
    frag_q = _frag_q_graph(dag_d, gasteiger_arr=None)
    num_nodes = dag_d["dag_num_nodes"]
    assert frag_q.shape == (num_nodes, 1), f"Expected ({num_nodes}, 1), got {frag_q.shape}"
    assert frag_q.dtype == th.float32, f"Expected float32, got {frag_q.dtype}"
    assert th.all(frag_q == 0.0), "frag_q should be all-zeros when gasteiger_arr=None"


# ---------------------------------------------------------------------------
# 3. Root node frag_q == sum of all Gasteiger charges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "c1ccccc1C(=O)O"])
def test_frag_q_root_equals_total_charge(smiles: str) -> None:
    """Root node (precursor, depth bit 0) contains all atoms → its frag_q is the total charge."""
    dag_d = _run_dag(smiles)
    gasteiger_arr = _gasteiger(smiles)
    frag_q = _frag_q_graph(dag_d, gasteiger_arr)

    dag = dag_d["dag"]
    depth_feat = get_node_feats(dag.x, dag.node_feat_idxs[0], "depth")
    root_idx = depth_feat[:, 0].bool().nonzero(as_tuple=True)[0][0].item()

    expected = float(gasteiger_arr.sum())
    actual = frag_q[root_idx, 0].item()
    assert abs(actual - expected) < 1e-4, (
        f"{smiles}: root frag_q={actual:.6f}, expected total charge={expected:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Per-fragment correctness: frag_q == sum of charges for cc atoms
# ---------------------------------------------------------------------------


def test_frag_q_matches_cc_atom_sum() -> None:
    """For every fragment, frag_q should equal the dot product of cc mask and gasteiger_arr."""
    smiles = "CCO"  # 3 heavy atoms (C, C, O)
    dag_d = _run_dag(smiles)
    gasteiger_arr = _gasteiger(smiles)
    frag_q = _frag_q_graph(dag_d, gasteiger_arr)

    dag = dag_d["dag"]
    cc_long = get_node_feats(dag.x, dag.node_feat_idxs[0], "cc").long()
    cc_binary = th_long_to_mask(cc_long)[:, : len(gasteiger_arr)].float()
    g_arr = th.as_tensor(gasteiger_arr)
    expected = (cc_binary @ g_arr).unsqueeze(1)

    assert frag_q.shape == expected.shape
    assert th.allclose(frag_q, expected, atol=1e-5), (
        f"frag_q mismatch:\n{frag_q.squeeze()}\nvs expected:\n{expected.squeeze()}"
    )


# ---------------------------------------------------------------------------
# 5. Shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["CO", "CCO", "c1ccccc1"])
def test_frag_q_shape_and_dtype(smiles: str) -> None:
    dag_d = _run_dag(smiles)
    gasteiger_arr = _gasteiger(smiles)
    frag_q = _frag_q_graph(dag_d, gasteiger_arr)
    num_nodes = dag_d["dag_num_nodes"]
    assert frag_q.shape == (num_nodes, 1), f"Expected ({num_nodes}, 1), got {frag_q.shape}"
    assert frag_q.dtype == th.float32, f"Expected float32, got {frag_q.dtype}"


# ---------------------------------------------------------------------------
# 6. frag_q survives PyG batching
# ---------------------------------------------------------------------------


def test_frag_q_survives_batching() -> None:
    """PyG Batch.from_data_list should preserve frag_q by concatenation."""
    import torch_geometric as pyg

    smiles_list = ["CCO", "c1ccccc1"]
    frag_pygs = []
    total_nodes = 0
    for smi in smiles_list:
        dag_d = _run_dag(smi)
        g_arr = _gasteiger(smi)
        fp = get_frag_graph(
            dag_d["dag"],
            frag_node_feats=["frag_q"],
            frag_edge_feats=[],
            edges=False,
            bigraph=False,
            gasteiger_arr=g_arr,
        )
        frag_pygs.append(fp)
        total_nodes += dag_d["dag_num_nodes"]

    batch = pyg.data.Batch.from_data_list(frag_pygs)
    assert hasattr(batch, "frag_q"), "frag_q missing after batching"
    assert batch.frag_q.shape == (total_nodes, 1), (
        f"Expected ({total_nodes}, 1) after batching, got {batch.frag_q.shape}"
    )
    # Verify the values are intact by comparing slice-by-slice
    offset = 0
    for fp in frag_pygs:
        n = fp.frag_q.shape[0]
        assert th.allclose(batch.frag_q[offset : offset + n], fp.frag_q, atol=1e-6)
        offset += n


# ---------------------------------------------------------------------------
# 7. Model forward pass with frag_q enabled
# ---------------------------------------------------------------------------


@pytest.fixture
def frag_q_model():
    """Smallest valid FraGNNetModel with frag_q in frag_node_feats."""
    from fragnnet.models.fragnnet.fragnnet_model import FraGNNetModel

    return FraGNNetModel(
        num_depth=4,
        num_hs=2,
        num_elements=17,
        int_embedder="fourier",
        int_embedder_tight=True,
        mol_node_feats=["a_onehot", "a_degree", "a_ring", "a_mass"],
        mol_edge_feats=["b_degree"],
        mol_pe_embed_k=0,
        mol_hidden_size=32,
        mol_num_layers=2,
        mol_gnn_type="MPNN",
        mol_normalization="layer",
        mol_dropout=0.0,
        mol_num_heads=1,
        mol_pool_type="mean",
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "frag_q"],
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
        inst_types=["ORB"],
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
    )


def test_model_forward_with_frag_q(frag_q_model) -> None:
    """Forward pass must complete without error and produce finite output."""
    import torch_geometric as pyg

    from fragnnet.utils.feat_utils import get_mol_graph

    smiles = "CCO"
    mol = mol_from_smiles(smiles)
    gasteiger_arr = _gasteiger(smiles)

    mol_pyg = get_mol_graph(
        mol,
        atom_feats=["a_onehot", "a_degree", "a_ring", "a_mass"],
        bond_feats=["b_degree"],
        pe_embed_k=0,
        bigraph=True,
    )
    dag_d = _run_dag(smiles)
    frag_pyg = get_frag_graph(
        dag_d["dag"],
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "frag_q"],
        frag_edge_feats=[],
        edges=False,
        bigraph=False,
        gasteiger_arr=gasteiger_arr,
    )

    num_formulae = len(dag_d["idx_to_formula"])
    formula_peak_mzs = dag_d["formula_peak_mzs"]
    formula_peak_probs = dag_d["formula_peak_probs"]
    nz = th.nonzero(formula_peak_probs)
    peak_idxs = nz[:, 0]
    peak_mzs = formula_peak_mzs[nz[:, 0], nz[:, 1]]
    peak_probs = formula_peak_probs[nz[:, 0], nz[:, 1]]

    frag_q_model.eval()
    with th.no_grad():
        out = frag_q_model(
            mol_pyg=pyg.data.Batch.from_data_list([mol_pyg]),
            frag_pyg=pyg.data.Batch.from_data_list([frag_pyg]),
            mol_num_nodes=th.tensor([0, mol_pyg.num_nodes]),
            frag_num_nodes=th.tensor([0, frag_pyg.num_nodes]),
            frag_formula_peak_idxs=peak_idxs,
            frag_formula_peak_mzs=peak_mzs,
            frag_formula_peak_probs=peak_probs,
            frag_formula_sizes=th.tensor([num_formulae]),
            frag_formula_cumsizes=th.tensor([0, num_formulae]),
            frag_formula_peak_sizes=th.tensor([peak_idxs.shape[0]]),
            spec_ce=th.tensor([30.0]),
            spec_ce_batch_idxs=th.tensor([0]),
            spec_prec_type=th.tensor([0]),
            spec_inst_type=th.tensor([0]),
            spec_frag_mode=th.tensor([0]),
            boundary_pair_frag_idxs=dag_d["dag"].boundary_pair_frag_idxs,
            boundary_pair_in_idxs=dag_d["dag"].boundary_pair_in_local,
            boundary_pair_out_idxs=dag_d["dag"].boundary_pair_out_local,
        )

    assert out is not None
    logprobs = out if isinstance(out, th.Tensor) else out.get("logprobs", next(iter(out.values())))
    assert th.isfinite(logprobs).all(), "Model output contains NaN or Inf"
