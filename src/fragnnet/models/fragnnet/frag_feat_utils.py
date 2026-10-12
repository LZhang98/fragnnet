"""FraGNNet fragment-DAG graphs: one molecule's DAG as a PyG graph, and DAG batching.

Moved out of core ``feat_utils`` in the reorganization (docs/plans/repo_reorg_plan.md, Phase 3), so
core featurization no longer imports FraGNNet's ``frag_utils``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch as th
import torch_geometric as pyg
from torch_geometric.data import Data

from fragnnet.models.fragnnet.frag_utils import (
    EDGE_FEAT_TO_IDX,
    NODE_FEAT_TO_IDX,
    get_edge_feats,
    get_node_feats,
)
from fragnnet.utils.formula_utils import th_long_to_mask


def batch_mols_frags(
    mol_pyg_list: Sequence[pyg.data.Data],
    frag_pyg_list: Sequence[pyg.data.Data],
    formula_peak_mzs_list: Sequence[th.Tensor],
    formula_peak_probs_list: Sequence[th.Tensor],
    debug_validate_outputs: bool = False,
) -> dict[str, Any]:
    """Batch molecule/fragment graphs and sparse formula peak tensors."""
    # batch size
    batch_size = len(mol_pyg_list)
    # Convert the input lists of molecules and fragments into PyTorch Geometric data batches
    mol_pyg = pyg.data.Batch.from_data_list(mol_pyg_list)
    frag_pyg = pyg.data.Batch.from_data_list(frag_pyg_list)

    # Assert that the number of graphs in both molecule and fragment batches matches the batch size
    assert mol_pyg.num_graphs == frag_pyg.num_graphs == batch_size
    # Assert that the node/edge feature indices are consistent. This is useful for
    # debugging data corruption, but expensive enough to keep off in normal training.
    if debug_validate_outputs:
        assert all(
            th.all(frag_pyg.node_feat_idxs[0] == frag_pyg.node_feat_idxs[i])
            for i in range(batch_size)
        )
        assert all(
            th.all(frag_pyg.edge_feat_idxs[0] == frag_pyg.edge_feat_idxs[i])
            for i in range(batch_size)
        )

    # Compute the cumulative sum of nodes for molecules and fragments, to track the total number of nodes per graph
    mol_num_nodes = [g.num_nodes for g in mol_pyg_list]
    frag_num_nodes = [g.num_nodes for g in frag_pyg_list]
    mol_num_nodes = th.cumsum(th.tensor([0] + mol_num_nodes), dim=0)
    frag_num_nodes = th.cumsum(th.tensor([0] + frag_num_nodes), dim=0)

    boundary_pair_frag_idxs = []
    boundary_pair_in_idxs = []
    boundary_pair_out_idxs = []
    for mol_idx, (_mol_pyg_item, frag_pyg_item) in enumerate(zip(mol_pyg_list, frag_pyg_list)):
        if not hasattr(frag_pyg_item, "boundary_pair_frag_idxs"):
            raise ValueError(
                "frag_pyg is missing precomputed boundary_pair_* attributes; "
                "regenerate DAGs before using the boundary_pair feature"
            )
        frag_offsets = frag_num_nodes[mol_idx]
        mol_offsets = mol_num_nodes[mol_idx]
        boundary_pair_frag_idxs.append(frag_pyg_item.boundary_pair_frag_idxs + frag_offsets)
        boundary_pair_in_idxs.append(frag_pyg_item.boundary_pair_in_local + mol_offsets)
        boundary_pair_out_idxs.append(frag_pyg_item.boundary_pair_out_local + mol_offsets)

    # Initialize lists
    frag_formula_peak_idxs = []  # 1D tensor, indices of the formula_i of fragment_j created peak_k in fragments, concatenated across all fragments.
    frag_formula_peak_mzs = []
    frag_formula_peak_probs = []
    frag_formula_sizes = []  # 1D tensor, offset for each fragment's formula indices in the concatenated batch.
    frag_formula_peak_sizes = []  # 1D tensor, number of nonzero peaks per fragment. in case of isotopes, each formula can have more than one peak.

    # Loop through the mz and probs for each sample
    for mzs, probs in zip(formula_peak_mzs_list, formula_peak_probs_list):
        # sparsify: retain only non-zero probability peaks
        idx = th.nonzero(probs)
        # Extract mass-to-charge ratios where probability is non-zero
        mzs = mzs[idx[:, 0], idx[:, 1]]
        # Extract probabilities where probability is non-zero
        probs = probs[idx[:, 0], idx[:, 1]]
        # Append the filtered results for each fragment
        frag_formula_peak_idxs.append(idx[:, 0])
        frag_formula_peak_mzs.append(mzs)
        frag_formula_peak_probs.append(probs)

        # Calculate the number of unique formula peaks and the total peak sizes
        # In case of isotopes, each formula can have more then one peak
        frag_formula_sizes.append(th.unique(idx, sorted=True).shape[0])
        frag_formula_peak_sizes.append(idx.shape[0])

    # Flatten the lists to single dimension tensor
    frag_formula_peak_idxs = th.cat(frag_formula_peak_idxs, dim=0)
    frag_formula_peak_mzs = th.cat(frag_formula_peak_mzs, dim=0)
    frag_formula_peak_probs = th.cat(frag_formula_peak_probs, dim=0)
    frag_formula_cumsizes = th.cumsum(th.tensor([0] + frag_formula_sizes), dim=0)
    frag_formula_sizes = th.tensor(frag_formula_sizes)
    frag_formula_peak_sizes = th.tensor(frag_formula_peak_sizes)
    batched_mol_frag = {
        "mol_pyg": mol_pyg,  # PyG batch for molecules
        "frag_pyg": frag_pyg,  # PyG batch for fragments
        "mol_num_nodes": mol_num_nodes,  # Cumulative node count for molecules
        "frag_num_nodes": frag_num_nodes,  # Cumulative node count for fragments
        "frag_formula_peak_idxs": frag_formula_peak_idxs,  # Flattened tensor mapping each peak to its local formula index (0-based per molecule)
        "frag_formula_peak_mzs": frag_formula_peak_mzs,  # Flattened tensor of m/z values for each peak
        "frag_formula_peak_probs": frag_formula_peak_probs,  # Flattened tensor of theoretical isotopic probabilities for each peak
        "frag_formula_sizes": frag_formula_sizes,  # Tensor containing the number of unique formulas (fragments) with peaks for each molecule
        "frag_formula_cumsizes": frag_formula_cumsizes,  # Cumulative sum of frag_formula_sizes (offsets for formula indices in batch)
        "frag_formula_peak_sizes": frag_formula_peak_sizes,  # Tensor containing the total number of peaks (isotopes) for each molecule
    }
    batched_mol_frag["boundary_pair_frag_idxs"] = th.cat(boundary_pair_frag_idxs, dim=0)
    batched_mol_frag["boundary_pair_in_idxs"] = th.cat(boundary_pair_in_idxs, dim=0)
    batched_mol_frag["boundary_pair_out_idxs"] = th.cat(boundary_pair_out_idxs, dim=0)
    return batched_mol_frag


def get_frag_graph(
    dag_pyg: Data,
    frag_node_feats: Sequence[str],
    frag_edge_feats: Sequence[str],
    edges: bool,
    bigraph: bool,
    gasteiger_arr: np.ndarray | None = None,
) -> Data:
    """Subselect DAG features and optionally symmetrize edges for fragments.

    Args:
        dag_pyg: Input PyTorch Geometric Data object with node and edge features.
        frag_node_feats: Sequence of node feature names to include.
        frag_edge_feats: Sequence of edge feature names to include.
        edges: Whether to include edges in the output graph.
        bigraph: Whether to symmetrize edges by adding reverse directions.
        gasteiger_arr: Per-atom Gasteiger partial charges of shape (num_atoms,).
            Required when "frag_q" is in frag_node_feats; zeros used if None.

    Returns:
        PyTorch Geometric Data object with selected features and optionally symmetrized edges.

    Raises:
        ValueError: If dag_pyg.x (node features) is None.
    """

    source_dag_pyg = dag_pyg
    device = dag_pyg.x.device
    # cast
    x = dag_pyg.x.long()
    edge_attr = dag_pyg.edge_attr.long()
    edge_index = dag_pyg.edge_index.long()
    node_feat_idxs = dag_pyg.node_feat_idxs.long()
    edge_feat_idxs = dag_pyg.edge_feat_idxs.long()
    # select node features
    # TODO: out_degree / in_degree / num_heavy_atoms are computed on-the-fly here for now
    # (avoids DAG regeneration during testing). Once validated, move the computation into
    # compute_dags in frag_utils.py and store them in dag.x like other features.
    _x = []
    _node_feat_idxs = [0]
    _node_feat_size = 0
    num_nodes = x.shape[0]
    for feat, _ in NODE_FEAT_TO_IDX.items():
        if feat in frag_node_feats:
            if feat == "out_degree":  # on-the-fly: count outgoing edges per node
                _x_cur = th.zeros(num_nodes, 1, device=device, dtype=th.int64)
                if edge_index.shape[1] > 0:
                    _x_cur.scatter_add_(
                        0, edge_index[0].unsqueeze(1), th.ones_like(edge_index[0].unsqueeze(1))
                    )
            elif feat == "in_degree":  # on-the-fly: count incoming edges per node
                _x_cur = th.zeros(num_nodes, 1, device=device, dtype=th.int64)
                if edge_index.shape[1] > 0:
                    _x_cur.scatter_add_(
                        0, edge_index[1].unsqueeze(1), th.ones_like(edge_index[1].unsqueeze(1))
                    )
            elif feat == "num_heavy_atoms":  # on-the-fly: sum base_formula (H col is already 0)
                _x_cur = get_node_feats(x, node_feat_idxs[0], "base_formula").sum(
                    dim=1, keepdim=True
                )
            else:
                _x_cur = get_node_feats(x, node_feat_idxs[0], feat)
            _x.append(_x_cur)
            _node_feat_size += _x_cur.shape[1]
            # print(feat,feat_idx,_node_feat_size)
        _node_feat_idxs.append(_node_feat_size)
    if len(_x) == 0:
        _x = th.zeros(num_nodes, 0, device=device, dtype=th.int64)
    else:
        _x = th.cat(_x, dim=1)
    _node_feat_idxs = th.tensor(_node_feat_idxs, device=device, dtype=th.int64).reshape(1, -1)
    # select edge features
    _edge_index = edge_index.clone()
    _edge_attr = []
    _edge_feat_idxs = [0]
    _edge_feat_size = 0
    for feat, _ in EDGE_FEAT_TO_IDX.items():
        if feat in frag_edge_feats:
            assert edges
            _edge_attr_cur = get_edge_feats(edge_attr, edge_feat_idxs[0], feat)
            _edge_attr.append(_edge_attr_cur)
            _edge_feat_size += _edge_attr_cur.shape[1]
            # print(feat,feat_idx,_edge_feat_size)
        _edge_feat_idxs.append(_edge_feat_size)
    if len(_edge_attr) == 0:
        _edge_attr = [th.zeros(edge_attr.shape[0], 1, device=device, dtype=th.int64)]
    _edge_attr = th.cat(_edge_attr, dim=1)
    _edge_feat_idxs = th.tensor(_edge_feat_idxs, device=device, dtype=th.int64).reshape(1, -1)
    if bigraph:
        assert edges
        _edge_index_c = th.stack([_edge_index[1], _edge_index[0]], dim=0)
        assert not th.any(th.all(_edge_index == _edge_index_c, dim=0), dim=0)
        _edge_attr_c = _edge_attr.clone()
        if "complement" in frag_edge_feats:
            _edge_attr_c[:, -1] = 1 - _edge_attr_c[:, -1]
        _edge_index = th.cat([_edge_index, _edge_index_c], dim=1)
        _edge_attr = th.cat([_edge_attr, _edge_attr_c], dim=0)
    if not edges:
        assert not bigraph
        _edge_index = _edge_index[:, :0]
        _edge_attr = _edge_attr[:0, :]
    # create pyg object
    dag_pyg = pyg.data.Data(
        x=_x,
        edge_index=_edge_index,
        edge_attr=_edge_attr,
        node_feat_idxs=_node_feat_idxs,
        edge_feat_idxs=_edge_feat_idxs,
    )
    for attr_name in (
        "boundary_pair_frag_idxs",
        "boundary_pair_in_local",
        "boundary_pair_out_local",
    ):
        if hasattr(source_dag_pyg, attr_name):
            setattr(dag_pyg, attr_name, getattr(source_dag_pyg, attr_name))
    if "frag_q" in frag_node_feats:
        if gasteiger_arr is None:
            frag_q = th.zeros(num_nodes, 1, dtype=th.float32, device=device)
        else:
            cc_long = get_node_feats(x, node_feat_idxs[0], "cc").long()
            cc_binary = th_long_to_mask(cc_long)[:, : len(gasteiger_arr)].float()
            g_arr = th.as_tensor(gasteiger_arr, dtype=th.float32, device=device)
            frag_q = (cc_binary @ g_arr).unsqueeze(1)
        dag_pyg.frag_q = frag_q
    return dag_pyg
