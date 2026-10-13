"""The fragment-DAG layout: index and feature dtypes, size constants, the node and edge
feature columns and their accessors, and the boundary masks the model reads.
"""

import numpy as np
import torch as th

# Fragment index dtype (change here to update everywhere)
# uint32 supports up to ~4B unique formulas per molecule (uint16 max is 65535)
FRAG_INDEX_DTYPE = np.uint32

MAX_H_TRANSFER = 4
MAX_NUM_MZS_PER_FORMULA = 5

NODE_FEAT_DTYPE = th.int64
EDGE_FEAT_DTYPE = th.int64
META_DATA_DTYPE = th.float32

NODE_FEAT_TO_IDX = {
    "depth": 0,
    "cc": 1,
    "base_formula": 2,
    "h_formulae_idx": 3,
    "h_counts": 4,
    "nb_iso_idx": 5,
    "cmf_h_formulae_idx": 6,
    "out_degree": 7,
    "in_degree": 8,
    "num_heavy_atoms": 9,
}

EDGE_FEAT_TO_IDX = {"cc": 0, "base_formula": 1, "h_range": 2, "complement": 3}


def get_node_feats(node_feats: th.Tensor, node_feat_idxs: th.Tensor, key: str) -> th.Tensor:
    """get node features by key used for pyg

    Args:
        node_feats (_type_): _description_
        node_feat_idxs (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """

    node_feat_idx = NODE_FEAT_TO_IDX[key]
    node_feats = node_feats[:, node_feat_idxs[node_feat_idx] : node_feat_idxs[node_feat_idx + 1]]
    # print(f"get_node_feats, node_feats tensor shape: {node_feats.shape}, num nodes: {len(node_feats)}, feature name: {key}" )
    return node_feats


def get_edge_feats(edge_feats: th.Tensor, edge_feat_idxs: int, key: str) -> th.Tensor:
    """get edege feats used for pyg

    Args:
        edge_feats (_type_): _description_
        edge_feat_idxs (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """

    edge_feat_idx = EDGE_FEAT_TO_IDX[key]
    edge_feats = edge_feats[:, edge_feat_idxs[edge_feat_idx] : edge_feat_idxs[edge_feat_idx + 1]]
    return edge_feats


def compute_boundary_mask(
    frag_node_mask: th.Tensor,
    mol_edge_index: th.Tensor,
    mol_num_nodes: th.Tensor,
    frag_num_nodes: th.Tensor,
) -> th.Tensor:
    """Compute boundary atom mask for each fragment on-the-fly.

    A boundary atom is one that belongs to the fragment AND is adjacent to at
    least one atom that does not (i.e. it sits at a cut bond).  The root
    fragment (full precursor) always produces an all-zero row.

    Args:
        frag_node_mask: Bool tensor of shape ``(F, mask_size)`` with LOCAL atom
            indices per molecule (as returned by ``th_long_to_mask`` on the cc
            node feature).
        mol_edge_index: Long tensor of shape ``(2, E)`` with GLOBAL atom indices
            across the batch (standard PyG format).
        mol_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative atom counts
            per molecule in the batch (prefix sum, starts at 0).
        frag_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative fragment
            counts per molecule in the batch (prefix sum, starts at 0).

    Returns:
        Bool tensor of shape ``(F, mask_size)`` where ``True`` marks boundary
        atoms for each fragment.
    """
    num_frags_total = frag_node_mask.shape[0]
    mask_size = frag_node_mask.shape[1]
    device = frag_node_mask.device

    u_global = mol_edge_index[0]
    v_global = mol_edge_index[1]
    edge_mol_idx = th.bucketize(u_global, mol_num_nodes, right=True) - 1

    boundary_mask = th.zeros(num_frags_total, mask_size, dtype=th.bool, device=device)
    num_mols = int(mol_num_nodes.shape[0]) - 1
    for m in range(num_mols):
        f_start = int(frag_num_nodes[m])
        f_end = int(frag_num_nodes[m + 1])
        if f_start == f_end:
            continue
        edge_sel = edge_mol_idx == m
        if not edge_sel.any():
            continue
        atom_off = int(mol_num_nodes[m])
        u_loc = u_global[edge_sel] - atom_off
        v_loc = v_global[edge_sel] - atom_off
        valid_e = (u_loc >= 0) & (u_loc < mask_size) & (v_loc >= 0) & (v_loc < mask_size)
        u_loc, v_loc = u_loc[valid_e], v_loc[valid_e]

        frags_m = frag_node_mask[f_start:f_end]  # (F_m, mask_size)
        f_m = f_end - f_start
        u_in = frags_m[:, u_loc].float()
        v_in = frags_m[:, v_loc].float()
        bm = th.zeros(f_m, mask_size, device=device)
        bm.scatter_add_(1, u_loc.unsqueeze(0).expand(f_m, -1), u_in * (1 - v_in))
        bm.scatter_add_(1, v_loc.unsqueeze(0).expand(f_m, -1), v_in * (1 - u_in))
        boundary_mask[f_start:f_end] = bm > 0

    return boundary_mask


def compute_boundary_pair_idxs(
    frag_node_mask: th.Tensor,
    mol_edge_index: th.Tensor,
    mol_num_nodes: th.Tensor,
    frag_num_nodes: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Compute boundary pair indices for each fragment on-the-fly.

    For each cut bond ``(u_in, v_out)`` where ``u`` is inside the fragment and
    ``v`` is outside (or vice versa), returns the global fragment index and the
    global atom indices of both endpoints.  The root fragment (full precursor)
    always produces zero pairs.

    Args:
        frag_node_mask: Bool tensor of shape ``(F, mask_size)`` with LOCAL atom
            indices per molecule (as returned by ``th_long_to_mask`` on the cc
            node feature).
        mol_edge_index: Long tensor of shape ``(2, E)`` with GLOBAL atom indices
            across the batch (standard PyG format).
        mol_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative atom counts
            per molecule in the batch (prefix sum, starts at 0).
        frag_num_nodes: Long tensor of shape ``(B+1,)`` — cumulative fragment
            counts per molecule in the batch (prefix sum, starts at 0).

    Returns:
        Tuple of three long tensors of shape ``(N_pairs,)``:

        - ``pair_frag_idxs``: global fragment index for each pair.
        - ``pair_in_global``: global atom index of the inside atom.
        - ``pair_out_global``: global atom index of the outside atom.

        All three tensors are empty (numel == 0) when no cut bonds exist.
    """
    device = frag_node_mask.device
    mask_size = frag_node_mask.shape[1]

    u_global = mol_edge_index[0]
    v_global = mol_edge_index[1]
    edge_mol_idx = th.bucketize(u_global, mol_num_nodes, right=True) - 1

    num_mols = int(mol_num_nodes.shape[0]) - 1

    all_pair_frag: list[th.Tensor] = []
    all_pair_in: list[th.Tensor] = []
    all_pair_out: list[th.Tensor] = []

    for m in range(num_mols):
        f_start = int(frag_num_nodes[m])
        f_end = int(frag_num_nodes[m + 1])
        if f_start == f_end:
            continue
        edge_sel = edge_mol_idx == m
        if not edge_sel.any():
            continue
        atom_off = int(mol_num_nodes[m])
        u_loc = u_global[edge_sel] - atom_off
        v_loc = v_global[edge_sel] - atom_off
        valid_e = (u_loc >= 0) & (u_loc < mask_size) & (v_loc >= 0) & (v_loc < mask_size)
        u_loc, v_loc = u_loc[valid_e], v_loc[valid_e]
        u_glob_m = u_global[edge_sel][valid_e]
        v_glob_m = v_global[edge_sel][valid_e]

        frags_m = frag_node_mask[f_start:f_end]  # (F_m, mask_size)

        u_in = frags_m[:, u_loc]  # (F_m, E_m) bool
        v_in = frags_m[:, v_loc]  # (F_m, E_m) bool

        # cut bonds: u inside, v outside
        cut_uv = u_in & ~v_in  # (F_m, E_m)
        if cut_uv.any():
            idxs = th.nonzero(cut_uv)  # (N, 2): [frag_local, edge_local]
            all_pair_frag.append(idxs[:, 0] + f_start)
            all_pair_in.append(u_glob_m[idxs[:, 1]])
            all_pair_out.append(v_glob_m[idxs[:, 1]])

        # cut bonds: v inside, u outside
        cut_vu = v_in & ~u_in  # (F_m, E_m)
        if cut_vu.any():
            idxs = th.nonzero(cut_vu)  # (N, 2): [frag_local, edge_local]
            all_pair_frag.append(idxs[:, 0] + f_start)
            all_pair_in.append(v_glob_m[idxs[:, 1]])
            all_pair_out.append(u_glob_m[idxs[:, 1]])

    if not all_pair_frag:
        empty = th.zeros(0, dtype=th.long, device=device)
        return empty, empty, empty

    return th.cat(all_pair_frag), th.cat(all_pair_in), th.cat(all_pair_out)
