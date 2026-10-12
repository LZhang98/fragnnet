"""Fragment-DAG construction: molecule info, the Cython enumeration (compute_frags,
multi_cut_bfs), the SMARTS/NL prepass injection, and batch generation into HDF5
(was frag_utils.py).
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops
import torch as th
import torch.nn.functional as F
import torch_geometric as pyg

import fragnnet.models.fragnnet.dag.compute_frags as compute_frags
from fragnnet.fragmentation.hydrogen_cap import compute_cc_h_cap
from fragnnet.fragmentation.smarts_prepass import (
    FRAG_RULES,
    NL_RULES,
    _apply_nl_prepass,
    _apply_smarts_prepass,
)
from fragnnet.models.fragnnet.dag.compute_frags import MASK_DTYPE
from fragnnet.models.fragnnet.dag.formula import (
    cc_bit_mask_to_atom_idx,
    cc_bitmask_to_formula_arr,
    compute_approximate_formula,
    formula_to_peak_mzs,
)
from fragnnet.models.fragnnet.dag.hashing import get_subgraph_hashes
from fragnnet.models.fragnnet.dag.layout import (
    EDGE_FEAT_DTYPE,
    FRAG_INDEX_DTYPE,
    MAX_NUM_MZS_PER_FORMULA,
    META_DATA_DTYPE,
    NODE_FEAT_DTYPE,
    compute_boundary_pair_idxs,
)
from fragnnet.models.fragnnet.dag.multi_cut_bfs import (
    compute_ccs_multi_cut as _compute_ccs_multi_cut_cy,
)
from fragnnet.models.fragnnet.dag.multi_cut_bfs import (
    get_ring_edge_mask,
)
from fragnnet.models.fragnnet.dag.storage import dump_dag_hdf5_packed, initialize_packed_dag_hdf5
from fragnnet.utils.formula_utils import (
    CANONICAL_H_IDX,
    ELEMENT_TO_IDX,
    ELEMENT_TO_VE,
    MASK_SIZE,
    MAX_NUM_EDGES,
    MAX_NUM_NODES,
    th_mask_to_long,
)
from fragnnet.utils.misc_utils import ProgressLogger, progress_wrapper
from fragnnet.utils.mol_utils import mol_from_smiles
from fragnnet.utils.parallel_utils import par_apply

if TYPE_CHECKING:
    import h5py as _h5py
    import pandas as pd

logger = logging.getLogger(__name__)

# The size limits live in core formula_utils; the Cython builder's compile-time copies must
# match them.
assert (compute_frags.MAX_NUM_NODES, compute_frags.MAX_NUM_EDGES) == (MAX_NUM_NODES, MAX_NUM_EDGES)

BOND_TYPE_TO_IDX = {
    Chem.rdchem.BondType.names["AROMATIC"]: 2,
    Chem.rdchem.BondType.names["DOUBLE"]: 2,
    Chem.rdchem.BondType.names["TRIPLE"]: 3,
    Chem.rdchem.BondType.names["SINGLE"]: 1,
}


def get_fraggen_input_arrays(
    mol_d: dict,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        mol_d (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_nodes = mol_d["atom_mask_arr"].shape[0]
    num_edges = mol_d["bond_mask_arr"].shape[0]
    assert num_nodes <= MAX_NUM_NODES, num_nodes
    assert num_edges <= MAX_NUM_EDGES, num_edges
    edges = np.zeros((MAX_NUM_EDGES, 2), dtype=np.intc)
    for bond_idx, bond in enumerate(mol_d["bonds"]):
        edges[bond_idx, 0] = bond[0]
        edges[bond_idx, 1] = bond[1]
    node_to_edge_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes, num_edges, edges)
    # print(node_to_edge_idx)
    node_mask = np.zeros((num_nodes,), dtype=MASK_DTYPE)
    node_mask[:num_nodes] = 1
    edge_mask = np.zeros((num_edges,), dtype=MASK_DTYPE)
    edge_mask[:num_edges] = 1
    return num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx


def extract_mol_info(
    smiles_or_mol: "str | Chem.rdchem.Mol", use_default_valence: bool = False
) -> dict:
    """Method to exatract mol infomation in to dict

    Args:
        smiles_or_mol (_type_): _description_
        use_default_valence (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        dict :
        mol_d["mol"]: mol object
        mol_d["num_hs"]: number of totoal Hs
        mol_d["sbond_arr"]: single bond array
        mol_d["ve_arr"]: max velance array
        mol_d["hs_arr"]: max Hs per atom arary
        mol_d["elems"]: elemns per atom
        mol_d["bonds"]: a list of from atom - to atom pairs
        mol_d["bond_mask_arr"]: defualt bond mask, this should be just 1s
        mol_d["atom_mask_arr"] : defualt atom  mask, this should be just 1s
        mol_d["atoms_to_bonds"]:
        mol_d["element_counts"] : atom count per element
    """
    if isinstance(smiles_or_mol, str):
        mol = mol_from_smiles(smiles_or_mol)
    else:
        assert isinstance(smiles_or_mol, Chem.rdchem.Mol), type(smiles_or_mol)
        mol = smiles_or_mol

    pt = Chem.GetPeriodicTable()
    # GetValenceList
    # some checks
    charge = rdmolops.GetFormalCharge(mol)
    if charge not in {0, 1}:
        raise ValueError(
            f"unsupported molecular charge {charge}; only 0 (neutral) or +1 ([M]+ cations) supported"
        )
    # enumerate atoms
    sbond_arr = []
    ve_arr = []
    hs_arr = []
    elems = []
    elem_idxs = []
    num_hs = 0
    num_atoms = 0
    num_radicals = 0
    element_counts = dict.fromkeys(ELEMENT_TO_VE.keys(), 0)

    for atom in mol.GetAtoms():
        cur_idx = atom.GetIdx()
        cur_num_hs = atom.GetTotalNumHs()
        cur_deg = atom.GetTotalDegree()
        # cur_num_bonds = atom.GetNumBonds()
        cur_element = atom.GetSymbol()
        cur_isotope = atom.GetIsotope()

        # Formulae and masses here are monoisotopic per element, so an isotope label is
        # silently dropped and every fragment m/z would be off by the label's mass defect.
        # Reject such molecules explicitly instead of emitting wrong masses.
        if cur_isotope != 0:
            raise ValueError(
                f"Molecules with isotope-labeled atoms currently not supported: "
                f"[{cur_isotope}{cur_element}] at atom index {cur_idx}"
            )

        if cur_element not in element_counts:
            raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")

        element_counts[cur_element] += 1
        elems.append(cur_element)
        elem_idxs.append(ELEMENT_TO_IDX[cur_element])
        # number of single bond need to attach to this atom to keep atom connected
        # this equals to replace all all bond to single, and count how many single bond
        sbond_arr.append(cur_deg - cur_num_hs)
        # ve_arr.append(ELEMENT_TO_VE[cur_element])

        # set valence value for each atom
        # assumption we will use current valance unless
        # use default valence flag is set to true
        # not defualt valence can be -1 for transition metals
        # on paper we should not encounter them at all
        ve_value = atom.GetTotalValence()
        # For cationic atoms with no H (e.g. [N+] in nitro, N-oxide), the
        # formal charge funds an extra heavy-atom bond rather than an H slot.
        # Subtracting FC prevents the H-cap from being inflated by 1 per such
        # atom.  The guard cur_num_hs == 0 preserves correctness for protonated
        # amines ([NH3+], [NH2+]) where the extra valence IS used for H.
        if atom.GetFormalCharge() > 0 and cur_num_hs == 0:
            ve_value -= atom.GetFormalCharge()
        if use_default_valence:
            default_valence = pt.GetDefaultValence(cur_element)
            if default_valence != -1:
                ve_value = min(ve_value, default_valence)

        ve_arr.append(ve_value)
        hs_arr.append(cur_num_hs)
        num_hs += cur_num_hs
        num_atoms += 1
        num_radicals += atom.GetNumRadicalElectrons()

    element_counts["H"] = num_hs
    assert num_radicals == 0, num_radicals
    sbond_arr = np.array(sbond_arr, dtype=np.int32)
    ve_arr = np.array(ve_arr, dtype=np.int32)
    hs_arr = np.array(hs_arr, dtype=np.int32)
    # enumerate bonds
    # Every atom maps to a (possibly empty) bond list: isolated atoms have no bonds
    # but are still valid fragment members.
    atoms_to_bonds = {atom_idx: [] for atom_idx in range(num_atoms)}
    bonds, bond_type_idxs, ring_bond_mask = [], [], []
    num_bonds = 0
    adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    # adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    for bond in mol.GetBonds():
        cur_idx = bond.GetIdx()
        from_idx = bond.GetBeginAtomIdx()
        to_idx = bond.GetEndAtomIdx()
        cur_type_idx = BOND_TYPE_TO_IDX[bond.GetBondType()]
        assert from_idx != to_idx
        adj[from_idx, to_idx] = 1

        bonds.append((from_idx, to_idx))
        bond_type_idxs.append(cur_type_idx)
        ring_bond_mask.append(1 if bond.IsInRing() else 0)

        atoms_to_bonds[from_idx].append(cur_idx)
        atoms_to_bonds[to_idx].append(cur_idx)
        num_bonds += 1
    # reshape keeps the (num_bonds, 2) contract for bond-free molecules (single atoms),
    # where np.array([]) would otherwise be 1-D and break the typed Cython buffers.
    bonds = np.array(bonds, dtype=np.int32).reshape(-1, 2)
    mol_d = {}
    mol_d["mol"] = mol
    mol_d["num_hs"] = num_hs
    mol_d["sbond_arr"] = sbond_arr
    mol_d["ve_arr"] = ve_arr
    mol_d["hs_arr"] = hs_arr
    mol_d["elems"] = elems
    mol_d["elem_idxs"] = elem_idxs
    mol_d["bonds"] = bonds
    mol_d["bond_type_idxs"] = bond_type_idxs
    mol_d["ring_bond_mask"] = np.array(ring_bond_mask, dtype=np.uint8)
    mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
    mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool)  # can be computed
    mol_d["atoms_to_bonds"] = atoms_to_bonds  # can be computed
    mol_d["element_counts"] = element_counts
    return mol_d


def compute_cc_h_floor(
    cc_atom_ids: np.ndarray,
    ve_arr: np.ndarray,
    sbond_arr: np.ndarray,
    num_radicals: int,
    bonds: np.ndarray,
    atoms_to_bonds: dict,
    bond_mask_arr: np.ndarray,
) -> int:
    """compute min amount of Hs a cc can have.

    Args:
        cc_atom_ids (np.ndarray): atom ids in the cc
        ve_arr (np.ndarray): _description_
        sbond_arr (np.ndarray): _description_
        bonds (np.ndarray): _description_
        atoms_to_bonds (dict): _description_
        bond_mask_arr (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    assert num_radicals == 0
    # if we could use update from single bond to double bond to get an electron pair
    diff_arr = np.maximum(ve_arr - sbond_arr, 0)
    # cc_atoms = list(cc_atom_ids)
    # print(diff_arr)
    # this computes a lower bound
    h_arr = np.copy(diff_arr)

    for _, atom in enumerate(cc_atom_ids):
        bond_idxs = atoms_to_bonds[atom]
        for bond_idx in bond_idxs:
            if h_arr[atom] == 0:
                break
            if not bond_mask_arr[bond_idx]:
                continue
            bond = bonds[bond_idx]
            if bond[0] == atom:
                other = bond[1]
            else:
                other = bond[0]
            # dont't form more than 3 bonds with anything!
            h_arr[atom] = max(0, h_arr[atom] - min(diff_arr[other], 2))
    # print(h_arr)
    cc_floor = sum(h_arr[atom] for atom in cc_atom_ids)
    cc_floor -= num_radicals
    cc_floor = max(cc_floor, 0)  # why can cc_floor be negative?
    return cc_floor


def _inject_smarts_pairs_into_dag(
    smarts_pairs: list[tuple[np.ndarray, np.ndarray]],
    nodes_mask_matrix: np.ndarray,
    nodes_depth_matrix: np.ndarray,
    dag_edges_matrix: np.ndarray,
    dag_frag_meta: dict,
    num_nodes: int,
    min_frag_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inject SMARTS prepass fragment pairs into a DAG produced by compute_frags.

    Appends fragment pairs from the SMARTS rearrangement prepass as direct
    children of the root node (index 0).  Prepass rules require at least 2 bond
    cuts plus 1 new bond formation, so injected fragments receive **node depth 2**
    (not 1) to reflect the two-cut distance from the root.  The root→fragment
    edge retains ``min_depth=1`` since it is introduced in the same expansion
    pass as depth-1 BFS edges.  See ``docs/reference/frag_gen_prepass.md`` for the full
    depth-semantics rationale.

    Fragments already present in the DAG are deduplicated via their byte-string
    node-mask key.  Fragments smaller than ``min_frag_atoms`` heavy atoms are
    silently dropped, consistent with the ``min_frag_atoms`` filter applied
    inside the BFS.

    Args:
        smarts_pairs: List of ``(node_mask_a, node_mask_b)`` uint8 array pairs
            returned by ``_apply_smarts_prepass``.
        nodes_mask_matrix: Shape ``(n_frags, num_nodes)`` uint8 — existing DAG
            fragment node masks, root at index 0.
        nodes_depth_matrix: Shape ``(n_frags, max_depth+1)`` uint8 — depth
            one-hot for each fragment.
        dag_edges_matrix: Shape ``(n_edges, 2)`` int64 — (parent, child) pairs.
        dag_frag_meta: Dict with keys ``reached_depth``, ``edges_min_depth``,
            ``nodes_min_depth``, ``force_stopped``.
        num_nodes: Number of atoms in the root molecule.
        min_frag_atoms: Minimum heavy-atom count for a fragment to be kept.

    Returns:
        Updated ``(nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix,
        dag_frag_meta)`` tuple with the same format as the input.
    """
    if not smarts_pairs:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    max_depth_cols = nodes_depth_matrix.shape[1]

    # Build bytes-key → fragment index lookup from existing rows.
    key_to_idx: dict[bytes, int] = {
        nodes_mask_matrix[i].tobytes(): i for i in range(len(nodes_mask_matrix))
    }
    # Build set of existing (parent, child) edges for O(1) dedup.
    existing_edges: set[tuple[int, int]] = {
        (int(dag_edges_matrix[i, 0]), int(dag_edges_matrix[i, 1]))
        for i in range(len(dag_edges_matrix))
    }

    new_masks: list[np.ndarray] = []
    new_depth_rows: list[np.ndarray] = []
    new_min_depths: list[int] = []
    new_edges: list[tuple[int, int]] = []
    new_edge_min_depths: list[int] = []

    for mask_a, mask_b in smarts_pairs:
        for mask in (mask_a, mask_b):
            if int(mask.sum()) < min_frag_atoms:
                continue
            key = mask.tobytes()
            if key not in key_to_idx:
                new_idx = len(nodes_mask_matrix) + len(new_masks)
                key_to_idx[key] = new_idx
                new_masks.append(mask.astype(MASK_DTYPE))
                depth_row = np.zeros(max_depth_cols, dtype=MASK_DTYPE)
                # Prepass rules require ≥2 bond cuts + 1 new bond from the root,
                # so the node depth label is 2 (not 1).  See docs/reference/frag_gen_prepass.md.
                depth_row[min(2, max_depth_cols - 1)] = 1
                new_depth_rows.append(depth_row)
                new_min_depths.append(2)
            frag_idx = key_to_idx[key]
            edge = (0, frag_idx)
            if edge not in existing_edges:
                existing_edges.add(edge)
                new_edges.append(edge)
                # Edge min_depth stays 1: the root→prepass edge is introduced
                # in the same expansion pass as depth-1 BFS edges.
                new_edge_min_depths.append(1)

    if not new_masks and not new_edges:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    if new_masks:
        nodes_mask_matrix = np.vstack(
            [nodes_mask_matrix] + [m.reshape(1, num_nodes) for m in new_masks]
        )
        nodes_depth_matrix = np.vstack([nodes_depth_matrix] + new_depth_rows)
        dag_frag_meta["nodes_min_depth"] = np.concatenate(
            [dag_frag_meta["nodes_min_depth"], np.array(new_min_depths, dtype=MASK_DTYPE)]
        )

    if new_edges:
        dag_edges_matrix = np.vstack([dag_edges_matrix, np.array(new_edges, dtype=np.int64)])
        dag_frag_meta["edges_min_depth"] = np.concatenate(
            [dag_frag_meta["edges_min_depth"], np.array(new_edge_min_depths, dtype=MASK_DTYPE)]
        )

    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta


def _inject_nl_masks_into_dag(
    nl_masks: list[np.ndarray],
    nodes_mask_matrix: np.ndarray,
    nodes_depth_matrix: np.ndarray,
    dag_edges_matrix: np.ndarray,
    dag_frag_meta: dict,
    num_nodes: int,
    min_frag_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inject neutral-loss product masks as depth-2 root children in the DAG.

    Each *nl_mask* represents a product formula node after a new-bond-forming
    rearrangement (SO₂ extrusion, SO₃ loss, N₂O loss, etc.).  Only the product
    is added — the neutral-loss group is not registered as a DAG node.

    Args:
        nl_masks: Product atom masks from :func:`_apply_nl_prepass`.
        nodes_mask_matrix: Shape ``(n_frags, num_nodes)`` uint8.
        nodes_depth_matrix: Shape ``(n_frags, max_depth+1)`` uint8.
        dag_edges_matrix: Shape ``(n_edges, 2)`` int64 — (parent, child) pairs.
        dag_frag_meta: Dict with keys ``reached_depth``, ``edges_min_depth``,
            ``nodes_min_depth``, ``force_stopped``.
        num_nodes: Number of atoms in the root molecule.
        min_frag_atoms: Minimum heavy-atom count for a fragment to be kept.

    Returns:
        Updated ``(nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix,
        dag_frag_meta)`` tuple.
    """
    if not nl_masks:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    max_depth_cols = nodes_depth_matrix.shape[1]
    key_to_idx: dict[bytes, int] = {
        nodes_mask_matrix[i].tobytes(): i for i in range(len(nodes_mask_matrix))
    }
    existing_edges: set[tuple[int, int]] = {
        (int(dag_edges_matrix[i, 0]), int(dag_edges_matrix[i, 1]))
        for i in range(len(dag_edges_matrix))
    }

    new_masks: list[np.ndarray] = []
    new_depth_rows: list[np.ndarray] = []
    new_min_depths: list[int] = []
    new_edges: list[tuple[int, int]] = []
    new_edge_min_depths: list[int] = []

    for mask in nl_masks:
        if int(mask.sum()) < min_frag_atoms:
            continue
        key = mask.tobytes()
        if key not in key_to_idx:
            new_idx = len(nodes_mask_matrix) + len(new_masks)
            key_to_idx[key] = new_idx
            new_masks.append(mask.astype(MASK_DTYPE))
            depth_row = np.zeros(max_depth_cols, dtype=MASK_DTYPE)
            depth_row[min(2, max_depth_cols - 1)] = 1
            new_depth_rows.append(depth_row)
            new_min_depths.append(2)
        frag_idx = key_to_idx[key]
        edge = (0, frag_idx)
        if edge not in existing_edges:
            existing_edges.add(edge)
            new_edges.append(edge)
            new_edge_min_depths.append(1)

    if not new_masks and not new_edges:
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    if new_masks:
        nodes_mask_matrix = np.vstack(
            [nodes_mask_matrix] + [m.reshape(1, num_nodes) for m in new_masks]
        )
        nodes_depth_matrix = np.vstack([nodes_depth_matrix] + new_depth_rows)
        dag_frag_meta["nodes_min_depth"] = np.concatenate(
            [dag_frag_meta["nodes_min_depth"], np.array(new_min_depths, dtype=MASK_DTYPE)]
        )

    if new_edges:
        dag_edges_matrix = np.vstack([dag_edges_matrix, np.array(new_edges, dtype=np.int64)])
        dag_frag_meta["edges_min_depth"] = np.concatenate(
            [dag_frag_meta["edges_min_depth"], np.array(new_edge_min_depths, dtype=MASK_DTYPE)]
        )

    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta


def compute_dags(
    mol_d: dict,
    max_depth: int,
    h_prior: bool,
    max_h_transfer: int,
    frag_max_time: int,
    isotopes: bool = False,
    nb_isomorphic: bool = False,
    # b_isomorphic:bool = False, # not used
    wl_max_iterations: int = -1,
    multi_cut_bfs: bool = False,
    max_cut_size: int = 2,
    smarts_prepass: bool = False,
    min_frag_atoms: int = 0,
    max_ring_cut_steps: int = 2,
    restrict_ring_open: bool = False,
) -> dict:
    """_summary_

    Args:
        mol_d (dict): _description_
        max_depth (int): _description_
        h_prior (bool): _description_
        max_h_transfer (int): _description_
        frag_max_time (int): _description_
        isotopes (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """

    assert h_prior in [True, False], h_prior
    num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = get_fraggen_input_arrays(
        mol_d
    )
    # time the recursive part
    # effectively this is infinite amount of time if None or <= 0
    if frag_max_time is None or frag_max_time <= 0:
        frag_max_time = int(1e6)

    node_mask = node_mask.astype(MASK_DTYPE)
    edge_mask = edge_mask.astype(MASK_DTYPE)
    if multi_cut_bfs:
        ring_edge_mask = get_ring_edge_mask(mol_d["mol"], num_edges)
        # Per-SSSR-ring bond groups: only rings that are part of a fused system
        # (i.e., share ≥1 bond with another ring).  Isolated single rings are
        # excluded: cut=2 on them produces the same fragments as sequential
        # single-cut BFS at the same depth budget, so they add O(k²) noise
        # candidates without contributing unique DAG nodes.
        _all_sssr = mol_d["mol"].GetRingInfo().BondRings()
        _bond_ring_count: dict[int, int] = {}
        for _r in _all_sssr:
            for _b in _r:
                _bond_ring_count[_b] = _bond_ring_count.get(_b, 0) + 1
        # Build ring_bond_groups with adaptive per-ring granularity:
        # - Isolated rings (no shared bonds): excluded (cut=2 is redundant with
        #   sequential bridge cuts at the same depth budget).
        # - Terminal rings in a fused system (exactly 1 shared bond): keep ALL
        #   bonds so cut=2 explores the full ring (simple bicyclic behaviour).
        # - Internal rings in a larger fused system (≥2 shared bonds): restrict to
        #   shared-bond pairs only, avoiding the O(ring_size²) cascade while
        #   retaining the chemically meaningful junction cuts.
        _ring_bond_groups_list: list[list[int]] = []
        for _r in _all_sssr:
            _ring_bonds = list(_r)
            _ring_shared = [_b for _b in _ring_bonds if _bond_ring_count.get(_b, 0) > 1]
            if not _ring_shared:
                continue  # isolated ring: skip
            if len(_ring_shared) == 2:
                # Ring flanked by exactly 2 other rings (linear/angular fusion):
                # restrict cut=2 to the 2 junction bonds, suppressing the
                # O(ring_size²) pair explosion while keeping junction cuts.
                _ring_bond_groups_list.append(_ring_shared)
            else:
                # Terminal ring (1 shared bond) or peri-fused ring (≥3 shared bonds):
                # keep all bonds so cut=2 explores the full ring.
                _ring_bond_groups_list.append(_ring_bonds)
        ring_bond_groups = _ring_bond_groups_list if _ring_bond_groups_list else None
        # Restrict ring-open to shared (inter-ring) bonds only: bonds that belong
        # to 2+ SSSR rings.  Opening a shared bond disconnects the fused ring
        # system, enabling new cut=2 pairs at the next BFS depth.  Opening
        # non-shared bonds adds O(n_ring_bonds) redundant BFS intermediates that
        # produce fragments already reachable via direct cut=2 paths.
        if restrict_ring_open:
            ring_open_mask = np.zeros(num_edges, dtype=MASK_DTYPE)
            for _b, _c in _bond_ring_count.items():
                if _c > 1:
                    ring_open_mask[_b] = 1
            if not ring_open_mask.any():
                ring_open_mask = None
        else:
            ring_open_mask = None
        # SMARTS prepass: compute fragment pairs in Python (RDKit API), then pass
        # as smarts_seed_pairs to the Cython BFS for injection as depth-1 children.
        smarts_seed_pairs = None
        if smarts_prepass:
            smarts_seed_pairs_tagged = _apply_smarts_prepass(
                FRAG_RULES,
                mol_d["mol"],
                edge_mask,
                edges,
                num_nodes,
                num_edges,
            )
            # Pass 3-tuples directly; the caller only needs the fragment masks.
            smarts_seed_pairs = smarts_seed_pairs_tagged
        nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
            _compute_ccs_multi_cut_cy(
                num_nodes,
                num_edges,
                node_mask,
                edges,
                edge_mask,
                node_to_edge_idx,
                max_depth,
                frag_max_time,
                ring_edge_mask=ring_edge_mask,
                max_cut_size=max_cut_size,
                smarts_seed_pairs=smarts_seed_pairs,
                ring_bond_groups=ring_bond_groups,
                min_frag_atoms=min_frag_atoms,
                max_ring_cut_steps=max_ring_cut_steps,
                ring_open_mask=ring_open_mask,
            )
        )
    else:
        nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
            compute_frags.compute_ccs(
                num_nodes,
                num_edges,
                node_mask,
                edges,
                edge_mask,
                node_to_edge_idx,
                max_depth,
                frag_max_time,
                min_frag_atoms,
            )
        )
        if smarts_prepass:
            # Post-process: inject SMARTS rearrangement fragments as root children
            # at node depth 2 (see _inject_smarts_pairs_into_dag).
            # compute_frags.compute_ccs runs unchanged
            # (preserving ring-opening via its n_ccs=1 path), and the SMARTS
            # pairs are merged in afterwards.  No further BFS expansion of the
            # SMARTS fragments is needed because their sub-fragments are already
            # covered by the BFS from the root molecule.
            smarts_pairs_tagged = _apply_smarts_prepass(
                FRAG_RULES,
                mol_d["mol"],
                edge_mask,
                edges,
                num_nodes,
                num_edges,
            )
            smarts_pairs = [(a, b) for a, b, _ in smarts_pairs_tagged]
            nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
                _inject_smarts_pairs_into_dag(
                    smarts_pairs,
                    nodes_mask_matrix,
                    nodes_depth_matrix,
                    dag_edges_matrix,
                    dag_frag_meta,
                    num_nodes,
                    min_frag_atoms,
                )
            )
    if smarts_prepass:
        # Inject rearrangement products after either BFS; their new bonds are
        # not represented in the original molecular connectivity.
        nl_products = _apply_nl_prepass(NL_RULES, mol_d["mol"], num_nodes)
        nl_masks = [mask for mask, _ in nl_products]
        nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta = (
            _inject_nl_masks_into_dag(
                nl_masks,
                nodes_mask_matrix,
                nodes_depth_matrix,
                dag_edges_matrix,
                dag_frag_meta,
                num_nodes,
                min_frag_atoms,
            )
        )

    if nb_isomorphic:
        node_nb_hashes = get_subgraph_hashes(
            nodes_mask_matrix=nodes_mask_matrix,
            elems=mol_d["elems"],
            bond_type_idxs=mol_d["bond_type_idxs"],
            edges=edges[:num_edges],
            node_to_edge_idx=node_to_edge_idx[:num_nodes],
            include_bond_type=False,
            max_iterations=wl_max_iterations,
        )
    else:
        node_nb_hashes = None

    # if b_isomorphic:
    #   node_b_hashes = get_subgraph_hashes(
    #       nodes_mask_matrix=nodes_mask_matrix,
    #       elems=mol_d["elems"],
    #       bond_type_idxs=mol_d["bond_type_idxs"],
    #       edges=edges[:num_edges],
    #       node_to_edge_idx=node_to_edge_idx[:num_nodes],
    #       include_bond_type=True)
    # else:
    #   node_b_hashes = None

    # get meta
    reached_depth = dag_frag_meta["reached_depth"]
    edges_min_depth = dag_frag_meta["edges_min_depth"]
    nodes_min_depth = dag_frag_meta["nodes_min_depth"]
    force_stop = dag_frag_meta["force_stopped"]

    # add node depth information
    # convert to a one hot encoding
    # node feature size
    depth_node_feat_size = max_depth + 1  # depth
    cc_node_feat_size = MASK_SIZE // 64  # mask
    base_formula_node_feat_size = len(ELEMENT_TO_IDX)  # base_formula
    formula_node_feat_size = (
        1 + 2 * max_h_transfer
    )  # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
    h_count_node_feat_size = formula_node_feat_size
    nb_iso_node_feat_size = 1 if nb_isomorphic else 0
    cmf_formula_node_feat_size = formula_node_feat_size  # 1+2*max_h_transfer # h_formulae_idx, at max we can have 1 + 2 * max_h_transfer different formula
    # edge feature sizes
    cc_edge_feat_size = cc_node_feat_size
    base_formula_edge_feat_size = base_formula_node_feat_size
    h_range_edge_feat_size = 2
    node_feat_shapes = [
        depth_node_feat_size,
        cc_node_feat_size,
        base_formula_node_feat_size,
        formula_node_feat_size,
        h_count_node_feat_size,
        nb_iso_node_feat_size,
        cmf_formula_node_feat_size,
    ]
    edge_feat_shapes = [cc_edge_feat_size, base_formula_edge_feat_size, h_range_edge_feat_size]

    # add node h count and element information
    hs_arr = mol_d["hs_arr"]
    elem_idxs = mol_d["elem_idxs"]
    cc_formula_list = []
    hs_arr_np = np.array(hs_arr)
    for cc_mask in nodes_mask_matrix:
        cc_h_count = np.sum(np.multiply(cc_mask, hs_arr_np))
        # Cast to uint16 before setting H count: uint8 overflows for fragments with >255 H atoms
        # (possible when MAX_NUM_NODES=128 and ~4 H per heavy atom → up to 512 H)
        cc_formula = cc_bitmask_to_formula_arr(cc_mask, elem_idxs).astype(np.uint16)
        cc_formula[ELEMENT_TO_IDX["H"]] = cc_h_count
        cc_formula_list.append(cc_formula)
    node_base_formula_matrix = np.stack(cc_formula_list, dtype=np.uint16)

    # map nodes to formulae
    formula_d_list, h_count_d_list = [], []
    formula_counts = {}
    for idx, cc_mask in enumerate(nodes_mask_matrix):
        base_formula = node_base_formula_matrix[idx]
        base_formula[ELEMENT_TO_IDX["H"]] = 0  # remove Hs
        delta_h_to_formula, delta_h_to_h_count = compute_approximate_formula(
            cc_mask, mol_d, max_h_transfer, formula_strs=True, base_formula=base_formula
        )
        formula_d_list.append(delta_h_to_formula)
        h_count_d_list.append(delta_h_to_h_count)
        formulae = list(delta_h_to_formula.values())
        for formula in formulae:
            formula_counts[formula] = formula_counts.get(formula, 0) + 1

    # map formulae indices
    formula_to_idx = {formula: idx for idx, formula in enumerate(sorted(formula_counts.keys()))}
    idx_to_formula = {idx: formula for formula, idx in formula_to_idx.items()}
    formula_idx_by_h_delta = [set() for _ in range(1 + 2 * max_h_transfer)]

    formula_idx_list, h_count_list = [], []
    for formulae_dict, h_count_dict in zip(formula_d_list, h_count_d_list):
        # idxs have value range 0 - uint32 max
        formulae_idxs = np.zeros(formula_node_feat_size, dtype=FRAG_INDEX_DTYPE)
        # h_counts have value range -1 - int16 max, we use int16 here, -1 is for invalid
        # we are not expecting h_counts to be that high anyway
        h_counts = np.zeros(h_count_node_feat_size, dtype=np.int16)
        for h_delta in formulae_dict:
            formula = formulae_dict[h_delta]
            h_count = h_count_dict[h_delta]
            formula_idx = formula_to_idx[formula]
            # h_delta [0,-1,1,-2,2,-3,3,-4,4]
            h_delta_idx = h_delta * 2 if h_delta >= 0 else (-h_delta * 2) - 1
            # sanity checks: indices/counts must be non-negative and fit uint16
            if h_delta_idx < 0 or h_delta_idx >= formula_node_feat_size:
                raise ValueError(
                    f"h_delta_idx out of bounds: {h_delta_idx}, size={formula_node_feat_size}"
                )
            if formula_idx < 0 or formula_idx > np.iinfo(FRAG_INDEX_DTYPE).max:
                raise ValueError(f"formula_idx out of {FRAG_INDEX_DTYPE} range: {formula_idx}")
            formulae_idxs[h_delta_idx] = formula_idx
            h_counts[h_delta_idx] = h_count
            formula_idx_by_h_delta[h_delta_idx].add(formula_idx)
        formula_idx_list.append(formulae_idxs)
        h_count_list.append(h_counts)

    # before stacking, ensure no negative or out-of-range values exist
    max_frag_index = np.iinfo(FRAG_INDEX_DTYPE).max
    for arr in formula_idx_list:
        if np.any(np.asarray(arr) < 0):
            raise ValueError("Negative value found in formula index list before stacking")
        if np.any(np.asarray(arr) > max_frag_index):
            raise ValueError(
                f"Value exceeds {FRAG_INDEX_DTYPE} max in formula index list before stacking"
            )

    # for arr in h_count_list:
    #   if np.any(np.asarray(arr) < 0):
    #       raise ValueError("Negative value found in h_count list before stacking")
    #   if np.any(np.asarray(arr) > max_uint16):
    #       raise ValueError("Value exceeds uint16 max in h_count list before stacking")

    node_formulae_matrix = np.stack(formula_idx_list, dtype=FRAG_INDEX_DTYPE)
    node_h_count_matrix = np.stack(h_count_list, dtype=np.int16)

    # map nodes to nb_isomorphism indices
    if nb_isomorphic and node_nb_hashes is not None:
        nb_iso_map = {hash: idx for idx, hash in enumerate(sorted(set(node_nb_hashes)))}
        node_nb_iso_idx = np.array(
            [nb_iso_map[hash] for hash in node_nb_hashes], dtype=FRAG_INDEX_DTYPE
        ).reshape(-1, 1)
        # validate nb iso indices
        if node_nb_iso_idx.size > 0 and (
            np.any(node_nb_iso_idx < 0) or np.any(node_nb_iso_idx > np.iinfo(FRAG_INDEX_DTYPE).max)
        ):
            raise ValueError(
                f"node_nb_iso_idx contains invalid values outside {FRAG_INDEX_DTYPE} range"
            )
        dag_num_nodes_nb = len(set(node_nb_hashes))
    else:
        node_nb_iso_idx = np.zeros((nodes_mask_matrix.shape[0], 0), dtype=FRAG_INDEX_DTYPE)
        dag_num_nodes_nb = -1

    assert nodes_mask_matrix.shape[1] <= MASK_SIZE
    node_cc_mask = th.as_tensor(nodes_mask_matrix, dtype=th.bool)
    node_cc_mask = F.pad(node_cc_mask, (0, MASK_SIZE - node_cc_mask.shape[1]), "constant", 0)
    node_cc_long = th_mask_to_long(node_cc_mask).type(NODE_FEAT_DTYPE)
    mol_edge_index = th.as_tensor(mol_d["bonds"].T, dtype=th.long)
    if mol_edge_index.numel() > 0:
        mol_edge_index = th.cat([mol_edge_index, mol_edge_index.flip(0)], dim=1)
    boundary_pair_frag_idxs, boundary_pair_in_local, boundary_pair_out_local = (
        compute_boundary_pair_idxs(
            node_cc_mask,
            mol_edge_index,
            th.tensor([0, num_nodes], dtype=th.long),
            th.tensor([0, node_cc_mask.shape[0]], dtype=th.long),
        )
    )
    # the order is important!
    pyg_node_feats = th.cat(
        [
            th.as_tensor(nodes_depth_matrix, dtype=NODE_FEAT_DTYPE),
            node_cc_long,
            th.as_tensor(node_base_formula_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_formulae_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_h_count_matrix, dtype=NODE_FEAT_DTYPE),
            th.as_tensor(node_nb_iso_idx, dtype=NODE_FEAT_DTYPE),
            # cmf features bit, not set here, this should set in dataloader, use 0 for NULL forumla
            th.zeros((node_cc_mask.shape[0], cmf_formula_node_feat_size), dtype=NODE_FEAT_DTYPE),
        ],
        dim=1,
    )

    # peak mzs array
    formula_peak_mzs = []
    formula_peak_probs = []
    peaks_for_element_cache = {}
    for formula, _idx in formula_to_idx.items():
        if formula == "":
            peak_mzs = np.zeros(MAX_NUM_MZS_PER_FORMULA, dtype=np.float32)
            peak_probs = np.zeros(MAX_NUM_MZS_PER_FORMULA, dtype=np.float32)
        else:
            peak_mzs, peak_probs = formula_to_peak_mzs(
                formula,
                "",
                isotopes=isotopes,
                return_probs=True,
                peaks_for_element_cache=peaks_for_element_cache,
            )
            peak_mzs, peak_probs = zip(
                *sorted(zip(peak_mzs, peak_probs), key=lambda x: x[1], reverse=True)
            )
            peak_mzs = np.array(peak_mzs[:MAX_NUM_MZS_PER_FORMULA], dtype=np.float32)
            peak_mzs = np.pad(
                peak_mzs,
                (0, MAX_NUM_MZS_PER_FORMULA - len(peak_mzs)),
                "constant",
                constant_values=0,
            )
            peak_probs = np.array(peak_probs[:MAX_NUM_MZS_PER_FORMULA], dtype=np.float32)
            peak_probs = np.pad(
                peak_probs,
                (0, MAX_NUM_MZS_PER_FORMULA - len(peak_probs)),
                "constant",
                constant_values=0,
            )
        formula_peak_mzs.append(peak_mzs)
        formula_peak_probs.append(peak_probs)

    # save as float32 for speed and lower ram usage
    # forumla and peak mz it can produce
    formula_peak_mzs = th.as_tensor(np.stack(formula_peak_mzs, axis=0), dtype=META_DATA_DTYPE)
    # forumla and peak intensity it can produce, respect to isotopes labels
    formula_peak_probs = th.as_tensor(np.stack(formula_peak_probs, axis=0), dtype=META_DATA_DTYPE)

    # add edges info
    edge_diff_cc_mask, edge_diff_formula_mask, edge_diff_h_range = [], [], []

    for edge in dag_edges_matrix:
        from_idx, to_idx = edge
        from_cc_mask = nodes_mask_matrix[from_idx]
        to_cc_mask = nodes_mask_matrix[to_idx]
        # Use signed subtraction to avoid uint8 underflow (0 - 1 = 255 in uint8)
        diff_cc_mask = np.clip(
            from_cc_mask.astype(np.int16) - to_cc_mask.astype(np.int16), 0, 1
        ).astype(MASK_DTYPE)
        # compute_cc_h_floor/cap expect atom indices, not a bitmask
        diff_atom_ids = cc_bit_mask_to_atom_idx(diff_cc_mask)

        from_formula_mask = node_base_formula_matrix[from_idx]
        to_formula_mask = node_base_formula_matrix[to_idx]
        # Use signed subtraction to avoid uint16 underflow (parent always >= child for heavy atoms,
        # but H is zeroed below, so sign safety is needed)
        diff_formula_mask = from_formula_mask.astype(np.int32) - to_formula_mask.astype(np.int32)
        diff_formula_mask[CANONICAL_H_IDX] = 0  # we don't care Hs for this

        diff_h_floor = compute_cc_h_floor(
            diff_atom_ids,
            mol_d["ve_arr"],
            mol_d["sbond_arr"],
            0,
            mol_d["bonds"],
            mol_d["atoms_to_bonds"],
            mol_d["bond_mask_arr"],
        )
        diff_h_cap = compute_cc_h_cap(diff_atom_ids, mol_d["ve_arr"], mol_d["sbond_arr"], 0)
        assert diff_h_floor <= diff_h_cap, (diff_h_floor, diff_h_cap)
        assert diff_h_floor >= 0, diff_h_floor
        diff_h_range = [diff_h_floor, diff_h_cap]

        edge_diff_cc_mask.append(diff_cc_mask)
        edge_diff_formula_mask.append(diff_formula_mask)
        edge_diff_h_range.append(diff_h_range)

    assert len(edge_diff_cc_mask) == len(edge_diff_formula_mask)
    assert (
        len(edge_diff_cc_mask) > 0 or pyg_node_feats.shape[0] == 1
    ), f"DAG has no edges, with {pyg_node_feats.shape[0]} nodes"

    # Handle case when there are no edges
    if len(edge_diff_cc_mask) == 0:
        # Create empty edge feature tensor with correct shape
        cc_edge_feat_size = MASK_SIZE // 64
        base_formula_edge_feat_size = len(ELEMENT_TO_IDX)
        h_range_edge_feat_size = 2
        total_edge_feat_size = (
            cc_edge_feat_size + base_formula_edge_feat_size + h_range_edge_feat_size
        )
        pyg_edge_feats = th.zeros((0, total_edge_feat_size), dtype=EDGE_FEAT_DTYPE)
    else:
        edge_diff_cc_mask = th.as_tensor(np.stack(edge_diff_cc_mask, axis=0), dtype=th.bool)
        edge_diff_cc_mask = F.pad(
            edge_diff_cc_mask, (0, MASK_SIZE - edge_diff_cc_mask.shape[1]), "constant", 0
        )
        edge_diff_cc_long = th_mask_to_long(edge_diff_cc_mask).type(EDGE_FEAT_DTYPE)
        edge_diff_formula_mask = th.as_tensor(
            np.stack(edge_diff_formula_mask, axis=0), dtype=EDGE_FEAT_DTYPE
        )
        edge_diff_h_range = th.as_tensor(np.stack(edge_diff_h_range, axis=0), dtype=EDGE_FEAT_DTYPE)
        pyg_edge_feats = th.cat(
            [edge_diff_cc_long, edge_diff_formula_mask, edge_diff_h_range], dim=1
        )

    # edge index need to be int64 or it will throw error where compute degree
    pyg_edge_index = th.tensor(dag_edges_matrix.T, dtype=th.int64)

    pyg_cc_g = pyg.data.Data(pyg_node_feats, pyg_edge_index, pyg_edge_feats)

    pyg_cc_g.node_feat_idxs = th.cumsum(
        th.tensor([0] + node_feat_shapes, dtype=th.long), 0
    ).reshape(1, -1)
    pyg_cc_g.edge_feat_idxs = th.cumsum(
        th.tensor([0] + edge_feat_shapes, dtype=th.long), 0
    ).reshape(1, -1)
    pyg_cc_g.boundary_pair_frag_idxs = boundary_pair_frag_idxs
    pyg_cc_g.boundary_pair_in_local = boundary_pair_in_local
    pyg_cc_g.boundary_pair_out_local = boundary_pair_out_local

    # convert to pyg
    frag_d = {}
    frag_d["max_depth"] = max_depth
    frag_d["reached_depth"] = reached_depth
    frag_d["h_prior"] = h_prior
    frag_d["max_h_transfer"] = max_h_transfer  # max number of Hs transfers
    frag_d["formula_peak_mzs"] = formula_peak_mzs  # forumla and peak mz it can produce
    frag_d["formula_peak_probs"] = (
        formula_peak_probs  # forumla and peak int it can produce, respect to isotopes labels
    )
    frag_d["idx_to_formula"] = idx_to_formula  # idx to formula str, useful for annotation
    frag_d["idx_by_h_delta"] = (
        formula_idx_by_h_delta  # formula idx for each h_delta value, used in compute_frag_peak_stats
    )
    frag_d["dag"] = pyg_cc_g

    frag_d["edges_min_depth"] = edges_min_depth
    frag_d["nodes_min_depth"] = nodes_min_depth

    # add stats here
    # we need change data type again
    # we just need to change this one place
    frag_d["dag_num_edges"] = pyg_cc_g.num_edges
    frag_d["dag_num_nodes"] = pyg_cc_g.num_nodes
    frag_d["dag_num_nodes_nb"] = dag_num_nodes_nb
    frag_d["dag_sparsity"] = (
        2 * pyg_cc_g.num_edges / (pyg_cc_g.num_nodes * (pyg_cc_g.num_nodes - 1))
        if pyg_cc_g.num_nodes > 1
        else 0.0
    )
    frag_d["formula_redundancy"] = sum([v for k, v in formula_counts.items() if k != ""]) / len(
        [k for k in formula_counts if k != ""]
    )
    frag_d["node_feature_size"] = pyg_cc_g.num_features
    frag_d["edge_feature_size"] = pyg_cc_g.num_edge_features
    frag_d["is_directed"] = pyg_cc_g.is_directed() if pyg_cc_g.num_edges > 0 else False
    frag_d["dag_num_edges_by_depth"] = {
        k: np.count_nonzero(edges_min_depth == k) for k in range(reached_depth + 1)
    }
    frag_d["dag_num_nodes_by_depth"] = {
        k: np.count_nonzero(nodes_min_depth == k) for k in range(reached_depth + 1)
    }
    frag_d["force_stopped"] = force_stop

    return frag_d


def timed_get_dags(
    mol: "str | Chem.rdchem.Mol",
    mol_id: str | int,
    max_depth: int,
    h_prior: bool,
    max_h_transfer: int,
    max_time: float,
    isotopes: bool,
    nb_isomorphic: bool,
    wl_max_iterations: int,
    multi_cut_bfs: bool = False,
    max_cut_size: int = 2,
    smarts_prepass: bool = False,
    min_frag_atoms: int = 0,
    max_ring_cut_steps: int = 2,
    restrict_ring_open: bool = False,
) -> tuple:
    try:
        mol_d = extract_mol_info(mol)
        # this maybe dangers in multi processing because of scopes
        dag_d = compute_dags(
            mol_d,
            max_depth,
            h_prior,
            max_h_transfer,
            max_time,
            isotopes,
            nb_isomorphic,
            # False, b_isomorphic, this not used
            wl_max_iterations,
            multi_cut_bfs=multi_cut_bfs,
            max_cut_size=max_cut_size,
            smarts_prepass=smarts_prepass,
            min_frag_atoms=min_frag_atoms,
            max_ring_cut_steps=max_ring_cut_steps,
            restrict_ring_open=restrict_ring_open,
        )
        return mol_id, dag_d
    except KeyboardInterrupt as e:
        # let these through
        raise e
    except Exception as e:
        # don't retry, theres a bug
        if type(mol) is not str:
            mol = Chem.MolToSmiles(mol)
        logger.exception("Non-timeout error, aborting: %r Input %s", e, mol)
    return mol_id, {}


def run_frag_gen_hdf5(
    mol_df: "pd.DataFrame",
    h5_file: "_h5py.File",
    max_depth: int,
    max_h_transfer: int,
    max_time: int,
    isotopes: bool,
    nb_isomorphic: bool,
    wl_max_iterations: int,
    multi_cut_bfs: bool,
    max_cut_size: int,
    smarts_prepass: bool,
    min_frag_atoms: int,
    max_ring_cut_steps: int = 2,
    restrict_ring_open: bool = False,
    disable_tqdm: bool = False,
) -> dict[str, list[int]]:
    """Run fragment generation for all molecules and write DAGs to an open HDF5 file.

    Iterates over rows in ``mol_df``, calls :func:`timed_get_dags` in parallel via
    :func:`~fragnnet.utils.parallel_utils.par_apply`, and writes each successfully
    fragmented molecule's DAG into ``h5_file`` using the packed HDF5 format (see
    :func:`dump_dag_hdf5_packed`).

    Args:
        mol_df: DataFrame with at least ``mol`` and ``mol_id`` columns.
        h5_file: Open writable ``h5py.File`` object; the caller is responsible
            for opening and closing the file.
        max_depth: Maximum fragmentation depth for DAG generation.
        max_h_transfer: Maximum hydrogen transfers per fragment.
        max_time: Per-molecule time budget in seconds.
        isotopes: Whether to compute isotopic peak distributions.
        nb_isomorphic: Whether to compute neighbourhood-isomorphism features.
        wl_max_iterations: Maximum Weisfeiler-Lehman iterations.
        multi_cut_bfs: Whether to use ring-aware multi-bond-cut BFS.
        max_cut_size: Maximum bonds cut per BFS step (only when ``multi_cut_bfs``).
        smarts_prepass: Whether to run the SMARTS rearrangement prepass.
        min_frag_atoms: Minimum heavy atoms per fragment.
        disable_tqdm: Suppress the tqdm progress bar.

    Returns:
        Dictionary mapping ``str(mol_id)`` to ``[num_nodes, num_edges]`` for every
        molecule that was successfully fragmented.
    """
    _log = logging.getLogger(__name__)

    mol_input_rows = (
        [
            row["mol"],
            row["mol_id"],
            max_depth,
            True,  # h_prior
            max_h_transfer,
            max_time,
            isotopes,
            nb_isomorphic,
            wl_max_iterations,
            multi_cut_bfs,
            max_cut_size,
            smarts_prepass,
            min_frag_atoms,
            max_ring_cut_steps,
            restrict_ring_open,
        ]
        for _, row in mol_df.iterrows()
    )

    from typing import cast

    frag_results_gen = par_apply(
        iter(mol_input_rows),
        timed_get_dags,
        True,
        return_as_generator=True,
    )
    if frag_results_gen is None:
        raise RuntimeError("Parallel fragment generation returned no results")
    frag_results_iter = cast("list[tuple[str, dict]]", frag_results_gen)

    initialize_packed_dag_hdf5(h5_file)
    meta_info: dict[str, list[int]] = {}
    n_total = mol_df.shape[0]

    _log.info("Executing fragment generation for %d molecules (packed HDF5 output)", n_total)
    progress = ProgressLogger(total=n_total, desc="Fragment generation", logger=_log)

    # ProgressLogger already emits the rate/ETA lines, so progress_wrapper is only used
    # for the interactive tqdm bar; its logging fallback would duplicate every line.
    frag_iter = frag_results_iter
    if not disable_tqdm:
        frag_iter = progress_wrapper(frag_results_iter, total=n_total, desc="Compute Frags")

    for fr in frag_iter:
        mol_id, dag_d = fr
        if len(dag_d) == 0:
            _log.warning("Empty DAG for mol_id=%s", mol_id)
            progress.update(is_error=True)
            continue
        meta_info[str(int(mol_id))] = [dag_d["dag"].num_nodes, dag_d["dag"].num_edges]
        dump_dag_hdf5_packed(h5_file, mol_id, dag_d)
        progress.update()
        if len(meta_info) % 1000 == 0:
            h5_file.flush()

    h5_file.flush()
    progress.finish()
    _log.info(
        "Stored %d DAGs in packed HDF5 (%d molecules failed)",
        len(meta_info),
        n_total - len(meta_info),
    )
    return meta_info
