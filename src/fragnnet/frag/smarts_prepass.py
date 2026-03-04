"""SMARTS-based rearrangement prepass for multi-bond BFS fragmentation.

Contains :class:`SmartsFragRule`, :data:`FRAG_RULES`, and
:func:`_apply_smarts_prepass` — the pure-Python prepass for reactions that
require new bond formation (1,3-elimination, acyclic ester CO₂ loss).

Kept in a separate module so the Cython ``.so`` can shadow ``multi_cut_bfs``
for the BFS entry-point while this prepass remains importable without any
``importlib`` hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

MASK_DTYPE = np.uint8


@dataclass
class SmartsFragRule:
    """SMARTS-based fragmentation rule for the rearrangement prepass.

    Encodes a 2-fragment reaction that cannot be described as a simple bond cut
    (e.g., 1,3-elimination, ester CO₂ loss) by specifying which bonds to cut and
    how to reassemble 3-component results into 2 fragments.

    All atom map numbers must be **1-based** (never 0).  RDKit treats map number 0
    as "no map", so :0 atoms cannot be retrieved by ``GetAtomMapNum()``.

    Attributes:
        name: Rule identifier for logging and debugging.
        reactant_smarts: SMARTS pattern for the reactant substructure only
            (no ">>" product part).  All mapped atoms use 1-based map numbers.
        cut_bond_map_pairs: Pairs of 1-based atom map numbers defining bonds to
            sever.  Each pair ``(a, b)`` identifies the bond between the matched
            atoms carrying map numbers *a* and *b*.
        merge_outer_map_nums: When cutting produces 3 CCs (n_ccs==3), the map
            numbers of the two "outer" atoms whose CCs are merged into one
            fragment.  Empty list discards n_ccs==3 results.
        filter_bond_map_nums: Optional pair of map numbers.  If the matched atoms
            are directly bonded in the original graph the rule is skipped — used
            to exclude cyclic cases already covered by BFS cut=2.

    Example:
        >>> rule = SmartsFragRule(
        ...     name="crf12_0",
        ...     reactant_smarts="[*+0:1][*+0:2][*+0:3][*+0:4]",
        ...     cut_bond_map_pairs=[(1, 2), (3, 4)],
        ...     merge_outer_map_nums=[1, 4],
        ...     filter_bond_map_nums=(1, 4),
        ... )
    """

    name: str
    reactant_smarts: str
    cut_bond_map_pairs: list[tuple[int, int]]
    merge_outer_map_nums: list[int] = field(default_factory=list)
    filter_bond_map_nums: tuple[int, int] | None = None


FRAG_RULES: list[SmartsFragRule] = [
    # crf12_0: acyclic 1,3-elimination (single bonds on both sides)
    # A–B–C–D  →  A·D (new bond) + B=C
    # Cut bonds A-B and C-D; outer atoms A,D merge; inner B,C stay together.
    # Skip if A and D are already directly bonded (4-membered ring → BFS cut=2).
    SmartsFragRule(
        name="crf12_0",
        reactant_smarts="[*+0:1][*+0:2][*+0:3][*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
    ),
    # crf12_1: acyclic 1,3-elimination (double bond between inner atoms)
    # A–B=C–D  →  A·D (new bond) + B≡C
    SmartsFragRule(
        name="crf12_1",
        reactant_smarts="[*+0:1][*+0:2]=[*+0:3][*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
    ),
    # crf13_2: CO2 loss from acyclic ester
    # R–C(=O)–O–R'  →  CO2  +  R·R' (new R-R' bond)
    # Cut bonds R-C and O-R'; outer R,R' merge; inner C,=O,O stay together (CO2).
    # Skip if R and R' are already directly bonded (lactone → BFS cut=2).
    SmartsFragRule(
        name="crf13_2",
        reactant_smarts="[*+0:1][C+0:2](=[O+0:3])[O+0:4][*+0:5]",
        cut_bond_map_pairs=[(1, 2), (4, 5)],
        merge_outer_map_nums=[1, 5],
        filter_bond_map_nums=(1, 5),
    ),
]


def _union_find_ccs_all(
    num_nodes: int,
    node_mask: np.ndarray,
    edges: np.ndarray,
    edge_mask: np.ndarray,
    num_edges: int,
) -> list[np.ndarray]:
    """Compute all connected components without a cap on their count.

    Used by the SMARTS prepass where reactions may produce 3 components
    (e.g., 1,3-elimination splits A-B-C-D into {A}, {B,C}, {D}).

    Args:
        num_nodes: Total atom count.
        node_mask: Shape ``(num_nodes,)`` uint8 — active atoms.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — active bonds.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        List of node-mask arrays (shape ``(num_nodes,)`` uint8), one per CC.
        Empty list if no active atoms.
    """
    parent = np.arange(num_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live_atoms = [i for i in range(num_nodes) if node_mask[i] == 1]
    if not live_atoms:
        return []

    for idx in range(num_edges):
        if edge_mask[idx] == 0:
            continue
        u, v = int(edges[idx, 0]), int(edges[idx, 1])
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    roots: dict[int, int] = {}
    for a in live_atoms:
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)

    cc_masks = [np.zeros(num_nodes, dtype=MASK_DTYPE) for _ in range(len(roots))]
    for a in live_atoms:
        cc_masks[roots[find(a)]][a] = 1

    return cc_masks


def _find_bond_idx(atom_a: int, atom_b: int, edges: np.ndarray, num_edges: int) -> int:
    """Find the edge-array index for the bond between atom_a and atom_b.

    Args:
        atom_a: First atom index.
        atom_b: Second atom index.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        Edge index (0-based) if found, ``-1`` otherwise.
    """
    src = edges[:num_edges, 0]
    dst = edges[:num_edges, 1]
    mask = ((src == atom_a) & (dst == atom_b)) | ((src == atom_b) & (dst == atom_a))
    idxs = np.where(mask)[0]
    return int(idxs[0]) if len(idxs) > 0 else -1


def _apply_smarts_prepass(
    rules: list[SmartsFragRule],
    mol: Chem.Mol,
    base_edge_mask: np.ndarray,
    edges: np.ndarray,
    num_nodes: int,
    num_edges: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply SMARTS fragmentation rules to the root molecule.

    For each rule, matches the reactant SMARTS against *mol*, cuts the specified
    bonds, and decomposes the result into exactly two fragment node-masks.
    When a cut produces 3 components (e.g., 1,3-elimination), the two "outer"
    components (containing ``merge_outer_map_nums`` atoms) are merged into one
    fragment.

    Only the root (un-fragmented) molecule is processed; the resulting fragment
    pairs are injected as depth-1 children in the DAG.

    Args:
        rules: List of :class:`SmartsFragRule` to apply.
        mol: RDKit molecule (same atoms/indices used to build the edge arrays).
        base_edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — root edge mask (all
            real bonds active).
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_nodes: Number of atoms.
        num_edges: Number of real bonds.

    Returns:
        List of ``(node_mask_a, node_mask_b)`` uint8 array pairs, one per valid
        match.  Duplicate fragment pairs are **not** deduplicated here; the BFS
        ``css_to_id_dict`` handles deduplication by node-mask key.
    """
    root_node_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
    results: list[tuple[np.ndarray, np.ndarray]] = []

    for rule in rules:
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        if patt is None:
            continue

        # Build: query atom index → atom map number (skip unmapped atoms, i.e. map==0)
        q_idx_to_map: dict[int, int] = {}
        for q_idx in range(patt.GetNumAtoms()):
            map_num = patt.GetAtomWithIdx(q_idx).GetAtomMapNum()
            if map_num != 0:
                q_idx_to_map[q_idx] = map_num

        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            # match[q_idx] = mol atom index for query atom q_idx
            map_num_to_mol: dict[int, int] = {
                map_num: match[q_idx] for q_idx, map_num in q_idx_to_map.items()
            }

            # Skip cyclic cases already covered by BFS cut=2
            if rule.filter_bond_map_nums is not None:
                fa, fb = rule.filter_bond_map_nums
                mol_a = map_num_to_mol.get(fa)
                mol_b = map_num_to_mol.get(fb)
                if mol_a is not None and mol_b is not None:
                    bond_idx = _find_bond_idx(mol_a, mol_b, edges, num_edges)
                    if bond_idx != -1 and base_edge_mask[bond_idx] == 1:
                        continue  # atoms directly bonded → cyclic; covered by cut=2

            # Cut the specified bonds in a temporary mask
            work_mask = base_edge_mask.copy()
            valid = True
            for map_a, map_b in rule.cut_bond_map_pairs:
                mol_a = map_num_to_mol.get(map_a)
                mol_b = map_num_to_mol.get(map_b)
                if mol_a is None or mol_b is None:
                    valid = False
                    break
                bond_idx = _find_bond_idx(mol_a, mol_b, edges, num_edges)
                if bond_idx == -1:
                    valid = False
                    break
                work_mask[bond_idx] = 0
            if not valid:
                continue

            cc_masks = _union_find_ccs_all(
                num_nodes, root_node_mask, edges, work_mask, num_edges
            )
            n_ccs = len(cc_masks)

            if n_ccs == 2:
                results.append((cc_masks[0], cc_masks[1]))
            elif n_ccs == 3 and rule.merge_outer_map_nums:
                # Identify which CCs contain the "outer" atoms
                outer_cc_set: set[int] = set()
                for map_num in rule.merge_outer_map_nums:
                    mol_idx = map_num_to_mol.get(map_num)
                    if mol_idx is not None:
                        for cc_idx, cc_mask in enumerate(cc_masks):
                            if cc_mask[mol_idx] == 1:
                                outer_cc_set.add(cc_idx)
                                break
                inner_cc_set = set(range(n_ccs)) - outer_cc_set
                if len(outer_cc_set) == 2 and len(inner_cc_set) == 1:
                    outer_mask = np.zeros(num_nodes, dtype=MASK_DTYPE)
                    for cc_idx in outer_cc_set:
                        outer_mask |= cc_masks[cc_idx]
                    inner_mask = cc_masks[next(iter(inner_cc_set))]
                    results.append((outer_mask, inner_mask))

    return results
