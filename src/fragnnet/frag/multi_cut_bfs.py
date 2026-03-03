"""Multi-bond-cut BFS fragmentation.

Drop-in augmentation of ``compute_frags.compute_ccs`` that supports
simultaneous 2- and 3-bond ring cleavage (``max_cut_size=2`` or ``3``),
plus a SMARTS-based rearrangement prepass for reactions that require new
bond formation (1,3-elimination, acyclic ester CO₂ loss).

The core change relative to the single-bond Cython BFS:

    Standard BFS (cut_size=1):
        For each live edge i → mask i → if num_ccs==2: add children

    Multi-cut BFS (cut_size=2):
        Same as above, PLUS for ring-bond pairs in the same ring system:
            mask both i and j → if num_ccs==2: add children at same depth

    Multi-cut BFS (cut_size=3):
        Same as cut_size=2, PLUS ring-bond triples at depth=0 only (O(R³) cost).

Only ring+ring pairs/triples within the same ring system are tried.
Cross-system cuts are pruned: they can never disconnect the molecule in one
step because the acyclic linker between systems is still intact.

PPGB_MS2 reaction coverage
---------------------------
The following table cross-references all 28 PPGB_MS2 reactions against this
BFS (+ H-transfer ±max_h_transfer applied independently per fragment).

Covered reactions
~~~~~~~~~~~~~~~~~
* crf1_0/1, crf2_0/1  — alpha-cleavage + H-rearrangement
  cut=1, H-transfer(−1) on the fragment that forms the new double/triple bond.

* crf3_0/1  — retro-Diels-Alder (2 ring bonds, same ring system)
  cut=2.  Two ring sigma bonds flanking the diene system are severed simultaneously.

* crf4_0  — 1,4-H rearrangement (retro-ene)
  cut=1 (the acyclic beta bond), H-transfer(±1).

* crf5_0  — double H-rearrangement (McLafferty-type, 2 H donors)
  cut=1, H-transfer(−1) independently on EACH fragment.

* crf7_0-3  — 6-membered ring → 3 diatomic fragments
  NOT a single-step 3-fragment event.  The BFS decomposes it in 2 steps:
    depth 1: cut=2 (2 alternating ring bonds) → pair fragment + quad fragment
    depth 2: cut=1 from the quad fragment → 2 pair fragments
  All 3 diatomic nodes appear at depth ≤ 2; no 3-fragment extension required.

* crf9_0  — ring CO extrusion (ring ketone/carbonyl)
  cut=2 (both ring bonds adjacent to the carbonyl carbon).  For any ring ≥ 4
  atoms the remaining ring atoms stay connected, giving n_ccs=2.

* crf13_0/1  — CO2 loss from carboxylic acid / formate ester
  cut=1 (acyclic R–C bond), H-transfer(−1).

* crf13_2  — CO2 loss from ester (CYCLIC case: lactone)
  cut=2 (both ring bonds flanking the ester linkage).

* cmf2_0/1  — charge-migration + single bond cleavage
  cut=1; the formal charge change does not affect atom sets.

* cmf7_0  — heterolytic bond cleavage (any single bond)
  cut=1.

* cmf13_0  — aromatic ring contraction (6-ring → 5-ring + exocyclic atom)
  cut=2 (both ring bonds adjacent to the departing atom).

Not covered by BFS alone
~~~~~~~~~~~~~~~~~~~~~~~~
* crf6_0  — 8π electrocyclic ring closure (octatriene → cyclohexadiene + ethylene)
  Requires new bond formation between non-adjacent atoms.

* crf8_0/1/2  — 1,3-/1,5-/1,7-H shift (H-migration, no bond cleavage)
  Single product; pure structural isomerization of the precursor ion.
  Since BFS tracks atom connectivity only (not bond orders), H-shift isomers
  yield identical atom-set partitions to the original — sub-type 1 prepass
  is a no-op in the current model.

* crf11_0/1  — H2 loss (dehydrogenation, no bond cleavage)
  Single product, −2 Da.  Handled as a fixed neutral loss (−2.016 Da).

* crf12_0/1  — 1,3-elimination (acyclic case only)
  New bond between atoms 0 and 3.  Cyclic case IS covered by cut=2.
  Handled by SMARTS prepass (sub-type 2) — see :data:`FRAG_RULES`.

* crf13_2  — CO2 loss from acyclic ester (acyclic case only)
  New R–R' bond required.  Cyclic (lactone) case IS covered by cut=2.
  Handled by SMARTS prepass (sub-type 2) — see :data:`FRAG_RULES`.

Rearrangement prepass
~~~~~~~~~~~~~~~~~~~~~
Sub-type 2 SMARTS rules apply SMARTS matching to the root molecule, cut the
specified bonds, decompose the result into exactly two fragment atom-sets
(merging outer components when 3 CCs are produced), and inject both fragments
as depth-1 DAG leaves identical to BFS cut=1/2 children.

The new internal bond formed by the rearrangement is irrelevant for mass
prediction because only atom composition + H count matters.

  Example: crf13_2 on methyl acetate gives CO₂ + C₂H₅.  The new C–C bond
  internal to the C₂H₅ fragment is transparent at the formula level.

All rules in :data:`FRAG_RULES` use 1-based atom map numbers (never 0) to
avoid the RDKit convention that map number 0 means "unmapped".

Extended SMARTS rule catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``docs/smarts.md`` contains 72 rules across 9 categories collected from the
literature.  Many have SMARTS syntax bugs (bimolecular reactants, invalid atom
symbols, atom-type remapping).  The valid rules map as follows:

* Already covered by BFS + H-transfer: extended neutral-loss rules (nl1-nl_h2o_alc),
  amide/peptide rules, ether_alpha, nitro_loss, nglycosidic, ortho_effect.
* Already covered by cut=2: cyclic_amides (lactam ring opening).
* Handled by SMARTS prepass: crf12_0/1 acyclic, crf13_2 acyclic (sub-type 2).
* crf8_0-2 (sub-type 1 isomerization prepass): no-op — BFS is bond-order-agnostic.
* Buggy / not salvageable without redesign: bimolecular reactions, invalid atom
  symbols ([CH2CH2N+]:5 outside brackets, [N:1H2:2] dual maps), no-op charge rules.
  See ``docs/smarts.md`` for per-rule annotations.

Usage::

    from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut, get_ring_edge_mask, FRAG_RULES

    mol_d = extract_mol_info(smiles)
    ring_edge_mask = get_ring_edge_mask(mol_d["mol"], len(mol_d["bonds"]))
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)

    result = compute_ccs_multi_cut(
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e,
        max_depth=3, time_limit=150,
        ring_edge_mask=ring_edge_mask,
        max_cut_size=2,
        smarts_rules=FRAG_RULES,
        mol=mol_d["mol"],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from rdkit import Chem

MASK_DTYPE = np.uint8
MAX_NUM_NODES = 128
MAX_NUM_EDGES = 4 * MAX_NUM_NODES


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


def get_ring_edge_mask(mol: Chem.Mol, num_bonds: int) -> np.ndarray:
    """Build a uint8 array marking which bonds are ring bonds.

    Args:
        mol: RDKit molecule (same one used to build mol_d).
        num_bonds: Number of bonds in the molecule (``len(mol_d["bonds"])``).

    Returns:
        Shape ``(num_bonds,)`` uint8 array; 1 if the bond is in a ring, 0 otherwise.
    """
    ring_mask = np.zeros(num_bonds, dtype=MASK_DTYPE)
    for bond in mol.GetBonds():
        if bond.IsInRing():
            ring_mask[bond.GetIdx()] = 1
    return ring_mask


def _build_ring_sys_labels(
    ring_edge_mask: np.ndarray,
    edges: np.ndarray,
    num_edges: int,
) -> dict[int, int]:
    """Map each ring bond index to its ring-system index via union-find on shared atoms.

    Two ring bonds belong to the same ring system if they share an atom (directly or
    transitively). Cross-system bond pairs can never disconnect a molecule by themselves
    — the acyclic linker between systems must also be cut — so they are always redundant
    in the multi-cut inner loops.

    Args:
        ring_edge_mask: Shape ``(num_edges,)`` uint8 — 1 for ring bonds.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_edges: Number of real bonds (un-padded).

    Returns:
        Dict mapping ring bond index → ring-system label (0-based integer).
        Only ring bonds (where ``ring_edge_mask[i] == 1``) are included.
    """
    ring_bond_idxs = [i for i in range(num_edges) if ring_edge_mask[i] == 1]

    # atom → list of ring bond indices that touch it
    atom_to_rbonds: dict[int, list[int]] = {}
    for bi in ring_bond_idxs:
        for atom in (int(edges[bi, 0]), int(edges[bi, 1])):
            atom_to_rbonds.setdefault(atom, []).append(bi)

    # Union-find over ring bond indices
    parent = {bi: bi for bi in ring_bond_idxs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for nbrs in atom_to_rbonds.values():
        for k in range(1, len(nbrs)):
            union(nbrs[0], nbrs[k])

    # Assign contiguous system labels
    root_to_label: dict[int, int] = {}
    bond_to_sys: dict[int, int] = {}
    for bi in ring_bond_idxs:
        root = find(bi)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        bond_to_sys[bi] = root_to_label[root]

    return bond_to_sys


def _union_find_ccs(
    num_nodes: int,
    node_mask: np.ndarray,
    edges: np.ndarray,
    edge_mask: np.ndarray,
    num_edges: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute connected components via union-find (capped at 2).

    Args:
        num_nodes: Total atom count (including padding).
        node_mask: Shape ``(num_nodes,)`` uint8 — 1 for atoms in this fragment.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints (padded).
        edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — 1 for live bonds.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        cc_nodes: Shape ``(2, num_nodes)`` uint8 — atom membership per CC.
        cc_edges: Shape ``(2, num_edges)`` uint8 — edge membership per CC.
        num_ccs: Number of connected components found (0, 1, or 2).
            Returns 99 if more than 2 CCs are found (treated as invalid cut).
    """
    parent = np.arange(num_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live_atoms: list[int] = [i for i in range(num_nodes) if node_mask[i] == 1]

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

    num_ccs = len(roots)
    if num_ccs > 2:
        empty_nodes = np.zeros((2, num_nodes), dtype=MASK_DTYPE)
        empty_edges = np.zeros((2, num_edges), dtype=MASK_DTYPE)
        return empty_nodes, empty_edges, 99

    cc_nodes = np.zeros((2, num_nodes), dtype=MASK_DTYPE)
    cc_edges = np.zeros((2, num_edges), dtype=MASK_DTYPE)

    for a in live_atoms:
        cc_idx = roots[find(a)]
        cc_nodes[cc_idx, a] = 1

    for idx in range(num_edges):
        if edge_mask[idx] == 0:
            continue
        u = int(edges[idx, 0])
        if node_mask[u] == 0:
            continue
        cc_idx = roots[find(u)]
        cc_edges[cc_idx, idx] = 1

    return cc_nodes, cc_edges, num_ccs


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


def _mask_to_key(arr: np.ndarray) -> bytes:
    return np.asarray(arr).tobytes()


def _edges_from_node_mask(
    node_mask: np.ndarray,
    base_edge_mask: np.ndarray,
    edges: np.ndarray,
    num_edges: int,
) -> np.ndarray:
    """Reconstruct an edge mask from a node mask + the base (root) edge mask.

    For any BFS-generated fragment, the edge mask is uniquely determined by its
    node mask: an edge is active iff it is active in the root graph AND both of
    its endpoints belong to the fragment.  This property holds because cut bonds
    always connect atoms in *different* fragments (never internal to one fragment).

    Args:
        node_mask: Shape ``(num_nodes,)`` uint8 — atoms belonging to the fragment.
        base_edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — root-fragment edge mask
            (all real bonds active, padding zeros).
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints (padded).
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        Edge mask of shape ``(MAX_NUM_EDGES,)`` uint8 (padding entries always 0).
    """
    em = base_edge_mask.copy()
    src = edges[:num_edges, 0]
    dst = edges[:num_edges, 1]
    em[:num_edges] &= node_mask[src]
    em[:num_edges] &= node_mask[dst]
    return em


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


def compute_ccs_multi_cut(
    num_nodes: int,
    num_edges: int,
    node_mask: np.ndarray,
    edges: np.ndarray,
    edge_mask: np.ndarray,
    node_to_edge_idx: np.ndarray,
    max_depth: int,
    time_limit: int,
    *,
    ring_edge_mask: np.ndarray | None = None,
    max_cut_size: int = 2,
    smarts_rules: list[SmartsFragRule] | None = None,
    mol: Chem.Mol | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Multi-bond-cut BFS fragmentation with optional SMARTS prepass.

    Extends ``compute_frags.compute_ccs`` to simultaneously break up to
    ``max_cut_size`` ring bonds per fragmentation step, enabling ring
    cleavage fragments at depth 1.  Optionally applies a SMARTS-based
    rearrangement prepass to inject depth-1 fragments for reactions that
    require new bond formation (see :data:`FRAG_RULES`).

    Args:
        num_nodes: Number of atoms in the molecule.
        num_edges: Number of bonds.
        node_mask: Shape ``(num_nodes,)`` uint8 — active atoms (all 1s for root).
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints, zero-padded.
        edge_mask: Shape ``(num_edges,)`` uint8 — active bonds (all 1s for root).
        node_to_edge_idx: Shape ``(num_nodes, MAX_EDGE_PER_NODE)`` int32.
            Kept for interface compatibility; not used in this Python implementation.
        max_depth: Maximum BFS recursion depth.
        time_limit: Wall-clock seconds before timeout.
        ring_edge_mask: Shape ``(num_edges,)`` uint8 — 1 if bond is in a ring.
            If ``None``, multi-bond cuts are disabled (falls back to
            ``max_cut_size=1`` behaviour, equivalent to standard BFS).
        max_cut_size: Maximum simultaneous bond breaks per fragmentation step.
            ``1`` = standard single-bond BFS (same as ``compute_ccs``).
            ``2`` = also tries all pairs of ring bonds (recommended).
            ``3`` = also tries triples of ring bonds at **depth 0 only**
            (bridged polycyclics, peri-fused systems like pyrene). Triples
            are restricted to depth 0 to avoid O(R³) blowup on the growing
            frontier at deeper levels.
        smarts_rules: List of :class:`SmartsFragRule` for the rearrangement
            prepass.  If ``None`` (default), the prepass is skipped.  Pass
            :data:`FRAG_RULES` to enable the default crf12/crf13_2 rules.
        mol: RDKit molecule corresponding to the input arrays.  Required when
            ``smarts_rules`` is not ``None``; ignored otherwise.

    Returns:
        Tuple matching the output of ``compute_frags.compute_ccs``:

        - ``nodes_mask_matrix``: ``(N, num_nodes)`` uint8 — atom mask per DAG node.
        - ``nodes_depth_matrix``: ``(N, max_depth+1)`` uint8 — depth flags per node.
        - ``dag_edges_matrix``: ``(E, 2)`` int64 — DAG edges as (src_id, dst_id).
        - ``dag_frag_meta``: dict with keys ``reached_depth``, ``edges_min_depth``,
          ``nodes_min_depth``, ``force_stopped``.
    """
    assert num_nodes <= MAX_NUM_NODES, num_nodes
    assert num_edges <= MAX_NUM_EDGES, num_edges

    # Fall back to single-bond BFS if no ring information provided
    effective_max_cut = max_cut_size if ring_edge_mask is not None else 1

    # Precompute ring-system labels once — used to prune cross-system pairs/triples
    # that can never disconnect the molecule (the acyclic linker between systems must
    # also be cut, but that is handled by the single-bond loop).
    bond_to_sys: dict[int, int] = (
        _build_ring_sys_labels(ring_edge_mask, edges, num_edges)
        if ring_edge_mask is not None
        else {}
    )

    start_time = time.monotonic()

    # Work with copies so we don't mutate caller's arrays
    node_mask = np.asarray(node_mask, dtype=MASK_DTYPE).copy()

    # edge_mask and edges may be padded to MAX_NUM_EDGES; use num_edges as the
    # real length throughout — padding entries are always 0.
    base_edge_mask = np.asarray(edge_mask, dtype=MASK_DTYPE).copy()

    root_node_key = _mask_to_key(node_mask)

    cc_node_mask_dict: dict[bytes, np.ndarray] = {root_node_key: node_mask.copy()}
    css_to_id_dict: dict[bytes, int] = {root_node_key: 0}
    ccs_depth_dict: dict[bytes, set[int]] = {root_node_key: {0}}
    # has_seen_dict: node_key → minimum depth at which this fragment was processed.
    # Fragments with the same node_mask have identical edge_masks (cut bonds are
    # always inter-fragment, so edge_mask = base_edge_mask & both-endpoints-in-frag).
    # Using node_key alone is therefore sufficient for deduplication.
    has_seen_dict: dict[bytes, int] = {}
    dag_edge_dict: dict[tuple[int, int], int] = {}

    # BFS frontier: list of node_mask keys
    ccs: list[bytes] = [root_node_key]

    ccs_idx = 0
    force_stop = False
    reached_depth = 0

    for current_depth in range(max_depth):
        ccs_end = len(ccs)

        for node_idx in range(ccs_idx, ccs_end):
            if time.monotonic() - start_time > time_limit:
                force_stop = True
                break

            cur_node_key = ccs[node_idx]
            cur_node_mask = cc_node_mask_dict[cur_node_key]

            # Deduplication: skip if this fragment was already processed at an
            # equal or shallower depth (all children were already registered).
            if cur_node_key in has_seen_dict and has_seen_dict[cur_node_key] <= current_depth:
                continue
            has_seen_dict[cur_node_key] = current_depth

            # Reconstruct the live edge mask from the node mask.
            # This avoids storing a separate per-fragment edge-mask dictionary.
            cur_edge_mask = _edges_from_node_mask(
                cur_node_mask, base_edge_mask, edges, num_edges
            )

            # ----------------------------------------------------------------
            # SMARTS prepass: runs once on the root fragment at depth 0.
            # Injects depth-1 DAG children for rearrangement reactions that
            # require new bond formation (crf12_0/1, crf13_2 acyclic).
            # Added to `ccs` beyond `ccs_end` → processed at depth 1.
            # ----------------------------------------------------------------
            if (
                current_depth == 0
                and smarts_rules is not None
                and mol is not None
                and cur_node_key == root_node_key
            ):
                for mask_a, mask_b in _apply_smarts_prepass(
                    smarts_rules, mol, base_edge_mask, edges, num_nodes, num_edges
                ):
                    for child_mask in (mask_a, mask_b):
                        child_key = _mask_to_key(child_mask)
                        if child_key not in cc_node_mask_dict:
                            cc_node_mask_dict[child_key] = child_mask
                            css_to_id_dict[child_key] = len(css_to_id_dict)
                        ccs_depth_dict.setdefault(child_key, set()).add(1)
                        if child_key != root_node_key:
                            ccs.append(child_key)
                            edge_pair = (0, css_to_id_dict[child_key])
                            if edge_pair not in dag_edge_dict:
                                dag_edge_dict[edge_pair] = 1

            # Bonds alive in this fragment
            live_edges = [i for i in range(num_edges) if cur_edge_mask[i] == 1]
            if not live_edges:
                continue

            ring_edges: list[int] = (
                [i for i in live_edges if ring_edge_mask[i] == 1]
                if ring_edge_mask is not None
                else []
            )

            # ----------------------------------------------------------------
            # Inner helper: try a cut set, register children if num_ccs == 2
            # Default args bind loop variables at definition time (B023).
            # ----------------------------------------------------------------
            def _try_cut(
                cut: tuple[int, ...],
                _em: np.ndarray = cur_edge_mask,
                _nm: np.ndarray = cur_node_mask,
                _nk: bytes = cur_node_key,
                _depth: int = current_depth,
                _id_dict: dict[bytes, int] = css_to_id_dict,
                _depth_dict: dict[bytes, set[int]] = ccs_depth_dict,
            ) -> None:
                work_mask = _em.copy()
                for idx in cut:
                    work_mask[idx] = 0
                cc_nodes, _, new_num_ccs = _union_find_ccs(
                    num_nodes, _nm, edges, work_mask, num_edges
                )
                if new_num_ccs != 2:
                    return
                for c in range(2):
                    child_node_key = _mask_to_key(cc_nodes[c])

                    if child_node_key not in cc_node_mask_dict:
                        cc_node_mask_dict[child_node_key] = cc_nodes[c].copy()
                        _id_dict[child_node_key] = len(_id_dict)

                    _depth_dict.setdefault(child_node_key, set()).add(_depth + 1)
                    ccs.append(child_node_key)

                    if _nk != child_node_key:
                        edge_pair = (
                            _id_dict[_nk],
                            _id_dict[child_node_key],
                        )
                        if edge_pair not in dag_edge_dict:
                            dag_edge_dict[edge_pair] = _depth + 1

            # ----------------------------------------------------------------
            # cut_size = 1: try every single bond
            # Track which single cuts disconnected — pairs containing those
            # bonds are redundant (equivalent to sequential BFS steps).
            # ----------------------------------------------------------------
            single_cut_disconnected: set[int] = set()

            for i in live_edges:
                cur_edge_mask[i] = 0
                _, _, n1 = _union_find_ccs(
                    num_nodes, cur_node_mask, edges, cur_edge_mask, num_edges
                )
                cur_edge_mask[i] = 1

                if n1 == 2:
                    _try_cut((i,))
                    single_cut_disconnected.add(i)

            reached_depth = current_depth + 1

            # ----------------------------------------------------------------
            # cut_size = 2: pairs of ring bonds
            # Only tried when the first bond alone did NOT disconnect.
            # Cross-system pairs are skipped — they can never disconnect the
            # molecule without also cutting the acyclic linker between systems.
            # ----------------------------------------------------------------
            if effective_max_cut >= 2 and len(ring_edges) >= 2:
                for i, j in combinations(ring_edges, 2):
                    if i in single_cut_disconnected or j in single_cut_disconnected:
                        continue  # redundant — covered by sequential steps
                    if bond_to_sys.get(i) != bond_to_sys.get(j):
                        continue  # cross-system pair — guaranteed num_ccs != 2
                    _try_cut((i, j))

            # ----------------------------------------------------------------
            # cut_size = 3: triples of ring bonds (bridged bicyclics, rare)
            # Restricted to depth 0 only — O(R³) triples on a large frontier
            # at depth 2+ causes combinatorial blowup and timeout.
            # ----------------------------------------------------------------
            if effective_max_cut >= 3 and current_depth == 0 and len(ring_edges) >= 3:
                for i, j, k in combinations(ring_edges, 3):
                    if any(x in single_cut_disconnected for x in (i, j, k)):
                        continue
                    if bond_to_sys.get(i) != bond_to_sys.get(j) or bond_to_sys.get(
                        j
                    ) != bond_to_sys.get(k):
                        continue  # cross-system triple — guaranteed num_ccs != 2
                    _try_cut((i, j, k))

        if force_stop:
            reached_depth = current_depth - 1 if current_depth > 0 else 0
            filtered: dict[bytes, set[int]] = {
                k: {d for d in v if d <= reached_depth} for k, v in ccs_depth_dict.items()
            }
            ccs_depth_dict = {k: v for k, v in filtered.items() if v}
            css_to_id_dict = {k: css_to_id_dict[k] for k in ccs_depth_dict}
            id_to_css = {v: k for k, v in css_to_id_dict.items()}
            dag_edge_dict = {
                (s, d): dep
                for (s, d), dep in dag_edge_dict.items()
                if dep <= reached_depth and s in id_to_css and d in id_to_css
            }
            break

        ccs_idx = ccs_end

    # ----------------------------------------------------------------
    # Build output matrices — same format as compute_frags.compute_ccs
    # ----------------------------------------------------------------
    num_un_ccs = len(css_to_id_dict)
    ordered_ccs: list[bytes] = [b""] * num_un_ccs
    for cc, idx in css_to_id_dict.items():
        ordered_ccs[idx] = cc

    nodes_mask_matrix = np.stack(
        [np.asarray(cc_node_mask_dict[cc]) for cc in ordered_ccs],
        dtype=MASK_DTYPE,
    )
    nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1), dtype=MASK_DTYPE)
    nodes_min_depth = np.zeros(num_un_ccs, dtype=MASK_DTYPE)

    for idx, cc in enumerate(ordered_ccs):
        for d in ccs_depth_dict[cc]:
            nodes_depth_matrix[idx, d] = 1
        nodes_min_depth[idx] = min(ccs_depth_dict[cc])

    if dag_edge_dict:
        dag_edges_matrix = np.array(list(dag_edge_dict.keys()), dtype=np.int64)
        edges_min_depth = np.array(list(dag_edge_dict.values()), dtype=MASK_DTYPE)
    else:
        dag_edges_matrix = np.empty((0, 2), dtype=np.int64)
        edges_min_depth = np.empty(0, dtype=MASK_DTYPE)

    dag_frag_meta: dict[str, Any] = {
        "reached_depth": reached_depth,
        "edges_min_depth": edges_min_depth,
        "nodes_min_depth": nodes_min_depth,
        "force_stopped": force_stop,
    }
    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta
