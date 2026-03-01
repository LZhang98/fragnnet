"""Multi-bond-cut BFS fragmentation.

Drop-in augmentation of ``compute_frags.compute_ccs`` that supports
simultaneous 2-bond ring cleavage (``max_cut_size=2``), enabling retro-
Diels-Alder and other ring-opening reactions at depth 1.

The core change relative to the Cython BFS:

    Standard BFS (cut_size=1):
        For each live edge i → mask i → if num_ccs==2: add children

    Multi-cut BFS (cut_size=2):
        Same as above, PLUS:
        For each ring edge i that gave num_ccs==1 (ring bond, didn't disconnect):
            For each ring edge j > i:
                mask both i and j → if num_ccs==2: add children at same depth

Only ring+ring pairs are tried for cut_size=2.  Acyclic+anything pairs
are always redundant — an acyclic bond alone already disconnects (num_ccs==2),
so pairs containing it would just be covered by sequential BFS steps.

Usage::

    from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut, get_ring_edge_mask

    mol_d = extract_mol_info(smiles)
    ring_edge_mask = get_ring_edge_mask(mol_d["mol"], len(mol_d["bonds"]))
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)

    result = compute_ccs_multi_cut(
        num_nodes, num_edges, node_mask, edges, edge_mask, n2e,
        max_depth=3, time_limit=150,
        ring_edge_mask=ring_edge_mask,
        max_cut_size=2,
    )
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any

import numpy as np
from rdkit import Chem

MASK_DTYPE = np.uint8
MAX_NUM_NODES = 128
MAX_NUM_EDGES = 4 * MAX_NUM_NODES


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
    """Compute connected components via union-find.

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


def _mask_to_key(arr: np.ndarray) -> bytes:
    return np.asarray(arr).tobytes()


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Multi-bond-cut BFS fragmentation.

    Extends ``compute_frags.compute_ccs`` to simultaneously break up to
    ``max_cut_size`` ring bonds per fragmentation step, enabling ring
    cleavage fragments at depth 1.

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
    root_edge_key = _mask_to_key(base_edge_mask)

    cc_node_mask_dict: dict[bytes, np.ndarray] = {root_node_key: node_mask.copy()}
    cc_edge_mask_dict: dict[bytes, np.ndarray] = {root_edge_key: base_edge_mask.copy()}
    css_to_id_dict: dict[bytes, int] = {root_node_key: 0}
    ccs_depth_dict: dict[bytes, set[int]] = {root_node_key: {0}}
    has_seen_dict: dict[bytes, int] = {}
    dag_edge_dict: dict[tuple[int, int], int] = {}

    # BFS frontier: parallel lists of (node_mask_key, edge_mask_key)
    ccs: list[bytes] = [root_node_key]
    cc_edges: list[bytes] = [root_edge_key]

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
            cur_edge_key = cc_edges[node_idx]
            cur_node_mask = cc_node_mask_dict[cur_node_key]
            cur_edge_mask = cc_edge_mask_dict[cur_edge_key].copy()

            full_key = cur_node_key + b"|" + cur_edge_key
            if full_key in has_seen_dict:
                if has_seen_dict[full_key] <= current_depth:
                    continue
            has_seen_dict[full_key] = current_depth

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
            # ----------------------------------------------------------------
            def _try_cut(cut: tuple[int, ...]) -> None:
                work_mask = cur_edge_mask.copy()
                for idx in cut:
                    work_mask[idx] = 0
                cc_nodes, new_cc_edges_arr, new_num_ccs = _union_find_ccs(
                    num_nodes, cur_node_mask, edges, work_mask, num_edges
                )
                if new_num_ccs != 2:
                    return
                for c in range(2):
                    child_node_key = _mask_to_key(cc_nodes[c])
                    child_edge_key = _mask_to_key(new_cc_edges_arr[c])

                    if child_node_key not in cc_node_mask_dict:
                        cc_node_mask_dict[child_node_key] = cc_nodes[c].copy()
                        css_to_id_dict[child_node_key] = len(css_to_id_dict)

                    if child_edge_key not in cc_edge_mask_dict:
                        cc_edge_mask_dict[child_edge_key] = new_cc_edges_arr[c].copy()

                    ccs_depth_dict.setdefault(child_node_key, set()).add(current_depth + 1)
                    ccs.append(child_node_key)
                    cc_edges.append(child_edge_key)

                    if cur_node_key != child_node_key:
                        edge_pair = (
                            css_to_id_dict[cur_node_key],
                            css_to_id_dict[child_node_key],
                        )
                        if edge_pair not in dag_edge_dict:
                            dag_edge_dict[edge_pair] = current_depth + 1

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
                    if bond_to_sys.get(i) != bond_to_sys.get(j) or bond_to_sys.get(j) != bond_to_sys.get(k):
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
