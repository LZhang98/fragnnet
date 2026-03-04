# cython: profile=False
# cython: linetrace=False
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""Multi-bond-cut BFS fragmentation — Cython implementation.

Cython port of ``multi_cut_bfs.py``; identical public interface, faster execution.
See ``multi_cut_bfs.py`` for the full PPGB_MS2 reaction coverage table and design
rationale.

Key implementation improvements over the Python version
-------------------------------------------------------
- ``_cc_uf``: union-find CC using C stack arrays — zero heap allocation per call.
- ``bond_to_sys_mv``: ``int[::1]`` typed array for O(1) ring-system lookup vs dict.
- ``disconnected_mv``: ``char[::1]`` bit flags (avoid Python ``set`` overhead).
- Single CC call per candidate cut (Python version called ``_union_find_ccs`` twice).
- In-place mask modify/restore — no extra array copy per pair/triple.
- Pre-allocated working buffers: bytes keys ARE the mask data; masks are reconstructed
  into fixed C buffers via ``PyBytes_AS_STRING + memcpy``, eliminating all per-node
  ``malloc``/``free`` and dict storage of mask arrays.
- ``PyBytes_FromStringAndSize`` for child key creation — direct C API, no buffer
  protocol overhead from ``bytes(memoryview_slice)``.
- ``seen_set`` (plain ``set``) replaces depth-tracking dict: BFS processes depths
  strictly in order so once-seen → always-skip is correct.
"""

import numpy as np

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.string cimport memcpy, memset
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from rdkit import Chem

cimport numpy as cnp

MASK_DTYPE = np.uint8
MAX_NUM_NODES = 128
MAX_NUM_EDGES = 4 * MAX_NUM_NODES   # 512

# ---------------------------------------------------------------------------
# Time helper (same as compute_frags.pyx)
# ---------------------------------------------------------------------------

cdef long _get_time() nogil:
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec


# ---------------------------------------------------------------------------
# Union-Find find with path-halving (operates on a C int pointer).
# ---------------------------------------------------------------------------

cdef inline int _uf_find(int* parent, int x) nogil:
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path halving
        x = parent[x]
    return x


# ---------------------------------------------------------------------------
# Zero scratch arrays before each CC call.
#
# _zero_scratch_full: memset-based full clear (used for first call per BFS node).
# _zero_scratch_live: selective clear — only live nodes/edges (subsequent calls).
# ---------------------------------------------------------------------------

cdef inline void _zero_scratch_full(
    int num_nodes,
    int num_edges,
    char[:, ::1] cc_nodes_s,
    char[:, ::1] cc_edges_s,
) nogil:
    memset(&cc_nodes_s[0, 0], 0, 2 * num_nodes)
    memset(&cc_edges_s[0, 0], 0, 2 * num_edges)


cdef inline void _zero_scratch_live(
    int n_live_nodes,
    int n_live_edges,
    int* live_node_arr,
    int* live_edge_arr,
    char[:, ::1] cc_nodes_s,
    char[:, ::1] cc_edges_s,
) nogil:
    """Reset only entries that _cc_uf can write — avoids clearing unvisited slots."""
    cdef int idx
    for idx in range(n_live_nodes):
        cc_nodes_s[0, live_node_arr[idx]] = 0
        cc_nodes_s[1, live_node_arr[idx]] = 0
    for idx in range(n_live_edges):
        cc_edges_s[0, live_edge_arr[idx]] = 0
        cc_edges_s[1, live_edge_arr[idx]] = 0


# ---------------------------------------------------------------------------
# Connected components via union-find.
#
# Writes results into pre-allocated, caller-zeroed scratch arrays.
# Returns num_ccs: 0, 1, 2, or 99 (> 2 components → bail early).
#
# Uses C stack arrays for the union-find structure — no heap allocation.
# ---------------------------------------------------------------------------

cdef int _cc_uf(
    int num_nodes,
    int num_edges,
    char[::1] node_mask,
    int[:, ::1] edges,
    char[::1] edge_mask,
    char[:, ::1] cc_nodes_out,   # (2, num_nodes) — must be zeroed by caller
    char[:, ::1] cc_edges_out,   # (2, num_edges) — must be zeroed by caller
) nogil:
    # Stack-allocated union-find arrays (MAX_NUM_NODES = 128)
    cdef int parent[128]
    # cc_root[atom_root] → CC index assigned to that root (-1 = not seen yet)
    cdef int cc_root[128]
    cdef int i, u, v, pu, pv, r, num_ccs, cc_idx

    for i in range(num_nodes):
        parent[i] = i
        cc_root[i] = -1

    # Union connected atoms via live edges
    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        v = edges[i, 1]
        pu = _uf_find(parent, u)
        pv = _uf_find(parent, v)
        if pu != pv:
            parent[pu] = pv

    # Assign CC indices to live atoms and populate cc_nodes_out
    num_ccs = 0
    for i in range(num_nodes):
        if node_mask[i] == 0:
            continue
        r = _uf_find(parent, i)
        if cc_root[r] == -1:
            if num_ccs == 2:
                return 99   # > 2 connected components → invalid cut
            cc_root[r] = num_ccs
            num_ccs += 1
        cc_nodes_out[cc_root[r], i] = 1

    # Assign live edges to their CC and populate cc_edges_out
    for i in range(num_edges):
        if edge_mask[i] == 0:
            continue
        u = edges[i, 0]
        if node_mask[u] == 0:
            continue
        r = _uf_find(parent, u)
        cc_idx = cc_root[r]
        if cc_idx >= 0:
            cc_edges_out[cc_idx, i] = 1

    return num_ccs


# ---------------------------------------------------------------------------
# Build ring-system labels (one per ring bond, -1 for non-ring bonds).
#
# Two ring bonds are in the same ring system if they share an atom
# (directly or transitively). Cross-system pairs can never disconnect
# a molecule by themselves, so they are pruned from the inner loops.
#
# Uses C stack arrays — no heap allocation.
# ---------------------------------------------------------------------------

cdef void _build_ring_sys_labels_c(
    int num_nodes,
    int num_edges,
    char[::1] ring_edge_mask,
    int[:, ::1] edges,
    int[::1] bond_to_sys_out,   # length >= num_edges; caller pre-fills with -1
) nogil:
    # Stack arrays (MAX_NUM_EDGES = 512, MAX_NUM_NODES = 128)
    cdef int parent[512]
    cdef int root_label[512]   # bond root → contiguous system label (-1 = unseen)
    cdef int atom_first[128]   # first ring bond seen at each atom (-1 = none)
    cdef int bi, atom, k, pa, pb, root
    cdef int num_labels = 0

    for bi in range(512):
        parent[bi] = bi
        root_label[bi] = -1
    for atom in range(128):
        atom_first[atom] = -1

    # Union ring bonds that share an atom
    for bi in range(num_edges):
        if ring_edge_mask[bi] == 0:
            continue
        for k in range(2):
            atom = edges[bi, k]
            if atom_first[atom] == -1:
                atom_first[atom] = bi
            else:
                pa = _uf_find(parent, atom_first[atom])
                pb = _uf_find(parent, bi)
                if pa != pb:
                    parent[pa] = pb

    # Assign contiguous system labels (bond_to_sys_out already -1 for non-ring)
    for bi in range(num_edges):
        if ring_edge_mask[bi] == 0:
            continue
        root = _uf_find(parent, bi)
        if root_label[root] == -1:
            root_label[root] = num_labels
            num_labels += 1
        bond_to_sys_out[bi] = root_label[root]


# ---------------------------------------------------------------------------
# Public Python helper: build ring-edge mask from RDKit mol
# (kept for API compatibility with multi_cut_bfs.py)
# ---------------------------------------------------------------------------

def get_ring_edge_mask(mol, int num_bonds):
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


# ---------------------------------------------------------------------------
# Main BFS — same interface as compute_ccs_multi_cut in multi_cut_bfs.py
# ---------------------------------------------------------------------------

def compute_ccs_multi_cut(
    int num_nodes,
    int num_edges,
    cnp.ndarray node_mask_arr,
    cnp.ndarray edges_arr,
    cnp.ndarray edge_mask_arr,
    cnp.ndarray node_to_edge_idx_arr,
    int max_depth = 3,
    long time_limit = 150,
    cnp.ndarray ring_edge_mask = None,
    int max_cut_size = 2,
    smarts_seed_pairs = None,
):
    """Multi-bond-cut BFS fragmentation (Cython).

    Identical interface and output format to
    ``multi_cut_bfs.compute_ccs_multi_cut`` and ``compute_frags.compute_ccs``.

    Args:
        num_nodes: Number of atoms.
        num_edges: Number of bonds.
        node_mask_arr: Shape ``(num_nodes,)`` uint8 — active atoms.
        edges_arr: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        edge_mask_arr: Shape ``(num_edges,)`` uint8 — active bonds.
        node_to_edge_idx_arr: Kept for interface compatibility; unused here.
        max_depth: Maximum BFS depth.
        time_limit: Wall-clock seconds before timeout.
        ring_edge_mask: Shape ``(num_edges,)`` uint8 — 1 for ring bonds.
            If ``None``, behaves like single-bond BFS.
        max_cut_size: Max simultaneous bond breaks per step (1, 2, or 3).
            Cut=3 is restricted to depth 0 to prevent O(R³) blowup.
        smarts_seed_pairs: Optional list of ``(node_mask_a, node_mask_b)``
            uint8 numpy array pairs pre-computed by ``_apply_smarts_prepass``
            in Python.  Each pair is injected as a depth-1 DAG child pair
            identical to BFS cut=1/2 children.  Pass ``None`` (default) to
            skip the SMARTS prepass.

    Returns:
        Tuple ``(nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix,
        dag_frag_meta)`` — same format as ``compute_frags.compute_ccs``.
    """
    assert num_nodes <= MAX_NUM_NODES, num_nodes
    assert num_edges <= MAX_NUM_EDGES, num_edges

    cdef long start_time = _get_time()
    cdef long cur_time

    # Whether multi-bond cuts are active
    cdef int effective_max_cut = max_cut_size if ring_edge_mask is not None else 1

    # -----------------------------------------------------------------------
    # Typed memoryviews into the input arrays
    # -----------------------------------------------------------------------
    cdef char[::1] node_mask_mv = np.asarray(node_mask_arr, dtype=MASK_DTYPE)
    cdef int[:, ::1] edges_mv = np.asarray(edges_arr, dtype=np.intc)
    cdef char[::1] base_edge_mask_mv = np.asarray(edge_mask_arr, dtype=MASK_DTYPE)

    # -----------------------------------------------------------------------
    # Ring-system labels: int[::1] array, -1 for non-ring bonds
    # -----------------------------------------------------------------------
    cdef cnp.ndarray bond_to_sys_np = np.full(num_edges, -1, dtype=np.intc)
    cdef int[::1] bond_to_sys_mv = bond_to_sys_np
    cdef char[::1] ring_edge_mask_mv

    if ring_edge_mask is not None:
        ring_edge_mask_mv = np.asarray(ring_edge_mask, dtype=MASK_DTYPE)
        _build_ring_sys_labels_c(
            num_nodes, num_edges,
            ring_edge_mask_mv, edges_mv,
            bond_to_sys_mv,
        )

    # -----------------------------------------------------------------------
    # Pre-allocated scratch arrays for CC computation (reused every cut attempt)
    # -----------------------------------------------------------------------
    cdef cnp.ndarray cc_nodes_np = np.zeros((2, num_nodes), dtype=MASK_DTYPE)
    cdef cnp.ndarray cc_edges_np = np.zeros((2, num_edges), dtype=MASK_DTYPE)
    cdef char[:, ::1] cc_nodes_s = cc_nodes_np
    cdef char[:, ::1] cc_edges_s = cc_edges_np

    # -----------------------------------------------------------------------
    # Pre-allocated working mask buffers.
    #
    # Key insight: the bytes keys in `ccs` and `cc_edges` ARE the mask data.
    # We reconstruct masks on demand into these fixed buffers using
    # const char[::1] typed views of the bytes objects — zero allocation,
    # one fast typed-loop memcpy per BFS node.  This eliminates the
    # per-node malloc/free that was present in the previous version.
    # -----------------------------------------------------------------------
    cdef cnp.ndarray cur_node_mask_buf = np.empty(num_nodes, dtype=MASK_DTYPE)
    cdef cnp.ndarray cur_edge_mask_buf = np.empty(num_edges, dtype=MASK_DTYPE)
    cdef char[::1] cur_node_mask_mv = cur_node_mask_buf
    cdef char[::1] cur_edge_mask_mv = cur_edge_mask_buf

    # Raw C pointers used to read bytes keys via PyBytes_AS_STRING + memcpy.
    # Avoids buffer-protocol type-mismatch (bytes export as 'unsigned char',
    # but char[::1] expects 'char').  PyBytes_AS_STRING is a direct macro —
    # zero overhead, no buffer acquisition needed.
    cdef char* key_ptr

    # -----------------------------------------------------------------------
    # BFS state.
    #
    # cc_node_mask_dict and cc_edge_mask_dict from the Python version are
    # ELIMINATED: the bytes key already encodes the mask, so we reconstruct
    # masks from keys on demand.  Only two dicts remain:
    #   css_to_id_dict   — unique node_key → integer DAG node ID
    #   ccs_min_depth    — unique node_key → int min depth seen (replaces set)
    #
    # Storing min-depth as a plain int avoids Python set creation per fragment
    # and simplifies the output construction loop.
    # -----------------------------------------------------------------------
    # Direct C API: no buffer-protocol overhead
    cdef bytes root_node_key = PyBytes_FromStringAndSize(&node_mask_mv[0], num_nodes)
    cdef bytes root_edge_key = PyBytes_FromStringAndSize(&base_edge_mask_mv[0], num_edges)

    css_to_id_dict  = {root_node_key: 0}
    # ccs_depth_bits: node_key → int bitmask of depths seen (bit d = depth d).
    # Replaces the set-per-node approach: `bits |= (1 << d)` is faster than
    # set.add() because it avoids Python set object allocation and hash overhead.
    ccs_depth_bits  = {root_node_key: 1}   # bit 0 = depth 0 for root
    ccs_min_depth   = {root_node_key: 0}   # node_key → int min depth seen
    # seen_set replaces has_seen_dict: BFS processes depths in order so
    # once-seen → always-skip is correct (no depth value needed).
    seen_set        = set()
    dag_edge_dict   = {}

    cdef list ccs      = [root_node_key]
    cdef list cc_edges = [root_edge_key]

    # -----------------------------------------------------------------------
    # C-level loop variables
    # -----------------------------------------------------------------------
    cdef int ccs_idx = 0
    cdef int ccs_end = 0
    cdef int node_idx
    cdef int current_depth
    cdef int i, j, k, ii, jj, kk
    cdef int n_live, n_ring, n_live_nodes
    cdef int n_ccs
    cdef int si, sj, sk
    cdef int c

    # C arrays for live-edge and ring-edge indices (avoid Python list per node)
    cdef int live_arr[512]    # MAX_NUM_EDGES
    cdef int ring_arr[512]
    # Live atom indices — used for selective scratch reset
    cdef int live_node_arr[128]   # MAX_NUM_NODES

    # Per-bond flag: did cutting this bond alone disconnect the molecule?
    cdef cnp.ndarray disconnected_np = np.zeros(num_edges, dtype=MASK_DTYPE)
    cdef char[::1] disconnected_mv = disconnected_np

    # Seed injection working buffers (smarts_seed_pairs)
    cdef cnp.ndarray seed_edge_buf
    cdef char[::1] seed_node_mv
    cdef char[::1] seed_edge_mv
    cdef int u_s, v_s

    force_stop = False
    reached_depth = 0

    # Pre-allocate seed edge buffer once if needed (injection happens at depth 0
    # inside the BFS loop so seeds are processed at depth 1, matching Python behaviour).
    if smarts_seed_pairs is not None:
        seed_edge_buf = np.empty(num_edges, dtype=MASK_DTYPE)
        seed_edge_mv = seed_edge_buf

    # -----------------------------------------------------------------------
    # BFS depth loop
    # -----------------------------------------------------------------------
    for current_depth in range(max_depth):
        ccs_end = len(ccs)

        for node_idx in range(ccs_idx, ccs_end):
            cur_time = _get_time()
            if cur_time - start_time > time_limit:
                force_stop = True
                break

            cur_node_key = ccs[node_idx]
            cur_edge_key = cc_edges[node_idx]

            # Deduplicate: same (node, edge) config can appear from multiple parents.
            # Tuple key reuses cached per-element hash values — cheaper than bytes concat.
            seen_key = (cur_node_key, cur_edge_key)
            if seen_key in seen_set:
                continue
            seen_set.add(seen_key)

            # ---------------------------------------------------------------
            # Reconstruct masks from bytes keys into pre-allocated buffers.
            # PyBytes_AS_STRING gives a raw char* with zero overhead (a
            # direct macro), then memcpy fills the working buffer — no heap
            # allocation, no buffer-protocol type checks.
            # ---------------------------------------------------------------
            key_ptr = PyBytes_AS_STRING(cur_node_key)
            memcpy(&cur_node_mask_mv[0], key_ptr, num_nodes)

            key_ptr = PyBytes_AS_STRING(cur_edge_key)
            memcpy(&cur_edge_mask_mv[0], key_ptr, num_edges)

            # Build live-edge, ring-edge, and live-node C arrays.
            # live_node_arr is used for selective scratch reset.
            n_live = 0
            n_ring = 0
            n_live_nodes = 0
            for i in range(num_edges):
                if cur_edge_mask_mv[i] == 0:
                    continue
                live_arr[n_live] = i
                n_live += 1
                if ring_edge_mask is not None and ring_edge_mask_mv[i] == 1:
                    ring_arr[n_ring] = i
                    n_ring += 1
            for i in range(num_nodes):
                if cur_node_mask_mv[i] != 0:
                    live_node_arr[n_live_nodes] = i
                    n_live_nodes += 1

            if n_live == 0:
                continue

            # Reset disconnected flags for this BFS node
            for i in range(n_live):
                disconnected_mv[live_arr[i]] = 0

            # -----------------------------------------------------------
            # Cut = 1: try every single live edge.
            # First call uses full memset; subsequent calls use selective
            # reset (only live nodes/edges touched by _cc_uf).
            # -----------------------------------------------------------
            _zero_scratch_full(num_nodes, num_edges, cc_nodes_s, cc_edges_s)
            for ii in range(n_live):
                i = live_arr[ii]
                cur_edge_mask_mv[i] = 0
                if ii > 0:
                    _zero_scratch_live(
                        n_live_nodes, n_live, live_node_arr, live_arr,
                        cc_nodes_s, cc_edges_s,
                    )
                n_ccs = _cc_uf(
                    num_nodes, num_edges,
                    cur_node_mask_mv, edges_mv, cur_edge_mask_mv,
                    cc_nodes_s, cc_edges_s,
                )
                cur_edge_mask_mv[i] = 1

                if n_ccs == 2:
                    disconnected_mv[i] = 1
                    child_depth_bit = 1 << (current_depth + 1)
                    for c in range(2):
                        child_node_key = PyBytes_FromStringAndSize(&cc_nodes_s[c, 0], num_nodes)
                        child_edge_key = PyBytes_FromStringAndSize(&cc_edges_s[c, 0], num_edges)

                        if child_node_key not in css_to_id_dict:
                            css_to_id_dict[child_node_key] = len(css_to_id_dict)

                        if child_node_key not in ccs_min_depth:
                            ccs_min_depth[child_node_key] = current_depth + 1
                            ccs_depth_bits[child_node_key] = child_depth_bit
                        else:
                            ccs_depth_bits[child_node_key] |= child_depth_bit
                        ccs.append(child_node_key)
                        cc_edges.append(child_edge_key)

                        if cur_node_key != child_node_key:
                            edge_pair = (css_to_id_dict[cur_node_key],
                                         css_to_id_dict[child_node_key])
                            if edge_pair not in dag_edge_dict:
                                dag_edge_dict[edge_pair] = current_depth + 1

            reached_depth = current_depth + 1

            # -----------------------------------------------------------
            # Cut = 2: pairs of ring edges (same ring system only).
            #
            # Optimization: pre-group ring bonds by system so we only
            # iterate within each system — eliminates the O(R²) cross-
            # system pairs that were previously rejected by an si != sj check.
            # -----------------------------------------------------------
            if effective_max_cut >= 2 and n_ring >= 2:
                # Group non-disconnecting ring bonds by their ring system.
                # Python dict keyed by int system id → Python list of bond indices.
                sys_groups = {}
                for ii in range(n_ring):
                    i = ring_arr[ii]
                    if disconnected_mv[i]:
                        continue
                    si = bond_to_sys_mv[i]
                    if si not in sys_groups:
                        sys_groups[si] = [i]
                    else:
                        sys_groups[si].append(i)

                for bonds_in_sys in sys_groups.values():
                    n_sys = len(bonds_in_sys)
                    if n_sys < 2:
                        continue
                    for ii in range(n_sys):
                        i = bonds_in_sys[ii]
                        for jj in range(ii + 1, n_sys):
                            j = bonds_in_sys[jj]

                            cur_edge_mask_mv[i] = 0
                            cur_edge_mask_mv[j] = 0
                            _zero_scratch_live(
                                n_live_nodes, n_live, live_node_arr, live_arr,
                                cc_nodes_s, cc_edges_s,
                            )
                            n_ccs = _cc_uf(
                                num_nodes, num_edges,
                                cur_node_mask_mv, edges_mv, cur_edge_mask_mv,
                                cc_nodes_s, cc_edges_s,
                            )
                            cur_edge_mask_mv[i] = 1
                            cur_edge_mask_mv[j] = 1

                            if n_ccs != 2:
                                continue

                            child_depth_bit = 1 << (current_depth + 1)
                            for c in range(2):
                                child_node_key = PyBytes_FromStringAndSize(&cc_nodes_s[c, 0], num_nodes)
                                child_edge_key = PyBytes_FromStringAndSize(&cc_edges_s[c, 0], num_edges)

                                if child_node_key not in css_to_id_dict:
                                    css_to_id_dict[child_node_key] = len(css_to_id_dict)

                                if child_node_key not in ccs_min_depth:
                                    ccs_min_depth[child_node_key] = current_depth + 1
                                    ccs_depth_bits[child_node_key] = child_depth_bit
                                else:
                                    ccs_depth_bits[child_node_key] |= child_depth_bit
                                ccs.append(child_node_key)
                                cc_edges.append(child_edge_key)

                                if cur_node_key != child_node_key:
                                    edge_pair = (css_to_id_dict[cur_node_key],
                                                 css_to_id_dict[child_node_key])
                                    if edge_pair not in dag_edge_dict:
                                        dag_edge_dict[edge_pair] = current_depth + 1

            # -----------------------------------------------------------
            # Cut = 3: triples of ring edges — depth 0 only (avoid O(R³) blowup).
            # Also pre-grouped by ring system.
            # -----------------------------------------------------------
            if effective_max_cut >= 3 and current_depth == 0 and n_ring >= 3:
                # Reuse sys_groups if already built; build it if cut=2 was skipped.
                if effective_max_cut < 2 or n_ring < 2:
                    sys_groups = {}
                    for ii in range(n_ring):
                        i = ring_arr[ii]
                        if disconnected_mv[i]:
                            continue
                        si = bond_to_sys_mv[i]
                        if si not in sys_groups:
                            sys_groups[si] = [i]
                        else:
                            sys_groups[si].append(i)

                for bonds_in_sys in sys_groups.values():
                    n_sys = len(bonds_in_sys)
                    if n_sys < 3:
                        continue
                    for ii in range(n_sys):
                        i = bonds_in_sys[ii]
                        for jj in range(ii + 1, n_sys):
                            j = bonds_in_sys[jj]
                            for kk in range(jj + 1, n_sys):
                                k = bonds_in_sys[kk]

                                cur_edge_mask_mv[i] = 0
                                cur_edge_mask_mv[j] = 0
                                cur_edge_mask_mv[k] = 0
                                _zero_scratch_live(
                                    n_live_nodes, n_live, live_node_arr, live_arr,
                                    cc_nodes_s, cc_edges_s,
                                )
                                n_ccs = _cc_uf(
                                    num_nodes, num_edges,
                                    cur_node_mask_mv, edges_mv, cur_edge_mask_mv,
                                    cc_nodes_s, cc_edges_s,
                                )
                                cur_edge_mask_mv[i] = 1
                                cur_edge_mask_mv[j] = 1
                                cur_edge_mask_mv[k] = 1

                                if n_ccs != 2:
                                    continue

                                child_depth_bit = 1 << (current_depth + 1)
                                for c in range(2):
                                    child_node_key = PyBytes_FromStringAndSize(&cc_nodes_s[c, 0], num_nodes)
                                    child_edge_key = PyBytes_FromStringAndSize(&cc_edges_s[c, 0], num_edges)

                                    if child_node_key not in css_to_id_dict:
                                        css_to_id_dict[child_node_key] = len(css_to_id_dict)

                                    if child_node_key not in ccs_min_depth:
                                        ccs_min_depth[child_node_key] = current_depth + 1
                                        ccs_depth_bits[child_node_key] = child_depth_bit
                                    else:
                                        ccs_depth_bits[child_node_key] |= child_depth_bit
                                    ccs.append(child_node_key)
                                    cc_edges.append(child_edge_key)

                                    if cur_node_key != child_node_key:
                                        edge_pair = (css_to_id_dict[cur_node_key],
                                                     css_to_id_dict[child_node_key])
                                        if edge_pair not in dag_edge_dict:
                                            dag_edge_dict[edge_pair] = current_depth + 1

        # -------------------------------------------------------------------
        # SMARTS prepass injection (depth 0 only).
        # Seeds are appended AFTER ccs_end was captured, so they fall beyond
        # the current depth's frontier and are processed at depth 1 — exactly
        # matching the Python version's behaviour.
        # -------------------------------------------------------------------
        if current_depth == 0 and smarts_seed_pairs is not None:
            for pair in smarts_seed_pairs:
                for child_mask_np in pair:
                    seed_node_mv = np.ascontiguousarray(child_mask_np, dtype=MASK_DTYPE)
                    child_node_key = PyBytes_FromStringAndSize(&seed_node_mv[0], num_nodes)
                    if child_node_key == root_node_key:
                        continue
                    for i in range(num_edges):
                        if base_edge_mask_mv[i] == 0:
                            seed_edge_mv[i] = 0
                        else:
                            u_s = edges_mv[i, 0]
                            v_s = edges_mv[i, 1]
                            seed_edge_mv[i] = (
                                1 if (seed_node_mv[u_s] == 1 and seed_node_mv[v_s] == 1) else 0
                            )
                    child_edge_key = PyBytes_FromStringAndSize(&seed_edge_mv[0], num_edges)
                    if child_node_key not in css_to_id_dict:
                        css_to_id_dict[child_node_key] = len(css_to_id_dict)
                    if child_node_key not in ccs_min_depth:
                        ccs_min_depth[child_node_key] = 1
                        ccs_depth_bits[child_node_key] = 1 << 1
                    else:
                        ccs_depth_bits[child_node_key] |= (1 << 1)
                    ccs.append(child_node_key)
                    cc_edges.append(child_edge_key)
                    edge_pair = (css_to_id_dict[root_node_key], css_to_id_dict[child_node_key])
                    if edge_pair not in dag_edge_dict:
                        dag_edge_dict[edge_pair] = 1

        if force_stop:
            reached_depth = current_depth - 1 if current_depth > 0 else 0
            # Mask off depth bits beyond reached_depth and prune empty entries.
            valid_bits = (1 << (reached_depth + 1)) - 1
            ccs_depth_bits = {
                k: (v & valid_bits) for k, v in ccs_depth_bits.items() if v & valid_bits
            }
            ccs_min_depth = {k: v for k, v in ccs_min_depth.items() if v <= reached_depth}
            css_to_id_dict = {k: css_to_id_dict[k] for k in ccs_min_depth}
            id_to_css = {v: k for k, v in css_to_id_dict.items()}
            dag_edge_dict = {
                (s, d): dep
                for (s, d), dep in dag_edge_dict.items()
                if dep <= reached_depth and s in id_to_css and d in id_to_css
            }
            break

        ccs_idx = ccs_end

    # -----------------------------------------------------------------------
    # Build output matrices — same format as compute_frags.compute_ccs
    # -----------------------------------------------------------------------
    num_un_ccs = len(css_to_id_dict)
    ordered_ccs = [b""] * num_un_ccs
    for cc, idx in css_to_id_dict.items():
        ordered_ccs[idx] = cc

    # Reconstruct node masks from bytes keys (zero-copy frombuffer + stack)
    nodes_mask_matrix = np.stack(
        [np.frombuffer(cc, dtype=MASK_DTYPE) for cc in ordered_ccs],
        dtype=MASK_DTYPE,
    )
    nodes_depth_matrix = np.zeros((num_un_ccs, max_depth + 1), dtype=MASK_DTYPE)
    nodes_min_depth    = np.zeros(num_un_ccs, dtype=MASK_DTYPE)

    # ccs_depth_bits is an int bitmask (bit d = 1 iff fragment seen at depth d).
    # Unpack into nodes_depth_matrix without creating intermediate Python sets.
    for idx, cc in enumerate(ordered_ccs):
        bits = ccs_depth_bits[cc]
        for d in range(max_depth + 1):
            if (bits >> d) & 1:
                nodes_depth_matrix[idx, d] = 1
        nodes_min_depth[idx] = ccs_min_depth[cc]

    if dag_edge_dict:
        dag_edges_matrix = np.array(list(dag_edge_dict.keys()), dtype=np.int64)
        edges_min_depth  = np.array(list(dag_edge_dict.values()), dtype=MASK_DTYPE)
    else:
        dag_edges_matrix = np.empty((0, 2), dtype=np.int64)
        edges_min_depth  = np.empty(0, dtype=MASK_DTYPE)

    dag_frag_meta = {
        "reached_depth":   reached_depth,
        "edges_min_depth": edges_min_depth,
        "nodes_min_depth": nodes_min_depth,
        "force_stopped":   force_stop,
    }
    return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta
