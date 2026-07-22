"""Structural audit: can per-edge sibling branching (beta) be made identifiable by DAG pruning?

Background
----------
The spectrum loss only observes the node marginal P(n), which sums flux over every fragmentation
path reaching a node. When a node has multiple parents, the split (beta) into it is not recoverable
from that marginal -- it is non-identifiable. A bottom-up flux target for beta is only well defined
where a node has a *single* parent, so that all of its flux is attributable to one incoming edge.

The proposal was: prune each node to its minimum-depth in-edges (the union of shortest-path trees,
which is exactly what the BFS in compute_frags already discovers), keeping every node but dropping
deeper redundant edges. The claim under test is that this leaves most nodes single-parent, so beta
becomes identifiable for the bulk of the DAG without an arbitrary tie-break.

This script measures that claim directly on the real DAGs. It compares three structures per molecule:

  1. full            -- the DAG as stored.
  2. min_depth       -- keep edge p->c iff depth[p] == depth[c] - 1 (union of shortest-path trees,
                        NO tie-break). Nodes preserved; only non-shortest-path edges dropped.
  3. arborescence    -- each non-root node keeps exactly ONE min-depth parent (lowest node index).
                        A spanning tree of the deduplicated DAG; requires a tie-break.

For each it reports the fraction of nodes that remain multi-parent (the non-identifiable set) and,
for the branching side, how many parents still have >= 2 children (competable) relative to the full
DAG. Depth is the BFS shortest-path distance from the in-degree-0 root(s).

Reading the result
------------------
  - min_depth multi-parent% small  -> the no-tie-break prune restores identifiability; build on it.
  - min_depth multi-parent% large  -> shortest-path pruning is NOT enough (order-invariant cut
                                      sequences give a node several same-depth parents); an
                                      arborescence (a real tie-break) is required, and then WHICH
                                      parent is chosen matters -> use a chemically meaningful prior.

Usage:
    python scripts/eval/beta_target_identifiability_audit.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --limit_mols 3000
"""

from __future__ import annotations

import argparse
import functools
import glob
import logging
import os
from collections import deque

import numpy as np
import pandas as pd

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bfs_depth(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Shortest-path depth of every node from the in-degree-0 root(s).

    Args:
        edge_index: Directed parent->child edges of shape (2, num_edges).
        num_nodes: Number of nodes in the DAG.

    Returns:
        Depth array of shape (num_nodes,); -1 for any node unreachable from a root.
    """
    indeg = np.bincount(edge_index[1], minlength=num_nodes)
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in zip(edge_index[0], edge_index[1]):
        adj[a].append(int(b))
    depth = np.full(num_nodes, -1, dtype=np.int64)
    dq: deque[int] = deque()
    for r in np.where(indeg == 0)[0]:
        depth[r] = 0
        dq.append(int(r))
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                dq.append(v)
    return depth


def _process_mol(mol_id: int, frag_dp: str, is_compressed: bool) -> dict | None:
    """Per-molecule multi-parent and branching-parent counts for the three structures."""
    dag_d = frag_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    if dag_d is None:
        return None
    dag = dag_d["dag"]
    edge_index = dag.edge_index.numpy()
    num_nodes = int(dag.x.shape[0])
    if edge_index.shape[1] == 0:
        return None

    depth = _bfs_depth(edge_index, num_nodes)
    src, dst = edge_index[0], edge_index[1]

    # min-depth (shortest-path) in-edge mask: parent sits exactly one layer above the child.
    keep = depth[src] == depth[dst] - 1

    full_indeg = np.bincount(dst, minlength=num_nodes)
    min_indeg = np.bincount(dst[keep], minlength=num_nodes)

    # arborescence: each node keeps its lowest-index min-depth parent.
    arbor_parent = np.full(num_nodes, -1, dtype=np.int64)
    for a, b, k in zip(src, dst, keep):
        if k and (arbor_parent[b] == -1 or a < arbor_parent[b]):
            arbor_parent[b] = a
    arbor_children = np.bincount(arbor_parent[arbor_parent >= 0], minlength=num_nodes)

    # branching parents (>= 2 distinct children) in full vs arborescence.
    full_parent_children: dict[int, int] = {}
    for p in np.unique(src):
        full_parent_children[int(p)] = int(np.unique(dst[src == p]).size)

    return {
        "mol_id": mol_id,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "num_edges_min_depth": int(keep.sum()),
        "unreachable_nodes": int((depth == -1).sum()),
        "multi_parent_full": int((full_indeg >= 2).sum()),
        "multi_parent_min_depth": int((min_indeg >= 2).sum()),
        # non-root nodes = nodes with >= 1 parent; used as the identifiability denominator.
        "non_root_nodes": int((full_indeg >= 1).sum()),
        "branch_parents_full": int(sum(v >= 2 for v in full_parent_children.values())),
        "branch_parents_arbor": int((arbor_children >= 2).sum()),
    }


def _pct(num: float, den: float) -> str:
    return "nan" if den == 0 else f"{100.0 * num / den:.2f}%"


def summarize(df: pd.DataFrame) -> None:
    """Print the three-structure identifiability comparison."""
    s = df.sum(numeric_only=True)
    nodes = s["num_nodes"]
    non_root = s["non_root_nodes"]
    bp_full = s["branch_parents_full"]

    print(f"\n{'=' * 88}")
    print("Beta-target identifiability by DAG pruning (structural, model-free)")
    print(f"{'=' * 88}")
    print(f"molecules: {len(df)}   nodes: {int(nodes)}   non-root nodes: {int(non_root)}")
    print(f"unreachable nodes (should be 0): {int(s['unreachable_nodes'])}")
    print(f"edge retention under min-depth prune: {_pct(s['num_edges_min_depth'], s['num_edges'])}")
    print()
    print("Multi-parent nodes = non-identifiable beta (as fraction of non-root nodes):")
    print(
        f"  full DAG:              {int(s['multi_parent_full']):>10d} "
        f"({_pct(s['multi_parent_full'], non_root)})"
    )
    print(
        f"  min-depth (no t-brk):  {int(s['multi_parent_min_depth']):>10d} "
        f"({_pct(s['multi_parent_min_depth'], non_root)})"
    )
    print(f"  arborescence (1 par):  {0:>10d} (0.00%  by construction)")
    print()
    print("Branching parents (>= 2 children) retained vs full DAG:")
    print(f"  full DAG:              {int(bp_full):>10d}")
    print(
        f"  arborescence:          {int(s['branch_parents_arbor']):>10d} "
        f"({_pct(s['branch_parents_arbor'], bp_full)} of full)"
    )
    print(f"\n{'-' * 88}")
    print("READ:")
    print("  If min-depth multi-parent% is still high, the no-tie-break prune does NOT make beta")
    print("  identifiable (order-invariant cut sequences give a node several same-depth parents).")
    print("  Only the arborescence zeroes it out -- so a tie-break IS required, and WHICH parent it")
    print("  picks decides who owns each branch: use a chemical bond-priority, not lowest-index.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frag_dp", required=True, help="DAG dir (or .tar/.zip).")
    parser.add_argument("--limit_mols", type=int, default=None, help="Subsample this many DAGs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of per-mol counts.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.frag_dp, "*")))
    if not files:
        raise ValueError(f"No DAG files under {args.frag_dp}")
    is_compressed = files[0].endswith(".bz2")
    mol_ids = [os.path.basename(f).split(".")[0] for f in files]
    if args.limit_mols is not None and args.limit_mols < len(mol_ids):
        rng = np.random.default_rng(args.seed)
        mol_ids = list(rng.choice(mol_ids, size=args.limit_mols, replace=False))

    worker = functools.partial(_process_mol, frag_dp=args.frag_dp, is_compressed=is_compressed)
    log.info("Processing %d DAGs", len(mol_ids))
    rows = [r for r in data_utils.par_apply(mol_ids, worker, n_jobs=args.n_jobs) if r is not None]
    if not rows:
        raise ValueError("No DAG produced audit rows; check --frag_dp.")

    df = pd.DataFrame(rows)
    summarize(df)
    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-mol counts to %s", args.out_fp)


if __name__ == "__main__":
    main()
