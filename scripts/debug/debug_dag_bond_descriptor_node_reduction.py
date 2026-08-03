"""Measure DAG node-count reduction from applying candidate bond-breakability masks at
depth 4, using the same (in_ring, element_pair, alpha_hetero) bond descriptor as
debug_dag_bond_descriptor_stats.py. That script measured the *recall-loss* side of the
masking tradeoff; this measures the *node-reduction* side
(docs/plans/bond_breakability_mask_plan.md step 4).

A depth-4 node survives a masking policy iff at least one of its incoming routes (parent
edges / multi-bond cuts) uses no bond from the masked bucket set - mirrors the recall
survival logic in debug_dag_bond_descriptor_stats.py, but applied to ALL depth-4 nodes,
not just ones that matched a real spectral peak.

Usage:
    python scripts/debug/debug_dag_bond_descriptor_node_reduction.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --n_sample 3000 \
        --seed 42
"""

import argparse
import bz2
import os
import pickle
import random
import sys
import tarfile
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/debug")
from debug_dag_bond_descriptor_stats import boundary_buckets, decode_cc

CANDIDATE_POLICIES = {
    "4-bucket (non-ring CC/XX)": {
        (False, "CC", False),
        (False, "CC", True),
        (False, "XX", False),
        (False, "XX", True),
    },
    "6-bucket (+ ring CX)": {
        (False, "CC", False),
        (False, "CC", True),
        (False, "XX", False),
        (False, "XX", True),
        (True, "CX", False),
        (True, "CX", True),
    },
}


def analyze_dag_node_reduction(
    dag: dict, mol
) -> tuple[int, dict[str, int], int, dict[str, int]] | None:
    """Count depth-4 nodes/edges and how many survive each candidate masking policy.

    Node survival uses OR-across-routes (a node survives if any one route to it is
    unmasked); edge survival is per-route (an edge/route is blocked if masking removes
    any of the bonds it needs to cut). Edge count is the more direct proxy for BFS
    enumeration cost, since redundant multi-parent routes to the same fragment inflate
    edges much more than distinct nodes (the 1.3M-node pathological case from step 1 had
    ~7 edges per node).
    """
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
    node_feat_idxs = g.node_feat_idxs[0]
    n_nodes = x.shape[0]
    num_atoms = mol.GetNumAtoms()

    cc_start, cc_end = node_feat_idxs[1].item(), node_feat_idxs[2].item()
    node_cc = [decode_cc(x[ni, cc_start:cc_end], num_atoms) for ni in range(n_nodes)]

    nmd = dag["nodes_min_depth"]

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    parent_of: dict[int, list[int]] = defaultdict(list)
    for ei in range(edge_index.shape[1]):
        parent_of[edge_index[1, ei].item()].append(edge_index[0, ei].item())

    nodes_at_d4 = [ni for ni in range(n_nodes) if node_depth(ni) == 4]
    total_d4 = len(nodes_at_d4)
    if total_d4 == 0:
        return None

    edge_bucket_cache: dict[tuple[int, int], set | None] = {}
    retained_counts = dict.fromkeys(CANDIDATE_POLICIES, 0)
    total_d4_edges = 0
    retained_edge_counts = dict.fromkeys(CANDIDATE_POLICIES, 0)

    for ni in nodes_at_d4:
        child_atoms = node_cc[ni]
        routes: list[set] = []
        for pi in parent_of[ni]:
            cache_key = (ni, pi)
            if cache_key not in edge_bucket_cache:
                complement_atoms = node_cc[pi] - child_atoms
                edge_bucket_cache[cache_key] = (
                    boundary_buckets(mol, child_atoms, complement_atoms)
                    if complement_atoms
                    else None
                )
            buckets = edge_bucket_cache[cache_key]
            if buckets is not None:
                routes.append(buckets)
        if not routes:
            continue
        total_d4_edges += len(routes)
        for policy_name, masked in CANDIDATE_POLICIES.items():
            retained_counts[policy_name] += any(
                not any(b in masked for b in route) for route in routes
            )
            retained_edge_counts[policy_name] += sum(
                not any(b in masked for b in route) for route in routes
            )

    return total_d4, retained_counts, total_d4_edges, retained_edge_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    args = parser.parse_args()

    print(f"Loading {args.mol_df} ...")
    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    print(f"Scanning {args.tar} ...")
    with tarfile.open(args.tar, "r") as tf:
        members = [
            m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name
        ]
    print(f"  Total DAG files: {len(members):,}")

    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    print(f"  Sampling {len(sample):,} molecules ...")

    total_d4_agg = 0
    retained_agg = dict.fromkeys(CANDIDATE_POLICIES, 0)
    total_d4_edges_agg = 0
    retained_edges_agg = dict.fromkeys(CANDIDATE_POLICIES, 0)
    n_analyzed = 0
    n_skipped_large = 0
    n_skipped_no_mol = 0

    with tarfile.open(args.tar, "r") as tf:
        for i, member in enumerate(sample):
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} ...", flush=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))
            if dag["reached_depth"] < 1:
                continue
            if dag["dag"].x.shape[0] > args.max_nodes:
                n_skipped_large += 1
                continue

            mol_id = int(os.path.basename(member.name).split(".")[0])
            mol = mol_by_id.get(mol_id)
            if mol is None:
                n_skipped_no_mol += 1
                continue

            result = analyze_dag_node_reduction(dag, mol)
            if result is None:
                continue
            total_d4, retained_counts, total_d4_edges, retained_edge_counts = result
            total_d4_agg += total_d4
            total_d4_edges_agg += total_d4_edges
            for k, v in retained_counts.items():
                retained_agg[k] += v
            for k, v in retained_edge_counts.items():
                retained_edges_agg[k] += v
            n_analyzed += 1

    print(
        f"  Skipped {n_skipped_large:,} molecules (>{args.max_nodes:,} nodes), "
        f"{n_skipped_no_mol:,} molecules (no mol_id match)"
    )
    print(
        f"\nAnalyzed {n_analyzed} molecules | {total_d4_agg:,} total depth-4 nodes | "
        f"{total_d4_edges_agg:,} total depth-4 edges "
        f"({total_d4_edges_agg / total_d4_agg:.2f} edges/node)"
    )

    print(f"\n{'='*90}")
    print("Node-level (survives if ANY route unmasked):")
    print(f"{'Policy':<30} {'Nodes retained':>15} {'Retained %':>12} {'Reduction %':>13}")
    print(f"{'-'*90}")
    for policy_name, retained in retained_agg.items():
        pct_retained = 100 * retained / total_d4_agg
        pct_reduction = 100 - pct_retained
        print(f"{policy_name:<30} {retained:>15,} {pct_retained:>11.2f}% {pct_reduction:>12.2f}%")

    print(f"\n{'='*90}")
    print("Edge-level (each route counted individually - proxy for BFS enumeration cost):")
    print(f"{'Policy':<30} {'Edges retained':>15} {'Retained %':>12} {'Reduction %':>13}")
    print(f"{'-'*90}")
    for policy_name, retained in retained_edges_agg.items():
        pct_retained = 100 * retained / total_d4_edges_agg
        pct_reduction = 100 - pct_retained
        print(f"{policy_name:<30} {retained:>15,} {pct_retained:>11.2f}% {pct_reduction:>12.2f}%")


if __name__ == "__main__":
    main()
