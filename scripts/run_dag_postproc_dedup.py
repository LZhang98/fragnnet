"""Post-process a DAG tar file: remove nodes whose heavy-atom formula already appears
at a strictly shallower depth (all_depths strategy), then fix orphaned nodes via
reachability from root.

Creates a deduplicated copy of the tar. Non-DAG members (meta_info.json,
spec_stats_df.pkl, etc.) are copied unchanged.

Usage:
    # D3
    python scripts/run_dag_postproc_dedup.py \
        --input_tar  data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \
        --output_tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_dedup.tar

    # D4
    python scripts/run_dag_postproc_dedup.py \
        --input_tar  data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --output_tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse_dedup.tar

Known limitations (see ISSUES printed at end):
  - h-formula recall may be slightly below 100%: structural isomers with the same
    base formula can carry different H-transfer variants; removing one isomer may
    silently drop some h-formulae even if the base formula is still covered.
  - Stats fields (formula_redundancy, dag_num_nodes_by_depth, dag_sparsity, etc.)
    are recomputed; dag_num_nodes_nb is set to -1 (unused sentinel in training code).
"""

import argparse
import bz2
import io
import json
import os
import pickle
import sys
import tarfile
import time
from collections import defaultdict, deque

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, "src")
from fragnnet.utils.frag_utils import ELEMENT_TO_IDX, get_node_feats


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def formula_arr_to_str(arr) -> str:
    parts = []
    for elem, idx in sorted(ELEMENT_TO_IDX.items(), key=lambda x: x[1]):
        if elem == "H":
            continue
        if arr[idx] > 0:
            parts.append(f"{elem}{int(arr[idx])}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core dedup logic
# ---------------------------------------------------------------------------

def dedup_dag(dag: dict) -> tuple[dict, dict]:
    """Apply all_depths formula dedup + orphan removal to one molecule DAG.

    Args:
        dag: Loaded DAG dict from pkl.bz2.

    Returns:
        Tuple of (new_dag, stats_dict).
    """
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
    edge_attr = g.edge_attr
    node_feat_idxs = g.node_feat_idxs[0]
    nmd: np.ndarray = dag["nodes_min_depth"]   # uint8 [n_nodes]
    emd: np.ndarray = dag["edges_min_depth"]   # uint8 [n_edges]

    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")   # [n_nodes, 12]
    h_formulae_idx = get_node_feats(x, node_feat_idxs, "h_formulae_idx")  # [n_nodes, 9]
    fpp = dag["formula_peak_probs"]  # [n_formulae, n_spectra]

    n_nodes = x.shape[0]
    n_edges = edge_index.shape[1]

    node_depth = nmd.tolist()
    node_fstr = [formula_arr_to_str(base_formula[ni].numpy()) for ni in range(n_nodes)]

    # Formula → minimum depth at which it first appears
    formula_min_depth: dict[str, int] = {}
    for ni in range(n_nodes):
        d = node_depth[ni]
        f = node_fstr[ni]
        if f not in formula_min_depth or d < formula_min_depth[f]:
            formula_min_depth[f] = d

    # Mark nodes for removal: depth > formula's minimum depth
    remove_set: set[int] = {
        ni for ni in range(n_nodes)
        if formula_min_depth[node_fstr[ni]] < node_depth[ni]
    }

    # Reachability fix: find orphaned nodes in the pruned graph and also remove them
    remaining = set(range(n_nodes)) - remove_set
    adj: dict[int, list[int]] = defaultdict(list)
    for ei in range(n_edges):
        src = edge_index[0, ei].item()
        dst = edge_index[1, ei].item()
        if src in remaining and dst in remaining:
            adj[src].append(dst)

    root = int(np.argmin(nmd))  # depth-0 node
    reachable: set[int] = set()
    queue = deque([root])
    while queue:
        ni = queue.popleft()
        if ni in reachable:
            continue
        reachable.add(ni)
        for child in adj[ni]:
            if child not in reachable:
                queue.append(child)

    orphans = remaining - reachable
    remove_set |= orphans

    # Final keep list (preserves original ordering for stable node feature matrix)
    keep_nodes = sorted(set(range(n_nodes)) - remove_set)
    node_remap = {old: new for new, old in enumerate(keep_nodes)}

    # Keep edges where both endpoints survive
    keep_edges = [
        ei for ei in range(n_edges)
        if edge_index[0, ei].item() in node_remap
        and edge_index[1, ei].item() in node_remap
    ]

    # Build new PyG graph
    new_x = x[keep_nodes]
    new_nmd = nmd[np.array(keep_nodes, dtype=np.intp)]
    new_emd = emd[np.array(keep_edges, dtype=np.intp)] if keep_edges else emd[:0]

    if keep_edges:
        ei_kept = edge_index[:, keep_edges]
        new_edge_index = torch.tensor(
            [[node_remap[ei_kept[0, i].item()], node_remap[ei_kept[1, i].item()]]
             for i in range(len(keep_edges))],
            dtype=torch.long,
        ).t().contiguous()
        new_edge_attr = edge_attr[keep_edges]
    else:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
        new_edge_attr = edge_attr[:0]

    new_g = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        node_feat_idxs=g.node_feat_idxs,
        edge_feat_idxs=g.edge_feat_idxs,
    )

    # Recompute depth-based stats
    new_n_nodes = len(keep_nodes)
    new_n_edges = len(keep_edges)
    max_d = int(new_nmd.max()) if new_n_nodes > 0 else 0

    nodes_by_depth: dict[int, int] = defaultdict(int)
    for d in new_nmd.tolist():
        nodes_by_depth[d] += 1

    edges_by_depth: dict[int, int] = defaultdict(int)
    for d in new_emd.tolist():
        edges_by_depth[d] += 1

    # formula_redundancy = avg nodes per unique formula
    new_fstrs = [formula_arr_to_str(new_x[ni, :12].numpy()) for ni in range(new_n_nodes)]
    n_unique = len(set(new_fstrs))
    new_formula_redundancy = new_n_nodes / n_unique if n_unique > 0 else 1.0

    # h-formula recall: count matched fidx still present in at least one kept node
    new_h_formulae_idx = get_node_feats(new_x, g.node_feat_idxs[0], "h_formulae_idx")
    kept_fidxs: set[int] = set()
    for ni in range(new_n_nodes):
        for v in new_h_formulae_idx[ni]:
            fidx = v.item()
            if fidx > 0:
                kept_fidxs.add(fidx)

    matched_fidxs = {
        fidx for fidx in range(1, fpp.shape[0])
        if fpp[fidx].sum() > 0
    }
    lost_matched = matched_fidxs - kept_fidxs
    n_lost_h = len(lost_matched)

    new_dag = dict(dag)
    new_dag["dag"] = new_g
    new_dag["nodes_min_depth"] = new_nmd
    new_dag["edges_min_depth"] = new_emd
    new_dag["dag_num_nodes"] = new_n_nodes
    new_dag["dag_num_edges"] = new_n_edges
    new_dag["dag_num_nodes_nb"] = -1          # sentinel; unused in training
    new_dag["dag_num_nodes_by_depth"] = dict(nodes_by_depth)
    new_dag["dag_num_edges_by_depth"] = dict(edges_by_depth)
    new_dag["formula_redundancy"] = new_formula_redundancy
    new_dag["dag_sparsity"] = new_n_edges / (new_n_nodes ** 2) if new_n_nodes > 1 else 0.0

    stats = {
        "n_nodes_before": n_nodes,
        "n_nodes_after": new_n_nodes,
        "n_edges_before": n_edges,
        "n_edges_after": new_n_edges,
        "n_orphans": len(orphans),
        "n_matched_h": len(matched_fidxs),
        "n_lost_h": n_lost_h,
    }
    return new_dag, stats


# ---------------------------------------------------------------------------
# Tar processing
# ---------------------------------------------------------------------------

def process_tar(input_tar: str, output_tar: str) -> None:
    print(f"Input:  {input_tar}")
    print(f"Output: {output_tar}\n")

    input_base = os.path.basename(input_tar).replace(".tar", "")
    output_base = os.path.basename(output_tar).replace(".tar", "")

    def rewrite_name(name: str) -> str:
        if name.startswith(input_base + "/"):
            return output_base + name[len(input_base):]
        return name

    with tarfile.open(input_tar, "r") as tf:
        members = tf.getmembers()

    dag_members = [m for m in members if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    other_members = [m for m in members if m not in dag_members]
    print(f"  DAG members:   {len(dag_members):,}")
    print(f"  Other members: {len(other_members):,}\n")

    totals: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()

    with tarfile.open(input_tar, "r") as tf_in, tarfile.open(output_tar, "w") as tf_out:
        # Copy non-DAG members as-is
        for m in other_members:
            f = tf_in.extractfile(m)
            if f is not None:
                data = f.read()
                info = tarfile.TarInfo(name=rewrite_name(m.name))
                info.size = len(data)
                tf_out.addfile(info, io.BytesIO(data))
            else:
                tf_out.addfile(m)

        # Process DAG members
        for i, member in enumerate(dag_members):
            if i % 2000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i:>6}/{len(dag_members):,}  ({elapsed:.0f}s)")

            f = tf_in.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))

            if dag["reached_depth"] < 1:
                # Trivial DAG (root only) — copy unchanged
                raw = bz2.compress(pickle.dumps(dag))
                info = tarfile.TarInfo(name=rewrite_name(member.name))
                info.size = len(raw)
                tf_out.addfile(info, io.BytesIO(raw))
                totals["n_nodes_before"] += dag["dag_num_nodes"]
                totals["n_nodes_after"] += dag["dag_num_nodes"]
                totals["n_edges_before"] += dag["dag_num_edges"]
                totals["n_edges_after"] += dag["dag_num_edges"]
                continue

            try:
                new_dag, stats = dedup_dag(dag)
            except Exception as e:
                print(f"  [WARN] {member.name}: dedup failed ({e}), copying unchanged")
                new_dag = dag
                stats = {
                    "n_nodes_before": dag["dag_num_nodes"],
                    "n_nodes_after": dag["dag_num_nodes"],
                    "n_edges_before": dag["dag_num_edges"],
                    "n_edges_after": dag["dag_num_edges"],
                    "n_orphans": 0,
                    "n_matched_h": 0,
                    "n_lost_h": 0,
                }

            raw = bz2.compress(pickle.dumps(new_dag))
            info = tarfile.TarInfo(name=rewrite_name(member.name))
            info.size = len(raw)
            tf_out.addfile(info, io.BytesIO(raw))

            for k, v in stats.items():
                totals[k] += v

    elapsed = time.perf_counter() - t0
    nn_b = totals["n_nodes_before"]
    nn_a = totals["n_nodes_after"]
    ne_b = totals["n_edges_before"]
    ne_a = totals["n_edges_after"]
    nm   = totals["n_matched_h"]
    nl   = totals["n_lost_h"]

    print(f"\nDone in {elapsed:.0f}s")
    print(f"\n{'='*64}")
    print(f"{'':32s} {'before':>12}  {'after':>12}")
    print(f"{'-'*64}")
    print(f"  {'nodes':<30} {nn_b:>12,}  {nn_a:>12,}")
    print(f"  {'edges':<30} {ne_b:>12,}  {ne_a:>12,}")
    print(f"  {'node reduction':<30} {'—':>12}  {100*(nn_b-nn_a)/nn_b:>11.1f}%")
    print(f"  {'edge reduction':<30} {'—':>12}  {100*(ne_b-ne_a)/ne_b:>11.1f}%")
    print(f"  {'orphans removed':<30} {totals['n_orphans']:>12,}")
    print(f"\n  {'matched h-formulae':<30} {nm:>12,}  {nm-nl:>12,}")
    print(f"  {'h-formula recall':<30} {'100.00%':>12}  {100*(nm-nl)/nm if nm else 0:>11.2f}%")

    print(f"\n{'='*64}")
    print("KNOWN ISSUES TO CHECK BEFORE TRAINING:")
    print(f"  1. h-formula recall is {100*(nm-nl)/nm if nm else 0:.2f}% (not 100%).")
    print(f"     Structural isomers with the same base formula can have different")
    print(f"     H-transfer variants. Removing one may silently drop specific h-formula")
    print(f"     indices even though the base formula remains covered.")
    print(f"     → Impact: the model trains on a slightly smaller peak-matched set.")
    print(f"     → Mitigation: use leaf_only strategy (only removes true leaves, safer).")
    print()
    print(f"  2. {totals['n_orphans']:,} orphan nodes removed (children of pruned intermediate nodes).")
    print(f"     These had unique formulas at their depth but lost their parent edges.")
    print(f"     → Impact: any matched formula carried exclusively by orphan nodes is lost.")
    print(f"       (counted in loss above).")
    print()
    print(f"  3. dag_num_nodes_nb set to -1 (sentinel). Verify training code ignores it.")
    print()
    print(f"  4. formula_redundancy, dag_sparsity, dag_num_nodes_by_depth are recomputed")
    print(f"     from the pruned graph. If any loss function uses these as weights,")
    print(f"     the distribution will shift vs the original tar.")
    print()
    print(f"  5. GNN graph structure changes: pruned DAGs have fewer nodes and a different")
    print(f"     depth distribution. Verify model architecture does not rely on a minimum")
    print(f"     number of nodes or a specific depth layout.")
    print(f"\nOutput written to: {output_tar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_tar", required=True)
    parser.add_argument("--output_tar", required=True)
    args = parser.parse_args()
    process_tar(args.input_tar, args.output_tar)


if __name__ == "__main__":
    main()
