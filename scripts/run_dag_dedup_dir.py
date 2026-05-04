"""Apply post-processing DAG deduplication to a directory-format dataset.

Reads .pickle.bz2 DAG files from <input_dir>/dags/, applies the chosen strategy,
and writes deduplicated files to <output_dir>/dags/. All non-dag files are copied
unchanged (global_stats.json, mol_stats_df.pkl, etc.).

Strategies:
  hybrid     : remove deeper nodes only when all h-indices already covered by kept
               shallower nodes — 0% h-recall loss, safe for training (recommended)
  h_coverage : remove any node whose h-index set is covered by already-seen nodes
               — 0% h-recall loss, most aggressive
  leaf_only  : remove max-depth nodes whose base formula exists at shallower depth
  all_depths : remove any node whose base formula exists at strictly shallower depth

Usage:
    python scripts/run_dag_dedup_dir.py \\
        --input_dir  data/frag/debug_tinyv4_d3h4 \\
        --output_dir data/frag/debug_tinyv4_d3h4_hybrid \\
        --strategy   hybrid
"""

import argparse
import bz2
import pickle
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, "src")
from fragnnet.utils.frag_utils import ELEMENT_TO_IDX, get_node_feats


# ---------------------------------------------------------------------------
# Helpers (shared with run_dag_dedup_compare.py)
# ---------------------------------------------------------------------------

def formula_arr_to_str(arr) -> str:
    parts = []
    for elem, idx in sorted(ELEMENT_TO_IDX.items(), key=lambda x: x[1]):
        if elem == "H":
            continue
        if arr[idx] > 0:
            parts.append(f"{elem}{int(arr[idx])}")
    return "".join(parts)


def build_remove_set(strategy: str, node_depth, node_fstr, h_formulae_idx, n_nodes) -> set[int]:
    if strategy == "leaf_only":
        formula_min_depth: dict[str, int] = {}
        for ni in range(n_nodes):
            d, f = node_depth[ni], node_fstr[ni]
            if f not in formula_min_depth or d < formula_min_depth[f]:
                formula_min_depth[f] = d
        max_depth = max(node_depth)
        return {
            ni for ni in range(n_nodes)
            if node_depth[ni] == max_depth
            and formula_min_depth[node_fstr[ni]] < max_depth
        }

    if strategy == "all_depths":
        formula_min_depth = {}
        for ni in range(n_nodes):
            d, f = node_depth[ni], node_fstr[ni]
            if f not in formula_min_depth or d < formula_min_depth[f]:
                formula_min_depth[f] = d
        return {ni for ni in range(n_nodes) if formula_min_depth[node_fstr[ni]] < node_depth[ni]}

    if strategy == "hybrid":
        formula_to_nodes: dict[str, list] = defaultdict(list)
        for ni in range(n_nodes):
            formula_to_nodes[node_fstr[ni]].append((node_depth[ni], ni))
        remove_set: set[int] = set()
        for node_list in formula_to_nodes.values():
            if len(node_list) == 1:
                continue
            node_list.sort(key=lambda x: x[0])
            min_depth = node_list[0][0]
            covered_h: set[int] = set()
            for depth, ni in node_list:
                node_h = {v.item() for v in h_formulae_idx[ni] if v.item() > 0}
                if depth == min_depth:
                    covered_h |= node_h
                elif node_h <= covered_h:
                    remove_set.add(ni)
                else:
                    covered_h |= node_h
        return remove_set

    if strategy == "h_coverage":
        formula_to_nodes = defaultdict(list)
        for ni in range(n_nodes):
            formula_to_nodes[node_fstr[ni]].append((node_depth[ni], ni))
        remove_set = set()
        for node_list in formula_to_nodes.values():
            if len(node_list) == 1:
                continue
            node_list.sort(key=lambda x: x[0])
            covered_h: set[int] = set()
            for _depth, ni in node_list:
                node_h = {v.item() for v in h_formulae_idx[ni] if v.item() > 0}
                if node_h <= covered_h:
                    remove_set.add(ni)
                else:
                    covered_h |= node_h
        return remove_set

    raise ValueError(f"Unknown strategy: {strategy!r}")


def apply_remove_set(dag: dict, remove_set: set[int]) -> tuple[dict, dict]:
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
    edge_attr = g.edge_attr
    nmd: np.ndarray = dag["nodes_min_depth"]
    emd: np.ndarray = dag["edges_min_depth"]
    n_nodes = x.shape[0]
    n_edges = edge_index.shape[1]

    remaining = set(range(n_nodes)) - remove_set
    adj: dict[int, list[int]] = defaultdict(list)
    for ei in range(n_edges):
        src, dst = edge_index[0, ei].item(), edge_index[1, ei].item()
        if src in remaining and dst in remaining:
            adj[src].append(dst)

    root = int(np.argmin(nmd))
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

    remove_set = remove_set | (remaining - reachable)
    keep_nodes = sorted(set(range(n_nodes)) - remove_set)
    node_remap = {old: new for new, old in enumerate(keep_nodes)}

    keep_edges = [
        ei for ei in range(n_edges)
        if edge_index[0, ei].item() in node_remap
        and edge_index[1, ei].item() in node_remap
    ]

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

    new_n_nodes, new_n_edges = len(keep_nodes), len(keep_edges)
    nodes_by_depth: dict[int, int] = defaultdict(int)
    edges_by_depth: dict[int, int] = defaultdict(int)
    for d in new_nmd.tolist():
        nodes_by_depth[d] += 1
    for d in new_emd.tolist():
        edges_by_depth[d] += 1

    new_fstrs = [formula_arr_to_str(new_x[ni, :12].numpy()) for ni in range(new_n_nodes)]
    n_unique = len(set(new_fstrs))

    new_dag = dict(dag)
    new_dag["dag"] = new_g
    new_dag["nodes_min_depth"] = new_nmd
    new_dag["edges_min_depth"] = new_emd
    new_dag["dag_num_nodes"] = new_n_nodes
    new_dag["dag_num_edges"] = new_n_edges
    new_dag["dag_num_nodes_nb"] = -1
    new_dag["dag_num_nodes_by_depth"] = dict(nodes_by_depth)
    new_dag["dag_num_edges_by_depth"] = dict(edges_by_depth)
    new_dag["formula_redundancy"] = new_n_nodes / n_unique if n_unique > 0 else 1.0
    new_dag["dag_sparsity"] = new_n_edges / (new_n_nodes ** 2) if new_n_nodes > 1 else 0.0

    stats = {
        "n_nodes_before": n_nodes,
        "n_nodes_after": new_n_nodes,
        "n_edges_before": n_edges,
        "n_edges_after": new_n_edges,
        "orphans": len(remaining - reachable),
    }
    return new_dag, stats


def dedup_dag(dag: dict, strategy: str) -> tuple[dict, dict]:
    g = dag["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    nmd = dag["nodes_min_depth"]
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")
    h_formulae_idx = get_node_feats(x, node_feat_idxs, "h_formulae_idx")
    n_nodes = x.shape[0]

    node_depth = nmd.tolist()
    node_fstr = [formula_arr_to_str(base_formula[ni].numpy()) for ni in range(n_nodes)]

    remove_set = build_remove_set(strategy, node_depth, node_fstr, h_formulae_idx, n_nodes)
    return apply_remove_set(dag, remove_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Source dataset directory")
    parser.add_argument("--output_dir", required=True, help="Destination directory")
    parser.add_argument(
        "--strategy",
        default="hybrid",
        choices=["leaf_only", "all_depths", "hybrid", "h_coverage"],
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dags_in = input_dir / "dags"
    dags_out = output_dir / "dags"

    if not dags_in.exists():
        raise FileNotFoundError(f"No dags/ subdirectory found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dags_out.mkdir(exist_ok=True)

    # Copy non-dag files unchanged
    for p in input_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, output_dir / p.name)
            print(f"  copied {p.name}")

    dag_files = sorted(dags_in.glob("*.pickle.bz2")) + sorted(dags_in.glob("*.pkl.bz2"))
    n_total = len(dag_files)
    print(f"\nInput : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"DAG files: {n_total:,}\n")

    totals: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()

    for i, dag_path in enumerate(dag_files):
        if i % 50 == 0:
            print(f"  {i:>5}/{n_total}  ({time.perf_counter()-t0:.0f}s)")

        with open(dag_path, "rb") as fh:
            dag = pickle.loads(bz2.decompress(fh.read()))

        if dag["reached_depth"] < 1:
            out_dag = dag
        else:
            try:
                out_dag, stats = dedup_dag(dag, args.strategy)
                for k, v in stats.items():
                    totals[k] += v
            except Exception as e:
                print(f"  [WARN] {dag_path.name}: {e}")
                out_dag = dag

        out_path = dags_out / dag_path.name
        with open(out_path, "wb") as fh:
            fh.write(bz2.compress(pickle.dumps(out_dag)))

    elapsed = time.perf_counter() - t0
    nn_b, nn_a = totals["n_nodes_before"], totals["n_nodes_after"]
    ne_b, ne_a = totals["n_edges_before"], totals["n_edges_after"]

    print(f"\nDone in {elapsed:.0f}s")
    print(f"\n{'='*52}")
    print(f"{'':30s} {'before':>10} {'after':>10}")
    print(f"{'-'*52}")
    print(f"  {'nodes':<28} {nn_b:>10,} {nn_a:>10,}")
    print(f"  {'edges':<28} {ne_b:>10,} {ne_a:>10,}")
    print(f"  {'node reduction':<28} {'—':>10} {100*(nn_b-nn_a)/nn_b if nn_b else 0:>9.1f}%")
    print(f"  {'edge reduction':<28} {'—':>10} {100*(ne_b-ne_a)/ne_b if ne_b else 0:>9.1f}%")
    print(f"  {'orphans removed':<28} {totals['orphans']:>10,}")
    print(f"\nOutput written to: {output_dir}")


if __name__ == "__main__":
    main()
