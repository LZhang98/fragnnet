"""Create deduplicated D3 tar files for all 4 strategies, then compute h-formula
recall broken down by h-delta (h1..h4) for each strategy on the full NIST20 set.

Strategies:
  leaf_only  : remove max-depth nodes whose base formula exists at shallower depth
  all_depths : remove any node whose base formula exists at a shallower depth
  hybrid     : remove deeper nodes only when all h-indices already covered by kept
               shallower nodes (0% h-recall loss, preserves min-depth structure)
  h_coverage : remove any node whose h-index set is covered by already-seen nodes
               (0% h-recall loss, most aggressive, deduplicates same-depth too)

Recall is computed per h-delta threshold:
  h1 = only formulas with |h_delta| <= 1 count as matchable
  h2 = |h_delta| <= 2, etc.
  This shows whether dedup disproportionately affects high-H-transfer formulas.

Usage:
    python scripts/run_dag_dedup_compare.py \
        --input_tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \
        --out_dir   data/frag/dedup \
        --strategies leaf_only all_depths hybrid h_coverage \
        --recall_sample 0          # 0 = all molecules
"""

import argparse
import bz2
import concurrent.futures
import io
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

ALL_STRATEGIES = ["leaf_only", "all_depths", "hybrid", "h_coverage"]


# ---------------------------------------------------------------------------
# Helpers
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
    """Return the set of node indices to remove for the given strategy."""
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

    raise ValueError(f"Unknown strategy: {strategy}")


def apply_remove_set(dag: dict, remove_set: set[int]) -> tuple[dict, dict]:
    """Remove nodes, fix orphans, rebuild PyG graph. Returns (new_dag, stats)."""
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
    edge_attr = g.edge_attr
    nmd: np.ndarray = dag["nodes_min_depth"]
    emd: np.ndarray = dag["edges_min_depth"]
    n_nodes = x.shape[0]
    n_edges = edge_index.shape[1]

    # Reachability: remove orphaned nodes too
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
# Tar creation
# ---------------------------------------------------------------------------

def create_dedup_tar(input_tar: str, output_tar: str, strategy: str) -> dict:
    """Write deduplicated tar; return aggregate stats."""
    totals: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()

    with tarfile.open(input_tar, "r") as tf:
        members = tf.getmembers()

    dag_members = [m for m in members if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    other_members = [m for m in members if m not in dag_members]

    with tarfile.open(input_tar, "r") as tf_in, tarfile.open(output_tar, "w") as tf_out:
        for m in other_members:
            f = tf_in.extractfile(m)
            if f is not None:
                data = f.read()
                info = tarfile.TarInfo(name=m.name)
                info.size = len(data)
                tf_out.addfile(info, io.BytesIO(data))
            else:
                tf_out.addfile(m)

        for i, member in enumerate(dag_members):
            if i % 5000 == 0:
                print(f"    [{strategy}] {i:>6}/{len(dag_members):,}  ({time.perf_counter()-t0:.0f}s)")
            f = tf_in.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))

            if dag["reached_depth"] < 1:
                raw = bz2.compress(pickle.dumps(dag))
            else:
                try:
                    new_dag, stats = dedup_dag(dag, strategy)
                    for k, v in stats.items():
                        totals[k] += v
                    raw = bz2.compress(pickle.dumps(new_dag))
                except Exception as e:
                    print(f"    [WARN] {member.name}: {e}")
                    raw = bz2.compress(pickle.dumps(dag))

            info = tarfile.TarInfo(name=member.name)
            info.size = len(raw)
            tf_out.addfile(info, io.BytesIO(raw))

    totals["elapsed"] = int(time.perf_counter() - t0)
    return dict(totals)


# ---------------------------------------------------------------------------
# Recall evaluation (h1..h4 breakdown)
# ---------------------------------------------------------------------------

def compute_recall(tar_path: str, n_sample: int = 0, seed: int = 42) -> dict:
    """Compute matched h-formula recall broken down by h-delta threshold.

    Returns dict: h_max → recall fraction (0..1).
    Also returns "all" key (no h_delta restriction).
    """
    import random

    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]

    if n_sample > 0:
        random.seed(seed)
        members = random.sample(members, min(n_sample, len(members)))

    # Accumulators: for each h_max, count total matched and kept
    H_MAXES = [1, 2, 3, 4]
    total_matched: dict[int | str, int] = defaultdict(int)
    total_kept: dict[int | str, int] = defaultdict(int)

    with tarfile.open(tar_path, "r") as tf:
        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))
            if dag["reached_depth"] < 1:
                continue

            fpp = dag["formula_peak_probs"]          # [n_formulae, n_spectra]
            idx_by_h_delta: dict = dag["idx_by_h_delta"]  # h_delta → [fidx, ...]

            g = dag["dag"]
            x = g.x
            h_formulae_idx = get_node_feats(x, g.node_feat_idxs[0], "h_formulae_idx")
            n_nodes = x.shape[0]

            # h-formula indices present in this (possibly dedup'd) DAG
            present_fidxs: set[int] = set()
            for ni in range(n_nodes):
                for v in h_formulae_idx[ni]:
                    fidx = v.item()
                    if fidx > 0:
                        present_fidxs.add(fidx)

            # All matched fidxs (fpp > 0) grouped by |h_delta|
            # idx_by_h_delta is a list: [0]=h0, [1]=+1, [2]=-1, [3]=+2, [4]=-2, ...
            for h_max in H_MAXES:
                eligible: set[int] = set()
                for slot_idx in range(min(1 + 2 * h_max, len(idx_by_h_delta))):
                    eligible.update(int(f) for f in idx_by_h_delta[slot_idx])
                matched = {f for f in eligible if 1 <= f < fpp.shape[0] and fpp[f].sum() > 0}
                kept = matched & present_fidxs
                total_matched[h_max] += len(matched)
                total_kept[h_max] += len(kept)

            # "all" = no h_delta restriction
            all_matched = {f for f in range(1, fpp.shape[0]) if fpp[f].sum() > 0}
            total_matched["all"] += len(all_matched)
            total_kept["all"] += len(all_matched & present_fidxs)

    return {
        k: total_kept[k] / total_matched[k] if total_matched[k] > 0 else 1.0
        for k in list(H_MAXES) + ["all"]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_tar", required=True, help="Path to original D3 tar")
    parser.add_argument("--out_dir", required=True, help="Directory to write dedup tars")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES)
    parser.add_argument("--recall_sample", type=int, default=0,
                        help="Molecules to sample for recall (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall_only", action="store_true",
                        help="Skip tar creation (tars already exist); only compute recall")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.basename(args.input_tar).replace(".tar", "")
    tar_paths: dict[str, str] = {}

    # ---- Step 1: create dedup tars in parallel ----
    print(f"Creating {len(args.strategies)} deduplicated tars in parallel ...\n")
    for strategy in args.strategies:
        out_tar = os.path.join(args.out_dir, f"{base}_{strategy}.tar")
        tar_paths[strategy] = out_tar
        print(f"  [{strategy}] → {out_tar}")

    t0_all = time.perf_counter()
    create_stats: dict[str, dict] = {}
    if args.recall_only:
        print("  (--recall_only: skipping tar creation)\n")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(args.strategies)) as pool:
            futures = {
                pool.submit(create_dedup_tar, args.input_tar, tar_paths[s], s): s
                for s in args.strategies
            }
            for fut in concurrent.futures.as_completed(futures):
                strategy = futures[fut]
                stats = fut.result()
                create_stats[strategy] = stats
                nn_b, nn_a = stats.get("n_nodes_before", 0), stats.get("n_nodes_after", 0)
                ne_b, ne_a = stats.get("n_edges_before", 0), stats.get("n_edges_after", 0)
                print(f"  [{strategy}] done in {stats['elapsed']}s — "
                      f"nodes {nn_b:,}→{nn_a:,} ({100*(nn_b-nn_a)/nn_b if nn_b else 0:.1f}%) | "
                      f"edges {ne_b:,}→{ne_a:,} ({100*(ne_b-ne_a)/ne_b if ne_b else 0:.1f}%)")
        print(f"\nAll tars done in {time.perf_counter()-t0_all:.0f}s")

    # Also include baseline (original tar) in recall eval
    tar_paths["baseline"] = args.input_tar

    # ---- Step 2: compute recall in parallel ----
    sample_str = f"{args.recall_sample:,}" if args.recall_sample else "all"
    print(f"\nComputing recall (h1..h4) on {sample_str} molecules per tar in parallel ...\n")

    eval_order = ["baseline"] + args.strategies
    recall_results: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(eval_order)) as pool:
        futures = {
            pool.submit(compute_recall, tar_paths[label], args.recall_sample, args.seed): label
            for label in eval_order
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            recall_results[label] = fut.result()
            print(f"  [{label}] recall computed")

    # ---- Step 3: print table ----
    H_MAXES = [1, 2, 3, 4, "all"]
    col_w = 12
    header_cols = ["baseline"] + args.strategies
    print(f"\n{'='*(32 + col_w * len(header_cols))}")
    print(f"  h-formula recall by h_delta threshold")
    print(f"  (fraction of matched h-formula indices surviving dedup)")
    print(f"\n  {'':30s}" + "".join(f"{s:>{col_w}}" for s in header_cols))
    print(f"  {'-'*(30 + col_w * len(header_cols))}")
    for h in H_MAXES:
        label = f"h_max={h}" if h != "all" else "all h_delta"
        row = f"  {label:<30}"
        for s in header_cols:
            v = recall_results[s][h]
            row += f"{100*v:>{col_w-1}.2f}%"
        print(row)

    # ---- Step 4: save CSV ----
    import csv
    csv_path = os.path.join(args.out_dir, f"{base}_dedup_recall.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["h_max"] + header_cols)
        for h in H_MAXES:
            writer.writerow([h] + [f"{100*recall_results[s][h]:.4f}" for s in header_cols])
    print(f"\nRecall table saved to: {csv_path}")
    print(f"Dedup tars written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
