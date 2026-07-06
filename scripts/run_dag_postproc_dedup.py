"""Post-process a DAG tar file: trim redundant depth->=cutoff nodes whose
(base_formula, h_min, h_max) is already supplied by another surviving node, then
write a deduplicated copy of the tar.

Dedup rule (see docs/dag_full_recall_tractability_plan.md and conversation record for
how this was derived and validated):
  - cutoff = the DAG's actual max node depth (true leaves only, per-molecule).
  - Nodes with depth < cutoff are NEVER removed (keeps the DAG's connectivity intact;
    since only true leaves are ever removed, this cannot orphan any other node).
  - Among nodes with depth >= cutoff, group by (base_formula, h_min, h_max) -- this
    triple fully determines the set of h-transfer formula indices a node can supply
    (verified: achievable h_counts always form a contiguous range, no gaps). Keep
    only the first node per group (processed shallowest-first), drop the rest.

Two strategies that looked plausible but are NOT used here, because they were tested
and found to silently lose real matched-formula recall:
  - Deduping across ALL depths (not gating by cutoff): can orphan depth-3+ descendants
    when a depth-1/2 parent gets removed, even though nothing in the removal criterion
    itself looks wrong (measured: 0.14% real recall loss on the worst-case molecule).
  - Grouping by base_formula alone (ignoring h_min/h_max): two fragments with the same
    heavy-atom formula can have different achievable H-transfer ranges depending on
    where they were cut, so collapsing them loses real formula-table entries (measured:
    10-29% real recall loss, independent of the depth-gating bug above).

This version also fixes a gap in the prior (removed) implementation: it now filters
and remaps `boundary_pair_frag_idxs` / `boundary_pair_in_local` / `boundary_pair_out_local`
(the prior script dropped these three arrays entirely, which would silently break the
`boundary_pair` node feature used by mbfs configs).

Usage:
    python scripts/run_dag_postproc_dedup.py \
        --input_tar  data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs.tar \
        --output_tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs_trimmed.tar
"""

import argparse
import bz2
import copy
import io
import os
import pickle
import sys
import tarfile
import time
from collections import defaultdict

import numpy as np
import torch as th
from torch_geometric.data import Data

sys.path.insert(0, "src")
from fragnnet.utils.frag_utils import get_node_feats  # noqa: E402


def get_remove_set_hrange_cutoff(node_depth: list, node_key: list, n_nodes: int, cutoff: int) -> set[int]:
    """depth < cutoff: never removed. depth >= cutoff: keep first-seen (base_formula,
    h_min, h_max) group member (shallowest first), drop the rest."""
    seen_keys: set = set()
    remove_set: set[int] = set()
    order = sorted(range(n_nodes), key=lambda ni: node_depth[ni])
    for ni in order:
        d = node_depth[ni]
        k = node_key[ni]
        if d < cutoff:
            seen_keys.add(k)
            continue
        if k in seen_keys:
            remove_set.add(ni)
        else:
            seen_keys.add(k)
    return remove_set


def dedup_dag(dag: dict) -> tuple[dict, dict]:
    """Apply the (base_formula, h_min, h_max) / depth-cutoff dedup to one molecule DAG.

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
    nmd: np.ndarray = np.asarray(dag["nodes_min_depth"])
    emd: np.ndarray = np.asarray(dag["edges_min_depth"])

    n_nodes = x.shape[0]
    n_edges = edge_index.shape[1]

    base_formula = get_node_feats(x, node_feat_idxs, "base_formula").numpy()
    h_counts = get_node_feats(x, node_feat_idxs, "h_counts").numpy()

    node_depth = nmd.tolist()
    cutoff = int(nmd.max()) if n_nodes > 0 else 0

    node_key = []
    for ni in range(n_nodes):
        valid = [v for v in h_counts[ni].tolist() if v != -1]
        hmin, hmax = (min(valid), max(valid)) if valid else (0, 0)
        node_key.append((tuple(base_formula[ni].tolist()), hmin, hmax))

    remove_set = get_remove_set_hrange_cutoff(node_depth, node_key, n_nodes, cutoff)

    keep_nodes = sorted(set(range(n_nodes)) - remove_set)
    remap_arr = np.full(n_nodes, -1, dtype=np.int64)
    remap_arr[keep_nodes] = np.arange(len(keep_nodes))
    remaining_mask = remap_arr >= 0

    keep_node_idx = th.as_tensor(keep_nodes, dtype=th.long)
    new_x = x[keep_node_idx]
    new_nmd = nmd[np.asarray(keep_nodes, dtype=np.intp)]

    src_np, dst_np = edge_index[0].numpy(), edge_index[1].numpy()
    edge_keep_mask = remaining_mask[src_np] & remaining_mask[dst_np]
    kept_edge_idx = np.nonzero(edge_keep_mask)[0]
    new_src = remap_arr[src_np[kept_edge_idx]]
    new_dst = remap_arr[dst_np[kept_edge_idx]]
    new_edge_index = th.as_tensor(np.stack([new_src, new_dst]), dtype=th.long)
    kept_edge_idx_t = th.as_tensor(kept_edge_idx, dtype=th.long)
    new_edge_attr = edge_attr[kept_edge_idx_t]
    new_emd = emd[kept_edge_idx] if len(kept_edge_idx) else emd[:0]

    # boundary_pair_* : one entry per (fragment/node, boundary atom pair). Filter to
    # surviving nodes and remap the frag-idx column; atom-local indices are unaffected
    # by node removal, so those two arrays are just row-selected, not remapped.
    bp_frag_np = g.boundary_pair_frag_idxs.numpy()
    bp_keep_mask = remaining_mask[bp_frag_np]
    new_bp_frag = th.as_tensor(remap_arr[bp_frag_np[bp_keep_mask]], dtype=g.boundary_pair_frag_idxs.dtype)
    bp_keep_mask_t = th.as_tensor(bp_keep_mask)
    new_bp_in = g.boundary_pair_in_local[bp_keep_mask_t]
    new_bp_out = g.boundary_pair_out_local[bp_keep_mask_t]

    new_g = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    new_g.node_feat_idxs = g.node_feat_idxs
    new_g.edge_feat_idxs = g.edge_feat_idxs
    new_g.boundary_pair_frag_idxs = new_bp_frag
    new_g.boundary_pair_in_local = new_bp_in
    new_g.boundary_pair_out_local = new_bp_out

    new_n_nodes = len(keep_nodes)
    new_n_edges = int(kept_edge_idx.size)

    nodes_by_depth: dict[int, int] = defaultdict(int)
    for d in new_nmd.tolist():
        nodes_by_depth[d] += 1
    edges_by_depth: dict[int, int] = defaultdict(int)
    for d in new_emd.tolist():
        edges_by_depth[d] += 1

    new_base_formula = get_node_feats(new_x, node_feat_idxs, "base_formula").numpy()
    n_unique_formula = len({tuple(row) for row in new_base_formula.tolist()})
    new_formula_redundancy = new_n_nodes / n_unique_formula if n_unique_formula > 0 else 1.0

    new_dag = dict(dag)
    new_dag["dag"] = new_g
    new_dag["nodes_min_depth"] = new_nmd
    new_dag["edges_min_depth"] = new_emd
    new_dag["dag_num_nodes"] = new_n_nodes
    new_dag["dag_num_edges"] = new_n_edges
    new_dag["dag_num_nodes_nb"] = -1  # sentinel; unused in training
    new_dag["dag_num_nodes_by_depth"] = dict(nodes_by_depth)
    new_dag["dag_num_edges_by_depth"] = dict(edges_by_depth)
    new_dag["formula_redundancy"] = new_formula_redundancy
    new_dag["dag_sparsity"] = new_n_edges / (new_n_nodes**2) if new_n_nodes > 1 else 0.0
    # formula_peak_mzs / formula_peak_probs / idx_to_formula / idx_by_h_delta are left
    # untouched: h_formulae_idx / cmf_h_formulae_idx values in new_x still index the
    # same (unmodified) global formula table, so no remap is needed there.

    stats = {
        "n_nodes_before": n_nodes,
        "n_nodes_after": new_n_nodes,
        "n_edges_before": n_edges,
        "n_edges_after": new_n_edges,
    }
    return new_dag, stats


def process_tar(input_tar: str, output_tar: str) -> None:
    print(f"Input:  {input_tar}")
    print(f"Output: {output_tar}\n")

    input_base = os.path.basename(input_tar).replace(".tar", "")
    output_base = os.path.basename(output_tar).replace(".tar", "")

    def rewrite_name(name: str) -> str:
        if name == input_base:
            return output_base
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
    n_failed = 0
    t0 = time.perf_counter()

    with tarfile.open(input_tar, "r") as tf_in, tarfile.open(output_tar, "w") as tf_out:
        for m in other_members:
            f = tf_in.extractfile(m)
            if f is not None:
                data = f.read()
                info = tarfile.TarInfo(name=rewrite_name(m.name))
                info.size = len(data)
                tf_out.addfile(info, io.BytesIO(data))
            else:
                # Directory (or other non-extractable) entry: extractfile() returns
                # None, so there's no content to re-read -- still must rewrite the
                # name, or the old input_base prefix leaks into the output tar.
                info = copy.copy(m)
                info.name = rewrite_name(m.name)
                tf_out.addfile(info)

        for i, member in enumerate(dag_members):
            if i % 2000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i:>6}/{len(dag_members):,}  ({elapsed:.0f}s)", flush=True)

            f = tf_in.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))

            if dag["reached_depth"] < 1:
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
                n_failed += 1
                print(f"  [WARN] {member.name}: dedup failed ({e}), copying unchanged")
                new_dag = dag
                stats = {
                    "n_nodes_before": dag["dag_num_nodes"],
                    "n_nodes_after": dag["dag_num_nodes"],
                    "n_edges_before": dag["dag_num_edges"],
                    "n_edges_after": dag["dag_num_edges"],
                }

            raw = bz2.compress(pickle.dumps(new_dag))
            info = tarfile.TarInfo(name=rewrite_name(member.name))
            info.size = len(raw)
            tf_out.addfile(info, io.BytesIO(raw))

            for k, v in stats.items():
                totals[k] += v

    elapsed = time.perf_counter() - t0
    nn_b, nn_a = totals["n_nodes_before"], totals["n_nodes_after"]
    ne_b, ne_a = totals["n_edges_before"], totals["n_edges_after"]

    print(f"\nDone in {elapsed:.0f}s ({n_failed} molecules failed dedup and were copied unchanged)")
    print(f"\n{'=' * 64}")
    print(f"{'':32s} {'before':>12}  {'after':>12}")
    print(f"{'-' * 64}")
    print(f"  {'nodes':<30} {nn_b:>12,}  {nn_a:>12,}")
    print(f"  {'edges':<30} {ne_b:>12,}  {ne_a:>12,}")
    print(f"  {'node reduction':<30} {'-':>12}  {100 * (nn_b - nn_a) / nn_b:>11.1f}%")
    print(f"  {'edge reduction':<30} {'-':>12}  {100 * (ne_b - ne_a) / ne_b:>11.1f}%")
    print(f"\nOutput written to: {output_tar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_tar", required=True)
    parser.add_argument("--output_tar", required=True)
    args = parser.parse_args()
    process_tar(args.input_tar, args.output_tar)


if __name__ == "__main__":
    main()
