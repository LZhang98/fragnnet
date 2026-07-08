"""Merge two DAG tar files: keep every node from ``--base_tar`` unchanged, and add
the leaf-depth nodes from ``--extra_tar`` whose ``(base_formula, h_min, h_max)`` key
is not already present anywhere in the base molecule's DAG.

Background (see conversation record / docs/bond_breakability_mask_plan.md lineage):
``base_tar`` is expected to be the plain d3 DAG (``..._smpFalse``) and ``extra_tar``
the unrestricted-ring-opening multi-cut BFS DAG (``..._mbfs_old``), whose extra
ring-opened fragments were shown to add real (non-random, chance-corrected) signal
for ~36% of molecules. A naive "add everything extra has beyond base" merge grows
the dataset ~2x, but ~81% of that added bulk is ``extra`` duplicating itself
(multiple cut paths converging on the same new formula/h-range) -- deduping the
added content against itself first (this script) cuts that growth to ~+21%.

Dedup/select rule for extra's nodes:
  - cutoff = extra's own max node depth (its true leaves).
  - Only nodes at that leaf depth are considered for addition (matches the
    depth-gating rationale in run_dag_postproc_dedup.py: shallower nodes are
    structurally load-bearing ancestors within `extra`'s own graph, not
    freestanding candidates).
  - A leaf node is added only if its (base_formula, h_min, h_max) key is not
    already present ANYWHERE in base (any depth) -- base is never touched.
  - Among the remaining candidates, keep first-seen (node-index order) per key.

Edge handling for added nodes: PLACEHOLDER ONLY. Each added node gets a single
edge from the root (node 0, verified invariant in this pipeline) directly to
itself, with a zero-filled edge_attr row. This is topologically wrong (the true
ancestor chain lives in `extra`'s own indexing, not base's, and reconstructing
it would require recursively merging `extra`'s ancestor subgraph too), but is
harmless for every config this was built against (`frag_gnn_type: NodeMLP`,
`pyg_edges: False` -- the model never reads edge_index). Do NOT point a
`pyg_edges: True` config at output from this script without fixing this.

Formula-table reconciliation: base and extra each carry their own independently
-generated per-molecule formula table (`idx_to_formula` / `formula_peak_mzs` /
`formula_peak_probs` / `idx_by_h_delta`), so `h_formulae_idx` / `cmf_h_formulae_idx`
values from `extra` are NOT valid indices into base's table. Base's table is kept
as-is (indices 0..N_base-1 unchanged); extra's formulas are matched to base's by
formula STRING (index 0 is always the '' sentinel in both, by construction --
empty string sorts first), and any extra formula not already in base is appended
as a new index. Extra's `h_formulae_idx`/`cmf_h_formulae_idx` columns for added
nodes are remapped through this old->new lookup before being copied in.

Usage:
    python scripts/run_dag_postproc_merge.py \\
        --base_tar   data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \\
        --extra_tar  data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs_old.tar \\
        --output_tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_ringmerge.tar
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
from fragnnet.utils.frag_utils import NODE_FEAT_TO_IDX, get_node_feats  # noqa: E402

ROOT_NODE_IDX = 0


def node_keys(x: th.Tensor, node_feat_idxs: th.Tensor) -> list[tuple]:
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula").numpy()
    h_counts = get_node_feats(x, node_feat_idxs, "h_counts").numpy()
    keys = []
    for ni in range(x.shape[0]):
        valid = [v for v in h_counts[ni].tolist() if v != -1]
        hmin, hmax = (min(valid), max(valid)) if valid else (0, 0)
        keys.append((tuple(base_formula[ni].tolist()), hmin, hmax))
    return keys


def select_added_nodes(base_dag: dict, extra_dag: dict) -> list[int]:
    """Return extra's node indices to add: leaf-depth, key not already in base,
    first-seen-wins among themselves."""
    g_base = base_dag["dag"]
    g_extra = extra_dag["dag"]
    n_extra = g_extra.x.shape[0]
    if n_extra == 0:
        return []

    base_keys = set(node_keys(g_base.x, g_base.node_feat_idxs[0]))

    depth_extra = np.asarray(extra_dag["nodes_min_depth"]).tolist()
    cutoff = max(depth_extra)
    keys_extra = node_keys(g_extra.x, g_extra.node_feat_idxs[0])

    seen: set = set()
    selected: list[int] = []
    for ni in range(n_extra):
        if depth_extra[ni] != cutoff:
            continue
        k = keys_extra[ni]
        if k in base_keys or k in seen:
            continue
        seen.add(k)
        selected.append(ni)
    return selected


def reconcile_formula_table(base_dag: dict, extra_dag: dict):
    """Build extra_old_idx -> new_idx array, and the merged formula-table fields.

    Returns:
        Tuple of (extra_old_to_new: np.ndarray, new_idx_to_formula: dict,
        new_formula_peak_mzs: th.Tensor, new_formula_peak_probs: th.Tensor,
        new_idx_by_h_delta: list[set]).
    """
    base_idx_to_formula: dict = base_dag["idx_to_formula"]
    extra_idx_to_formula: dict = extra_dag["idx_to_formula"]
    formula_to_new_idx = {f: i for i, f in base_idx_to_formula.items()}

    new_idx_to_formula = dict(base_idx_to_formula)
    new_mzs_rows = [base_dag["formula_peak_mzs"]]
    new_probs_rows = [base_dag["formula_peak_probs"]]

    n_extra_formula = len(extra_idx_to_formula)
    extra_old_to_new = np.zeros(n_extra_formula, dtype=np.int64)
    next_idx = len(base_idx_to_formula)
    for old_idx in range(n_extra_formula):
        formula = extra_idx_to_formula[old_idx]
        if formula in formula_to_new_idx:
            extra_old_to_new[old_idx] = formula_to_new_idx[formula]
        else:
            formula_to_new_idx[formula] = next_idx
            new_idx_to_formula[next_idx] = formula
            extra_old_to_new[old_idx] = next_idx
            new_mzs_rows.append(extra_dag["formula_peak_mzs"][old_idx : old_idx + 1])
            new_probs_rows.append(extra_dag["formula_peak_probs"][old_idx : old_idx + 1])
            next_idx += 1

    new_formula_peak_mzs = th.cat(new_mzs_rows, dim=0)
    new_formula_peak_probs = th.cat(new_probs_rows, dim=0)

    new_idx_by_h_delta = [set(s) for s in base_dag["idx_by_h_delta"]]
    for slot, extra_slot_set in enumerate(extra_dag["idx_by_h_delta"]):
        for old_idx in extra_slot_set:
            new_idx_by_h_delta[slot].add(int(extra_old_to_new[old_idx]))

    return extra_old_to_new, new_idx_to_formula, new_formula_peak_mzs, new_formula_peak_probs, new_idx_by_h_delta


def remap_formula_columns(x: th.Tensor, node_feat_idxs: th.Tensor, extra_old_to_new: np.ndarray) -> th.Tensor:
    """Return a copy of x with h_formulae_idx / cmf_h_formulae_idx columns remapped
    through extra_old_to_new. Other columns are left untouched."""
    x = x.clone()
    lut = th.as_tensor(extra_old_to_new, dtype=x.dtype)
    for key in ("h_formulae_idx", "cmf_h_formulae_idx"):
        idx = NODE_FEAT_TO_IDX[key]
        start, end = int(node_feat_idxs[idx]), int(node_feat_idxs[idx + 1])
        if end == start:
            continue
        x[:, start:end] = lut[x[:, start:end]]
    return x


def merge_dag(base_dag: dict, extra_dag: dict) -> tuple[dict, dict]:
    g_base = base_dag["dag"]
    n_base = g_base.x.shape[0]

    selected = select_added_nodes(base_dag, extra_dag)
    if not selected:
        stats = dict(n_added=0, n_base=n_base, n_final=n_base)
        return base_dag, stats

    g_extra = extra_dag["dag"]
    extra_old_to_new, new_idx_to_formula, new_mzs, new_probs, new_idx_by_h_delta = reconcile_formula_table(
        base_dag, extra_dag
    )

    extra_x_remapped = remap_formula_columns(g_extra.x, g_extra.node_feat_idxs[0], extra_old_to_new)
    sel_t = th.as_tensor(selected, dtype=th.long)
    added_x = extra_x_remapped[sel_t]
    new_x = th.cat([g_base.x, added_x], dim=0)

    depth_extra = np.asarray(extra_dag["nodes_min_depth"])
    added_depth = depth_extra[np.asarray(selected)]
    new_nmd = np.concatenate([np.asarray(base_dag["nodes_min_depth"]), added_depth])

    n_added = len(selected)
    n_final = n_base + n_added

    # base edges are untouched (node indices 0..n_base-1 unchanged); each added
    # node gets one placeholder root edge (see module docstring).
    new_node_idxs = th.arange(n_base, n_final, dtype=th.long)
    placeholder_src = th.full((n_added,), ROOT_NODE_IDX, dtype=th.long)
    placeholder_edges = th.stack([placeholder_src, new_node_idxs], dim=0)
    new_edge_index = th.cat([g_base.edge_index, placeholder_edges], dim=1)

    placeholder_edge_attr = th.zeros((n_added, g_base.edge_attr.shape[1]), dtype=g_base.edge_attr.dtype)
    new_edge_attr = th.cat([g_base.edge_attr, placeholder_edge_attr], dim=0)

    base_emd = np.asarray(base_dag["edges_min_depth"])
    new_emd = np.concatenate([base_emd, added_depth])

    # boundary_pair_*: base rows kept as-is; extra rows filtered to selected nodes
    # and frag_idx remapped to the node's new position in the merged graph.
    old_to_new_node = {old: n_base + i for i, old in enumerate(selected)}
    bp_frag_extra = g_extra.boundary_pair_frag_idxs.numpy()
    bp_keep_mask = np.isin(bp_frag_extra, selected)
    bp_frag_kept = bp_frag_extra[bp_keep_mask]
    bp_frag_remapped = np.array([old_to_new_node[i] for i in bp_frag_kept], dtype=np.int64)

    new_bp_frag = th.cat(
        [g_base.boundary_pair_frag_idxs, th.as_tensor(bp_frag_remapped, dtype=g_base.boundary_pair_frag_idxs.dtype)]
    )
    bp_keep_mask_t = th.as_tensor(bp_keep_mask)
    new_bp_in = th.cat([g_base.boundary_pair_in_local, g_extra.boundary_pair_in_local[bp_keep_mask_t]])
    new_bp_out = th.cat([g_base.boundary_pair_out_local, g_extra.boundary_pair_out_local[bp_keep_mask_t]])

    new_g = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    new_g.node_feat_idxs = g_base.node_feat_idxs
    new_g.edge_feat_idxs = g_base.edge_feat_idxs
    new_g.boundary_pair_frag_idxs = new_bp_frag
    new_g.boundary_pair_in_local = new_bp_in
    new_g.boundary_pair_out_local = new_bp_out

    nodes_by_depth: dict[int, int] = defaultdict(int)
    for d in new_nmd.tolist():
        nodes_by_depth[d] += 1
    edges_by_depth: dict[int, int] = defaultdict(int)
    for d in new_emd.tolist():
        edges_by_depth[d] += 1

    new_base_formula = get_node_feats(new_x, new_g.node_feat_idxs[0], "base_formula").numpy()
    n_unique_formula = len({tuple(row) for row in new_base_formula.tolist()})
    new_formula_redundancy = n_final / n_unique_formula if n_unique_formula > 0 else 1.0

    new_dag = dict(base_dag)
    new_dag["dag"] = new_g
    new_dag["nodes_min_depth"] = new_nmd
    new_dag["edges_min_depth"] = new_emd
    new_dag["dag_num_nodes"] = n_final
    new_dag["dag_num_edges"] = int(new_edge_index.shape[1])
    new_dag["dag_num_nodes_nb"] = -1  # sentinel; unused in training
    new_dag["dag_num_nodes_by_depth"] = dict(nodes_by_depth)
    new_dag["dag_num_edges_by_depth"] = dict(edges_by_depth)
    new_dag["formula_redundancy"] = new_formula_redundancy
    new_dag["dag_sparsity"] = new_dag["dag_num_edges"] / (n_final**2) if n_final > 1 else 0.0
    new_dag["idx_to_formula"] = new_idx_to_formula
    new_dag["formula_peak_mzs"] = new_mzs
    new_dag["formula_peak_probs"] = new_probs
    new_dag["idx_by_h_delta"] = new_idx_by_h_delta
    new_dag["reached_depth"] = max(base_dag["reached_depth"], extra_dag["reached_depth"])
    new_dag["force_stopped"] = bool(base_dag["force_stopped"] or extra_dag["force_stopped"])

    stats = dict(n_added=n_added, n_base=n_base, n_final=n_final)
    return new_dag, stats


def process_tars(base_tar: str, extra_tar: str, output_tar: str) -> None:
    print(f"Base:   {base_tar}")
    print(f"Extra:  {extra_tar}")
    print(f"Output: {output_tar}\n")

    base_base = os.path.basename(base_tar).replace(".tar", "")
    output_base = os.path.basename(output_tar).replace(".tar", "")

    def rewrite_name(name: str) -> str:
        if name == base_base:
            return output_base
        if name.startswith(base_base + "/"):
            return output_base + name[len(base_base):]
        return name

    with tarfile.open(base_tar, "r") as tf:
        base_members = tf.getmembers()
    with tarfile.open(extra_tar, "r") as tf:
        extra_names = {m.name.split("/")[-1]: m.name for m in tf.getmembers() if m.name.endswith(".pkl.bz2")}

    dag_members = [m for m in base_members if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    other_members = [m for m in base_members if m not in dag_members]
    print(f"  DAG members:   {len(dag_members):,}")
    print(f"  Other members: {len(other_members):,}\n")

    totals: dict[str, int] = defaultdict(int)
    n_no_extra_match = 0
    n_failed = 0
    t0 = time.perf_counter()

    with tarfile.open(base_tar, "r") as tf_base, tarfile.open(extra_tar, "r") as tf_extra, tarfile.open(
        output_tar, "w"
    ) as tf_out:
        for m in other_members:
            f = tf_base.extractfile(m)
            if f is not None:
                data = f.read()
                info = tarfile.TarInfo(name=rewrite_name(m.name))
                info.size = len(data)
                tf_out.addfile(info, io.BytesIO(data))
            else:
                info = copy.copy(m)
                info.name = rewrite_name(m.name)
                tf_out.addfile(info)

        for i, member in enumerate(dag_members):
            if i % 2000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i:>6}/{len(dag_members):,}  ({elapsed:.0f}s)", flush=True)

            f = tf_base.extractfile(member)
            if f is None:
                continue
            base_dag = pickle.loads(bz2.decompress(f.read()))

            leaf_name = member.name.split("/")[-1]
            extra_member_name = extra_names.get(leaf_name)
            if extra_member_name is None:
                n_no_extra_match += 1
                new_dag = base_dag
                stats = dict(n_added=0, n_base=base_dag["dag_num_nodes"], n_final=base_dag["dag_num_nodes"])
            else:
                f_extra = tf_extra.extractfile(extra_member_name)
                extra_dag = pickle.loads(bz2.decompress(f_extra.read()))
                try:
                    new_dag, stats = merge_dag(base_dag, extra_dag)
                except Exception as e:
                    n_failed += 1
                    print(f"  [WARN] {member.name}: merge failed ({e}), copying base unchanged")
                    new_dag = base_dag
                    stats = dict(n_added=0, n_base=base_dag["dag_num_nodes"], n_final=base_dag["dag_num_nodes"])

            raw = bz2.compress(pickle.dumps(new_dag))
            info = tarfile.TarInfo(name=rewrite_name(member.name))
            info.size = len(raw)
            tf_out.addfile(info, io.BytesIO(raw))

            totals["n_base"] += stats["n_base"]
            totals["n_final"] += stats["n_final"]
            totals["n_added"] += stats["n_added"]

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s ({n_failed} molecules failed merge, copied base unchanged;")
    print(f"  {n_no_extra_match} molecules had no matching extra_tar entry)")
    print(f"\n{'=' * 64}")
    print(f"  base nodes:  {totals['n_base']:>14,}")
    print(f"  final nodes: {totals['n_final']:>14,}")
    print(f"  added nodes: {totals['n_added']:>14,}  ({100 * totals['n_added'] / totals['n_base']:.1f}% growth)")
    print(f"\nOutput written to: {output_tar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_tar", required=True)
    parser.add_argument("--extra_tar", required=True)
    parser.add_argument("--output_tar", required=True)
    args = parser.parse_args()
    process_tars(args.base_tar, args.extra_tar, args.output_tar)


if __name__ == "__main__":
    main()
