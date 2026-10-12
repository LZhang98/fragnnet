"""Analyze DAG boundary bond types (C-C vs heteroatom) by depth and their contribution
to matched formula recall. Used to evaluate whether suppressing C-C cuts at deeper
depths can reduce DAG size without significant recall loss.

Usage:
    python scripts/debug/dag/debug_dag_boundary_bond_stats.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --n_sample 500 \
        --seed 42
"""

import argparse
import bz2
import pickle
import random
import sys
import tarfile
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
from fragnnet.models.fragnnet.frag_utils import get_edge_feats, get_node_feats


def is_cc_cut(complement_vec: list[float]) -> bool:
    """Returns True if the complement fragment contains no heteroatoms (pure C-C cut).

    The base_formula element layout is always [C, ...heavy elements..., H]
    (C at index 0, H at the last index), regardless of how many heavy
    elements the schema had at the time the DAG was generated. So heteroatoms
    are everything strictly between index 0 and the last index.
    """
    return all(v == 0 for v in complement_vec[1:-1])


def analyze_dag(dag: dict) -> tuple[dict, int] | None:
    """Analyze a single molecule's fragmentation DAG.

    For each unique formula that matches at least one spectrum peak, finds the
    minimum depth at which it first appears and classifies the incoming cut type
    (C-C only vs. heteroatom-containing).

    Args:
        dag: Loaded DAG dict from pkl.bz2 file.

    Returns:
        Tuple of (depth_stats, total_matched_formulas), or None if no matches.
        depth_stats maps depth -> {'n_formulas', 'cc_only', 'hetero'}.
    """
    g = dag["dag"]
    x = g.x
    edge_attr = g.edge_attr
    edge_index = g.edge_index
    node_feat_idxs = g.node_feat_idxs[0]
    edge_feat_idxs = g.edge_feat_idxs[0]
    n_nodes = x.shape[0]

    nmd = dag["nodes_min_depth"]  # [n_nodes] int array
    emd = dag["edges_min_depth"]  # [n_edges] int array

    edge_formula = get_edge_feats(edge_attr, edge_feat_idxs, "base_formula")  # [n_edges, 12]
    h_formulae_idx = get_node_feats(x, node_feat_idxs, "h_formulae_idx")  # [n_nodes, 9]
    fpp = dag["formula_peak_probs"]  # [n_formulae, n_spectra]

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    def edge_depth(ei: int) -> int:
        v = emd[ei]
        return v.item() if hasattr(v, "item") else int(v)

    # For each unique formula that matches any peak, find its minimum node depth.
    formula_to_min_depth: dict[int, int] = {}
    for ni in range(n_nodes):
        d = node_depth(ni)
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0 or fidx >= fpp.shape[0]:
                continue
            if fpp[fidx].sum() == 0:
                continue
            if fidx not in formula_to_min_depth or d < formula_to_min_depth[fidx]:
                formula_to_min_depth[fidx] = d

    if not formula_to_min_depth:
        return None

    # Build formula -> list of node indices
    formula_to_nodes: dict[int, list[int]] = defaultdict(list)
    for ni in range(n_nodes):
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx > 0:
                formula_to_nodes[fidx].append(ni)

    # Build node -> depth -> [is_cc_cut] for incoming edges
    incoming_by_depth: dict[int, dict[int, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ei in range(edge_index.shape[1]):
        child = edge_index[1, ei].item()
        d = edge_depth(ei)
        cc = is_cc_cut(edge_formula[ei].tolist())
        incoming_by_depth[child][d].append(cc)

    # For each matched formula, classify cut type at its min depth
    depth_stats: dict[int, dict] = defaultdict(lambda: {"n_formulas": 0, "cc_only": 0, "hetero": 0})

    for fidx, min_d in formula_to_min_depth.items():
        depth_stats[min_d]["n_formulas"] += 1

        if min_d == 0:
            continue  # root node, no cut

        nodes_at_min_d = [
            ni for ni in formula_to_nodes[fidx] if node_depth(ni) == min_d
        ]

        has_hetero = False
        has_cc = False
        for ni in nodes_at_min_d:
            cuts = incoming_by_depth[ni].get(min_d, [])
            if not cuts:
                continue
            if all(cuts):
                has_cc = True
            else:
                has_hetero = True

        if has_hetero:
            depth_stats[min_d]["hetero"] += 1
        elif has_cc:
            depth_stats[min_d]["cc_only"] += 1

    return dict(depth_stats), len(formula_to_min_depth)


def print_results(agg: dict, total: int, n_analyzed: int) -> None:
    print(f"\nAnalyzed {n_analyzed} molecules | {total:,} total matched formula-peak pairs")
    print(f"\n{'='*88}")
    print(
        f"{'Depth':<8} {'NewForms':>10} {'Cumul%':>10} "
        f"{'CC-only#':>10} {'CC%':>8} {'Hetero#':>10} {'Hetero%':>8}"
    )
    print(f"{'-'*88}")

    cumul = 0
    for d in sorted(agg.keys()):
        s = agg[d]
        n = s["n_formulas"]
        cumul += n
        cc = s["cc_only"]
        het = s["hetero"]
        print(
            f"  d={d}    {n:>10,} {100*cumul/total:>9.2f}% "
            f"{cc:>10,} {100*cc/n if n else 0:>7.1f}% "
            f"{het:>10,} {100*het/n if n else 0:>7.1f}%"
        )

    max_d = max(agg.keys())
    d_max = agg[max_d]

    print(f"\n--- What if we skip C-C cuts at depth {max_d} only? ---")
    cum_before_max = sum(agg[d]["n_formulas"] for d in sorted(agg.keys()) if d < max_d)
    retained = cum_before_max + d_max["hetero"]
    print(f"  Full recall:                 100.00%")
    print(f"  D{max_d-1}-only recall:              {100*cum_before_max/total:.2f}%")
    print(f"  Skip CC at d={max_d} recall:     {100*retained/total:.2f}%")
    print(f"  Loss from skipping CC at d={max_d}: {100*(total-retained)/total:.2f}%")

    if max_d >= 4:
        print(f"\n--- What if we skip C-C cuts at depth >= {max_d-1}? ---")
        cum_before = sum(agg[d]["n_formulas"] for d in sorted(agg.keys()) if d < max_d - 1)
        het_gain = sum(agg[d]["hetero"] for d in sorted(agg.keys()) if d >= max_d - 1)
        retained2 = cum_before + het_gain
        print(f"  Skip CC at d>={max_d-1} recall:   {100*retained2/total:.2f}%")
        print(f"  Loss:                        {100*(total-retained2)/total:.2f}%")

    print(f"\n--- C-C fraction trend by depth ---")
    for d in sorted(agg.keys()):
        s = agg[d]
        n = s["n_formulas"]
        cc_pct = 100 * s["cc_only"] / n if n > 0 else 0
        bar = "█" * int(cc_pct / 5)
        print(f"  d={d}: {cc_pct:5.1f}% CC  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tar",
        required=True,
        help="Path to DAG tar file (e.g. data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar)",
    )
    parser.add_argument("--n_sample", type=int, default=500, help="Number of molecules to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=50000,
        help="Skip DAGs with more nodes than this (combinatorial-blowup outliers)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.tar} ...")
    with tarfile.open(args.tar, "r") as tf:
        members = [
            m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name
        ]
    print(f"  Total DAG files: {len(members):,}")

    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    print(f"  Sampling {len(sample):,} molecules ...")

    agg: dict[int, dict] = defaultdict(lambda: {"n_formulas": 0, "cc_only": 0, "hetero": 0})
    total_matched = 0
    n_analyzed = 0
    n_skipped_large = 0

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
            result = analyze_dag(dag)
            if result is None:
                continue
            depth_stats, n_matched = result
            for d, s in depth_stats.items():
                agg[d]["n_formulas"] += s["n_formulas"]
                agg[d]["cc_only"] += s["cc_only"]
                agg[d]["hetero"] += s["hetero"]
            total_matched += n_matched
            n_analyzed += 1

    print(f"  Skipped {n_skipped_large:,} molecules with > {args.max_nodes:,} nodes (combinatorial-blowup outliers)")

    if total_matched == 0:
        print("No matched formulas found.")
        return

    print_results(dict(agg), total_matched, n_analyzed)


if __name__ == "__main__":
    main()
