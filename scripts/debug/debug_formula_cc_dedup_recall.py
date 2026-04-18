"""Check matched-formula recall of the formula+CC-cut dedup vs baseline.

For a sample of molecules from the D3 tar:
  1. Load baseline DAG (from tar) → find formulas matched to peaks (fpp > 0)
  2. Run dedup BFS on the same molecule → compute its formula set
  3. Recall = matched formulas still present in dedup / matched formulas in baseline

Reports per-depth breakdown: which matched formulas are lost and at what depth.

Usage:
    python scripts/debug/debug_formula_cc_dedup_recall.py \
        --tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --n_sample 200 --max_depth 3
"""

import argparse
import bz2
import pickle
import random
import sys
import tarfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.frag import compute_frags
from fragnnet.utils.frag_utils import ELEMENT_TO_IDX, IDX_TO_ELEMENT


def formula_arr_to_str(arr: np.ndarray) -> str:
    parts = []
    for elem, idx in sorted(ELEMENT_TO_IDX.items(), key=lambda x: x[1]):
        if elem == "H":
            continue
        if arr[idx] > 0:
            parts.append(f"{elem}{arr[idx]}")
    return "".join(parts)


def get_matched_formula_strs(dag: dict) -> set[str]:
    """Return set of heavy-atom formula strings that match at least one peak."""
    from fragnnet.utils.frag_utils import get_node_feats, NODE_FEAT_TO_IDX
    import torch

    g = dag["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    h_formulae_idx = get_node_feats(x, node_feat_idxs, "h_formulae_idx")  # [n_nodes, 9]
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")  # [n_nodes, 12]
    fpp = dag["formula_peak_probs"]  # [n_formulae, n_spectra]
    nmd = dag["nodes_min_depth"]

    matched = set()
    for ni in range(x.shape[0]):
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0 or fidx >= fpp.shape[0]:
                continue
            if fpp[fidx].sum() > 0:
                # Use base formula of this node as the key (no H, depth-independent)
                fstr = formula_arr_to_str(base_formula[ni].numpy())
                matched.add(fstr)
                break
    return matched


def get_matched_formula_strs_by_depth(dag: dict) -> dict[int, set[str]]:
    """Return matched formula strings grouped by their min_depth in the baseline DAG."""
    from fragnnet.utils.frag_utils import get_node_feats
    import torch

    g = dag["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    h_formulae_idx = get_node_feats(x, node_feat_idxs, "h_formulae_idx")
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")
    fpp = dag["formula_peak_probs"]
    nmd = dag["nodes_min_depth"]

    by_depth: dict[int, set[str]] = defaultdict(set)
    for ni in range(x.shape[0]):
        d = int(nmd[ni]) if hasattr(nmd[ni], "item") else nmd[ni]
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0 or fidx >= fpp.shape[0]:
                continue
            if fpp[fidx].sum() > 0:
                fstr = formula_arr_to_str(base_formula[ni].numpy())
                by_depth[d].add(fstr)
                break
    return dict(by_depth)


def get_dedup_formula_strs(smiles: str, max_depth: int, max_h: int, max_time: int) -> set[str]:
    """Run dedup BFS and return the set of heavy-atom formula strings generated."""
    from fragnnet.utils.frag_utils import get_node_feats

    mol_d = frag_utils.extract_mol_info(smiles)
    _, dag_d = frag_utils.timed_get_dags(
        mol_d["mol"], 0,
        max_depth=max_depth, h_prior=True, max_h_transfer=max_h,
        max_time=max_time, isotopes=False, nb_isomorphic=False,
        wl_max_iterations=-1, multi_cut_bfs=False,
    )
    if not dag_d:
        return set()

    g = dag_d["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")  # [n_nodes, 12]
    formula_strs = set()
    for ni in range(x.shape[0]):
        formula_strs.add(formula_arr_to_str(base_formula[ni].numpy()))
    return formula_strs


def get_baseline_formula_strs(smiles: str, max_depth: int, max_h: int, max_time: int) -> set[str]:
    """Run baseline BFS (no dedup) and return formula strings."""
    _orig = compute_frags.compute_ccs

    def _no_dedup(n_nodes, n_edges, nm, edges, em, nte, md, tl, mfa,
                  is_cc_bond=None, elem_idxs_c=None):
        return _orig(n_nodes, n_edges, nm, edges, em, nte, md, tl, mfa)

    compute_frags.compute_ccs = _no_dedup
    mol_d = frag_utils.extract_mol_info(smiles)
    _, dag_d = frag_utils.timed_get_dags(
        mol_d["mol"], 0,
        max_depth=max_depth, h_prior=True, max_h_transfer=max_h,
        max_time=max_time, isotopes=False, nb_isomorphic=False,
        wl_max_iterations=-1, multi_cut_bfs=False,
    )
    from fragnnet.utils.frag_utils import get_node_feats

    compute_frags.compute_ccs = _orig
    if not dag_d:
        return set()

    g = dag_d["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    base_formula = get_node_feats(x, node_feat_idxs, "base_formula")  # [n_nodes, 12]
    formula_strs = set()
    for ni in range(x.shape[0]):
        formula_strs.add(formula_arr_to_str(base_formula[ni].numpy()))
    return formula_strs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True, help="Path to D3 DAG tar file")
    parser.add_argument("--mol_df", required=True, help="Path to mol_df.pkl")
    parser.add_argument("--n_sample", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_h", type=int, default=4)
    parser.add_argument("--max_time", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    mol_df = pickle.load(open(args.mol_df, "rb"))
    mol_id_to_smiles = dict(zip(mol_df["mol_id"], mol_df["smiles"]))

    print(f"Scanning {args.tar} ...")
    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    print(f"  {len(members):,} DAG files found")

    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    print(f"  Sampling {len(sample)} molecules ...\n")

    # Per-depth accumulators
    depth_total:  dict[int, int] = defaultdict(int)
    depth_kept:   dict[int, int] = defaultdict(int)

    total_matched = 0
    total_kept = 0
    total_lost = 0
    n_analyzed = 0
    n_skipped = 0

    with tarfile.open(args.tar, "r") as tf:
        for i, member in enumerate(sample):
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} ...")

            mol_id = int(member.name.split("/dags/")[1].replace(".pkl.bz2", ""))
            smiles = mol_id_to_smiles.get(mol_id)
            if smiles is None or smiles.startswith("*"):
                n_skipped += 1
                continue

            f = tf.extractfile(member)
            if f is None:
                n_skipped += 1
                continue
            baseline_dag = pickle.loads(bz2.decompress(f.read()))

            # Matched formulas from baseline tar DAG (ground truth peaks)
            matched_by_depth = get_matched_formula_strs_by_depth(baseline_dag)
            all_matched = set().union(*matched_by_depth.values()) if matched_by_depth else set()
            if not all_matched:
                n_skipped += 1
                continue

            # Formulas generated by the new dedup BFS
            try:
                dedup_formulas = get_dedup_formula_strs(smiles, args.max_depth, args.max_h, args.max_time)
            except Exception as e:
                print(f"  [mol {mol_id}] dedup error: {e}")
                n_skipped += 1
                continue

            # Per-depth recall
            for d, fset in matched_by_depth.items():
                for fstr in fset:
                    depth_total[d] += 1
                    if fstr in dedup_formulas:
                        depth_kept[d] += 1

            kept = all_matched & dedup_formulas
            lost = all_matched - dedup_formulas
            total_matched += len(all_matched)
            total_kept += len(kept)
            total_lost += len(lost)
            n_analyzed += 1

    print(f"\nAnalyzed {n_analyzed} molecules (skipped {n_skipped})")
    print(f"\n{'='*70}")
    print(f"Overall matched-formula recall (dedup vs baseline tar):")
    print(f"  Total matched formulas: {total_matched:,}")
    print(f"  Kept by dedup:          {total_kept:,}  ({100*total_kept/total_matched:.2f}%)")
    print(f"  Lost by dedup:          {total_lost:,}  ({100*total_lost/total_matched:.2f}%)")

    print(f"\nRecall by depth of first appearance in baseline:")
    print(f"  {'depth':<8} {'matched':>10} {'kept':>10} {'recall':>10}")
    print(f"  {'-'*40}")
    for d in sorted(depth_total.keys()):
        tot = depth_total[d]
        kept = depth_kept[d]
        print(f"  d={d}      {tot:>10,} {kept:>10,} {100*kept/tot:>9.2f}%")


if __name__ == "__main__":
    main()
