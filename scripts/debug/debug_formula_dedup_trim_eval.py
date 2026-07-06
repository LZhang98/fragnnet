"""Test an exact (non-statistical) trimming rule: at depth >= dedup_from_depth, drop any
node whose ENTIRE set of matched formulas is already achievable at a strictly shallower
depth. Since real recall (peak-to-formula matching) only cares whether SOME node exists
for a given formula, not which specific fragment structure represents it, such a node adds
nothing -- multiple structurally distinct fragments sharing the same elemental formula are
redundant for recall purposes even though they're distinct DAG nodes. This should be a
FREE cut: zero wrecall cost by construction, verified empirically below as a sanity check
on the implementation, not because it's actually a tradeoff.

A node is KEPT iff it is a "first achiever" of at least one of its matched formulas (its
own depth equals that formula's minimum depth anywhere in the DAG); DROPPED iff every one
of its matched formulas is already achieved by some strictly-shallower node.

Uses the exact same f_depth machinery (formula_depth_map) as
scripts/stats_scripts/dag_depth_recall_nist20_d3.py, on the same 97-molecule test split
used throughout this investigation.

Usage:
    python scripts/debug/debug_formula_dedup_trim_eval.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --spec_df data/proc/nist20/spec_df.pkl \
        --n_sample 400 \
        --seed 42 \
        --dedup_from_depth 3
"""

import argparse
import bz2
import os
import pickle
import random
import sys
import tarfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats_scripts"))
from dag_depth_recall_nist20_d3 import formula_depth_map, ibd_allowed, vectorized_recall  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES = {"FT", "QTOF", "IT"}
FRAG_MODES = {"CID", "HCD"}


def dedup_keep_set(dag: dict, f_depth: dict[int, int], dedup_from_depth: int) -> tuple[set[int], int]:
    """Return (kept_node_indices, total_nodes). A node at depth >= dedup_from_depth is
    kept only if it is a first-achiever of at least one of its matched formulas; nodes at
    depth < dedup_from_depth are always kept untouched.
    """
    g = dag["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    n_nodes = x.shape[0]
    h_start, h_end = node_feat_idxs[3].item(), node_feat_idxs[4].item()
    h_formulae_idx = x[:, h_start:h_end]
    nmd = dag["nodes_min_depth"]

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    kept: set[int] = set()
    for ni in range(n_nodes):
        d = node_depth(ni)
        if d < dedup_from_depth:
            kept.add(ni)
            continue
        is_first_achiever = False
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0:
                continue
            if f_depth.get(fidx) == d:
                is_first_achiever = True
                break
        if is_first_achiever:
            kept.add(ni)
    return kept, n_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--spec_df", required=True)
    parser.add_argument("--n_sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--radii", type=str, default="1,2")
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--dedup_from_depth", type=int, default=3)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

    print("=" * 90)
    print("Reproducing the identical test-molecule split used throughout this investigation")
    print("=" * 90)
    examples = build_examples(args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, radii)
    mol_ids = np.array([e[0] for e in examples])
    labels = np.array([e[2] for e in examples])
    dummy_x = csr_matrix((len(mol_ids), 1))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    _train_idx, test_idx = next(splitter.split(dummy_x, labels, groups=mol_ids))
    test_mol_ids = set(mol_ids[test_idx].tolist())
    print(f"Test molecules: {len(test_mol_ids)}")

    print("\nLoading real spectra ...")
    spec_df = pd.read_pickle(args.spec_df)
    spec_filt = spec_df[
        spec_df["prec_type"].isin(PREC_TYPES)
        & spec_df["inst_type"].isin(INST_TYPES)
        & spec_df["frag_mode"].isin(FRAG_MODES)
        & (spec_df["spec_type"] == "MS2")
        & spec_df["mol_id"].isin(test_mol_ids)
    ][["spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", "peaks"]].copy()
    del spec_df
    print(f"  {len(spec_filt):,} real spectra across {spec_filt['mol_id'].nunique():,} test molecules")
    spec_by_mol: dict[int, list] = {}
    for mol_id, grp in spec_filt.groupby("mol_id"):
        spec_by_mol[int(mol_id)] = grp[
            ["spec_id", "prec_type", "inst_type", "frag_mode", "peaks"]
        ].to_dict("records")

    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    test_members = [
        m for m in sample if int(os.path.basename(m.name).split(".")[0]) in test_mol_ids
    ]
    print(f"  {len(test_members)} test-molecule DAG files to re-scan")

    baseline_rows: list[tuple[float, float]] = []
    dedup_rows: list[tuple[float, float]] = []
    total_nodes_all = 0
    kept_nodes_all = 0
    n_mols_used = 0
    n_specs_used = 0

    with tarfile.open(args.tar, "r") as tf:
        for i, member in enumerate(test_members):
            if i % 20 == 0:
                print(f"  {i}/{len(test_members)} ...", flush=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))
            if dag["dag"].x.shape[0] > args.max_nodes:
                continue
            mol_id = int(os.path.basename(member.name).split(".")[0])
            spec_rows = spec_by_mol.get(mol_id, [])
            if not spec_rows:
                continue

            f_depth = formula_depth_map(dag)
            idx_by_h_delta = dag["idx_by_h_delta"]
            fpm = dag["formula_peak_mzs"].numpy()
            fpp = dag["formula_peak_probs"].numpy()

            all_formulas = set(f_depth.keys())
            allowed_baseline = ibd_allowed(idx_by_h_delta, all_formulas)

            kept, total_nodes = dedup_keep_set(dag, f_depth, args.dedup_from_depth)
            # Formulas still allowed after dedup: any formula whose min-depth-achieving
            # node (or any node) survives. Since dedup only drops non-first-achievers,
            # every formula's first achiever is always kept by construction -- so the
            # allowed set should be identical to baseline. Verify this explicitly rather
            # than assuming it.
            g = dag["dag"]
            x = g.x
            node_feat_idxs = g.node_feat_idxs[0]
            h_start, h_end = node_feat_idxs[3].item(), node_feat_idxs[4].item()
            h_formulae_idx = x[:, h_start:h_end]
            surviving_formulas = set()
            for ni in kept:
                for h_idx_val in h_formulae_idx[ni]:
                    fidx = h_idx_val.item()
                    if fidx > 0:
                        surviving_formulas.add(fidx)
            allowed_dedup = ibd_allowed(idx_by_h_delta, surviving_formulas & all_formulas)

            for sr in spec_rows:
                r_b, w_b = vectorized_recall(
                    sr["peaks"], fpm, fpp, allowed_baseline, sr["prec_type"], sr["inst_type"]
                )
                r_d, w_d = vectorized_recall(
                    sr["peaks"], fpm, fpp, allowed_dedup, sr["prec_type"], sr["inst_type"]
                )
                baseline_rows.append((r_b, w_b))
                dedup_rows.append((r_d, w_d))

            total_nodes_all += total_nodes
            kept_nodes_all += len(kept)
            n_mols_used += 1
            n_specs_used += len(spec_rows)

    print(f"\nUsed {n_mols_used} test molecules, {n_specs_used} real spectra")
    print(f"Dedup applied at depth >= {args.dedup_from_depth}")

    baseline_w = np.mean([w for _r, w in baseline_rows])
    dedup_w = np.mean([w for _r, w in dedup_rows])
    node_reduction = 100 * (1 - kept_nodes_all / total_nodes_all)

    print(f"\n{'='*70}")
    print(f"Baseline wrecall:        {baseline_w:.4f}")
    print(f"Dedup wrecall:           {dedup_w:.4f}")
    print(f"Wrecall delta:           {dedup_w - baseline_w:+.4f}")
    print(f"Total node reduction:    {node_reduction:.2f}%  ({kept_nodes_all:,}/{total_nodes_all:,} kept)")


if __name__ == "__main__":
    main()
