"""Follow-up to debug_true_wrecall_eval.py: that script found depth-4-specific trimming
barely moves real wrecall (0.5-2.3 points even at aggressive thresholds), on the same 90
held-out test molecules used throughout this investigation. This computes the natural
explanation -- how much real, intensity-weighted recall comes from each depth layer
(0,1,2,3,4), cumulatively, with NO trimming at all -- to see whether depths 0-3 already
capture most of it (in which case depth-4-only trimming was always going to matter little,
and the real remaining question is whether depths 1-3 could be trimmed instead, at bigger
compounding cost since pruning an early branch removes its whole downstream subtree).

Reuses the exact same matching machinery as debug_true_wrecall_eval.py
(formula_depth_map / filtered_allowed / ibd_allowed / vectorized_recall from
scripts/stats_scripts/dag_depth_recall_nist20_d3.py), on the identical 97-molecule
test split (same --seed, --n_sample, --test_frac), so results are directly comparable to
every prior number in this investigation.

Usage:
    python scripts/debug/debug_depth_attribution_wrecall.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --spec_df data/proc/nist20/spec_df.pkl \
        --n_sample 400 \
        --seed 42
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
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats_scripts"))
from dag_depth_recall_nist20_d3 import (  # noqa: E402
    filtered_allowed,
    formula_depth_map,
    ibd_allowed,
    vectorized_recall,
)
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES = {"FT", "QTOF", "IT"}
FRAG_MODES = {"CID", "HCD"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--spec_df", required=True)
    parser.add_argument("--n_sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--radii", type=str, default="1,2")
    parser.add_argument("--n_features", type=int, default=2**16)
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--max_depth", type=int, default=4)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

    # Reuse build_examples purely to reproduce the identical train/test molecule split
    # used throughout this investigation (same seed/n_sample/radii/test_frac) -- no new
    # model is trained here, this script needs no scorer at all.
    print("=" * 90)
    print("Reproducing the identical test-molecule split used throughout this investigation")
    print("=" * 90)
    examples = build_examples(args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, radii)
    mol_ids = np.array([e[0] for e in examples])
    labels = np.array([e[2] for e in examples])
    from scipy.sparse import csr_matrix

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

    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    test_members = [
        m for m in sample if int(os.path.basename(m.name).split(".")[0]) in test_mol_ids
    ]
    print(f"  {len(test_members)} test-molecule DAG files to re-scan")

    depths = list(range(args.max_depth + 1))
    rows: dict[int, list[tuple[float, float]]] = {d: [] for d in depths}
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
            mol = mol_by_id.get(mol_id)
            if mol is None:
                continue
            spec_rows = spec_by_mol.get(mol_id, [])
            if not spec_rows:
                continue

            f_depth = formula_depth_map(dag)
            idx_by_h_delta = dag["idx_by_h_delta"]
            fpm = dag["formula_peak_mzs"].numpy()
            fpp = dag["formula_peak_probs"].numpy()

            depth_allowed = {
                d: ibd_allowed(idx_by_h_delta, filtered_allowed(f_depth, d)) for d in depths
            }

            for sr in spec_rows:
                for d in depths:
                    r, w = vectorized_recall(
                        sr["peaks"], fpm, fpp, depth_allowed[d], sr["prec_type"], sr["inst_type"]
                    )
                    rows[d].append((r, w))

            n_mols_used += 1
            n_specs_used += len(spec_rows)

    print(f"\nUsed {n_mols_used} test molecules, {n_specs_used} real spectra")

    print(f"\n{'='*70}")
    print(f"{'depth<=':>8} {'recall':>10} {'wrecall':>10} {'d(recall)':>11} {'d(wrecall)':>12}")
    print(f"{'-'*70}")
    prev_r = prev_w = 0.0
    for d in depths:
        recalls = [r for r, _w in rows[d]]
        wrecalls = [w for _r, w in rows[d]]
        r_mean = np.mean(recalls)
        w_mean = np.mean(wrecalls)
        print(
            f"{d:>8} {r_mean:>10.4f} {w_mean:>10.4f} {r_mean-prev_r:>+11.4f} {w_mean-prev_w:>+12.4f}"
        )
        prev_r, prev_w = r_mean, w_mean


if __name__ == "__main__":
    main()
