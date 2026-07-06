"""Correction to the rest of this investigation: every "recall" number computed by
debug_fanout_cap_recall_eval.py / debug_global_cutoff_recall_eval.py /
debug_corpus_level_cutoff_eval.py / debug_absolute_threshold_recall_eval.py measured
formula reachability WITHIN the DAG's own combinatorially-generated formula set -- NOT
recall against real observed spectral peaks. `dag["formula_peak_probs"]` is the
*theoretical isotope pattern* for a formula (computed for every formula the DAG generates,
270/271 nonzero in a spot check), not evidence the formula was ever actually observed.

This script computes the real thing: `wrecall` (intensity-weighted recall against actual
observed spectrum peaks), reusing the exact matching methodology already proven in
scripts/stats_scripts/dag_depth_recall_nist20_d3.py (ppm-tolerance m/z matching against
formula_peak_mzs, precursor-type mass offset, per-instrument tolerance) -- not a
reimplementation, the same functions.

For each held-out test molecule, allowed formulas = {formulas reachable at depth <= 3,
always kept, untouched by any depth-4 trimming policy} UNION {formulas whose depth-4 node
survives the trimming policy under test}. Then real recall/wrecall is computed against
that molecule's actual spectra from data/proc/nist20/spec_df.pkl.

Usage:
    python scripts/debug/debug_true_wrecall_eval.py \
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
from debug_dag_bond_radius_env_stats import bond_env_bucket, bond_env_str  # noqa: E402
from debug_fanout_cap_recall_eval import extract_fanout_data, make_hand_mined_priority  # noqa: E402
from debug_global_cutoff_recall_eval import per_node_single_route_score  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES = {"FT", "QTOF", "IT"}
FRAG_MODES = {"CID", "HCD"}
TAU_GRID_NORMALIZED = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
HAND_MINED_MAX = 4.17


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
    parser.add_argument("--target_depth", type=int, default=4)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

    print("=" * 90)
    print("Training learned scorer (identical setup to prior scripts)")
    print("=" * 90)
    examples = build_examples(args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, radii)
    mol_ids = np.array([e[0] for e in examples])
    feat_dicts = [e[1] for e in examples]
    labels = np.array([e[2] for e in examples])

    hasher = FeatureHasher(n_features=args.n_features, input_type="dict")
    x = hasher.transform(feat_dicts)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups=mol_ids))
    test_mol_ids = set(mol_ids[test_idx].tolist())
    print(f"Test molecules: {len(test_mol_ids)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], labels[train_idx])

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

    single_route_rng_hm = random.Random(11000)
    single_route_rng_learned = random.Random(12000)
    naive_rng = random.Random(13000)
    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]

    def learned_priority_proba(buckets, pairs):
        # predict_proba (bounded [0,1]), not decision_function (unbounded logit, can be
        # negative) -- a fixed tau grid starting at 0.0 only means "keep everything" if
        # the score is on a genuine probability scale.
        mol = mol_id_ref[1]
        cache = mol_cache_by_mol_id.setdefault(mol_id_ref[0], {})
        best = -1.0
        for a, b in pairs:
            feats = {}
            for r in radii:
                bucket = bond_env_bucket(mol, a, b, r, cache)
                feats[f"r{r}:{bond_env_str(bucket)}"] = 1.0
            proba = clf.predict_proba(hasher.transform([feats]))[0, 1]
            best = max(best, proba)
        return best

    learned_priority = learned_priority_proba
    hand_mined_priority = make_hand_mined_priority()

    # Per policy per tau: list of (recall, wrecall) per real spectrum.
    results: dict[str, dict[float, list[tuple[float, float]]]] = {
        "hand_mined": {t: [] for t in TAU_GRID_NORMALIZED},
        "learned": {t: [] for t in TAU_GRID_NORMALIZED},
        "naive": {t: [] for t in TAU_GRID_NORMALIZED},
    }
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

            result = extract_fanout_data(dag, mol, args.target_depth)
            if result is None:
                continue
            children_of_parent, formula_target_nodes = result
            mol_id_ref[0], mol_id_ref[1] = mol_id, mol

            # Real-recall machinery: formula depth map (full untrimmed DAG), h-delta
            # expansion, theoretical peak mzs/probs -- exactly as the proven reference.
            f_depth = formula_depth_map(dag)
            idx_by_h_delta = dag["idx_by_h_delta"]
            fpm = dag["formula_peak_mzs"].numpy()
            fpp = dag["formula_peak_probs"].numpy()

            allowed_shallow = ibd_allowed(idx_by_h_delta, filtered_allowed(f_depth, args.target_depth - 1))

            # Map depth-4 node -> its formula indices (reverse of formula_target_nodes),
            # needed to translate "this node survives" into "these formulas stay allowed."
            node_to_formulas: dict[int, list[int]] = {}
            for fidx, nodes in formula_target_nodes.items():
                for ni in nodes:
                    node_to_formulas.setdefault(ni, []).append(fidx)

            hm_node_score = per_node_single_route_score(
                children_of_parent, hand_mined_priority, single_route_rng_hm
            )
            learned_node_score = per_node_single_route_score(
                children_of_parent, learned_priority, single_route_rng_learned
            )
            naive_node_score = {ni: naive_rng.random() for ni in hm_node_score}

            for policy_name, raw_scores, max_val in [
                ("hand_mined", hm_node_score, HAND_MINED_MAX),
                ("learned", learned_node_score, 1.0),
                ("naive", naive_node_score, 1.0),
            ]:
                for tau in TAU_GRID_NORMALIZED:
                    survivors = {ni for ni, s in raw_scores.items() if (s / max_val) >= tau}
                    survivor_formulas = set()
                    for ni in survivors:
                        survivor_formulas.update(node_to_formulas.get(ni, []))
                    allowed = allowed_shallow | ibd_allowed(
                        idx_by_h_delta, survivor_formulas
                    )
                    for sr in spec_rows:
                        r, w = vectorized_recall(
                            sr["peaks"], fpm, fpp, allowed, sr["prec_type"], sr["inst_type"]
                        )
                        results[policy_name][tau].append((r, w))

            n_mols_used += 1
            n_specs_used += len(spec_rows)

    print(f"\nUsed {n_mols_used} test molecules, {n_specs_used} real spectra")

    print(f"\n{'='*90}")
    print(f"{'Policy':<14} {'tau':>6} {'recall':>10} {'wrecall':>10} {'n_specs':>10}")
    print(f"{'-'*90}")
    for policy_name in ["hand_mined", "learned", "naive"]:
        for tau in TAU_GRID_NORMALIZED:
            rows = results[policy_name][tau]
            if not rows:
                continue
            recalls = [r for r, _w in rows]
            wrecalls = [w for _r, w in rows]
            print(
                f"{policy_name:<14} {tau:>6.2f} {np.mean(recalls):>10.4f} "
                f"{np.mean(wrecalls):>10.4f} {len(rows):>10,}"
            )


if __name__ == "__main__":
    main()
