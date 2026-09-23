"""Step 5 (revised) of docs/plans/fragnnet/dag_full_recall_tractability_plan.md: score depth-4 NODES
directly from features intrinsic to the fragment itself (element composition, size,
aromaticity, ring membership, boundary-bond count), instead of deriving a node score from
its incoming BOND edges and having to combine multiple redundant routes' scores (max /
average / single-route -- all tried in debug_global_cutoff_recall_eval.py, all left a
residual gap to naive random).

Motivation: every combination rule tried so far starts from a bond-level score and has to
aggregate over a node's redundant routes, and every aggregation rule has its own failure
mode. This sidesteps the aggregation problem entirely by not deriving the score from edges
at all -- there is exactly one score per node, so there is nothing to combine.

Label: is this depth-4 fragment one of the nodes matching a formula first-matched at
depth 4 (same recall target as all prior scripts in this investigation).

Usage:
    python scripts/debug/debug_node_level_scorer_prototype.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
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
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_dag_bond_descriptor_stats import decode_cc  # noqa: E402

RETENTION_FRACTIONS = [0.85, 0.79, 0.66, 0.56, 0.44]


def compute_node_features(mol, atom_indices: frozenset) -> dict[str, float]:
    """Features intrinsic to the fragment itself -- no bond/edge/route information."""
    counts: Counter = Counter()
    n_aromatic = 0
    n_ring_atoms = 0
    n_boundary_bonds = 0
    for a in atom_indices:
        atom = mol.GetAtomWithIdx(a)
        counts[atom.GetSymbol()] += 1
        if atom.GetIsAromatic():
            n_aromatic += 1
        if atom.IsInRing():
            n_ring_atoms += 1
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in atom_indices:
                n_boundary_bonds += 1
    feats = {f"elem_{sym}": float(cnt) for sym, cnt in counts.items()}
    feats["frag_size"] = float(len(atom_indices))
    feats["n_aromatic"] = float(n_aromatic)
    feats["n_ring_atoms"] = float(n_ring_atoms)
    feats["n_boundary_bonds"] = float(n_boundary_bonds)
    return feats


def extract_node_level_data(dag: dict, mol, target_depth: int):
    """Return (node_examples, formula_target_nodes) for one molecule.
    node_examples: list of (node_idx, atom_indices, label) for every node at target_depth.
    formula_target_nodes: fidx -> list of node indices at target_depth matching fidx,
    for the recall computation (same convention as debug_fanout_cap_recall_eval.py).
    """
    g = dag["dag"]
    x = g.x
    node_feat_idxs = g.node_feat_idxs[0]
    n_nodes = x.shape[0]
    num_atoms = mol.GetNumAtoms()
    cc_start, cc_end = node_feat_idxs[1].item(), node_feat_idxs[2].item()
    h_start, h_end = node_feat_idxs[3].item(), node_feat_idxs[4].item()
    node_cc = [decode_cc(x[ni, cc_start:cc_end], num_atoms) for ni in range(n_nodes)]
    h_formulae_idx = x[:, h_start:h_end]

    nmd = dag["nodes_min_depth"]
    fpp = dag["formula_peak_probs"]

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    formula_to_min_depth: dict[int, int] = {}
    formula_to_nodes: dict[int, list[int]] = defaultdict(list)
    for ni in range(n_nodes):
        d = node_depth(ni)
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0 or fidx >= fpp.shape[0]:
                continue
            formula_to_nodes[fidx].append(ni)
            if fpp[fidx].sum() == 0:
                continue
            if fidx not in formula_to_min_depth or d < formula_to_min_depth[fidx]:
                formula_to_min_depth[fidx] = d

    target_formulas = [fidx for fidx, d in formula_to_min_depth.items() if d == target_depth]
    if not target_formulas:
        return None

    formula_target_nodes = {
        fidx: [ni for ni in formula_to_nodes[fidx] if node_depth(ni) == target_depth]
        for fidx in target_formulas
    }
    matched_nodes = {ni for nodes in formula_target_nodes.values() for ni in nodes}

    nodes_at_target = [ni for ni in range(n_nodes) if node_depth(ni) == target_depth]
    if not nodes_at_target:
        return None

    node_examples = [
        (ni, node_cc[ni], 1 if ni in matched_nodes else 0) for ni in nodes_at_target
    ]
    return node_examples, formula_target_nodes


def build_examples(tar_fp, mol_df_fp, n_sample, seed, max_nodes, target_depth):
    print(f"Loading {mol_df_fp} ...")
    mol_df = pd.read_pickle(mol_df_fp)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    print(f"Scanning {tar_fp} ...")
    with tarfile.open(tar_fp, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    print(f"  Total DAG files: {len(members):,}")

    random.seed(seed)
    sample = random.sample(members, min(n_sample, len(members)))
    print(f"  Sampling {len(sample):,} molecules ...")

    examples: list[tuple[int, dict[str, float], int]] = []
    n_analyzed = 0
    n_skipped_large = 0
    n_skipped_no_mol = 0
    n_pos = 0
    n_neg = 0

    with tarfile.open(tar_fp, "r") as tf:
        for i, member in enumerate(sample):
            if i % 50 == 0:
                print(f"  {i}/{len(sample)} ...", flush=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))
            if dag["reached_depth"] < 1:
                continue
            if dag["dag"].x.shape[0] > max_nodes:
                n_skipped_large += 1
                continue

            mol_id = int(os.path.basename(member.name).split(".")[0])
            mol = mol_by_id.get(mol_id)
            if mol is None:
                n_skipped_no_mol += 1
                continue

            result = extract_node_level_data(dag, mol, target_depth)
            if result is None:
                continue
            node_examples, _ = result
            for _ni, atom_indices, label in node_examples:
                feats = compute_node_features(mol, atom_indices)
                examples.append((mol_id, feats, label))
                n_pos += label
                n_neg += 1 - label
            n_analyzed += 1

    print(
        f"  Skipped {n_skipped_large:,} molecules (>{max_nodes:,} nodes), "
        f"{n_skipped_no_mol:,} molecules (no mol_id match)"
    )
    print(f"Analyzed {n_analyzed} molecules -> {n_pos:,} positive nodes, {n_neg:,} negative nodes")
    return examples


def simulate_global_cutoff_direct(
    node_score: dict[int, float], formula_target_nodes, retention_fraction: float
) -> tuple[int, int, int, int]:
    total_nodes = len(node_score)
    keep_n = max(1, round(total_nodes * retention_fraction))
    ranked = sorted(node_score.items(), key=lambda kv: -kv[1])
    retained_nodes = {ni for ni, _ in ranked[:keep_n]}
    total_formulas = len(formula_target_nodes)
    retained_formulas = sum(
        1 for nodes in formula_target_nodes.values() if any(ni in retained_nodes for ni in nodes)
    )
    return retained_formulas, total_formulas, len(retained_nodes), total_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--n_sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--target_depth", type=int, default=4)
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--naive_seeds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 90)
    print("Building node-level examples and training scorer")
    print("=" * 90)
    examples = build_examples(
        args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, args.target_depth
    )
    mol_ids = np.array([e[0] for e in examples])
    feat_dicts = [e[1] for e in examples]
    labels = np.array([e[2] for e in examples])

    vec = DictVectorizer(sparse=True)
    x = vec.fit_transform(feat_dicts)
    print(f"Feature dimensionality: {len(vec.feature_names_)} ({vec.feature_names_})")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups=mol_ids))
    train_mol_ids = set(mol_ids[train_idx].tolist())
    test_mol_ids = set(mol_ids[test_idx].tolist())
    print(f"Train molecules: {len(train_mol_ids)}, Test molecules: {len(test_mol_ids)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], labels[train_idx])

    scores = clf.decision_function(x[test_idx])
    y_test = labels[test_idx]
    print(f"\nHeld-out AUC: {roc_auc_score(y_test, scores):.4f}")
    print(
        f"Held-out Average Precision: {average_precision_score(y_test, scores):.4f} "
        f"(base rate: {y_test.mean():.4f})"
    )
    for name, coef in sorted(
        zip(vec.feature_names_, clf.coef_[0].tolist(), strict=True), key=lambda kv: -kv[1]
    ):
        print(f"  {name:<16} {coef:+.3f}")

    print("\n" + "=" * 90)
    print(f"Simulating GLOBAL cutoff (direct node score, no aggregation) on {len(test_mol_ids)} held-out molecules")
    print("=" * 90)

    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    test_members = [
        m for m in sample if int(os.path.basename(m.name).split(".")[0]) in test_mol_ids
    ]

    naive_rngs = [random.Random(4000 + i) for i in range(args.naive_seeds)]
    policies = ["node_scorer"] + [f"naive_{i}" for i in range(args.naive_seeds)]
    agg = {policy: {f: [0, 0, 0, 0] for f in RETENTION_FRACTIONS} for policy in policies}
    n_used = 0

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

            result = extract_node_level_data(dag, mol, args.target_depth)
            if result is None:
                continue
            node_examples, formula_target_nodes = result

            node_feats = [compute_node_features(mol, atoms) for _ni, atoms, _label in node_examples]
            x_nodes = vec.transform(node_feats)
            model_scores = clf.decision_function(x_nodes)

            model_node_score = {
                ni: float(s) for (ni, _atoms, _label), s in zip(node_examples, model_scores, strict=True)
            }
            for policy_name in policies:
                if policy_name == "node_scorer":
                    node_score = model_node_score
                else:
                    rng = naive_rngs[int(policy_name.split("_")[1])]
                    node_score = {ni: rng.random() for ni, _atoms, _label in node_examples}
                for frac in RETENTION_FRACTIONS:
                    rf, tf_, rn, tn = simulate_global_cutoff_direct(
                        node_score, formula_target_nodes, frac
                    )
                    slot = agg[policy_name][frac]
                    slot[0] += rf
                    slot[1] += tf_
                    slot[2] += rn
                    slot[3] += tn
            n_used += 1

    print(f"\nUsed {n_used} test molecules with depth-{args.target_depth} matched formulas")

    print(f"\n{'='*90}")
    print(f"{'Policy':<14} {'Target frac':>12} {'Recall retained':>16} {'Recall loss':>13}")
    print(f"{'-'*90}")
    for frac in RETENTION_FRACTIONS:
        for policy_name in policies:
            rf, tf_, rn, tn = agg[policy_name][frac]
            if tf_ == 0:
                continue
            recall_pct = 100 * rf / tf_
            print(f"{policy_name:<14} {frac:>11.2f} {recall_pct:>15.2f}% {100-recall_pct:>12.2f}%")

    print("\nNaive baseline averaged across seeds:")
    for frac in RETENTION_FRACTIONS:
        recalls = []
        for i in range(args.naive_seeds):
            rf, tf_, rn, tn = agg[f"naive_{i}"][frac]
            if tf_ == 0:
                continue
            recalls.append(100 * rf / tf_)
        if recalls:
            print(f"  target_frac={frac}: recall={np.mean(recalls):.2f}% (+/-{np.std(recalls):.2f})")


if __name__ == "__main__":
    main()
