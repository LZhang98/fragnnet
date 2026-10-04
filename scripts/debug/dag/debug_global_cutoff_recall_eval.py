"""Step 4 of docs/plans/fragnnet/dag_full_recall_tractability_plan.md: test whether the MECHANISM change
alone -- global (not per-parent) cutoff with max-over-redundant-routes, modeled on
src/fragnnet/iceberg/pl_model.py's autoregressive generation loop -- recovers the recall
that Step 3 (debug_fanout_cap_recall_eval.py) found lost when hand-mined/learned priority
was applied via an independent per-parent top-K cap (both lost to random tie-break there).

Reuses the exact same two priority signals as Step 3 (hand-mined BUCKET_LOSS_RANK, learned
logistic-regression scorer) -- only the SELECTION MECHANISM changes:
  Step 3 (superseded): per-parent independent top-K -- each parent keeps its own best K
    candidates; a node survives if ANY parent's independent decision keeps it.
  Step 4 (this script): per-node score = max(priority over ALL incoming edges) --
    mirroring ICEBERG's prob_gen = max(existing, new) rule for redundant routes -- then a
    SINGLE global ranking over all depth-4 nodes in the molecule, keeping the global top
    fraction by node count (matching ICEBERG's global max_nodes cutoff, not a per-parent
    budget).

This isolates "is it the per-parent independence that hurt" from "do we need a smarter
score" before training any new model, per the plan's step 4.

Usage:
    python scripts/debug/dag/debug_global_cutoff_recall_eval.py \
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

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_fanout_cap_recall_eval import (  # noqa: E402
    extract_fanout_data,
    make_hand_mined_priority,
    make_learned_priority,
)
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

# Target global retention fractions chosen to roughly match the average node-retention
# levels observed at K=8,6,4,3,2 in Step 3's per-parent cap (85.5%, 78-79%, 65-67%, 55-57%,
# 43-44%), so results are comparable at similar node budgets.
RETENTION_FRACTIONS = [0.85, 0.79, 0.66, 0.56, 0.44]


def per_node_max_score(children_of_parent, priority_fn) -> dict[int, float]:
    """Score each depth-4 node by the max priority over ALL its incoming edges (from any
    parent) -- mirrors ICEBERG's prob_gen = max(existing, new) rule for redundant routes.
    """
    node_score: dict[int, float] = {}
    for _parent, candidates in children_of_parent.items():
        for ni, buckets, pairs in candidates:
            score = priority_fn(buckets, pairs)
            if ni not in node_score or score > node_score[ni]:
                node_score[ni] = score
    return node_score


def per_node_single_route_score(
    children_of_parent, priority_fn, rng: random.Random
) -> dict[int, float]:
    """Score each depth-4 node by ONE uniformly-random incoming edge's priority (not the
    max over all of them). This removes the max-of-N-samples order-statistic inflation
    that high-in-degree nodes get under per_node_max_score, while keeping everything else
    (global ranking, same underlying priority_fn) identical -- isolates whether that
    inflation, rather than score comparability across the molecule, drove Step 4's result.
    """
    edges_by_node: dict[int, list[tuple[set, set]]] = {}
    for _parent, candidates in children_of_parent.items():
        for ni, buckets, pairs in candidates:
            edges_by_node.setdefault(ni, []).append((buckets, pairs))
    return {
        ni: priority_fn(*rng.choice(edges)) for ni, edges in edges_by_node.items()
    }


def per_node_average_route_score(children_of_parent, priority_fn) -> dict[int, float]:
    """Score each depth-4 node by the MEAN priority over all its incoming edges, instead of
    max (order-statistic-biased toward high in-degree) or a single random route (throws
    away signal from the other routes entirely). Uses all the redundancy information
    without the max's degree bias -- the natural next ablation after per_node_max_score
    and per_node_single_route_score.
    """
    node_scores: dict[int, list[float]] = {}
    for _parent, candidates in children_of_parent.items():
        for ni, buckets, pairs in candidates:
            node_scores.setdefault(ni, []).append(priority_fn(buckets, pairs))
    return {ni: sum(scores) / len(scores) for ni, scores in node_scores.items()}


def simulate_global_cutoff(
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
    parser.add_argument("--radii", type=str, default="1,2")
    parser.add_argument("--n_features", type=int, default=2**16)
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--target_depth", type=int, default=4)
    parser.add_argument("--naive_seeds", type=int, default=3)
    parser.add_argument(
        "--single_route_seeds",
        type=int,
        default=1,
        help="Number of independent random draws for single-route scoring, to measure "
        "variance from which route happens to be sampled per node.",
    )
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

    print("=" * 90)
    print("Training learned scorer (identical setup to Step 2/3 scripts)")
    print("=" * 90)
    examples = build_examples(args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, radii)
    mol_ids = np.array([e[0] for e in examples])
    feat_dicts = [e[1] for e in examples]
    labels = np.array([e[2] for e in examples])

    hasher = FeatureHasher(n_features=args.n_features, input_type="dict")
    x = hasher.transform(feat_dicts)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups=mol_ids))
    train_mol_ids = set(mol_ids[train_idx].tolist())
    test_mol_ids = set(mol_ids[test_idx].tolist())
    print(f"Train molecules: {len(train_mol_ids)}, Test molecules: {len(test_mol_ids)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], labels[train_idx])

    print("\n" + "=" * 90)
    print(f"Simulating GLOBAL cutoff on {len(test_mol_ids)} held-out test molecules only")
    print("=" * 90)

    print(f"Loading {args.mol_df} ...")
    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    with tarfile.open(args.tar, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name]
    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    test_members = [
        m for m in sample if int(os.path.basename(m.name).split(".")[0]) in test_mol_ids
    ]
    print(f"  {len(test_members)} test-molecule DAG files to re-scan for simulation")

    naive_rngs = [random.Random(2000 + i) for i in range(args.naive_seeds)]
    single_route_rngs = [random.Random(3000 + i) for i in range(args.single_route_seeds)]
    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]
    learned_priority = make_learned_priority(clf, hasher, radii, mol_cache_by_mol_id, mol_id_ref)
    hand_mined_priority = make_hand_mined_priority()

    single_policies = [
        f"{name}_single_{i}"
        for name in ("hand_mined", "learned")
        for i in range(args.single_route_seeds)
    ]
    policies = (
        ["hand_mined", "learned"]
        + single_policies
        + ["hand_mined_avg", "learned_avg"]
        + [f"naive_{i}" for i in range(args.naive_seeds)]
    )
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

            result = extract_fanout_data(dag, mol, args.target_depth)
            if result is None:
                continue
            children_of_parent, formula_target_nodes = result
            mol_id_ref[0], mol_id_ref[1] = mol_id, mol

            for policy_name in policies:
                if policy_name == "hand_mined":
                    node_score = per_node_max_score(children_of_parent, hand_mined_priority)
                elif policy_name == "learned":
                    node_score = per_node_max_score(children_of_parent, learned_priority)
                elif policy_name.startswith("hand_mined_single_"):
                    rng = single_route_rngs[int(policy_name.rsplit("_", 1)[1])]
                    node_score = per_node_single_route_score(
                        children_of_parent, hand_mined_priority, rng
                    )
                elif policy_name.startswith("learned_single_"):
                    rng = single_route_rngs[int(policy_name.rsplit("_", 1)[1])]
                    node_score = per_node_single_route_score(
                        children_of_parent, learned_priority, rng
                    )
                elif policy_name == "hand_mined_avg":
                    node_score = per_node_average_route_score(children_of_parent, hand_mined_priority)
                elif policy_name == "learned_avg":
                    node_score = per_node_average_route_score(children_of_parent, learned_priority)
                else:
                    rng = naive_rngs[int(policy_name.split("_")[1])]
                    node_score = per_node_max_score(
                        children_of_parent, lambda _b, _p, rng=rng: rng.random()
                    )
                for frac in RETENTION_FRACTIONS:
                    rf, tf_, rn, tn = simulate_global_cutoff(node_score, formula_target_nodes, frac)
                    slot = agg[policy_name][frac]
                    slot[0] += rf
                    slot[1] += tf_
                    slot[2] += rn
                    slot[3] += tn
            n_used += 1

    print(f"\nUsed {n_used} test molecules with depth-{args.target_depth} matched formulas")

    print(f"\n{'='*90}")
    print(f"{'Policy':<14} {'Target frac':>12} {'Recall retained':>16} {'Recall loss':>13} {'Actual node frac':>17}")
    print(f"{'-'*90}")
    for frac in RETENTION_FRACTIONS:
        for policy_name in policies:
            rf, tf_, rn, tn = agg[policy_name][frac]
            if tf_ == 0 or tn == 0:
                continue
            recall_pct = 100 * rf / tf_
            actual_node_frac = 100 * rn / tn
            print(
                f"{policy_name:<14} {frac:>11.2f} {recall_pct:>15.2f}% {100-recall_pct:>12.2f}% "
                f"{actual_node_frac:>16.2f}%"
            )

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

    if args.single_route_seeds > 1:
        print("\nSingle-route variance across independent random draws (same priority signal):")
        for name in ("hand_mined", "learned"):
            print(f"  {name}:")
            for frac in RETENTION_FRACTIONS:
                recalls = []
                node_fracs = []
                for i in range(args.single_route_seeds):
                    rf, tf_, rn, tn = agg[f"{name}_single_{i}"][frac]
                    if tf_ == 0:
                        continue
                    recalls.append(100 * rf / tf_)
                    node_fracs.append(100 * rn / tn if tn else 0)
                if recalls:
                    print(
                        f"    target_frac={frac}: recall={np.mean(recalls):.2f}% "
                        f"(+/-{np.std(recalls):.2f}, min={min(recalls):.2f}, max={max(recalls):.2f}) "
                        f"node_frac={np.mean(node_fracs):.2f}%"
                    )


if __name__ == "__main__":
    main()
