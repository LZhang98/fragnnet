"""Test whether a TRUE corpus-level (aggregate) node budget changes the picture from
docs/plans/fragnnet/dag_full_recall_tractability_plan.md's per-molecule uniform-retention-fraction
experiments.

Every prior experiment in this investigation (debug_fanout_cap_recall_eval.py,
debug_global_cutoff_recall_eval.py, debug_node_level_scorer_prototype.py) applied the SAME
retention fraction to EVERY molecule independently -- e.g. "keep 66% of nodes" meant every
molecule, regardless of size, lost 34% of its own nodes. That's a hard per-molecule cap.

But src/fragnnet/model/fragnnet_model.py batches fragmentation graphs with CSR-style
pointer arrays (mol_num_nodes/frag_num_nodes), like PyTorch Geometric -- there is no fixed
per-molecule node budget in the actual model. The real constraint is TOTAL nodes across
training (for signal-to-noise and to afford going deeper into the DAG), which is an
aggregate/corpus-level budget, not a per-molecule one.

Since ~99% of molecules are naturally small (500-4,000 nodes) and only ~1% blow up
(bond_breakability_mask_plan.md Step 1: 33/3,000 sampled molecules exceeded 50,000 nodes at
D4), a corpus-level budget should let untouched small molecules keep ALL their nodes while
concentrating any cuts on the few large ones -- a fundamentally different allocation than
uniformly taxing every molecule by the same fraction.

This script ranks ALL depth-4 nodes from ALL held-out test molecules TOGETHER on one
combined scale, then keeps the global top-N by TOTAL count (not per-molecule fraction),
using the same priority signals (hand-mined, learned, single-route aggregation -- the best
mechanism found so far) plus a true random baseline for comparison at matched TOTAL node
count.

Usage:
    python scripts/debug/debug_corpus_level_cutoff_eval.py \
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
from debug_fanout_cap_recall_eval import extract_fanout_data, make_hand_mined_priority, make_learned_priority  # noqa: E402
from debug_global_cutoff_recall_eval import per_node_single_route_score  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

RETENTION_FRACTIONS = [0.85, 0.66, 0.44, 0.25, 0.10]


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
    train_mol_ids = set(mol_ids[train_idx].tolist())
    test_mol_ids = set(mol_ids[test_idx].tolist())
    print(f"Train molecules: {len(train_mol_ids)}, Test molecules: {len(test_mol_ids)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], labels[train_idx])

    print("\n" + "=" * 90)
    print(f"Building CORPUS-LEVEL node pool from {len(test_mol_ids)} held-out molecules")
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
    print(f"  {len(test_members)} test-molecule DAG files to re-scan")

    single_route_rng_hm = random.Random(5000)
    single_route_rng_learned = random.Random(6000)
    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]
    learned_priority = make_learned_priority(clf, hasher, radii, mol_cache_by_mol_id, mol_id_ref)
    hand_mined_priority = make_hand_mined_priority()

    # Global pools: key = (mol_id, node_idx) -> score. Also a mapping from a globally
    # unique formula key -> list of global node keys, for corpus-wide recall.
    hm_scores: dict[tuple[int, int], float] = {}
    learned_scores: dict[tuple[int, int], float] = {}
    naive_scores: dict[tuple[int, int], float] = {}
    formula_nodes_global: dict[tuple[int, int], list[tuple[int, int]]] = {}
    per_mol_node_count: dict[int, int] = {}
    naive_rng = random.Random(7000)

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

            hm_node_score = per_node_single_route_score(
                children_of_parent, hand_mined_priority, single_route_rng_hm
            )
            learned_node_score = per_node_single_route_score(
                children_of_parent, learned_priority, single_route_rng_learned
            )
            per_mol_node_count[mol_id] = len(hm_node_score)
            for ni, score in hm_node_score.items():
                hm_scores[(mol_id, ni)] = score
            for ni, score in learned_node_score.items():
                learned_scores[(mol_id, ni)] = score
            for ni in hm_node_score:
                naive_scores[(mol_id, ni)] = naive_rng.random()

            for fidx, nodes in formula_target_nodes.items():
                formula_nodes_global[(mol_id, fidx)] = [(mol_id, ni) for ni in nodes]
            n_used += 1

    print(f"\nUsed {n_used} test molecules with depth-{args.target_depth} matched formulas")
    grand_total_nodes = len(hm_scores)
    grand_total_formulas = len(formula_nodes_global)
    per_mol_sizes = sorted(per_mol_node_count.values())
    print(
        f"Grand total depth-4 nodes across corpus: {grand_total_nodes:,} | "
        f"formulas: {grand_total_formulas:,}"
    )
    print(
        f"Per-molecule node count: min={per_mol_sizes[0]}, "
        f"median={per_mol_sizes[len(per_mol_sizes)//2]}, max={per_mol_sizes[-1]}, "
        f"mean={np.mean(per_mol_sizes):.1f}"
    )

    def evaluate(scores: dict[tuple[int, int], float], keep_n: int) -> tuple[float, int]:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        retained = {k for k, _ in ranked[:keep_n]}
        retained_formulas = sum(
            1 for nodes in formula_nodes_global.values() if any(k in retained for k in nodes)
        )
        mols_retained_counts: dict[int, int] = {}
        for mol_id, _ni in retained:
            mols_retained_counts[mol_id] = mols_retained_counts.get(mol_id, 0) + 1
        n_mols_fully_kept = sum(
            1
            for mol_id, total in per_mol_node_count.items()
            if mols_retained_counts.get(mol_id, 0) == total
        )
        n_mols_untouched_at_all = sum(1 for c in mols_retained_counts.values() if c > 0)
        return 100 * retained_formulas / grand_total_formulas, n_mols_fully_kept, n_mols_untouched_at_all

    print(f"\n{'='*90}")
    print(
        f"{'Policy':<16} {'Target frac':>12} {'Total nodes kept':>17} {'Recall retained':>16} "
        f"{'Mols fully kept':>15}"
    )
    print(f"{'-'*90}")
    total_mols = len(per_mol_node_count)
    for frac in RETENTION_FRACTIONS:
        keep_n = max(1, round(grand_total_nodes * frac))
        for name, scores in [
            ("hand_mined_single", hm_scores),
            ("learned_single", learned_scores),
            ("naive_random", naive_scores),
        ]:
            recall_pct, n_full, _n_touched = evaluate(scores, keep_n)
            print(
                f"{name:<16} {frac:>11.2f} {keep_n:>17,} {recall_pct:>15.2f}% "
                f"{n_full:>10,}/{total_mols}"
            )

    # Compare against the earlier PER-MOLECULE uniform-fraction numbers at the same
    # nominal fraction, to make the corpus-vs-per-molecule contrast explicit.
    print("\nFor reference, this script's grand-total node count equals the sum of every")
    print("test molecule's own depth-4 node count -- the same denominator the earlier")
    print("per-molecule experiments used, just pooled instead of applied uniformly.")


if __name__ == "__main__":
    main()
