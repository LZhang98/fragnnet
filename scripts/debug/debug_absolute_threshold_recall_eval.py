"""Test branch/subgraph trimming via an ABSOLUTE per-candidate threshold, instead of a
budget (a hard per-molecule fraction, or a corpus-wide top-N quota -- both tried in
debug_global_cutoff_recall_eval.py / debug_corpus_level_cutoff_eval.py).

Every prior experiment in this investigation picked a TARGET retention fraction (or total
node count) and kept exactly that many candidates, ranked by score. That requires either
(a) knowing each molecule's own candidate count in advance (per-molecule fraction) or
(b) seeing every molecule's scores at once before deciding cutoffs (corpus-level quota) --
neither of which matches how preproc_scripts/03_prepare_dag_feats.py actually generates
DAGs, which is one molecule at a time.

A branch (a specific candidate cut from a specific parent, leading to one child fragment)
is trimmed independently of every other branch: compute its score, keep it if score >= a
FIXED threshold tau, drop it otherwise. No ranking against siblings, no ranking against the
rest of the corpus, no target fraction. The number of surviving branches per molecule
emerges naturally -- a clean, small molecule with uniformly high-scoring candidates keeps
everything; a messy, blown-up molecule with mostly low-scoring candidates gets trimmed hard
-- purely a byproduct of tau, not something specified per molecule.

Comparison baseline: applying the SAME threshold mechanism to a uniform random score is
mathematically equivalent (in expectation) to the ORIGINAL per-molecule uniform-fraction
experiments (P(Uniform[0,1] >= tau) = 1-tau identically for every candidate, regardless of
molecule) -- so this also connects the three framings (per-molecule fraction, corpus quota,
absolute threshold) into one coherent comparison.

Usage:
    python scripts/debug/debug_absolute_threshold_recall_eval.py \
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
from debug_dag_bond_radius_env_stats import bond_env_bucket, bond_env_str  # noqa: E402
from debug_fanout_cap_recall_eval import extract_fanout_data, make_hand_mined_priority, make_learned_priority  # noqa: E402
from debug_global_cutoff_recall_eval import per_node_single_route_score  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402


def evaluate_threshold(
    node_score: dict[int, float], formula_target_nodes, tau: float
) -> tuple[int, int, int, int]:
    """Keep every node independently: score >= tau. No ranking, no target count."""
    retained_nodes = {ni for ni, score in node_score.items() if score >= tau}
    total_formulas = len(formula_target_nodes)
    retained_formulas = sum(
        1 for nodes in formula_target_nodes.values() if any(ni in retained_nodes for ni in nodes)
    )
    return retained_formulas, total_formulas, len(retained_nodes), len(node_score)


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

    # Scores need to be on a comparable [0,1]-ish scale to sweep one tau across both
    # hand-mined and learned signals and the random baseline -- use predict_proba (a
    # monotonic transform of decision_function, so it doesn't change *ranking*, only
    # makes an absolute threshold swept in the same units meaningful for all three).
    print("\n" + "=" * 90)
    print(f"Evaluating per-branch absolute threshold on {len(test_mol_ids)} held-out molecules")
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

    single_route_rng_hm = random.Random(8000)
    single_route_rng_learned = random.Random(9000)
    naive_rng = random.Random(10000)
    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]
    hand_mined_priority = make_hand_mined_priority()

    def learned_priority_proba(buckets, pairs):
        # predict_proba instead of decision_function -- monotonic transform, same ranking,
        # but puts the score on a genuine [0,1] probability scale so a fixed tau is
        # meaningful and comparable to the hand-mined table's own [0, 4.17] scale once both
        # are min-max normalized below.
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

    # Hand-mined scores are already bounded [0, 4.17]; normalize to [0,1] by the known max
    # so the same tau grid is meaningful for both signals.
    HAND_MINED_MAX = 4.17

    def hand_mined_priority_normalized(buckets, pairs):
        return hand_mined_priority(buckets, pairs) / HAND_MINED_MAX

    all_hm_scores: list[tuple[int, dict[int, float]]] = []
    all_learned_scores: list[tuple[int, dict[int, float]]] = []
    all_naive_scores: list[tuple[int, dict[int, float]]] = []
    all_formula_target_nodes: list[dict] = []
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
                children_of_parent, hand_mined_priority_normalized, single_route_rng_hm
            )
            learned_node_score = per_node_single_route_score(
                children_of_parent, learned_priority_proba, single_route_rng_learned
            )
            naive_node_score = {ni: naive_rng.random() for ni in hm_node_score}

            all_hm_scores.append((mol_id, hm_node_score))
            all_learned_scores.append((mol_id, learned_node_score))
            all_naive_scores.append((mol_id, naive_node_score))
            all_formula_target_nodes.append(formula_target_nodes)
            n_used += 1

    print(f"\nUsed {n_used} test molecules with depth-{args.target_depth} matched formulas")

    tau_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    print(f"\n{'='*100}")
    print(f"{'Policy':<14} {'tau':>6} {'Total nodes kept':>17} {'Node reduction':>15} {'Recall retained':>16}")
    print(f"{'-'*100}")

    for name, all_scores in [
        ("hand_mined", all_hm_scores),
        ("learned", all_learned_scores),
        ("naive", all_naive_scores),
    ]:
        for tau in tau_grid:
            total_retained_formulas = 0
            total_formulas = 0
            total_retained_nodes = 0
            total_nodes = 0
            for (mol_id, node_score), formula_target_nodes in zip(
                all_scores, all_formula_target_nodes, strict=True
            ):
                rf, tf_, rn, tn = evaluate_threshold(node_score, formula_target_nodes, tau)
                total_retained_formulas += rf
                total_formulas += tf_
                total_retained_nodes += rn
                total_nodes += tn
            recall_pct = 100 * total_retained_formulas / total_formulas
            node_reduction_pct = 100 * (1 - total_retained_nodes / total_nodes)
            print(
                f"{name:<14} {tau:>6.2f} {total_retained_nodes:>17,} "
                f"{node_reduction_pct:>14.2f}% {recall_pct:>15.2f}%"
            )


if __name__ == "__main__":
    main()
