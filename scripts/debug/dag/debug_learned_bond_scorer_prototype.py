"""Learned bond-priority scorer prototype (Step 2 of docs/plans/fragnnet/dag_full_recall_tractability_plan.md).

Trains a small linear (logistic regression over hashed features) scorer that predicts
whether a bond is "load-bearing" -- appears in at least one route to a formula first-matched
at the deepest mined depth -- as a candidate replacement for the hand-mined BUCKET_LOSS_RANK
table used by docs/plans/fragnnet/dag_fanout_cap_plan.md's per-parent fan-out cap.

Positive ("critical") examples: bonds that appear in EVERY route (per analyze_dag, reused
from debug_dag_bond_radius_env_stats.py) to some depth-max matched formula -- i.e. masking
that single bond alone would make the formula entirely unreachable. This mirrors
bond_breakability_mask_plan.md's per-bucket `bucket_loss` computation, applied per-bond
instead of per-bucket. A weaker "appears in at least one route" label was tried first and
was degenerate (95%+ positive rate at full D4 enumeration, since redundant multi-parent
routes mean nearly every bond gets touched by *some* route to *some* formula -- see
bond_breakability_mask_plan.md Step 4's "3.00 edges per node" finding for why).
Negative examples: every other real bond in the molecule (a valid BFS candidate cut that
is never singly-critical for any matched formula at the deepest depth).

Features are radius-1 and radius-2 bond environment buckets (same descriptor as the sibling
script), hashed via sklearn's FeatureHasher so the model needs no fixed vocabulary -- this is
the point of a learned scorer over a hand-mined table per docs/plans/fragnnet/dag_bond_radius_env_plan.md's
finding that cardinality (~2,000 buckets at radius 2) is too large to hand-curate even though
it isn't data-starved.

Usage:
    python scripts/debug/dag/debug_learned_bond_scorer_prototype.py \
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_dag_bond_radius_env_stats import analyze_dag, bond_env_bucket, bond_env_str  # noqa: E402


def build_examples(
    tar_fp: str, mol_df_fp: str, n_sample: int, seed: int, max_nodes: int, radii: list[int]
) -> list[tuple[int, dict[str, float], int]]:
    """Build (mol_id, hashed-feature-dict, label) triples for every real bond in each
    sampled molecule, labeled by whether the bond appears in any route to a depth-max
    matched formula.
    """
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

            result = analyze_dag(dag, mol)
            if result is None:
                continue
            global_max_d = max(result.keys())

            positive_bonds: set[tuple[int, int]] = set()
            for routes in result[global_max_d]:
                if not routes:
                    continue
                # A bond is "critical" for this formula only if every route (every tied
                # min-depth parent edge) requires it -- masking it alone would block the
                # formula entirely. Weaker "appears in some route" is degenerate (see
                # module docstring).
                common = set(routes[0])
                for pairs in routes[1:]:
                    common &= set(pairs)
                for a, b in common:
                    positive_bonds.add((min(a, b), max(a, b)))

            mol_cache: dict = {}
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                pair = (min(a, b), max(a, b))
                label = 1 if pair in positive_bonds else 0
                feats: dict[str, float] = {}
                for r in radii:
                    bucket = bond_env_bucket(mol, a, b, r, mol_cache)
                    feats[f"r{r}:{bond_env_str(bucket)}"] = 1.0
                examples.append((mol_id, feats, label))
                n_pos += label
                n_neg += 1 - label

            n_analyzed += 1

    print(
        f"  Skipped {n_skipped_large:,} molecules (>{max_nodes:,} nodes), "
        f"{n_skipped_no_mol:,} molecules (no mol_id match)"
    )
    print(f"Analyzed {n_analyzed} molecules -> {n_pos:,} positive bonds, {n_neg:,} negative bonds")
    return examples


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
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

    examples = build_examples(
        args.tar, args.mol_df, args.n_sample, args.seed, args.max_nodes, radii
    )
    if not examples:
        print("No examples built, aborting.")
        return

    mol_ids = np.array([e[0] for e in examples])
    feat_dicts = [e[1] for e in examples]
    labels = np.array([e[2] for e in examples])

    hasher = FeatureHasher(n_features=args.n_features, input_type="dict")
    x = hasher.transform(feat_dicts)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups=mol_ids))
    print(
        f"\nTrain: {len(train_idx):,} bonds ({len(set(mol_ids[train_idx])):,} molecules), "
        f"Test: {len(test_idx):,} bonds ({len(set(mol_ids[test_idx])):,} molecules)"
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx], labels[train_idx])

    scores = clf.decision_function(x[test_idx])
    y_test = labels[test_idx]
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    print(f"\nHeld-out AUC: {auc:.4f}")
    print(f"Held-out Average Precision: {ap:.4f} (base rate: {y_test.mean():.4f})")

    # Sanity check: precision among the top-scored test bonds vs. overall base rate.
    order = np.argsort(-scores)
    for k_frac in [0.05, 0.10, 0.25]:
        k = max(1, int(len(order) * k_frac))
        top_k_labels = y_test[order[:k]]
        print(
            f"  Top {int(k_frac*100)}% by score: precision={top_k_labels.mean():.4f} "
            f"(n={k:,}, base_rate={y_test.mean():.4f})"
        )

    # Human-readable check: inspect the highest-scored bonds' raw environment strings to
    # sanity-check the model recovers known chemistry (e.g. ring C-heteroatom bonds should
    # score high, per docs/plans/fragnnet/bond_breakability_mask_plan.md's Step 3 findings). Feature hashing
    # collapses names to indices, so coefficients aren't directly inspectable -- read the
    # examples themselves instead.
    print(f"\nTop {args.top_k} highest-scored test bonds (env string, true label):")
    test_feat_dicts = [feat_dicts[i] for i in test_idx]
    for rank in order[: args.top_k]:
        env_keys = list(test_feat_dicts[rank].keys())
        print(f"  score={scores[rank]:+.3f} label={y_test[rank]}  {env_keys}")


if __name__ == "__main__":
    main()
