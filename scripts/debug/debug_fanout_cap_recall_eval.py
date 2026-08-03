"""Step 3 of docs/plans/dag_full_recall_tractability_plan.md: measure actual depth-4 formula
recall retained vs. node reduction under a per-parent fan-out cap, comparing three
candidate-ranking functions on the SAME held-out molecules:

1. "hand-mined" -- the (in_ring, element_pair, alpha_hetero) bucket-loss table already
   verified by debug_dag_bond_descriptor_stats.py (values hardcoded below from
   docs/plans/bond_breakability_mask_plan.md's Step 3 table; that script still exists and can
   reproduce them).
2. "learned" -- the logistic-regression scorer prototyped in
   debug_learned_bond_scorer_prototype.py, trained here on the same train/test molecule
   split (GroupShuffleSplit, seed=42) so evaluation is on genuinely held-out molecules.
3. "naive" -- random tie-break at matched K, averaged over a few seeds, per
   docs/plans/dag_fanout_cap_plan.md's still-open "step 2" sensitivity check.

IMPORTANT CONTEXT: docs/plans/dag_fanout_cap_plan.md cites a K=8 baseline (16.24% node
reduction / 6.30% recall loss) attributed to a script
`debug_dag_fanout_cap_node_reduction.py` that does not exist anywhere in this repo or its
git history. Those numbers cannot be reproduced or verified. This script is a fresh,
from-scratch implementation of the same fan-out-cap-simulation idea, not a rerun of that
missing script -- treat any resemblance to the old doc's numbers as coincidental, and treat
this script's own printed numbers as the only currently-verified ones.

Usage:
    python scripts/debug/debug_fanout_cap_recall_eval.py \
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
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_dag_bond_descriptor_stats import bond_bucket, decode_cc  # noqa: E402
from debug_dag_bond_radius_env_stats import bond_env_bucket, bond_env_str, boundary_bond_pairs  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

# Hardcoded from docs/plans/bond_breakability_mask_plan.md's Step 3 table (verified, reproducible
# via debug_dag_bond_descriptor_stats.py, which still exists in this repo).
BUCKET_LOSS_RANK = {
    (False, "XX", False): 0.03,
    (False, "CC", True): 0.03,
    (True, "XX", False): 0.03,
    (False, "CC", False): 0.04,
    (False, "CX", True): 0.28,
    (False, "CX", False): 0.50,
    (True, "CX", True): 0.93,
    (True, "CX", False): 1.73,
    (True, "CC", False): 1.83,
    (True, "CC", True): 4.17,
}


def extract_fanout_data(dag: dict, mol, target_depth: int):
    """Return (children_of_parent, formula_target_nodes) for one molecule, restricted to
    depth-`target_depth` children. children_of_parent[parent_ni] = list of
    (child_ni, bucket_set, bond_pairs). formula_target_nodes[fidx] = node indices at
    target_depth that match formula fidx and whose min matched depth IS target_depth
    (i.e. formulas first reachable at this depth, matching the existing scripts'
    "first-matched here" convention).
    """
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
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

    parent_of: dict[int, list[int]] = defaultdict(list)
    for ei in range(edge_index.shape[1]):
        parent_of[edge_index[1, ei].item()].append(edge_index[0, ei].item())

    nodes_at_target = [ni for ni in range(n_nodes) if node_depth(ni) == target_depth]
    children_of_parent: dict[int, list[tuple[int, set, set]]] = defaultdict(list)
    for ni in nodes_at_target:
        child_atoms = node_cc[ni]
        for pi in parent_of[ni]:
            complement = node_cc[pi] - child_atoms
            if not complement:
                continue
            pairs = boundary_bond_pairs(mol, child_atoms, complement)
            if not pairs:
                continue
            buckets = {bond_bucket(mol, a, b) for a, b in pairs}
            children_of_parent[pi].append((ni, buckets, pairs))

    formula_target_nodes = {
        fidx: [ni for ni in formula_to_nodes[fidx] if node_depth(ni) == target_depth]
        for fidx in target_formulas
    }
    return children_of_parent, formula_target_nodes


def rank_candidates(children_of_parent, priority_fn) -> dict[int, list[tuple[int, set, set]]]:
    """Score and sort each parent's candidates once, so a single ranking can be reused
    across every K value instead of recomputing priority_fn per (parent, K) pair.
    """
    return {
        parent: sorted(candidates, key=lambda c: -priority_fn(c[1], c[2]))
        for parent, candidates in children_of_parent.items()
    }


def simulate_cap(ranked_children_of_parent, formula_target_nodes, k: int) -> tuple[int, int, int, int]:
    """Apply a per-parent top-K cap to already-ranked candidates. Returns
    (retained_formulas, total_formulas, retained_nodes, total_nodes).
    """
    retained_nodes: set[int] = set()
    total_nodes: set[int] = set()
    for _parent, sorted_candidates in ranked_children_of_parent.items():
        for ni, _, _ in sorted_candidates:
            total_nodes.add(ni)
        keep = sorted_candidates[:k] if len(sorted_candidates) > k else sorted_candidates
        for ni, _, _ in keep:
            retained_nodes.add(ni)

    total_formulas = len(formula_target_nodes)
    retained_formulas = sum(
        1 for nodes in formula_target_nodes.values() if any(ni in retained_nodes for ni in nodes)
    )
    return retained_formulas, total_formulas, len(retained_nodes), len(total_nodes)


def make_hand_mined_priority():
    def priority(buckets: set, _pairs: set) -> float:
        return max((BUCKET_LOSS_RANK.get(b, 0.0) for b in buckets), default=0.0)

    return priority


def make_learned_priority(clf, hasher, radii, mol_cache_by_mol_id, mol_id_ref: list[int]):
    """mol_id_ref is a 1-element list acting as a mutable "current molecule id" cell, set
    by the caller before each simulate_cap call, so the same priority_fn closure can be
    reused across molecules without rebuilding it.
    """

    def priority(_buckets: set, pairs: set) -> float:
        mol = mol_id_ref[1]
        cache = mol_cache_by_mol_id.setdefault(mol_id_ref[0], {})
        best = -np.inf
        for a, b in pairs:
            feats = {}
            for r in radii:
                bucket = bond_env_bucket(mol, a, b, r, cache)
                feats[f"r{r}:{bond_env_str(bucket)}"] = 1.0
            score = clf.decision_function(hasher.transform([feats]))[0]
            best = max(best, score)
        return best

    return priority


def make_naive_priority(rng: random.Random):
    def priority(_buckets: set, _pairs: set) -> float:
        return rng.random()

    return priority


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
    parser.add_argument("--ks", type=str, default="2,3,4,6,8")
    parser.add_argument("--naive_seeds", type=int, default=3)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    print("=" * 90)
    print("Training learned scorer (identical setup to debug_learned_bond_scorer_prototype.py)")
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
    print(f"Simulating fan-out cap on {len(test_mol_ids)} held-out test molecules only")
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

    naive_rngs = [random.Random(1000 + i) for i in range(args.naive_seeds)]
    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]  # [mol_id, mol] cell shared with make_learned_priority closure
    learned_priority = make_learned_priority(clf, hasher, radii, mol_cache_by_mol_id, mol_id_ref)
    hand_mined_priority = make_hand_mined_priority()

    agg = {
        policy: {k: [0, 0, 0, 0] for k in ks}  # [retained_f, total_f, retained_n, total_n]
        for policy in ["hand_mined", "learned"] + [f"naive_{i}" for i in range(args.naive_seeds)]
    }
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

            policies = [
                ("hand_mined", hand_mined_priority),
                ("learned", learned_priority),
            ] + [(f"naive_{i}", make_naive_priority(rng)) for i, rng in enumerate(naive_rngs)]
            for policy_name, priority_fn in policies:
                ranked = rank_candidates(children_of_parent, priority_fn)
                for k in ks:
                    rf, tf_, rn, tn = simulate_cap(ranked, formula_target_nodes, k)
                    slot = agg[policy_name][k]
                    slot[0] += rf
                    slot[1] += tf_
                    slot[2] += rn
                    slot[3] += tn
            n_used += 1

    print(f"\nUsed {n_used} test molecules with depth-{args.target_depth} matched formulas")

    print(f"\n{'='*90}")
    print(f"{'Policy':<14} {'K':>4} {'Recall retained':>16} {'Recall loss':>13} {'Node reduction':>16}")
    print(f"{'-'*90}")
    for k in ks:
        for policy_name in ["hand_mined", "learned"] + [f"naive_{i}" for i in range(args.naive_seeds)]:
            rf, tf_, rn, tn = agg[policy_name][k]
            if tf_ == 0 or tn == 0:
                continue
            recall_pct = 100 * rf / tf_
            node_reduction_pct = 100 * (1 - rn / tn)
            print(
                f"{policy_name:<14} {k:>4} {recall_pct:>15.2f}% {100-recall_pct:>12.2f}% "
                f"{node_reduction_pct:>15.2f}%"
            )

    print("\nNaive baseline averaged across seeds:")
    for k in ks:
        recalls = []
        reductions = []
        for i in range(args.naive_seeds):
            rf, tf_, rn, tn = agg[f"naive_{i}"][k]
            if tf_ == 0 or tn == 0:
                continue
            recalls.append(100 * rf / tf_)
            reductions.append(100 * (1 - rn / tn))
        if recalls:
            print(
                f"  K={k}: recall={np.mean(recalls):.2f}% (+/-{np.std(recalls):.2f}), "
                f"node_reduction={np.mean(reductions):.2f}%"
            )


if __name__ == "__main__":
    main()
