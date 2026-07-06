"""Apply the same three policies (hand-mined, learned, naive) as a per-branch absolute
threshold trim, tested SEPARATELY at each depth d in {1,2,3,4} -- not just depth 4 like
debug_true_wrecall_eval.py.

Trimming a branch at depth d cascades: if a depth-d candidate is pruned, any depth
d+1..4 descendant that ONLY existed through that pruned node must also become
unreachable, UNLESS it has another surviving path via a different, still-alive parent
(this DAG has heavy redundant multi-parent routing, per bond_breakability_mask_plan.md).
This is computed by propagating an "alive" node set forward one depth layer at a time:
depths other than d are never independently pruned, but they DO inherit cascading loss
from an upstream trim at d, since a node's parent(s) might no longer be alive.

At the trim depth itself, each (parent, child) candidate edge is scored and kept
independently iff score >= tau AND its parent is alive -- this is a genuine per-branch
decision, not an aggregation over a node's multiple routes (no max/avg/single-route
ambiguity here, since each edge is evaluated on its own and a node survives if ANY of its
alive-parent edges clears tau).

Real recall/wrecall is computed exactly as in debug_true_wrecall_eval.py, reusing
scripts/stats_scripts/dag_depth_recall_nist20_d3.py's proven matching functions -- a
formula is "allowed" if ANY of its matching nodes (at any depth) survived the cascade.

Usage:
    python scripts/debug/debug_depth_trim_cascade_eval.py \
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
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats_scripts"))
from dag_depth_recall_nist20_d3 import ibd_allowed, vectorized_recall  # noqa: E402
from debug_dag_bond_descriptor_stats import bond_bucket, decode_cc  # noqa: E402
from debug_dag_bond_radius_env_stats import bond_env_bucket, bond_env_str, boundary_bond_pairs  # noqa: E402
from debug_fanout_cap_recall_eval import make_hand_mined_priority  # noqa: E402
from debug_learned_bond_scorer_prototype import build_examples  # noqa: E402

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES = {"FT", "QTOF", "IT"}
FRAG_MODES = {"CID", "HCD"}
HAND_MINED_MAX = 4.17
TAU_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]
TRIM_DEPTHS = [1, 2, 3, 4]


def build_full_dag_structure(dag: dict, mol):
    """Extract per-node atom sets, depths, ALL edges (not restricted to one target
    depth), and formula->matching-nodes (any depth), for the cascading-trim simulation.
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

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    node_depths = [node_depth(ni) for ni in range(n_nodes)]

    formula_to_nodes: dict[int, list[int]] = defaultdict(list)
    for ni in range(n_nodes):
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx > 0:
                formula_to_nodes[fidx].append(ni)

    parent_of: dict[int, list[int]] = defaultdict(list)
    for ei in range(edge_index.shape[1]):
        parent_of[edge_index[1, ei].item()].append(edge_index[0, ei].item())

    nodes_by_depth: dict[int, list[int]] = defaultdict(list)
    for ni in range(n_nodes):
        nodes_by_depth[node_depths[ni]].append(ni)

    return node_cc, node_depths, parent_of, nodes_by_depth, formula_to_nodes, n_nodes


def cascading_alive_set(
    node_cc, parent_of, nodes_by_depth, mol, trim_depth: int, priority_fn, tau: float, max_depth: int
) -> set[int]:
    """Propagate node aliveness forward one depth at a time. At trim_depth, each
    (parent, child) edge is scored and kept independently; a node survives if any edge
    from an alive parent clears tau. At every other depth, a node survives if it has any
    edge from an alive parent at all (no independent pruning there, but cascading loss
    from an upstream trim is still respected, since parent_of is only checked against the
    already-computed alive set from the previous depth).
    """
    alive: set[int] = set(nodes_by_depth[0])
    for d in range(1, max_depth + 1):
        for ni in nodes_by_depth[d]:
            survives = False
            for pi in parent_of[ni]:
                if pi not in alive:
                    continue
                if d != trim_depth:
                    survives = True
                    break
                complement = node_cc[pi] - node_cc[ni]
                if not complement:
                    continue
                pairs = boundary_bond_pairs(mol, node_cc[ni], complement)
                if not pairs:
                    continue
                buckets = {bond_bucket(mol, a, b) for a, b in pairs}
                if priority_fn(buckets, pairs) >= tau:
                    survives = True
                    break
            if survives:
                alive.add(ni)
    return alive


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

    mol_cache_by_mol_id: dict = {}
    mol_id_ref = [None, None]
    naive_rng = random.Random(14000)

    def learned_priority_proba(buckets, pairs):
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

    hand_mined_raw = make_hand_mined_priority()

    def hand_mined_priority_normalized(buckets, pairs):
        return hand_mined_raw(buckets, pairs) / HAND_MINED_MAX

    def naive_priority(_buckets, _pairs):
        return naive_rng.random()

    policies = {
        "hand_mined": hand_mined_priority_normalized,
        "learned": learned_priority_proba,
        "naive": naive_priority,
    }

    # results[(trim_depth, policy, tau)] = list of (recall, wrecall) per real spectrum,
    # plus node-count aggregates for context.
    results: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    node_counts: dict[tuple, list[int]] = defaultdict(list)
    total_node_counts: dict[int, list[int]] = defaultdict(list)
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

            node_cc, _node_depths, parent_of, nodes_by_depth, formula_to_nodes, n_nodes = (
                build_full_dag_structure(dag, mol)
            )
            mol_id_ref[0], mol_id_ref[1] = mol_id, mol
            idx_by_h_delta = dag["idx_by_h_delta"]
            fpm = dag["formula_peak_mzs"].numpy()
            fpp = dag["formula_peak_probs"].numpy()

            for trim_depth in TRIM_DEPTHS:
                for policy_name, priority_fn in policies.items():
                    for tau in TAU_GRID:
                        alive = cascading_alive_set(
                            node_cc, parent_of, nodes_by_depth, mol, trim_depth,
                            priority_fn, tau, args.max_depth,
                        )
                        allowed_formulas = {
                            fidx for fidx, nodes in formula_to_nodes.items()
                            if any(ni in alive for ni in nodes)
                        }
                        allowed = ibd_allowed(idx_by_h_delta, allowed_formulas)
                        for sr in spec_rows:
                            r, w = vectorized_recall(
                                sr["peaks"], fpm, fpp, allowed, sr["prec_type"], sr["inst_type"]
                            )
                            results[(trim_depth, policy_name, tau)].append((r, w))
                        node_counts[(trim_depth, policy_name, tau)].append(len(alive))
                total_node_counts[trim_depth].append(n_nodes)

            n_mols_used += 1
            n_specs_used += len(spec_rows)

    print(f"\nUsed {n_mols_used} test molecules, {n_specs_used} real spectra")

    for trim_depth in TRIM_DEPTHS:
        print(f"\n{'='*90}")
        print(f"TRIM AT DEPTH {trim_depth}")
        print(f"{'='*90}")
        print(f"{'Policy':<14} {'tau':>6} {'wrecall':>10} {'d(wrecall)':>12} {'nodes kept':>12}")
        print(f"{'-'*90}")
        total_n = np.mean(total_node_counts[trim_depth])
        for policy_name in policies:
            base_w = None
            for tau in TAU_GRID:
                rows = results[(trim_depth, policy_name, tau)]
                if not rows:
                    continue
                w_mean = np.mean([w for _r, w in rows])
                if base_w is None:
                    base_w = w_mean
                node_frac = 100 * np.mean(node_counts[(trim_depth, policy_name, tau)]) / total_n
                print(
                    f"{policy_name:<14} {tau:>6.2f} {w_mean:>10.4f} {w_mean-base_w:>+12.4f} "
                    f"{node_frac:>11.2f}%"
                )


if __name__ == "__main__":
    main()
