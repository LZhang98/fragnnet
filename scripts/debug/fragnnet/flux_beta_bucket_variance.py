"""Does the trained FluxMPNN pool beta across edges sharing a bond environment?

Tests the "capacity beats tying" diagnosis of the Stage 3 null: a GNN-produced per-edge
beta can give every edge its own value, so evidence never pools across molecules and beta
stays unidentified.  CFM-style coarse tying (beta = f(bond features)) would instead force
edges in the same bucket to share.

beta is NOT in the inference dump, but is exactly recoverable for single-parent children
from the exported node-level flow latents, since such a child's inflow is one edge's flux:

    log_f_in    = log_f_sink - log_alpha_halt
    log_beta(p->c) = log_f_in(c) - log_f_in(p) - log(1 - alpha_halt(p))

Variance of log_beta is decomposed three ways:

  1. within-edge   : same edge across spectra -> CE/context modulation
  2. within-bucket : different edges, same bond bucket -> THE POOLING TEST
  3. between-bucket: bond chemistry actually captured

Large (2) relative to (3) means the model gave near-arbitrary per-edge values to
chemically identical cleavages -> nothing pooled -> the diagnosis holds.

Usage:
    python scripts/debug/fragnnet/flux_beta_bucket_variance.py \\
        --pred_fp output/inference_eval/w4q1b1px/predictions_val.hdf5 \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --mol_df data/proc/nist20/mol_df.pkl --limit_mols 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dag")
)
from dag_oracle_scores import _detect_is_compressed  # noqa: E402
from debug_dag_bond_descriptor_stats import bond_bucket, decode_cc  # noqa: E402
from debug_dag_bond_radius_env_stats import bond_env_bucket  # noqa: E402

from fragnnet.utils import dag_format_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOG_FLUX_MIN = -20.0  # ignore edges whose parent carries numerically negligible flux


def edge_bucket_key(
    mol, child_atoms: frozenset, parent_atoms: frozenset, radius: int | None, env_cache: dict
) -> tuple | None:
    """Bond-environment key for one DAG edge: the sorted bucket set of its cut bonds.

    Args:
        radius: None for the coarse (in_ring, element_pair, alpha_hetero) bucket, else the
            radius-N circular bond environment.
    """
    complement = parent_atoms - child_atoms
    if not complement:
        return None
    buckets = set()
    for a in child_atoms:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            b = nbr.GetIdx()
            if b in complement:
                if radius is None:
                    buckets.add(bond_bucket(mol, a, b))
                else:
                    buckets.add(bond_env_bucket(mol, a, b, radius, env_cache))
    if not buckets:
        return None
    return tuple(sorted(buckets))


def log1mexp(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    out = np.full_like(x, -np.inf)
    ok = x < 0
    out[ok] = np.where(
        x[ok] < -0.693,
        np.log1p(-np.exp(x[ok])),
        np.log(-np.expm1(x[ok])),
    )
    return out


def collect(
    pred_fp: str,
    frag_dp: str,
    mol_by_id: dict,
    limit_mols: int | None,
    seed: int,
    radii: list[int],
) -> dict[tuple, dict]:
    """Accumulate log-beta sums per (mol, parent, child) edge, tagged with bucket + depth."""
    with h5py.File(pred_fp, "r") as f:
        for key in ("pred_fluxmpnn_flow_node_logprobs", "pred_fluxmpnn_log_alpha_halt"):
            if key not in f:
                raise ValueError(f"{pred_fp} lacks {key}; needs a flow-model inference dump.")
        mol_ids = f["mol_id"][:]
        n_start = f["node_start_points"][:]
        n_end = f["node_end_points"][:]

        uniq = np.unique(mol_ids)
        if limit_mols is not None and limit_mols < uniq.size:
            uniq = np.random.default_rng(seed).choice(uniq, size=limit_mols, replace=False)
        keep = np.isin(mol_ids, uniq)
        rows_by_mol: dict[int, list[int]] = defaultdict(list)
        for ri in np.nonzero(keep)[0]:
            rows_by_mol[int(mol_ids[ri])].append(int(ri))
        log.info("Processing %d mols (%d spectra)", len(rows_by_mol), int(keep.sum()))

        is_compressed = _detect_is_compressed(frag_dp, next(iter(rows_by_mol)))
        acc: dict[tuple, dict] = {}
        n_skip_mol = 0

        for mi, (mol_id, rows) in enumerate(rows_by_mol.items()):
            if mi % 50 == 0:
                log.info("  %d/%d mols", mi, len(rows_by_mol))
            dag_d = dag_format_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
            mol = mol_by_id.get(mol_id)
            if dag_d is None or mol is None:
                n_skip_mol += 1
                continue
            dag = dag_d["dag"]
            edge_index = dag.edge_index.numpy()
            num_nodes = dag.x.shape[0]
            num_atoms = mol.GetNumAtoms()

            indeg = np.bincount(edge_index[1], minlength=num_nodes)
            outdeg = np.bincount(edge_index[0], minlength=num_nodes)
            sp_edges = np.nonzero(indeg[edge_index[1]] == 1)[0]
            if sp_edges.size == 0:
                continue

            feat_idxs = dag.node_feat_idxs[0]
            cc_start, cc_end = int(feat_idxs[1]), int(feat_idxs[2])
            node_depth = dag_d["nodes_min_depth"]

            cc_cache: dict[int, frozenset] = {}
            env_cache: dict = {}

            def cc_of(ni: int) -> frozenset:
                if ni not in cc_cache:
                    cc_cache[ni] = decode_cc(dag.x[ni, cc_start:cc_end], num_atoms)
                return cc_cache[ni]

            # Bond bucket is spectrum-independent: resolve once per edge.
            edge_keys: dict[int, tuple] = {}
            for ei in sp_edges:
                p, c = int(edge_index[0, ei]), int(edge_index[1, ei])
                keys = tuple(
                    edge_bucket_key(mol, cc_of(c), cc_of(p), r, env_cache) for r in radii
                )
                if all(k is not None for k in keys):
                    edge_keys[int(ei)] = keys
            if not edge_keys:
                continue

            for ri in rows:
                s, e = int(n_start[ri]), int(n_end[ri])
                if e - s != num_nodes:
                    raise ValueError(
                        f"node slice {e - s} != DAG nodes {num_nodes} for mol {mol_id}; "
                        "frag_dp does not match the run's DAGs."
                    )
                sink = f["pred_fluxmpnn_flow_node_logprobs"][s:e].astype(np.float64)
                halt = f["pred_fluxmpnn_log_alpha_halt"][s:e].astype(np.float64)
                f_in = sink - halt
                cont = log1mexp(halt)

                for ei, key in edge_keys.items():
                    p, c = int(edge_index[0, ei]), int(edge_index[1, ei])
                    if f_in[p] < LOG_FLUX_MIN or not np.isfinite(cont[p]):
                        continue
                    lb = f_in[c] - f_in[p] - cont[p]
                    if not np.isfinite(lb) or lb > 1e-6:  # beta is a probability
                        continue
                    # Excess over a uniform split: removes the -log(out_degree) baseline that
                    # the sibling softmax bakes into log beta, leaving the chemistry signal.
                    lb_x = lb + np.log(max(int(outdeg[p]), 1))
                    ek = (mol_id, p, c)
                    a = acc.get(ek)
                    if a is None:
                        a = acc[ek] = {
                            "keys": key,
                            "depth": int(node_depth[c]),
                            "outdeg": int(outdeg[p]),
                            "n": 0,
                            "s": 0.0,
                            "ss": 0.0,
                            "sx": 0.0,
                            "ssx": 0.0,
                        }
                    a["n"] += 1
                    a["s"] += lb
                    a["ss"] += lb * lb
                    a["sx"] += lb_x
                    a["ssx"] += lb_x * lb_x

    log.info("Collected %d edges (%d mols skipped)", len(acc), n_skip_mol)
    return acc


def summarize(acc: dict[tuple, dict], min_bucket_edges: int, radii: list[int]) -> None:
    """ICC across radii, on the full set and on the edge set the finest radius retains."""
    edges = [a for a in acc.values() if a["n"] >= 1]
    if not edges:
        raise ValueError("No usable edges recovered.")

    def icc(subset: list, ri: int) -> tuple:
        by_bucket: dict[tuple, list[float]] = defaultdict(list)
        for a in subset:
            by_bucket[a["keys"][ri]].append(a["sx"] / a["n"])
        kept = {k: v for k, v in by_bucket.items() if len(v) >= min_bucket_edges}
        if not kept:
            return float("nan"), float("nan"), float("nan"), 0, 0
        grand = np.array([m for v in kept.values() for m in v]).mean()
        wb, bb, n = 0.0, 0.0, 0
        for v in kept.values():
            arr = np.asarray(v)
            wb += ((arr - arr.mean()) ** 2).sum()
            bb += len(arr) * (arr.mean() - grand) ** 2
            n += len(arr)
        wb, bb = wb / n, bb / n
        return bb / (wb + bb), wb, bb, n, len(kept)

    fine = len(radii) - 1
    fine_buckets: dict[tuple, int] = defaultdict(int)
    for a in edges:
        fine_buckets[a["keys"][fine]] += 1
    fixed = [a for a in edges if fine_buckets[a["keys"][fine]] >= min_bucket_edges]

    print(f"\n{'=' * 88}")
    print("FluxMPNN log-beta: does finer bond environment explain more? (excess-over-uniform)")
    print(f"{'=' * 88}")
    print(f"edges recovered: {len(edges):,}   fixed set (radius-{radii[fine]} support): {len(fixed):,}")
    print()
    print(f"  {'radius':>7s} {'ICC(all)':>10s} {'n(all)':>9s} {'buckets':>8s}   "
          f"{'ICC(fixed)':>11s} {'n(fixed)':>9s} {'buckets':>8s}")
    for ri, r in enumerate(radii):
        ia, _, _, na, ba = icc(edges, ri)
        ix, _, _, nx, bx = icc(fixed, ri)
        print(f"  {r:>7d} {ia:>10.4f} {na:>9,d} {ba:>8d}   {ix:>11.4f} {nx:>9,d} {bx:>8d}")
    print(f"\n{'-' * 88}")
    print("READ:")
    print("  ICC(fixed) RISING with radius -> the GNN really does encode finer bond chemistry;")
    print("    the earlier low coarse-bucket ICC was bucket crudeness, not per-edge arbitrariness.")
    print("  ICC(fixed) FLAT -> the rise was selection (finer buckets retain stereotyped motifs).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_fp", required=True, help="Flow-model inference HDF5.")
    parser.add_argument("--frag_dp", required=True, help="DAG dir matching the run.")
    parser.add_argument("--mol_df", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--limit_mols", type=int, default=None)
    parser.add_argument("--min_bucket_edges", type=int, default=20)
    parser.add_argument("--radii", default="0,1,2", help="Comma-separated radii, coarsest first.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))
    radii = [int(r) for r in args.radii.split(",")]
    acc = collect(args.pred_fp, args.frag_dp, mol_by_id, args.limit_mols, args.seed, radii)
    summarize(acc, args.min_bucket_edges, radii)


if __name__ == "__main__":
    main()
