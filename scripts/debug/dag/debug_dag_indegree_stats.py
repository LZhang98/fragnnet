"""DAG in-degree (multi-parent) distribution, for the FluxMPNN Stage 1/2 review.

Stage 1 collapses a per-EDGE sibling softmax `β(p->c)` into a per-NODE scalar residual, so the way
multi-parent nodes combine their incoming branching evidence is the crux of the approximation (and of
whether a zero-init residual is truly zero at init). This quantifies how common multi-parent nodes
are: if most nodes are single-parent, the per-node projection is nearly lossless; if many are
multi-parent, the combiner choice (logsumexp / mean / flux-weighted) matters and biases init.

Usage:
    python scripts/debug/dag/debug_dag_indegree_stats.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags --limit_mols 300
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import fragnnet.models.fragnnet.frag_utils as frag_utils


def _detect_is_compressed(frag_dp: str) -> bool:
    """Whether DAG pickle files under frag_dp are bz2-compressed (self-contained, no eval import)."""
    if os.path.isfile(frag_dp) and str(frag_dp).endswith((".tar", ".zip")):
        return True
    if glob.glob(os.path.join(frag_dp, "*.pkl.bz2")):
        return True
    if glob.glob(os.path.join(frag_dp, "*.pkl")):
        return False
    raise FileNotFoundError(f"No DAG .pkl or .pkl.bz2 files found under {frag_dp}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frag_dp", default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--limit_mols", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_pickle(args.spec_fp)
    mol_ids = df["mol_id"].drop_duplicates().values
    is_compressed = _detect_is_compressed(args.frag_dp)

    indeg_all, nnodes, frac_multi = [], [], []
    n_loaded = 0
    for m in mol_ids:
        if n_loaded >= args.limit_mols:
            break
        try:
            dag_d = frag_utils.load_frag_d(int(m), args.frag_dp, is_compressed=is_compressed)
        except FileNotFoundError:
            continue  # molecule not fragmented in this dir
        if dag_d is None:
            continue
        n_loaded += 1
        edge_index = dag_d["dag"].edge_index.numpy()
        num_nodes = dag_d["dag"].x.shape[0]
        indeg = np.bincount(edge_index[1], minlength=num_nodes)
        indeg_all.append(indeg)
        nnodes.append(num_nodes)
        frac_multi.append((indeg >= 2).sum() / max(num_nodes, 1))

    if not indeg_all:
        raise ValueError(f"No DAGs loaded from {args.frag_dp}")
    all_indeg = np.concatenate(indeg_all)
    print(f"DAGs loaded: {n_loaded}   mean nodes/DAG: {np.mean(nnodes):.1f}")
    print(
        f"in-degree: mean {all_indeg.mean():.2f}  median {int(np.median(all_indeg))}  "
        f"max {int(all_indeg.max())}"
    )
    print(
        f"node in-degree share: ==0 (root) {(all_indeg == 0).mean():.3f}  "
        f"==1 {(all_indeg == 1).mean():.3f}  >=2 (multi-parent) {(all_indeg >= 2).mean():.3f}"
    )
    print(f"mean per-DAG fraction of multi-parent nodes: {np.mean(frac_multi):.3f}")


if __name__ == "__main__":
    main()
