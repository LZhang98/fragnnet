#!/usr/bin/env python3
"""Scan frag pickles and report any negative h_formulae_idx values.

Usage:
  python scripts/check_h_formulae_idx.py --frag_dp data/frag --max 100
"""

import argparse
import bz2
import os
import pickle

import torch as th

from fragnnet.utils.frag_utils import get_node_feats


def load_frag(fp: str):
    if fp.endswith(".bz2"):
        with bz2.open(fp, "rb") as f:
            return pickle.load(f)
    else:
        with open(fp, "rb") as f:
            return pickle.load(f)


def check_file(fp: str):
    try:
        d = load_frag(fp)
    except Exception as e:
        return True, f"load_error:{e}"

    dag = d.get("dag")
    if dag is None:
        return True, "no_dag"

    node_feats = getattr(dag, "x", None)
    node_feat_idxs = getattr(dag, "node_feat_idxs", None)
    if node_feats is None or node_feat_idxs is None:
        return True, "missing_feats"

    h_idx = get_node_feats(node_feats, node_feat_idxs[0], "h_formulae_idx")
    if (h_idx < 0).any():
        neg_mask = h_idx < 0
        cnt = int(neg_mask.sum().item())
        uniq = [int(x) for x in th.unique(h_idx[neg_mask]).cpu().tolist()]
        return True, {"count": cnt, "unique": uniq}

    return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frag_dp", default="data/ccx/ccx_nist23_d3/dags", help="directory with frag pickles"
    )
    parser.add_argument("--max", type=int, default=None, help="max files to scan")
    args = parser.parse_args()

    files = []
    for root, _, fnames in os.walk(args.frag_dp):
        for fn in fnames:
            if fn.endswith(".pickle") or fn.endswith(".pickle.bz2"):
                files.append(os.path.join(root, fn))
    files.sort()

    total = 0
    bad = 0
    for i, fp in enumerate(files):
        if args.max is not None and i >= args.max:
            break
        total += 1
        has_issue, info = check_file(fp)
        if has_issue:
            bad += 1
            print(f"ISSUE in {fp}: {info}")

    print(
        f"Scanned {total} frag files, found {bad} with issues (negative h_formulae_idx or load problems)"
    )
    if bad > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
