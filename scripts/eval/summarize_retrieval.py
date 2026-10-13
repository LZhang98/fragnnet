"""Summarize candidate scores into retrieval hit@k with intervals and paired model comparisons.

Takes one or more per-candidate score pickles from ``score_retrieval_candidates.py`` (one per
model, each tagged ``label=path``), a query table giving each query's molecule key (for the
molecule bootstrap) and optional instrument, and writes one row per (label, score, subset):

* expected hit@1/5/20 under random tie-breaking (``retrieval_utils.expected_hits``; any truth
  counts; never list order, since MassSpecGym lists the truth first);
* a 95% molecule-bootstrap interval (resamples molecules; spectra of one molecule move together);
* ms-pred's 99.9% i.i.d. BCa interval (``retrieval_utils.mspred_bootstrap_ci``), for comparison
  with published numbers only: it treats correlated spectra as independent.

``--compare A:B`` adds paired hit@1 differences (A - B, molecule bootstrap) for every score both
labels have. Score pickles may cover several shards: pass a glob in quotes.

Example::

    python scripts/eval/summarize_retrieval.py \\
        --scores "masster=output/.../test/shard_*/scores_masster.pkl" \\
                 "glacier=output/.../test/shard_*/scores_glacier.pkl" \\
        --queries output/.../test/queries.csv --query_id_col spec_id \\
        --mol_col truth_inchikey_s --subset_col inst_type --compare glacier:masster \\
        --out output/.../test/retrieval_summary.csv
"""

import argparse
import glob
import logging

import numpy as np
import pandas as pd

from fragnnet.utils.retrieval_utils import expected_hits, molecule_bootstrap_ci, mspred_bootstrap_ci

logger = logging.getLogger(__name__)
KS = (1, 5, 20)
ID_COLS = {"cand_uid", "query_spec_id", "is_truth"}


def per_query_hits(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Per-query expected hit@k for one score column (index: query_spec_id)."""
    rec = {}
    for qid, g in df.groupby("query_spec_id", sort=True):
        rec[qid] = expected_hits(g[score].to_numpy(float), g["is_truth"].to_numpy(bool), KS)
    return pd.DataFrame.from_dict(rec, orient="index", columns=[f"hit@{k}" for k in KS])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores", nargs="+", required=True, help="label=path_or_glob")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--query_id_col", default="spec_id")
    parser.add_argument("--mol_col", required=True)
    parser.add_argument(
        "--subset_col", default=None, help="optional column to break results down by"
    )
    parser.add_argument("--compare", nargs="*", default=[], help="A:B pairs of labels")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queries = pd.read_csv(args.queries).set_index(args.query_id_col)
    hits: dict[tuple[str, str], pd.DataFrame] = {}
    for item in args.scores:
        label, pattern = item.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no score files match {pattern}")
        df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        if df["cand_uid"].duplicated().any():
            raise ValueError(f"{label}: duplicate cand_uid across {len(files)} files")
        for score in [c for c in df.columns if c not in ID_COLS]:
            hits[(label, score)] = per_query_hits(df, score)
        logger.info(
            "%s: %d files, %d candidates, %d queries",
            label,
            len(files),
            len(df),
            df["query_spec_id"].nunique(),
        )

    rows = []
    for (label, score), h in hits.items():
        q = queries.loc[h.index]
        subsets = [("all", np.ones(len(h), bool))]
        if args.subset_col:
            subsets += [
                (str(v), (q[args.subset_col] == v).to_numpy())
                for v in sorted(q[args.subset_col].unique())
            ]
        for name, mask in subsets:
            v = h.to_numpy()[mask]
            mean, lo, hi = molecule_bootstrap_ci(
                v[:, 0], q[args.mol_col].to_numpy()[mask], n_resamples=args.n_boot
            )
            _, blo, bhi = mspred_bootstrap_ci(v[:, 0])
            rows.append(
                {
                    "label": label,
                    "score": score,
                    "subset": name,
                    "n_queries": int(mask.sum()),
                    **{f"hit@{k}": v[:, j].mean() for j, k in enumerate(KS)},
                    "hit@1_mol95_lo": lo,
                    "hit@1_mol95_hi": hi,
                    "hit@1_mspred999_lo": blo,
                    "hit@1_mspred999_hi": bhi,
                }
            )
    for pair in args.compare:
        a, b = pair.split(":")
        for score in sorted(
            {s for (lab, s) in hits if lab == a} & {s for (lab, s) in hits if lab == b}
        ):
            ha, hb = hits[(a, score)], hits[(b, score)]
            common = ha.index.intersection(hb.index)
            d = ha.loc[common, "hit@1"].to_numpy() - hb.loc[common, "hit@1"].to_numpy()
            mean, lo, hi = molecule_bootstrap_ci(
                d, queries.loc[common, args.mol_col].to_numpy(), n_resamples=args.n_boot
            )
            rows.append(
                {
                    "label": f"{a}-{b}",
                    "score": score,
                    "subset": "all",
                    "n_queries": len(common),
                    "hit@1": mean,
                    "hit@1_mol95_lo": lo,
                    "hit@1_mol95_hi": hi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.precision", 4):
        logger.info("\n%s", out[out["subset"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
