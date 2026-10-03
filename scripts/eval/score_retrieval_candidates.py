"""Score retrieval candidates' predicted spectra against their query spectra.

Inputs
------
* ``--candidates``: CSV with one row per candidate spectrum: ``cand_uid`` (the spectrum id in the
  prediction file), ``query_spec_id`` (row of ``--query_spec_df``) and ``is_truth``.
* ``--query_spec_df``: spec_df pickle holding the query spectra (``spec_id``, ``peaks``,
  ``prec_mz``, ``inst_type``), e.g. ``data/proc/ms_gym_benchmark_1_5/spec_df.pkl``.
* ``--pred_fp`` with ``--pred_format``:
  ``masster`` (``run_save_inference_eval.py`` HDF5: linear intensities, ``spec_id`` = cand_uid;
  export with ``--min_peak_int -1`` so the low-intensity tail is kept) or
  ``mspred`` (ms-pred ``predict_smis_joint.py`` HDF5, e.g. GLACIER: sqrt intensities under
  ``pred_<name>``; ``--mspred_prefix`` maps ``cand_uid`` to ``<name>``).

Scores (``--scores``), all higher-is-better, written per candidate:
* ``mspred_cos`` / ``mspred_ent``: ms-pred's binned cosine / entropy similarity (GLACIER's
  published protocol; ``fragnnet.utils.retrieval_utils.mspred_*``).
* ``ent_inst_tol``: the same entropy similarity with one-to-one peak matching at instrument
  tolerance; peaks above precursor m/z + tolerance removed from both spectra.
* ``loglik``: query log-likelihood under the full prediction at instrument tolerance
  (MASSter exports only; a top-k sparse ms-pred output would floor most of the query).
``--min_rel_intensity`` (e.g. 0.01) drops prediction peaks below that fraction of the base peak
before ``mspred_*`` and ``ent_inst_tol`` (not ``loglik``).

Example::

    python scripts/eval/score_retrieval_candidates.py \\
        --candidates output/.../test/shard_00/candidates.csv \\
        --query_spec_df data/proc/ms_gym_benchmark_1_5/spec_df.pkl \\
        --pred_format masster --pred_fp output/.../masster/.../predictions_test.hdf5 \\
        --scores mspred_cos mspred_ent ent_inst_tol loglik --out output/.../scores_masster.pkl
"""

import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import torch as th

from fragnnet.utils.retrieval_utils import (
    entropy_similarity_inst_tol,
    mspred_cosine,
    mspred_entropy,
    mspred_prediction,
    mspred_target,
    peak_tolerance,
    prepare_sqrt_peaks,
    query_log_likelihood,
)

logger = logging.getLogger(__name__)
SCORES = ("mspred_cos", "mspred_ent", "ent_inst_tol", "loglik")


def parse_inst_tol(items: list[str]) -> dict[str, tuple[float, float]]:
    """Parse ``INST:rel:min_mz`` items, e.g. ``ORB:1e-5:200``."""
    out = {}
    for item in items:
        inst, rel, min_mz = item.split(":")
        out[inst] = (float(rel), float(min_mz))
    return out


def load_predictions(
    fp: str, fmt: str, uids: np.ndarray, prefix: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return ``{cand_uid: (mzs, linear_intensities)}`` for the requested candidates."""
    wanted = {int(u) for u in uids}
    preds = {}
    with h5py.File(fp, "r") as f:
        if fmt == "masster":
            sid, st, en = f["spec_id"][:], f["start_points"][:], f["end_points"][:]
            mzs, ints = f["pred_mzs"], f["pred_ints"]
            for u, a, b in zip(sid, st, en):
                if int(u) in wanted:
                    preds[int(u)] = (mzs[a:b].astype(np.float64), ints[a:b].astype(np.float64))
        else:
            for u in wanted:
                key = f"pred_{prefix}{u}"
                if key not in f:
                    continue
                (ikey_group,) = f[key].values()
                (ce_group,) = ikey_group.values()
                arr = ce_group["f"][:].astype(np.float64)
                arr = arr[arr[:, 1] > 0]
                preds[u] = (arr[:, 0], arr[:, 1] ** 2)  # ms-pred stores sqrt intensity
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--query_spec_df", required=True)
    parser.add_argument("--pred_fp", required=True)
    parser.add_argument("--pred_format", required=True, choices=["masster", "mspred"])
    parser.add_argument(
        "--mspred_prefix", default="P", help="ms-pred names are pred_<prefix><cand_uid>"
    )
    parser.add_argument("--scores", nargs="+", default=list(SCORES), choices=SCORES)
    parser.add_argument("--inst_tol", nargs="+", default=["ORB:1e-5:200", "QTOF:2e-5:750"])
    parser.add_argument("--min_rel_intensity", type=float, default=None)
    parser.add_argument("--loglik_eps", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if "loglik" in args.scores and args.pred_format != "masster":
        raise ValueError("loglik needs a full MASSter export; ms-pred outputs are top-k sparse")
    tol_table = parse_inst_tol(args.inst_tol)
    cand = pd.read_csv(args.candidates)
    spec = pd.read_pickle(args.query_spec_df).set_index("spec_id")
    preds = load_predictions(
        args.pred_fp, args.pred_format, cand["cand_uid"].to_numpy(), args.mspred_prefix
    )
    logger.info(
        "%d candidates, %d predictions, %d queries",
        len(cand),
        len(preds),
        cand["query_spec_id"].nunique(),
    )

    rows = []
    for qid, cq in cand.groupby("query_spec_id", sort=True):
        q = spec.loc[qid]
        inst = str(q["inst_type"])
        if inst not in tol_table:
            raise ValueError(f"no --inst_tol entry for instrument {inst!r}")
        rel, min_mz = tol_table[inst]
        prec = float(q["prec_mz"])
        pk = np.asarray(q["peaks"], dtype=np.float64)
        q_mz, q_int = pk[:, 0], pk[:, 1]
        uids = [int(u) for u in cq["cand_uid"] if int(u) in preds]
        res = {int(u): {"cand_uid": int(u)} for u in cq["cand_uid"]}

        # 1. ms-pred binned similarities.
        if {"mspred_cos", "mspred_ent"} & set(args.scores):
            tgt = mspred_target(q_mz, q_int, parent_mass=prec)
            for u in uids:
                p = mspred_prediction(
                    *preds[u], sqrt_input=False, min_rel_intensity=args.min_rel_intensity
                )
                if "mspred_cos" in args.scores:
                    res[u]["mspred_cos"] = mspred_cosine(tgt, p)
                if "mspred_ent" in args.scores:
                    res[u]["mspred_ent"] = mspred_entropy(tgt, p)

        # 2. Entropy similarity at instrument tolerance, above-precursor peaks removed.
        if "ent_inst_tol" in args.scores:
            cut = prec + float(peak_tolerance(np.array([prec]), rel, min_mz)[0])
            tgt = prepare_sqrt_peaks(q_mz, q_int, max_mz=cut)
            prepared = [
                prepare_sqrt_peaks(*preds[u], max_mz=cut, min_rel_intensity=args.min_rel_intensity)
                for u in uids
            ]
            for u, s in zip(
                uids, entropy_similarity_inst_tol(tgt, prepared, rel, min_mz, device=args.device)
            ):
                res[u]["ent_inst_tol"] = s

        # 3. Query log-likelihood at instrument tolerance.
        if "loglik" in args.scores:
            y = q_int / q_int.sum()
            for u in uids:
                mz, lin = preds[u]
                order = np.argsort(mz)
                res[u]["loglik"] = query_log_likelihood(
                    q_mz, y, mz[order], lin[order] / lin.sum(), rel, min_mz, eps=args.loglik_eps
                )
        rows += list(res.values())

    out = cand[["cand_uid", "query_spec_id", "is_truth"]].merge(
        pd.DataFrame(rows), on="cand_uid", how="left"
    )
    missing = out[list(args.scores)].isna().any(axis=1).sum()
    logger.info("writing %s (%d rows, %d with a missing score)", args.out, len(out), missing)
    out.to_pickle(args.out)


if __name__ == "__main__":
    main()
