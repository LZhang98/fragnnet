"""End-to-end MassSpecGym 1.5 retrieval for a MASSter checkpoint: prepare, infer, score, summarize.

Runs locally or as a Fir/Vulcan job (``scripts/manual_slurm/prepare_msgym_retrieval.sh``).
Every stage is resumable: a finished shard leaves a marker file and is skipped on re-run.

Queries are the rows of the config's benchmark split for ``--fold`` (the MSGYM 1.5 benchmark
split is the simulation-challenge ``[M+H]+`` spectra with a CE). Each candidate row is the
query's benchmark ``spec_df`` row with only ``mol_id`` / ``spec_id`` / ``group_id`` replaced, so
the model sees the query's CE, instrument and dual-CE extras. The truth is every candidate whose
2D InChIKey matches MSGYM's source SMILES; MSGYM lists the truth first, so hit@k never uses list
order (``retrieval_utils.expected_hits``).

Layout under ``--run_dp`` (data is checkpoint-independent; outputs are per checkpoint label)::

    data/                       01_prepare_msgym_retrieval.py export + build-proc of the fold
    queries.csv                 one row per query, with truth_inchikey_s and inst_type
    shard_NNN/                  spec_df.pkl, mol_df.pkl, split/test_ids.csv, candidates.csv
    shard_NNN/<label>/          predictions_test.hdf5, scores.pkl, inferred / scored markers
                                (<label> is the checkpoint file stem)
    summary_<label>.csv         summarize_retrieval.py output

Example::

    python scripts/ms2c/run_msgym_retrieval.py \\
        --custom_fp config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_glacier_feats.yml \\
        --model_ckpt output/annotation_retrieval_pilot_2026_10_02/ckpt/ll04i00m_ep060_orb.ckpt \\
        --tsv_fp data/raw/msgym_1_5/MassSpecGym1.5.tsv \\
        --candidates_fp data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_formula.json \\
        --fold test --run_dp output/msgym_retrieval/formula_test
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from fragnnet.runner import load_config
from fragnnet.utils.misc_utils import booltype

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
STAGES = ("prepare", "infer", "score", "summarize")


def run(cmd: list[str], log_fp: Path) -> None:
    """Run a subprocess from the repository root, logging to ``log_fp``; raise on failure."""
    logger.info("running %s (log %s)", " ".join(cmd[:3]), log_fp)
    with open(log_fp, "w") as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}; see {log_fp}")


def prepare(args: argparse.Namespace, config_d: dict, run_dp: Path) -> None:
    """Export the fold's candidates, then write query-sharded spec/mol dataframes."""
    # 1. Candidate molecules for every spectrum of the fold (export + build-proc).
    data_dp = run_dp / "data"
    proc_dp = data_dp / "proc"
    prep = str(REPO / "preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py")
    if not (proc_dp / "spec_df.pkl").exists():
        data_dp.mkdir(parents=True, exist_ok=True)
        run(
            [sys.executable, prep, "export", "--tsv_fp", args.tsv_fp,
             "--candidates_fp", args.candidates_fp, "--output_dp", str(data_dp),
             "--subset", "all", "--fold", args.fold],
            data_dp / "export.log",
        )  # fmt: skip
        run(
            [sys.executable, prep, "build-proc",
             "--manifest_fp", str(data_dp / "query_candidates.pkl.gz"), "--output_dp", str(proc_dp)],
            data_dp / "build_proc.log",
        )  # fmt: skip
    if (run_dp / "queries.csv").exists():
        logger.info("shards exist under %s; skipping prepare", run_dp)
        return

    # 2. Queries: the config's benchmark split rows, with MSGYM's source key.
    RDLogger.DisableLog("rdApp.*")
    b_spec = pd.read_pickle(config_d["spec_fp"])
    b_mol = pd.read_pickle(config_d["mol_fp"])[["mol_id", "inchikey_s"]]
    split = pd.read_csv(os.path.join(config_d["split_dp"], f"{args.fold}_ids.csv"))
    tsv = pd.read_csv(
        args.tsv_fp,
        sep="\t",
        usecols=["identifier", "smiles", "adduct", "instrument_type", "collision_energy",
                 "precursor_mz"],
    )  # fmt: skip
    q = b_spec[b_spec["spec_id"].isin(split["spec_id"])][
        ["spec_id", "dset_spec_id", "mol_id", "inst_type"]
    ]
    n_split = len(q)
    q = q.merge(b_mol, on="mol_id").merge(tsv, left_on="dset_spec_id", right_on="identifier")
    if len(q) != n_split:
        raise ValueError(f"{n_split - len(q)} benchmark {args.fold} spectra are not in {args.tsv_fp}")
    q["truth_inchikey_s"] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in q["smiles"]]
    q = q.sort_values("spec_id").reset_index(drop=True)

    # 3. Candidates in list order; truth by the source key.
    r_spec = pd.read_pickle(proc_dp / "spec_df.pkl")[["source_identifier", "mol_id", "spec_id"]]
    r_spec = r_spec[r_spec["source_identifier"].isin(q["dset_spec_id"])].sort_values("spec_id")
    r_spec["list_pos"] = r_spec.groupby("source_identifier", observed=True).cumcount()
    r_mol = pd.read_pickle(proc_dp / "mol_df.pkl")
    cand = r_spec.merge(r_mol[["mol_id", "inchikey_s"]], on="mol_id").rename(
        columns={"mol_id": "cand_mol_id", "inchikey_s": "cand_inchikey_s"}
    )
    cand = cand.merge(
        q[["dset_spec_id", "spec_id", "truth_inchikey_s"]].rename(
            columns={"dset_spec_id": "source_identifier", "spec_id": "query_spec_id"}
        ),
        on="source_identifier",
    )
    cand["is_truth"] = cand["cand_inchikey_s"] == cand["truth_inchikey_s"]
    n_truth = cand.groupby("query_spec_id")["is_truth"].sum()
    if len(n_truth) != len(q) or (n_truth < 1).any():
        raise ValueError(
            f"{len(q) - len(n_truth)} queries have no candidates and "
            f"{int((n_truth < 1).sum())} have no truth candidate"
        )
    cand = cand.sort_values(["query_spec_id", "list_pos"]).reset_index(drop=True)
    cand["cand_uid"] = np.arange(len(cand))

    # Drop candidates MASSter's dataset cannot load (GetMolFrags raises, e.g. a hypervalent
    # perchlorate); they would crash a shard's inference. cand_uid is kept, so a dropped
    # candidate is simply absent, i.e. never outranks the truth.
    bad = []
    used = r_mol[r_mol["mol_id"].isin(cand["cand_mol_id"])]
    for mid, m in zip(used["mol_id"], used["mol"]):
        try:
            Chem.GetMolFrags(m, asMols=True)
        except Exception:  # RDKit raises several sanitization error types
            bad.append(mid)
    excluded = cand["cand_mol_id"].isin(bad)
    if cand.loc[excluded, "is_truth"].any():
        raise ValueError(f"an unloadable candidate is a truth: {sorted(bad)}")
    cand[excluded].to_csv(run_dp / "excluded_candidates.csv", index=False)
    cand = cand[~excluded]
    logger.info(
        "%s: %d queries, %d candidates (%d unloadable excluded), truths per query %s",
        args.fold,
        len(q),
        len(cand),
        int(excluded.sum()),
        n_truth.value_counts().to_dict(),
    )

    # 4. Shards of --queries_per_shard queries; queries.csv last, as the completion marker.
    q["shard"] = np.arange(len(q)) // args.queries_per_shard
    cand["shard"] = cand["query_spec_id"].map(q.set_index("spec_id")["shard"])
    b_idx = b_spec.set_index("spec_id")
    r_mol_idx = r_mol.set_index("mol_id")
    for s, cs in cand.groupby("shard"):
        sd = run_dp / f"shard_{s:03d}"
        (sd / "split").mkdir(parents=True, exist_ok=True)
        spec = b_idx.loc[cs["query_spec_id"]].reset_index(drop=True)
        spec["spec_id"] = cs["cand_uid"].to_numpy()
        spec["mol_id"] = cs["cand_mol_id"].to_numpy()
        spec["group_id"] = cs["cand_uid"].to_numpy()  # one spectrum per group: no merging
        mol = r_mol_idx.loc[np.unique(cs["cand_mol_id"])].reset_index()[r_mol.columns]
        spec.to_pickle(sd / "spec_df.pkl")
        mol.to_pickle(sd / "mol_df.pkl")
        spec[["spec_id", "mol_id", "group_id"]].to_csv(sd / "split/test_ids.csv", index=False)
        cs.drop(columns=["shard"]).to_csv(sd / "candidates.csv", index=False)
        logger.info("shard %03d: %d queries, %d candidates", s, cs["query_spec_id"].nunique(),
                    len(cs))  # fmt: skip
    q.drop(columns=["smiles"]).to_csv(run_dp / "queries.csv", index=False)


def infer_and_score(args: argparse.Namespace, sd: Path, label: str) -> None:
    """Run MASSter inference on one shard, then score its candidates."""
    out_dp = sd / label
    pred_fp = out_dp / "predictions_test.hdf5"
    # 1. Inference: the full predicted distribution (no intensity floor), scored in-model;
    # eval_bin_sqrt adds cos_sim_sqrt_<res> next to every binned score.
    if "infer" in args.stages and not (out_dp / "inferred").exists():
        out_dp.mkdir(parents=True, exist_ok=True)
        run(
            [sys.executable, str(REPO / "scripts/eval/run_save_inference_eval.py"),
             "--template_fp", args.template_fp, "--custom_fp", args.custom_fp,
             "--model_ckpt", args.model_ckpt, "--eval_split", "test",
             "--save_dp", str(sd), "--eval_batch_size", str(args.eval_batch_size),
             "--min_peak_int", "-1", "--output_subset", "pred_mzs", "pred_ints",
             "--auxiliary_scores", *args.auxiliary_scores,
             "--override", f"spec_fp={sd / 'spec_df.pkl'}", f"mol_fp={sd / 'mol_df.pkl'}",
             f"split_dp={sd / 'split'}", "eval_bin_sqrt=True"],
            out_dp / "infer.log",
        )  # fmt: skip
        if not pred_fp.is_file() or pred_fp.stat().st_size == 0:
            raise RuntimeError(f"inference wrote no predictions: {pred_fp}")
        (out_dp / "inferred").touch()

    # 2. Per-candidate retrieval scores against the benchmark query spectra.
    if "score" in args.stages and not (out_dp / "scored").exists():
        if not (out_dp / "inferred").exists():
            raise RuntimeError(f"cannot score {sd}: inference has not finished")
        run(
            [sys.executable, str(REPO / "scripts/eval/score_retrieval_candidates.py"),
             "--candidates", str(sd / "candidates.csv"), "--query_spec_df", args.query_spec_fp,
             "--pred_format", "masster", "--pred_fp", str(pred_fp), "--scores", *args.scores,
             "--inmodel_scores", *args.inmodel_scores, "--out", str(out_dp / "scores.pkl")],
            out_dp / "score.log",
        )  # fmt: skip
        (out_dp / "scored").touch()
        if not args.keep_predictions:
            pred_fp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--template_fp", default="config")
    parser.add_argument("--custom_fp", required=True, help="the checkpoint's training config")
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--tsv_fp", required=True, help="MassSpecGym1.5.tsv")
    parser.add_argument("--candidates_fp", required=True, help="retrieval candidates JSON")
    parser.add_argument("--fold", required=True, choices=["val", "test"])
    parser.add_argument("--run_dp", required=True)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--shards", nargs="*", type=int, default=None, help="default: all")
    parser.add_argument("--queries_per_shard", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument(
        "--auxiliary_scores", nargs="+", default=["cos_sim", "cos_hun"]
    )
    parser.add_argument(
        "--inmodel_scores", nargs="+", default=["cos_sim_0.01", "cos_sim_sqrt_0.01", "cos_hun"]
    )
    parser.add_argument("--scores", nargs="+", default=["loglik", "loglik_cal", "mspred_ent"])
    parser.add_argument(
        "--keep_predictions",
        type=booltype,
        default=True,
        help="false deletes each shard's HDF5 once scored (~1 GB per 500 formula queries)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Subprocesses run from the repository root, so every path is made absolute here.
    for k in ("template_fp", "custom_fp", "model_ckpt", "tsv_fp", "candidates_fp", "run_dp"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    run_dp = Path(args.run_dp)
    run_dp.mkdir(parents=True, exist_ok=True)
    label = Path(args.model_ckpt).stem  # the output dir run_save_inference_eval.py writes
    os.chdir(REPO)  # load_config and the config's relative data paths are repo-relative
    config_d = load_config(args.template_fp, args.custom_fp, [])
    args.query_spec_fp = os.path.abspath(config_d["spec_fp"])
    logger.info("run %s, label %s, stages %s", run_dp, label, args.stages)

    # 1. Shards.
    if "prepare" in args.stages:
        prepare(args, config_d, run_dp)

    # 2. Inference and scoring, shard by shard.
    shard_dps = sorted(run_dp.glob("shard_*"))
    if args.shards is not None:
        shard_dps = [sd for sd in shard_dps if int(sd.name.split("_")[1]) in args.shards]
    if {"infer", "score"} & set(args.stages):
        if not shard_dps:
            raise FileNotFoundError(f"no shards under {run_dp}; run the prepare stage first")
        for i, sd in enumerate(shard_dps):
            logger.info("shard %d/%d: %s", i + 1, len(shard_dps), sd.name)
            infer_and_score(args, sd, label)

    # 3. hit@k with molecule-bootstrap intervals, overall and per instrument.
    if "summarize" in args.stages:
        missing = [sd.name for sd in sorted(run_dp.glob("shard_*"))
                   if not (sd / label / "scored").exists()]  # fmt: skip
        if missing:
            raise RuntimeError(f"{len(missing)} shards are not scored yet: {missing[:5]}")
        run(
            [sys.executable, str(REPO / "scripts/eval/summarize_retrieval.py"),
             "--scores", f"{label}={run_dp}/shard_*/{label}/scores.pkl",
             "--queries", str(run_dp / "queries.csv"), "--query_id_col", "spec_id",
             "--mol_col", "truth_inchikey_s", "--subset_col", "inst_type",
             "--out", str(run_dp / f"summary_{label}.csv")],
            run_dp / f"summarize_{label}.log",
        )  # fmt: skip
        logger.info("summary: %s", run_dp / f"summary_{label}.csv")


if __name__ == "__main__":
    main()
