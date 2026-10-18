"""Compare peak-list predictions on a split with the model's own eval metrics, and tabulate.

Stages:

``score``
    Score one prediction file against the split's targets with a MASSter checkpoint's
    ``metric_fn``: targets from ``model.preproc_spec`` (pred=False), predictions L1-normalized,
    the eval's tolerance table and flags. Metrics: ``cos_sim_0.01``, ``cos_hun``,
    ``cos_hun_inst_tol``, each linear and sqrt. Two sources:

    - ``--export``: a ``run_save_inference_eval.py`` HDF5. ``--min_peak_int`` re-filters its peaks
      on the exporter's 0-1000 scale (strict ``>``, as the exporter filters). With no filter, the
      offline scores are checked against the export's logged in-model scores.
    - ``--glacier``: an ms-pred ``predict_smis_joint.py`` PredSpecDB, keyed ``pred_<identifier>``.
      GLACIER predicts sqrt(intensity) (ms-pred trains it on sqrt(inten / max)), so its output is
      squared.

    Writes a per-spectrum pickle with ``spec_id``, ``mol_id``, ``inst_type``, ``ace`` and the metrics.

``table``
    Wide CSV: one row per metric, each model's mean, and each model's paired difference from
    ``--reference`` with a 95% molecule-bootstrap CI (all of a molecule's spectra resampled
    together, since they are not independent).

Example (MSGYM test, see
docs/findings/masster/masster_glacier_released_ckpt.md):

    python scripts/eval/compare_msgym_predictions.py score --config <cfg> --ckpt <ckpt> \
        --split test --export <predictions_test.hdf5> --out ours.pkl
    python scripts/eval/compare_msgym_predictions.py score ... --glacier <preds.hdf5> --out glacier.pkl
    python scripts/eval/compare_msgym_predictions.py table --scores "ours=ours.pkl" \
        "GLACIER=glacier.pkl" --reference GLACIER --out table.csv
"""

import argparse

import numpy as np
import pandas as pd

METRICS = [
    "cos_sim_0.01",
    "cos_hun_inst_tol",
    "cos_sim_sqrt_0.01",
    "cos_hun_inst_tol_sqrt",
    "cos_hun",
    "cos_hun_sqrt",
]
N_BOOT = 5000


def load_export_peaks(path: str, min_peak_int: float) -> tuple[dict, dict]:
    """Peaks and logged scores from a run_save_inference_eval.py HDF5, keyed by spec_id.

    Args:
        path: Export HDF5.
        min_peak_int: Keep peaks with intensity strictly above this (0-1000 scale); 0 keeps all.

    Returns:
        (spec_id -> array ``(n, 2)`` of m/z and intensity, metric -> {spec_id: logged score}).
    """
    import h5py

    with h5py.File(path, "r") as f:
        sid = f["spec_id"][:]
        st, en = f["start_points"][:], f["end_points"][:]
        mz, it = f["pred_mzs"][:], f["pred_ints"][:].astype(np.float64)
        logged = {m: dict(zip(sid.tolist(), f[m][:])) for m in METRICS if m in f}
    peaks = {}
    for s, a, b in zip(sid.tolist(), st, en):
        pk = np.stack([mz[a:b], it[a:b]], 1)
        peaks[s] = pk[pk[:, 1] > min_peak_int] if min_peak_int > 0 else pk
    return peaks, logged


def load_glacier_peaks(path: str, ident_to_sid: dict) -> dict:
    """GLACIER peaks keyed by spec_id, squared from sqrt to linear intensity.

    Args:
        path: ms-pred PredSpecDB HDF5 (``pred_<identifier>/ikey .../collision <ce>/f``).
        ident_to_sid: Dataset identifier (``dset_spec_id``) -> spec_id.

    Returns:
        spec_id -> array ``(n, 2)`` of m/z and linear intensity.
    """
    import h5py

    peaks = {}
    with h5py.File(path, "r") as f:
        for key in f:
            if not key.startswith("pred_"):
                continue
            ikeys = list(f[key].values())
            ces = list(ikeys[0].values())
            if len(ikeys) != 1 or len(ces) != 1:
                raise ValueError(f"{key}: expected one ikey and one collision energy")
            a = ces[0]["f"][:].astype(np.float64)
            a = a[a[:, 1] > 0]
            peaks[int(ident_to_sid[key[len("pred_"):]])] = np.stack([a[:, 0], a[:, 1] ** 2], 1)
    return peaks


def run_score(args: argparse.Namespace) -> None:
    import torch as th
    from pandarallel import pandarallel

    from fragnnet.inference.checkpoint import (
        apply_inference_config,
        init_model_from_ckpt,
        setup_device,
    )
    from fragnnet.runner import init_dataloader, load_config
    from fragnnet.utils.misc_utils import to_device
    from fragnnet.utils.scatter_utils import safelog
    from fragnnet.utils.setup_utils import get_dataset_cls

    pandarallel.initialize(progress_bar=False, nb_workers=6)
    # 1. Config, dataset and model (for preproc_spec and metric_fn), as the eval builds them.
    config_d = load_config("config", args.config, ["eval_bin_sqrt=True"])
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=["cos_sim", "cos_hun", "cos_hun_inst_tol"],
        skip_extra_losses=False,
        disable_preproc=True,
        output_formula_str=False,
        train_batch_size=4,
        eval_batch_size=args.batch_size,
        dynamic_batch_sampler=False,
    )
    config_d["eval_hun_sqrt"] = True
    ds = get_dataset_cls(config_d["model_type"])(split=args.split, **config_d)
    dl = init_dataloader(ds, config_d)
    device = setup_device(config_d)
    model = init_model_from_ckpt(args.ckpt, config_d, device)
    model.eval()
    if model.extra_metric_args:
        raise ValueError(f"unsupported extra metric args: {model.extra_metric_args}")

    # 2. Predictions keyed by spec_id.
    logged = {}
    if args.export:
        peaks, logged = load_export_peaks(args.export, args.min_peak_int)
    else:
        ident_to_sid = dict(zip(ds.spec_df["dset_spec_id"], ds.spec_df["spec_id"]))
        peaks = load_glacier_peaks(args.glacier, ident_to_sid)
    missing = set(ds.spec_df["spec_id"].tolist()) - set(peaks)
    if missing:
        raise ValueError(f"{len(missing)} spectra of split {args.split} have no prediction")

    # 3. Score batch by batch.
    rows = []
    for batch in dl:
        batch = to_device(batch, device)
        bsz = batch["batch_size"]
        with th.inference_mode():
            t_mz, t_int, t_b = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                log_in=False,
                log_out=False,
                batch_size=bsz,
            )
        ids = batch["spec_unique_id"].cpu().numpy().tolist()
        pk = [peaks[s] for s in ids]
        pm = th.as_tensor(np.concatenate([p[:, 0] for p in pk]), dtype=th.float64, device=device)
        pi = th.as_tensor(
            np.concatenate([p[:, 1] / p[:, 1].sum() for p in pk]), dtype=t_int.dtype, device=device
        )
        pb = th.as_tensor(
            np.concatenate([np.full(len(p), b) for b, p in enumerate(pk)]), dtype=th.long, device=device
        )
        inp = {
            "true_mzs": t_mz,
            "true_ints": t_int,
            "true_logprobs": safelog(t_int, eps=model.hparams.log_min),
            "true_batch_idxs": t_b,
            "pred_mzs": pm,
            "pred_ints": pi,
            "pred_logprobs": safelog(pi, eps=model.hparams.log_min),
            "pred_batch_idxs": pb,
            "true_prec_mzs": batch["spec_prec_mz"],
            "split": "test",
        }
        for arg in model.batch_metric_args:
            if arg in batch:
                inp[arg] = batch[arg]
        with th.inference_mode():
            out = model.metric_fn(**inp)
        vals = {m: out[m].detach().cpu().numpy() for m in METRICS}
        for b, s in enumerate(ids):
            rows.append({"spec_id": s, "n_pred_peaks": len(pk[b]), **{m: float(vals[m][b]) for m in METRICS}})
    df = pd.DataFrame(rows)

    # 4. Reproduction check against the export's in-model scores. Those are computed before the
    # exporter's own --min_peak_int filter, so they only match an export written with 0.
    if args.export and args.min_peak_int <= 0:
        print("[check] logged scores predate the exporter's peak filter: expect a match only for "
              "an export written with --min_peak_int 0")
        for m, ref in logged.items():
            d = (df[m] - df["spec_id"].map(ref)).abs()
            print(f"[check] {m}: max |offline - in-model| {d.max():.2e}, n>1e-4 {(d > 1e-4).sum()}")

    meta = ds.spec_df[["spec_id", "mol_id", "inst_type", "ace"]].copy()
    meta["ace"] = meta["ace"].map(lambda v: float(np.asarray(v, dtype=float).reshape(-1)[0]))
    df = df.merge(meta, on="spec_id", how="left", validate="one_to_one")
    df.to_pickle(args.out)
    print(f"n={len(df)} | " + " | ".join(f"{m} {df[m].mean():.4f}" for m in METRICS))


def paired_diff_ci(
    diff: pd.Series, mol: pd.Series, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """Mean paired difference and its 95% molecule-bootstrap CI.

    Args:
        diff: Per-spectrum difference, shape ``(n,)``.
        mol: Molecule id per spectrum, aligned with ``diff``.
        rng: Random generator.
        n_boot: Bootstrap resamples of molecules.

    Returns:
        (mean, 2.5th percentile, 97.5th percentile).
    """
    g = pd.DataFrame({"d": diff.to_numpy(), "mol": mol.to_numpy()}).groupby("mol")["d"].agg(["sum", "count"])
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_table(scores: dict[str, pd.DataFrame], reference: str, seed: int = 0) -> pd.DataFrame:
    """One row per metric: model means, and each model's difference from the reference.

    Args:
        scores: Model name -> per-spectrum frame with ``spec_id``, ``mol_id`` and the metrics.
        reference: Name of the model differences are taken against.
        seed: Bootstrap seed.

    Returns:
        Wide table with ``metric``, ``n``, ``<model>`` means and, for every other model,
        ``<model> - <reference>`` with ``ci95_lo`` / ``ci95_hi``.
    """
    rng = np.random.default_rng(seed)
    ref = scores[reference].set_index("spec_id")
    rows = []
    for m in METRICS:
        row = {"metric": m, "n": len(ref)}
        for name, df in scores.items():
            row[name] = df[m].mean()
        for name, df in scores.items():
            if name == reference:
                continue
            d = df.set_index("spec_id")
            if set(d.index) != set(ref.index):
                raise ValueError(f"{name} and {reference} cover different spectra")
            diff = d[m] - ref[m].reindex(d.index)
            mean, lo, hi = paired_diff_ci(diff, d["mol_id"], rng)
            col = f"{name} - {reference}"
            row.update({col: mean, f"{col} ci95_lo": lo, f"{col} ci95_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def run_table(args: argparse.Namespace) -> None:
    scores = {}
    for item in args.scores:
        name, path = item.split("=", 1)
        scores[name] = pd.read_pickle(path)
    if args.reference not in scores:
        raise ValueError(f"reference {args.reference!r} not among {list(scores)}")
    table = build_table(scores, args.reference)
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(table.round(4).to_string(index=False))


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("score", help="score one prediction file with the model's metric_fn")
    s.add_argument("--config", required=True, help="experiment config the targets are built with")
    s.add_argument("--ckpt", required=True, help="MASSter checkpoint providing metric_fn")
    s.add_argument("--split", default="test")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--export", help="run_save_inference_eval.py HDF5")
    src.add_argument("--glacier", help="ms-pred PredSpecDB HDF5 (sqrt intensities)")
    s.add_argument("--min_peak_int", type=float, default=0.0, help="export only, 0-1000 scale")
    s.add_argument("--batch_size", type=int, default=64)
    s.add_argument("--out", required=True, help="per-spectrum pickle")
    t = sub.add_parser("table", help="wide CSV of means and paired differences")
    t.add_argument("--scores", nargs="+", required=True, help="name=path.pkl, one per model")
    t.add_argument("--reference", required=True)
    t.add_argument("--out", required=True, help="CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.stage == "score":
        if args.glacier and args.min_peak_int > 0:
            raise ValueError("--min_peak_int applies to --export only")
        run_score(args)
    else:
        run_table(args)
