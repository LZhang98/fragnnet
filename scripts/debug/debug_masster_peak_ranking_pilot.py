"""Test peak-ranking supervision in a matched, intensity-head-only CPU continuation.

The encoder, proposal head, candidate support, and dropout stay frozen. Training targets
alone provide gradients; validation identities are split into tuning and confirmation.
This bounded diagnostic does not establish the effect of end-to-end training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from fragnnet.runner import load_config
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt
from fragnnet.utils.setup_utils import get_dataset_cls

try:
    from .debug_masster_cut_intensity_probe import identity, metric_cache, score_channels
except ImportError:
    from debug_masster_cut_intensity_probe import identity, metric_cache, score_channels

LOGGER = logging.getLogger(__name__)


def peak_ranking_loss(
    mzs: th.Tensor,
    logp: th.Tensor,
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    rel: float,
    min_mz: float,
    minimum_gap: float,
) -> tuple[th.Tensor, bool]:
    """Compare summed base-compatible probability with the strongest incompatible mass.

    Inputs are single-spectrum vectors. Channels with identical four-decimal masses
    are summed first. All masses compatible with any tied measured base peak are excluded
    from competitors. Uncovered or near-tied targets return a differentiable zero.
    The positive group sums all base-compatible masses, preserving structural ambiguity.
    """
    zero = logp.sum() * 0
    if not len(true_ints) or not len(logp):
        return zero, False
    pm, inverse = th.unique(th.round(mzs.detach(), decimals=4), return_inverse=True)
    # Stable summation retains gradients through every channel in a mass group.
    probabilities = logp.exp().new_zeros(len(pm)).scatter_add(0, inverse, logp.exp())
    tm, ti_idx = th.unique(th.round(true_mzs.detach(), decimals=4), return_inverse=True)
    ti = true_ints.new_zeros(len(tm)).scatter_add(0, ti_idx, true_ints)
    top = th.topk(ti, min(2, len(ti))).values
    gap = (top[0] - top[1]) / top[0] if len(top) == 2 else top[0].new_tensor(1.0)
    if float(gap) < minimum_gap:
        return zero, False
    base_mz = tm[ti.argmax()]
    compatible = (pm - base_mz).abs() <= rel * th.maximum(base_mz, base_mz.new_tensor(min_mz))
    if not compatible.any() or compatible.all():
        return zero, False
    tiny = th.finfo(logp.dtype).tiny
    positive = probabilities[compatible].sum().clamp_min(tiny).log()
    negative = probabilities[~compatible].max().clamp_min(tiny).log()
    return gap.detach() * F.softplus(negative - positive), True


def sample_specs(frame: pd.DataFrame, molecules: int, cap: int, seed: int) -> pd.DataFrame:
    """Sample full identities and conditions before examining target intensities."""
    rng = np.random.default_rng(seed)
    keys = np.sort(frame.inchikey_s.unique())
    chosen = rng.choice(keys, min(molecules, len(keys)), replace=False)
    indices = []
    for _, group in frame[frame.inchikey_s.isin(chosen)].groupby("inchikey_s", sort=True):
        indices.extend(rng.choice(group.index, min(cap, len(group)), replace=False))
    return frame.loc[indices].sort_values("spec_id").reset_index(drop=True)


def prepare_batch(model, batch: dict) -> tuple[dict, tuple]:
    """Match the ordinary training target/prediction preprocessing and tolerance lookup."""
    out = model(**batch)
    size = int(batch["batch_size"])
    tm, tl, tb = model.preproc_spec(
        batch["spec_mzs"],
        batch["spec_ints"],
        batch["spec_batch_idxs"],
        train=True,
        pred=False,
        log_out=True,
        batch_size=size,
    )
    pm, pl, pb = model.preproc_spec(
        out["pred_mzs"],
        out["pred_logprobs"],
        out["pred_batch_idxs"],
        train=True,
        pred=True,
        log_in=True,
        log_out=True,
    )
    inst = batch["spec_inst_type"].reshape(-1)
    rel = model.inst_tol_lookup[inst].to(tl)
    floor = model.inst_min_mz_lookup[inst].to(tl)
    values = dict(out)
    values.update(
        true_mzs=tm,
        true_logprobs=tl,
        true_intensities=tl.exp(),
        true_batch_idxs=tb,
        pred_mzs=pm,
        pred_logprobs=pl,
        pred_intensities=pl.exp(),
        pred_batch_idxs=pb,
        batch_size=size,
        tol_per_sample=rel,
        min_mz_per_sample=floor,
    )
    return values, (tm, tl.exp(), tb, pm, pl, pb, rel, floor)


def evaluate(model, loader, arm: str, partition: dict[int, str]) -> pd.DataFrame:
    """Score full norms with production-equivalent one-to-one mass matching."""
    rows = []
    with th.no_grad():
        for batch in loader:
            out = model(**batch)
            tm, ti, tb = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=int(batch["batch_size"]),
            )
            for i, sid in enumerate(batch["spec_unique_id"].tolist()):
                p = out["pred_batch_idxs"] == i
                t = tb == i
                inst = int(batch["spec_inst_type"].reshape(-1)[i])
                cache = metric_cache(
                    out["pred_mzs"][p].numpy(),
                    tm[t].numpy(),
                    ti[t].numpy(),
                    float(model.inst_tol_lookup[inst]),
                    float(model.inst_min_mz_lookup[inst]),
                )
                cosine, hit, rank = score_channels(out["pred_logprobs"][p].numpy(), cache)
                rows.append(
                    {
                        "spec_id": sid,
                        "arm": arm,
                        "partition": partition[sid],
                        "cosine": cosine,
                        "hit": hit,
                        "base_rank": rank,
                        "covered": bool(cache["base_mask"].any()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the predeclared matched arms and unlock confirmation after tuning selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis/masster_peak_ranking_pilot.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    th.set_num_threads(4)
    th.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002
    random.seed(cfg["seed"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # 1. Audit identities, then sample target-independent train/validation conditions.
    mols = pd.read_pickle("data/proc/nist20/mol_df.pkl")[["mol_id", "inchikey_s"]]
    frames = {}
    for split in ("train", "val"):
        frame = pd.read_pickle(cfg[f"{split}_spec_fp"])
        if "inchikey_s" not in frame:
            frame = frame.merge(mols, on="mol_id", validate="many_to_one")
        frame["inst_type"] = frame.inst_type.replace({"OB": "ORB", "OB_F": "ORB_F"})
        frames[split] = frame
    if set(frames["train"].inchikey_s) & set(frames["val"].inchikey_s):
        raise RuntimeError("train/validation molecular identity overlap")
    for split in frames:
        frames[split] = sample_specs(
            frames[split],
            cfg[f"{split}_molecules"],
            cfg["conditions_per_molecule"],
            cfg["seed"] + (split == "val"),
        )
    val_keys = np.sort(frames["val"].inchikey_s.unique())
    np.random.default_rng(cfg["seed"] + 2).shuffle(val_keys)
    tuning = set(val_keys[: len(val_keys) // 2])
    frames["val"]["partition"] = np.where(
        frames["val"].inchikey_s.isin(tuning), "tuning", "confirmation"
    )
    manifest = pd.concat([frames[s].assign(split=s) for s in frames])
    manifest[
        ["spec_id", "mol_id", "inchikey_s", "split", "partition", "nce", "inst_type"]
    ].to_pickle(output / "manifest.pkl")
    config = load_config("config", cfg["custom_fp"], [])
    config = apply_inference_config(config, disable_preproc=True, num_workers=0)
    config["masster_candidate_backend"] = "cpu"
    loaders = {}
    for split, frame in frames.items():
        path = output / f"spec_{split}.pkl"
        frame.to_pickle(path)
        dc = copy.deepcopy(config)
        dc["spec_fp"] = str(path)
        dataset = get_dataset_cls(config["model_type"])(split=split, **dc)
        if set(dataset.spec_df.spec_id) != set(frame.spec_id):
            raise RuntimeError("dataset changed sampled spectrum membership")
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=dataset.get_collate_fn(),
            num_workers=0,
        )
    provenance = {
        "config": cfg,
        "resolved_model_config": config,
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "inputs": [
            identity(cfg[k]) for k in ("checkpoint", "train_spec_fp", "val_spec_fp", "custom_fp")
        ],
        "source": identity(__file__),
        "analysis_config": identity(args.config),
        "torch": th.__version__,
        "device": "cpu",
        "training_scope": "intensity_head; eval mode; same ordered batches",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    partition = frames["val"].set_index("spec_id").partition.to_dict()
    train_partition = dict.fromkeys(frames["train"].spec_id, "train")
    scores, history = [], []
    # 2. Matched continuation arms: only the additional ranking weight differs.
    for weight in cfg["ranking_weights"]:
        th.manual_seed(cfg["seed"])
        model = init_model_from_ckpt(cfg["checkpoint"], copy.deepcopy(config), th.device("cpu"))
        model.requires_grad_(False)
        model.model.intensity_head.requires_grad_(True)
        model.eval()
        optimizer = th.optim.AdamW(
            model.model.intensity_head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        if weight == 0:
            scores.append(evaluate(model, loaders["val"], "baseline", partition))
            scores.append(evaluate(model, loaders["train"], "baseline", train_partition))
        for epoch in range(cfg["epochs"]):
            losses = []
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                values, arrays = prepare_batch(model, batch)
                primary = model.loss_fn(**values)["loss"].mean()
                tm, ti, tb, pm, pl, pb, rel, floor = arrays
                auxiliary, eligible = [], 0
                for i in range(int(batch["batch_size"])):
                    p, t = pb == i, tb == i
                    value, valid = peak_ranking_loss(
                        pm[p],
                        pl[p],
                        tm[t],
                        ti[t],
                        float(rel[i]),
                        float(floor[i]),
                        cfg["minimum_base_gap"],
                    )
                    auxiliary.append(value)
                    eligible += valid
                ranking = th.stack(auxiliary).mean()
                loss = primary + weight * ranking
                if not th.isfinite(loss):
                    raise RuntimeError("nonfinite pilot loss")
                loss.backward()
                norm = th.nn.utils.clip_grad_norm_(model.model.intensity_head.parameters(), 5.0)
                if not th.isfinite(norm):
                    raise RuntimeError("nonfinite pilot gradient")
                optimizer.step()
                losses.append((float(primary.detach()), float(ranking.detach()), eligible))
            record = {
                "weight": weight,
                "epoch": epoch,
                "primary": np.mean([v[0] for v in losses]),
                "ranking": np.mean([v[1] for v in losses]),
                "eligible": sum(v[2] for v in losses),
            }
            history.append(record)
            LOGGER.info("arm %s epoch %s: %s", weight, epoch, record)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        arm = str(weight)
        scores.append(evaluate(model, loaders["val"], arm, partition))
        scores.append(evaluate(model, loaders["train"], arm, train_partition))
        th.save(
            {
                "state_dict": model.model.intensity_head.state_dict(),
                "config": cfg,
                "ranking_weight": weight,
            },
            output / f"head_{arm}.pt",
        )
        pd.concat(scores).to_pickle(output / "scores.pkl")
        del model, optimizer
    # 3. Lock strength on tuning cosine, then report clustered confirmation differences.
    results = pd.concat(scores).merge(
        manifest[["spec_id", "inchikey_s", "nce", "inst_type"]],
        on="spec_id",
        validate="many_to_one",
    )
    results.to_pickle(output / "scores.pkl")
    means = results.groupby(["partition", "arm"])[["cosine", "hit", "covered"]].mean()
    tuning_means = means.loc["tuning", "cosine"].drop("baseline")
    selected = max(tuning_means.index, key=lambda a: (tuning_means[a], -float(a)))
    (output / "selection.json").write_text(
        json.dumps({"selected": selected, "tuning": tuning_means.to_dict()}, indent=2)
    )
    confirmation = results[results.partition == "confirmation"]
    summary = {
        "selected": selected,
        "means": means.reset_index().to_dict("records"),
        "comparisons": {},
    }
    for reference in ("baseline", "0.0"):
        for metric in ("cosine", "hit"):
            pivot = confirmation.pivot(index="spec_id", columns="arm", values=metric)
            delta = pivot[selected] - pivot[reference]
            identities = manifest.set_index("spec_id").inchikey_s.loc[delta.index]
            group = (
                pd.DataFrame({"delta": delta, "key": identities})
                .groupby("key")
                .delta.agg(["sum", "count"])
                .to_numpy()
            )
            rng = np.random.default_rng(cfg["seed"] + 3)
            bootstrap = []
            for _ in range(cfg["bootstrap_samples"]):
                sample = group[rng.integers(len(group), size=len(group))]
                bootstrap.append(sample[:, 0].sum() / sample[:, 1].sum())
            summary["comparisons"][f"{metric}_vs_{reference}"] = {
                "delta": float(delta.mean()),
                "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            }
    summary["elapsed_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    results.groupby(["partition", "arm", "inst_type"])[["cosine", "hit"]].mean().to_csv(
        output / "instrument_slices.csv"
    )
    results["nce_bin"] = pd.cut(results.nce, [-np.inf, 20, 40, 60, np.inf])
    results.groupby(["partition", "arm", "nce_bin"], observed=True)[
        ["cosine", "hit"]
    ].mean().to_csv(output / "energy_slices.csv")
    LOGGER.info("Complete: %s", summary["comparisons"])


if __name__ == "__main__":
    main()
