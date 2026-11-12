"""Summarize training runs in a checkpoint directory into one table.

One row per config in ``--config_dir`` (run order taken from ``--order_from`` if given, else
alphabetical). For each run, reads ``<checkpoint_dir>/<config name>/metrics.csv`` and reports the
best train / val value of each metric over epochs with the epoch it occurred on, plus the single
test value (test is evaluated once, on the best checkpoint, after training), the epoch of the
saved best checkpoint and the last epoch trained.

A run is summarized whenever its ``metrics.csv`` exists (a run still training reports the epochs
so far, with NA test values until the final test evaluation is logged). A missing CSV means the
run has not started and gets NA in every data column.

Example::

    python scripts/eval/summarize_train.py
"""

import argparse
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# (label, metric name without split prefix/suffix, direction of "best")
METRICS = [
    ("cos_sim_0.01", "cos_sim_0.01_epoch/mean", "max"),
    ("cos_hun", "cos_hun_epoch/mean", "max"),
    ("jss_0.01", "jss_0.01_epoch/mean", "max"),
    ("jss_hun", "jss_hun_epoch/mean", "max"),
    ("loss", "loss_epoch/mean", "min"),
    ("primary_loss", "primary_loss_epoch/mean", "min"),
    ("spec_ce", "spec_ce_epoch/mean", "min"),
]


def parse_name(name: str) -> tuple[str, str, str]:
    """``fragnnet_d3_ms_gym_scaffold_s0`` -> (model, dataset, split)."""
    m = re.match(r"^(fragnnet_d3|masster)_(.+)_(inchikey|scaffold|predefined)_s\d+$", name)
    return m.groups() if m else (name, "", "")


def summarize_run(ckpt_dp: str) -> dict | None:
    """Return the data columns for a run with a metrics.csv, or None if there is none."""
    csv_fp = os.path.join(ckpt_dp, "metrics.csv")
    if not os.path.exists(csv_fp):
        return None
    df = pd.read_csv(csv_fp)
    test_cols = [c for c in df.columns if c.startswith("test_")]
    has_test = bool(test_cols) and df[test_cols].notna().any().any()

    out = {}
    ckpts = sorted(f for f in os.listdir(ckpt_dp) if f.startswith("model-epoch="))
    out["best_ckpt_epoch"] = int(re.search(r"epoch=(\d+)", ckpts[-1]).group(1)) if ckpts else None
    # the CSV's final "epoch" value belongs to the test row, so take it from the train rows
    train_rows = df[df["train_loss_epoch/mean"].notna()]
    out["last_epoch"] = int(train_rows["epoch"].max())

    test_row = df[df[test_cols].notna().any(axis=1)].iloc[-1] if has_test else None
    for split in ("train", "val"):
        for label, col, mode in METRICS:
            s = df[f"{split}_{col}"].dropna()
            idx = s.idxmax() if mode == "max" else s.idxmin()
            out[f"{split}_{label}"] = s[idx]
            out[f"{split}_{label}_ep"] = int(df.loc[idx, "epoch"])
    for label, col, _ in METRICS:
        out[f"test_{label}"] = test_row[f"test_{col}"] if has_test else None
    out["has_test"] = has_test
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_dir", default="config/luke_exp")
    p.add_argument("--checkpoint_dir", default="/mnt/data/masster_luke_exp_checkpoints")
    p.add_argument("--order_from", default="scripts/bash/train_luke_exp.sh", help="script whose run order to follow")
    p.add_argument("--out_dir", default="output/training_metrics")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    names = sorted(f[:-4] for f in os.listdir(args.config_dir) if f.endswith(".yml"))
    if args.order_from and os.path.exists(args.order_from):
        text = open(args.order_from).read()
        ordered = sorted(names, key=lambda n: m.start() if (m := re.search(rf"^\s*{re.escape(n)}\s*$", text, re.M)) else 1e9)
        names = ordered

    data_cols = ["best_ckpt_epoch", "last_epoch"]
    for label, _, _ in METRICS:
        data_cols += [f"train_{label}", f"train_{label}_ep", f"val_{label}", f"val_{label}_ep", f"test_{label}"]

    rows = []
    for i, name in enumerate(names, 1):
        model, dataset, split = parse_name(name)
        res = summarize_run(os.path.join(args.checkpoint_dir, name))
        status = "not started" if res is None else ("finished" if res.pop("has_test") else "training")
        rows.append({"#": i, "config": name, "model": model, "dataset": dataset, "split": split, "status": status, **(res or {c: None for c in data_cols})})
    table = pd.DataFrame(rows)[["#", "config", "model", "dataset", "split", "status"] + data_cols]

    for c in data_cols:
        if c.endswith("_ep") or c.endswith("epoch"):
            table[c] = table[c].astype("Int64")  # keep epochs integer despite NA rows

    os.makedirs(args.out_dir, exist_ok=True)
    csv_fp = os.path.join(args.out_dir, os.path.basename(os.path.normpath(args.checkpoint_dir)) + ".csv")
    table.to_csv(csv_fp, index=False, na_rep="NA")

    logger.info("Wrote %s", csv_fp)


if __name__ == "__main__":
    main()
