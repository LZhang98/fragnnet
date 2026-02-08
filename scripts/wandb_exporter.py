"""
Minimal WandB exporter.
Writes full history CSV and a human-readable TXT with sparklines.
Call `export_wandb_history(out_dir=...)` from a shell trap or external handler.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import wandb

SPARK_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def _sparkline(values: np.ndarray, width: int | None = 200) -> str:
    if values is None or len(values) == 0:
        return ""
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return ""
    if width and vals.size > width:
        idx = (np.linspace(0, vals.size - 1, width)).astype(int)
        vals = vals[idx]
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        return SPARK_CHARS[0] * len(vals)
    norm = (vals - vmin) / (vmax - vmin)
    idx = (norm * (len(SPARK_CHARS) - 1)).round().astype(int)
    return "".join(SPARK_CHARS[i] for i in idx)


def export_wandb_history(
    run_path: str | None = None, out_dir: str | None = None, spark_width: int = 200
) -> str:
    api = wandb.Api()
    if run_path is None:
        if wandb.run is None:
            raise RuntimeError("No active wandb.run and no run_path provided")
        run_path = wandb.run.path
    run = api.run(run_path)

    try:
        df = run.history(pandas=True)
    except Exception:
        rows = list(run.history())
        df = pd.DataFrame(rows)

    out_dir = "." if out_dir is None else out_dir
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    run_id = run.name or run.path.split("/")[-1]
    csv_path = out_dir_path / f"wandb_run_{run_id}_history.csv"
    txt_path = out_dir_path / f"wandb_run_{run_id}_history.txt"

    tmp_csv = str(csv_path) + ".tmp"
    try:
        df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, str(csv_path))
    except Exception:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    with open(txt_path, "w", encoding="utf-8") as f:

        def w(s: str = ""):
            f.write("wandb: " + s + "\n")

        w("Run history:")
        for col in df.columns:
            try:
                series = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            except Exception:
                series = np.array([], dtype=float)
            if series.size > 0:
                spark = _sparkline(series, width=spark_width)
                w(f"{col:>30} {spark}")
            else:
                vals = df[col].dropna().unique()
                if len(vals) == 0:
                    continue
                if len(vals) > 50:
                    w(f"{col:>30} <{len(vals)} unique values; sample: {list(vals[:3])} ...>")
                else:
                    w(f"{col:>30} {list(vals)}")

        w("")
        w("Run summary:")
        summary = getattr(run, "summary", {}) or {}
        for k, v in sorted(summary.items(), key=lambda x: x[0]):
            w(f"{k:>30} {v}")

    return str(txt_path)
