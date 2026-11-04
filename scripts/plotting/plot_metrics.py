"""Plot train and val metric curves from a checkpoint folder's metrics.csv.

The training runner writes ``metrics.csv`` (and ``hparams.yaml``) beside the
checkpoints via Lightning's ``CSVLogger``. The metrics to plot come from a fixed list
(``DEFAULT_AUXILIARY_SCORES``). Each metric is logged per epoch under
``train_<metric>_epoch/mean`` and ``val_<metric>_epoch/mean`` columns; a metric
may appear with extra suffixes (such as ``cos_sim_0.01`` for a mass-tolerance
variant), so each configured name is matched as a substring of the column.

Each metric has its own plotting function; the composer draws one subplot per metric that has
both a train and a val column in ``metrics.csv`` and saves a single figure to
``figs/metrics/<model_name>.png``.

Usage:
    python scripts/plotting/plot_metrics.py -c ckpts/<job_id>

Args:
    ckpt_dp (str): Checkpoint folder containing ``metrics.csv``.
    out_dp (str): Directory to write the figure; model-name is the checkpoint folder's name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

TRAIN_PREFIX = "train_"
VAL_PREFIX = "val_"
DPI = 300
DEFAULT_AUXILIARY_SCORES = [
    "cos_sim",
    "recall",
    "precision",
    "wrecall",
    "wprecision",
    "cos_hun",
    "jss",
    "jss_hun",
]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#c9c8c4",
        "axes.linewidth": 0.8,
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)


def _find_metric_csv(ckpt_dp: Path) -> Path:
    """Return the metrics.csv path, raising if missing."""
    path = ckpt_dp / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no metrics.csv found in {ckpt_dp}")
    return path


def _metric_columns(
    df: pd.DataFrame, metric: str, requested: set[str]
) -> tuple[str | None, str | None]:
    """Return (train_col, val_col) logging a metric's per-epoch curve.

    Columns have the shape ``<split>_<segment>_<stage>/mean`` where ``segment`` is
    the metric plus optional suffixes (e.g. ``cos_sim_0.01``) and ``stage`` is
    ``epoch`` (a curve) or ``best`` (a running best). A configured metric matches a
    column when its segment equals the metric or starts with ``<metric>_``; segments that are
    themselves a separately requested metric (e.g. ``jss_hun`` for ``jss``) are
    excluded. The ``epoch`` variant is preferred over ``best`` so curves are drawn.
    """
    train_cands: dict[str, str] = {}
    val_cands: dict[str, str] = {}
    for col in df.columns:
        prefix = ""
        remainder = ""
        if col.startswith(TRAIN_PREFIX):
            prefix, remainder = TRAIN_PREFIX, col[len(TRAIN_PREFIX) :]
        elif col.startswith(VAL_PREFIX):
            prefix, remainder = VAL_PREFIX, col[len(VAL_PREFIX) :]
        else:
            continue
        for stage in ("epoch", "best"):
            marker = f"_{stage}/mean"
            if not remainder.endswith(marker):
                continue
            segment = remainder[: -len(marker)]
            # Exclude segments that are another requested metric (e.g. jss_hun for jss).
            if segment in requested and segment != metric:
                continue
            if not (segment == metric or segment.startswith(metric + "_")):
                continue
            cands = train_cands if prefix == TRAIN_PREFIX else val_cands
            # Prefer the epoch curve; only fall back to best when no epoch exists.
            if stage == "epoch" or segment not in cands:
                cands[segment] = col

    def _prefer(cands: dict[str, str]) -> str | None:
        exact = next((c for seg, c in cands.items() if seg == metric), None)
        if exact is not None:
            return exact
        # Prefer the shortest matching segment (fewest modifier suffixes).
        return cands[min(cands, key=len)] if cands else None

    return _prefer(train_cands), _prefer(val_cands)


def plot_cos_sim(ax, x, y_train, y_val) -> None:
    """Plot train vs val cosine similarity curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("cos_sim")


def plot_recall(ax, x, y_train, y_val) -> None:
    """Plot train vs val recall curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("recall")


def plot_precision(ax, x, y_train, y_val) -> None:
    """Plot train vs val precision curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("precision")


def plot_wrecall(ax, x, y_train, y_val) -> None:
    """Plot train vs val weighted recall curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("wrecall")


def plot_wprecision(ax, x, y_train, y_val) -> None:
    """Plot train vs val weighted precision curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("wprecision")


def plot_cos_hun(ax, x, y_train, y_val) -> None:
    """Plot train vs val HUN cosine curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("cos_hun")


def plot_jss(ax, x, y_train, y_val) -> None:
    """Plot train vs val JSS curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("jss")


def plot_jss_hun(ax, x, y_train, y_val) -> None:
    """Plot train vs val HUN JSS curves on ``ax``."""
    ax.plot(x, y_train, color="#2a78d6", label="train")
    ax.plot(x, y_val, color="#008300", label="val")
    ax.set_ylabel("jss_hun")


_METRIC_PLOTTERS = {
    "cos_sim": plot_cos_sim,
    "recall": plot_recall,
    "precision": plot_precision,
    "wrecall": plot_wrecall,
    "wprecision": plot_wprecision,
    "cos_hun": plot_cos_hun,
    "jss": plot_jss,
    "jss_hun": plot_jss_hun,
}


def _dense_series(df: pd.DataFrame, col: str, epoch_col: str) -> tuple[list, list]:
    """Return (x, y) pairs for a metric column with NaN rows dropped.

    The CSV stores each per-epoch metric on exactly one row per epoch (train and val land
    on different rows of the same epoch, e.g. val on the first and train on the second),
    leaving NaN elsewhere. Dropping NaN rows yields a dense epoch->value table for every plot.
    """
    pairs = df[[epoch_col, col]].dropna()
    return pairs[epoch_col].tolist(), pairs[col].tolist()


def make_and_save_plot(
    df: pd.DataFrame, metrics: list[str], out_dp: Path, model_name: str
) -> tuple[Path, list[str], list[str]]:
    """Compose one subplot per available metric into a single figure and save it.

    Args:
        df: Loaded metrics.csv with train_*/val_* metric columns.
        metrics: Configured metric names; each is attempted and plotted when present.
        out_dp: Destination directory for the figure.
        model_name: Stem used in the output filename.

    Returns:
        Path of the saved figure, the plotted metric names, and the missing metric names.
    """
    found: list[tuple[str, str, str]] = []
    missing: list[str] = []
    requested = set(metrics)
    for metric in metrics:
        train_col, val_col = _metric_columns(df, metric, requested)
        if train_col is not None and val_col is not None:
            found.append((metric, train_col, val_col))
        else:
            missing.append(metric)
    if not found:
        raise ValueError(
            f"none of the configured metrics {metrics} have train/val columns in metrics.csv"
        )

    epoch_col = "epoch" if "epoch" in df else df.index.name or "row"
    n = len(found)
    fig, axes = plt.subplots(
        n, 1, figsize=(8, 3.4 * n), constrained_layout=True, squeeze=False
    )
    for ax, (metric, train_col, val_col) in zip(axes[:, 0], found):
        x_train, y_train = _dense_series(df, train_col, epoch_col)
        x_val, y_val = _dense_series(df, val_col, epoch_col)
        if not y_train or not y_val:
            raise ValueError(f"metric {metric} has no data rows in metrics.csv")
        ax.set_title(metric)
        _METRIC_PLOTTERS[metric](ax, x_train, y_train, y_val)
        ax.set_xlabel(epoch_col)
        ax.legend()

    fig.suptitle(f"{model_name} train/val metrics")
    out_dp = Path(out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    path = out_dp / f"{model_name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, [name for name, _, _ in found], missing


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--ckpt_dp", required=True, help="Checkpoint folder with metrics.csv"
    )
    parser.add_argument("--out_dp", default="figs/metrics", help="Output directory for figures")
    args = parser.parse_args()

    ckpt_dp = Path(args.ckpt_dp)
    out_dp = Path(args.out_dp)
    if not ckpt_dp.is_dir():
        raise NotADirectoryError(f"checkpoint folder not found: {ckpt_dp}")

    metrics = list(DEFAULT_AUXILIARY_SCORES)

    df = pd.read_csv(_find_metric_csv(ckpt_dp))
    if df.empty:
        raise ValueError(f"metrics.csv in {ckpt_dp} is empty")

    path, found, missing = make_and_save_plot(df, metrics, out_dp, ckpt_dp.name)
    print(f"saved {path}")
    print(f"plotted metrics: {found}")
    if missing:
        print(f"not found in metrics.csv: {missing}")


if __name__ == "__main__":
    main()
