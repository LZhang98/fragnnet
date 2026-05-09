"""
Minimal WandB exporter.
Writes full history CSV and a human-readable TXT with sparklines.
Call `export_wandb_history(out_dir=...)` from a shell trap or external handler.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import wandb

SPARK_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def _sparkline(values: Sequence[float] | None, width: int | None = 200) -> str:
    if values is None:
        return ""
    vals = [float(v) for v in values if v is not None]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) == 0:
        return ""
    n = len(vals)
    if width and n > width:
        step = n / width
        idxs = [min(n - 1, int(i * step)) for i in range(width)]
        vals = [vals[i] for i in idxs]
        n = len(vals)
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        return SPARK_CHARS[0] * n
    chars = []
    levels = len(SPARK_CHARS) - 1
    for v in vals:
        norm = (v - vmin) / (vmax - vmin)
        idx = int(round(norm * levels))
        idx = max(0, min(levels, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def export_wandb_history(
    run_path: str | None = None, out_dir: str | None = None, spark_width: int = 200
) -> str:
    api = wandb.Api()
    if run_path is None:
        if wandb.run is None:
            raise RuntimeError("No active wandb.run and no run_path provided")
        run_path = wandb.run.path
    run = api.run(run_path)

    # Always request history as list of records (avoid DataFrame handling).
    try:
        rows = list(run.history())
    except Exception:
        rows = []

    # If the iterator produced no rows, some wandb runtimes (or tests/mocks)
    # require calling history(pandas=True) to get a DataFrame-like object.
    if not rows:
        try:
            hist = run.history(pandas=True)
            if hasattr(hist, "to_dict"):
                try:
                    rows = hist.to_dict(orient="records")
                except Exception:
                    rows = list(hist)
            elif isinstance(hist, list):
                rows = hist
            else:
                rows = list(hist)
        except Exception:
            # keep rows as-is (empty)
            pass

    # Determine CSV columns in order of first appearance
    columns: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k not in columns:
                columns.append(k)

    out_dir = "." if out_dir is None else out_dir
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    run_id = run.name or run.path.split("/")[-1]
    csv_path = out_dir_path / f"wandb_run_{run_id}_history.csv"
    txt_path = out_dir_path / f"wandb_run_{run_id}_history.txt"

    tmp_csv = str(csv_path) + ".tmp"
    try:
        with open(tmp_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                if not isinstance(r, dict):
                    # write a single-value row under a synthetic column
                    writer.writerow({columns[0] if columns else "value": str(r)})
                    continue
                row_out = {}
                for k in columns:
                    v = r.get(k)
                    if v is None:
                        row_out[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        row_out[k] = str(v)
                    else:
                        try:
                            row_out[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            row_out[k] = str(v)
                writer.writerow(row_out)
        os.replace(tmp_csv, str(csv_path))
    except Exception:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    with open(txt_path, "w", encoding="utf-8") as f:

        def w(s: str = ""):
            f.write("wandb: " + s + "\n")

        w("Run history:")
        for col in columns:
            # collect numeric series and non-numeric samples
            nums: list[float] = []
            samples: list[str] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                v = r.get(col)
                if v is None:
                    continue
                # attempt numeric conversion
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        nums.append(fv)
                        continue
                except Exception:
                    pass
                # fallback sample
                try:
                    samples.append(
                        json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                    )
                except Exception:
                    samples.append(str(v))

            if len(nums) > 0:
                spark = _sparkline(nums, width=spark_width)
                w(f"{col:>30} {spark}")
            else:
                uniq = []
                seen = set()
                for s in samples:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                    if len(uniq) > 51:
                        break
                if len(uniq) == 0:
                    continue
                if len(uniq) > 50:
                    w(f"{col:>30} <{len(uniq)} unique values; sample: {uniq[:3]} ...>")
                else:
                    w(f"{col:>30} {uniq}")

        w("")
        w("Run summary:")
        summary = getattr(run, "summary", {}) or {}
        for k, v in sorted(summary.items(), key=lambda x: x[0]):
            w(f"{k:>30} {v}")

    return str(txt_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export wandb run history and a small human-readable summary"
    )
    parser.add_argument("--out-dir", default="./wandb", help="Output directory (default: ./wandb)")
    parser.add_argument(
        "--run", action="append", help="Specific run path(s) to export (owner/proj/run)"
    )
    parser.add_argument("--project", help="Project path to export all runs from (owner/proj)")
    parser.add_argument("--spark-width", type=int, default=200, help="Sparkline width")

    args = parser.parse_args(argv)

    api = wandb.Api()

    if args.project:
        try:
            runs = list(api.runs(args.project))
        except Exception as e:
            print(f"Error listing runs for project {args.project}: {e}", file=sys.stderr)
            return 2
        for run in runs:
            out_dir = Path(args.out_dir) / run.path.replace("/", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                print(f"Exporting {run.path} -> {out_dir}")
                export_wandb_history(
                    run_path=run.path, out_dir=str(out_dir), spark_width=args.spark_width
                )
            except Exception as e:
                print(f"Failed to export {run.path}: {e}", file=sys.stderr)
        return 0

    if args.run:
        for rp in args.run:
            out_dir = Path(args.out_dir) / rp.replace("/", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Exporting {rp} -> {out_dir}")
            try:
                export_wandb_history(
                    run_path=rp, out_dir=str(out_dir), spark_width=args.spark_width
                )
            except Exception as e:
                print(f"Failed to export {rp}: {e}", file=sys.stderr)
        return 0

    # Default: if running inside a wandb run, export that one; otherwise error
    if wandb.run is not None:
        print(f"Exporting active run {wandb.run.path} -> {args.out_dir}")
        export_wandb_history(
            run_path=wandb.run.path, out_dir=args.out_dir, spark_width=args.spark_width
        )
        return 0

    print("No --project or --run given and no active wandb.run detected.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


def scan_local_runs(out_dir: str | Path = "./wandb") -> list[dict]:
    """Scan a local export directory for wandb run exports.

    Looks for files named like `wandb_run_<id>_history.csv` or
    `wandb_run_<id>_history.txt` and returns a list of records with keys:
    - "id": the run id (filename-derived)
    - "dir": Path to the directory containing the files
    - "csv": Path to csv file or None
    - "txt": Path to txt file or None

    Default `out_dir` is `./wandb` to match the CLI default.
    """
    out_p = Path(out_dir)
    runs = {}
    if not out_p.exists():
        return []

    for csv_path in out_p.rglob("wandb_run_*_history.csv"):
        stem = csv_path.stem
        # stem: wandb_run_<id>_history
        if stem.startswith("wandb_run_") and stem.endswith("_history"):
            run_id = stem[len("wandb_run_") : -len("_history")]
        else:
            run_id = stem
        rec = runs.setdefault(
            run_id, {"id": run_id, "dir": csv_path.parent, "csv": None, "txt": None}
        )
        rec["csv"] = csv_path

    for txt_path in out_p.rglob("wandb_run_*_history.txt"):
        stem = txt_path.stem
        if stem.startswith("wandb_run_") and stem.endswith("_history"):
            run_id = stem[len("wandb_run_") : -len("_history")]
        else:
            run_id = stem
        rec = runs.setdefault(
            run_id, {"id": run_id, "dir": txt_path.parent, "csv": None, "txt": None}
        )
        rec["txt"] = txt_path

    # also consider directories directly under out_p that look like earlier CLI layout
    for sub in out_p.iterdir():
        if not sub.is_dir():
            continue
        # look for txt/csv files in this subdir
        csvf = next(sub.glob("wandb_run_*_history.csv"), None)
        txtf = next(sub.glob("wandb_run_*_history.txt"), None)
        if csvf or txtf:
            # derive id from dirname if not present
            run_id = sub.name
            rec = runs.setdefault(run_id, {"id": run_id, "dir": sub, "csv": None, "txt": None})
            if csvf:
                rec["csv"] = csvf
            if txtf:
                rec["txt"] = txtf

    return list(runs.values())
