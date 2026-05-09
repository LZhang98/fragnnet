"""Upload local checkpoint file(s) to an existing wandb run.

Useful for crashed runs where checkpoints were saved locally but not synced.

Usage:
    # Upload a single checkpoint file
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run <run_id> --ckpt path/to/model-epoch=010.ckpt

    # Upload all .ckpt files in a directory
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run <run_id> --ckpt-dir path/to/ckpt/

    # Upload from the standard wandb run dir (auto-discovers ckpt/ subdirectory)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run <run_id> --wandb-run-dir wandb/run-20260421_120318-omp634h1/files

    # Specify entity/project explicitly (default: frag-gnn/frag-gnn)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run <run_id> --ckpt model.ckpt --entity myorg --project myproj
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def collect_ckpt_files(args: argparse.Namespace) -> list[Path]:
    """Collect checkpoint paths from the various input options."""
    paths: list[Path] = []

    if args.ckpt:
        for fp in args.ckpt:
            p = Path(fp)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            if not p.suffix == ".ckpt":
                log.warning("File does not have .ckpt extension: %s", p)
            paths.append(p)

    if args.ckpt_dir:
        d = Path(args.ckpt_dir)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        found = sorted(d.glob("*.ckpt"))
        if not found:
            raise FileNotFoundError(f"No .ckpt files found in: {d}")
        paths.extend(found)

    if args.wandb_run_dir:
        run_dir = Path(args.wandb_run_dir)
        # Try <run_dir>/ckpt/ first, then <run_dir>/ directly
        ckpt_subdir = run_dir / "ckpt"
        search_dir = ckpt_subdir if ckpt_subdir.is_dir() else run_dir
        found = sorted(search_dir.glob("*.ckpt"))
        if not found:
            raise FileNotFoundError(f"No .ckpt files found under: {search_dir}")
        paths.extend(found)

    # deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="WandB run ID (short ID like 'abc123de')")
    parser.add_argument("--entity", default="frag-gnn", help="WandB entity (default: frag-gnn)")
    parser.add_argument("--project", default="frag-gnn", help="WandB project (default: frag-gnn)")
    parser.add_argument(
        "--ckpt",
        action="append",
        metavar="FILE",
        help="Path to a .ckpt file (repeatable)",
    )
    parser.add_argument(
        "--ckpt-dir",
        metavar="DIR",
        help="Directory containing .ckpt files to upload",
    )
    parser.add_argument(
        "--wandb-run-dir",
        metavar="DIR",
        help="Local wandb run directory (e.g. wandb/run-<timestamp>-<id>/files); "
        "the script looks for a ckpt/ subdirectory first",
    )
    parser.add_argument(
        "--root",
        default="ckpt",
        help="Remote directory prefix inside the run (default: 'ckpt')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args(argv)

    if not args.ckpt and not args.ckpt_dir and not args.wandb_run_dir:
        parser.error("Provide at least one of --ckpt, --ckpt-dir, or --wandb-run-dir")

    try:
        ckpt_files = collect_ckpt_files(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        log.error("%s", exc)
        return 1

    if not ckpt_files:
        log.error("No checkpoint files resolved.")
        return 1

    log.info("Checkpoints to upload (%d):", len(ckpt_files))
    for p in ckpt_files:
        log.info("  %s  (%s)", p, _human_size(p.stat().st_size))

    run_path = f"{args.entity}/{args.project}/{args.run}"
    if args.dry_run:
        log.info("[dry-run] Would upload to run: %s", run_path)
        log.info("[dry-run] Remote root: %s", args.root)
        return 0

    import wandb

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as exc:
        log.error("Could not fetch run %s: %s", run_path, exc)
        return 1

    log.info("Uploading to run: %s (state: %s)", run.name, run.state)

    n_ok = 0
    n_fail = 0
    for p in ckpt_files:
        remote_path = f"{args.root}/{p.name}"
        log.info("  Uploading %s -> %s ...", p.name, remote_path)
        try:
            run.upload_file(str(p), root=str(p.parent))
            log.info("    OK")
            n_ok += 1
        except Exception as exc:
            log.error("    FAILED: %s", exc)
            n_fail += 1

    log.info("Done: %d uploaded, %d failed", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


if __name__ == "__main__":
    raise SystemExit(main())
