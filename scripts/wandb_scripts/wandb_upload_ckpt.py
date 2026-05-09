"""Upload local checkpoint file(s) to an existing wandb run.

Useful for crashed runs where checkpoints were saved locally but not synced.
By default, searches the local `wandb/` directory for the run matching the
given run ID, then uploads all checkpoints from its `ckpt/` subdirectory.

Usage:
    # Auto-discover ckpts from the local wandb/ run dir
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run-id omp634h1

    # Custom wandb directory (if not running from project root)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run-id omp634h1 --wandb-dir /path/to/wandb

    # Upload specific files instead
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run-id omp634h1 --ckpt path/to/model-epoch=010.ckpt

    # Upload all .ckpt files from a specific directory
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run-id omp634h1 --ckpt-dir path/to/ckpt/

    # Specify entity/project (default: frag-gnn/frag-gnn)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run-id omp634h1 --entity myorg --project myproj
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_local_run_dir(run_id: str, wandb_dir: Path) -> Path:
    """Find the local wandb run directory for a given run ID.

    Args:
        run_id: Short wandb run ID (e.g. 'omp634h1').
        wandb_dir: Root local wandb directory to search under.

    Returns:
        Path to the run's files directory (e.g. wandb/run-<ts>-<id>/files).

    Raises:
        FileNotFoundError: If no matching run directory is found.
    """
    if not wandb_dir.is_dir():
        raise FileNotFoundError(f"wandb directory not found: {wandb_dir}")

    matches = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.endswith(f"-{run_id}")]
    if not matches:
        raise FileNotFoundError(
            f"No local run directory found for run ID '{run_id}' under {wandb_dir}"
        )
    if len(matches) > 1:
        log.warning("Multiple run dirs matched '%s': %s — using first", run_id, matches)

    run_dir = matches[0]
    # WandB stores run files under <run_dir>/files/
    files_dir = run_dir / "files"
    return files_dir if files_dir.is_dir() else run_dir


def collect_ckpt_files(args: argparse.Namespace) -> list[Path]:
    """Collect checkpoint paths from the various input options.

    Args:
        args: Parsed argument namespace with ckpt, ckpt_dir, run_id, and wandb_dir fields.

    Returns:
        Deduplicated list of resolved checkpoint Paths.

    Raises:
        FileNotFoundError: If any specified path or directory does not exist.
        NotADirectoryError: If ckpt_dir is not a directory.
    """
    paths: list[Path] = []

    if args.ckpt:
        for fp in args.ckpt:
            p = Path(fp)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            if p.suffix != ".ckpt":
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

    if not args.ckpt and not args.ckpt_dir:
        # Auto-discover from local wandb run directory
        wandb_dir = Path(args.wandb_dir)
        run_files_dir = find_local_run_dir(args.run_id, wandb_dir)
        ckpt_dir = run_files_dir / "ckpt"
        search_dir = ckpt_dir if ckpt_dir.is_dir() else run_files_dir
        found = sorted(search_dir.glob("*.ckpt"))
        if not found:
            raise FileNotFoundError(f"No .ckpt files found under: {search_dir}")
        log.info("Auto-discovered ckpt dir: %s", search_dir)
        paths.extend(found)

    # Deduplicate while preserving order
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
    parser.add_argument(
        "--run-id",
        required=True,
        help="WandB run ID (short ID like 'omp634h1')",
    )
    parser.add_argument(
        "--entity", default="frag-gnn", help="WandB entity (default: frag-gnn)"
    )
    parser.add_argument(
        "--project", default="frag-gnn", help="WandB project (default: frag-gnn)"
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        metavar="DIR",
        help="Local wandb directory to search for the run (default: wandb/)",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        metavar="FILE",
        help="Specific .ckpt file to upload (repeatable); overrides auto-discovery",
    )
    parser.add_argument(
        "--ckpt-dir",
        metavar="DIR",
        help="Directory of .ckpt files to upload; overrides auto-discovery",
    )
    parser.add_argument(
        "--root",
        default=None,
        metavar="ROOT",
        help="Strip this local path prefix when computing the remote filename "
        "(default: parent of each ckpt file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args(argv)

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

    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    if args.dry_run:
        log.info("[dry-run] Would upload to run: %s", run_path)
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
        root = args.root if args.root is not None else str(p.parent)
        log.info("  Uploading %s ...", p.name)
        try:
            run.upload_file(str(p), root=root)
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
