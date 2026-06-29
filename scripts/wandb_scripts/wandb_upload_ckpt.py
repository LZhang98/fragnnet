"""Upload local checkpoint file(s) to one or more existing wandb runs.

Useful for crashed runs where checkpoints were saved locally but not synced.
By default, searches the local `wandb/` directory for each run matching the
given run ID(s), then uploads all checkpoints from their `ckpt/` subdirectory.

Usage:
    # Single run — auto-discover ckpts from the local wandb/ run dir
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1

    # Multiple runs at once
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1 abc123de

    # Custom wandb directory (if not running from project root)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1 --wandb_dir /path/to/wandb

    # Upload specific files instead (single run only)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1 --ckpt path/to/model-epoch=010.ckpt

    # Upload all .ckpt files from a specific directory (single run only)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1 --ckpt_dir path/to/ckpt/

    # Specify entity/project (default: frag-gnn/frag-gnn)
    python scripts/wandb_scripts/wandb_upload_ckpt.py --run_ids omp634h1 abc123de --entity myorg --project myproj
"""

import argparse
import datetime
import logging
import os
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
    files_dir = run_dir / "files"
    return files_dir if files_dir.is_dir() else run_dir


def collect_ckpt_files_for_run(run_id: str, args: argparse.Namespace) -> list[Path]:
    """Collect checkpoint paths for a single run.

    When --ckpt or --ckpt_dir are given they take precedence over auto-discovery.
    Auto-discovery is only used when neither override is provided.

    Args:
        run_id: Short wandb run ID.
        args: Parsed argument namespace.

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
        wandb_dir = Path(args.wandb_dir)
        run_files_dir = find_local_run_dir(run_id, wandb_dir)
        ckpt_dir = run_files_dir / "ckpt"
        search_dir = ckpt_dir if ckpt_dir.is_dir() else run_files_dir
        found = sorted(search_dir.glob("*.ckpt"))
        if not found:
            raise FileNotFoundError(f"No .ckpt files found under: {search_dir}")
        log.info("Auto-discovered ckpt dir: %s", search_dir)
        paths.extend(found)

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
        "--run_ids",
        nargs="+",
        required=True,
        metavar="RUN_ID",
        help="One or more WandB run IDs (e.g. omp634h1 abc123de)",
    )
    parser.add_argument("--entity", default="frag-gnn", help="WandB entity (default: frag-gnn)")
    parser.add_argument("--project", default="frag-gnn", help="WandB project (default: frag-gnn)")
    parser.add_argument(
        "--wandb_dir",
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
        "--ckpt_dir",
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
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if the remote file is newer than the local file",
    )
    args = parser.parse_args(argv)

    if (args.ckpt or args.ckpt_dir) and len(args.run_ids) > 1:
        parser.error("--ckpt / --ckpt_dir can only be used with a single --run_ids entry")

    import wandb

    api = wandb.Api()

    total_ok = 0
    total_fail = 0

    for run_id in args.run_ids:
        log.info("=== Run %s ===", run_id)

        try:
            ckpt_files = collect_ckpt_files_for_run(run_id, args)
        except (FileNotFoundError, NotADirectoryError) as exc:
            log.error("  %s", exc)
            total_fail += 1
            continue

        log.info("  Checkpoints to upload (%d):", len(ckpt_files))
        for p in ckpt_files:
            log.info("    %s  (%s)", p, _human_size(p.stat().st_size))

        run_path = f"{args.entity}/{args.project}/{run_id}"
        if args.dry_run:
            log.info(
                "  [dry-run] Would upload to run: %s%s",
                run_path,
                " (--force: skip timestamp check)" if args.force else " (skipping if remote is newer)",
            )
            continue

        try:
            run = api.run(run_path)
        except Exception as exc:
            log.error("  Could not fetch run %s: %s", run_path, exc)
            total_fail += 1
            continue

        log.info("  Uploading to: %s (state: %s)", run.name, run.state)

        remote_mtimes = _remote_file_mtimes(run)

        for p in ckpt_files:
            root = args.root if args.root is not None else str(p.parent)
            remote_name = os.path.relpath(str(p), root)

            if not args.force and remote_name in remote_mtimes:
                local_mtime = _local_mtime_utc(p)
                remote_mtime = remote_mtimes[remote_name]
                if local_mtime <= remote_mtime:
                    log.info(
                        "    SKIP %s (local %s <= remote %s)",
                        p.name,
                        local_mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        remote_mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                    continue

            log.info("    %s ...", p.name)
            try:
                run.upload_file(str(p), root=root)
                log.info("      OK")
                total_ok += 1
            except Exception as exc:
                log.error("      FAILED: %s", exc)
                total_fail += 1

    log.info("Done: %d uploaded, %d failed", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


def _remote_file_mtimes(run) -> dict[str, datetime.datetime]:
    """Return a mapping of remote filename -> updatedAt (UTC-aware) for all files in a run.

    Args:
        run: A wandb Run object returned by api.run().

    Returns:
        Dict mapping each remote file's path (as stored in W&B) to its last-updated datetime.
    """
    result: dict[str, datetime.datetime] = {}
    for f in run.files():
        raw = f.updatedAt.replace("Z", "+00:00")
        result[f.name] = datetime.datetime.fromisoformat(raw)
    return result


def _local_mtime_utc(p: Path) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(p.stat().st_mtime, tz=datetime.UTC)


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


if __name__ == "__main__":
    raise SystemExit(main())
