"""Copy and rename checkpoints from the local wandb/ directory to a target directory.

Reads the same JSON format as wandb_download_ckpts.py and produces identically-named
output files ({run_name}_{run_id}.ckpt), but sources the checkpoints from a local
wandb/ run directory instead of downloading them from the W&B API.

The local wandb run directory is located by matching the run ID suffix
(e.g. wandb/run-20240101_120000-{run_id}/files/ckpt/).

Usage:
    python scripts/wandb_scripts/wandb_copy_local_ckpts.py \\
        --input_file data/wandb_downloads/spectraverse_neims_original.json \\
        --save_dir saved_ckpts/spectraverse_neims_original/ \\
        --wandb_dir wandb/

    # Skip files that already exist in save_dir
    python scripts/wandb_scripts/wandb_copy_local_ckpts.py \\
        --input_file data/wandb_downloads/spectraverse_neims_original.json \\
        --save_dir saved_ckpts/ \\
        --use_cached
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_local_run_dir(run_id: str, wandb_dir: Path) -> Path | None:
    """Find the files/ subdirectory for a run ID inside the local wandb/ tree.

    Args:
        run_id: Short wandb run ID (e.g. 'fmblwr8w').
        wandb_dir: Root local wandb directory.

    Returns:
        Path to the run's files/ directory, or None if not found.

    Raises:
        FileNotFoundError: If wandb_dir does not exist.
    """
    if not wandb_dir.is_dir():
        raise FileNotFoundError(f"wandb directory not found: {wandb_dir}")

    matches = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.endswith(f"-{run_id}")]
    if not matches:
        return None
    if len(matches) > 1:
        log.warning("Multiple run dirs matched '%s': %s — using most recent", run_id, matches)
        matches.sort(key=lambda d: d.name)

    run_dir = matches[-1]
    files_dir = run_dir / "files"
    return files_dir if files_dir.is_dir() else run_dir


def get_run_name(files_dir: Path, run_id: str) -> str:
    """Read the run name from local wandb-metadata.json.

    Args:
        files_dir: Path to the run's files/ directory.
        run_id: Short run ID used as fallback when metadata is missing.

    Returns:
        Run name string.
    """
    metadata_fp = files_dir / "wandb-metadata.json"
    if metadata_fp.exists():
        with open(metadata_fp) as f:
            meta = json.load(f)
        name = meta.get("name")
        if name:
            return name

    log.warning("  wandb-metadata.json not found or has no name field; using run_id as name")
    return run_id


def copy_ckpt_for_run(
    run_id: str,
    save_dir: Path,
    wandb_dir: Path,
    best_only: bool = True,
    use_cached: bool = False,
) -> None:
    """Copy checkpoint(s) for one run into save_dir with the standard naming convention.

    Naming convention (mirrors wandb_download_ckpts.py):
      - best_only=True  → {run_name}_{run_id}.ckpt
      - best_only=False → {run_name}_{run_id}_epoch={N}.ckpt  (one file per epoch)

    Args:
        run_id: Short wandb run ID.
        save_dir: Destination directory for renamed checkpoints.
        wandb_dir: Root local wandb directory to search.
        best_only: If True, copy only the highest-epoch checkpoint.
        use_cached: If True, skip copying when the destination file already exists.
    """
    files_dir = find_local_run_dir(run_id, wandb_dir)
    if files_dir is None:
        log.warning("> SKIP %s: no local run directory found under %s", run_id, wandb_dir)
        return

    run_name = get_run_name(files_dir, run_id)
    model_tag = f"{run_name}_{run_id}"

    ckpt_subdir = files_dir / "ckpt"
    search_dir = ckpt_subdir if ckpt_subdir.is_dir() else files_dir

    epoch_ckpts: list[tuple[int, Path]] = []
    for ckpt_fp in search_dir.glob("model-epoch=*.ckpt"):
        try:
            epoch_num = int(ckpt_fp.stem.removeprefix("model-epoch="))
            epoch_ckpts.append((epoch_num, ckpt_fp))
        except ValueError:
            log.warning("  Could not parse epoch from %s, skipping", ckpt_fp.name)

    if not epoch_ckpts:
        log.warning("> SKIP %s: no model-epoch=*.ckpt files found in %s", run_id, search_dir)
        return

    epoch_ckpts.sort(key=lambda x: x[0])

    if best_only:
        _epoch, src = epoch_ckpts[-1]
        dest = save_dir / f"{model_tag}.ckpt"
        if dest.exists() and use_cached:
            log.info("> found cached %s, skip", dest)
            return
        log.info("> Copy %s -> %s", src, dest)
        shutil.copy2(src, dest)
    else:
        for epoch_num, src in epoch_ckpts:
            dest = save_dir / f"{model_tag}_epoch={epoch_num}.ckpt"
            if dest.exists() and use_cached:
                log.info("> found cached %s, skip", dest)
                continue
            log.info("> Copy %s -> %s", src, dest)
            shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSON file containing run IDs (same format as wandb_download_ckpts.py)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_ckpts/",
        help="Directory to write renamed checkpoints into (default: ./saved_ckpts/)",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="wandb/",
        help="Local wandb directory to search for run subdirectories (default: wandb/)",
    )
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="Skip copying when the destination file already exists",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = Path(args.wandb_dir)

    input_fp = Path(args.input_file)
    if not input_fp.exists():
        raise FileNotFoundError(f"Input file not found: {input_fp}")

    log.info("> Reading run IDs from %s", input_fp)
    with open(input_fp) as f:
        runs = json.load(f)

    for row in runs:
        run_id = row["run_id"]
        description = row.get("description", "")
        best_only = row.get("best_only", True)
        log.info("> Processing %s (%s)", run_id, description)
        copy_ckpt_for_run(
            run_id,
            save_dir,
            wandb_dir,
            best_only=best_only,
            use_cached=args.use_cached,
        )


if __name__ == "__main__":
    main()
