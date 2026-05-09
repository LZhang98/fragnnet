"""Create job_id files for resuming wandb runs that were started without --job_id.

Usage:
    python scripts/create_resume_job_ids.py <wandb_run_id> [<wandb_run_id> ...]

    python scripts/create_resume_job_ids.py 5ka1k9uo ap2wmk7m

Each run ID gets a job_id file at job_id/<run_id>.id containing:
    <run_id>;<abs_path_to_wandb_files_dir>

Pass the resulting job_id to run_pl_model_fit.py via --job_id <run_id>.
"""

import argparse
import glob
import os
import sys


def find_wandb_files_dir(wandb_root: str, run_id: str) -> str | None:
    """Find the files/ directory for a given wandb run ID.

    Args:
        wandb_root: Path to the wandb/ directory.
        run_id: WandB run ID (e.g. '5ka1k9uo').

    Returns:
        Absolute path to the files/ subdirectory, or None if not found.
    """
    pattern = os.path.join(wandb_root, f"run-*-{run_id}", "files")
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        # Multiple matches — take the most recent by directory mtime
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"  [warn] multiple dirs matched for {run_id}, using newest: {matches[0]}")
    return os.path.abspath(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+", help="WandB run IDs to create job_id files for")
    parser.add_argument(
        "--wandb_root",
        default="wandb",
        help="Path to the wandb/ directory (default: wandb/)",
    )
    parser.add_argument(
        "--job_id_dir",
        default="job_id",
        help="Directory to write job_id files into (default: job_id/)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be written without creating files",
    )
    args = parser.parse_args()

    os.makedirs(args.job_id_dir, exist_ok=True)

    ok = True
    for run_id in args.run_ids:
        files_dir = find_wandb_files_dir(args.wandb_root, run_id)
        if files_dir is None:
            print(f"[ERROR] no wandb dir found for run_id={run_id} under {args.wandb_root}/")
            ok = False
            continue

        ckpt_dir = os.path.join(files_dir, "ckpt")
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.isfile(last_ckpt):
            existing = glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if os.path.isdir(ckpt_dir) else []
            if existing:
                print(f"  [warn] last.ckpt missing for {run_id}; found: {[os.path.basename(f) for f in existing]}")
            else:
                print(f"  [warn] no .ckpt files found for {run_id} at {ckpt_dir}")

        job_id_fp = os.path.join(args.job_id_dir, f"{run_id}.id")
        content = f"{run_id};{files_dir}"

        if args.dry_run:
            print(f"[dry-run] would write {job_id_fp}:\n  {content}")
        else:
            if os.path.isfile(job_id_fp):
                print(f"  [warn] overwriting existing {job_id_fp}")
            with open(job_id_fp, "w") as f:
                f.write(content)
            print(f"[ok] {job_id_fp}\n     {content}")

    if not ok:
        sys.exit(1)

    if not args.dry_run:
        print(f"\nResume with:")
        for run_id in args.run_ids:
            print(f"  python scripts/run_pl_model_fit.py --config <config.yaml> --job_id {run_id}")


if __name__ == "__main__":
    main()
