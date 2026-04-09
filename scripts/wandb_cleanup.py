import argparse
import os
from datetime import UTC, datetime, timedelta

# from tqdm import tqdm
import wandb


def safe_get_files(run):
    """Safely get files from a run, handling API pagination bugs."""
    try:
        return list(run.files(per_page=100))
    except (TypeError, KeyError) as e:
        if "NoneType" in str(e) or "subscriptable" in str(e):
            print(f"  Skipping run {run.id} - API files() failed")
            return []
        raise


def should_skip_run(run):
    """Skip mces pretraining runs."""
    # Normalize once so all matching is case-insensitive.
    run_name_lower = run.name.lower()
    tags_lower = [tag.lower() for tag in (run.tags or [])]

    if "mces" in run_name_lower and (
        "pretrain" in run_name_lower or "pretraining" in run_name_lower
    ):
        return True

    if "mces" in tags_lower and ("pretrain" in tags_lower or "pretraining" in tags_lower):
        return True

    return False


def should_delete_all_ckpts(run):
    """Delete ALL ckpts if: (massformer OR neims OR graff) AND NOT (massspecgym OR nist20v3) in name."""
    run_name_lower = run.name.lower()

    # These model families are explicitly marked for full checkpoint cleanup.
    target_keywords = ["massformer", "neims"]
    has_target = any(keyword in run_name_lower for keyword in target_keywords)

    # Keep all checkpoints for protected benchmark datasets.
    exclude_keywords = ["massspecgym", "nist20v3"]
    has_exclude = any(keyword in run_name_lower for keyword in exclude_keywords)

    return has_target and not has_exclude


def is_file_recent(file, days=30):
    """Check if file was updated within last N days."""
    try:
        file_time = datetime.fromisoformat(file.updated_at.replace("Z", "+00:00"))
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        return file_time >= cutoff_date
    except Exception:
        # Fail-safe: if timestamp parsing fails, do not delete the file.
        return True  # Assume recent if can't parse timestamp


def main():
    parser = argparse.ArgumentParser(description="Clean W&B checkpoints")
    parser.add_argument(
        "--delete", dest="dry_run", action="store_false", help="Actually delete files"
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument("--entity", default="frag-gnn")
    parser.add_argument("--project", default="frag-gnn")
    parser.add_argument(
        "--recent-days",
        type=int,
        default=30,
        help="Don't delete files newer than N days (default: 30)",
    )
    args = parser.parse_args()

    mode = "DRY RUN MODE (SAFE)" if args.dry_run else "LIVE DELETE MODE"
    print(f"{mode} (Protecting files < {args.recent_days} days old)")

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    total_size_saved = 0
    total_files_deleted = 0
    runs_processed = 0
    runs_skipped_mces = 0
    recent_files_skipped = 0
    total_runs = len(runs)
    print(f"Found {total_runs} runs in {args.entity}/{args.project}\n")

    for i, run in enumerate(runs):
        if should_skip_run(run):
            runs_skipped_mces += 1
            print(f"{i + 1}/{total_runs} Skipping mces pretraining: {run.name}")
            continue

        files = safe_get_files(run)
        if not files:
            continue

        runs_processed += 1

        # Find best checkpoint
        best_ckpt_wandb_fp = None
        best_epoch = -1

        delete_all_ckpts = should_delete_all_ckpts(run)
        if delete_all_ckpts:
            print(f"{i + 1}/{total_runs} Run {run.name[:50]}: Marked for ALL checkpoint deletion")
            best_ckpt_wandb_fp = None  # No checkpoint will be kept
        else:
            for file in files:
                if file.name.endswith(".ckpt") and "last" not in file.name.lower():
                    # Expected naming pattern: model-epoch=<N>.ckpt
                    epoch_str = (
                        os.path.basename(file.name)
                        .removeprefix("model-epoch=")
                        .removesuffix(".ckpt")
                    )

                    if not epoch_str.isdigit():
                        continue

                    epoch = int(epoch_str)
                    if epoch > best_epoch:
                        # Retain only the highest epoch checkpoint per run.
                        best_ckpt_wandb_fp = file.name
                        best_epoch = epoch

            if best_ckpt_wandb_fp is None and any(file.name.endswith(".ckpt") for file in files):
                print(
                    f"{i + 1}/{total_runs} Run {run.name[:50]}: No epoch checkpoints found; deleting all .ckpt files"
                )

        # Delete all other .ckpt files (but skip recent ones)
        deleted_in_run = 0
        for file in files:
            if file.name.endswith(".ckpt") and file.name != best_ckpt_wandb_fp:
                # Skip if file is recent
                if is_file_recent(file, args.recent_days):
                    recent_files_skipped += 1
                    print(
                        f"{i + 1}/{total_runs} Run {run.name[:50]} {run.id} {file.name} ({file.size / (1024 * 1024):.1f} MB) [RECENT - SKIPPED]"
                    )
                    continue

                size_mb = file.size / (1024 * 1024)
                action = "WOULD DELETE" if args.dry_run else "DELETED"

                print(
                    f"{i + 1}/{total_runs} Run {run.name[:50]} {run.id} {file.name} ({size_mb:.1f} MB) [{action}]"
                )

                if not args.dry_run:  # LIVE DELETE
                    try:
                        # Keep delete call commented while validating cleanup policy.
                        file.delete()
                        total_size_saved += file.size
                        total_files_deleted += 1
                        deleted_in_run += 1
                    except Exception as e:
                        print(
                            f"{i + 1}/{total_runs} Run {run.name[:50]} {run.id} Delete failed: {e}"
                        )
                else:
                    total_size_saved += file.size
                    total_files_deleted += 1
                    deleted_in_run += 1

        if best_ckpt_wandb_fp:
            status = "Kept epoch" if deleted_in_run > 0 else "Only best checkpoint"
            print(
                f"{i + 1}/{total_runs} Run {run.name[:50]} {run.id} {status} {best_epoch}, Deleted {deleted_in_run} other ckpts"
            )
        else:
            print(
                f"{i + 1}/{total_runs} Run {run.name[:50]} {run.id} Deleted all {deleted_in_run} ckpts"
            )

    total_mb = total_size_saved / (1024 * 1024)
    total_gb = total_mb / 1024
    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"\n{mode} COMPLETE!")
    print(f"Processed: {runs_processed} runs")
    print(f"Skipped mces: {runs_skipped_mces}")
    print(f"Recent files protected: {recent_files_skipped}")
    print(
        f"{total_files_deleted} files ({total_gb:.1f} GB) {'WOULD BE' if args.dry_run else 'DELETED'}"
    )


if __name__ == "__main__":
    main()
