"""Remove derived (now persistent=False) buffers from saved checkpoints.

Buffers that were previously persistent=True but are now persistent=False:
  - *.mz_bins          (FraGNNetModel and SpecFFN)
  - *.mzs              (SpecFFN)
  - *._h_counts_buf    (FraGNNetModel)
  - *._h_counts_range_buf (FraGNNetModel)

These are all deterministic functions of hyperparameters and must not appear
in the state dict, otherwise strict checkpoint loading will fail.

Usage:
    python scripts/migrate_ckpt_remove_derived_buffers.py path/to/model.ckpt
    python scripts/migrate_ckpt_remove_derived_buffers.py path/to/model.ckpt --out path/to/model_migrated.ckpt
    python scripts/migrate_ckpt_remove_derived_buffers.py path/to/dir/  # migrates all *.ckpt in dir
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DERIVED_BUFFER_SUFFIXES = (
    ".mz_bins",
    ".mzs",
    "._h_counts_buf",
    "._h_counts_range_buf",
)


def migrate_checkpoint(src: Path, dst: Path) -> int:
    """Remove derived buffer keys from a checkpoint file.

    Args:
        src: Source checkpoint path.
        dst: Destination checkpoint path (may equal src for in-place).

    Returns:
        Number of keys removed.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"{src}: no 'state_dict' key found — not a Lightning checkpoint?")

    sd = ckpt["state_dict"]
    to_remove = [k for k in sd if any(k.endswith(s) for s in DERIVED_BUFFER_SUFFIXES)]

    if not to_remove:
        logging.info(f"{src.name}: no derived buffer keys found, skipping")
        return 0

    for k in to_remove:
        logging.info(f"  removing: {k}  shape={list(sd[k].shape)}")
        del sd[k]

    if src != dst:
        shutil.copy2(src, dst.with_suffix(".bak" + dst.suffix))
    torch.save(ckpt, dst)
    logging.info(f"{src.name}: removed {len(to_remove)} key(s) -> saved to {dst}")
    return len(to_remove)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="checkpoint file or directory of *.ckpt files")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (file or dir); defaults to in-place overwrite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print keys that would be removed without writing",
    )
    args = parser.parse_args()

    if args.path.is_dir():
        ckpt_files = sorted(args.path.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No *.ckpt files found in {args.path}")
        out_dir = args.out or args.path
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in ckpt_files:
            dst = out_dir / src.name
            _run(src, dst, args.dry_run)
    else:
        if not args.path.exists():
            raise FileNotFoundError(args.path)
        dst = args.out or args.path
        _run(args.path, dst, args.dry_run)


def _run(src: Path, dst: Path, dry_run: bool) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        logging.warning(f"{src.name}: no state_dict, skipping")
        return
    sd = ckpt["state_dict"]
    to_remove = [k for k in sd if any(k.endswith(s) for s in DERIVED_BUFFER_SUFFIXES)]
    if not to_remove:
        logging.info(f"{src.name}: nothing to remove")
        return
    for k in to_remove:
        logging.info(f"  {'[dry-run] ' if dry_run else ''}remove: {k}  shape={list(sd[k].shape)}")
    if not dry_run:
        for k in to_remove:
            del sd[k]
        torch.save(ckpt, dst)
        logging.info(f"saved -> {dst}  ({len(to_remove)} key(s) removed)")


if __name__ == "__main__":
    main()
