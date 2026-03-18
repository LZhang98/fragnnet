"""Migrate legacy DAG pickle filenames to the canonical plain-integer .pkl format.

Two legacy conventions are normalised:

- **Zero-padded**:  ``00000123.pkl[.gz|.bz2]``  →  ``123.pkl[.gz|.bz2]``
- **Old extension**: ``123.pickle[.bz2]``          →  ``123.pkl[.bz2]``

Supports two backends:

- **Directory**: renames files in-place.
- **Tar archive**: reads the source tar and writes a new tar with renamed members.
  The output path defaults to ``<stem>_migrated.tar`` next to the source file.

Both modes are idempotent: members already in canonical format are kept unchanged,
and renamed entries are only written when no collision would result.

Usage
-----
    # Directory – dry run
    python migrate_dag_zero_padded_to_plain_int.py \\
        --dag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse/dags

    # Directory – apply
    python migrate_dag_zero_padded_to_plain_int.py \\
        --dag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse/dags --apply

    # Tar – dry run (prints what would be renamed, no output file written)
    python migrate_dag_zero_padded_to_plain_int.py \\
        --dag_dp data/frag/nist20.tar

    # Tar – apply (writes data/frag/nist20_migrated.tar by default)
    python migrate_dag_zero_padded_to_plain_int.py \\
        --dag_dp data/frag/nist20.tar --apply

    # Tar – apply with explicit output path
    python migrate_dag_zero_padded_to_plain_int.py \\
        --dag_dp data/frag/nist20.tar --output data/frag/nist20_canonical.tar --apply
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

# Matches zero-padded pkl:  00000123.pkl  /  00000123.pkl.gz  /  00000123.pkl.bz2
_ZERO_PADDED_RE = re.compile(r"^(0\d+)\.pkl(\.gz|\.bz2)?$")

# Matches old .pickle extension:  123.pickle  /  123.pickle.bz2
# Also catches zero-padded .pickle:  00000123.pickle[.bz2]
_OLD_EXT_RE = re.compile(r"^(\d+)\.pickle(\.bz2)?$")


def _resolve_name(name: str) -> tuple[str, bool]:
    """Return ``(canonical_name, was_changed)`` for a single filename.

    Applies both legacy→canonical transformations:

    - Zero-padded .pkl → plain-int .pkl
    - .pickle[.bz2]   → .pkl[.bz2]   (stripping zero-padding too if present)

    Args:
        name: Original filename (basename only, no directory component).

    Returns:
        Tuple of the canonical filename and whether it differs from *name*.
    """
    m = _ZERO_PADDED_RE.match(name)
    if m:
        suffix = m.group(2) or ""
        return f"{int(m.group(1))}.pkl{suffix}", True

    m = _OLD_EXT_RE.match(name)
    if m:
        suffix = m.group(2) or ""
        return f"{int(m.group(1))}.pkl{suffix}", True

    return name, False


# ---------------------------------------------------------------------------
# Directory backend
# ---------------------------------------------------------------------------


def migrate_dir(dag_dp: Path, apply: bool) -> tuple[int, int, int]:
    """Rename legacy files inside *dag_dp* to canonical names in-place.

    Args:
        dag_dp: Directory containing DAG pickle files.
        apply: If ``False``, only log what would happen without touching disk.

    Returns:
        Tuple of ``(renamed, skipped_collision, already_canonical)`` counts.
    """
    if not dag_dp.is_dir():
        raise ValueError(f"Not a directory: {dag_dp}")

    renamed = 0
    skipped_collision = 0
    already_canonical = 0

    for src in sorted(dag_dp.iterdir()):
        new_name, was_changed = _resolve_name(src.name)
        if not was_changed:
            already_canonical += 1
            continue

        dst = src.parent / new_name
        if dst.exists():
            logger.warning(
                "SKIP (collision) %s → %s (destination already exists)", src.name, new_name
            )
            skipped_collision += 1
            continue

        logger.info("%s %s → %s", "RENAME" if apply else "WOULD RENAME", src.name, new_name)
        if apply:
            src.rename(dst)
        renamed += 1

    return renamed, skipped_collision, already_canonical


# ---------------------------------------------------------------------------
# Tar backend
# ---------------------------------------------------------------------------


def migrate_tar(src_tar: Path, dst_tar: Path, apply: bool) -> tuple[int, int, int]:
    """Read *src_tar* and write *dst_tar* with legacy member names renamed to canonical.

    Members already in canonical format are copied unchanged.  The output tar
    is written atomically via a temporary file so *dst_tar* is never left in a
    partial state if the process is interrupted.

    Args:
        src_tar: Source tar archive.
        dst_tar: Output tar archive path.
        apply: If ``False``, only log planned renames without writing *dst_tar*.

    Returns:
        Tuple of ``(renamed, skipped_collision, already_canonical)`` counts.

    Raises:
        ValueError: If a member cannot be extracted from the archive.
    """
    renamed = 0
    skipped_collision = 0
    already_canonical = 0

    with tarfile.open(src_tar, "r") as tf_in:
        members = tf_in.getmembers()
        existing_names = {m.name for m in members}

        # Build rename map, detecting collisions up front.
        # Two collision cases:
        #   1. Target name already exists as another member in the archive.
        #   2. Two different legacy names both resolve to the same canonical name.
        rename_map: dict[str, str] = {}
        claimed_targets: set[str] = set()
        for member in members:
            new_name, was_changed = _resolve_name(member.name)
            if not was_changed:
                already_canonical += 1
                continue
            if new_name in existing_names:
                logger.warning(
                    "SKIP (collision) %s → %s (canonical name already exists in archive)",
                    member.name,
                    new_name,
                )
                skipped_collision += 1
                continue
            if new_name in claimed_targets:
                logger.warning(
                    "SKIP (collision) %s → %s (another member already maps to this name)",
                    member.name,
                    new_name,
                )
                skipped_collision += 1
                continue
            rename_map[member.name] = new_name
            claimed_targets.add(new_name)
            renamed += 1
            logger.info(
                "%s %s → %s", "RENAME" if apply else "WOULD RENAME", member.name, new_name
            )

        if not apply:
            return renamed, skipped_collision, already_canonical

        # Write output tar atomically.
        dst_tar.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dst_tar.parent, suffix=".tar.tmp")
        try:
            with open(tmp_fd, "wb") as tmp_fh, tarfile.open(fileobj=tmp_fh, mode="w") as tf_out:
                for member in members:
                    out_name = rename_map.get(member.name, member.name)
                    out_member = tarfile.TarInfo(name=out_name)
                    out_member.size = member.size
                    out_member.mtime = member.mtime
                    out_member.mode = member.mode
                    out_member.type = member.type
                    out_member.uid = member.uid
                    out_member.gid = member.gid
                    out_member.uname = member.uname
                    out_member.gname = member.gname

                    if member.isfile():
                        f = tf_in.extractfile(member)
                        if f is None:
                            raise ValueError(f"Could not extract member '{member.name}'")
                        tf_out.addfile(out_member, f)
                    else:
                        tf_out.addfile(out_member)

            Path(tmp_path).replace(dst_tar)
            logger.info("Written → %s", dst_tar)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    return renamed, skipped_collision, already_canonical


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy DAG pickle filenames to canonical plain-int .pkl format. "
            "Handles both zero-padded (00000123.pkl) and old-extension (123.pickle) conventions. "
            "Supports directory and tar-archive inputs."
        )
    )
    parser.add_argument(
        "--dag_dp",
        required=True,
        type=Path,
        help="Directory or .tar file containing DAG pickle files to migrate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the migrated tar archive (tar mode only). "
            "Defaults to <stem>_migrated.tar next to the source file."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually apply changes. Without this flag the script is a dry run.",
    )
    args = parser.parse_args()

    if not args.apply:
        logger.info("DRY RUN — pass --apply to apply changes")

    is_tar = args.dag_dp.suffix == ".tar" and args.dag_dp.is_file()
    is_dir = args.dag_dp.is_dir()

    if not is_tar and not is_dir:
        logger.error("--dag_dp must be an existing directory or .tar file: %s", args.dag_dp)
        sys.exit(1)

    if is_tar:
        dst_tar = args.output or args.dag_dp.with_name(args.dag_dp.stem + "_migrated.tar")
        if args.apply and dst_tar.exists():
            logger.error("Output file already exists: %s (remove it or choose --output)", dst_tar)
            sys.exit(1)
        logger.info("Source tar : %s", args.dag_dp)
        logger.info("Output tar : %s", dst_tar)
        renamed, skipped, canonical = migrate_tar(args.dag_dp, dst_tar, apply=args.apply)
    else:
        if args.output is not None:
            logger.warning("--output is ignored in directory mode")
        renamed, skipped, canonical = migrate_dir(args.dag_dp, apply=args.apply)

    logger.info(
        "Done. renamed=%d  skipped_collision=%d  already_canonical=%d",
        renamed,
        skipped,
        canonical,
    )
    if skipped:
        logger.warning("%d file(s) skipped due to collision — review manually.", skipped)
        sys.exit(1)


if __name__ == "__main__":
    main()
