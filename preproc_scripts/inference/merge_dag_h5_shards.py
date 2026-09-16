"""Merge per-shard packed DAG HDF5 files into a single dags.h5.

``preproc_infer_synced_dag_arr_hdf5.sh`` runs 03_prepare_dag_feats.py as a SLURM
array, so each task writes its own ``shard_<task>/dags.h5`` plus ``meta_info.json``.
``SpecMolFragDataset`` takes a single ``.h5`` path, so the shards have to be
concatenated once the array finishes.

The packed format (``packed_v1``) is a flat byte blob plus an index keyed by a dense
``mol_id``. Shards slice mol_df without renumbering, so mol_ids are already global and
disjoint: merging is a blob concatenation with the shard's index offsets shifted by the
running blob length.

Example:
    python preproc_scripts/inference/merge_dag_h5_shards.py \
        --shard_dp /scratch/feiw/frag-gnn-data/inference/frags/synced_d3_h4_hdf5 \
        --output_dp /scratch/feiw/frag-gnn-data/inference/frags/synced_d3_h4_hdf5_merged
"""

import argparse
import glob
import json
import logging
import os

import h5py
import numpy as np

from fragnnet.utils.frag_utils import (
    PACKED_DAG_BYTES_DATASET,
    PACKED_DAG_INDEX_DATASET,
    initialize_packed_dag_hdf5,
)

# Copy the byte blob in slices rather than in one read, so a multi-GB shard does not
# have to be materialized in memory all at once.
COPY_CHUNK_BYTES = 256 * 1024 * 1024


def find_shards(shard_dp: str) -> list[str]:
    """Return the shard dags.h5 paths under ``shard_dp``, ordered by shard index.

    Accepts both the ``shard_<n>/dags.h5`` layout written by the array job and a flat
    directory of ``*.h5`` files.
    """
    shard_fps = glob.glob(os.path.join(shard_dp, "shard_*", "dags.h5"))
    if shard_fps:
        return sorted(
            shard_fps, key=lambda fp: int(os.path.basename(os.path.dirname(fp)).split("_")[-1])
        )

    shard_fps = sorted(glob.glob(os.path.join(shard_dp, "*.h5")))
    if not shard_fps:
        raise FileNotFoundError(f"no shard_*/dags.h5 or *.h5 found under {shard_dp}")
    return shard_fps


def merge_shard(src_h5: h5py.File, dst_h5: h5py.File) -> int:
    """Append one shard's DAGs to the merged file. Returns the number of molecules copied.

    Raises:
        ValueError: If a mol_id is already present in the merged file, which would mean
            the shards overlap.
    """
    src_bytes = src_h5[PACKED_DAG_BYTES_DATASET]
    src_index = src_h5[PACKED_DAG_INDEX_DATASET][:]
    dst_bytes = dst_h5[PACKED_DAG_BYTES_DATASET]
    dst_index = dst_h5[PACKED_DAG_INDEX_DATASET]

    base = int(dst_bytes.shape[0])
    n_src_bytes = int(src_bytes.shape[0])
    dst_bytes.resize((base + n_src_bytes,))
    for start in range(0, n_src_bytes, COPY_CHUNK_BYTES):
        stop = min(start + COPY_CHUNK_BYTES, n_src_bytes)
        dst_bytes[base + start : base + stop] = src_bytes[start:stop]

    present = np.nonzero(src_index["length"] != 0)[0]
    if present.size == 0:
        return 0

    max_mol_id = int(present.max())
    if max_mol_id >= dst_index.shape[0]:
        dst_index.resize((max_mol_id + 1,))

    existing = dst_index[:][present]
    n_collisions = int(np.count_nonzero(existing["length"] != 0))
    if n_collisions:
        raise ValueError(
            f"{n_collisions} mol_id(s) already present in the merged file — shards overlap"
        )

    shifted = src_index[present]
    shifted["offset"] = shifted["offset"] + base
    # h5py fancy-index assignment needs increasing indices, which np.nonzero gives.
    dst_index[present] = shifted
    return int(present.size)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    shard_fps = find_shards(args.shard_dp)
    logging.info(f"merging {len(shard_fps)} shards from {args.shard_dp}")

    os.makedirs(args.output_dp, exist_ok=True)
    out_h5_fp = os.path.join(args.output_dp, "dags.h5")
    if os.path.exists(out_h5_fp) and not args.overwrite:
        raise FileExistsError(f"{out_h5_fp} exists (pass --overwrite to replace)")

    meta_info: dict[str, list[int]] = {}
    n_total = 0
    with h5py.File(out_h5_fp, "w") as dst_h5:
        initialize_packed_dag_hdf5(dst_h5)
        for shard_fp in shard_fps:
            with h5py.File(shard_fp, "r") as src_h5:
                n_copied = merge_shard(src_h5, dst_h5)
            n_total += n_copied

            shard_meta_fp = os.path.join(os.path.dirname(shard_fp), "meta_info.json")
            if os.path.exists(shard_meta_fp):
                with open(shard_meta_fp) as f:
                    shard_meta = json.load(f)
                meta_info.update(shard_meta)
            else:
                logging.warning(f"no meta_info.json next to {shard_fp}")

            logging.info(f"{shard_fp}: {n_copied} molecules (running total {n_total})")

    meta_info_fp = os.path.join(args.output_dp, "meta_info.json")
    with open(meta_info_fp, "w") as f:
        json.dump(meta_info, f)

    logging.info(f"wrote {out_h5_fp} ({n_total} molecules, {os.path.getsize(out_h5_fp)} bytes)")
    logging.info(f"wrote {meta_info_fp} ({len(meta_info)} entries)")
    if len(meta_info) != n_total:
        logging.warning(f"meta_info entries ({len(meta_info)}) != molecules in dags.h5 ({n_total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard_dp",
        type=str,
        required=True,
        help="Directory holding shard_<n>/dags.h5 (or a flat dir of *.h5 shards)",
    )
    parser.add_argument(
        "--output_dp",
        type=str,
        required=True,
        help="Directory to write the merged dags.h5 and meta_info.json into",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing dags.h5 in --output_dp"
    )
    main(parser.parse_args())
