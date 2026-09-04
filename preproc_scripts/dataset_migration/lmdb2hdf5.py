"""Convert an existing LMDB DAG database to HDF5 format.

Reads the LMDB database at ``<frag_dp>/dags/`` produced by
``03_prepare_dag_feats.py --use_lmdb True`` and writes a single
``<frag_dp>/dags.h5`` file.  The output can then be used by passing
``frag_dp=<frag_dp>/dags.h5`` to :class:`SpecMolFragDataset`.

Usage::

    python preproc_scripts/dataset_migration/lmdb2hdf5.py \\
        --frag_dp /path/to/frag_dp \\
        --num_entries -1
"""

import argparse
import logging
import os
import pickle

import h5py
import lmdb
from tqdm import tqdm

from fragnnet.utils.frag_utils import dump_dag_hdf5_packed, initialize_packed_dag_hdf5
from fragnnet.utils.misc_utils import booltype


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    lmdb_dp = os.path.join(args.frag_dp, "dags")
    if not os.path.isdir(lmdb_dp):
        raise FileNotFoundError(
            f"LMDB directory not found: {lmdb_dp}. "
            "Run 03_prepare_dag_feats.py with --use_lmdb True first."
        )

    h5_fp = os.path.join(args.frag_dp, "dags.h5")
    if os.path.exists(h5_fp) and not args.overwrite:
        raise FileExistsError(f"{h5_fp} already exists. Pass --overwrite True to overwrite.")

    env = lmdb.open(
        lmdb_dp,
        readonly=True,
        readahead=False,
        max_readers=1,
        lock=False,
        meminit=False,
    )

    with env.begin() as txn:
        total = txn.stat()["entries"]

    logging.info(f"LMDB entries: {total}")
    if args.num_entries > 0:
        total = min(total, args.num_entries)
        logging.info(f"Processing first {total} entries (--num_entries={args.num_entries})")

    num_ok = 0
    num_err = 0
    with env.begin() as txn, h5py.File(h5_fp, "w") as h5_file:
        initialize_packed_dag_hdf5(h5_file)
        cursor = txn.cursor()
        for key_bytes, val_bytes in tqdm(cursor.iternext_dup() if False else cursor, total=total):
            key = key_bytes.decode()
            try:
                dag_d = pickle.loads(bytes(val_bytes))
                dump_dag_hdf5_packed(h5_file, key, dag_d)
                num_ok += 1
            except Exception as exc:
                logging.warning(f"Failed to convert mol_id={key}: {exc}")
                num_err += 1
            if args.num_entries > 0 and (num_ok + num_err) >= args.num_entries:
                break

    env.close()
    logging.info(f"Done. converted={num_ok}, errors={num_err}, output={h5_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LMDB DAG database to HDF5.")
    parser.add_argument(
        "--frag_dp",
        type=str,
        required=True,
        help="Directory containing the LMDB 'dags/' subdirectory.",
    )
    parser.add_argument(
        "--num_entries",
        type=int,
        default=-1,
        help="Maximum number of entries to convert (-1 for all).",
    )
    parser.add_argument(
        "--overwrite",
        type=booltype,
        default=False,
        help="Overwrite existing dags.h5 if present.",
    )
    args = parser.parse_args()
    main(args)
