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
import heapq
import logging
import os
import pickle
from collections.abc import Iterator

import h5py
import lmdb
from tqdm import tqdm

from fragnnet.utils.frag_utils import dump_dag_hdf5_packed, initialize_packed_dag_hdf5
from fragnnet.utils.misc_utils import booltype


def _key_value_pairs(txn: "lmdb.Transaction", mol_ids: list[int]) -> Iterator[tuple[bytes, bytes]]:
    """Look up ``(key_bytes, value_bytes)`` pairs for the given mol_ids, one get() each."""
    for mol_id in mol_ids:
        key_bytes = str(mol_id).encode()
        yield key_bytes, txn.get(key_bytes)


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

    num_ok = 0
    num_err = 0
    with env.begin() as txn, h5py.File(h5_fp, "w") as h5_file:
        total = txn.stat()["entries"]
        logging.info(f"LMDB entries: {total}")

        # Packed HDF5 storage indexes DAGs by mol_id used directly as a dense array
        # position, but LMDB's cursor iterates keys in lexicographic byte order, not
        # numeric order. Selecting the smallest mol_ids explicitly keeps a partial
        # (--num_entries) migration's output size proportional to what was requested.
        entries: Iterator[tuple[bytes, bytes]]
        if args.num_entries > 0:
            selected_ids = heapq.nsmallest(
                args.num_entries, (int(k.decode()) for k, _ in txn.cursor())
            )
            total = len(selected_ids)
            logging.info(
                f"Processing {total} entries with the smallest mol_id (--num_entries="
                f"{args.num_entries})"
            )
            entries = _key_value_pairs(txn, selected_ids)
        else:
            entries = txn.cursor()

        initialize_packed_dag_hdf5(h5_file)
        for key_bytes, val_bytes in tqdm(entries, total=total):
            key = key_bytes.decode()
            try:
                dag_d = pickle.loads(bytes(val_bytes))
                dump_dag_hdf5_packed(h5_file, key, dag_d)
                num_ok += 1
            except Exception as exc:
                logging.warning(f"Failed to convert mol_id={key}: {exc}")
                num_err += 1

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
