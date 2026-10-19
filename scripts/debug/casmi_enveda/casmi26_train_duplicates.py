"""Exact copies and near-duplicate spectra in the CASMI 2026 training set.

Two counts, nothing removed from the data:

1. **Exact copies**: byte-identical peak arrays (m/z and intensity), over the whole train set and
   whatever the labels. A copy filed under two molecules is a label conflict.
2. **Near-duplicates** by the Spectraverse dedup rule (user decisions 2026-10-04,
   ``docs/plans/masster/spectraverse_dedup_plan.md``), reusing its pair logic
   (``preproc_scripts.spectraverse.dedup_train_split._group_edges``):
   - pairs share molecule (InChIKey first block, the CASMI metric's key) and adduct;
   - instrument and collision energy must match, and a missing value matches anything;
   - the instrument is the class from ``casmi26_train_stats.instrument_class``, with ``UNK``
     counted as missing;
   - CE is ``collision_energy_ev`` (NCE already converted), rounded to 0.1 eV;
   - the Hungarian cosine at 5 ppm (at least 1 mDa) must exceed 0.99, after a binned 0.01 Da
     prefilter above 0.95.

   ``same`` edges form clusters, and each cluster keeps the spectrum with the most peaks. A
   ``wildcard`` edge drops the less-specified spectrum. Each edge records the median |m/z
   difference| over matched peaks: 0 means the same m/z list, a redistributed copy.

Usage:
    PYTHONPATH=src:. python scripts/debug/casmi_enveda/casmi26_train_duplicates.py \
        --out_dp output/casmi26_train_check_2026_10_05/duplicates --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from preproc_scripts.spectraverse.dedup_train_split import _group_edges
from scripts.debug.casmi_enveda.casmi26_train_stats import instrument_class

logger = logging.getLogger(__name__)

# Peak arrays shared with forked workers (copy-on-write, never written after the fork).
_MZ: np.ndarray | None = None
_INT: np.ndarray | None = None
_OFF: np.ndarray | None = None
_INST: list[str] | None = None
_CE: list[str] | None = None


def load_peaks(train_fp: Path, num_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattened m/z and intensity arrays with row offsets, in file order.

    The arrays are preallocated from the release's ``num_peaks`` (checked against every list
    length), so the parquet is read one row group at a time without a second full copy.
    """
    offsets = np.concatenate([[0], np.cumsum(num_peaks)]).astype(np.int64)
    mz_all = np.empty(offsets[-1], dtype=np.float64)
    int_all = np.empty(offsets[-1], dtype=np.float64)
    parquet = pq.ParquetFile(train_fp)
    row = 0
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=["ms2_mzs", "ms2_normalized_intensities"])
        mzs = table.column("ms2_mzs").combine_chunks()
        ints = table.column("ms2_normalized_intensities").combine_chunks()
        local = mzs.offsets.to_numpy()
        n_rows = len(local) - 1
        if not np.array_equal(local, ints.offsets.to_numpy()) or not np.array_equal(
            np.diff(local), num_peaks[row : row + n_rows]
        ):
            raise ValueError(f"Row group {group}: list lengths disagree with num_peaks")
        start, end = offsets[row], offsets[row + n_rows]
        # Slice the values by the offsets: a list array's values buffer can be longer.
        mz_all[start:end] = mzs.values.to_numpy()[local[0] : local[-1]]
        int_all[start:end] = ints.values.to_numpy()[local[0] : local[-1]]
        row += n_rows
    if row != len(num_peaks):
        raise ValueError("Row groups do not cover every spectrum")
    return mz_all, int_all, offsets


def ce_signature(ce_lists: pd.Series) -> list[str]:
    """Collision energies (eV) as a sorted string rounded to 0.1 eV; "" when missing."""
    return [
        "" if values is None or len(values) == 0 else "|".join(f"{v:.1f}" for v in sorted(values))
        for values in ce_lists
    ]


def exact_copy_keys(mz: np.ndarray, ints: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """A 16-byte digest of each spectrum's m/z and intensity bytes."""
    keys = np.empty(len(offsets) - 1, dtype=object)
    for row in range(len(offsets) - 1):
        start, end = offsets[row], offsets[row + 1]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(mz[start:end].tobytes())
        digest.update(b"|")
        digest.update(ints[start:end].tobytes())
        keys[row] = digest.hexdigest()
    return keys


def _edges_for_group(task: tuple) -> list[tuple]:
    rows, bin_da, prefilter, ppm, min_mz, min_cosine = task
    peaks = [
        list(zip(_MZ[_OFF[r] : _OFF[r + 1]].tolist(), _INT[_OFF[r] : _OFF[r + 1]].tolist()))
        for r in rows
    ]
    return _group_edges(
        (
            [int(r) for r in rows],
            peaks,
            [_INST[r] for r in rows],
            [_CE[r] for r in rows],
            [len(p) for p in peaks],
            bin_da,
            prefilter,
            ppm,
            min_mz,
            min_cosine,
        )
    )


def cluster_rows(n_rows: int, num_peaks: np.ndarray, edges: pd.DataFrame) -> pd.DataFrame:
    """Clusters and kept flags, as in the Spectraverse dedup script.

    ``same`` edges are joined into connected components; each keeps the row with the most peaks,
    then the lowest row index. A ``wildcard`` loser is dropped in favour of its twin.
    """
    same = edges[edges["kind"] == "same"]
    adj = coo_matrix(
        (np.ones(len(same)), (same["spec_a"].to_numpy(), same["spec_b"].to_numpy())),
        shape=(n_rows, n_rows),
    )
    _, label = connected_components(adj, directed=False)
    frame = pd.DataFrame({"row": np.arange(n_rows), "npk": num_peaks, "label": label})
    rep = frame.sort_values(["npk", "row"], ascending=[False, True]).groupby("label")["row"].first()
    cluster = frame["label"].map(rep).to_numpy(dtype=np.int64, copy=True)
    wild = edges[edges["kind"] == "wildcard"]
    winner = np.where(wild["loser"] == wild["spec_a"], wild["spec_b"], wild["spec_a"])
    winner_of = pd.Series(winner, index=wild["loser"].to_numpy()).groupby(level=0).first()
    is_loser = frame["row"].isin(winner_of.index).to_numpy()
    cluster[is_loser] = frame.loc[is_loser, "row"].map(winner_of).to_numpy()
    frame["cluster"] = cluster
    frame["wildcard_loser"] = is_loser
    frame["kept"] = (frame["cluster"] == frame["row"]) & ~is_loser
    frame["cluster_size"] = frame.groupby("cluster")["row"].transform("size")
    return frame.drop(columns=["label", "npk"])


def main() -> None:
    global _MZ, _INT, _OFF, _INST, _CE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    parser.add_argument("--out_dp", type=Path, required=True)
    parser.add_argument("--min_cosine", type=float, default=0.99)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--min_mz", type=float, default=200.0)
    parser.add_argument("--bin_da", type=float, default=0.01)
    parser.add_argument("--prefilter", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_groups", type=int, default=None, help="Smoke runs only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dp.mkdir(parents=True, exist_ok=True)
    start = time.time()

    meta = pd.read_parquet(
        args.train_fp,
        columns=[
            "ingest_lib", "inchikey14", "adduct", "instrument_type", "collision_energy_ev",
            "num_peaks",
        ],
    )
    num_peaks = meta["num_peaks"].to_numpy(dtype=np.int64)
    _MZ, _INT, _OFF = load_peaks(args.train_fp, num_peaks)
    logger.info("loaded %d spectra, %d peaks", len(meta), len(_MZ))

    # 1. Exact copies.
    meta["copy_key"] = exact_copy_keys(_MZ, _INT, _OFF)
    logger.info("hashed peak arrays")

    # 2. Near-duplicate edges per (molecule, adduct) group.
    _INST = ["NA" if c == "UNK" else c for c in map(instrument_class, meta["instrument_type"])]
    _CE = ce_signature(meta["collision_energy_ev"])
    groups = [
        idx
        for idx in meta.groupby(["inchikey14", "adduct"]).indices.values()
        if len(idx) >= 2
    ][: args.max_groups]
    tasks = [
        (idx, args.bin_da, args.prefilter, args.ppm, args.min_mz, args.min_cosine) for idx in groups
    ]
    with Pool(args.workers) as pool:
        edge_lists = pool.map(_edges_for_group, tasks, chunksize=256)
    edges = pd.DataFrame(
        [edge for group_edges in edge_lists for edge in group_edges],
        columns=["spec_a", "spec_b", "cos_5ppm", "median_dmz", "kind", "loser"],
    )
    logger.info("%d near-duplicate edges from %d groups", len(edges), len(groups))
    clusters = cluster_rows(len(meta), num_peaks, edges)
    clusters["copy_key"] = meta["copy_key"].to_numpy()
    clusters["inst_sig"] = _INST
    clusters["ce_sig"] = _CE

    edges.to_parquet(args.out_dp / "casmi26_train_dup_edges.parquet", index=False)
    clusters.to_parquet(args.out_dp / "casmi26_train_dup_clusters.parquet", index=False)
    run_info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "n_spectra": len(meta),
        "n_groups_ge2": len(groups),
        "n_edges": len(edges),
        "n_same_edges": int((edges["kind"] == "same").sum()),
        "n_wildcard_edges": int((edges["kind"] == "wildcard").sum()),
        "n_dropped": int((~clusters["kept"]).sum()),
        "n_exact_copy_redundant": int(meta["copy_key"].duplicated().sum()),
        "elapsed_s": round(time.time() - start, 1),
    }
    (args.out_dp / "run_info.json").write_text(json.dumps(run_info, indent=2))
    logger.info("done: %s", run_info)


if __name__ == "__main__":
    main()
