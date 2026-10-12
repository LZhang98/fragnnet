"""FraGNNet's HDF5 DAG storage: the packed ``dags.h5`` format (one zlib-compressed pickle per
molecule plus an offset/length index), its writer and readers, the reader for the legacy
one-group-per-molecule layout, and the list of molecules a DAG directory or ``dags.h5`` holds
(moved from ``utils/dag_format_utils.py`` and ``utils/script_utils.py``). The per-molecule
pickle layout that core reads stays in ``utils/dag_format_utils.py``.
"""

import json
import os
import pickle
import zlib
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch as th
import torch_geometric as pyg

if TYPE_CHECKING:
    import h5py as _h5py


PACKED_DAG_FORMAT_ATTR = "fragnnet_dag_format"
# Bump whenever PACKED_DAG_INDEX_DTYPE's on-disk layout changes, so
# initialize_packed_dag_hdf5 rejects an old file instead of silently
# reusing an incompatible index dtype.
PACKED_DAG_FORMAT = "packed_v1"
PACKED_DAG_BYTES_DATASET = "dag_bytes"
PACKED_DAG_INDEX_DATASET = "dag_index"
PACKED_DAG_BYTES_CHUNK_SIZE = 64 * 1024
PACKED_DAG_INDEX_CHUNK_SIZE = 8192
# zlib level for the per-molecule DAG payload. Measured on 185 drug-like d3/h4 DAGs
# (150 MB of raw pickle): level 4 -> 13.5 MB, level 6 -> 12.2 MB, level 9 -> 12.0 MB.
# Level 6 is strictly better than 4 here — 9.5% smaller AND ~10% faster to inflate
# (0.83 vs 0.92 ms/DAG, since there is less to read back) for +3 ms/molecule of write
# CPU, against ~90 ms/molecule of fragmentation. Level 9 costs 20x the write CPU for a
# further 1.5%. No HDF5-level filter is applied on top: the bytes dataset already holds
# compressed payloads, so a gzip filter would mostly duplicate this work.
PACKED_DAG_ZLIB_LEVEL = 6
PACKED_DAG_INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u8"),
    ]
)


def initialize_packed_dag_hdf5(h5_file: "_h5py.File") -> None:
    """Initialize an open HDF5 file for packed DAG storage.

    Creates the shared byte-blob and index datasets used by
    :func:`dump_dag_hdf5_packed`. Idempotent: calling this again on a file
    already initialized for packed storage is a no-op.

    Args:
        h5_file: Open writable ``h5py.File`` object.

    Raises:
        ValueError: If the file already stores DAGs in an unsupported packed
            format, or if the file is non-empty but not yet initialized for
            packed DAG storage.
    """
    stored_format = h5_file.attrs.get(PACKED_DAG_FORMAT_ATTR)
    if stored_format is not None:
        if stored_format != PACKED_DAG_FORMAT:
            raise ValueError(f"Unsupported packed DAG format: {stored_format}")
        return
    if len(h5_file) != 0:
        raise ValueError("Cannot initialize packed DAG storage in a non-empty HDF5 file")

    h5_file.attrs[PACKED_DAG_FORMAT_ATTR] = PACKED_DAG_FORMAT
    h5_file.create_dataset(
        PACKED_DAG_BYTES_DATASET,
        shape=(0,),
        maxshape=(None,),
        chunks=(PACKED_DAG_BYTES_CHUNK_SIZE,),
        dtype=np.uint8,
    )
    h5_file.create_dataset(
        PACKED_DAG_INDEX_DATASET,
        shape=(0,),
        maxshape=(None,),
        chunks=(PACKED_DAG_INDEX_CHUNK_SIZE,),
        dtype=PACKED_DAG_INDEX_DTYPE,
    )


def dump_dag_hdf5_packed(h5_file: "_h5py.File", mol_id: str | int, dag_d: dict) -> None:
    """Append one compressed DAG payload to a packed HDF5 file.

    Args:
        h5_file: Open writable ``h5py.File``, already initialized via
            :func:`initialize_packed_dag_hdf5`.
        mol_id: Molecule identifier. Used directly as a dense index into the
            packed index dataset, so it must be a small non-negative integer.
        dag_d: DAG payload dictionary as returned by :func:`compute_dags`.

    Raises:
        ValueError: If ``mol_id`` is negative, or a DAG has already been
            stored for it.
        KeyError: If ``h5_file`` was not initialized via
            :func:`initialize_packed_dag_hdf5`.
    """
    mol_idx = int(mol_id)
    if mol_idx < 0:
        raise ValueError(f"mol_id must be non-negative for packed HDF5 storage: {mol_id}")

    index_dataset = h5_file[PACKED_DAG_INDEX_DATASET]
    if mol_idx < index_dataset.shape[0] and int(index_dataset[mol_idx]["length"]) != 0:
        raise ValueError(f"Duplicate mol_id in packed HDF5 file: {mol_id}")

    payload = pickle.dumps(dag_d, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_payload = zlib.compress(payload, level=PACKED_DAG_ZLIB_LEVEL)
    payload_array = np.frombuffer(compressed_payload, dtype=np.uint8)

    bytes_dataset = h5_file[PACKED_DAG_BYTES_DATASET]
    offset = int(bytes_dataset.shape[0])
    length = int(payload_array.size)
    bytes_dataset.resize((offset + length,))
    bytes_dataset[offset : offset + length] = payload_array

    if mol_idx >= index_dataset.shape[0]:
        index_dataset.resize((mol_idx + 1,))
    index_dataset[mol_idx] = (offset, length)


def _load_dag_hdf5_packed(h5_file, mol_id: str | int) -> dict:
    """Load one DAG payload from a packed HDF5 file."""
    mol_idx = int(mol_id)
    index_dataset = h5_file[PACKED_DAG_INDEX_DATASET]
    if mol_idx < 0 or mol_idx >= index_dataset.shape[0]:
        raise KeyError(f"mol_id {mol_id} is not present in packed HDF5 file")

    record = index_dataset[mol_idx]
    length = int(record["length"])
    if length == 0:
        raise KeyError(f"mol_id {mol_id} is not present in packed HDF5 file")

    offset = int(record["offset"])
    compressed_payload = h5_file[PACKED_DAG_BYTES_DATASET][offset : offset + length].tobytes()
    return pickle.loads(zlib.decompress(compressed_payload))


def load_dag_hdf5(h5_file: "_h5py.File", mol_id: str | int) -> dict:
    """Read one DAG dictionary from an open HDF5 file, packed or legacy format.

    Args:
        h5_file: Open h5py.File in the packed format (:func:`dump_dag_hdf5_packed`) or the
            legacy one-group-per-molecule format.
        mol_id: Molecule identifier to look up.

    Returns:
        DAG payload dictionary compatible with the format returned by
        ``compute_dags``.
    """
    if h5_file.attrs.get(PACKED_DAG_FORMAT_ATTR) == PACKED_DAG_FORMAT:
        return _load_dag_hdf5_packed(h5_file, mol_id)
    return load_dag_hdf5_from_group(h5_file[f"{int(mol_id)}"])


def load_dag_hdf5_from_group(h5_group: "_h5py.Group") -> dict:
    """Read one DAG dictionary from a legacy per-molecule HDF5 group.

    Numpy arrays are converted back to PyTorch tensors with their original
    dtypes.  JSON-encoded attributes are decoded to their original Python types.

    Args:
        h5_group: An open legacy molecule group (one group per molecule, the layout before
            the packed format).

    Returns:
        DAG payload dictionary compatible with the format returned by
        ``compute_dags``.
    """
    g = h5_group["dag"]
    dag = pyg.data.Data(
        x=th.from_numpy(g["x"][:]),
        edge_index=th.from_numpy(g["edge_index"][:]),
        edge_attr=th.from_numpy(g["edge_attr"][:]),
    )
    dag.node_feat_idxs = th.from_numpy(g["node_feat_idxs"][:])
    dag.edge_feat_idxs = th.from_numpy(g["edge_feat_idxs"][:])
    for name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
        if name in g:
            setattr(dag, name, th.from_numpy(g[name][:]))

    dag_d: dict = {
        "dag": dag,
        "formula_peak_mzs": th.from_numpy(h5_group["formula_peak_mzs"][:]),
        "formula_peak_probs": th.from_numpy(h5_group["formula_peak_probs"][:]),
        "edges_min_depth": h5_group["edges_min_depth"][:],
        "nodes_min_depth": h5_group["nodes_min_depth"][:],
    }

    # JSON-encoded variable-structure fields
    dag_d["idx_to_formula"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["idx_to_formula"]).items()
    }
    # Stored as list-of-lists; restore as list-of-sets to match original type
    dag_d["idx_by_h_delta"] = [set(s) for s in json.loads(h5_group.attrs["idx_by_h_delta"])]
    dag_d["dag_num_edges_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_edges_by_depth"]).items()
    }
    dag_d["dag_num_nodes_by_depth"] = {
        int(k): v for k, v in json.loads(h5_group.attrs["dag_num_nodes_by_depth"]).items()
    }

    # Integer scalars
    for key in [
        "max_depth",
        "reached_depth",
        "max_h_transfer",
        "dag_num_edges",
        "dag_num_nodes",
        "dag_num_nodes_nb",
        "node_feature_size",
        "edge_feature_size",
    ]:
        dag_d[key] = int(h5_group.attrs[key])

    # Float scalars
    for key in ["dag_sparsity", "formula_redundancy"]:
        dag_d[key] = float(h5_group.attrs[key])

    # Bool scalars
    for key in ["h_prior", "is_directed", "force_stopped"]:
        dag_d[key] = bool(h5_group.attrs[key])

    return dag_d


def available_frag_mol_ids(frag_dp: str) -> set[str]:
    """Return the mol_ids (as strings) that have a DAG in ``frag_dp``.

    ``frag_dp`` is either a directory of per-molecule pickles (the folder backend,
    ``<name>/dags/``) or a packed ``dags.h5`` (the HDF5 backend). Both layouts skip
    molecules whose fragmentation failed or timed out, so the caller must drop those
    rows from mol_df / spec_df before building the dataset — the HDF5 backend does no
    missing-file check of its own and raises a KeyError on the first absent mol_id.

    For the HDF5 backend the ids come from the ``meta_info.json`` written next to the
    .h5 when available (its keys are exactly the successfully fragmented molecules),
    and otherwise from the packed index: it is dense in mol_id, so an entry with
    ``length == 0`` means "no DAG stored".

    Args:
        frag_dp: Directory of DAG files, or path to a packed ``dags.h5``.

    Returns:
        Set of mol_ids, as strings, for which a DAG can be loaded.
    """
    if not frag_dp.endswith(".h5"):
        return {f.split(".")[0] for f in os.listdir(frag_dp)}

    meta_info_fp = os.path.join(os.path.dirname(os.path.abspath(frag_dp)), "meta_info.json")
    if os.path.exists(meta_info_fp):
        with open(meta_info_fp) as f:
            return {str(mol_id) for mol_id in json.load(f)}

    with h5py.File(frag_dp, "r") as h5_file:
        index = h5_file[PACKED_DAG_INDEX_DATASET][:]
    return {str(mol_id) for mol_id in np.nonzero(index["length"])[0]}
